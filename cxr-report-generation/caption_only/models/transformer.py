import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def _build_sinusoidal_positions(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class TransformerDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feat_dim: int = 512,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.feature_proj = nn.Linear(feat_dim, d_model)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(length, length, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def _decode(self, feats: torch.Tensor, input_seq: torch.Tensor) -> torch.Tensor:
        # feats: (B, N, D), input_seq: (B, T)
        memory = self.feature_proj(feats)
        memory = memory + _build_sinusoidal_positions(memory.size(1), self.d_model, memory.device)

        tgt = self.embedding(input_seq) * (self.d_model ** 0.5)
        tgt = self.pos_enc(tgt)

        tgt_mask = self._causal_mask(input_seq.size(1), input_seq.device)
        tgt_key_padding_mask = input_seq.eq(self.pad_idx)
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.fc_out(out)
        return logits

    def forward(
        self,
        feats: torch.Tensor,
        input_seq: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        if teacher_forcing_ratio >= 1.0:
            return self._decode(feats, input_seq)

        bsz, seq_len = input_seq.shape
        generated = input_seq[:, :1]
        logits = []
        for t in range(seq_len):
            out = self._decode(feats, generated)
            step_logits = out[:, -1, :]
            logits.append(step_logits)
            if t + 1 < seq_len:
                pred_next = torch.argmax(step_logits, dim=-1, keepdim=True)
                gt_next = input_seq[:, t + 1].unsqueeze(1)
                use_teacher = (torch.rand(bsz, 1, device=input_seq.device) < teacher_forcing_ratio)
                next_tok = torch.where(use_teacher, gt_next, pred_next)
                generated = torch.cat([generated, next_tok], dim=1)
        return torch.stack(logits, dim=1)

    def generate(
        self,
        feats: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> torch.Tensor:
        bsz = feats.size(0)
        generated = torch.full((bsz, 1), sos_idx, dtype=torch.long, device=feats.device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=feats.device)

        for _ in range(max_len):
            logits = self._decode(feats, generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if eos_idx >= 0:
                eos_fill = torch.full_like(next_token, eos_idx)
                next_token = torch.where(finished.unsqueeze(1), eos_fill, next_token)
                finished = finished | (next_token.squeeze(1) == eos_idx)
            generated = torch.cat([generated, next_token], dim=1)
            if finished.all():
                break

        seq = generated[:, 1:]
        if eos_idx >= 0:
            for i in range(seq.size(0)):
                eos_pos = (seq[i] == eos_idx).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    first = int(eos_pos[0].item())
                    if first + 1 < seq.size(1):
                        seq[i, first + 1 :] = eos_idx
        return seq

    @torch.no_grad()
    def beam_search(
        self,
        feats: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
        beam_size: int = 3,
        length_penalty: float = 0.7,
    ) -> torch.Tensor:
        if feats.size(0) != 1:
            raise ValueError("beam_search expects batch size 1 features.")

        beams = [([int(sos_idx)], 0.0, False)]

        for _ in range(max_len):
            candidates = []
            for seq, score, finished in beams:
                if finished:
                    candidates.append((seq, score, finished))
                    continue

                inp = torch.tensor([seq], dtype=torch.long, device=feats.device)
                logits = self._decode(feats, inp)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                top_vals, top_idx = torch.topk(log_probs, k=min(beam_size, log_probs.numel()))

                for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
                    token = int(idx)
                    new_seq = seq + [token]
                    candidates.append((new_seq, score + float(v), token == eos_idx))

            def rank_key(item):
                seq_len = max(1, len(item[0]) - 1)
                norm = (seq_len ** length_penalty) if length_penalty > 0 else 1.0
                return item[1] / norm

            beams = sorted(candidates, key=rank_key, reverse=True)[:beam_size]
            if all(b[2] for b in beams):
                break

        best_seq = beams[0][0][1:]
        if eos_idx >= 0 and eos_idx in best_seq:
            best_seq = best_seq[: best_seq.index(eos_idx) + 1]
        return torch.tensor(best_seq, dtype=torch.long, device=feats.device).unsqueeze(0)
