import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feat_dim: int = 512,
        emb_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.init_h = nn.Linear(feat_dim, hidden_dim * num_layers)
        self.init_c = nn.Linear(feat_dim, hidden_dim * num_layers)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def _init_state(self, global_feat: torch.Tensor):
        b = global_feat.size(0)
        h0 = self.init_h(global_feat).view(self.num_layers, b, self.hidden_dim).contiguous()
        c0 = self.init_c(global_feat).view(self.num_layers, b, self.hidden_dim).contiguous()
        return h0, c0

    def forward(
        self,
        global_feat: torch.Tensor,
        input_seq: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        if teacher_forcing_ratio >= 1.0:
            emb = self.embedding(input_seq)
            h0, c0 = self._init_state(global_feat)
            out, _ = self.lstm(emb, (h0, c0))
            out = self.dropout(out)
            logits = self.fc(out)
            return logits

        bsz, seq_len = input_seq.shape
        h, c = self._init_state(global_feat)
        cur = input_seq[:, 0].unsqueeze(1)
        logits = []

        for t in range(seq_len):
            emb = self.embedding(cur)
            out, (h, c) = self.lstm(emb, (h, c))
            step_logits = self.fc(self.dropout(out[:, -1, :]))
            logits.append(step_logits)

            if t + 1 < seq_len:
                pred_next = torch.argmax(step_logits, dim=-1)
                gt_next = input_seq[:, t + 1]
                use_teacher = torch.rand(bsz, device=input_seq.device) < teacher_forcing_ratio
                next_tok = torch.where(use_teacher, gt_next, pred_next)
                cur = next_tok.unsqueeze(1)

        return torch.stack(logits, dim=1)

    def generate(
        self,
        global_feat: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> torch.Tensor:
        bsz = global_feat.size(0)
        h, c = self._init_state(global_feat)

        cur = torch.full((bsz, 1), sos_idx, dtype=torch.long, device=global_feat.device)
        generated = []
        finished = torch.zeros(bsz, dtype=torch.bool, device=global_feat.device)

        for _ in range(max_len):
            emb = self.embedding(cur)
            out, (h, c) = self.lstm(emb, (h, c))
            logit = self.fc(out[:, -1, :])
            nxt = torch.argmax(logit, dim=-1)
            if eos_idx >= 0:
                nxt = torch.where(finished, torch.full_like(nxt, eos_idx), nxt)
                finished = finished | (nxt == eos_idx)
            generated.append(nxt)
            cur = nxt.unsqueeze(1)
            if finished.all():
                break

        seq = torch.stack(generated, dim=1)

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
        global_feat: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
        beam_size: int = 3,
        length_penalty: float = 0.7,
    ) -> torch.Tensor:
        if global_feat.size(0) != 1:
            raise ValueError("beam_search expects batch size 1 features.")

        h0, c0 = self._init_state(global_feat)
        beams = [([int(sos_idx)], h0, c0, 0.0, False)]

        for _ in range(max_len):
            candidates = []
            for seq, h, c, score, finished in beams:
                if finished:
                    candidates.append((seq, h, c, score, finished))
                    continue

                cur = torch.tensor([[seq[-1]]], dtype=torch.long, device=global_feat.device)
                emb = self.embedding(cur)
                out, (nh, nc) = self.lstm(emb, (h, c))
                log_probs = F.log_softmax(self.fc(out[:, -1, :]), dim=-1).squeeze(0)
                top_vals, top_idx = torch.topk(log_probs, k=min(beam_size, log_probs.numel()))

                for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
                    token = int(idx)
                    new_seq = seq + [token]
                    candidates.append((new_seq, nh, nc, score + float(v), token == eos_idx))

            def rank_key(item):
                seq_len = max(1, len(item[0]) - 1)
                norm = (seq_len ** length_penalty) if length_penalty > 0 else 1.0
                return item[3] / norm

            beams = sorted(candidates, key=rank_key, reverse=True)[:beam_size]
            if all(b[4] for b in beams):
                break

        best_seq = beams[0][0][1:]
        if eos_idx >= 0 and eos_idx in best_seq:
            best_seq = best_seq[: best_seq.index(eos_idx) + 1]
        return torch.tensor(best_seq, dtype=torch.long, device=global_feat.device).unsqueeze(0)
