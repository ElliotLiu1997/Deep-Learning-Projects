import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, attn_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, feats: torch.Tensor, hidden: torch.Tensor):
        # feats: (B, N, D), hidden: (B, H)
        attn_input = torch.tanh(self.feat_proj(feats) + self.hidden_proj(hidden).unsqueeze(1))
        e = self.score(attn_input).squeeze(-1)
        alpha = torch.softmax(e, dim=1)
        context = torch.sum(feats * alpha.unsqueeze(-1), dim=1)
        return context, alpha


class LSTMAttnDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feat_dim: int = 512,
        emb_dim: int = 256,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = AdditiveAttention(feat_dim=feat_dim, hidden_dim=hidden_dim, attn_dim=attn_dim)

        self.init_h = nn.Linear(feat_dim, hidden_dim)
        self.init_c = nn.Linear(feat_dim, hidden_dim)

        self.lstm_cell = nn.LSTMCell(emb_dim + feat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def _init_state(self, feats: torch.Tensor):
        mean_feat = feats.mean(dim=1)
        h0 = self.init_h(mean_feat)
        c0 = self.init_c(mean_feat)
        return h0, c0

    def forward(
        self,
        feats: torch.Tensor,
        input_seq: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        # feats: (B, 49, 512), input_seq: (B, T)
        bsz, seq_len = input_seq.shape
        h, c = self._init_state(feats)

        logits = []
        cur = input_seq[:, 0]
        for t in range(seq_len):
            emb = self.embedding(cur)
            context, _ = self.attention(feats, h)
            step_in = torch.cat([emb, context], dim=-1)
            h, c = self.lstm_cell(step_in, (h, c))
            logit = self.fc(self.dropout(h))
            logits.append(logit)

            if t + 1 < seq_len:
                pred_next = torch.argmax(logit, dim=-1)
                gt_next = input_seq[:, t + 1]
                if teacher_forcing_ratio >= 1.0:
                    cur = gt_next
                else:
                    use_teacher = torch.rand(bsz, device=input_seq.device) < teacher_forcing_ratio
                    cur = torch.where(use_teacher, gt_next, pred_next)

        return torch.stack(logits, dim=1)

    def generate(
        self,
        feats: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> torch.Tensor:
        bsz = feats.size(0)
        h, c = self._init_state(feats)
        cur = torch.full((bsz,), sos_idx, dtype=torch.long, device=feats.device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=feats.device)

        generated = []
        for _ in range(max_len):
            emb = self.embedding(cur)
            context, _ = self.attention(feats, h)
            step_in = torch.cat([emb, context], dim=-1)
            h, c = self.lstm_cell(step_in, (h, c))
            logit = self.fc(h)
            nxt = torch.argmax(logit, dim=-1)
            if eos_idx >= 0:
                nxt = torch.where(finished, torch.full_like(nxt, eos_idx), nxt)
                finished = finished | (nxt == eos_idx)
            generated.append(nxt)
            cur = nxt
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
        feats: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
        beam_size: int = 3,
        length_penalty: float = 0.7,
    ) -> torch.Tensor:
        if feats.size(0) != 1:
            raise ValueError("beam_search expects batch size 1 features.")

        h0, c0 = self._init_state(feats)
        beams = [([int(sos_idx)], h0, c0, 0.0, False)]

        for _ in range(max_len):
            candidates = []
            for seq, h, c, score, finished in beams:
                if finished:
                    candidates.append((seq, h, c, score, finished))
                    continue

                cur = torch.tensor([seq[-1]], dtype=torch.long, device=feats.device)
                emb = self.embedding(cur)
                context, _ = self.attention(feats, h)
                step_in = torch.cat([emb, context], dim=-1)
                nh, nc = self.lstm_cell(step_in, (h, c))
                log_probs = F.log_softmax(self.fc(nh), dim=-1).squeeze(0)
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
        return torch.tensor(best_seq, dtype=torch.long, device=feats.device).unsqueeze(0)
