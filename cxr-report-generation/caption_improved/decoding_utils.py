from __future__ import annotations

from typing import List

import torch


def has_repeat_ngram(seq, n: int = 3):
    ngrams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
    return len(ngrams) != len(set(ngrams))


def apply_repetition_penalty(logits_1d: torch.Tensor, generated_tokens: List[int], penalty: float = 1.2) -> torch.Tensor:
    # Required behavior from spec: divide logits of seen tokens by 1.2 before selecting next token.
    if logits_1d.dim() != 1:
        raise ValueError("apply_repetition_penalty expects 1D logits")
    adjusted = logits_1d.clone()
    for token in set(generated_tokens):
        if 0 <= token < adjusted.numel():
            adjusted[token] /= penalty
    return adjusted


def select_with_topk_and_ngram(
    logits_1d: torch.Tensor,
    generated_tokens: List[int],
    top_k: int = 5,
    ngram_n: int = 3,
) -> int:
    if logits_1d.dim() != 1:
        raise ValueError("select_with_topk_and_ngram expects 1D logits")

    k = max(1, min(int(top_k), logits_1d.numel()))
    _top_vals, top_idx = torch.topk(logits_1d, k=k)

    # Select the first top-k token that does not form repeated n-gram.
    for token in top_idx.tolist():
        cand = generated_tokens + [int(token)]
        if not has_repeat_ngram(cand, n=ngram_n):
            return int(token)

    # Fallback to argmax when all top-k violate.
    return int(torch.argmax(logits_1d).item())
