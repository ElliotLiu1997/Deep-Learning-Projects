from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from caption_only.utils import compute_bleu_scores, compute_cider, compute_rouge_l


def _sentence_repetition_rate(tokens, n: int = 3) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def compute_repetition_rate_stats(hypotheses, n: int = 3):
    rates = [_sentence_repetition_rate(tokens, n=n) for tokens in hypotheses]
    if not rates:
        return 0.0, 0.0
    return float(sum(rates) / len(rates)), float(statistics.median(rates))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate improved caption outputs.")
    p.add_argument("--results_json", type=str, required=True, help="Path to generated results json")
    p.add_argument("--model_type", type=str, required=True, choices=["lstm", "lstm_attn", "transformer", "share_encoder"])
    p.add_argument("--metrics_csv", type=str, default="caption_improved/metrics.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.load(open(args.results_json, "r", encoding="utf-8"))

    refs = []
    rouge_refs = []
    hyps = []
    for r in rows:
        gt_tokens = (r.get("ground_truth") or "").split()
        pred_tokens = (r.get("prediction") or "").split()
        refs.append([gt_tokens])
        rouge_refs.append(gt_tokens)
        hyps.append(pred_tokens)

    bleu = compute_bleu_scores(refs, hyps, max_n=4)
    rouge_l = compute_rouge_l(rouge_refs, hyps)
    cider = compute_cider(refs, hyps)
    repetition_rate_mean, repetition_rate_median = compute_repetition_rate_stats(hyps, n=3)

    metrics = {
        "model": args.model_type,
        "BLEU-1": bleu["BLEU-1"],
        "BLEU-2": bleu["BLEU-2"],
        "BLEU-3": bleu["BLEU-3"],
        "BLEU-4": bleu["BLEU-4"],
        "ROUGE-L": rouge_l,
        "CIDEr": cider,
        "RepetitionRateMean": repetition_rate_mean,
        "RepetitionRateMedian": repetition_rate_median,
    }
    print(metrics)

    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"Appended metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
