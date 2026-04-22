#!/usr/bin/env python3
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    metrics = ["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr"]

    lstm_attn_path = Path("caption_only/outputs/lstm_attn/metrics.json")
    share_encoder_path = Path("share_encoder/outputs/lstm_attn/clsw_2.0/metrics.json")

    lstm_attn = load_metrics(lstm_attn_path)
    share_encoder = load_metrics(share_encoder_path)

    lstm_values = [float(lstm_attn[m]) for m in metrics]
    share_values = [float(share_encoder[m]) for m in metrics]

    x = list(range(len(metrics)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7, 6))
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        lstm_values,
        width=width,
        color="#4C78A8",
        label="LSTM_Attn",
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        share_values,
        width=width,
        color="#F58518",
        label="Share_encoder",
    )

    ax.set_title("LSTM_Attn vs Share_encoder")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(h + 0.015, 0.995),
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/fig4_lstm_attn_vs_share_encoder.pdf"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
