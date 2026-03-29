"""
Plot training/evaluation figures from project outputs.

Generates:
1) training loss line plot
2) validation loss line plot
3) BLEU histogram (grouped bars)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from config import get_config

plt.rcParams.update({"font.size": 13})


def display_name(name: str) -> str:
    mapping = {
        "rnn": "RNN",
        "gru": "GRU",
        "lstm": "LSTM",
        "lstm_dropout": "LSTP_dp",
    }
    return mapping.get(name, name)


def load_training_history(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"training_loss.npy not found: {path}")
    arr = np.load(path, allow_pickle=True)
    history = arr.item() if hasattr(arr, "item") else arr
    if not isinstance(history, dict):
        raise ValueError("Unexpected training_loss.npy format: expected dict")
    return history


def load_metrics(path: Path) -> Tuple[List[str], List[Dict[str, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {path}")

    rows: List[Dict[str, float]] = []
    models: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            models.append(model)
            rows.append(
                {
                    "BLEU-1": float(row["BLEU-1"]),
                    "BLEU-2": float(row["BLEU-2"]),
                    "BLEU-3": float(row["BLEU-3"]),
                    "BLEU-4": float(row["BLEU-4"]),
                }
            )
    return models, rows


def plot_training_loss(history: Dict[str, Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for variant, v in history.items():
        losses = v.get("train_losses", [])
        if not losses:
            continue
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, marker="o", linewidth=2, label=display_name(variant))

    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_validation_loss(history: Dict[str, Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for variant, v in history.items():
        losses = v.get("val_losses", [])
        if not losses:
            continue
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, marker="o", linewidth=2, label=display_name(variant))

    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bleu_histogram(models: List[str], rows: List[Dict[str, float]], out_path: Path) -> None:
    bleu_keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    x = np.arange(len(bleu_keys))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model, row) in enumerate(zip(models, rows)):
        values = [row[key] for key in bleu_keys]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=display_name(model))

    ax.set_title("Evaluation BLEU Scores")
    ax.set_xlabel("BLEU Score")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(bleu_keys)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    default_output_dir = cfg.outputs_dir / "figures"

    parser = argparse.ArgumentParser(description="Generate training/evaluation figures.")
    parser.add_argument(
        "--training-loss",
        type=Path,
        default=cfg.training_loss_path,
        help="Path to training_loss.npy",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=cfg.metrics_csv_path,
        help="Path to metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to save generated figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history = load_training_history(args.training_loss)
    models, metrics_rows = load_metrics(args.metrics_csv)

    train_fig = args.output_dir / "training_loss.png"
    val_fig = args.output_dir / "val_loss.png"
    bleu_fig = args.output_dir / "evaluate_histogram.png"

    plot_training_loss(history, train_fig)
    plot_validation_loss(history, val_fig)
    plot_bleu_histogram(models, metrics_rows, bleu_fig)

    print(f"Saved: {train_fig}")
    print(f"Saved: {val_fig}")
    print(f"Saved: {bleu_fig}")


if __name__ == "__main__":
    main()
