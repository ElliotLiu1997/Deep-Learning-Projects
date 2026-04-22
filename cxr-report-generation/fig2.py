#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import pandas as pd


SELECTED_LABELS = [
    "normal",
    "cardiomegaly",
    "atelectasis",
    "pleural effusion",
    "lung opacity",
    "nodule",
    "airspace disease",
    "atherosclerosis",
    "granulomatous disease",
    "calcified granuloma",
    "calcinosis",
    "cicatrix",
    "scoliosis",
]


def load_per_class_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["class_idx"] = pd.to_numeric(df["class_idx"], errors="coerce")
    df = df.dropna(subset=["class_idx"]).copy()
    df["class_idx"] = df["class_idx"].astype(int)
    df = df.sort_values("class_idx").copy()
    df["label"] = df["class_idx"].apply(
        lambda i: SELECTED_LABELS[i] if 0 <= i < len(SELECTED_LABELS) else f"class_{i}"
    )
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    return df


def plot_metric(df: pd.DataFrame, model_name: str, metric: str, out_path: str) -> None:
    title_metric = metric.upper()
    color = "#4C78A8" if metric == "f1" else "#F58518"

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = range(len(df))
    bars = ax.bar(x_pos, df[metric], color=color)
    ax.set_title(f"{model_name} Per-Class {title_metric}", fontsize=16)
    ax.set_xlabel("Label", fontsize=13)
    ax.set_ylabel(title_metric, fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(df["label"].tolist(), rotation=45, ha="right", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)

    for bar, value in zip(bars, df[metric]):
        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(float(value) + 0.015, 0.995),
                f"{float(value):.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

    resnet_path = "classification_only/outputs/resnet/eval_per_class_metrics.csv"
    vit_path = "classification_only/outputs/vit/eval_per_class_metrics.csv"

    resnet_df = load_per_class_metrics(resnet_path)
    vit_df = load_per_class_metrics(vit_path)

    plot_metric(
        resnet_df,
        "ResNet",
        "f1",
        os.path.join(out_dir, "resnet_per_class_f1.pdf"),
    )
    plot_metric(
        resnet_df,
        "ResNet",
        "auc",
        os.path.join(out_dir, "resnet_per_class_auc.pdf"),
    )
    plot_metric(
        vit_df,
        "ViT",
        "f1",
        os.path.join(out_dir, "vit_per_class_f1.pdf"),
    )
    plot_metric(
        vit_df,
        "ViT",
        "auc",
        os.path.join(out_dir, "vit_per_class_auc.pdf"),
    )

    print("Saved:")
    print("- figures/resnet_per_class_f1.pdf")
    print("- figures/resnet_per_class_auc.pdf")
    print("- figures/vit_per_class_f1.pdf")
    print("- figures/vit_per_class_auc.pdf")


if __name__ == "__main__":
    main()
