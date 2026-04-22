#!/usr/bin/env python3
import ast
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_token_list(value: str) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return len(parsed)
    except (ValueError, SyntaxError):
        pass
    return len(text.split())


def parse_is_normal(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "normal"}:
        return "Normal"
    if v in {"0", "false", "no", "abnormal"}:
        return "Abnormal"
    return "Unknown"


def parse_multi_labels(value: str) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def main():
    csv_path = "info.csv"
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 1) Label distribution from Filtered_labels
    problem_series = (
        df["Filtered_labels"]
        .apply(parse_multi_labels)
        .explode()
        .dropna()
        .astype(str)
        .str.strip()
    )
    problem_counts = (
        problem_series[problem_series != ""]
        .value_counts()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = problem_counts.index.tolist()
    freqs = problem_counts.values.tolist()
    ax.bar(range(len(problem_counts)), freqs, color="#4C78A8")
    ax.set_title("Label Distribution", fontsize=18)
    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_xticks(range(len(problem_counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_label_distribution.pdf"), dpi=300)
    plt.close(fig)

    # 2) Normal vs abnormal ratio (pie chart)
    normal_counts = df["is_normal"].apply(parse_is_normal).value_counts()

    fig, ax = plt.subplots(figsize=(6, 6))
    pie_labels = []
    pie_sizes = []
    for key in ["Normal", "Abnormal", "Unknown"]:
        if key in normal_counts and normal_counts[key] > 0:
            count = int(normal_counts[key])
            pie_labels.append(f"{key} (n={count})")
            pie_sizes.append(count)
    total_samples = int(sum(pie_sizes))
    colors = ["#59A14F", "#E15759", "#B07AA1"][: len(pie_labels)]
    ax.pie(
        pie_sizes,
        labels=pie_labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=colors,
    )
    ax.set_title(f"Normal vs Abnormal (n={total_samples})")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_normal_vs_abnormal.pdf"), dpi=300)
    plt.close(fig)

    # 3) Label count per sample (label_len)
    label_len = pd.to_numeric(df["label_len"], errors="coerce").dropna().astype(int)
    label_len_counts = label_len.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals = label_len_counts.index.tolist()
    y_vals = label_len_counts.values.tolist()
    ax.bar(x_vals, y_vals, width=0.8, color="#F28E2B")
    ax.set_title("Label Count Per Sample")
    ax.set_xlabel("label_len")
    ax.set_ylabel("Number of Samples")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_label_count_per_sample.pdf"), dpi=300)
    plt.close(fig)

    # 4) Caption length distribution (tokens)
    caption_len = df["tokens"].apply(parse_token_list)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(caption_len, bins=30, color="#76B7B2", edgecolor="black")
    ax.set_title("Caption Length Distribution")
    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("Number of Samples")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_caption_length_distribution.pdf"), dpi=300)
    plt.close(fig)

    print("Saved:")
    print(f"- {out_dir}/fig1_label_distribution.pdf")
    print(f"- {out_dir}/fig2_normal_vs_abnormal.pdf")
    print(f"- {out_dir}/fig3_label_count_per_sample.pdf")
    print(f"- {out_dir}/fig4_caption_length_distribution.pdf")


if __name__ == "__main__":
    main()
