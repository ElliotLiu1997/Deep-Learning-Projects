import argparse
import re
from collections import Counter

import pandas as pd


def safe_text(x) -> str:
    return "" if pd.isna(x) else str(x)


def normalize_text(text: str) -> str:
    text = safe_text(text).lower()
    replacements = {
        "c/w": "consistent_with",
        "r/l": "right_left",
        "and/or": "and_or",
        "w/o": "without",
        "w/": "with",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"x{3,}", " ", text)
    text = re.sub(r"\b\d+\.\b", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\-/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_tokens(tokens):
    out = []
    for t in tokens:
        t = t.strip().rstrip(".")
        if not t:
            continue
        if re.match(r"x-\w+", t):
            continue
        if re.fullmatch(r"x{3,}", t):
            continue
        out.append(t)
    return out


def preprocess_impression(text: str):
    cleaned = clean_text(text)
    tokens = clean_tokens(cleaned.split())
    if len(tokens) < 3:
        return "", [], []
    clean = " ".join(tokens)
    sentences = [s.strip() for s in re.split(r"\.+", clean) if s.strip()]
    return clean, tokens, sentences


def encode_tokens(tokens, word2idx, max_len: int):
    seq = [word2idx["<SOS>"]]
    seq += [word2idx.get(t, word2idx["<UNK>"]) for t in tokens]
    seq.append(word2idx["<EOS>"])

    if len(seq) < max_len:
        seq += [word2idx["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
        seq[-1] = word2idx["<EOS>"]
    return seq


def main():
    parser = argparse.ArgumentParser(description="Build caption CSV using impression-only text.")
    parser.add_argument("--input_csv", type=str, default="info.csv")
    parser.add_argument("--output_csv", type=str, default="info_impression.csv")
    parser.add_argument("--min_word_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=90)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "impression" not in df.columns:
        raise ValueError("Input CSV does not contain 'impression' column.")
    if "split" not in df.columns:
        raise ValueError("Input CSV does not contain 'split' column.")

    processed = df["impression"].apply(preprocess_impression)
    df["clean_text"] = processed.apply(lambda x: x[0])
    df["tokens"] = processed.apply(lambda x: x[1])
    df["sentences"] = processed.apply(lambda x: x[2])

    df = df[df["tokens"].apply(lambda x: isinstance(x, list) and len(x) >= 3)].copy()

    train_tokens = df.loc[df["split"].astype(str).str.lower() == "train", "tokens"]
    counter = Counter()
    for toks in train_tokens:
        counter.update(toks)

    vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + [w for w, c in counter.items() if c >= args.min_word_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}

    df["caption_seq"] = df["tokens"].apply(lambda t: encode_tokens(t, word2idx=word2idx, max_len=args.max_len))

    df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")
    print("Split counts:")
    print(df["split"].value_counts())
    print(f"vocab_size={len(vocab)} max_len={args.max_len}")


if __name__ == "__main__":
    main()
