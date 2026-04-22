import ast
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _parse_list_field(raw: str):
    if isinstance(raw, list):
        return raw
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return ast.literal_eval(raw)


def _infer_special_indices(rows: List[Dict[str, str]]) -> Tuple[int, int, int, int]:
    from collections import Counter

    first_counter = Counter()
    eos_counter = Counter()
    pad_counter = Counter()

    for row in rows:
        seq = [int(x) for x in _parse_list_field(row.get("caption_seq", ""))]
        if not seq:
            continue

        first_counter[seq[0]] += 1

        tokens = _parse_list_field(row.get("tokens", ""))
        eos_pos = min(len(tokens) + 1, len(seq) - 1)
        eos_counter[seq[eos_pos]] += 1

        for idx in range(eos_pos + 1, len(seq)):
            pad_counter[seq[idx]] += 1

    sos_idx = first_counter.most_common(1)[0][0] if first_counter else 1
    eos_idx = eos_counter.most_common(1)[0][0] if eos_counter else 2
    pad_idx = pad_counter.most_common(1)[0][0] if pad_counter else 0
    unk_idx = 3
    return pad_idx, sos_idx, eos_idx, unk_idx


def _reconstruct_vocab_from_alignment(
    rows: List[Dict[str, str]],
    vocab_size: int,
    pad_idx: int,
    sos_idx: int,
    eos_idx: int,
    unk_idx: int,
) -> Tuple[List[str], Dict[str, int], List[str]]:
    from collections import Counter, defaultdict

    id_to_word_counts = defaultdict(Counter)

    for row in rows:
        seq = [int(x) for x in _parse_list_field(row.get("caption_seq", ""))]
        if len(seq) < 2:
            continue

        tokens = [str(t) for t in _parse_list_field(row.get("tokens", ""))]
        token_count = min(len(tokens), max(0, len(seq) - 2))

        for idx in range(token_count):
            token_id = int(seq[idx + 1])
            token = tokens[idx]
            if token_id in {pad_idx, sos_idx, eos_idx}:
                continue
            if not token:
                continue
            id_to_word_counts[token_id][token] += 1

    idx2word = ["<UNK>"] * vocab_size
    for token_id, counts in id_to_word_counts.items():
        if 0 <= token_id < vocab_size:
            idx2word[token_id] = counts.most_common(1)[0][0]

    if 0 <= pad_idx < vocab_size:
        idx2word[pad_idx] = "<PAD>"
    if 0 <= sos_idx < vocab_size:
        idx2word[sos_idx] = "<SOS>"
    if 0 <= eos_idx < vocab_size:
        idx2word[eos_idx] = "<EOS>"
    if 0 <= unk_idx < vocab_size:
        idx2word[unk_idx] = "<UNK>"

    vocab = list(idx2word)
    word2idx = {}
    for i, w in enumerate(vocab):
        if w not in word2idx:
            word2idx[w] = i

    return vocab, word2idx, idx2word


def load_caption_metadata(data_csv: str) -> Dict:
    csv_path = Path(data_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("No rows found in CSV.")

    fieldnames = set(rows[0].keys())
    pad_idx, sos_idx, eos_idx, unk_idx = _infer_special_indices(rows)

    has_vocab_cols = "vocab" in fieldnames and "word2idx" in fieldnames
    vocab = None
    word2idx = None

    if has_vocab_cols:
        for row in rows:
            raw_vocab = (row.get("vocab") or "").strip()
            raw_word2idx = (row.get("word2idx") or "").strip()
            if raw_vocab and raw_word2idx:
                vocab = _parse_list_field(raw_vocab)
                word2idx = ast.literal_eval(raw_word2idx)
                break

    max_token_id = 0
    max_seq_len = 0

    for row in rows:
        seq = [int(x) for x in _parse_list_field(row.get("caption_seq", ""))]
        if not seq:
            continue
        max_seq_len = max(max_seq_len, len(seq))
        max_token_id = max(max_token_id, max(seq))

    if vocab is None or word2idx is None:
        vocab_size = max_token_id + 1
        vocab, word2idx, idx2word = _reconstruct_vocab_from_alignment(
            rows=rows,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            unk_idx=unk_idx,
        )
    else:
        vocab_size = max(max_token_id + 1, len(vocab))
        idx2word = ["<UNK>"] * vocab_size
        for w, i in word2idx.items():
            if 0 <= i < vocab_size:
                idx2word[i] = w

        if 0 <= pad_idx < vocab_size:
            idx2word[pad_idx] = "<PAD>"
        if 0 <= sos_idx < vocab_size:
            idx2word[sos_idx] = "<SOS>"
        if 0 <= eos_idx < vocab_size:
            idx2word[eos_idx] = "<EOS>"
        if 0 <= unk_idx < vocab_size:
            idx2word[unk_idx] = "<UNK>"

    return {
        "vocab": vocab,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "vocab_size": vocab_size,
        "pad_idx": pad_idx,
        "sos_idx": sos_idx,
        "eos_idx": eos_idx,
        "unk_idx": unk_idx,
        "max_seq_len": max_seq_len,
    }


class CaptionDataset(Dataset):
    def __init__(
        self,
        data_csv: str,
        image_dir: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.data_csv = Path(data_csv)
        self.image_dir = Path(image_dir)
        self.split = split.strip().lower()
        self.transform = transform

        if not self.data_csv.exists():
            raise FileNotFoundError(f"CSV file not found: {self.data_csv}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.samples: List[Tuple[Path, List[int], str]] = []
        self._load_rows()

        if not self.samples:
            raise ValueError(f"No samples found for split='{self.split}'")

    def _load_rows(self) -> None:
        with self.data_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = (row.get("split") or "").strip().lower()
                if row_split != self.split:
                    continue

                filename = row.get("filename")
                caption_raw = row.get("caption_seq")
                if not filename or caption_raw is None:
                    continue

                caption_seq = [int(x) for x in _parse_list_field(caption_raw)]
                if not caption_seq:
                    continue

                image_path = self.image_dir / filename
                self.samples.append((image_path, caption_seq, filename))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, caption_seq, filename = self.samples[idx]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        caption = torch.tensor(caption_seq, dtype=torch.long)
        return image, caption, filename


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
