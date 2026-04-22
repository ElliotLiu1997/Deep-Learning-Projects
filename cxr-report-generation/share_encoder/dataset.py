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


def _parse_list_field(raw):
    if isinstance(raw, list):
        return raw
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    return ast.literal_eval(s)


def load_caption_metadata(data_csv: str) -> Dict:
    csv_path = Path(data_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No rows found in CSV.")

    max_token_id = 0
    max_seq_len = 0
    for row in rows:
        seq = [int(x) for x in _parse_list_field(row.get("caption_seq", ""))]
        if not seq:
            continue
        max_token_id = max(max_token_id, max(seq))
        max_seq_len = max(max_seq_len, len(seq))

    vocab_size = max_token_id + 1
    idx2word = ["<UNK>"] * vocab_size

    # Reconstruct token strings from aligned tokens/caption_seq when possible.
    for row in rows:
        seq = [int(x) for x in _parse_list_field(row.get("caption_seq", ""))]
        toks = [str(t) for t in _parse_list_field(row.get("tokens", ""))]
        if len(seq) < 3 or not toks:
            continue
        for i, tok in enumerate(toks):
            j = i + 1  # after <SOS>
            if j >= len(seq) - 1:
                break
            tid = int(seq[j])
            if 0 <= tid < vocab_size and idx2word[tid] == "<UNK>" and tok:
                idx2word[tid] = tok

    # Infer special ids from sequence statistics.
    sos_idx = 1
    eos_idx = 2
    pad_idx = 0
    if rows:
        first_tokens = []
        end_tokens = []
        pad_tokens = []
        for row in rows:
            seq = [int(x) for x in _parse_list_field(row.get("caption_seq", ""))]
            if not seq:
                continue
            first_tokens.append(seq[0])
            if len(seq) >= 2:
                toks = _parse_list_field(row.get("tokens", ""))
                eos_pos = min(len(toks) + 1, len(seq) - 1)
                end_tokens.append(seq[eos_pos])
                pad_tokens.extend(seq[eos_pos + 1 :])
        if first_tokens:
            sos_idx = max(set(first_tokens), key=first_tokens.count)
        if end_tokens:
            eos_idx = max(set(end_tokens), key=end_tokens.count)
        if pad_tokens:
            pad_idx = max(set(pad_tokens), key=pad_tokens.count)

    if 0 <= pad_idx < vocab_size:
        idx2word[pad_idx] = "<PAD>"
    if 0 <= sos_idx < vocab_size:
        idx2word[sos_idx] = "<SOS>"
    if 0 <= eos_idx < vocab_size:
        idx2word[eos_idx] = "<EOS>"

    return {
        "vocab_size": vocab_size,
        "pad_idx": pad_idx,
        "sos_idx": sos_idx,
        "eos_idx": eos_idx,
        "max_seq_len": max_seq_len,
        "idx2word": idx2word,
    }


class MultiTaskDataset(Dataset):
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

        self.samples: List[Tuple[Path, List[int], torch.Tensor, str]] = []
        self.num_classes: Optional[int] = None
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
                cap_raw = row.get("caption_seq")
                label_raw = row.get("label_vec")
                if not filename or cap_raw is None or label_raw is None:
                    continue

                caption_seq = [int(x) for x in _parse_list_field(cap_raw)]
                if not caption_seq:
                    continue

                labels = [float(x) for x in _parse_list_field(label_raw)]
                if self.num_classes is None:
                    self.num_classes = len(labels)
                elif len(labels) != self.num_classes:
                    raise ValueError(
                        f"Inconsistent label_vec length: expected {self.num_classes}, got {len(labels)}"
                    )

                self.samples.append(
                    (
                        self.image_dir / filename,
                        caption_seq,
                        torch.tensor(labels, dtype=torch.float32),
                        filename,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, caption_seq, labels, filename = self.samples[idx]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        caption = torch.tensor(caption_seq, dtype=torch.long)
        return image, caption, labels, filename


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

