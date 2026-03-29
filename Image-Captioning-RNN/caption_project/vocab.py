"""
Vocabulary utilities for Flickr8k captions.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import json
import re
from typing import Dict, List


def tokenize(text: str) -> List[str]:
    """Lowercase and tokenize caption text into word tokens."""
    text = text.lower().strip()
    return re.findall(r"[a-z0-9']+", text)


def parse_flickr8k_captions(captions_file: Path) -> Dict[str, List[str]]:
    """
    Parse Flickr8k.token.txt into:
    {image_name: [caption1, ..., caption5]}
    """
    image_to_captions: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: image.jpg#0<TAB>caption text
            left, caption = line.split("\t", maxsplit=1)
            image_name = left.split("#")[0]
            image_to_captions.setdefault(image_name, []).append(caption.strip())
    return image_to_captions


@dataclass
class Vocabulary:
    """Bidirectional word-index mapping with special token support."""

    token_to_idx: Dict[str, int]
    idx_to_token: Dict[int, str]

    PAD: str = "<pad>"
    START: str = "<start>"
    END: str = "<end>"
    UNK: str = "<unk>"

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[self.PAD]

    @property
    def start_idx(self) -> int:
        return self.token_to_idx[self.START]

    @property
    def end_idx(self) -> int:
        return self.token_to_idx[self.END]

    @property
    def unk_idx(self) -> int:
        return self.token_to_idx[self.UNK]

    @property
    def size(self) -> int:
        return len(self.token_to_idx)

    def numericalize(self, tokens: List[str]) -> List[int]:
        """Convert token list to index list."""
        return [self.token_to_idx.get(t, self.unk_idx) for t in tokens]

    def denumericalize(self, indices: List[int], stop_at_end: bool = True) -> List[str]:
        """Convert index list back to tokens."""
        out = []
        for idx in indices:
            tok = self.idx_to_token.get(int(idx), self.UNK)
            if stop_at_end and tok == self.END:
                break
            out.append(tok)
        return out

    def encode_caption(self, caption: str) -> List[int]:
        """Encode raw caption text with <start> and <end>."""
        tokens = tokenize(caption)
        return [self.start_idx] + self.numericalize(tokens) + [self.end_idx]

    @classmethod
    def build_from_captions(
        cls,
        captions_by_image: Dict[str, List[str]],
        min_word_freq: int = 1,
    ) -> "Vocabulary":
        """Build vocabulary from all captions."""
        counter = Counter()
        for captions in captions_by_image.values():
            for cap in captions:
                counter.update(tokenize(cap))

        specials = [cls.PAD, cls.START, cls.END, cls.UNK]
        token_to_idx: Dict[str, int] = {tok: i for i, tok in enumerate(specials)}

        for token, freq in counter.items():
            if freq >= min_word_freq and token not in token_to_idx:
                token_to_idx[token] = len(token_to_idx)

        idx_to_token = {idx: tok for tok, idx in token_to_idx.items()}
        return cls(token_to_idx=token_to_idx, idx_to_token=idx_to_token)

    def to_json(self, path: Path) -> None:
        """Save vocabulary to JSON."""
        data = {
            "token_to_idx": self.token_to_idx,
            "idx_to_token": {str(k): v for k, v in self.idx_to_token.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        token_to_idx = {str(k): int(v) for k, v in data["token_to_idx"].items()}
        idx_to_token = {int(k): str(v) for k, v in data["idx_to_token"].items()}
        return cls(token_to_idx=token_to_idx, idx_to_token=idx_to_token)


def load_split_image_names(split_file: Path) -> List[str]:
    """Read image names from Flickr8k split file."""
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_and_save_vocab(
    captions_file: Path,
    vocab_path: Path,
    min_word_freq: int = 1,
) -> Vocabulary:
    """Convenience helper to build vocabulary from caption file and save it."""
    captions_by_image = parse_flickr8k_captions(captions_file)
    vocab = Vocabulary.build_from_captions(captions_by_image, min_word_freq=min_word_freq)
    vocab.to_json(vocab_path)
    return vocab
