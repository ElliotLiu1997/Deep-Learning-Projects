"""
Dataset and dataloader
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional

import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from vocab import Vocabulary, parse_flickr8k_captions, load_split_image_names


def build_image_transform(image_size: int, mean: Tuple[float, ...], std: Tuple[float, ...]):
    """Image transform for ResNet18 preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _pad_or_truncate(seq: List[int], max_len: int, pad_idx: int) -> List[int]:
    """Pad or truncate sequence to fixed length."""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


class Flickr8kCaptionDataset(Dataset):
    """
    Flickr8k dataset for captioning.

    Returns:
        feature_or_image: FloatTensor
            - Cached mode: shape [feature_dim]
            - Raw mode: shape [3, H, W]
        input_seq: LongTensor [max_len - 1]
        target_seq: LongTensor [max_len - 1]
    """

    def __init__(
        self,
        images_dir: Path,
        captions_file: Path,
        split_file: Path,
        vocab: Vocabulary,
        max_caption_length: int = 30,
        use_feature_cache: bool = True,
        feature_npy_path: Optional[Path] = None,
        feature_keys_json_path: Optional[Path] = None,
        image_transform=None,
    ) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.max_caption_length = max_caption_length
        self.use_feature_cache = use_feature_cache
        self.image_transform = image_transform

        captions_by_image = parse_flickr8k_captions(Path(captions_file))
        split_images = set(load_split_image_names(Path(split_file)))

        # Build flat sample list: one row per (image, caption)
        self.samples: List[Tuple[str, str]] = []
        for image_name in split_images:
            for caption in captions_by_image.get(image_name, []):
                self.samples.append((image_name, caption))

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split file: {split_file}")

        # Cached feature structures
        self.feature_matrix = None
        self.feature_index_by_image = {}

        if self.use_feature_cache:
            if feature_npy_path is None or feature_keys_json_path is None:
                raise ValueError("feature_npy_path and feature_keys_json_path are required in cached mode.")

            self.feature_matrix = np.load(feature_npy_path)  # [num_images, feature_dim]
            with open(feature_keys_json_path, "r", encoding="utf-8") as f:
                keys = json.load(f)  # ordered image names aligned with rows in feature matrix

            self.feature_index_by_image = {name: i for i, name in enumerate(keys)}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_name: str) -> torch.Tensor:
        image_path = self.images_dir / image_name
        image = Image.open(image_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image

    def _load_feature(self, image_name: str) -> torch.Tensor:
        if image_name not in self.feature_index_by_image:
            raise KeyError(f"Image feature not found in cache: {image_name}")
        idx = self.feature_index_by_image[image_name]
        feat = self.feature_matrix[idx]  # [feature_dim]
        return torch.tensor(feat, dtype=torch.float32)

    def _encode_caption_pair(self, caption: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build pair:
        full sequence: <start> ... <end>
        input:  full[:-1]
        target: full[1:]
        """
        full = self.vocab.encode_caption(caption)
        full = _pad_or_truncate(full, self.max_caption_length, self.vocab.pad_idx)

        input_seq = torch.tensor(full[:-1], dtype=torch.long)
        target_seq = torch.tensor(full[1:], dtype=torch.long)
        return input_seq, target_seq

    def __getitem__(self, idx: int):
        image_name, caption = self.samples[idx]

        if self.use_feature_cache:
            feature_or_image = self._load_feature(image_name)
        else:
            feature_or_image = self._load_image(image_name)

        input_seq, target_seq = self._encode_caption_pair(caption)
        return feature_or_image, input_seq, target_seq


def caption_collate_fn(batch):
    """Collate function for fixed-length tensors."""
    features_or_images, input_seqs, target_seqs = zip(*batch)
    features_or_images = torch.stack(features_or_images, dim=0)
    input_seqs = torch.stack(input_seqs, dim=0)
    target_seqs = torch.stack(target_seqs, dim=0)
    return features_or_images, input_seqs, target_seqs
