"""
Extract and cache frozen ResNet18 features for Flickr8k splits.

Outputs:
- features_train.npy / features_val.npy / features_test.npy
- feature_keys_train.json / feature_keys_val.json / feature_keys_test.json
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from config import get_config
from dataset import build_image_transform
from vocab import load_split_image_names
from model import CNNGlobalEncoder


class ImageOnlyDataset(Dataset):
    """Loads images only (no captions) for feature extraction."""

    def __init__(self, images_dir: Path, image_names: List[str], transform=None):
        self.images_dir = Path(images_dir)
        self.image_names = image_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image = Image.open(self.images_dir / image_name).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image_name, image


def _collate_image_only(batch):
    names, images = zip(*batch)
    return list(names), torch.stack(images, dim=0)


@torch.no_grad()
def extract_features_for_split(
    encoder: CNNGlobalEncoder,
    images_dir: Path,
    split_file: Path,
    image_size: int,
    image_mean: Tuple[float, ...],
    image_std: Tuple[float, ...],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    """
    Extract raw 512-d frozen ResNet features (before projection layer).
    Returns:
        image_names (ordered list), features ndarray [N, 512]
    """
    image_names = load_split_image_names(split_file)
    transform = build_image_transform(image_size=image_size, mean=image_mean, std=image_std)

    ds = ImageOnlyDataset(images_dir=images_dir, image_names=image_names, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_image_only,
    )

    encoder.eval()
    all_names: List[str] = []
    all_features = []

    for batch_names, batch_images in tqdm(dl, desc=f"Extracting {split_file.name}", leave=False):
        batch_images = batch_images.to(device, non_blocking=True)
        feats = encoder.cnn(batch_images)      # [B, 512, 1, 1]
        feats = feats.flatten(1).cpu().numpy() # [B, 512]

        all_names.extend(batch_names)
        all_features.append(feats)

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    return all_names, features


def save_feature_cache(feature_path: Path, keys_path: Path, keys: List[str], features: np.ndarray) -> None:
    """Save feature matrix and aligned image-name key list."""
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, features)
    with open(keys_path, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2)


def build_all_feature_caches() -> None:
    """Build train/val/test feature caches."""
    cfg = get_config()
    device = cfg.device

    encoder = CNNGlobalEncoder(embed_dim=cfg.embed_dim).to(device)
    encoder.eval()

    split_specs = [
        (cfg.train_split_file, cfg.features_train_path, cfg.feature_keys_train_path),
        (cfg.val_split_file, cfg.features_val_path, cfg.feature_keys_val_path),
        (cfg.test_split_file, cfg.features_test_path, cfg.feature_keys_test_path),
    ]

    for split_file, feat_path, key_path in split_specs:
        keys, feats = extract_features_for_split(
            encoder=encoder,
            images_dir=cfg.images_dir,
            split_file=split_file,
            image_size=cfg.image_size,
            image_mean=cfg.image_mean,
            image_std=cfg.image_std,
            device=device,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and device.type == "cuda",
        )
        save_feature_cache(feat_path, key_path, keys, feats)
        print(f"Saved {split_file.name}: {feats.shape} -> {feat_path}")


if __name__ == "__main__":
    build_all_feature_caches()
