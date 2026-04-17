from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NpyImageDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path)

        if self.images.ndim != 4:
            raise ValueError(f"Expected images shape (N,H,W,C), got {self.images.shape}")

        images = self.images.astype(np.float32)
        if images.max() > 1.5:
            images = images / 127.5 - 1.0
        else:
            images = images * 2.0 - 1.0

        self.images_t = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        self.labels_t = torch.from_numpy(self.labels).long().view(-1)

    def __len__(self) -> int:
        return self.images_t.shape[0]

    def __getitem__(self, idx: int):
        return self.images_t[idx], self.labels_t[idx]


def resolve_data_dir(data_dir: str) -> Path:
    p = Path(data_dir)
    if p.exists():
        return p

    alt = Path("..").joinpath(data_dir)
    if alt.exists():
        return alt

    raise FileNotFoundError(f"Could not find data directory: {data_dir}")


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root = resolve_data_dir(data_dir)

    train_ds = NpyImageDataset(root / "train_images.npy", root / "train_labels.npy")
    val_ds = NpyImageDataset(root / "val_images.npy", root / "val_labels.npy")
    test_ds = NpyImageDataset(root / "test_images.npy", root / "test_labels.npy")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
