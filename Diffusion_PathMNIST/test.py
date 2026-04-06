import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import NpyImageDataset, create_dataloaders, resolve_data_dir


def stats_np(name: str, arr: np.ndarray) -> None:
    print(
        f"[NPY] {name}: shape={arr.shape}, dtype={arr.dtype}, "
        f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, std={arr.std():.6f}"
    )


def stats_torch(name: str, x: torch.Tensor) -> None:
    print(
        f"[TORCH] {name}: shape={tuple(x.shape)}, dtype={x.dtype}, "
        f"min={x.min().item():.6f}, max={x.max().item():.6f}, "
        f"mean={x.mean().item():.6f}, std={x.std().item():.6f}"
    )


def check_range(name: str, x: torch.Tensor, low: float = -1.0, high: float = 1.0, eps: float = 1e-4) -> None:
    xmin = x.min().item()
    xmax = x.max().item()
    ok = (xmin >= low - eps) and (xmax <= high + eps)
    status = "PASS" if ok else "FAIL"
    print(f"[RANGE] {name}: {status} (expected in [{low}, {high}], got [{xmin:.6f}, {xmax:.6f}])")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check PathMNIST scaling and tensor ranges")
    parser.add_argument("--data_dir", type=str, default="pathmnist")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    root = resolve_data_dir(args.data_dir)
    print(f"Data dir: {root}")

    # 1) Raw .npy stats
    train_images = np.load(Path(root) / "train_images.npy")
    val_images = np.load(Path(root) / "val_images.npy")
    test_images = np.load(Path(root) / "test_images.npy")

    stats_np("train_images.npy", train_images)
    stats_np("val_images.npy", val_images)
    stats_np("test_images.npy", test_images)

    # 2) Dataset-transformed stats (after normalization + NHWC->NCHW)
    train_ds = NpyImageDataset(Path(root) / "train_images.npy", Path(root) / "train_labels.npy")
    val_ds = NpyImageDataset(Path(root) / "val_images.npy", Path(root) / "val_labels.npy")
    test_ds = NpyImageDataset(Path(root) / "test_images.npy", Path(root) / "test_labels.npy")

    stats_torch("train_ds.images_t", train_ds.images_t)
    check_range("train_ds.images_t", train_ds.images_t)
    stats_torch("val_ds.images_t", val_ds.images_t)
    check_range("val_ds.images_t", val_ds.images_t)
    stats_torch("test_ds.images_t", test_ds.images_t)
    check_range("test_ds.images_t", test_ds.images_t)

    # 3) Dataloader batch check
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    xb, yb = next(iter(train_loader))
    print(f"[BATCH] train: x.shape={tuple(xb.shape)}, y.shape={tuple(yb.shape)}")
    stats_torch("train batch x", xb)
    check_range("train batch x", xb)

    xb, yb = next(iter(val_loader))
    print(f"[BATCH] val: x.shape={tuple(xb.shape)}, y.shape={tuple(yb.shape)}")
    stats_torch("val batch x", xb)
    check_range("val batch x", xb)

    xb, yb = next(iter(test_loader))
    print(f"[BATCH] test: x.shape={tuple(xb.shape)}, y.shape={tuple(yb.shape)}")
    stats_torch("test batch x", xb)
    check_range("test batch x", xb)


if __name__ == "__main__":
    main()
