import csv
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from .utils import parse_label_vector
except ImportError:
    from utils import parse_label_vector


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChestXrayDataset(Dataset):
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

        self.samples = []
        self.num_classes = None
        self._load_rows()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split='{self.split}'")

    def _load_rows(self) -> None:
        with self.data_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = (row.get("split") or "").strip().lower()
                if row_split != self.split:
                    continue

                filename = row.get("filename")
                label_raw = row.get("label_vec")
                if not filename or label_raw is None:
                    continue

                label_list = parse_label_vector(label_raw)
                if self.num_classes is None:
                    self.num_classes = len(label_list)
                elif len(label_list) != self.num_classes:
                    raise ValueError(
                        "Inconsistent label vector length found in CSV. "
                        f"Expected {self.num_classes}, got {len(label_list)}"
                    )

                image_path = self.image_dir / filename
                self.samples.append((image_path, torch.tensor(label_list, dtype=torch.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_vec = self.samples[idx]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label_vec


def build_train_transforms(encoder: str = "resnet") -> transforms.Compose:
    encoder = encoder.lower()
    if encoder == "vit":
        # ViT is data-hungry; stronger augmentation helps reduce overfitting.
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_transforms() -> transforms.Compose:
    # Backward-compatible alias (evaluation/no augmentation pipeline).
    return build_eval_transforms()
