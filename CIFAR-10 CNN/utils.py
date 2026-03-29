import os
import random
import csv
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_wrap_data_parallel(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def get_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ]
    )

    full_train = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=False)
    test_set = datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=False)

    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [45000, 5000], generator=split_generator)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    classes = full_train.classes
    return train_loader, val_loader, test_loader, classes


def save_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    output_path: str,
    model_title: str = "Model",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_title} Train/Validation Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_val_accuracy_curve(
    train_accuracies: List[float],
    val_accuracies: List[float],
    output_path: str,
    model_title: str = "Model",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_title} Train/Validation Accuracy Curve")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    output_path: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_training_history_csv(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    output_path: str,
) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for idx, (tr_loss, va_loss, tr_acc, va_acc) in enumerate(
            zip(train_losses, val_losses, train_accuracies, val_accuracies), start=1
        ):
            writer.writerow([idx, tr_loss, va_loss, tr_acc, va_acc])


def save_test_predictions_csv(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    output_path: str,
) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "true_label", "pred_label", "true_class", "pred_class"])
        for idx, (true_id, pred_id) in enumerate(zip(y_true, y_pred)):
            writer.writerow([idx, true_id, pred_id, classes[true_id], classes[pred_id]])
