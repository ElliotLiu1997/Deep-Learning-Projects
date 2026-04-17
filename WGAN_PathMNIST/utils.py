import csv
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image


METRIC_COLUMNS = ["model", "steps", "FID", "IS", "Precision", "Recall", "sampling_time"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_gpu_ids(gpu_ids: str) -> List[int]:
    ids = [s.strip() for s in gpu_ids.split(",") if s.strip()]
    return [int(x) for x in ids]


def resolve_device(device_name: str, gpu_ids: Sequence[int]) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        if len(gpu_ids) > 0:
            return torch.device(f"cuda:{gpu_ids[0]}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def save_tensor_grid(x: torch.Tensor, path: Path, nrow: int = 4) -> None:
    ensure_dir(path.parent)
    save_image(denorm_to_01(x), str(path), nrow=nrow)


def save_real_fake_comparison(real: torch.Tensor, fake: torch.Tensor, path: Path) -> None:
    ensure_dir(path.parent)
    pair = torch.cat([real, fake], dim=0)
    save_image(denorm_to_01(pair), str(path), nrow=4)


def plot_gan_losses(g_losses: Iterable[float], d_losses: Iterable[float], path: Path) -> None:
    ensure_dir(path.parent)
    g_losses = list(g_losses)
    d_losses = list(d_losses)

    plt.figure(figsize=(7, 4))
    if d_losses:
        plt.plot(d_losses, label="Critic Loss")
    if g_losses:
        plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("WGAN-GP Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def append_metrics_row(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_latent_interpolation(
    generator: torch.nn.Module,
    z_dim: int,
    device: torch.device,
    path: Path,
    steps: int = 8,
) -> None:
    generator.eval()
    with torch.no_grad():
        z0 = torch.randn(1, z_dim, device=device)
        z1 = torch.randn(1, z_dim, device=device)
        alphas = torch.linspace(0.0, 1.0, steps=steps, device=device).view(-1, 1)
        z = (1.0 - alphas) * z0 + alphas * z1
        imgs = generator(z)
    save_tensor_grid(imgs.cpu(), path, nrow=steps)
