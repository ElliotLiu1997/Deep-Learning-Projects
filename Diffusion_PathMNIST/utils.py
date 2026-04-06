import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_device(device_name: str = "cuda") -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)



def save_tensor_grid(x: torch.Tensor, path: Path, nrow: int = 8) -> None:
    ensure_dir(path.parent)
    save_image(denorm_to_01(x), str(path), nrow=nrow)



def plot_loss_curve(
    loss_history: List[float],
    path: Path,
    val_loss_history: Optional[List[float]] = None,
) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label="Train Loss")
    if val_loss_history is not None and len(val_loss_history) > 0:
        plt.plot(val_loss_history, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("DDPM Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()



def save_diffusion_process_grid(
    forward_steps: torch.Tensor,
    reverse_steps: torch.Tensor,
    path: Path,
) -> None:
    ensure_dir(path.parent)

    fwd = denorm_to_01(forward_steps)
    rev = denorm_to_01(reverse_steps)

    top = make_grid(fwd, nrow=fwd.shape[0], padding=2)
    bot = make_grid(rev, nrow=rev.shape[0], padding=2)
    merged = torch.cat([top, bot], dim=1)
    save_image(merged, str(path))



class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.register(model)

    def register(self, model: nn.Module) -> None:
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])



def save_checkpoint(
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss_history: List[float],
    path: Path,
    extra: Optional[dict] = None,
) -> None:
    ensure_dir(path.parent)
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.shadow,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss_history": loss_history,
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, str(path))



def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        # Backward compatibility for older torch versions without weights_only.
        return torch.load(str(path), map_location=map_location)



def append_metrics_row(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    file_exists = path.exists()
    columns = ["model", "steps", "FID", "IS", "Precision", "Recall", "sampling_time"]

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
