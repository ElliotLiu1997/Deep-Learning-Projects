from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class Config:
    data_dir: str = "pathmnist"
    output_dir: str = "diffusion_project/outputs"

    image_channels: int = 3
    base_channels: int = 64
    channel_mults: tuple = (1, 2, 4, 8)
    num_groups: int = 8

    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    noise_schedule: str = "cosine"

    epochs: int = 50
    batch_size: int = 128
    lr: float = 5e-5
    weight_decay: float = 0.0
    num_workers: int = 4

    ema_decay: float = 0.999
    grad_clip: float = 1.0

    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    sample_every: int = 5
    save_every: int = 5
    sample_grid_size: int = 16

    eval_num_samples: int = 2000
    eval_batch_size: int = 128
    eval_ddim_steps: tuple = (100, 50)

    ckpt_name: str = "latest.pt"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / self.ckpt_name

    @property
    def samples_dir(self) -> Path:
        return Path(self.output_dir) / "samples"

    @property
    def plots_dir(self) -> Path:
        return Path(self.output_dir) / "plots"

    @property
    def process_dir(self) -> Path:
        return Path(self.output_dir) / "diffusion_process"

    @property
    def metrics_path(self) -> Path:
        return Path(self.output_dir) / "metrics.csv"


DEFAULT_CONFIG = Config()
