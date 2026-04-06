import argparse
import math
import time
from pathlib import Path

import torch

from config import Config
from dataset import create_dataloaders
from diffusion import GaussianDiffusion
from model_unet import UNet
from utils import (
    denorm_to_01,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_diffusion_process_grid,
    save_tensor_grid,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images using DDPM and DDIM")
    parser.add_argument("--checkpoint", type=str, default="diffusion_project/outputs/latest.pt")
    parser.add_argument("--data_dir", type=str, default="pathmnist")
    parser.add_argument("--output_dir", type=str, default="diffusion_project/outputs")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, nargs="+", default=[100, 50])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated GPU ids, e.g. 0,1")
    return parser.parse_args()



def load_ema_model(ckpt: dict, device: torch.device) -> UNet:
    extra = ckpt["extra"]
    image_shape = extra["image_shape"]

    model = UNet(
        in_channels=image_shape[0],
        base_channels=extra["base_channels"],
        channel_mults=tuple(extra["channel_mults"]),
    ).to(device)

    model.load_state_dict(ckpt["model"])

    for name, param in model.named_parameters():
        if name in ckpt["ema"]:
            param.data.copy_(ckpt["ema"][name].to(device))

    model.eval()
    return model



def main() -> None:
    args = parse_args()
    cfg = Config(output_dir=args.output_dir)
    gpu_ids = []
    if torch.cuda.is_available() and args.device == "cuda":
        if args.gpu_ids.strip():
            gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
    use_data_parallel = len(gpu_ids) > 1
    if torch.cuda.is_available() and args.device == "cuda":
        device = torch.device(f"cuda:{gpu_ids[0] if gpu_ids else 0}")
    else:
        device = get_device(args.device)
    ensure_dir(Path(cfg.output_dir))
    ensure_dir(cfg.samples_dir)
    ensure_dir(cfg.process_dir)

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model = load_ema_model(ckpt, device)
    if use_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    image_shape = ckpt["extra"]["image_shape"]
    timesteps = ckpt["extra"]["timesteps"]
    beta_start = ckpt["extra"].get("beta_start", 1e-4)
    beta_end = ckpt["extra"].get("beta_end", 2e-2)
    noise_schedule = ckpt["extra"].get("noise_schedule", "linear")
    diffusion = GaussianDiffusion(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        noise_schedule=noise_schedule,
    ).to(device)

    n = args.num_samples
    shape = (n, image_shape[0], image_shape[1], image_shape[2])
    nrow = max(int(math.sqrt(n)), 1)

    t0 = time.perf_counter()
    ddpm = diffusion.sample_ddpm(model, shape=shape, device=device)
    ddpm_time = time.perf_counter() - t0
    save_tensor_grid(ddpm, cfg.samples_dir / f"ddpm_steps_{timesteps}.png", nrow=nrow)

    ddim_results = []
    for steps in args.ddim_steps:
        t1 = time.perf_counter()
        ddim = diffusion.sample_ddim(model, shape=shape, device=device, steps=steps, eta=0.0)
        ddim_time = time.perf_counter() - t1
        ddim_results.append((steps, ddim, ddim_time))
        save_tensor_grid(ddim, cfg.samples_dir / f"ddim_steps_{steps}.png", nrow=nrow)

    # Save DDPM vs DDIM visual comparison
    comp_left = denorm_to_01(ddpm[: nrow])
    comp_right = denorm_to_01(ddim_results[0][1][: nrow])
    comparison = torch.cat([comp_left, comp_right], dim=0)
    save_tensor_grid(comparison * 2 - 1, cfg.samples_dir / "ddpm_vs_ddim.png", nrow=nrow)

    # Diffusion process visualization: forward noising + reverse denoising
    _, _, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=1,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    x0, _ = next(iter(test_loader))
    x0 = x0.to(device)

    t_seq = torch.linspace(0, timesteps - 1, steps=8, device=device).long()
    forward_steps = []
    for t in t_seq:
        tt = torch.tensor([t.item()], device=device, dtype=torch.long)
        xt = diffusion.q_sample(x0, tt)
        forward_steps.append(xt[0].detach().cpu())
    forward_steps = torch.stack(forward_steps, dim=0)

    _, trajectory = diffusion.sample_ddpm_with_trajectory(
        model,
        shape=(1, image_shape[0], image_shape[1], image_shape[2]),
        device=device,
        snapshots=8,
    )
    reverse_steps = torch.stack([s[0].detach().cpu() for s in trajectory], dim=0)

    save_diffusion_process_grid(
        forward_steps=forward_steps,
        reverse_steps=reverse_steps,
        path=cfg.process_dir / "process.png",
    )

    print(f"DDPM sampling time ({timesteps} steps): {ddpm_time:.3f}s")
    for steps, _, s_time in ddim_results:
        print(f"DDIM sampling time ({steps} steps): {s_time:.3f}s")


if __name__ == "__main__":
    main()
