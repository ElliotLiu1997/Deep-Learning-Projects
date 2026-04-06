import argparse
import math
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from diffusion import GaussianDiffusion
from model_unet import UNet
from utils import (
    EMA,
    ensure_dir,
    get_device,
    plot_loss_curve,
    save_checkpoint,
    save_tensor_grid,
    set_seed,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPM on PathMNIST (.npy)")
    parser.add_argument("--data_dir", type=str, default="pathmnist")
    parser.add_argument("--output_dir", type=str, default="diffusion_project/outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated GPU ids, e.g. 0,1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        timesteps=args.timesteps,
        noise_schedule=args.noise_schedule,
        ema_decay=args.ema_decay,
        sample_every=args.sample_every,
        save_every=args.save_every,
        device=args.device,
        seed=args.seed,
    )

    set_seed(cfg.seed)

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
        device = get_device(cfg.device)

    ensure_dir(Path(cfg.output_dir))
    ensure_dir(cfg.samples_dir)
    ensure_dir(cfg.plots_dir)
    ensure_dir(cfg.process_dir)

    train_loader, val_loader, _ = create_dataloaders(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    image_shape = train_loader.dataset[0][0].shape

    model = UNet(
        in_channels=image_shape[0],
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
    ).to(device)
    train_model = (
        torch.nn.DataParallel(model, device_ids=gpu_ids) if use_data_parallel else model
    )

    ema_model = UNet(
        in_channels=image_shape[0],
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
    ).to(device)

    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False

    ema = EMA(model, decay=cfg.ema_decay)
    diffusion = GaussianDiffusion(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        noise_schedule=cfg.noise_schedule,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if use_data_parallel:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        print(f"Using device: {device}")

    loss_history = []
    val_loss_history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            t = torch.randint(0, cfg.timesteps, (x.shape[0],), device=device).long()

            loss = diffusion.training_losses(train_model, x, t)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(model)
            ema.apply_to(ema_model)

            total_loss += loss.item() * x.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = total_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)

        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                t_val = torch.randint(0, cfg.timesteps, (x_val.shape[0],), device=device).long()
                val_loss = diffusion.training_losses(model, x_val, t_val)
                val_total_loss += val_loss.item() * x_val.shape[0]

        epoch_val_loss = val_total_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)

        print(
            f"[Epoch {epoch}] train_loss={epoch_loss:.6f} "
            f"val_loss={epoch_val_loss:.6f}"
        )

        plot_loss_curve(
            loss_history,
            cfg.plots_dir / "loss_curve.png",
            val_loss_history=val_loss_history,
        )

        if epoch % cfg.sample_every == 0 or epoch == 1:
            ema_model.eval()
            with torch.no_grad():
                n = cfg.sample_grid_size
                samples = diffusion.sample_ddim(
                    ema_model,
                    shape=(n, image_shape[0], image_shape[1], image_shape[2]),
                    device=device,
                    steps=100,
                    eta=0.0,
                )
            nrow = int(math.sqrt(n))
            save_tensor_grid(samples, cfg.samples_dir / f"epoch_{epoch}.png", nrow=max(nrow, 1))

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            save_checkpoint(
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                loss_history=loss_history,
                path=cfg.checkpoint_path,
                extra={
                    "image_shape": tuple(image_shape),
                    "timesteps": cfg.timesteps,
                    "base_channels": cfg.base_channels,
                    "channel_mults": tuple(cfg.channel_mults),
                    "beta_start": cfg.beta_start,
                    "beta_end": cfg.beta_end,
                    "noise_schedule": cfg.noise_schedule,
                    "data_dir": cfg.data_dir,
                    "val_loss_history": val_loss_history,
                },
            )

    print(f"Training complete. Checkpoint: {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
