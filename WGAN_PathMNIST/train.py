import argparse
from pathlib import Path

import torch
from torch import autograd
from torch.optim import Adam
from tqdm import tqdm

from dataset import create_dataloaders
from metrics import (
    InceptionExtractor,
    compute_fid,
    compute_inception_score,
    compute_precision_recall,
    extract_features_and_probs,
    gather_real_images,
    generate_images,
)
from models import Critic, Generator, init_weights
from utils import (
    append_metrics_row,
    ensure_dir,
    parse_gpu_ids,
    plot_gan_losses,
    resolve_device,
    save_latent_interpolation,
    save_tensor_grid,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WGAN-GP on PathMNIST")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_ids", type=str, default="0,1")

    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--lambda_gp", type=float, default=5.0)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    return parser.parse_args()


def gradient_penalty(
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    bsz = real.shape[0]
    alpha = torch.rand(bsz, 1, 1, 1, device=device)
    interpolates = alpha * real + (1.0 - alpha) * fake
    interpolates.requires_grad_(True)

    with torch.cuda.amp.autocast(enabled=False):
        critic_interpolates = critic(interpolates.float()).view(-1)

    grad_outputs = torch.ones_like(critic_interpolates, device=device)
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(bsz, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = "pathmnist"
    output_dir = Path("outputs")

    gpu_ids = []
    if torch.cuda.is_available() and args.device == "cuda":
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        if len(gpu_ids) == 0:
            gpu_ids = list(range(torch.cuda.device_count()))

    device = resolve_device(args.device, gpu_ids)
    use_dp = device.type == "cuda" and len(gpu_ids) > 1
    use_amp = device.type == "cuda"

    samples_dir = output_dir / "samples"
    plots_dir = output_dir / "plots"
    ckpt_dir = output_dir / "checkpoints"
    ensure_dir(samples_dir)
    ensure_dir(plots_dir)
    ensure_dir(ckpt_dir)

    train_loader, val_loader, _ = create_dataloaders(
        data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    image_shape = train_loader.dataset[0][0].shape
    img_channels = image_shape[0]

    generator = Generator(z_dim=args.z_dim, img_channels=img_channels).to(device)
    critic = Critic(img_channels=img_channels).to(device)
    generator.apply(init_weights)
    critic.apply(init_weights)

    if use_dp:
        generator = torch.nn.DataParallel(generator, device_ids=gpu_ids)
        critic = torch.nn.DataParallel(critic, device_ids=gpu_ids)
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    else:
        print(f"Using device: {device}")

    opt_g = Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = Adam(critic.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_d = torch.cuda.amp.GradScaler(enabled=use_amp)

    g_losses = []
    d_losses = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        generator.train()
        critic.train()

        epoch_g = 0.0
        epoch_d = 0.0
        g_updates = 0
        d_updates = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for real, _ in pbar:
            real = real.to(device, non_blocking=True)
            bsz = real.shape[0]

            z = torch.randn(bsz, args.z_dim, device=device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake = generator(z).detach()
                d_real = critic(real).view(-1)
                d_fake = critic(fake).view(-1)
                wasserstein = d_fake.mean() - d_real.mean()

            gp = gradient_penalty(critic, real, fake, device=device)
            d_loss = wasserstein + args.lambda_gp * gp

            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            if args.grad_clip > 0:
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
            scaler_d.step(opt_d)
            scaler_d.update()

            d_loss_item = float(d_loss.detach().item())
            d_losses.append(d_loss_item)
            epoch_d += d_loss_item
            d_updates += 1

            if (global_step + 1) % args.n_critic == 0:
                z = torch.randn(bsz, args.z_dim, device=device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    gen = generator(z)
                    g_loss = -critic(gen).view(-1).mean()

                opt_g.zero_grad(set_to_none=True)
                scaler_g.scale(g_loss).backward()
                if args.grad_clip > 0:
                    scaler_g.unscale_(opt_g)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
                scaler_g.step(opt_g)
                scaler_g.update()

                g_loss_item = float(g_loss.detach().item())
                g_losses.append(g_loss_item)
                epoch_g += g_loss_item
                g_updates += 1
                pbar.set_postfix(d_loss=f"{d_loss_item:.4f}", g_loss=f"{g_loss_item:.4f}")
            else:
                pbar.set_postfix(d_loss=f"{d_loss_item:.4f}")

            global_step += 1

        avg_d = epoch_d / max(d_updates, 1)
        avg_g = epoch_g / max(g_updates, 1)
        print(
            f"[Epoch {epoch}] d_loss={avg_d:.6f} g_loss={avg_g:.6f} "
            f"train_steps={len(train_loader)} val_steps={len(val_loader)}"
        )

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            generator.eval()
            with torch.no_grad():
                z = torch.randn(16, args.z_dim, device=device)
                fake16 = generator(z).cpu()
            save_tensor_grid(fake16, samples_dir / f"gan_epoch_{epoch}.png", nrow=4)

        if epoch % 25 == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "generator": generator.module.state_dict() if isinstance(generator, torch.nn.DataParallel) else generator.state_dict(),
                "critic": critic.module.state_dict() if isinstance(critic, torch.nn.DataParallel) else critic.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "g_losses": g_losses,
                "d_losses": d_losses,
                "z_dim": args.z_dim,
                "image_shape": tuple(image_shape),
            }
            torch.save(ckpt, ckpt_dir / "wgan_gp_latest.pt")

    plot_gan_losses(g_losses, d_losses, plots_dir / "gan_loss_curve.png")

    _, _, test_loader = create_dataloaders(
        data_dir,
        batch_size=args.eval_batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    real_images = gather_real_images(test_loader, args.num_samples)
    fake_images, sampling_time = generate_images(
        generator=generator,
        num_samples=args.num_samples,
        batch_size=args.eval_batch_size,
        z_dim=args.z_dim,
        device=device,
    )

    extractor = InceptionExtractor()
    real_feats, _ = extract_features_and_probs(
        extractor,
        real_images,
        batch_size=args.eval_batch_size,
        device=device,
    )
    gen_feats, gen_probs = extract_features_and_probs(
        extractor,
        fake_images,
        batch_size=args.eval_batch_size,
        device=device,
    )

    fid = compute_fid(real_images, fake_images)
    is_score = compute_inception_score(gen_probs)
    precision, recall = compute_precision_recall(real_feats, gen_feats, k=5)

    row = {
        "model": "WGAN-GP",
        "steps": "N/A",
        "FID": f"{fid:.6f}",
        "IS": f"{is_score:.6f}",
        "Precision": f"{precision:.6f}",
        "Recall": f"{recall:.6f}",
        "sampling_time": f"{sampling_time:.6f}",
    }
    append_metrics_row(output_dir / "metrics.csv", row)

    real16 = real_images[:16]
    fake16 = fake_images[:16]
    save_tensor_grid(real16, samples_dir / "gan_real_test_4x4.png", nrow=4)
    save_tensor_grid(fake16, samples_dir / "gan_generated_4x4.png", nrow=4)
    save_latent_interpolation(generator, args.z_dim, device, samples_dir / "gan_latent_interp.png", steps=8)

    print("GAN evaluation row:", row)
    print(f"Saved loss curve to: {plots_dir / 'gan_loss_curve.png'}")
    print(f"Saved samples to: {samples_dir}")


if __name__ == "__main__":
    main()
