import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import Inception_V3_Weights, inception_v3

from config import Config
from dataset import create_dataloaders
from diffusion import GaussianDiffusion
from model_unet import UNet
from utils import append_metrics_row, ensure_dir, get_device, load_checkpoint



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DDPM/DDIM on PathMNIST")
    parser.add_argument("--checkpoint", type=str, default="diffusion_project/outputs/latest.pt")
    parser.add_argument("--data_dir", type=str, default="pathmnist")
    parser.add_argument("--output_dir", type=str, default="diffusion_project/outputs")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ddim_steps", type=int, nargs="+", default=[100, 50])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_ids", type=str, default="", help="Comma-separated GPU ids, e.g. 0,1")
    return parser.parse_args()



class InceptionExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # torchvision compatibility: pretrained inception_v3 may enforce aux_logits=True.
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.model.eval()
        self._features = None

        def hook_fn(_, __, output):
            self._features = torch.flatten(output, 1)

        self.model.avgpool.register_forward_hook(hook_fn)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x in [-1,1] -> [0,1] -> resize -> ImageNet normalize
        x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        logits = self.model(x)
        if not torch.is_tensor(logits):
            # Handle InceptionOutputs or tuple outputs across torchvision versions.
            if hasattr(logits, "logits"):
                logits = logits.logits
            else:
                logits = logits[0]
        feats = self._features
        return feats, logits



def load_ema_model(ckpt: dict, device: torch.device) -> UNet:
    extra = ckpt["extra"]
    image_shape = extra["image_shape"]
    model = UNet(
        in_channels=image_shape[0],
        base_channels=extra["base_channels"],
        channel_mults=tuple(extra["channel_mults"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    for name, p in model.named_parameters():
        if name in ckpt["ema"]:
            p.data.copy_(ckpt["ema"][name].to(device))

    model.eval()
    return model



def gather_real_images(test_loader: DataLoader, num_samples: int) -> torch.Tensor:
    chunks = []
    total = 0
    for x, _ in test_loader:
        chunks.append(x)
        total += x.shape[0]
        if total >= num_samples:
            break
    return torch.cat(chunks, dim=0)[:num_samples]



@torch.no_grad()
def generate_images(
    diffusion: GaussianDiffusion,
    model: UNet,
    num_samples: int,
    batch_size: int,
    image_shape: Tuple[int, int, int],
    device: torch.device,
    method: str,
    steps: int,
) -> Tuple[torch.Tensor, float]:
    all_samples = []
    start = time.perf_counter()
    generated = 0

    while generated < num_samples:
        cur_bs = min(batch_size, num_samples - generated)
        shape = (cur_bs, image_shape[0], image_shape[1], image_shape[2])

        if method == "DDPM":
            x = diffusion.sample_ddpm(model, shape=shape, device=device)
        elif method == "DDIM":
            x = diffusion.sample_ddim(model, shape=shape, device=device, steps=steps, eta=0.0)
        else:
            raise ValueError(f"Unknown method: {method}")

        all_samples.append(x.cpu())
        generated += cur_bs

    elapsed = time.perf_counter() - start
    return torch.cat(all_samples, dim=0), elapsed



@torch.no_grad()
def extract_features_and_probs(
    extractor: InceptionExtractor,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = TensorDataset(images)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    feats_list = []
    prob_list = []

    extractor = extractor.to(device)
    extractor.eval()

    for (x,) in dl:
        x = x.to(device)
        feats, logits = extractor(x)
        if feats is None:
            raise RuntimeError(
                "Inception feature extraction failed (got None). "
                "Avoid wrapping InceptionExtractor with DataParallel when using forward hooks."
            )
        probs = torch.softmax(logits, dim=1)
        feats_list.append(feats.cpu())
        prob_list.append(probs.cpu())

    return torch.cat(feats_list, dim=0), torch.cat(prob_list, dim=0)



def compute_fid(real_feats: torch.Tensor, gen_feats: torch.Tensor) -> float:
    r = real_feats.numpy().astype(np.float64)
    g = gen_feats.numpy().astype(np.float64)

    mu_r, mu_g = r.mean(axis=0), g.mean(axis=0)
    sigma_r = np.cov(r, rowvar=False)
    sigma_g = np.cov(g, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean)
    return float(np.real(fid))



def compute_inception_score(probs: torch.Tensor, splits: int = 10) -> float:
    probs = probs.numpy().astype(np.float64)
    n = probs.shape[0]
    split_size = max(n // splits, 1)

    scores = []
    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        if len(part) == 0:
            continue
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))

    return float(np.mean(scores)) if scores else float("nan")



def compute_precision_recall(
    real_feats: torch.Tensor,
    gen_feats: torch.Tensor,
    k: int = 5,
) -> Tuple[float, float]:
    real = real_feats.float()
    gen = gen_feats.float()

    d_rr = torch.cdist(real, real)
    d_rr.fill_diagonal_(float("inf"))
    r_real = d_rr.topk(k, largest=False).values[:, -1]

    d_gg = torch.cdist(gen, gen)
    d_gg.fill_diagonal_(float("inf"))
    r_gen = d_gg.topk(k, largest=False).values[:, -1]

    d_gr = torch.cdist(gen, real)
    precision = (d_gr <= r_real.unsqueeze(0)).any(dim=1).float().mean().item()

    d_rg = torch.cdist(real, gen)
    recall = (d_rg <= r_gen.unsqueeze(0)).any(dim=1).float().mean().item()

    return float(precision), float(recall)



def evaluate_one_setting(
    method: str,
    steps: int,
    diffusion: GaussianDiffusion,
    model: UNet,
    extractor: InceptionExtractor,
    real_images: torch.Tensor,
    real_feats: torch.Tensor,
    image_shape: Tuple[int, int, int],
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    gen_images, sampling_time = generate_images(
        diffusion=diffusion,
        model=model,
        num_samples=num_samples,
        batch_size=batch_size,
        image_shape=image_shape,
        device=device,
        method=method,
        steps=steps,
    )

    gen_feats, gen_probs = extract_features_and_probs(
        extractor,
        gen_images,
        batch_size=batch_size,
        device=device,
    )

    fid = compute_fid(real_feats, gen_feats)
    is_score = compute_inception_score(gen_probs)
    precision, recall = compute_precision_recall(real_feats, gen_feats, k=5)

    return {
        "model": method,
        "steps": steps,
        "FID": f"{fid:.6f}",
        "IS": f"{is_score:.6f}",
        "Precision": f"{precision:.6f}",
        "Recall": f"{recall:.6f}",
        "sampling_time": f"{sampling_time:.6f}",
    }



def main() -> None:
    args = parse_args()

    cfg = Config(output_dir=args.output_dir)
    ensure_dir(Path(cfg.output_dir))
    ensure_dir(cfg.metrics_path.parent)
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

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model = load_ema_model(ckpt, device)
    if use_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    image_shape = tuple(ckpt["extra"]["image_shape"])
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

    _, _, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    real_images = gather_real_images(test_loader, args.num_samples)

    extractor = InceptionExtractor()
    real_feats, _ = extract_features_and_probs(
        extractor,
        real_images,
        batch_size=args.batch_size,
        device=device,
    )

    rows = []

    rows.append(
        evaluate_one_setting(
            method="DDPM",
            steps=timesteps,
            diffusion=diffusion,
            model=model,
            extractor=extractor,
            real_images=real_images,
            real_feats=real_feats,
            image_shape=image_shape,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=device,
        )
    )

    for s in args.ddim_steps:
        rows.append(
            evaluate_one_setting(
                method="DDIM",
                steps=s,
                diffusion=diffusion,
                model=model,
                extractor=extractor,
                real_images=real_images,
                real_feats=real_feats,
                image_shape=image_shape,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                device=device,
            )
        )

    for row in rows:
        append_metrics_row(cfg.metrics_path, row)
        print(row)

    print(f"Metrics saved to: {cfg.metrics_path}")


if __name__ == "__main__":
    main()
