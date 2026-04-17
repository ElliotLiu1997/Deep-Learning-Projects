import time
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from torchvision.models import Inception_V3_Weights, inception_v3


class InceptionExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.model.eval()
        self._features = None

        def hook_fn(_, __, output):
            self._features = torch.flatten(output, 1)

        self.model.avgpool.register_forward_hook(hook_fn)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        logits = self.model(x)
        if not torch.is_tensor(logits):
            logits = logits.logits if hasattr(logits, "logits") else logits[0]
        feats = self._features
        return feats, logits


@torch.no_grad()
def generate_images(
    generator: torch.nn.Module,
    num_samples: int,
    batch_size: int,
    z_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    generator.eval()
    all_samples = []
    start = time.perf_counter()
    generated = 0

    while generated < num_samples:
        cur_bs = min(batch_size, num_samples - generated)
        z = torch.randn(cur_bs, z_dim, device=device)
        x = generator(z)
        all_samples.append(x.cpu())
        generated += cur_bs

    elapsed = time.perf_counter() - start
    return torch.cat(all_samples, dim=0), elapsed


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
            raise RuntimeError("Inception feature extraction failed (features None).")
        probs = torch.softmax(logits, dim=1)
        feats_list.append(feats.cpu())
        prob_list.append(probs.cpu())

    return torch.cat(feats_list, dim=0), torch.cat(prob_list, dim=0)


def compute_fid_internal(real_feats: torch.Tensor, gen_feats: torch.Tensor) -> float:
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


def _save_tensor_folder(images: torch.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # [-1,1] -> [0,1] for clean-fid image loading.
    imgs = ((images + 1.0) * 0.5).clamp(0.0, 1.0)
    for i in range(imgs.shape[0]):
        save_image(imgs[i], str(out_dir / f"{i:06d}.png"))


def compute_fid(real_images: torch.Tensor, gen_images: torch.Tensor) -> float:
    """
    Standard FID via clean-fid. Falls back to internal FID if clean-fid is unavailable.
    """
    try:
        from cleanfid import fid as clean_fid
    except ImportError:
        # Fallback keeps script runnable in environments without clean-fid.
        # For paper-level comparability, install clean-fid and avoid fallback.
        extractor = InceptionExtractor()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        real_feats, _ = extract_features_and_probs(extractor, real_images, batch_size=128, device=device)
        gen_feats, _ = extract_features_and_probs(extractor, gen_images, batch_size=128, device=device)
        return compute_fid_internal(real_feats, gen_feats)

    with tempfile.TemporaryDirectory(prefix="fid_real_") as real_dir, tempfile.TemporaryDirectory(
        prefix="fid_fake_"
    ) as fake_dir:
        real_path = Path(real_dir)
        fake_path = Path(fake_dir)
        _save_tensor_folder(real_images, real_path)
        _save_tensor_folder(gen_images, fake_path)
        return float(
            clean_fid.compute_fid(
                str(real_path),
                str(fake_path),
                mode="clean",
            )
        )


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
