import argparse
from pathlib import Path

from dataset import create_dataloaders
from utils import ensure_dir, save_tensor_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save real PathMNIST test grid (4x4 by default)")
    parser.add_argument("--data_dir", type=str, default="pathmnist")
    parser.add_argument("--output_dir", type=str, default="diffusion_project/outputs")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    samples_dir = out_dir / "samples"
    ensure_dir(samples_dir)

    _, _, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.num_samples,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    real_images, _ = next(iter(test_loader))
    n = min(args.num_samples, real_images.shape[0])
    nrow = int(n ** 0.5)
    if nrow * nrow != n:
        nrow = max(1, nrow)

    save_tensor_grid(real_images[:n], samples_dir / "real_test_4x4.png", nrow=max(nrow, 1))
    print(f"Saved real grid: {samples_dir / 'real_test_4x4.png'}")


if __name__ == "__main__":
    main()
