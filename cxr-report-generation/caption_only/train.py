import argparse
import csv
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from .dataset import CaptionDataset, build_transforms, load_caption_metadata
    from .models import CaptioningModel
    from .utils import (
        get_model_state_dict,
        plot_loss_curve,
        save_json,
        set_seed,
        setup_device_and_parallel,
    )
except ImportError:
    from dataset import CaptionDataset, build_transforms, load_caption_metadata
    from models import CaptioningModel
    from utils import (
        get_model_state_dict,
        plot_loss_curve,
        save_json,
        set_seed,
        setup_device_and_parallel,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image captioning model")
    parser.add_argument("--decoder_type", type=str, default="lstm", choices=["lstm", "lstm_attn", "transformer"])

    parser.add_argument("--data_csv", type=str, default="info.csv")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        default="classification_only/outputs/resnet/best_model.pt",
        help="Path to pretrained classification checkpoint used to initialize ResNet encoder",
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4, help="Decoder learning rate")
    parser.add_argument("--encoder_lr", type=float, default=0.0, help="Encoder learning rate. 0 means frozen encoder.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.8)
    parser.add_argument(
        "--caption_balance_alpha",
        type=float,
        default=0.0,
        help="Inverse-frequency strength for caption-level sampling. 0 disables balancing.",
    )
    parser.add_argument(
        "--token_balance_alpha",
        type=float,
        default=0.0,
        help="Inverse-frequency strength for token-level CE weighting. 0 disables weighting.",
    )
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "plateau", "cosine"])
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--warmup_epochs", type=int, default=1)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _make_optimizer(model: nn.Module, lr: float, encoder_lr: float, weight_decay: float) -> Adam:
    base_model = _unwrap_model(model)
    decoder_params = [p for p in base_model.decoder.parameters() if p.requires_grad]
    encoder_params = [p for p in base_model.encoder.parameters() if p.requires_grad]

    param_groups = []
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": lr})
    if encoder_lr > 0 and encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})

    if not param_groups:
        raise ValueError("No trainable parameters found.")

    return Adam(param_groups, weight_decay=weight_decay)


def _make_scheduler(optimizer: Adam, args: argparse.Namespace):
    if args.scheduler == "none":
        return None

    if args.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    cosine_tmax = max(1, args.epochs - max(args.warmup_epochs, 0))
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_tmax, eta_min=args.min_lr)
    if args.warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    return cosine


def train_one_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio: float, grad_clip: float):
    model.train()
    total_loss = 0.0

    for images, captions, _ in loader:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        input_seq = captions[:, :-1]
        target_seq = captions[:, 1:]

        optimizer.zero_grad()
        logits = model(images, input_seq, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for images, captions, _ in loader:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        input_seq = captions[:, :-1]
        target_seq = captions[:, 1:]

        logits = model(images, input_seq, teacher_forcing_ratio=1.0)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def _save_history_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "val_loss"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_caption_sampler(train_dataset: CaptionDataset, alpha: float):
    if alpha <= 0:
        return None

    caption_keys = [tuple(caption_seq) for _, caption_seq, _ in train_dataset.samples]
    counts = Counter(caption_keys)
    sample_weights = [1.0 / (counts[key] ** alpha) for key in caption_keys]
    weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def _make_token_weights(train_dataset: CaptionDataset, vocab_size: int, pad_idx: int, alpha: float):
    if alpha <= 0:
        return None

    counts = torch.zeros(vocab_size, dtype=torch.float32)
    for _, caption_seq, _ in train_dataset.samples:
        if len(caption_seq) < 2:
            continue
        target_tokens = torch.tensor(caption_seq[1:], dtype=torch.long)
        valid = (target_tokens >= 0) & (target_tokens < vocab_size)
        if valid.any():
            binc = torch.bincount(target_tokens[valid], minlength=vocab_size).float()
            counts += binc

    counts = torch.clamp(counts, min=1.0)
    weights = counts.pow(-alpha)
    if 0 <= pad_idx < vocab_size:
        weights[pad_idx] = 0.0

    mean_w = weights[weights > 0].mean().item() if (weights > 0).any() else 1.0
    if mean_w > 0:
        weights = weights / mean_w
    return weights


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else Path("caption_only") / "outputs" / args.decoder_type
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_caption_metadata(args.data_csv)

    transform = build_transforms()
    train_dataset = CaptionDataset(args.data_csv, args.image_dir, split="train", transform=transform)
    val_dataset = CaptionDataset(args.data_csv, args.image_dir, split="val", transform=transform)
    train_sampler = _make_caption_sampler(train_dataset, args.caption_balance_alpha)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = CaptioningModel(
        decoder_type=args.decoder_type,
        vocab_size=metadata["vocab_size"],
        encoder_checkpoint=args.encoder_checkpoint,
        pad_idx=metadata["pad_idx"],
    )

    if args.encoder_lr <= 0:
        model.freeze_encoder()
    else:
        model.unfreeze_encoder()

    device, model, gpu_ids = setup_device_and_parallel(model, args.gpu_ids)
    print(f"Using device={device}, gpu_ids={gpu_ids if gpu_ids else 'cpu'}")

    token_weights = _make_token_weights(
        train_dataset=train_dataset,
        vocab_size=metadata["vocab_size"],
        pad_idx=metadata["pad_idx"],
        alpha=args.token_balance_alpha,
    )
    if token_weights is not None:
        token_weights = token_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=token_weights,
        ignore_index=metadata["pad_idx"],
        label_smoothing=args.label_smoothing,
    )
    optimizer = _make_optimizer(model, lr=args.lr, encoder_lr=args.encoder_lr, weight_decay=args.weight_decay)
    scheduler = _make_scheduler(optimizer, args)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    best_model_path = output_dir / "best_model.pt"
    history_csv = output_dir / "train_history.csv"
    loss_curve_path = output_dir / "loss_curve.png"
    metadata_path = output_dir / "metadata.json"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            grad_clip=args.grad_clip,
        )
        val_loss = validate(model, val_loader, criterion, device)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(get_model_state_dict(model), best_model_path)
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={args.early_stopping_patience}).")
            break

    _save_history_csv(history, history_csv)
    plot_ok = plot_loss_curve(
        train_losses=[r["train_loss"] for r in history],
        val_losses=[r["val_loss"] for r in history],
        output_path=str(loss_curve_path),
    )

    metadata_to_save = {
        "decoder_type": args.decoder_type,
        "data_csv": args.data_csv,
        "image_dir": args.image_dir,
        "encoder_checkpoint": args.encoder_checkpoint,
        "vocab_size": metadata["vocab_size"],
        "pad_idx": metadata["pad_idx"],
        "sos_idx": metadata["sos_idx"],
        "eos_idx": metadata["eos_idx"],
        "max_seq_len": metadata["max_seq_len"],
        "best_val_loss": best_val_loss,
        "loss_curve_saved": plot_ok,
        "grad_clip": args.grad_clip,
        "label_smoothing": args.label_smoothing,
        "teacher_forcing_ratio": args.teacher_forcing_ratio,
        "caption_balance_alpha": args.caption_balance_alpha,
        "token_balance_alpha": args.token_balance_alpha,
        "scheduler": args.scheduler,
        "lr_patience": args.lr_patience,
        "lr_decay_factor": args.lr_decay_factor,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
    }
    save_json(metadata_to_save, str(metadata_path))

    print(f"Saved best model to: {best_model_path}")
    print(f"Saved train history to: {history_csv}")
    print(f"Saved loss curve to: {loss_curve_path}")


if __name__ == "__main__":
    main()
