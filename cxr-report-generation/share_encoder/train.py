import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    from .dataset import MultiTaskDataset, build_transforms, load_caption_metadata
    from .model import SharedEncoderMultiTaskModel
except ImportError:
    from dataset import MultiTaskDataset, build_transforms, load_caption_metadata
    from model import SharedEncoderMultiTaskModel

from caption_only.utils import get_model_state_dict, save_json, set_seed, setup_device_and_parallel


def parse_args():
    p = argparse.ArgumentParser(description="Train shared-encoder multi-task model (caption + classification)")
    p.add_argument("--data_csv", type=str, default="info.csv")
    p.add_argument("--image_dir", type=str, default="images/images_normalized")
    p.add_argument("--encoder_checkpoint", type=str, default="classification_only/outputs/resnet/best_model.pt")
    p.add_argument("--output_dir", type=str, default="share_encoder/outputs/lstm_attn")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--encoder_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--teacher_forcing_ratio", type=float, default=0.8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cls_loss_weight", type=float, default=0.2)
    p.add_argument("--early_stopping_patience", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu_ids", type=str, default="0,1")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _build_optimizer(model: nn.Module, lr: float, encoder_lr: float, weight_decay: float):
    base = _unwrap(model)
    dec_params = [p for p in base.decoder.parameters() if p.requires_grad]
    cls_params = [p for p in base.cls_head.parameters() if p.requires_grad]
    enc_params = [p for p in base.encoder.parameters() if p.requires_grad]

    param_groups = []
    if dec_params:
        param_groups.append({"params": dec_params, "lr": lr})
    if cls_params:
        param_groups.append({"params": cls_params, "lr": lr})
    if enc_params and encoder_lr > 0:
        param_groups.append({"params": enc_params, "lr": encoder_lr})
    return Adam(param_groups, weight_decay=weight_decay)


def _macro_f1_from_logits(cls_logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(cls_logits) >= threshold).float()
    eps = 1e-8
    f1s = []
    for c in range(labels.size(1)):
        p = preds[:, c]
        y = labels[:, c]
        tp = (p * y).sum()
        fp = (p * (1 - y)).sum()
        fn = ((1 - p) * y).sum()
        denom = (2 * tp + fp + fn + eps)
        f1s.append(((2 * tp) / denom).item())
    return float(sum(f1s) / max(len(f1s), 1))


def train_one_epoch(model, loader, optimizer, cap_criterion, cls_criterion, device, teacher_forcing_ratio, grad_clip, cls_w):
    model.train()
    total = 0.0
    total_cap = 0.0
    total_cls = 0.0
    for images, captions, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        input_seq = captions[:, :-1]
        target_seq = captions[:, 1:]

        optimizer.zero_grad()
        cap_logits, cls_logits = model(images, input_seq, teacher_forcing_ratio=teacher_forcing_ratio)
        cap_loss = cap_criterion(cap_logits.reshape(-1, cap_logits.size(-1)), target_seq.reshape(-1))
        cls_loss = cls_criterion(cls_logits, labels)
        loss = cap_loss + cls_w * cls_loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        bs = images.size(0)
        total += loss.item() * bs
        total_cap += cap_loss.item() * bs
        total_cls += cls_loss.item() * bs

    n = len(loader.dataset)
    return total / n, total_cap / n, total_cls / n


@torch.no_grad()
def validate(model, loader, cap_criterion, cls_criterion, device, cls_w):
    model.eval()
    total = 0.0
    total_cap = 0.0
    total_cls = 0.0
    all_cls_logits = []
    all_labels = []
    for images, captions, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        input_seq = captions[:, :-1]
        target_seq = captions[:, 1:]
        cap_logits, cls_logits = model(images, input_seq, teacher_forcing_ratio=1.0)

        cap_loss = cap_criterion(cap_logits.reshape(-1, cap_logits.size(-1)), target_seq.reshape(-1))
        cls_loss = cls_criterion(cls_logits, labels)
        loss = cap_loss + cls_w * cls_loss

        bs = images.size(0)
        total += loss.item() * bs
        total_cap += cap_loss.item() * bs
        total_cls += cls_loss.item() * bs
        all_cls_logits.append(cls_logits.detach())
        all_labels.append(labels.detach())

    n = len(loader.dataset)
    cls_logits = torch.cat(all_cls_logits, dim=0)
    cls_labels = torch.cat(all_labels, dim=0)
    macro_f1 = _macro_f1_from_logits(cls_logits, cls_labels, threshold=0.5)
    return total / n, total_cap / n, total_cls / n, macro_f1


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_caption_metadata(args.data_csv)
    train_ds = MultiTaskDataset(args.data_csv, args.image_dir, split="train", transform=build_transforms())
    val_ds = MultiTaskDataset(args.data_csv, args.image_dir, split="val", transform=build_transforms())
    num_classes = train_ds.num_classes or 13

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SharedEncoderMultiTaskModel(
        vocab_size=metadata["vocab_size"],
        num_classes=num_classes,
        encoder_checkpoint=args.encoder_checkpoint,
        pad_idx=metadata["pad_idx"],
    )
    if args.encoder_lr <= 0:
        model.freeze_encoder()
    else:
        model.unfreeze_encoder()

    device, model, gpu_ids = setup_device_and_parallel(model, args.gpu_ids)
    print(f"Using device={device}, gpu_ids={gpu_ids if gpu_ids else 'cpu'}")

    cap_criterion = nn.CrossEntropyLoss(ignore_index=metadata["pad_idx"])
    cls_criterion = nn.BCEWithLogitsLoss()
    optimizer = _build_optimizer(model, args.lr, args.encoder_lr, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-7)

    best_val = float("inf")
    no_improve = 0
    history = []
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        tr, tr_cap, tr_cls = train_one_epoch(
            model,
            train_loader,
            optimizer,
            cap_criterion,
            cls_criterion,
            device,
            args.teacher_forcing_ratio,
            args.grad_clip,
            args.cls_loss_weight,
        )
        va, va_cap, va_cls, va_f1 = validate(
            model,
            val_loader,
            cap_criterion,
            cls_criterion,
            device,
            args.cls_loss_weight,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train={tr:.4f} (cap={tr_cap:.4f}, cls={tr_cls:.4f}) | "
            f"val={va:.4f} (cap={va_cap:.4f}, cls={va_cls:.4f}) | "
            f"val_cls_macro_f1={va_f1:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr,
                "train_caption_loss": tr_cap,
                "train_cls_loss": tr_cls,
                "val_loss": va,
                "val_caption_loss": va_cap,
                "val_cls_loss": va_cls,
                "val_cls_macro_f1": va_f1,
            }
        )

        if va < best_val:
            best_val = va
            no_improve = 0
            torch.save(get_model_state_dict(model), best_path)
        else:
            no_improve += 1
            if no_improve >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (patience={args.early_stopping_patience})")
                break

    hist_csv = out_dir / "train_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else ["epoch"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    save_json(
        {
            "data_csv": args.data_csv,
            "image_dir": args.image_dir,
            "encoder_checkpoint": args.encoder_checkpoint,
            "vocab_size": metadata["vocab_size"],
            "num_classes": num_classes,
            "pad_idx": metadata["pad_idx"],
            "sos_idx": metadata["sos_idx"],
            "eos_idx": metadata["eos_idx"],
            "max_seq_len": metadata["max_seq_len"],
            "cls_loss_weight": args.cls_loss_weight,
            "best_val_loss": best_val,
        },
        str(out_dir / "metadata.json"),
    )
    print(f"Saved best model to: {best_path}")
    print(f"Saved train history to: {hist_csv}")


if __name__ == "__main__":
    main()

