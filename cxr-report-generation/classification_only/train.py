import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from .dataset import ChestXrayDataset, build_eval_transforms, build_train_transforms
    from .eval import evaluate_logits
    from .model import MultiLabelClassifier
    from .utils import get_model_state_dict, set_seed, setup_device_and_parallel
except ImportError:
    from dataset import ChestXrayDataset, build_eval_transforms, build_train_transforms
    from eval import evaluate_logits
    from model import MultiLabelClassifier
    from utils import get_model_state_dict, set_seed, setup_device_and_parallel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-label chest X-ray classifier")
    parser.add_argument("--encoder", type=str, default="resnet", choices=["resnet", "vit"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_csv", type=str, default="info.csv")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="classification_only/best_model.pt")
    parser.add_argument("--history_csv", type=str, default="classification_only/train_history.csv")
    parser.add_argument("--loss_plot", type=str, default="classification_only/train_loss.png")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help='GPU ids, e.g. "0,1" or "all"')

    parser.add_argument("--use_pos_weight", dest="use_pos_weight", action="store_true")
    parser.add_argument("--no_pos_weight", dest="use_pos_weight", action="store_false")
    parser.set_defaults(use_pos_weight=True)

    parser.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau", "cosine"])
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=-1,
        help="Stop training if val_macro_f1_valid does not improve for N epochs. Disabled when < 0.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="Minimum improvement in val_macro_f1_valid to reset early stopping counter.",
    )

    parser.add_argument(
        "--unfreeze_epoch",
        type=int,
        default=-1,
        help="If >0 and freeze_backbone=True, unfreeze backbone at this epoch.",
    )
    parser.add_argument("--compute_auc_val", action="store_true", help="Compute validation AUC each epoch.")

    parser.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    return parser.parse_args()


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _split_head_and_backbone_params(model: nn.Module):
    base_model = _unwrap_model(model).model

    head_module = None
    if hasattr(base_model, "fc") and isinstance(base_model.fc, nn.Module):
        head_module = base_model.fc
    elif hasattr(base_model, "head") and isinstance(base_model.head, nn.Module):
        head_module = base_model.head
    elif hasattr(base_model, "heads") and hasattr(base_model.heads, "head") and isinstance(base_model.heads.head, nn.Module):
        head_module = base_model.heads.head

    if head_module is None:
        return list(base_model.parameters()), []

    head_param_ids = {id(p) for p in head_module.parameters()}
    head_params = []
    backbone_params = []
    for p in base_model.parameters():
        if id(p) in head_param_ids:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return head_params, backbone_params


def _make_optimizer(model: nn.Module, args: argparse.Namespace, include_backbone: bool) -> Adam:
    head_params, backbone_params = _split_head_and_backbone_params(model)
    head_params = [p for p in head_params if p.requires_grad]
    backbone_params = [p for p in backbone_params if p.requires_grad]

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr})

    if include_backbone and backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})

    if not param_groups:
        raise ValueError("No trainable parameters found. Check freeze strategy.")

    return Adam(param_groups, weight_decay=args.weight_decay)


def _make_scheduler(optimizer: Adam, args: argparse.Namespace):
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_decay_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)


def _compute_pos_weight(train_dataset: ChestXrayDataset) -> torch.Tensor:
    labels = torch.stack([label for _, label in train_dataset.samples], dim=0)
    pos_count = labels.sum(dim=0)
    n_samples = labels.shape[0]

    pos_weight = torch.where(
        pos_count > 0,
        (n_samples - pos_count) / pos_count,
        torch.ones_like(pos_count),
    )
    return pos_weight.float()


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device, threshold=0.5, compute_auc=False):
    model.eval()
    val_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            val_loss += loss.item() * images.size(0)
            all_logits.append(logits)
            all_targets.append(targets)

    avg_val_loss = val_loss / len(dataloader.dataset)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = evaluate_logits(logits, targets, threshold=threshold, compute_auc=compute_auc)

    return avg_val_loss, metrics


def _save_training_history(history_rows, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_macro_f1_valid",
        "val_macro_f1_all",
        "val_micro_f1",
        "val_macro_auc",
        "head_lr",
        "backbone_lr",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)


def _save_loss_plot(history_rows, plot_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history_rows]
    train_losses = [row["train_loss"] for row in history_rows]
    val_losses = [row["val_loss"] for row in history_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_transform = build_train_transforms(args.encoder)
    eval_transform = build_eval_transforms()
    train_dataset = ChestXrayDataset(args.data_csv, args.image_dir, split="train", transform=train_transform)
    val_dataset = ChestXrayDataset(args.data_csv, args.image_dir, split="val", transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    num_classes = train_dataset.num_classes
    model = MultiLabelClassifier(
        num_classes=num_classes,
        encoder=args.encoder,
        freeze_backbone=args.freeze_backbone,
    )
    device, model, gpu_ids = setup_device_and_parallel(model, args.gpu_ids)
    if torch.cuda.is_available():
        print(f"Using CUDA device: {device} | DataParallel GPUs: {gpu_ids}")
    else:
        print("CUDA not available, using CPU.")

    include_backbone = not args.freeze_backbone
    optimizer = _make_optimizer(model, args, include_backbone=include_backbone)
    scheduler = _make_scheduler(optimizer, args)

    if args.use_pos_weight:
        pos_weight = _compute_pos_weight(train_dataset).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(
            "Using pos_weight: "
            f"min={pos_weight.min().item():.4f}, "
            f"mean={pos_weight.mean().item():.4f}, "
            f"max={pos_weight.max().item():.4f}"
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss without pos_weight")

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Num classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    backbone_unfrozen = not args.freeze_backbone
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        if (
            args.freeze_backbone
            and not backbone_unfrozen
            and args.unfreeze_epoch > 0
            and epoch >= args.unfreeze_epoch
        ):
            _, backbone_params = _split_head_and_backbone_params(model)
            for p in backbone_params:
                p.requires_grad = True
            optimizer = _make_optimizer(model, args, include_backbone=True)
            scheduler = _make_scheduler(optimizer, args)
            backbone_unfrozen = True
            print(f"Backbone unfrozen at epoch {epoch}. Using backbone_lr={args.backbone_lr}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            threshold=args.threshold,
            compute_auc=args.compute_auc_val,
        )

        macro_f1_valid = val_metrics["macro_f1_valid"]
        macro_f1_all = val_metrics["macro_f1_all"]
        micro_f1 = val_metrics["micro_f1"]

        log_msg = (
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1_valid={macro_f1_valid:.4f} | "
            f"val_macro_f1_all={macro_f1_all:.4f} | "
            f"val_micro_f1={micro_f1:.4f}"
        )
        if args.compute_auc_val:
            log_msg += f" | val_macro_auc={val_metrics.get('macro_auc')}"
        print(log_msg)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_macro_f1_valid": float(macro_f1_valid),
            "val_macro_f1_all": float(macro_f1_all),
            "val_micro_f1": float(micro_f1),
            "val_macro_auc": val_metrics.get("macro_auc") if args.compute_auc_val else "",
            "head_lr": float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else "",
            "backbone_lr": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else "",
        }
        history_rows.append(row)

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(macro_f1_valid)
            else:
                scheduler.step()

        improved = macro_f1_valid > (best_macro_f1 + args.early_stop_min_delta)
        if improved:
            best_macro_f1 = macro_f1_valid
            epochs_without_improvement = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": get_model_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "macro_f1": macro_f1_valid,
                "args": vars(args),
                "num_classes": num_classes,
            }
            if scheduler is not None:
                ckpt["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(ckpt, save_path)
            print(f"Saved new best model to {save_path} (macro_f1_valid={best_macro_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience >= 0:
                print(
                    "No macro F1 improvement this epoch "
                    f"(patience_counter={epochs_without_improvement}/{args.early_stop_patience})."
                )
                if epochs_without_improvement >= args.early_stop_patience:
                    print(
                        "Early stopping triggered at epoch "
                        f"{epoch}. Best val macro F1 (valid classes): {best_macro_f1:.4f}"
                    )
                    break

    history_csv_path = Path(args.history_csv)
    _save_training_history(history_rows, history_csv_path)
    print(f"Saved training history CSV to {history_csv_path}")

    loss_plot_path = Path(args.loss_plot)
    if _save_loss_plot(history_rows, loss_plot_path):
        print(f"Saved loss plot to {loss_plot_path}")
    else:
        print("Skipped loss plot: matplotlib is not installed.")

    print(f"Training complete. Best val macro F1 (valid classes): {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
