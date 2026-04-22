import argparse
import csv
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .dataset import ChestXrayDataset, build_eval_transforms
    from .model import MultiLabelClassifier
    from .utils import load_model_state_dict_flexible, setup_device_and_parallel
except ImportError:
    from dataset import ChestXrayDataset, build_eval_transforms
    from model import MultiLabelClassifier
    from utils import load_model_state_dict_flexible, setup_device_and_parallel


def evaluate_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    compute_auc: bool = False,
) -> Dict[str, object]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int()
    targets_int = targets.int()

    tp = (preds & targets_int).sum(dim=0).float()
    fp = (preds & (1 - targets_int)).sum(dim=0).float()
    fn = ((1 - preds) & targets_int).sum(dim=0).float()

    per_class_denom = (2 * tp + fp + fn)
    per_class_f1 = torch.where(
        per_class_denom > 0,
        (2 * tp) / per_class_denom,
        torch.zeros_like(per_class_denom),
    )

    macro_f1_all = per_class_f1.mean().item()
    valid_class_mask = per_class_denom > 0
    if valid_class_mask.any():
        macro_f1_valid = per_class_f1[valid_class_mask].mean().item()
    else:
        macro_f1_valid = 0.0

    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_denom = 2 * micro_tp + micro_fp + micro_fn
    micro_f1 = ((2 * micro_tp) / micro_denom).item() if micro_denom > 0 else 0.0

    metrics = {
        "per_class_f1": per_class_f1.tolist(),
        "macro_f1": macro_f1_valid,
        "macro_f1_valid": macro_f1_valid,
        "macro_f1_all": macro_f1_all,
        "valid_class_count": int(valid_class_mask.sum().item()),
        "micro_f1": micro_f1,
    }

    if compute_auc:
        try:
            from sklearn.metrics import roc_auc_score

            y_true = targets.cpu().numpy()
            y_prob = probs.cpu().numpy()
            auc_per_class = []
            for i in range(y_true.shape[1]):
                if len(set(y_true[:, i].tolist())) < 2:
                    auc_per_class.append(float("nan"))
                    continue
                auc_per_class.append(float(roc_auc_score(y_true[:, i], y_prob[:, i])))

            valid_auc = [x for x in auc_per_class if x == x]
            metrics["per_class_auc"] = auc_per_class
            metrics["macro_auc"] = sum(valid_auc) / len(valid_auc) if valid_auc else None
        except ImportError:
            metrics["per_class_auc"] = None
            metrics["macro_auc"] = None

    return metrics


def run_inference(model, dataloader, device) -> Dict[str, torch.Tensor]:
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            all_logits.append(logits)
            all_targets.append(targets)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return {"logits": logits, "targets": targets}


def evaluate_model(model, dataloader, device, threshold: float = 0.5, compute_auc: bool = False):
    outputs = run_inference(model, dataloader, device)
    return evaluate_logits(outputs["logits"], outputs["targets"], threshold=threshold, compute_auc=compute_auc)


def search_best_global_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
):
    best_threshold = threshold_min
    best_macro_f1 = -1.0
    search_rows = []

    t = threshold_min
    while t <= threshold_max + 1e-12:
        metrics = evaluate_logits(logits, targets, threshold=float(t), compute_auc=False)
        macro_f1 = float(metrics["macro_f1_valid"])
        search_rows.append({"threshold": float(t), "macro_f1_valid": macro_f1})
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(t)
        t += threshold_step

    return best_threshold, best_macro_f1, search_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-label chest X-ray classifier")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="resnet", choices=["resnet", "vit"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_csv", type=str, default="info.csv")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--search_threshold_on_val", action="store_true")
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--threshold_search_csv", type=str, default="classification_only/threshold_search.csv")
    parser.add_argument("--compute_auc", action="store_true")
    parser.add_argument("--metrics_csv", type=str, default="classification_only/eval_metrics.csv")
    parser.add_argument("--per_class_csv", type=str, default="classification_only/eval_per_class_metrics.csv")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help='GPU ids, e.g. "0,1" or "all"')
    parser.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    return parser.parse_args()


def _save_eval_metrics_csv(
    metrics: Dict[str, object],
    avg_loss: float,
    encoder: str,
    threshold: float,
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "encoder": encoder,
        "threshold": float(threshold),
        "loss": float(avg_loss),
        "macro_f1_valid": float(metrics["macro_f1_valid"]),
        "macro_f1_all": float(metrics["macro_f1_all"]),
        "valid_class_count": int(metrics["valid_class_count"]),
        "micro_f1": float(metrics["micro_f1"]),
        "macro_auc": metrics.get("macro_auc", ""),
    }
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _save_per_class_csv(metrics: Dict[str, object], encoder: str, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    per_class_f1 = metrics["per_class_f1"]
    per_class_auc = metrics.get("per_class_auc")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder", "class_idx", "f1", "auc"])
        writer.writeheader()
        for idx, f1 in enumerate(per_class_f1):
            auc_val = ""
            if per_class_auc is not None and idx < len(per_class_auc):
                auc_val = per_class_auc[idx]
            writer.writerow({"encoder": encoder, "class_idx": idx, "f1": f1, "auc": auc_val})


def _save_threshold_search_csv(search_rows, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "macro_f1_valid"])
        writer.writeheader()
        for row in search_rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    transform = build_eval_transforms()
    test_dataset = ChestXrayDataset(args.data_csv, args.image_dir, split="test", transform=transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MultiLabelClassifier(
        num_classes=test_dataset.num_classes,
        encoder=args.encoder,
        freeze_backbone=args.freeze_backbone,
    )
    device, model, gpu_ids = setup_device_and_parallel(model, args.gpu_ids)
    if torch.cuda.is_available():
        print(f"Using CUDA device: {device} | DataParallel GPUs: {gpu_ids}")
    else:
        print("CUDA not available, using CPU.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    load_model_state_dict_flexible(model, ckpt["model_state_dict"])

    threshold_to_use = args.threshold
    if args.search_threshold_on_val:
        if args.threshold_step <= 0:
            raise ValueError("--threshold_step must be > 0")
        if args.threshold_min <= 0 or args.threshold_max >= 1 or args.threshold_min >= args.threshold_max:
            raise ValueError("Threshold range must satisfy 0 < threshold_min < threshold_max < 1")

        val_dataset = ChestXrayDataset(args.data_csv, args.image_dir, split="val", transform=transform)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_outputs = run_inference(model, val_dataloader, device)
        best_threshold, best_val_macro_f1, search_rows = search_best_global_threshold(
            val_outputs["logits"],
            val_outputs["targets"],
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
        )
        threshold_to_use = best_threshold
        print(
            "Threshold search on val complete: "
            f"best_threshold={best_threshold:.4f}, "
            f"best_val_macro_f1_valid={best_val_macro_f1:.4f}"
        )
        search_csv = Path(args.threshold_search_csv)
        _save_threshold_search_csv(search_rows, search_csv)
        print(f"Saved threshold search CSV to {search_csv}")
    else:
        print(f"Using fixed threshold: {threshold_to_use:.4f}")

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, targets in test_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            all_logits.append(logits)
            all_targets.append(targets)

    avg_loss = total_loss / len(test_dataset)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = evaluate_logits(logits, targets, threshold=threshold_to_use, compute_auc=args.compute_auc)

    print("Split: test")
    print(f"Threshold: {threshold_to_use:.4f}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Macro F1 (valid classes): {metrics['macro_f1_valid']:.4f}")
    print(f"Macro F1 (all classes): {metrics['macro_f1_all']:.4f}")
    print(f"Valid class count: {metrics['valid_class_count']}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Per-class F1: {metrics['per_class_f1']}")

    if args.compute_auc:
        print(f"Macro AUC: {metrics.get('macro_auc')}")
        print(f"Per-class AUC: {metrics.get('per_class_auc')}")

    metrics_csv = Path(args.metrics_csv)
    _save_eval_metrics_csv(metrics, avg_loss, args.encoder, threshold_to_use, metrics_csv)
    print(f"Saved eval metrics CSV to {metrics_csv}")

    per_class_csv = Path(args.per_class_csv)
    _save_per_class_csv(metrics, args.encoder, per_class_csv)
    print(f"Saved per-class metrics CSV to {per_class_csv}")


if __name__ == "__main__":
    main()
