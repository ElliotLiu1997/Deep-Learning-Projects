import argparse
import os

import torch

from models import MODEL_REGISTRY
from train import checkpoint_path_for, evaluate_test, fit, load_best_checkpoint
from utils import (
    ensure_dir,
    get_dataloaders,
    get_device,
    maybe_wrap_data_parallel,
    save_confusion_matrix,
    save_loss_curve,
    save_test_predictions_csv,
    save_training_history_csv,
    save_val_accuracy_curve,
    set_seed,
)


def model_display_name(model_name: str) -> str:
    base_map = {
        "baseline": "BaselineCNN",
        "deeper": "DeeperCNN",
        "residual": "ResidualCNN",
        "baseline_bn": "BaselineCNN+BN",
        "deeper_bn": "DeeperCNN+BN",
        "residual_bn": "ResidualCNN+BN",
    }
    return base_map.get(model_name, model_name)


def run_single_model(
    model_name: str,
    data_dir: str,
    output_root: str,
    batch_size: int,
    epochs: int,
    lr: float,
    num_workers: int,
    seed: int,
    scheduler: str,
    step_size: int,
    gamma: float,
    t_max: int,
    eta_min: float,
) -> None:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Choices: {list(MODEL_REGISTRY.keys())}")

    device = get_device()
    print(f"Running model: {model_name} on device: {device}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    model = MODEL_REGISTRY[model_name]()
    model = model.to(device)
    model = maybe_wrap_data_parallel(model, device)

    model_output_dir = os.path.join(output_root, model_name)
    ensure_dir(model_output_dir)

    checkpoint_path = checkpoint_path_for(model_output_dir, model_name)
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        save_path=checkpoint_path,
        scheduler_type=scheduler,
        step_size=step_size,
        gamma=gamma,
        t_max=t_max,
        eta_min=eta_min,
    )

    load_best_checkpoint(model, checkpoint_path, device)
    _, test_acc, y_true, y_pred = evaluate_test(model, test_loader, device)

    save_loss_curve(
        history["train_loss"],
        history["val_loss"],
        os.path.join(model_output_dir, "train_loss_curve.png"),
        model_title=model_display_name(model_name),
    )
    save_val_accuracy_curve(
        history["train_acc"],
        history["val_acc"],
        os.path.join(model_output_dir, "val_accuracy_curve.png"),
        model_title=model_display_name(model_name),
    )
    save_training_history_csv(
        history["train_loss"],
        history["val_loss"],
        history["train_acc"],
        history["val_acc"],
        os.path.join(model_output_dir, "training_history.csv"),
    )
    save_test_predictions_csv(
        y_true,
        y_pred,
        classes,
        os.path.join(model_output_dir, "test_predictions.csv"),
    )
    save_confusion_matrix(y_true, y_pred, classes, os.path.join(model_output_dir, "confusion_matrix.png"))

    print(f"{model_name} final test accuracy: {test_acc:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 image classification with 6 CNN models")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing CIFAR-10 data files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save checkpoints and plots")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler", type=str, default="step", choices=["none", "step", "cosine"])
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--t_max", type=int, default=25)
    parser.add_argument("--eta_min", type=float, default=1e-5)
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["baseline", "deeper", "residual", "baseline_bn", "deeper_bn", "residual_bn", "all"],
        help="Choose one model or run all models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    model_names = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for model_name in model_names:
        run_single_model(
            model_name=model_name,
            data_dir=args.data_dir,
            output_root=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            seed=args.seed,
            scheduler=args.scheduler,
            step_size=args.step_size,
            gamma=args.gamma,
            t_max=args.t_max,
            eta_min=args.eta_min,
        )


if __name__ == "__main__":
    main()
