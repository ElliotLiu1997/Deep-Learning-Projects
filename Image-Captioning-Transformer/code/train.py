"""
Training script for Flickr8k image captioning models.

Trains and compares:
1) CNN + RNN
2) CNN + GRU
3) CNN + LSTM
4) CNN + LSTM + Dropout
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_config
from vocab import Vocabulary, build_and_save_vocab
from dataset import Flickr8kCaptionDataset, caption_collate_fn, build_image_transform
from model import ImageCaptioningModel
from feature_cache import build_all_feature_caches
from utils import set_seed, save_checkpoint


def _maybe_load_or_build_vocab(cfg) -> Vocabulary:
    if cfg.vocab_path.exists():
        return Vocabulary.from_json(cfg.vocab_path)
    return build_and_save_vocab(
        captions_file=cfg.captions_file,
        vocab_path=cfg.vocab_path,
        min_word_freq=cfg.min_word_freq,
    )


def _ensure_feature_cache(cfg) -> None:
    required = [
        cfg.features_train_path,
        cfg.features_val_path,
        cfg.features_test_path,
        cfg.feature_keys_train_path,
        cfg.feature_keys_val_path,
        cfg.feature_keys_test_path,
    ]
    if all(p.exists() for p in required):
        return
    print("Feature cache missing. Building caches now...")
    build_all_feature_caches()


def _build_dataloaders(cfg, vocab: Vocabulary, use_feature_cache: bool):
    image_transform = build_image_transform(
        image_size=cfg.image_size,
        mean=cfg.image_mean,
        std=cfg.image_std,
    )

    if use_feature_cache:
        _ensure_feature_cache(cfg)

    train_ds = Flickr8kCaptionDataset(
        images_dir=cfg.images_dir,
        captions_file=cfg.captions_file,
        split_file=cfg.train_split_file,
        vocab=vocab,
        max_caption_length=cfg.max_caption_length,
        use_feature_cache=use_feature_cache,
        feature_npy_path=cfg.features_train_path if use_feature_cache else None,
        feature_keys_json_path=cfg.feature_keys_train_path if use_feature_cache else None,
        image_transform=None if use_feature_cache else image_transform,
    )

    val_ds = Flickr8kCaptionDataset(
        images_dir=cfg.images_dir,
        captions_file=cfg.captions_file,
        split_file=cfg.val_split_file,
        vocab=vocab,
        max_caption_length=cfg.max_caption_length,
        use_feature_cache=use_feature_cache,
        feature_npy_path=cfg.features_val_path if use_feature_cache else None,
        feature_keys_json_path=cfg.feature_keys_val_path if use_feature_cache else None,
        image_transform=None if use_feature_cache else image_transform,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and cfg.device.type == "cuda",
        collate_fn=caption_collate_fn,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and cfg.device.type == "cuda",
        collate_fn=caption_collate_fn,
    )
    return train_dl, val_dl


def _forward_model(model: ImageCaptioningModel, x: torch.Tensor, input_seq: torch.Tensor, use_feature_cache: bool):
    if use_feature_cache:
        return model.forward_features(x, input_seq)
    return model(x, input_seq)


def train_one_epoch(
    model: ImageCaptioningModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_feature_cache: bool,
) -> float:
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for x, input_seq, target_seq in pbar:
        x = x.to(device, non_blocking=True)
        input_seq = input_seq.to(device, non_blocking=True)
        target_seq = target_seq.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = _forward_model(model, x, input_seq, use_feature_cache=use_feature_cache)  # [B, T, V]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_seq.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(dataloader), 1)


@torch.no_grad()
def validate_one_epoch(
    model: ImageCaptioningModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_feature_cache: bool,
) -> float:
    model.eval()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Val", leave=False)
    for x, input_seq, target_seq in pbar:
        x = x.to(device, non_blocking=True)
        input_seq = input_seq.to(device, non_blocking=True)
        target_seq = target_seq.to(device, non_blocking=True)

        logits = _forward_model(model, x, input_seq, use_feature_cache=use_feature_cache)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_seq.reshape(-1),
        )
        running_loss += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(dataloader), 1)


def train_single_variant(cfg, vocab: Vocabulary, variant: str):
    print(f"\n=== Training variant: {variant} ===")
    is_transformer = "transformer" in variant
    model_embed_dim = cfg.transformer_embed_dim if is_transformer else cfg.embed_dim
    model_hidden_dim = (4 * cfg.transformer_embed_dim) if is_transformer else cfg.hidden_dim
    model_num_layers = cfg.transformer_num_layers if is_transformer else cfg.num_layers

    model = ImageCaptioningModel(
        vocab_size=vocab.size,
        embed_dim=model_embed_dim,
        hidden_dim=model_hidden_dim,
        num_layers=model_num_layers,
        rnn_type=variant,
        model_type=variant,
        dropout=cfg.dropout,
        pad_idx=vocab.pad_idx,
        transformer_d_model=cfg.transformer_embed_dim,
        transformer_num_layers=cfg.transformer_num_layers,
        transformer_nhead=cfg.transformer_nhead,
        transformer_ff_dim=(4 * cfg.transformer_embed_dim),
    ).to(cfg.device)

    # Keep visual encoders frozen as required.
    if hasattr(model.encoder, "cnn"):
        for p in model.encoder.cnn.parameters():
            p.requires_grad = False
    if hasattr(model.encoder, "vit"):
        for p in model.encoder.vit.parameters():
            p.requires_grad = False

    use_feature_cache = cfg.use_feature_cache and model.supports_feature_cache
    print(f"[{variant}] use_feature_cache={use_feature_cache}")
    train_dl, val_dl = _build_dataloaders(cfg, vocab, use_feature_cache=use_feature_cache)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    lr = cfg.learning_rate * 0.1 if "transformer" in variant else cfg.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.weight_decay,
    )
    print(f"[{variant}] learning_rate={lr}")

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")
    best_epoch = 0
    no_improve_count = 0
    best_ckpt = cfg.checkpoints_dir / f"{variant}_best.pt"

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"[{variant}] Epoch {epoch}/{cfg.num_epochs}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dl,
            optimizer=optimizer,
            criterion=criterion,
            device=cfg.device,
            use_feature_cache=use_feature_cache,
        )
        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_dl,
            criterion=criterion,
            device=cfg.device,
            use_feature_cache=use_feature_cache,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[{variant}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve_count = 0
            save_checkpoint(
                path=best_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra={
                    "variant": variant,
                    "val_loss": val_loss,
                    "use_feature_cache": use_feature_cache,
                },
            )
        else:
            no_improve_count += 1

        if cfg.early_stopping_patience > 0 and no_improve_count >= cfg.early_stopping_patience:
            print(
                f"[{variant}] Early stopping at epoch {epoch}. "
                f"Best epoch={best_epoch}, best val_loss={best_val:.4f}"
            )
            break

    return {
        "variant": variant,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "stopped_epoch": len(train_losses),
        "best_checkpoint": str(best_ckpt),
    }


def main():
    cfg = get_config()
    set_seed(cfg.seed)

    print(f"Using device: {cfg.device}")
    print(f"Feature cache mode: {cfg.use_feature_cache}")

    vocab = _maybe_load_or_build_vocab(cfg)
    all_histories: Dict[str, Dict] = {}
    for variant in cfg.model_variants:
        history = train_single_variant(cfg, vocab, variant)
        all_histories[variant] = history

    # Required output: training_loss.npy
    np.save(cfg.training_loss_path, all_histories, allow_pickle=True)
    print(f"Saved training history to: {cfg.training_loss_path}")


if __name__ == "__main__":
    main()
