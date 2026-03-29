"""
Evaluation script for Flickr8k captioning models.

Computes:
- BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4
Saves:
- metrics.csv
- captions.json
"""

from __future__ import annotations
from typing import Dict, List

import json
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from config import get_config
from vocab import Vocabulary, parse_flickr8k_captions, load_split_image_names
from dataset import Flickr8kCaptionDataset, caption_collate_fn, build_image_transform
from model import ImageCaptioningModel
from utils import load_checkpoint, clean_caption_tokens


def _build_test_loader(cfg, vocab: Vocabulary):
    image_transform = build_image_transform(
        image_size=cfg.image_size,
        mean=cfg.image_mean,
        std=cfg.image_std,
    )

    test_ds = Flickr8kCaptionDataset(
        images_dir=cfg.images_dir,
        captions_file=cfg.captions_file,
        split_file=cfg.test_split_file,
        vocab=vocab,
        max_caption_length=cfg.max_caption_length,
        use_feature_cache=cfg.use_feature_cache,
        feature_npy_path=cfg.features_test_path if cfg.use_feature_cache else None,
        feature_keys_json_path=cfg.feature_keys_test_path if cfg.use_feature_cache else None,
        image_transform=None if cfg.use_feature_cache else image_transform,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and cfg.device.type == "cuda",
        collate_fn=caption_collate_fn,
    )
    return test_ds, test_dl


def _build_image_to_references(cfg, vocab: Vocabulary) -> Dict[str, List[List[str]]]:
    """
    Returns:
        image_name -> list of reference captions
        each caption is tokenized list without special tokens
    """
    all_caps = parse_flickr8k_captions(cfg.captions_file)
    test_images = set(load_split_image_names(cfg.test_split_file))

    refs = {}
    for img in test_images:
        ref_caps = []
        for c in all_caps.get(img, []):
            enc = vocab.encode_caption(c)  # includes <start>, <end>
            toks = vocab.denumericalize(enc, stop_at_end=False)
            toks = clean_caption_tokens(toks)
            ref_caps.append(toks)
        refs[img] = ref_caps
    return refs


@torch.no_grad()
def _generate_for_image(
    model: ImageCaptioningModel,
    feature_or_image: torch.Tensor,
    cfg,
    vocab: Vocabulary,
) -> List[str]:
    """
    Generate one caption via greedy decoding.
    """
    if cfg.use_feature_cache:
        # Input is cached 512-d feature vector.
        image_emb = model.project_cached_features(feature_or_image.unsqueeze(0))
    else:
        # Input is image tensor [3,H,W].
        image_emb = model.encoder(feature_or_image.unsqueeze(0))

    pred_ids = model.greedy_decode(
        image_embedding=image_emb,
        start_idx=vocab.start_idx,
        end_idx=vocab.end_idx,
        max_len=cfg.max_decode_length,
    ).tolist()

    pred_tokens = vocab.denumericalize(pred_ids, stop_at_end=True)
    pred_tokens = clean_caption_tokens(pred_tokens)
    return pred_tokens


def _load_model_for_variant(cfg, vocab: Vocabulary, variant: str) -> ImageCaptioningModel:
    ckpt_path = cfg.checkpoints_dir / f"{variant}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for variant '{variant}': {ckpt_path}")

    model = ImageCaptioningModel(
        vocab_size=vocab.size,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        rnn_type=variant,
        dropout=cfg.dropout,
        pad_idx=vocab.pad_idx,
    ).to(cfg.device)

    load_checkpoint(ckpt_path, model=model, optimizer=None, map_location=cfg.device)
    model.eval()
    return model


def evaluate_variant(cfg, vocab: Vocabulary, variant: str):
    print(f"\n=== Evaluating variant: {variant} ===")
    model = _load_model_for_variant(cfg, vocab, variant)

    test_ds, _test_dl = _build_test_loader(cfg, vocab)
    image_to_refs = _build_image_to_references(cfg, vocab)

    # Reconstruct image order from dataset samples (one per caption).
    # We'll evaluate once per image for BLEU corpus predictions.
    seen = set()
    unique_images = []
    for image_name, _caption in test_ds.samples:
        if image_name not in seen:
            seen.add(image_name)
            unique_images.append(image_name)

    # Build per-image input tensors by first occurrence index in dataset
    first_index_by_image = {}
    for idx, (img_name, _caption) in enumerate(test_ds.samples):
        if img_name not in first_index_by_image:
            first_index_by_image[img_name] = idx

    hypotheses = []
    references = []
    generated_json = {}

    pbar = tqdm(unique_images, desc=f"Decode {variant}", leave=False)
    for image_name in pbar:
        ds_idx = first_index_by_image[image_name]
        feature_or_image, _, _ = test_ds[ds_idx]
        feature_or_image = feature_or_image.to(cfg.device)

        pred_tokens = _generate_for_image(model, feature_or_image, cfg, vocab)
        ref_tokens = image_to_refs.get(image_name, [])

        if len(ref_tokens) == 0:
            continue

        hypotheses.append(pred_tokens)
        references.append(ref_tokens)
        generated_json[image_name] = {
            "prediction": " ".join(pred_tokens),
            "references": [" ".join(r) for r in ref_tokens],
        }

    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    metrics = {
        "model": variant,
        "BLEU-1": float(bleu1),
        "BLEU-2": float(bleu2),
        "BLEU-3": float(bleu3),
        "BLEU-4": float(bleu4),
    }
    return metrics, generated_json


def main():
    cfg = get_config()
    vocab = Vocabulary.from_json(cfg.vocab_path)

    all_metrics = []
    all_captions = {}

    for variant in cfg.model_variants:
        metrics, captions = evaluate_variant(cfg, vocab, variant)
        all_metrics.append(metrics)
        all_captions[variant] = captions
        print(metrics)

    # Save metrics.csv
    with open(cfg.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"])
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)

    # Save captions.json
    with open(cfg.captions_json_path, "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Saved metrics to: {cfg.metrics_csv_path}")
    print(f"Saved captions to: {cfg.captions_json_path}")


if __name__ == "__main__":
    main()
