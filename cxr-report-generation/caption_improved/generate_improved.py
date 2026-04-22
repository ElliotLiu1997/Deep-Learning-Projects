from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from caption_improved.decoding_utils import apply_repetition_penalty, select_with_topk_and_ngram
from caption_only.dataset import CaptionDataset, build_transforms, load_caption_metadata
from caption_only.models import CaptioningModel
from caption_only.utils import decode_sequence, load_model_state_dict_flexible, save_json, set_seed, setup_device_and_parallel
from share_encoder.model import SharedEncoderMultiTaskModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate captions with improved decoding (no retraining).")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["lstm", "lstm_attn", "transformer", "share_encoder"],
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=100)

    parser.add_argument("--data_csv", type=str, default="info.csv")
    parser.add_argument("--image_dir", type=str, default="images/images_normalized")
    parser.add_argument("--encoder_checkpoint", type=str, default="classification_only/outputs/resnet/best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--ngram_n", type=int, default=3)
    return parser.parse_args()


@torch.no_grad()
def _generate_lstm(decoder, feat: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int, top_k: int, repetition_penalty: float, ngram_n: int):
    h, c = decoder._init_state(feat)
    cur = torch.tensor([[sos_idx]], dtype=torch.long, device=feat.device)
    generated = []
    for _ in range(max_len):
        emb = decoder.embedding(cur)
        out, (h, c) = decoder.lstm(emb, (h, c))
        logits = decoder.fc(out[:, -1, :]).squeeze(0)
        logits = apply_repetition_penalty(logits, generated, penalty=repetition_penalty)
        nxt = select_with_topk_and_ngram(logits, generated, top_k=top_k, ngram_n=ngram_n)
        generated.append(nxt)
        if eos_idx >= 0 and nxt == eos_idx:
            break
        cur = torch.tensor([[nxt]], dtype=torch.long, device=feat.device)
    return generated


@torch.no_grad()
def _generate_lstm_attn(decoder, feat_tokens: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int, top_k: int, repetition_penalty: float, ngram_n: int):
    h, c = decoder._init_state(feat_tokens)
    cur = torch.tensor([sos_idx], dtype=torch.long, device=feat_tokens.device)
    generated = []
    for _ in range(max_len):
        emb = decoder.embedding(cur)
        context, _ = decoder.attention(feat_tokens, h)
        step_in = torch.cat([emb, context], dim=-1)
        h, c = decoder.lstm_cell(step_in, (h, c))
        logits = decoder.fc(h).squeeze(0)
        logits = apply_repetition_penalty(logits, generated, penalty=repetition_penalty)
        nxt = select_with_topk_and_ngram(logits, generated, top_k=top_k, ngram_n=ngram_n)
        generated.append(nxt)
        if eos_idx >= 0 and nxt == eos_idx:
            break
        cur = torch.tensor([nxt], dtype=torch.long, device=feat_tokens.device)
    return generated


@torch.no_grad()
def _generate_transformer(decoder, feat_tokens: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int, top_k: int, repetition_penalty: float, ngram_n: int):
    generated = [sos_idx]
    out_tokens = []
    for _ in range(max_len):
        inp = torch.tensor([generated], dtype=torch.long, device=feat_tokens.device)
        logits = decoder._decode(feat_tokens, inp)[:, -1, :].squeeze(0)
        logits = apply_repetition_penalty(logits, out_tokens, penalty=repetition_penalty)
        nxt = select_with_topk_and_ngram(logits, out_tokens, top_k=top_k, ngram_n=ngram_n)
        out_tokens.append(nxt)
        generated.append(nxt)
        if eos_idx >= 0 and nxt == eos_idx:
            break
    return out_tokens


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    metadata = load_caption_metadata(args.data_csv)
    pad_idx = metadata["pad_idx"]
    sos_idx = metadata["sos_idx"]
    eos_idx = metadata["eos_idx"]
    idx2word = metadata["idx2word"]

    ds = CaptionDataset(args.data_csv, args.image_dir, split="test", transform=build_transforms())
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.model_type == "share_encoder":
        model = SharedEncoderMultiTaskModel(
            vocab_size=metadata["vocab_size"],
            num_classes=13,
            encoder_checkpoint=args.encoder_checkpoint,
            pad_idx=pad_idx,
        )
    else:
        model = CaptioningModel(
            decoder_type=args.model_type,
            vocab_size=metadata["vocab_size"],
            encoder_checkpoint=args.encoder_checkpoint,
            pad_idx=pad_idx,
        )
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    load_model_state_dict_flexible(model, state_dict)

    model.freeze_encoder()
    device, model, gpu_ids = setup_device_and_parallel(model, args.gpu_ids)
    model.eval()
    print(f"Using device={device}, gpu_ids={gpu_ids if gpu_ids else 'cpu'}")

    rows = []
    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    for images, captions, filenames in loader:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        encoded = target_model.encoder.encode(images)

        gt_ids = captions[:, 1:].detach().cpu().tolist()

        for i, fname in enumerate(filenames):
            if args.model_type == "lstm":
                feat = encoded["global"][i : i + 1]
                pred_seq = _generate_lstm(
                    target_model.decoder,
                    feat,
                    sos_idx=sos_idx,
                    eos_idx=eos_idx,
                    max_len=args.max_len,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    ngram_n=args.ngram_n,
                )
            elif args.model_type in ("lstm_attn", "share_encoder"):
                feat_tokens = encoded["tokens"][i : i + 1]
                pred_seq = _generate_lstm_attn(
                    target_model.decoder,
                    feat_tokens,
                    sos_idx=sos_idx,
                    eos_idx=eos_idx,
                    max_len=args.max_len,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    ngram_n=args.ngram_n,
                )
            else:
                feat_tokens = encoded["tokens"][i : i + 1]
                pred_seq = _generate_transformer(
                    target_model.decoder,
                    feat_tokens,
                    sos_idx=sos_idx,
                    eos_idx=eos_idx,
                    max_len=args.max_len,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    ngram_n=args.ngram_n,
                )

            pred_tokens = decode_sequence(pred_seq, idx2word, pad_idx, sos_idx, eos_idx)
            gt_tokens = decode_sequence(gt_ids[i], idx2word, pad_idx, sos_idx, eos_idx)
            rows.append(
                {
                    "image_id": fname,
                    "prediction": " ".join(pred_tokens),
                    "ground_truth": " ".join(gt_tokens),
                }
            )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(rows, str(output_path))
    print(f"Saved improved captions to: {output_path}")


if __name__ == "__main__":
    main()
