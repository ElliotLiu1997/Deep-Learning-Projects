import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import MultiTaskDataset, build_transforms, load_caption_metadata
    from .model import SharedEncoderMultiTaskModel
except ImportError:
    from dataset import MultiTaskDataset, build_transforms, load_caption_metadata
    from model import SharedEncoderMultiTaskModel

from caption_only.utils import (
    compute_bleu_scores,
    compute_cider,
    compute_rouge_l,
    decode_sequence,
    load_model_state_dict_flexible,
    save_json,
    set_seed,
    setup_device_and_parallel,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate shared-encoder multi-task model")
    p.add_argument("--data_csv", type=str, default="info.csv")
    p.add_argument("--image_dir", type=str, default="images/images_normalized")
    p.add_argument("--encoder_checkpoint", type=str, default="classification_only/outputs/resnet/best_model.pt")
    p.add_argument("--model_path", type=str, default="share_encoder/outputs/lstm_attn/best_model.pt")
    p.add_argument("--output_dir", type=str, default="share_encoder/outputs/lstm_attn")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu_ids", type=str, default="0,1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_gen_len", type=int, default=None)
    return p.parse_args()


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


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    metadata = load_caption_metadata(args.data_csv)
    test_ds = MultiTaskDataset(args.data_csv, args.image_dir, split="test", transform=build_transforms())
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SharedEncoderMultiTaskModel(
        vocab_size=metadata["vocab_size"],
        num_classes=test_ds.num_classes or 13,
        encoder_checkpoint=args.encoder_checkpoint,
        pad_idx=metadata["pad_idx"],
    )
    sd = torch.load(model_path, map_location="cpu")
    load_model_state_dict_flexible(model, sd)
    model.freeze_encoder()
    device, model, gpu_ids = setup_device_and_parallel(model, args.gpu_ids)
    model.eval()
    print(f"Using device={device}, gpu_ids={gpu_ids if gpu_ids else 'cpu'}")

    pad_idx = metadata["pad_idx"]
    sos_idx = metadata["sos_idx"]
    eos_idx = metadata["eos_idx"]
    idx2word = metadata["idx2word"]
    max_gen_len = args.max_gen_len if args.max_gen_len is not None else max(1, metadata["max_seq_len"] - 1)

    caption_rows = []
    refs = []
    rouge_refs = []
    hyps = []
    all_cls_logits = []
    all_cls_labels = []

    for images, captions, labels, filenames in loader:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        pred_ids = target_model.generate(images, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_gen_len)
        input_seq = captions[:, :-1]
        _, cls_logits = target_model(images, input_seq, teacher_forcing_ratio=1.0)

        all_cls_logits.append(cls_logits.detach().cpu())
        all_cls_labels.append(labels.detach().cpu())

        pred_ids = pred_ids.detach().cpu().tolist()
        gt_ids = captions[:, 1:].detach().cpu().tolist()
        for fname, pred_seq, gt_seq in zip(filenames, pred_ids, gt_ids):
            pred_tokens = decode_sequence(pred_seq, idx2word, pad_idx, sos_idx, eos_idx)
            gt_tokens = decode_sequence(gt_seq, idx2word, pad_idx, sos_idx, eos_idx)
            pred_text = " ".join(pred_tokens)
            gt_text = " ".join(gt_tokens)
            caption_rows.append({"image_id": fname, "prediction": pred_text, "ground_truth": gt_text})
            refs.append([gt_tokens])
            rouge_refs.append(gt_tokens)
            hyps.append(pred_tokens)

    cls_logits = torch.cat(all_cls_logits, dim=0)
    cls_labels = torch.cat(all_cls_labels, dim=0)
    cls_macro_f1 = _macro_f1_from_logits(cls_logits, cls_labels)

    bleu_scores = compute_bleu_scores(refs, hyps, max_n=4)
    rouge_l = compute_rouge_l(rouge_refs, hyps)
    cider = compute_cider(refs, hyps)

    metrics = {
        "model": "shared_resnet_lstm_attn",
        "BLEU-1": bleu_scores["BLEU-1"],
        "BLEU-2": bleu_scores["BLEU-2"],
        "BLEU-3": bleu_scores["BLEU-3"],
        "BLEU-4": bleu_scores["BLEU-4"],
        "ROUGE-L": rouge_l,
        "CIDEr": cider,
        "classification_macro_f1": cls_macro_f1,
    }

    save_json(caption_rows, str(out_dir / "captions.json"))
    save_json(metrics, str(out_dir / "metrics.json"))
    print(f"Saved captions to: {out_dir / 'captions.json'}")
    print(f"Saved metrics to: {out_dir / 'metrics.json'}")
    print(metrics)


if __name__ == "__main__":
    main()

