import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import CaptionDataset, build_transforms, load_caption_metadata
    from .models import CaptioningModel
    from .utils import (
        compute_bleu_scores,
        compute_cider,
        compute_rouge_l,
        decode_sequence,
        load_model_state_dict_flexible,
        save_json,
        set_seed,
        setup_device_and_parallel,
    )
except ImportError:
    from dataset import CaptionDataset, build_transforms, load_caption_metadata
    from models import CaptioningModel
    from utils import (
        compute_bleu_scores,
        compute_cider,
        compute_rouge_l,
        decode_sequence,
        load_model_state_dict_flexible,
        save_json,
        set_seed,
        setup_device_and_parallel,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image captioning model")
    parser.add_argument("--decoder_type", type=str, required=True, choices=["lstm", "lstm_attn", "transformer"])

    parser.add_argument("--data_csv", type=str, default="info.csv")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--encoder_checkpoint", type=str, default="classification_only/outputs/resnet/best_model.pt")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_ids", type=str, default="0,1")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_gen_len", type=int, default=None)
    parser.add_argument("--decode_method", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=0.7)
    parser.add_argument("--postprocess_repeats", dest="postprocess_repeats", action="store_true")
    parser.add_argument("--no_postprocess_repeats", dest="postprocess_repeats", action="store_false")
    parser.set_defaults(postprocess_repeats=True)
    return parser.parse_args()


def _strip_repeat_loops(token_ids, eos_idx: int):
    """
    Lightweight decode-time cleanup for degenerate loops:
    - remove immediate token repeats: A A -> A
    - remove immediate repeated bi/tri-grams: A B A B -> A B, A B C A B C -> A B C
    """
    out = []
    for tok in token_ids:
        t = int(tok)
        if eos_idx >= 0 and t == eos_idx:
            break
        out.append(t)

        if len(out) >= 2 and out[-1] == out[-2]:
            out.pop()
            continue

        for n in (3, 2):
            while len(out) >= 2 * n and out[-2 * n : -n] == out[-n:]:
                del out[-n:]

    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else Path("caption_only") / "outputs" / args.decoder_type
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path) if args.model_path else output_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained caption model not found: {model_path}")

    metadata = load_caption_metadata(args.data_csv)

    test_dataset = CaptionDataset(
        data_csv=args.data_csv,
        image_dir=args.image_dir,
        split="test",
        transform=build_transforms(),
    )
    test_loader = DataLoader(
        test_dataset,
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
    state_dict = torch.load(model_path, map_location="cpu")
    load_model_state_dict_flexible(model, state_dict)

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

    for images, captions, filenames in test_loader:
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        if args.decode_method == "beam":
            pred_ids = target_model.generate_beam(
                images,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_len=max_gen_len,
                beam_size=args.beam_size,
                length_penalty=args.length_penalty,
            )
        else:
            pred_ids = target_model.generate(images, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_gen_len)

        pred_ids = pred_ids.detach().cpu().tolist()
        gt_ids = captions[:, 1:].detach().cpu().tolist()

        for fname, pred_seq, gt_seq in zip(filenames, pred_ids, gt_ids):
            if args.postprocess_repeats:
                pred_seq = _strip_repeat_loops(pred_seq, eos_idx=eos_idx)
            pred_tokens = decode_sequence(pred_seq, idx2word, pad_idx, sos_idx, eos_idx)
            gt_tokens = decode_sequence(gt_seq, idx2word, pad_idx, sos_idx, eos_idx)

            pred_text = " ".join(pred_tokens)
            gt_text = " ".join(gt_tokens)

            caption_rows.append(
                {
                    "image_id": fname,
                    "prediction": pred_text,
                    "ground_truth": gt_text,
                }
            )
            refs.append([gt_tokens])
            rouge_refs.append(gt_tokens)
            hyps.append(pred_tokens)

    bleu_scores = compute_bleu_scores(refs, hyps, max_n=4)
    rouge_l = compute_rouge_l(rouge_refs, hyps)
    cider = compute_cider(refs, hyps)

    metrics = {
        "model": args.decoder_type,
        "BLEU-1": bleu_scores["BLEU-1"],
        "BLEU-2": bleu_scores["BLEU-2"],
        "BLEU-3": bleu_scores["BLEU-3"],
        "BLEU-4": bleu_scores["BLEU-4"],
        "ROUGE-L": rouge_l,
        "CIDEr": cider,
        "decode_method": args.decode_method,
        "beam_size": args.beam_size if args.decode_method == "beam" else 1,
        "length_penalty": args.length_penalty if args.decode_method == "beam" else 0.0,
    }

    captions_path = output_dir / "captions.json"
    metrics_path = output_dir / "metrics.json"

    save_json(caption_rows, str(captions_path))
    save_json(metrics, str(metrics_path))

    print(f"Saved captions to: {captions_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    main()
