import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_gpu_ids(gpu_ids_arg: str) -> List[int]:
    if gpu_ids_arg.lower() == "all":
        return list(range(torch.cuda.device_count()))
    return [int(x.strip()) for x in gpu_ids_arg.split(",") if x.strip()]


def setup_device_and_parallel(model: nn.Module, gpu_ids_arg: str):
    if not torch.cuda.is_available():
        return torch.device("cpu"), model, []

    gpu_ids = parse_gpu_ids(gpu_ids_arg)
    if not gpu_ids:
        raise ValueError("No valid GPU ids were provided. Example: --gpu_ids 0,1")

    available = set(range(torch.cuda.device_count()))
    invalid = [gid for gid in gpu_ids if gid not in available]
    if invalid:
        raise ValueError(f"Invalid GPU ids {invalid}. Available ids: {sorted(available)}")

    device = torch.device(f"cuda:{gpu_ids[0]}")
    model = model.to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    return device, model, gpu_ids


def get_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state_dict_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    target = model.module if isinstance(model, nn.DataParallel) else model
    try:
        target.load_state_dict(state_dict)
        return
    except RuntimeError as err:
        original_error = err

    if any(k.startswith("module.") for k in state_dict.keys()):
        stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        target.load_state_dict(stripped)
        return

    raise original_error


def save_json(data, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_loss_curve(train_losses: Sequence[float], val_losses: Sequence[float], output_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
    return True


def decode_sequence(
    seq: Sequence[int],
    idx2word: Sequence[str],
    pad_idx: int,
    sos_idx: int,
    eos_idx: int,
) -> List[str]:
    words: List[str] = []
    for token_id in seq:
        token_id = int(token_id)
        if token_id == eos_idx:
            break
        if token_id in (pad_idx, sos_idx):
            continue

        if 0 <= token_id < len(idx2word):
            w = idx2word[token_id]
        else:
            w = "<UNK>"

        if w in ("<PAD>", "<SOS>", "<EOS>"):
            continue
        words.append(w)
    return words


def _count_ngrams(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i : i + n])
        counts[g] = counts.get(g, 0) + 1
    return counts


def compute_bleu_scores(
    references: Sequence[Sequence[Sequence[str]]],
    hypotheses: Sequence[Sequence[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have same length")

    ref_len = 0
    hyp_len = 0
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n

    for ref_list, hyp in zip(references, hypotheses):
        if not ref_list:
            raise ValueError("Each hypothesis must have at least one reference.")

        ref_lens = [len(r) for r in ref_list]
        best_ref_len = min(ref_lens, key=lambda x: (abs(x - len(hyp)), x))
        ref_len += best_ref_len
        hyp_len += len(hyp)

        for n in range(1, max_n + 1):
            hyp_counts = _count_ngrams(hyp, n)
            max_ref_counts: Dict[Tuple[str, ...], int] = {}
            for ref in ref_list:
                ref_counts = _count_ngrams(ref, n)
                for g, c in ref_counts.items():
                    max_ref_counts[g] = max(max_ref_counts.get(g, 0), c)

            total_counts[n - 1] += sum(hyp_counts.values())
            for g, c in hyp_counts.items():
                clipped_counts[n - 1] += min(c, max_ref_counts.get(g, 0))

    precisions = []
    for i in range(max_n):
        if total_counts[i] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[i] / total_counts[i])

    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = float(np.exp(1.0 - (ref_len / max(hyp_len, 1))))

    scores: Dict[str, float] = {}
    for n in range(1, max_n + 1):
        p = precisions[:n]
        if min(p) <= 0:
            bleu_n = 0.0
        else:
            bleu_n = bp * float(np.exp(np.mean(np.log(p))))
        scores[f"BLEU-{n}"] = bleu_n
    return scores


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def compute_rouge_l(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have same length")

    f1_scores = []
    for ref, hyp in zip(references, hypotheses):
        lcs = _lcs_len(ref, hyp)
        if lcs == 0:
            f1_scores.append(0.0)
            continue

        prec = lcs / max(len(hyp), 1)
        rec = lcs / max(len(ref), 1)
        if prec + rec == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append((2 * prec * rec) / (prec + rec))

    if not f1_scores:
        return 0.0
    return float(np.mean(f1_scores))


def _tf_vector(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], float]:
    counts = _count_ngrams(tokens, n)
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {g: c / total for g, c in counts.items()}


def compute_cider(
    references: Sequence[Sequence[Sequence[str]]],
    hypotheses: Sequence[Sequence[str]],
    max_n: int = 4,
    sigma: float = 6.0,
) -> float:
    """
    Corpus-level CIDEr-style score (supports one or more references per sample).
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have same length")
    if not references:
        return 0.0

    num_samples = len(references)
    doc_freq = [dict() for _ in range(max_n)]

    for ref_list in references:
        seen_per_n = [set() for _ in range(max_n)]
        for ref in ref_list:
            for n in range(1, max_n + 1):
                tf = _tf_vector(ref, n)
                seen_per_n[n - 1].update(tf.keys())
        for n in range(max_n):
            for g in seen_per_n[n]:
                doc_freq[n][g] = doc_freq[n].get(g, 0) + 1

    def _idf(n_idx: int, gram: Tuple[str, ...]) -> float:
        df = doc_freq[n_idx].get(gram, 0)
        return float(np.log((num_samples + 1.0) / (df + 1.0)))

    sample_scores = []
    for ref_list, hyp in zip(references, hypotheses):
        hyp_len = len(hyp)
        hyp_vecs = []
        hyp_norms = []
        for n in range(1, max_n + 1):
            tf = _tf_vector(hyp, n)
            vec = {g: v * _idf(n - 1, g) for g, v in tf.items()}
            norm = float(np.sqrt(sum(x * x for x in vec.values())))
            hyp_vecs.append(vec)
            hyp_norms.append(norm)

        ref_scores = []
        ref_lens = [len(r) for r in ref_list] if ref_list else [0]
        avg_ref_len = float(np.mean(ref_lens))
        len_penalty = float(np.exp(-((hyp_len - avg_ref_len) ** 2) / (2 * (sigma ** 2))))

        for ref in ref_list:
            sim_n = []
            for n in range(1, max_n + 1):
                ref_tf = _tf_vector(ref, n)
                ref_vec = {g: v * _idf(n - 1, g) for g, v in ref_tf.items()}
                ref_norm = float(np.sqrt(sum(x * x for x in ref_vec.values())))

                if hyp_norms[n - 1] == 0.0 or ref_norm == 0.0:
                    sim_n.append(0.0)
                    continue

                overlap = set(hyp_vecs[n - 1]).intersection(ref_vec)
                dot = float(sum(hyp_vecs[n - 1][g] * ref_vec[g] for g in overlap))
                sim_n.append(dot / (hyp_norms[n - 1] * ref_norm))

            ref_scores.append(float(np.mean(sim_n)) * len_penalty)

        if ref_scores:
            sample_scores.append(10.0 * float(np.mean(ref_scores)))
        else:
            sample_scores.append(0.0)

    return float(np.mean(sample_scores))
