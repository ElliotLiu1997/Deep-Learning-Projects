import ast
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn


def parse_label_vector(raw_value: str) -> List[float]:
    """Parse a stringified list like "[1, 0, ...]" into a float list."""
    if isinstance(raw_value, list):
        return [float(v) for v in raw_value]

    parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, list):
        raise ValueError(f"label_vec must parse to a list, got: {type(parsed)}")
    return [float(v) for v in parsed]


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
        raise ValueError(
            f"Invalid GPU ids {invalid}. Available ids: {sorted(available)}"
        )

    device = torch.device(f"cuda:{gpu_ids[0]}")
    model = model.to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    return device, model, gpu_ids


def get_model_state_dict(model: nn.Module):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state_dict_flexible(model: nn.Module, state_dict):
    target = model.module if isinstance(model, nn.DataParallel) else model
    try:
        target.load_state_dict(state_dict)
        return
    except RuntimeError as err:
        original_error = err

    # Handle checkpoints saved with/without DataParallel "module." prefix.
    if any(k.startswith("module.") for k in state_dict.keys()):
        stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        target.load_state_dict(stripped)
        return

    raise original_error
