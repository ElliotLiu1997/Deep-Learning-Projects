from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torchvision.models as tv_models


class PretrainedResnetEncoder(nn.Module):
    """
    ResNet18 encoder loaded from the classification checkpoint.
    Uses features only (no classification logits).
    """

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Pretrained encoder checkpoint not found: {ckpt_path}. "
                "Expected classification_only/outputs/resnet/best_model.pt"
            )

        try:
            backbone = tv_models.resnet18(weights=None)
        except TypeError:
            backbone = tv_models.resnet18(pretrained=False)

        state_dict = self._load_checkpoint_state_dict(ckpt_path)
        state_dict = self._normalize_state_dict_keys(state_dict)

        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        missing, unexpected = backbone.load_state_dict(filtered_state_dict, strict=False)

        allowed_missing = {"fc.weight", "fc.bias"}
        missing_set = set(missing)
        if not missing_set.issubset(allowed_missing):
            raise RuntimeError(
                "Unexpected missing keys while loading pretrained ResNet encoder: "
                f"{sorted(missing_set)}"
            )
        if unexpected:
            raise RuntimeError(
                "Unexpected keys while loading pretrained ResNet encoder: "
                f"{sorted(unexpected)}"
            )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.out_channels = backbone.layer4[-1].conv2.out_channels

    def _load_checkpoint_state_dict(self, checkpoint_path: Path) -> Dict[str, torch.Tensor]:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ["state_dict", "model_state_dict"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
        if isinstance(ckpt, dict):
            return ckpt
        raise ValueError("Checkpoint format is not supported. Expected a state_dict-like object.")

    def _normalize_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized = {}
        for key, value in state_dict.items():
            k = key
            if k.startswith("module."):
                k = k.replace("module.", "", 1)
            if k.startswith("model."):
                k = k.replace("model.", "", 1)
            normalized[k] = value
        return normalized

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.forward_spatial(x)
        pooled = self.avgpool(spatial)
        return torch.flatten(pooled, 1)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.forward_spatial(x)
        b, c, h, w = spatial.shape
        return spatial.view(b, c, h * w).permute(0, 2, 1).contiguous()

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        spatial = self.forward_spatial(x)
        pooled = self.avgpool(spatial)
        global_feat = torch.flatten(pooled, 1)
        b, c, h, w = spatial.shape
        tokens = spatial.view(b, c, h * w).permute(0, 2, 1).contiguous()
        return {
            "spatial": spatial,
            "global": global_feat,
            "tokens": tokens,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_global(x)
