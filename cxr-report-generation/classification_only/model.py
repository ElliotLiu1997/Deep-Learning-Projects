import torch
import torch.nn as nn
import torchvision.models as tv_models


def _freeze_all_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


class MultiLabelClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        encoder: str = "resnet",
        freeze_backbone: bool = True,
        vit_model_name: str = "vit_base_patch16_224",
    ) -> None:
        super().__init__()

        encoder = encoder.lower()
        if encoder == "resnet":
            self.model = self._build_resnet(num_classes, freeze_backbone)
        elif encoder == "vit":
            self.model = self._build_vit(num_classes, freeze_backbone, vit_model_name)
        else:
            raise ValueError(f"Unsupported encoder '{encoder}'. Use 'resnet' or 'vit'.")

    def _build_resnet(self, num_classes: int, freeze_backbone: bool) -> nn.Module:
        try:
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            backbone = tv_models.resnet18(pretrained=True)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            _freeze_all_params(backbone)
            for p in backbone.fc.parameters():
                p.requires_grad = True

        return backbone

    def _build_vit(self, num_classes: int, freeze_backbone: bool, vit_model_name: str) -> nn.Module:
        # Prefer timm for ViT. Fall back to torchvision if timm is unavailable.
        try:
            import timm

            model = timm.create_model(vit_model_name, pretrained=True, num_classes=num_classes)
            if freeze_backbone:
                _freeze_all_params(model)
                if hasattr(model, "head") and model.head is not None:
                    for p in model.head.parameters():
                        p.requires_grad = True
                else:
                    for p in model.parameters():
                        p.requires_grad = True
            return model
        except ImportError:
            pass

        try:
            vit = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT)
        except AttributeError:
            vit = tv_models.vit_b_16(pretrained=True)

        in_features = vit.heads.head.in_features
        vit.heads.head = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            _freeze_all_params(vit)
            for p in vit.heads.head.parameters():
                p.requires_grad = True

        return vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
