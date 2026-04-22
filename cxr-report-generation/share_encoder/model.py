import torch
import torch.nn as nn

from caption_only.models.encoder import PretrainedResnetEncoder
from caption_only.models.lstm_attn import LSTMAttnDecoder


class SharedEncoderMultiTaskModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        encoder_checkpoint: str,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.encoder = PretrainedResnetEncoder(encoder_checkpoint)
        feat_dim = self.encoder.out_channels

        self.decoder = LSTMAttnDecoder(vocab_size=vocab_size, feat_dim=feat_dim, pad_idx=pad_idx)
        self.cls_head = nn.Linear(feat_dim, num_classes)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, images: torch.Tensor, input_seq: torch.Tensor, teacher_forcing_ratio: float = 1.0):
        encoded = self.encoder.encode(images)
        caption_logits = self.decoder(encoded["tokens"], input_seq, teacher_forcing_ratio=teacher_forcing_ratio)
        cls_logits = self.cls_head(encoded["global"])
        return caption_logits, cls_logits

    @torch.no_grad()
    def generate(self, images: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int) -> torch.Tensor:
        encoded = self.encoder.encode(images)
        return self.decoder.generate(encoded["tokens"], sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len)

