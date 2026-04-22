import torch
import torch.nn as nn

from .encoder import PretrainedResnetEncoder
from .lstm import LSTMDecoder
from .lstm_attn import LSTMAttnDecoder
from .transformer import TransformerDecoderModel


class CaptioningModel(nn.Module):
    def __init__(
        self,
        decoder_type: str,
        vocab_size: int,
        encoder_checkpoint: str,
        pad_idx: int,
        transformer_d_model: int = 256,
    ) -> None:
        super().__init__()
        self.decoder_type = decoder_type

        self.encoder = PretrainedResnetEncoder(encoder_checkpoint)
        feat_dim = self.encoder.out_channels

        if decoder_type == "lstm":
            self.decoder = LSTMDecoder(vocab_size=vocab_size, feat_dim=feat_dim, pad_idx=pad_idx)
        elif decoder_type == "lstm_attn":
            self.decoder = LSTMAttnDecoder(vocab_size=vocab_size, feat_dim=feat_dim, pad_idx=pad_idx)
        elif decoder_type == "transformer":
            self.decoder = TransformerDecoderModel(
                vocab_size=vocab_size,
                feat_dim=feat_dim,
                d_model=transformer_d_model,
                nhead=4,
                num_layers=2,
                pad_idx=pad_idx,
            )
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        input_seq: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        encoded = self.encoder.encode(images)
        if self.decoder_type == "lstm":
            return self.decoder(encoded["global"], input_seq, teacher_forcing_ratio=teacher_forcing_ratio)

        return self.decoder(encoded["tokens"], input_seq, teacher_forcing_ratio=teacher_forcing_ratio)

    def generate(self, images: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int) -> torch.Tensor:
        encoded = self.encoder.encode(images)
        if self.decoder_type == "lstm":
            return self.decoder.generate(encoded["global"], sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len)

        return self.decoder.generate(encoded["tokens"], sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len)

    @torch.no_grad()
    def generate_beam(
        self,
        images: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
        beam_size: int = 3,
        length_penalty: float = 0.7,
    ) -> torch.Tensor:
        encoded = self.encoder.encode(images)
        outputs = []

        if self.decoder_type == "lstm":
            feats = encoded["global"]
        else:
            feats = encoded["tokens"]

        for i in range(feats.size(0)):
            feat_i = feats[i : i + 1]
            seq = self.decoder.beam_search(
                feat_i,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
            ).squeeze(0)
            outputs.append(seq)

        max_out_len = max((x.numel() for x in outputs), default=1)
        padded = []
        for seq in outputs:
            if seq.numel() < max_out_len:
                pad = torch.full((max_out_len - seq.numel(),), eos_idx, dtype=torch.long, device=seq.device)
                seq = torch.cat([seq, pad], dim=0)
            padded.append(seq)
        return torch.stack(padded, dim=0)
