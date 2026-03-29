"""
Model definitions for Flickr8k image captioning:
- CNN encoder (ResNet18, frozen)
- RNN decoder (RNN / GRU / LSTM / LSTM+Dropout)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class CNNEncoder(nn.Module):
    """
    Frozen ResNet18 feature extractor.
    - Removes final FC layer
    - Outputs 512-d feature
    - Projects to embedding dimension with a linear layer
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove final FC: keep up to global average pooling output.
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # [B, 512, 1, 1]

        # Freeze CNN parameters.
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.feature_proj = nn.Linear(512, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224]
        Returns:
            projected features: [B, embed_dim]
        """
        with torch.no_grad():
            feats = self.cnn(images)  # [B, 512, 1, 1]
        feats = feats.flatten(1)      # [B, 512]
        return self.feature_proj(feats)


class RNNDecoder(nn.Module):
    """
    Caption decoder with configurable recurrent cell:
    - "rnn"
    - "gru"
    - "lstm"
    - "lstm_dropout" (dropout applied to embeddings + recurrent output)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_dropout = self.rnn_type == "lstm_dropout"

        # Normalize internal type for recurrent module construction.
        cell_type = "lstm" if self.rnn_type == "lstm_dropout" else self.rnn_type
        if cell_type not in {"rnn", "gru", "lstm"}:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        if cell_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        elif cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        else:  # lstm
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )

        # Map image embedding to recurrent hidden state.
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.init_c = nn.Linear(embed_dim, hidden_dim) if cell_type == "lstm" else None

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def _init_state(self, image_emb: torch.Tensor):
        """
        Build initial recurrent state from image embedding.
        image_emb: [B, embed_dim]
        """
        h0 = torch.tanh(self.init_h(image_emb))  # [B, H]
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [L, B, H]

        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.tanh(self.init_c(image_emb))  # [B, H]
            c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [L, B, H]
            return (h0, c0)
        return h0

    def forward(self, image_emb: torch.Tensor, input_captions: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Args:
            image_emb: [B, embed_dim]
            input_captions: [B, T] (typically starts with <start> token)
        Returns:
            logits: [B, T, vocab_size]
        """
        x = self.embedding(input_captions)  # [B, T, E]
        if self.use_dropout:
            x = self.dropout(x)

        state = self._init_state(image_emb)
        out, _ = self.rnn(x, state)  # [B, T, H]
        if self.use_dropout:
            out = self.dropout(out)

        logits = self.fc_out(out)    # [B, T, V]
        return logits


class ImageCaptioningModel(nn.Module):
    """
    Full captioning model wrapper:
    - If using cached CNN features: call forward_features(...)
    - If using raw images: call forward_images(...)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=embed_dim)
        self.decoder = RNNDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rnn_type=rnn_type,
            dropout=dropout,
            pad_idx=pad_idx,
        )

    def project_cached_features(self, cached_features: torch.Tensor) -> torch.Tensor:
        """
        Project cached raw ResNet features (shape [B, 512]) into embedding space.
        """
        return self.encoder.feature_proj(cached_features)

    def forward_features(self, cached_features: torch.Tensor, input_captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass when using cached 512-d CNN features.
        """
        image_emb = self.project_cached_features(cached_features)
        return self.decoder(image_emb, input_captions)

    def forward_images(self, images: torch.Tensor, input_captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass when using raw images.
        """
        image_emb = self.encoder(images)
        return self.decoder(image_emb, input_captions)

    @torch.no_grad()
    def greedy_decode(
        self,
        image_embedding: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 25,
    ) -> torch.Tensor:
        """
        Greedy caption decoding for a single image embedding.

        Args:
            image_embedding: [1, embed_dim]
        Returns:
            token_ids: [<= max_len]
        """
        self.eval()
        device = image_embedding.device

        # Initialize recurrent state from image embedding.
        state = self.decoder._init_state(image_embedding)

        token = torch.tensor([[start_idx]], dtype=torch.long, device=device)
        generated = []

        for _ in range(max_len):
            emb = self.decoder.embedding(token)  # [1, 1, E]
            if self.decoder.use_dropout:
                emb = self.decoder.dropout(emb)

            out, state = self.decoder.rnn(emb, state)  # out: [1, 1, H]
            logits = self.decoder.fc_out(out[:, -1, :])  # [1, V]
            next_token = torch.argmax(logits, dim=-1)    # [1]

            token_id = int(next_token.item())
            generated.append(token_id)

            if token_id == end_idx:
                break

            token = next_token.unsqueeze(1)  # [1, 1]

        return torch.tensor(generated, dtype=torch.long, device=device)
