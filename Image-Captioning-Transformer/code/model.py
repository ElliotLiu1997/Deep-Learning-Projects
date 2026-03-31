"""
Model definitions for Flickr8k image captioning:
- CNN encoder (ResNet18 spatial tokens for Transformer)
- ViT encoder (patch tokens)
- RNN decoder (RNN / GRU / LSTM / LSTM+Dropout)
- Transformer decoder (masked self-attn + cross-attn)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights


class CNNGlobalEncoder(nn.Module):
    """
    Original frozen ResNet18 global feature extractor for RNN-family decoders.
    - Removes final FC layer
    - Outputs 512-d pooled feature
    - Projects to embedding dimension with a linear layer
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Keep up to global average pooling output.
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


class CNNEncoder(nn.Module):
    """
    Frozen ResNet18 token encoder for Transformer decoder.
    - Removes avgpool and fc
    - Uses spatial feature map [B, 512, 7, 7]
    - Flattens to 49 tokens, projects to d_model
    - Adds learnable positional encoding for 49 spatial tokens
    """

    def __init__(self, d_model: int):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Keep convolutional trunk up to layer4 output (before avgpool/fc).
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])  # [B, 512, 7, 7]
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.token_proj = nn.Linear(512, d_model)
        self.pos_embed = nn.Parameter(torch.randn(49, d_model) * 0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224]
        Returns:
            image tokens: [B, 49, d_model]
        """
        with torch.no_grad():
            feat_map = self.cnn(images)  # [B, 512, 7, 7]

        tokens = feat_map.flatten(2).transpose(1, 2)  # [B, 49, 512]
        tokens = self.token_proj(tokens)              # [B, 49, d_model]
        tokens = tokens + self.pos_embed.unsqueeze(0)
        return tokens


class ViTEncoder(nn.Module):
    """
    Frozen ViT-B/16 encoder for Transformer decoder.
    - Uses all patch tokens (drops CLS token)
    - Projects token dim 768 -> d_model
    - Does NOT add extra positional encoding (ViT already has positional embeddings)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        for p in self.vit.parameters():
            p.requires_grad = False

        self.token_proj = nn.Linear(768, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224]
        Returns:
            patch tokens: [B, num_patches, d_model] (num_patches=196 for 224/16)
        """
        with torch.no_grad():
            x = self.vit._process_input(images)  # [B, 196, 768]
            bsz = x.shape[0]
            cls = self.vit.class_token.expand(bsz, -1, -1)
            x = torch.cat([cls, x], dim=1)      # [B, 197, 768]
            x = self.vit.encoder(x)              # [B, 197, 768]
            x = x[:, 1:, :]                      # keep patch tokens only

        return self.token_proj(x)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for caption tokens.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        t = x.size(1)
        return x + self.pe[:, :t, :]


class TransformerDecoderLayer(nn.Module):
    """
    One Transformer decoder block:
    1) masked self-attention on caption tokens
    2) cross-attention where caption queries attend to image memory tokens
    3) feed-forward network
    Each sub-layer uses residual + layer norm.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        self_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: caption states [B, T, d_model]
            memory: image tokens [B, N, d_model]
            causal_mask: upper-triangular bool mask [T, T], True means masked
            self_key_padding_mask: [B, T], True means padding token
        """
        # Masked self-attention: Q,K,V all from captions.
        self_attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            key_padding_mask=self_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout1(self_attn_out))

        # Cross-attention: Q from captions, K,V from image features.
        cross_attn_out, _ = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            need_weights=False,
        )
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer caption decoder for teacher forcing and autoregressive decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        pad_idx: int,
        max_len: int = 512,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper triangular causal mask [T, T].
        True entries are masked (future positions).
        """
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, memory: torch.Tensor, input_captions: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forcing forward pass.

        Args:
            memory: image features [B, N, d_model]
            input_captions: caption token ids [B, T]
        Returns:
            logits: [B, T, vocab_size]
        """
        x = self.embedding(input_captions) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        seq_len = input_captions.size(1)
        causal_mask = self.generate_causal_mask(seq_len, input_captions.device)
        self_key_padding_mask = input_captions.eq(self.pad_idx)

        for layer in self.layers:
            x = layer(
                x=x,
                memory=memory,
                causal_mask=causal_mask,
                self_key_padding_mask=self_key_padding_mask,
            )

        return self.fc_out(x)


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
    Full captioning model wrapper.

    Supported model_type values:
    - "rnn"
    - "gru"
    - "lstm"
    - "lstm_dropout"
    - "transformer_cnn"
    - "transformer_vit"
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        model_type: str | None = None,
        dropout: float = 0.5,
        pad_idx: int = 0,
        transformer_d_model: int | None = None,
        transformer_num_layers: int | None = None,
        transformer_nhead: int = 8,
        transformer_ff_dim: int | None = None,
        transformer_max_len: int = 512,
    ):
        super().__init__()

        self.model_type = (model_type or rnn_type).lower()
        self.is_transformer = self.model_type in {"transformer_cnn", "transformer_vit"}
        transformer_d_model = transformer_d_model or embed_dim
        transformer_num_layers = transformer_num_layers or num_layers

        if self.model_type in {"rnn", "gru", "lstm", "lstm_dropout"}:
            self.encoder = CNNGlobalEncoder(embed_dim=embed_dim)
            self.decoder = RNNDecoder(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                rnn_type=self.model_type,
                dropout=dropout,
                pad_idx=pad_idx,
            )
            self.supports_feature_cache = True
        elif self.model_type == "transformer_cnn":
            self.encoder = CNNEncoder(d_model=transformer_d_model)
            self.decoder = TransformerDecoder(
                vocab_size=vocab_size,
                d_model=transformer_d_model,
                num_layers=transformer_num_layers,
                nhead=transformer_nhead,
                dim_feedforward=transformer_ff_dim or (4 * transformer_d_model),
                dropout=dropout,
                pad_idx=pad_idx,
                max_len=transformer_max_len,
            )
            self.supports_feature_cache = False
        elif self.model_type == "transformer_vit":
            self.encoder = ViTEncoder(d_model=transformer_d_model)
            self.decoder = TransformerDecoder(
                vocab_size=vocab_size,
                d_model=transformer_d_model,
                num_layers=transformer_num_layers,
                nhead=transformer_nhead,
                dim_feedforward=transformer_ff_dim or (4 * transformer_d_model),
                dropout=dropout,
                pad_idx=pad_idx,
                max_len=transformer_max_len,
            )
            self.supports_feature_cache = False
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def project_cached_features(self, cached_features: torch.Tensor) -> torch.Tensor:
        """
        Project cached raw ResNet features (shape [B, 512]) into embedding space.
        Only valid for RNN-family variants.
        """
        if self.is_transformer:
            raise RuntimeError(
                f"Feature cache mode is not supported for {self.model_type}. "
                "Use raw-image mode for Transformer variants."
            )
        return self.encoder.feature_proj(cached_features)

    def forward_features(self, cached_features: torch.Tensor, input_captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass when using cached 512-d CNN features.
        Only valid for RNN-family variants.
        """
        image_emb = self.project_cached_features(cached_features)
        return self.decoder(image_emb, input_captions)

    def forward_images(self, images: torch.Tensor, input_captions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass when using raw images.
        """
        image_features = self.encoder(images)
        if self.is_transformer:
            return self.decoder(image_features, input_captions)
        return self.decoder(image_features, input_captions)

    def forward(self, images: torch.Tensor, input_captions: torch.Tensor) -> torch.Tensor:
        """
        Standard API: outputs = model(images, captions)
        """
        return self.forward_images(images, input_captions)

    @torch.no_grad()
    def greedy_decode(
        self,
        image_embedding: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 25,
    ) -> torch.Tensor:
        """
        Greedy caption decoding.

        Args:
            image_embedding:
                - RNN variants: [1, embed_dim]
                - Transformer variants: [1, N, embed_dim]
        Returns:
            token_ids: [<= max_len]
        """
        self.eval()
        device = image_embedding.device

        if self.is_transformer:
            generated = []
            token_seq = torch.tensor([[start_idx]], dtype=torch.long, device=device)

            for _ in range(max_len):
                logits = self.decoder(image_embedding, token_seq)  # [1, T, V]
                next_token = torch.argmax(logits[:, -1, :], dim=-1)  # [1]

                token_id = int(next_token.item())
                generated.append(token_id)

                if token_id == end_idx:
                    break

                token_seq = torch.cat([token_seq, next_token.unsqueeze(1)], dim=1)

            return torch.tensor(generated, dtype=torch.long, device=device)

        # RNN-family decoding path.
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
