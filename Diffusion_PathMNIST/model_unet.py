import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freq[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(get_num_groups(in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(get_num_groups(out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 4),
    ):
        super().__init__()
        time_dim = base_channels * 4

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = [base_channels * m for m in channel_mults]

        self.in_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        self.skip_channels: List[int] = []

        for i, ch in enumerate(channels):
            block = nn.ModuleDict(
                {
                    "res": ResBlock(in_ch, ch, time_dim),
                    "down": nn.Identity()
                    if i == len(channels) - 1
                    else nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1),
                }
            )
            self.down_blocks.append(block)
            self.skip_channels.append(ch)
            in_ch = ch

        self.mid1 = ResBlock(in_ch, in_ch, time_dim)
        self.mid2 = ResBlock(in_ch, in_ch, time_dim)

        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        rev_skips = list(reversed(self.skip_channels))

        up_in = in_ch
        for i, (ch, skip_ch) in enumerate(zip(rev_channels, rev_skips)):
            out_ch = ch
            block = nn.ModuleDict(
                {
                    "res": ResBlock(up_in + skip_ch, out_ch, time_dim),
                    "up": nn.Identity()
                    if i == len(rev_channels) - 1
                    else nn.ConvTranspose2d(
                        out_ch,
                        rev_channels[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                }
            )
            self.up_blocks.append(block)
            up_in = rev_channels[i + 1] if i < len(rev_channels) - 1 else out_ch

        self.out_norm = nn.GroupNorm(get_num_groups(channels[0]), channels[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        h = self.in_conv(x)
        skips = []

        for block in self.down_blocks:
            h = block["res"](h, t_emb)
            skips.append(h)
            h = block["down"](h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for block in self.up_blocks:
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = block["res"](h, t_emb)
            h = block["up"](h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h
