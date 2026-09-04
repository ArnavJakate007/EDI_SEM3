"""Shared network building blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def conv_block(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    """Conv2d followed by a channel-wise PReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, dilation=dilation, bias=True),
        nn.PReLU(out_ch),
    )


class ResBlock(nn.Module):
    """Residual block that preserves shape."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = conv_block(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.PReLU(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


def backwarp(x: Tensor, flow: Tensor) -> Tensor:
    """Backward-warp `x` by `flow` (pixels; channel 0 = right, channel 1 = down).

    Output pixel (y, x0) reads from input pixel (y + flow_y, x0 + flow_x), bilinearly,
    clamping to the border outside the frame.
    """
    n, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    src_x = grid_x[None] + flow[:, 0]
    src_y = grid_y[None] + flow[:, 1]

    norm_x = 2.0 * src_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(h - 1, 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)          # (N, H, W, 2)

    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
