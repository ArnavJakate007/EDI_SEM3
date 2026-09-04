"""RAFT-lite: an iteratively refined local-correlation optical-flow estimator."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sattsr.config import ModelConfig
from sattsr.models.blocks import ResBlock, backwarp, conv_block


class FeatureEncoder(nn.Module):
    """Single-channel image -> dense features at 1/8 resolution."""

    def __init__(self, out_ch: int) -> None:
        super().__init__()
        c1, c2 = max(out_ch // 4, 4), max(out_ch // 2, 8)
        self.net = nn.Sequential(
            conv_block(1, c1, 3, 2, 1),
            conv_block(c1, c2, 3, 2, 1),
            conv_block(c2, out_ch, 3, 2, 1),
            ResBlock(out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def local_correlation(f1: Tensor, f2: Tensor, radius: int) -> Tensor:
    """Correlate each `f1` location against a (2r+1)^2 window of `f2`.

    Returns (N, (2r+1)^2, H, W), scaled by 1/sqrt(C) so the magnitude does not grow
    with feature width.
    """
    n, c, h, w = f1.shape
    k = 2 * radius + 1
    patches = F.unfold(f2, kernel_size=k, padding=radius).view(n, c, k * k, h, w)
    return (f1.unsqueeze(2) * patches).sum(dim=1) / (c**0.5)


class FlowRefiner(nn.Module):
    """Maps (correlation, features, current flow) to a flow increment."""

    def __init__(self, in_ch: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_ch, hidden),
            conv_block(hidden, hidden),
            nn.Conv2d(hidden, 2, 3, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RaftLiteFlow(nn.Module):
    """Bidirectional optical flow at 1/8 resolution, refined over `iters` steps."""

    def __init__(self, channels: int = 96, radius: int = 3, iters: int = 4) -> None:
        super().__init__()
        self.radius = int(radius)
        self.iters = int(iters)
        self.encoder = FeatureEncoder(channels)
        corr_ch = (2 * self.radius + 1) ** 2
        self.refiner = FlowRefiner(corr_ch + channels + 2, hidden=max(channels, 64))

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> RaftLiteFlow:
        """Build from a validated ModelConfig."""
        return cls(channels=cfg.flow_channels, radius=cfg.flow_radius, iters=cfg.flow_iters)

    def _estimate(self, fa: Tensor, fb: Tensor) -> Tensor:
        n, _, h, w = fa.shape
        flow = torch.zeros(n, 2, h, w, device=fa.device, dtype=fa.dtype)
        for _ in range(self.iters):
            corr = local_correlation(fa, backwarp(fb, flow), self.radius)
            flow = flow + self.refiner(torch.cat([corr, fa, flow], dim=1))
        return flow

    def forward(self, i0: Tensor, i2: Tensor) -> tuple[Tensor, Tensor]:
        """Return (flow_02, flow_20) at 1/8 resolution, in 1/8-resolution pixels."""
        f0 = self.encoder(i0)
        f2 = self.encoder(i2)
        return self._estimate(f0, f2), self._estimate(f2, f0)
