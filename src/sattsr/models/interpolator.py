"""The end-to-end frame interpolator: flow initialisation, fusion, refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sattsr.config import ModelConfig
from sattsr.models.blocks import ResBlock, conv_block
from sattsr.models.flow import RaftLiteFlow
from sattsr.models.ifnet import IFNet

_RESIDUAL_LIMIT = 0.5


class RefineNet(nn.Module):
    """Small U-Net that predicts a bounded correction to the fused frame."""

    def __init__(self, in_ch: int, channels: int = 32) -> None:
        super().__init__()
        self.down1 = conv_block(in_ch, channels, 3, 2, 1)
        self.down2 = conv_block(channels, channels * 2, 3, 2, 1)
        self.mid = ResBlock(channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)
        self.act1 = nn.PReLU(channels)
        self.up2 = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)
        self.act2 = nn.PReLU(channels)
        self.out = nn.Conv2d(channels, 1, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        d1 = self.down1(x)
        m = self.mid(self.down2(d1))
        u1 = self.act1(self.up1(m))
        u2 = self.act2(self.up2(torch.cat([u1, d1], dim=1)))
        return torch.tanh(self.out(u2)) * _RESIDUAL_LIMIT


class FrameInterpolator(nn.Module):
    """RAFT-lite flow initialisation -> RIFE-style fusion -> residual refinement."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.flow_net = RaftLiteFlow.from_config(cfg) if cfg.use_raft_init else None
        self.ifnet = IFNet(cfg)
        # i0, i2, warped0, warped2, merged, mask
        self.refine = RefineNet(in_ch=6, channels=max(cfg.base_channels // 2, 8))

    @property
    def size_multiple(self) -> int:
        """Input height and width must both be divisible by this."""
        return max(self.ifnet.size_multiple, 8)

    def _flow_init(self, i0: Tensor, i2: Tensor, t: Tensor) -> Tensor | None:
        if self.flow_net is None:
            return None
        flow_02, flow_20 = self.flow_net(i0, i2)
        tt = t.reshape(-1, 1, 1, 1).to(i0.dtype)
        # linear-motion identity, then lift from 1/8 resolution back to full pixels
        f_t0 = -tt * flow_02
        f_t2 = -(1.0 - tt) * flow_20
        coarse = torch.cat([f_t0, f_t2], dim=1)
        return (
            F.interpolate(coarse, size=i0.shape[-2:], mode="bilinear", align_corners=False) * 8.0
        )

    def forward(self, i0: Tensor, i2: Tensor, t: Tensor | float = 0.5) -> dict[str, Tensor]:
        """Synthesize the frame at normalised time `t` between `i0` and `i2`."""
        if not torch.is_tensor(t):
            t = torch.full((i0.shape[0],), float(t), device=i0.device, dtype=i0.dtype)
        t = t.to(device=i0.device, dtype=i0.dtype).reshape(-1)

        out = self.ifnet(i0, i2, t, flow_init=self._flow_init(i0, i2, t))
        residual = self.refine(
            torch.cat([i0, i2, out["warped0"], out["warped2"], out["merged"], out["mask"]], dim=1)
        )
        out["residual"] = residual
        out["pred"] = torch.clamp(out["merged"] + residual, 0.0, 1.0)
        return out

    @torch.no_grad()
    def interpolate(self, i0: Tensor, i2: Tensor, t: Tensor | float = 0.5) -> Tensor:
        """Inference shortcut returning just the predicted frame."""
        return self.forward(i0, i2, t)["pred"]


def build_model(cfg: ModelConfig) -> FrameInterpolator:
    """Construct the interpolator described by `cfg`."""
    return FrameInterpolator(cfg)
