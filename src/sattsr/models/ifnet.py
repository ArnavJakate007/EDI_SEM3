"""RIFE-style coarse-to-fine intermediate-flow and fusion network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sattsr.config import ModelConfig
from sattsr.models.blocks import ResBlock, backwarp, conv_block


class IFBlock(nn.Module):
    """One coarse-to-fine refinement stage.

    Consumes the current warped pair plus the current flow and mask, and predicts
    increments to both. `scale` selects the resolution the stage reasons at: large
    scale sees big displacements, small scale sharpens detail.
    """

    def __init__(self, in_ch: int, channels: int) -> None:
        super().__init__()
        half = max(channels // 2, 8)
        self.stem = nn.Sequential(
            conv_block(in_ch, half, 3, 2, 1),
            conv_block(half, channels, 3, 2, 1),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(4)])
        self.head = nn.ConvTranspose2d(channels, 5, 4, 2, 1)

    def forward(self, x: Tensor, flow: Tensor, scale: int = 1) -> tuple[Tensor, Tensor]:
        """Return (flow_delta, mask_delta) at the resolution of `x`."""
        if scale != 1:
            inv = 1.0 / float(scale)
            x = F.interpolate(x, scale_factor=inv, mode="bilinear", align_corners=False)
            flow = (
                F.interpolate(flow, scale_factor=inv, mode="bilinear", align_corners=False)
                / float(scale)
            )
        out = self.head(self.body(self.stem(torch.cat([x, flow], dim=1))))
        out = F.interpolate(
            out, scale_factor=float(scale) * 2.0, mode="bilinear", align_corners=False
        )
        return out[:, :4] * (float(scale) * 2.0), out[:, 4:5]


class IFNet(nn.Module):
    """Stacks IFBlocks from coarse to fine and fuses the two warped views."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.scales = list(cfg.scales)
        if not self.scales:
            raise ValueError("ModelConfig.scales must not be empty")
        # 4 image/context channels (warped0, warped2, t_map, mask) + 4 flow channels
        self.blocks = nn.ModuleList(
            IFBlock(in_ch=8, channels=cfg.base_channels) for _ in self.scales
        )

    @property
    def size_multiple(self) -> int:
        """Input height and width must both be divisible by this."""
        return 4 * max(self.scales)

    def forward(
        self,
        i0: Tensor,
        i2: Tensor,
        t: Tensor,
        flow_init: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict intermediate flows, a fusion mask, and the blended frame."""
        n, _, h, w = i0.shape
        m = self.size_multiple
        if h % m or w % m:
            raise ValueError(f"input {h}x{w} must be divisible by {m}")

        t_map = t.reshape(-1, 1, 1, 1).to(i0.dtype).expand(n, 1, h, w)
        flow = (
            torch.zeros(n, 4, h, w, device=i0.device, dtype=i0.dtype)
            if flow_init is None
            else flow_init.to(i0.dtype)
        )
        mask = torch.zeros(n, 1, h, w, device=i0.device, dtype=i0.dtype)
        warped0 = backwarp(i0, flow[:, :2])
        warped2 = backwarp(i2, flow[:, 2:4])

        for block, scale in zip(self.blocks, self.scales, strict=True):
            x = torch.cat([warped0, warped2, t_map, mask], dim=1)
            d_flow, d_mask = block(x, flow, scale=scale)
            flow = flow + d_flow
            mask = mask + d_mask
            warped0 = backwarp(i0, flow[:, :2])
            warped2 = backwarp(i2, flow[:, 2:4])

        sigma = torch.sigmoid(mask)
        return {
            "flow": flow,
            "mask": sigma,
            "warped0": warped0,
            "warped2": warped2,
            "merged": warped0 * sigma + warped2 * (1.0 - sigma),
        }
