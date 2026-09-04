"""The weighted training objective."""

from __future__ import annotations

from torch import Tensor, nn

from sattsr.config import LossConfig
from sattsr.losses.functional import (
    charbonnier,
    flow_smoothness,
    radiometric_consistency,
    warp_consistency,
)
from sattsr.losses.ssim import SSIMLoss


class CompositeLoss(nn.Module):
    """Reconstruction + structure + flow smoothness + temporal + radiometric."""

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ssim_loss = SSIMLoss(data_range=1.0)

    def forward(
        self, out: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        """Return the scalar total and a float breakdown for logging."""
        pred = out["pred"]
        target = batch["i1"]
        mask = batch.get("valid")
        cfg = self.cfg

        recon = charbonnier(pred, target, mask=mask)
        ssim = self.ssim_loss(pred, target)
        smooth = flow_smoothness(out["flow"], target)
        consistency = warp_consistency(out["warped0"], out["warped2"], mask=mask)
        radiometric = radiometric_consistency(
            pred, batch["i0"], batch["i2"], batch["t"], mask=mask
        )

        total = (
            cfg.w_recon * recon
            + cfg.w_ssim * ssim
            + cfg.w_smooth * smooth
            + cfg.w_consistency * consistency
            + cfg.w_radiometric * radiometric
        )

        parts = {
            "recon": float(cfg.w_recon * recon.detach()),
            "ssim": float(cfg.w_ssim * ssim.detach()),
            "smooth": float(cfg.w_smooth * smooth.detach()),
            "consistency": float(cfg.w_consistency * consistency.detach()),
            "radiometric": float(cfg.w_radiometric * radiometric.detach()),
        }
        parts["total"] = float(total.detach())
        return total, parts
