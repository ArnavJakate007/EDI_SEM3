"""Individual loss terms."""

from __future__ import annotations

import torch
from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    if values.shape[1] != mask.shape[1]:
        mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def charbonnier(
    pred: Tensor, target: Tensor, *, mask: Tensor | None = None, eps: float = 1e-3
) -> Tensor:
    """Robust L1 reconstruction loss, optionally restricted to valid pixels."""
    return _masked_mean(torch.sqrt((pred - target) ** 2 + eps * eps), mask)


def flow_smoothness(flow: Tensor, image: Tensor) -> Tensor:
    """Edge-aware first-order smoothness: penalise flow gradients away from image edges."""
    d_x = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()
    d_y = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()
    g_x = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    g_y = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    return (d_x * torch.exp(-g_x)).mean() + (d_y * torch.exp(-g_y)).mean()


def warp_consistency(warped0: Tensor, warped2: Tensor, *, mask: Tensor | None = None) -> Tensor:
    """Temporal consistency: the forward and backward warped views must agree."""
    return _masked_mean((warped0 - warped2).abs(), mask)


def radiometric_consistency(
    pred: Tensor, i0: Tensor, i2: Tensor, t: Tensor, *, mask: Tensor | None = None
) -> Tensor:
    """Keep the synthesized frame's mean brightness on the line between its neighbours."""
    if mask is None:
        mask = torch.ones_like(pred)
    weight = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mean_pred = (pred * mask).sum(dim=(1, 2, 3)) / weight
    mean_0 = (i0 * mask).sum(dim=(1, 2, 3)) / weight
    mean_2 = (i2 * mask).sum(dim=(1, 2, 3)) / weight
    tt = t.reshape(-1).to(pred.dtype)
    return ((mean_pred - ((1.0 - tt) * mean_0 + tt * mean_2)).abs()).mean()
