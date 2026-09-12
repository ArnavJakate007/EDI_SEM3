"""Differentiable structural similarity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gaussian_window(size: int, sigma: float) -> Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return torch.outer(g, g)[None, None]


class SSIM(nn.Module):
    """Mean SSIM over a batch, using Gaussian weighting (no sample-covariance bias)."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.register_buffer("window", _gaussian_window(window_size, sigma), persistent=False)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Scalar mean SSIM between two (N, 1, H, W) tensors."""
        window = self.window.to(dtype=a.dtype, device=a.device)
        pad = self.window_size // 2

        mu_a = F.conv2d(a, window, padding=pad)
        mu_b = F.conv2d(b, window, padding=pad)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

        sigma_a2 = F.conv2d(a * a, window, padding=pad) - mu_a2
        sigma_b2 = F.conv2d(b * b, window, padding=pad) - mu_b2
        sigma_ab = F.conv2d(a * b, window, padding=pad) - mu_ab

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
        den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
        return (num / den).mean()


class SSIMLoss(nn.Module):
    """1 - SSIM, so lower is better."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0) -> None:
        super().__init__()
        self.ssim = SSIM(window_size, sigma, data_range)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return 1.0 - self.ssim(a, b)


#: Standard MS-SSIM scale weights (Wang et al. 2003). Sum to 1, so the multi-scale
#: score stays comparable in magnitude to the single-scale one.
MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


class MSSSIM(nn.Module):
    """Multi-scale SSIM: SSIM evaluated on a Gaussian-ish pyramid and weighted.

    Thermal cloud structure lives at several scales at once -- a mesoscale convective
    system's anvil edge and its internal texture are different spatial frequencies --
    so a single 11-pixel window scores only one of them. This is the perceptual term
    used in place of VGG features; see `losses/composite.py`.

    Falls back to fewer scales when the input is too small to halve `levels` times,
    rather than erroring, so 64x64 test tiles and 256x256 training tiles both work.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        weights: tuple[float, ...] = MS_SSIM_WEIGHTS,
    ) -> None:
        super().__init__()
        self.ssim = SSIM(window_size, sigma, data_range)
        self.window_size = window_size
        self.register_buffer(
            "weights", torch.tensor(weights, dtype=torch.float32), persistent=False
        )

    def usable_levels(self, height: int, width: int) -> int:
        """How many pyramid levels this input size supports."""
        smallest = min(height, width)
        levels = 0
        while levels < len(self.weights) and smallest >= self.window_size:
            levels += 1
            smallest //= 2
        return max(1, levels)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Scalar mean MS-SSIM between two (N, 1, H, W) tensors."""
        levels = self.usable_levels(a.shape[-2], a.shape[-1])
        weights = self.weights[:levels].to(device=a.device, dtype=a.dtype)
        weights = weights / weights.sum()

        scores = []
        for level in range(levels):
            if level > 0:
                a = F.avg_pool2d(a, kernel_size=2)
                b = F.avg_pool2d(b, kernel_size=2)
            scores.append(self.ssim(a, b))
        return torch.stack(scores).mul(weights).sum()


class MSSSIMLoss(nn.Module):
    """1 - MS-SSIM, so lower is better."""

    def __init__(
        self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0
    ) -> None:
        super().__init__()
        self.ms_ssim = MSSSIM(window_size, sigma, data_range)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return 1.0 - self.ms_ssim(a, b)
