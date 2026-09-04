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
