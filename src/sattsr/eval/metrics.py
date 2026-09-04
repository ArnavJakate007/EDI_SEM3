"""Image quality metrics, computed only over pixels valid in both inputs."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity

_EPS = 1e-12
_FSIM_T1 = 0.85
_FSIM_T2 = 160.0


def _pair(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return a, b, np.isfinite(a) & np.isfinite(b)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error in Kelvin^2."""
    a, b, mask = _pair(pred, target)
    if not mask.any():
        return float("nan")
    return float(np.mean((a[mask] - b[mask]) ** 2))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error in Kelvin."""
    value = mse(pred, target)
    return float(np.sqrt(value)) if np.isfinite(value) else value


def psnr(pred: np.ndarray, target: np.ndarray, *, data_range: float) -> float:
    """Peak signal-to-noise ratio in dB; capped at 100 for an exact match."""
    value = mse(pred, target)
    if not np.isfinite(value):
        return float("nan")
    if value <= _EPS:
        return 100.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(value))


_SSIM_SIGMA = 1.5
_SSIM_TRUNCATE = 3.5
# skimage's Gaussian radius, and the border its scalar mssim excludes.
_SSIM_PAD = int(_SSIM_TRUNCATE * _SSIM_SIGMA + 0.5)


def ssim(pred: np.ndarray, target: np.ndarray, *, data_range: float) -> float:
    """Structural similarity, averaged over valid pixels only.

    The filter border is excluded, matching scikit-image's scalar convention: those
    pixels are computed against padding rather than real data.
    """
    a, b, mask = _pair(pred, target)
    if not mask.any():
        return float("nan")
    fill = float(b[mask].mean())
    a_f = np.where(mask, a, fill)
    b_f = np.where(mask, b, fill)
    _, smap = structural_similarity(
        a_f, b_f, data_range=data_range, gaussian_weights=True, sigma=_SSIM_SIGMA,
        truncate=_SSIM_TRUNCATE, use_sample_covariance=False, full=True,
    )

    pad = _SSIM_PAD
    if smap.shape[0] > 2 * pad and smap.shape[1] > 2 * pad:
        interior = np.zeros_like(mask)
        interior[pad:-pad, pad:-pad] = True
        mask = mask & interior
        if not mask.any():
            return float("nan")
    return float(smap[mask].mean())


def _log_gabor_and_riesz(
    shape: tuple[int, int], n_scales: int, min_wavelength: float, mult: float, sigma_onf: float
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    rows, cols = shape
    u1 = (np.arange(cols) - cols // 2) / float(cols)
    u2 = (np.arange(rows) - rows // 2) / float(rows)
    u1, u2 = np.meshgrid(u1, u2)

    radius = np.fft.ifftshift(np.sqrt(u1**2 + u2**2))
    u1 = np.fft.ifftshift(u1)
    u2 = np.fft.ifftshift(u2)
    radius[0, 0] = 1.0

    low_pass = 1.0 / (1.0 + (radius / 0.45) ** 30)     # Butterworth, kills corner artefacts
    filters = []
    for scale in range(n_scales):
        f0 = 1.0 / (min_wavelength * (mult**scale))
        lg = np.exp(-((np.log(radius / f0)) ** 2) / (2.0 * np.log(sigma_onf) ** 2)) * low_pass
        lg[0, 0] = 0.0
        filters.append(lg)

    denom = np.sqrt(u1**2 + u2**2)
    denom[0, 0] = 1.0
    return filters, 1j * u1 / denom, 1j * u2 / denom


def phase_congruency(
    image: np.ndarray,
    *,
    n_scales: int = 4,
    min_wavelength: float = 6.0,
    mult: float = 2.0,
    sigma_onf: float = 0.55,
) -> np.ndarray:
    """Monogenic-signal phase congruency in [0, 1].

    Uses log-Gabor bandpass filters plus the Riesz transform for the odd components.
    Kovesi's noise compensation is deliberately omitted: FSIM only needs a relative
    weighting of structurally significant pixels, and the omission keeps this
    dependency-free and fast.
    """
    img = np.asarray(image, dtype=np.float64)
    spectrum = np.fft.fft2(img)
    filters, hx, hy = _log_gabor_and_riesz(img.shape, n_scales, min_wavelength, mult, sigma_onf)

    sum_even = np.zeros_like(img)
    sum_odd_x = np.zeros_like(img)
    sum_odd_y = np.zeros_like(img)
    sum_amp = np.zeros_like(img)

    for lg in filters:
        band = spectrum * lg
        even = np.real(np.fft.ifft2(band))
        odd_x = np.real(np.fft.ifft2(band * hx))
        odd_y = np.real(np.fft.ifft2(band * hy))
        sum_even += even
        sum_odd_x += odd_x
        sum_odd_y += odd_y
        sum_amp += np.sqrt(even**2 + odd_x**2 + odd_y**2)

    energy = np.sqrt(sum_even**2 + sum_odd_x**2 + sum_odd_y**2)
    return np.clip(energy / (sum_amp + 1e-4), 0.0, 1.0)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = sobel(image, axis=1, mode="nearest")
    gy = sobel(image, axis=0, mode="nearest")
    return np.sqrt(gx**2 + gy**2)


def fsim(pred: np.ndarray, target: np.ndarray, *, data_range: float) -> float:
    """Feature similarity index over valid pixels.

    Both images are mapped onto [0, 255] using the target's minimum and the shared
    `data_range`, so the standard T1/T2 constants apply.
    """
    a, b, mask = _pair(pred, target)
    if not mask.any():
        return float("nan")

    lo = float(b[mask].min())
    fill = float(b[mask].mean())
    scale = 255.0 / max(data_range, _EPS)
    a_s = (np.where(mask, a, fill) - lo) * scale
    b_s = (np.where(mask, b, fill) - lo) * scale

    pc1, pc2 = phase_congruency(a_s), phase_congruency(b_s)
    g1, g2 = _gradient_magnitude(a_s), _gradient_magnitude(b_s)

    s_pc = (2.0 * pc1 * pc2 + _FSIM_T1) / (pc1**2 + pc2**2 + _FSIM_T1)
    s_g = (2.0 * g1 * g2 + _FSIM_T2) / (g1**2 + g2**2 + _FSIM_T2)
    weight = np.maximum(pc1, pc2) * mask

    denominator = float(weight.sum())
    if denominator <= _EPS:
        return float("nan")
    return float(np.clip((s_pc * s_g * weight).sum() / denominator, 0.0, 1.0))


def metric_suite(pred: np.ndarray, target: np.ndarray, *, data_range: float) -> dict[str, float]:
    """Every required image-quality metric in one call."""
    return {
        "mse": mse(pred, target),
        "rmse": rmse(pred, target),
        "psnr": psnr(pred, target, data_range=data_range),
        "ssim": ssim(pred, target, data_range=data_range),
        "fsim": fsim(pred, target, data_range=data_range),
    }
