"""Radiometric conversion and normalisation of thermal-IR observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sattsr.config import NormalizationConfig


def planck_radiance_to_bt(
    rad: np.ndarray, fk1: float, fk2: float, bc1: float, bc2: float
) -> np.ndarray:
    """Invert the Planck function: ABI L1b spectral radiance -> brightness temperature (K)."""
    r = np.asarray(rad, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        bt = (fk2 / np.log(fk1 / r + 1.0) - bc1) / bc2
    bt = np.where(r > 0, bt, np.nan)
    return np.where(np.isfinite(bt), bt, np.nan).astype(np.float32)


def apply_lut(counts: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map integer sensor counts through a count -> temperature table (INSAT L1B/L1C)."""
    # Cast first: NumPy 2 refuses to put the -1 sentinel into an unsigned array.
    c = np.asarray(counts, dtype=np.float64)
    table = np.asarray(lut, dtype=np.float32).ravel()
    idx = np.rint(np.where(np.isfinite(c), c, -1.0)).astype(np.int64)
    inside = (idx >= 0) & (idx < table.size)
    out = np.full(idx.shape, np.nan, dtype=np.float32)
    out[inside] = table[idx[inside]]
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def valid_mask(bt: np.ndarray) -> np.ndarray:
    """True where the brightness temperature is a usable finite value."""
    return np.isfinite(np.asarray(bt))


def normalize_bt(bt: np.ndarray, cfg: NormalizationConfig) -> np.ndarray:
    """Map Kelvin onto [0, 1]. Out-of-range values are clipped; NaN becomes 0.0."""
    a = np.asarray(bt, dtype=np.float32)
    x = (a - np.float32(cfg.bt_min)) / np.float32(cfg.data_range)
    x = np.clip(x, 0.0, 1.0)
    return np.where(np.isfinite(x), x, np.float32(0.0)).astype(np.float32)


def denormalize_bt(x: np.ndarray, cfg: NormalizationConfig) -> np.ndarray:
    """Map [0, 1] back onto Kelvin."""
    a = np.asarray(x, dtype=np.float32)
    return (a * np.float32(cfg.data_range) + np.float32(cfg.bt_min)).astype(np.float32)


@dataclass(frozen=True)
class SensorStats:
    """Brightness-temperature mean and standard deviation for one sensor."""

    mean: float
    std: float


def sensor_stats(bt: np.ndarray) -> SensorStats:
    """Compute BT statistics over the valid pixels of an array or stack."""
    a = np.asarray(bt, dtype=np.float64)
    return SensorStats(mean=float(np.nanmean(a)), std=float(np.nanstd(a)))


def match_statistics(bt: np.ndarray, src: SensorStats, dst: SensorStats) -> np.ndarray:
    """Renormalise BT from one sensor's radiometric statistics to another's.

    This is the cross-sensor adaptation step: it removes the systematic mean/scale
    offset between, say, GOES ABI C13 and INSAT TIR1 before fine-tuning.
    """
    a = np.asarray(bt, dtype=np.float32)
    scale = np.float32(dst.std / src.std) if src.std > 1e-6 else np.float32(1.0)
    return ((a - np.float32(src.mean)) * scale + np.float32(dst.mean)).astype(np.float32)
