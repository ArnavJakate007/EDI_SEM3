"""Non-learned interpolation baselines the model must beat."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from sattsr.eval.motion import farneback_flow


def linear_blend(i0: np.ndarray, i2: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Time-weighted average of the two neighbours."""
    a = np.asarray(i0, dtype=np.float32)
    b = np.asarray(i2, dtype=np.float32)
    return ((1.0 - t) * a + t * b).astype(np.float32)


def _remap(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp `image` by `flow` (pixels), filling invalid pixels with the field mean."""
    h, w = image.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    finite = np.isfinite(image)
    filled = np.where(finite, image, float(image[finite].mean()) if finite.any() else 0.0)
    warped = cv2.remap(
        filled.astype(np.float32),
        (grid_x + flow[..., 0]).astype(np.float32),
        (grid_y + flow[..., 1]).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped.astype(np.float32)


def farneback_warp(
    i0: np.ndarray, i2: np.ndarray, t: float = 0.5, *, data_range: float
) -> np.ndarray:
    """Classical optical-flow interpolation: warp both neighbours towards `t` and blend.

    The intermediate flow is approximated by scaling the endpoint flow, which is the
    standard cheap construction and exactly the "assumes smooth motion" behaviour the
    learned model is meant to improve on.
    """
    a = np.asarray(i0, dtype=np.float32)
    b = np.asarray(i2, dtype=np.float32)
    flow_02 = farneback_flow(a, b, data_range=data_range)
    flow_20 = farneback_flow(b, a, data_range=data_range)

    warped0 = _remap(a, -float(t) * flow_02)
    warped2 = _remap(b, -(1.0 - float(t)) * flow_20)
    out = (1.0 - t) * warped0 + t * warped2

    invalid = ~np.isfinite(a) & ~np.isfinite(b)
    return np.where(invalid, np.nan, out).astype(np.float32)


BASELINES: dict[str, Callable[..., np.ndarray]] = {
    "linear": lambda i0, i2, t, data_range: linear_blend(i0, i2, t),
    "farneback": lambda i0, i2, t, data_range: farneback_warp(i0, i2, t, data_range=data_range),
}


def run_baseline(
    name: str, i0: np.ndarray, i2: np.ndarray, t: float = 0.5, *, data_range: float
) -> np.ndarray:
    """Dispatch to a named baseline."""
    try:
        fn = BASELINES[name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}") from exc
    return fn(i0, i2, t, data_range)
