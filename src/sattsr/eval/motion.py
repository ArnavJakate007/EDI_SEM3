"""Metrics that score cloud motion rather than per-pixel similarity."""

from __future__ import annotations

import cv2
import numpy as np

_EPS = 1e-12
_FARNEBACK = {
    "pyr_scale": 0.5,
    "levels": 4,
    "winsize": 21,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


def to_uint8(image: np.ndarray, *, data_range: float, lo: float | None = None) -> np.ndarray:
    """Map Kelvin onto bytes for OpenCV; NaN becomes the field mean."""
    a = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, dtype=np.uint8)
    base = float(a[finite].min()) if lo is None else float(lo)
    filled = np.where(finite, a, float(a[finite].mean()))
    scaled = (filled - base) / max(data_range, _EPS) * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def farneback_flow(a: np.ndarray, b: np.ndarray, *, data_range: float) -> np.ndarray:
    """Dense classical optical flow from `a` to `b`, in pixels, shape (H, W, 2)."""
    lo = float(np.nanmin([np.nanmin(a), np.nanmin(b)]))
    return cv2.calcOpticalFlowFarneback(
        to_uint8(a, data_range=data_range, lo=lo),
        to_uint8(b, data_range=data_range, lo=lo),
        None,
        **_FARNEBACK,
    ).astype(np.float32)


def flow_endpoint_error(f1: np.ndarray, f2: np.ndarray) -> float:
    """Mean Euclidean distance between two flow fields."""
    return float(np.mean(np.linalg.norm(np.asarray(f1) - np.asarray(f2), axis=-1)))


def motion_displacement_error(
    i0: np.ndarray, pred_mid: np.ndarray, gt_mid: np.ndarray, *, data_range: float
) -> float:
    """How far the predicted cloud field moved compared with how far it really moved.

    A blurred prediction scores badly here even when its MSE looks respectable, which
    is exactly the failure mode the spec calls out.
    """
    return flow_endpoint_error(
        farneback_flow(i0, pred_mid, data_range=data_range),
        farneback_flow(i0, gt_mid, data_range=data_range),
    )


def cold_cloud_scores(
    pred: np.ndarray, target: np.ndarray, *, threshold_k: float = 235.0
) -> dict[str, float]:
    """Categorical skill for deep convection, thresholded on brightness temperature.

    Returns NaN rather than 0 when the category is absent from both fields, so empty
    scenes do not drag the reported average down.
    """
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    p = (a < threshold_k) & valid
    t = (b < threshold_k) & valid

    hits = float(np.count_nonzero(p & t))
    misses = float(np.count_nonzero(~p & t))
    false_alarms = float(np.count_nonzero(p & ~t))

    union = hits + misses + false_alarms
    observed = hits + misses
    predicted = hits + false_alarms
    return {
        "csi": hits / union if union > 0 else float("nan"),
        "pod": hits / observed if observed > 0 else float("nan"),
        "far": false_alarms / predicted if predicted > 0 else float("nan"),
    }


def mean_abs_vorticity(a: np.ndarray, b: np.ndarray, *, data_range: float) -> float:
    """Mean |dv/dx - du/dy| of the flow between two frames: how rotational the motion is."""
    flow = farneback_flow(a, b, data_range=data_range)
    du_dy = np.gradient(flow[..., 0], axis=0)
    dv_dx = np.gradient(flow[..., 1], axis=1)
    return float(np.mean(np.abs(dv_dx - du_dy)))


def motion_suite(
    i0: np.ndarray,
    pred_mid: np.ndarray,
    gt_mid: np.ndarray,
    *,
    data_range: float,
    threshold_k: float = 235.0,
) -> dict[str, float]:
    """Every cloud-motion metric in one call."""
    result = {
        "displacement_error": motion_displacement_error(
            i0, pred_mid, gt_mid, data_range=data_range
        )
    }
    result.update(cold_cloud_scores(pred_mid, gt_mid, threshold_k=threshold_k))
    return result
