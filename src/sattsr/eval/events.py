"""Heuristic stratification of scenes into meteorological event categories.

This is a screening heuristic on brightness temperature and flow vorticity, not a
validated meteorological classifier. Its purpose is to keep convective and cyclonic
performance from being averaged away in the report.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from sattsr.eval.motion import mean_abs_vorticity


class EventCategory(str, Enum):
    """The four categories the report is broken down by."""

    CLEAR = "clear"
    STRATIFORM = "stratiform"
    CONVECTIVE = "convective"
    CYCLONIC = "cyclonic"


@dataclass(frozen=True)
class EventThresholds:
    """Tunable decision boundaries. Defaults follow common IR nowcasting practice.

    `vorticity` was calibrated against the fixtures in tests/eval: a pure translation
    scores around 0.01-0.02 and a 0.25 rad rotation around 0.1, so 0.05 separates them
    with margin on both sides.
    """

    cloud_bt_k: float = 270.0             # below this a pixel is cloudy
    cloud_fraction: float = 0.15          # cloudier than this and the scene is not clear
    deep_bt_k: float = 220.0              # below this is deep convective cloud top
    # 1% of a scene with tops below 220 K is already an active convective cell; a
    # single storm easily covers that much of a regional tile.
    deep_fraction: float = 0.01           # this much deep cloud makes a scene convective
    cyclonic_deep_fraction: float = 0.15  # a large cold shield, plus rotation, is cyclonic
    vorticity: float = 0.05               # mean |curl| of the inter-frame flow, px/px


def classify_event(
    frames: np.ndarray,
    *,
    data_range: float = 150.0,
    thresholds: EventThresholds = EventThresholds(),
) -> EventCategory:
    """Classify a short sequence of Kelvin frames, shape (T, H, W) with T >= 2."""
    stack = np.asarray(frames, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError(f"expected (T, H, W) with T >= 2, got shape {stack.shape}")

    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return EventCategory.CLEAR

    cloud_fraction = float(np.mean(finite < thresholds.cloud_bt_k))
    deep_fraction = float(np.mean(finite < thresholds.deep_bt_k))

    if cloud_fraction < thresholds.cloud_fraction:
        return EventCategory.CLEAR
    if deep_fraction < thresholds.deep_fraction:
        return EventCategory.STRATIFORM
    if deep_fraction >= thresholds.cyclonic_deep_fraction:
        vorticity = mean_abs_vorticity(stack[0], stack[-1], data_range=data_range)
        if vorticity >= thresholds.vorticity:
            return EventCategory.CYCLONIC
    return EventCategory.CONVECTIVE


def category_counts(categories: Iterable[EventCategory]) -> dict[str, int]:
    """Count each category, always reporting all four keys."""
    counts = {c.value: 0 for c in EventCategory}
    for category in categories:
        counts[EventCategory(category).value] += 1
    return counts
