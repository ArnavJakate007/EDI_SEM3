from __future__ import annotations

import numpy as np
import pytest

from sattsr.eval.events import EventCategory, EventThresholds, category_counts, classify_event


def _stack(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    return np.stack([frame_a, frame_b]).astype(np.float32)


def _warm(size: int = 96, value: float = 292.0) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def test_warm_scene_is_clear():
    warm = _warm()
    assert classify_event(_stack(warm, np.roll(warm, 2, axis=1))) == EventCategory.CLEAR


def test_uniform_cool_cloud_deck_is_stratiform():
    deck = _warm()
    deck[:, :] = 255.0                       # cloudy but nowhere near deep convection
    assert classify_event(_stack(deck, np.roll(deck, 2, axis=1))) == EventCategory.STRATIFORM


def test_small_very_cold_core_is_convective():
    scene = _warm()
    scene[:, :] = 260.0
    scene[40:52, 40:52] = 205.0              # ~1.5% of the scene below 220 K
    assert classify_event(_stack(scene, np.roll(scene, 2, axis=1))) == EventCategory.CONVECTIVE


def test_large_rotating_cold_shield_is_cyclonic():
    size = 128
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    radius = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    scene = np.where(radius < size * 0.38, 200.0, 265.0).astype(np.float32)
    scene += (2.0 * np.sin(xx * 0.5) * np.cos(yy * 0.5)).astype(np.float32)  # texture to track

    angle = 0.25
    cy = cx = size / 2.0
    sy = np.cos(angle) * (yy - cy) - np.sin(angle) * (xx - cx) + cy
    sx = np.sin(angle) * (yy - cy) + np.cos(angle) * (xx - cx) + cx
    rotated = scene[np.clip(sy.round().astype(int), 0, size - 1),
                    np.clip(sx.round().astype(int), 0, size - 1)]

    assert classify_event(_stack(scene, rotated)) == EventCategory.CYCLONIC


def test_all_nan_scene_is_clear_not_a_crash():
    nan_frame = np.full((32, 32), np.nan, dtype=np.float32)
    assert classify_event(_stack(nan_frame, nan_frame)) == EventCategory.CLEAR


def test_thresholds_are_configurable():
    scene = _warm()
    scene[:, :] = 255.0
    strict = EventThresholds(cloud_bt_k=250.0)     # 255 K no longer counts as cloud
    assert classify_event(_stack(scene, scene), thresholds=strict) == EventCategory.CLEAR


def test_single_frame_input_is_rejected():
    with pytest.raises(ValueError):
        classify_event(_warm()[None])


def test_category_values_match_the_spec_spelling():
    assert [c.value for c in EventCategory] == ["clear", "stratiform", "convective", "cyclonic"]


def test_category_counts_zero_fills_every_category():
    counts = category_counts([EventCategory.CLEAR, EventCategory.CLEAR,
                              EventCategory.CONVECTIVE])
    assert counts == {"clear": 2, "stratiform": 0, "convective": 1, "cyclonic": 0}
