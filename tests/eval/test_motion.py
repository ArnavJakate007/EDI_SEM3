from __future__ import annotations

import numpy as np
import pytest

from sattsr.eval.motion import (
    cold_cloud_scores,
    farneback_flow,
    flow_endpoint_error,
    mean_abs_vorticity,
    motion_displacement_error,
    motion_suite,
    to_uint8,
)

DR = 150.0


def _texture(size: int = 96, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    field = 250.0 + 20.0 * np.sin(xx * 0.4) * np.cos(yy * 0.35)
    return (field + rng.normal(0.0, 0.4, size=(size, size))).astype(np.float32)


def test_to_uint8_spans_the_full_byte_range():
    out = to_uint8(np.array([[180.0, 330.0]], dtype=np.float32), data_range=DR, lo=180.0)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_to_uint8_handles_nan():
    out = to_uint8(np.array([[np.nan, 250.0]], dtype=np.float32), data_range=DR, lo=180.0)
    assert np.isfinite(out).all()


def test_farneback_recovers_a_known_translation():
    a = _texture()
    b = np.roll(a, 4, axis=1)               # content moves 4 px right
    flow = farneback_flow(a, b, data_range=DR)
    assert flow.shape == (96, 96, 2)
    interior = flow[20:76, 20:76, 0]
    assert float(np.median(interior)) == pytest.approx(4.0, abs=1.5)


def test_flow_endpoint_error_is_zero_for_identical_fields():
    f = np.ones((8, 8, 2), dtype=np.float32)
    assert flow_endpoint_error(f, f) == pytest.approx(0.0)
    assert flow_endpoint_error(f, f * 2.0) == pytest.approx(np.sqrt(2.0), rel=1e-5)


def test_motion_displacement_error_rewards_correct_motion():
    i0 = _texture()
    gt_mid = np.roll(i0, 3, axis=1)
    good = np.roll(i0, 3, axis=1)
    blurry = 0.5 * i0 + 0.5 * np.roll(i0, 6, axis=1)   # a linear-blend style guess
    assert motion_displacement_error(i0, good, gt_mid, data_range=DR) < \
        motion_displacement_error(i0, blurry, gt_mid, data_range=DR)


def test_cold_cloud_scores_are_perfect_for_an_exact_match():
    target = np.full((16, 16), 260.0, dtype=np.float32)
    target[:8, :8] = 210.0
    scores = cold_cloud_scores(target.copy(), target, threshold_k=235.0)
    assert scores["csi"] == pytest.approx(1.0)
    assert scores["pod"] == pytest.approx(1.0)
    assert scores["far"] == pytest.approx(0.0)


def test_cold_cloud_scores_penalise_a_missed_cold_core():
    target = np.full((16, 16), 260.0, dtype=np.float32)
    target[:8, :8] = 210.0
    pred = np.full((16, 16), 260.0, dtype=np.float32)     # missed it entirely
    scores = cold_cloud_scores(pred, target, threshold_k=235.0)
    assert scores["pod"] == pytest.approx(0.0)
    assert scores["csi"] == pytest.approx(0.0)


def test_cold_cloud_scores_with_no_cold_pixels_return_nan_not_zero():
    warm = np.full((16, 16), 280.0, dtype=np.float32)
    scores = cold_cloud_scores(warm, warm, threshold_k=235.0)
    assert np.isnan(scores["pod"])


def test_vorticity_is_higher_for_a_rotating_field_than_a_translating_one():
    size = 96
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    base = (250.0 + 20.0 * np.sin(xx * 0.4) * np.cos(yy * 0.35)).astype(np.float32)

    translated = np.roll(base, 4, axis=1)

    cy = cx = size / 2.0
    angle = 0.12
    sy = np.cos(angle) * (yy - cy) - np.sin(angle) * (xx - cx) + cy
    sx = np.sin(angle) * (yy - cy) + np.cos(angle) * (xx - cx) + cx
    rotated = base[np.clip(sy.round().astype(int), 0, size - 1),
                   np.clip(sx.round().astype(int), 0, size - 1)]

    assert mean_abs_vorticity(base, rotated, data_range=DR) > \
        mean_abs_vorticity(base, translated, data_range=DR)


def test_motion_suite_returns_every_key():
    i0 = _texture()
    mid = np.roll(i0, 3, axis=1)
    result = motion_suite(i0, mid, mid, data_range=DR)
    assert set(result) == {"displacement_error", "csi", "pod", "far"}
    assert all(isinstance(v, float) for v in result.values())
