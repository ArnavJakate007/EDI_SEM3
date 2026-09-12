from __future__ import annotations

import numpy as np
import pytest

from sattsr.config import NormalizationConfig
from sattsr.data.radiometry import (
    SensorStats,
    apply_lut,
    denormalize_bt,
    match_statistics,
    normalize_bt,
    planck_radiance_to_bt,
    sensor_stats,
    valid_mask,
)

# Representative GOES ABI band 13 Planck coefficients.
FK1, FK2, BC1, BC2 = 1.03413e4, 1.39177e3, 0.20351, 0.99958


def test_planck_inversion_lands_in_a_physical_range():
    bt = planck_radiance_to_bt(
        np.array([[10.0, 60.0, 120.0]], dtype=np.float32), FK1, FK2, BC1, BC2
    )
    assert bt.dtype == np.float32
    assert np.all(bt > 150.0) and np.all(bt < 350.0)
    assert np.all(np.diff(bt, axis=1) > 0)      # radiance and BT increase together


def test_planck_inversion_rejects_non_positive_radiance():
    bt = planck_radiance_to_bt(np.array([0.0, -5.0], dtype=np.float32), FK1, FK2, BC1, BC2)
    assert np.all(np.isnan(bt))


def test_apply_lut_maps_counts_to_temperature():
    lut = np.linspace(150.0, 350.0, 1024).astype(np.float32)
    bt = apply_lut(np.array([[0, 512, 1023]], dtype=np.uint16), lut)
    np.testing.assert_allclose(bt, [[lut[0], lut[512], lut[1023]]], rtol=1e-6)


def test_apply_lut_marks_out_of_range_counts_nan():
    lut = np.linspace(150.0, 350.0, 16).astype(np.float32)
    bt = apply_lut(np.array([-1, 3, 99]), lut)
    assert np.isnan(bt[0]) and np.isfinite(bt[1]) and np.isnan(bt[2])


def test_normalize_and_denormalize_round_trip():
    cfg = NormalizationConfig(bt_min=180.0, bt_max=330.0)
    bt = np.array([[180.0, 255.0, 330.0]], dtype=np.float32)
    x = normalize_bt(bt, cfg)
    np.testing.assert_allclose(x, [[0.0, 0.5, 1.0]], atol=1e-6)
    np.testing.assert_allclose(denormalize_bt(x, cfg), bt, atol=1e-3)


def test_normalize_clips_and_zero_fills_nan():
    cfg = NormalizationConfig(bt_min=200.0, bt_max=300.0)
    x = normalize_bt(np.array([150.0, np.nan, 400.0], dtype=np.float32), cfg)
    assert x[0] == pytest.approx(0.0)
    assert x[1] == pytest.approx(0.0)
    assert x[2] == pytest.approx(1.0)
    assert not np.isnan(x).any()


def test_valid_mask_flags_only_finite_pixels():
    bt = np.array([[250.0, np.nan], [np.inf, 300.0]], dtype=np.float32)
    np.testing.assert_array_equal(valid_mask(bt), [[True, False], [False, True]])


def test_match_statistics_shifts_mean_and_scale():
    rng = np.random.default_rng(0)
    bt = rng.normal(240.0, 12.0, size=(64, 64)).astype(np.float32)
    out = match_statistics(bt, sensor_stats(bt), SensorStats(mean=260.0, std=6.0))
    assert float(np.nanmean(out)) == pytest.approx(260.0, abs=0.5)
    assert float(np.nanstd(out)) == pytest.approx(6.0, abs=0.5)


def test_match_statistics_preserves_nan_positions():
    bt = np.array([[250.0, np.nan]], dtype=np.float32)
    out = match_statistics(bt, SensorStats(250.0, 1.0), SensorStats(260.0, 2.0))
    assert np.isnan(out[0, 1])
