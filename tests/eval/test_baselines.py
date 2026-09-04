from __future__ import annotations

import numpy as np
import pytest

from sattsr.eval.baselines import BASELINES, farneback_warp, linear_blend, run_baseline
from sattsr.eval.metrics import psnr

DR = 150.0


def _texture(size: int = 96, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    field = 250.0 + 20.0 * np.sin(xx * 0.4) * np.cos(yy * 0.35)
    return (field + rng.normal(0.0, 0.4, size=(size, size))).astype(np.float32)


def test_linear_blend_at_the_midpoint_is_the_average():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.full((4, 4), 10.0, dtype=np.float32)
    np.testing.assert_allclose(linear_blend(a, b, 0.5), 5.0)
    np.testing.assert_allclose(linear_blend(a, b, 0.25), 2.5)


def test_linear_blend_preserves_dtype_and_nan():
    a = np.array([[np.nan, 1.0]], dtype=np.float32)
    b = np.array([[np.nan, 3.0]], dtype=np.float32)
    out = linear_blend(a, b)
    assert out.dtype == np.float32
    assert np.isnan(out[0, 0]) and out[0, 1] == pytest.approx(2.0)


def test_farneback_warp_beats_a_linear_blend_on_pure_translation():
    i0 = _texture()
    # The test texture has a ~16 px period, so keep the displacement well under half
    # of that: a larger shift is genuinely ambiguous and any flow estimator will
    # legitimately lock onto the wrapped-around solution.
    i2 = np.roll(i0, 4, axis=1)
    truth = np.roll(i0, 2, axis=1)
    warped = farneback_warp(i0, i2, 0.5, data_range=DR)
    blended = linear_blend(i0, i2, 0.5)
    assert psnr(warped, truth, data_range=DR) > psnr(blended, truth, data_range=DR)


def test_farneback_warp_returns_the_right_shape_and_dtype():
    i0, i2 = _texture(seed=0), _texture(seed=1)
    out = farneback_warp(i0, i2, 0.5, data_range=DR)
    assert out.shape == i0.shape
    assert out.dtype == np.float32


def test_registry_exposes_both_required_baselines():
    assert set(BASELINES) == {"linear", "farneback"}


def test_run_baseline_dispatches_and_rejects_unknown_names():
    i0, i2 = _texture(seed=0), _texture(seed=1)
    np.testing.assert_allclose(
        run_baseline("linear", i0, i2, 0.5, data_range=DR), linear_blend(i0, i2, 0.5)
    )
    assert run_baseline("farneback", i0, i2, 0.5, data_range=DR).shape == i0.shape
    with pytest.raises(KeyError):
        run_baseline("magic", i0, i2, 0.5, data_range=DR)
