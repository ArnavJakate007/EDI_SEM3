from __future__ import annotations

import numpy as np
import pytest

from sattsr.eval.metrics import fsim, metric_suite, mse, phase_congruency, psnr, rmse, ssim

DR = 150.0


def _cloud_field(seed: int = 0, shift: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    base = 250.0 + 25.0 * np.sin((xx + shift) * 0.25) * np.cos(yy * 0.2)
    return (base + rng.normal(0.0, 0.5, size=(64, 64))).astype(np.float32)


def test_identical_arrays_score_perfectly():
    a = _cloud_field()
    assert mse(a, a) == pytest.approx(0.0, abs=1e-6)
    assert rmse(a, a) == pytest.approx(0.0, abs=1e-6)
    assert psnr(a, a, data_range=DR) == pytest.approx(100.0)
    assert ssim(a, a, data_range=DR) == pytest.approx(1.0, abs=1e-4)
    assert fsim(a, a, data_range=DR) == pytest.approx(1.0, abs=1e-4)


def test_metrics_degrade_monotonically_with_noise():
    a = _cloud_field()
    rng = np.random.default_rng(1)
    small = a + rng.normal(0.0, 1.0, size=a.shape).astype(np.float32)
    large = a + rng.normal(0.0, 8.0, size=a.shape).astype(np.float32)

    assert mse(a, small) < mse(a, large)
    assert psnr(a, small, data_range=DR) > psnr(a, large, data_range=DR)
    assert ssim(a, small, data_range=DR) > ssim(a, large, data_range=DR)
    assert fsim(a, small, data_range=DR) > fsim(a, large, data_range=DR)


def test_rmse_is_the_square_root_of_mse():
    a, b = _cloud_field(0), _cloud_field(1)
    assert rmse(a, b) == pytest.approx(np.sqrt(mse(a, b)), rel=1e-6)


def test_psnr_matches_its_definition():
    a = np.full((16, 16), 250.0, dtype=np.float32)
    b = a + 1.5
    assert psnr(a, b, data_range=DR) == pytest.approx(
        20.0 * np.log10(DR) - 10.0 * np.log10(1.5**2), rel=1e-4
    )


def test_nan_pixels_are_excluded_not_propagated():
    a = _cloud_field()
    b = a.copy()
    b[:8, :] = np.nan
    for value in (mse(a, b), psnr(a, b, data_range=DR), ssim(a, b, data_range=DR),
                  fsim(a, b, data_range=DR)):
        assert np.isfinite(value)
    assert mse(a, b) == pytest.approx(0.0, abs=1e-6)


def test_all_nan_returns_nan_not_a_crash():
    a = np.full((16, 16), np.nan, dtype=np.float32)
    assert np.isnan(mse(a, a))
    assert np.isnan(ssim(a, a, data_range=DR))
    assert np.isnan(fsim(a, a, data_range=DR))


def test_ssim_agrees_with_skimage_on_clean_data():
    skimage = pytest.importorskip("skimage.metrics")
    a, b = _cloud_field(0), _cloud_field(0, shift=2)
    theirs = skimage.structural_similarity(
        a.astype(np.float64), b.astype(np.float64), data_range=DR,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    assert ssim(a, b, data_range=DR) == pytest.approx(theirs, abs=1e-6)


def test_phase_congruency_peaks_on_a_structural_edge():
    img = np.zeros((64, 64), dtype=np.float64)
    img[:, 32:] = 200.0
    pc = phase_congruency(img)
    assert pc.shape == img.shape
    assert np.isfinite(pc).all()
    assert float(pc[:, 30:34].mean()) > float(pc[:, 5:15].mean())


def test_fsim_is_bounded():
    a, b = _cloud_field(0), _cloud_field(5, shift=7)
    value = fsim(a, b, data_range=DR)
    assert 0.0 <= value <= 1.0


def test_fsim_prefers_structural_agreement_over_a_constant_offset():
    a = _cloud_field()
    shifted = a + 4.0                      # same structure, different bias
    scrambled = _cloud_field(9, shift=17)  # different structure
    assert fsim(a, shifted, data_range=DR) > fsim(a, scrambled, data_range=DR)


def test_metric_suite_returns_every_required_metric():
    result = metric_suite(_cloud_field(0), _cloud_field(0, shift=1), data_range=DR)
    assert set(result) == {"mse", "rmse", "psnr", "ssim", "fsim"}
    assert all(isinstance(v, float) for v in result.values())
