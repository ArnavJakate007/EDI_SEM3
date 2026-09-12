"""Quantile renormalisation, on synthetic BT distributions."""

from __future__ import annotations

import numpy as np
import pytest

from sattsr.config import NormalizationConfig
from sattsr.data.normalize import (
    QuantileMap,
    RenormRegistry,
    fit_quantile_map,
    from_model_range,
    to_model_range,
)

NORM = NormalizationConfig(bt_min=180.0, bt_max=330.0)


def _samples(rng, *, mean, std, n=5000):
    return rng.normal(mean, std, n).astype(np.float32)


def _fit(rng, *, src_mean=250.0, dst_mean=260.0):
    return fit_quantile_map(
        _samples(rng, mean=src_mean, std=12.0),
        _samples(rng, mean=dst_mean, std=12.0),
        sensor="insat3dr",
        reference="goes19",
    )


def test_quantile_map_moves_the_distribution_onto_the_reference():
    rng = np.random.default_rng(0)
    qmap = _fit(rng)

    source = _samples(rng, mean=250.0, std=12.0)
    mapped = qmap.apply(source)

    assert abs(float(np.mean(mapped)) - 260.0) < 1.0, "mean should land on the reference"
    assert abs(float(np.mean(source)) - 250.0) < 1.0, "source must not be mutated"


def test_mapping_is_monotone():
    rng = np.random.default_rng(1)
    qmap = _fit(rng)
    probe = np.linspace(200.0, 300.0, 200, dtype=np.float32)
    mapped = qmap.apply(probe)
    assert np.all(np.diff(mapped) >= -1e-4), "a BT mapping that reorders temperatures is wrong"


def test_nan_survives_the_mapping():
    rng = np.random.default_rng(2)
    qmap = _fit(rng)
    out = qmap.apply(np.array([250.0, np.nan, 260.0], dtype=np.float32))
    assert np.isnan(out[1])
    assert np.isfinite(out[0]) and np.isfinite(out[2])


def test_out_of_range_values_are_clamped_not_extrapolated():
    """An absurdly cold pixel must not map to a physically impossible temperature."""
    rng = np.random.default_rng(3)
    qmap = _fit(rng)
    out = qmap.apply(np.array([50.0, 500.0], dtype=np.float32))
    assert float(out[0]) == pytest.approx(float(qmap.target_values[0]), abs=1e-3)
    assert float(out[1]) == pytest.approx(float(qmap.target_values[-1]), abs=1e-3)


def test_summary_reports_the_median_shift():
    rng = np.random.default_rng(4)
    qmap = _fit(rng, src_mean=250.0, dst_mean=262.0)
    summary = qmap.summary()
    assert summary["median_shift_k"] == pytest.approx(12.0, abs=1.0)
    assert summary["iqr_ratio"] == pytest.approx(1.0, abs=0.15)


def test_fit_rejects_too_few_samples():
    with pytest.raises(ValueError, match="at least"):
        fit_quantile_map(
            np.array([250.0, 251.0]), np.arange(100.0) + 200.0,
            sensor="insat3dr", reference="goes19",
        )


def test_fit_ignores_nan_samples():
    rng = np.random.default_rng(5)
    src = _samples(rng, mean=250.0, std=10.0)
    src[::3] = np.nan
    qmap = fit_quantile_map(
        src, _samples(rng, mean=250.0, std=10.0), sensor="a", reference="b"
    )
    assert np.all(np.isfinite(qmap.source_values))
    assert qmap.n_source_samples < src.size


# ------------------------------------------------------------------------ registry


def test_registry_round_trips_through_disk(tmp_path):
    rng = np.random.default_rng(6)
    registry = RenormRegistry(reference="goes19")
    registry.add(_fit(rng))
    registry.save(tmp_path)

    loaded = RenormRegistry.load(tmp_path)
    assert loaded.reference == "goes19"
    assert "insat3dr" in loaded

    probe = np.array([240.0, 250.0, 260.0], dtype=np.float32)
    np.testing.assert_allclose(
        loaded.apply(probe, "insat3dr"), registry.apply(probe, "insat3dr"), rtol=1e-5
    )


def test_registry_passes_the_reference_sensor_through_untouched():
    registry = RenormRegistry(reference="goes19")
    probe = np.array([250.0, 260.0], dtype=np.float32)
    np.testing.assert_array_equal(registry.apply(probe, "goes19"), probe)


def test_registry_passes_unknown_sensors_through_rather_than_zeroing():
    registry = RenormRegistry(reference="goes19")
    probe = np.array([250.0, 260.0], dtype=np.float32)
    np.testing.assert_array_equal(registry.apply(probe, "himawari8"), probe)


def test_registry_rejects_a_map_with_the_wrong_reference():
    qmap = QuantileMap(
        sensor="x", reference="himawari8",
        quantiles=np.linspace(0, 1, 4),
        source_values=np.arange(4.0, dtype=np.float32),
        target_values=np.arange(4.0, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="reference"):
        RenormRegistry(reference="goes19").add(qmap)


def test_loading_a_missing_registry_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="fit_renorm"):
        RenormRegistry.load(tmp_path)


# --------------------------------------------------------------------- model range


def test_model_range_round_trips():
    bt = np.array([200.0, 250.0, 300.0], dtype=np.float32)
    back = from_model_range(to_model_range(bt, NORM), NORM)
    np.testing.assert_allclose(back, bt, atol=1e-3)


def test_model_range_defaults_to_zero_one_to_match_the_interpolator_clamp():
    """FrameInterpolator clamps to [0, 1]; emitting [-1, 1] would clip half the range."""
    bt = np.array([180.0, 255.0, 330.0], dtype=np.float32)
    x = to_model_range(bt, NORM)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    assert float(x[1]) == pytest.approx(0.5, abs=1e-3)


def test_model_range_supports_an_explicit_symmetric_range():
    bt = np.array([180.0, 255.0, 330.0], dtype=np.float32)
    x = to_model_range(bt, NORM, lo=-1.0, hi=1.0)
    assert float(x[0]) == pytest.approx(-1.0, abs=1e-3)
    assert float(x[1]) == pytest.approx(0.0, abs=1e-3)
    np.testing.assert_allclose(from_model_range(x, NORM, lo=-1.0, hi=1.0), bt, atol=1e-3)


def test_to_model_range_agrees_with_radiometry_normalize_bt():
    """The two normalisers must not drift apart."""
    from sattsr.data.radiometry import normalize_bt

    bt = np.array([190.0, 240.0, 290.0, 340.0], dtype=np.float32)
    np.testing.assert_allclose(to_model_range(bt, NORM), normalize_bt(bt, NORM), atol=1e-6)


def test_nan_becomes_the_low_end():
    x = to_model_range(np.array([np.nan], dtype=np.float32), NORM)
    assert float(x[0]) == 0.0
