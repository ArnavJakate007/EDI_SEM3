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


# --------------------------------------------------------- scene classification


def _scene_frame():
    """A synthetic frame with one patch per scene category."""
    from sattsr.data.normalize import CLEAR_MIN_K, CONVECTIVE_MAX_K

    rng = np.random.default_rng(0)
    a = np.full((60, 60), 255.0, dtype=np.float32)
    a[:20, :] = CONVECTIVE_MAX_K - 15.0
    a[40:, :] = CLEAR_MIN_K + 10.0
    a[20:40, :] += rng.normal(0, 4.0, (20, 60))
    return a


def test_classify_scene_separates_the_three_categories():
    from sattsr.data.normalize import CLEAR_WARM, CONVECTIVE, STRATIFORM, classify_scene

    labels = classify_scene(_scene_frame())
    assert (labels[:15, :] == CONVECTIVE).all()
    assert (labels[45:, :] == CLEAR_WARM).all()
    assert (labels[25:35, :] == STRATIFORM).any()


def test_classify_scene_marks_invalid_pixels_empty():
    from sattsr.data.normalize import classify_scene

    a = _scene_frame()
    a[0, 0] = np.nan
    assert classify_scene(a)[0, 0] == ""


def test_textured_warm_cloud_is_not_called_clear():
    """The uniformity test is what stops broken warm cloud masquerading as clear sky."""
    from sattsr.data.normalize import CLEAR_MIN_K, CLEAR_WARM, classify_scene

    rng = np.random.default_rng(1)
    busy = (CLEAR_MIN_K + 10.0 + rng.normal(0, 8.0, (40, 40))).astype(np.float32)
    assert (classify_scene(busy) == CLEAR_WARM).mean() < 0.2


def test_land_mask_excludes_land_from_clear_warm():
    from sattsr.data.normalize import CLEAR_WARM, classify_scene

    a = _scene_frame()
    mask = np.zeros(a.shape, dtype=bool)
    mask[40:, :30] = True
    labels = classify_scene(a, land_mask=mask)

    assert not (labels[40:, :30] == CLEAR_WARM).any(), "masked land must never be clear_warm"
    # Rows 45+ rather than 40+: the 5x5 texture window straddles the band boundary at
    # rows 40-41, so those are legitimately not uniform enough to call clear.
    assert (labels[45:, 30:] == CLEAR_WARM).all()


def test_scene_samples_partition_the_valid_pixels():
    from sattsr.data.normalize import scene_samples

    a = _scene_frame()
    total = sum(v.size for v in scene_samples(a).values())
    assert total == int(np.isfinite(a).sum())


# ------------------------------------------------------------ plausibility guard


def _shift_map(shift_k, iqr_scale=1.0, scene="all"):
    from sattsr.data.normalize import QuantileMap

    q = np.linspace(0.0, 1.0, 64)
    src = np.linspace(220.0, 300.0, 64).astype(np.float32)
    mid = float(np.median(src))
    dst = ((src - mid) * iqr_scale + mid + shift_k).astype(np.float32)
    return QuantileMap(sensor="insat3dr", reference="goes19", quantiles=q,
                       source_values=src, target_values=dst, scene=scene)


def test_a_small_shift_is_plausible():
    assert _shift_map(2.0).is_plausible()
    assert _shift_map(2.0).implausibility() == []


def test_a_large_shift_is_flagged():
    reasons = _shift_map(18.0).implausibility()
    assert reasons and "plausibility limit" in reasons[0]
    assert not _shift_map(18.0).is_plausible()


def test_a_distorted_spread_is_flagged():
    reasons = _shift_map(0.0, iqr_scale=0.5).implausibility()
    assert any("IQR ratio" in r for r in reasons)


def test_the_threshold_is_configurable():
    assert _shift_map(8.0).is_plausible(max_shift_k=10.0)
    assert not _shift_map(8.0).is_plausible(max_shift_k=5.0)


def test_sensor_renorm_reports_which_scenes_are_implausible():
    from sattsr.data.normalize import SensorRenorm

    r = SensorRenorm("insat3dr", "goes19", {
        "clear_warm": _shift_map(1.0, scene="clear_warm"),
        "convective": _shift_map(25.0, scene="convective"),
    })
    assert set(r.implausible_scenes()) == {"convective"}
    assert not r.is_plausible()


# ------------------------------------------------ scene-stratified application


def test_sensor_renorm_applies_a_different_map_per_scene():
    from sattsr.data.normalize import SensorRenorm

    r = SensorRenorm("insat3dr", "goes19", {
        "convective": _shift_map(-3.0, scene="convective"),
        "clear_warm": _shift_map(+4.0, scene="clear_warm"),
        "stratiform": _shift_map(0.0, scene="stratiform"),
    })
    a = _scene_frame()
    out = r.apply(a)
    assert out[:15, :].mean() < a[:15, :].mean(), "convective should shift down"
    assert out[45:, :].mean() > a[45:, :].mean(), "clear should shift up"


def test_sensor_renorm_with_only_a_pooled_map_applies_it_everywhere():
    from sattsr.data.normalize import POOLED, SensorRenorm

    r = SensorRenorm("insat3dr", "goes19", {POOLED: _shift_map(5.0)})
    a = _scene_frame()
    out = r.apply(a)
    assert np.isfinite(out).sum() == np.isfinite(a).sum()
    assert out.mean() > a.mean()


def test_empty_renorm_passes_through_unchanged():
    from sattsr.data.normalize import SensorRenorm

    a = _scene_frame()
    np.testing.assert_array_equal(SensorRenorm("x", "goes19", {}).apply(a), a)


# --------------------------------------------------------------- provenance


def test_provenance_round_trips_and_is_saved(tmp_path):
    from sattsr.data.normalize import Provenance, RenormRegistry, fit_quantile_map

    rng = np.random.default_rng(7)
    prov = Provenance(
        n_source_frames=12, n_target_frames=146,
        source_bounds={"lat_min": -10.0, "lat_max": 40.0,
                       "lon_min": 60.0, "lon_max": 110.0},
        target_bounds={"lat_min": 10.0, "lat_max": 45.0,
                       "lon_min": -105.0, "lon_max": -60.0},
        source_dates=("2019-07-01T02:00:00", "2019-07-01T03:50:00"),
        notes="disjoint disks",
    )
    qmap = fit_quantile_map(
        rng.normal(250, 10, 4000), rng.normal(252, 10, 4000),
        sensor="himawari8", reference="goes19", scene="clear_warm", provenance=prov,
    )
    reg = RenormRegistry("goes19")
    reg.add(qmap)
    reg.save(tmp_path)

    got = RenormRegistry.load(tmp_path).maps["himawari8"]["clear_warm"].provenance
    assert got.n_source_frames == 12
    assert got.source_bounds["lon_min"] == 60.0
    assert got.notes == "disjoint disks"
    assert got.fitted_at, "fit time must be stamped automatically"
    assert "lon 60.0..110.0" in got.describe()


def test_saved_json_records_the_summary_and_any_implausibility(tmp_path):
    import json

    from sattsr.data.normalize import RenormRegistry

    reg = RenormRegistry("goes19")
    reg.add(_shift_map(18.0, scene="convective"))
    reg.save(tmp_path)
    entry = json.loads(
        (tmp_path / "insat3dr.json").read_text(encoding="utf-8")
    )["scenes"]["convective"]
    assert entry["summary"]["median_shift_k"] == pytest.approx(18.0, abs=0.1)
    assert entry["implausibility"], "an 18 K fit must be recorded as implausible on disk"


def test_registry_keeps_scenes_separate(tmp_path):
    from sattsr.data.normalize import RenormRegistry

    reg = RenormRegistry("goes19")
    reg.add(_shift_map(1.0, scene="clear_warm"))
    reg.add(_shift_map(2.0, scene="convective"))
    reg.save(tmp_path)

    back = RenormRegistry.load(tmp_path)
    assert set(back.maps["insat3dr"]) == {"clear_warm", "convective"}
    assert back.for_sensor("insat3dr") is not None
    assert back.for_sensor("nope") is None
