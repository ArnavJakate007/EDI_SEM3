"""Manifest-driven preprocessing: coverage gating, idempotency and failure handling."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sattsr.config import GridConfig
from sattsr.data.index import cache_path_for
from sattsr.data.prepare import build_tasks, prepare_one, run_prepare
from sattsr.io.manifest import scan_raw_files
from tests.conftest import make_goes_file

T0 = datetime(2026, 2, 10, 6, 0, tzinfo=timezone.utc)

# Small grid inside the synthetic GOES fixture's footprint (lon_origin 82.0).
GRID = GridConfig(lat_min=5.0, lat_max=15.0, lon_min=78.0, lon_max=88.0, resolution_deg=1.0)
# A grid far from the fixture's disc, so every regridded pixel is NaN.
OFF_DISK = GridConfig(lat_min=-80.0, lat_max=-70.0, lon_min=-170.0, lon_max=-160.0,
                      resolution_deg=1.0)


@pytest.fixture
def raw_rows(tmp_path):
    raw = tmp_path / "raw" / "goes19"
    for i in range(3):
        make_goes_file(raw, T0 + timedelta(minutes=10 * i))
    return scan_raw_files(raw, "goes19").rows


def _tasks(rows, cache_root, *, grid=GRID, min_coverage=0.5, force=False):
    return build_tasks(rows, cache_root=cache_root, grid_config=grid,
                       min_coverage=min_coverage, force=force)


def test_frames_are_written_to_the_cache_layout(tmp_path, raw_rows):
    cache = tmp_path / "cache" / "goes19"
    stats = run_prepare(_tasks(raw_rows, cache))

    assert stats.written == 3
    assert stats.failed == 0
    written = sorted(cache.rglob("*.npy"))
    assert len(written) == 3
    # Layout must match what the rest of the pipeline reads.
    assert written[0].parent.name == "20260210"
    assert written[0].stem == "060000"


def test_cached_array_is_float32_on_the_target_grid(tmp_path, raw_rows):
    cache = tmp_path / "cache" / "goes19"
    run_prepare(_tasks(raw_rows[:1], cache))

    arr = np.load(next(cache.rglob("*.npy")))
    assert arr.dtype == np.float32
    assert arr.shape == (10, 10), "must match the 10x10 target grid from GRID"


def test_preprocessing_is_idempotent(tmp_path, raw_rows):
    cache = tmp_path / "cache" / "goes19"
    run_prepare(_tasks(raw_rows, cache))
    again = run_prepare(_tasks(raw_rows, cache))

    assert again.written == 0
    assert again.skipped_cached == 3


def test_force_rewrites_cached_frames(tmp_path, raw_rows):
    cache = tmp_path / "cache" / "goes19"
    run_prepare(_tasks(raw_rows, cache))
    forced = run_prepare(_tasks(raw_rows, cache, force=True))

    assert forced.written == 3
    assert forced.skipped_cached == 0


def test_low_coverage_frames_are_skipped_and_logged(tmp_path, raw_rows, caplog):
    cache = tmp_path / "cache" / "goes19"
    with caplog.at_level(logging.WARNING, logger="sattsr.data.prepare"):
        stats = run_prepare(_tasks(raw_rows, cache, grid=OFF_DISK, min_coverage=0.5))

    assert stats.skipped_coverage == 3
    assert stats.written == 0
    assert list(cache.rglob("*.npy")) == [], "nothing should be cached"
    assert any("below" in m and "threshold" in m for m in caplog.messages)


def test_a_zero_threshold_accepts_an_off_disk_frame(tmp_path, raw_rows):
    """The gate is the only thing rejecting these, so it must be the gate deciding."""
    cache = tmp_path / "cache" / "goes19"
    stats = run_prepare(_tasks(raw_rows, cache, grid=OFF_DISK, min_coverage=0.0))
    assert stats.written == 3


def test_an_unreadable_file_is_counted_not_raised(tmp_path, raw_rows, caplog):
    cache = tmp_path / "cache" / "goes19"
    tasks = _tasks(raw_rows, cache)
    # Corrupt one source file in place; the others must still succeed.
    tasks[1].source.write_text("not netcdf at all", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="sattsr.data.prepare"):
        stats = run_prepare(tasks)

    assert stats.failed == 1
    assert stats.written == 2, "one bad file must not abort the batch"
    assert any("failed" in m for m in caplog.messages)


def test_refs_are_returned_sorted_for_the_index(tmp_path, raw_rows):
    cache = tmp_path / "cache" / "goes19"
    stats = run_prepare(_tasks(raw_rows, cache))
    stamps = [r.timestamp for r in stats.refs]
    assert stamps == sorted(stamps)
    assert all(r.sensor == "goes19" for r in stats.refs)


def test_dest_matches_cache_path_for(tmp_path, raw_rows):
    """build_tasks must agree with the canonical cache path helper."""
    from sattsr.data.index import FrameRef

    cache = tmp_path / "cache" / "goes19"
    task = _tasks(raw_rows[:1], cache)[0]
    ref = FrameRef(raw_rows[0].timestamp_utc, raw_rows[0].path, "goes19")
    assert task.dest == cache_path_for(cache, ref)


def test_goes19_cache_root_still_does_not_duplicate_the_sensor_segment(tmp_path, raw_rows):
    """Guards the nested cache/goes19/goes19 regression from the manifest path too."""
    cache = tmp_path / "cache" / "goes19"
    task = _tasks(raw_rows[:1], cache)[0]
    assert list(task.dest.parts).count("goes19") == 1


def test_empty_task_list_is_safe():
    stats = run_prepare([])
    assert (stats.read, stats.written, stats.failed) == (0, 0, 0)


def test_prepare_one_reports_coverage(tmp_path, raw_rows):
    cache = tmp_path / "cache" / "goes19"
    outcome = prepare_one(_tasks(raw_rows[:1], cache)[0])
    assert outcome.status == "written"
    assert 0.0 < outcome.coverage <= 1.0


@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_and_serial_paths_agree(tmp_path, raw_rows, workers):
    cache = tmp_path / "cache" / f"goes19_w{workers}"
    stats = run_prepare(_tasks(raw_rows, cache), workers=workers)
    assert stats.written == 3
    assert len(sorted(cache.rglob("*.npy"))) == 3


# ------------------------------------------------ renormalisation at cache time


def _identity_shift_map(shift_k: float):
    """A SensorRenorm whose pooled map adds `shift_k` K, for an exact assertion."""
    from sattsr.data.normalize import POOLED, QuantileMap, SensorRenorm

    src = np.linspace(150.0, 350.0, 64).astype(np.float32)
    qmap = QuantileMap(
        sensor="himawari8", reference="goes19",
        quantiles=np.linspace(0.0, 1.0, 64),
        source_values=src,
        target_values=(src + shift_k).astype(np.float32),
        scene=POOLED,
    )
    return SensorRenorm(sensor="himawari8", reference="goes19", maps={POOLED: qmap})


def test_no_renorm_leaves_the_cached_array_untouched(tmp_path, raw_rows):
    cache = tmp_path / "cache" / "goes19"
    run_prepare(_tasks(raw_rows[:1], cache))
    baseline = np.load(next(cache.rglob("*.npy")))
    assert np.isfinite(baseline).any()


def test_renorm_is_applied_before_writing_the_cache(tmp_path, raw_rows):
    """Renormalisation is baked into the .npy, not applied per batch later."""
    plain = tmp_path / "cache" / "plain"
    shifted = tmp_path / "cache" / "shifted"

    run_prepare(build_tasks(raw_rows[:1], cache_root=plain, grid_config=GRID))
    run_prepare(build_tasks(raw_rows[:1], cache_root=shifted, grid_config=GRID,
                            renorm=_identity_shift_map(5.0)))

    a = np.load(next(plain.rglob("*.npy")))
    b = np.load(next(shifted.rglob("*.npy")))
    both = np.isfinite(a) & np.isfinite(b)
    assert both.any()
    np.testing.assert_allclose(b[both], a[both] + 5.0, atol=0.2)


def test_renorm_preserves_the_invalid_mask(tmp_path, raw_rows):
    """A renorm map must not turn NaN into a number or vice versa."""
    plain = tmp_path / "cache" / "plain"
    shifted = tmp_path / "cache" / "shifted"
    run_prepare(build_tasks(raw_rows[:1], cache_root=plain, grid_config=GRID))
    run_prepare(build_tasks(raw_rows[:1], cache_root=shifted, grid_config=GRID,
                            renorm=_identity_shift_map(5.0)))

    a = np.load(next(plain.rglob("*.npy")))
    b = np.load(next(shifted.rglob("*.npy")))
    np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))


def test_renorm_survives_the_process_pool(tmp_path, raw_rows):
    """PrepareTask carries the map across a spawn boundary, so it must pickle."""
    cache = tmp_path / "cache" / "parallel"
    stats = run_prepare(
        build_tasks(raw_rows, cache_root=cache, grid_config=GRID,
                    renorm=_identity_shift_map(3.0)),
        workers=2,
    )
    assert stats.written == 3
    assert stats.failed == 0
