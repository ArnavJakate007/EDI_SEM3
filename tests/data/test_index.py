from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from sattsr.data.index import (
    FrameRef,
    build_index,
    cache_path_for,
    load_cached,
    load_index,
    prepare_cache,
    save_index,
)
from sattsr.geo.grid import TargetGrid
from sattsr.io.goes import Goes19Reader


def _grid() -> TargetGrid:
    return TargetGrid(lat_min=5.0, lat_max=15.0, lon_min=78.0, lon_max=88.0, resolution_deg=1.0)


def test_build_index_is_sorted_and_complete(goes_dir):
    refs = build_index(Goes19Reader(), goes_dir)
    assert len(refs) == 5
    assert [r.timestamp for r in refs] == sorted(r.timestamp for r in refs)
    assert all(r.sensor == "goes19" for r in refs)


def test_build_index_skips_unparseable_files(goes_dir):
    (goes_dir / "stray_C13_G19_file.nc").write_text("junk", encoding="utf-8")
    assert len(build_index(Goes19Reader(), goes_dir)) == 5


def test_index_json_round_trip(tmp_path, goes_dir):
    refs = build_index(Goes19Reader(), goes_dir)
    out = save_index(refs, tmp_path / "index.json")
    assert load_index(out) == refs


def test_cache_path_layout(tmp_path):
    ref = FrameRef(datetime(2026, 9, 4, 6, 30, 20, tzinfo=timezone.utc),
                   tmp_path / "x.nc", "goes19")
    assert cache_path_for(tmp_path / "cache", ref) == (
        tmp_path / "cache" / "goes19" / "20260904" / "063020.npy"
    )


def test_prepare_cache_writes_kelvin_arrays(tmp_path, goes_dir):
    grid = _grid()
    refs = build_index(Goes19Reader(), goes_dir)
    cached = prepare_cache(Goes19Reader(), refs, grid, tmp_path / "cache")

    assert len(cached) == len(refs)
    assert [c.timestamp for c in cached] == [r.timestamp for r in refs]
    for c in cached:
        assert c.path.suffix == ".npy"
        arr = load_cached(c)
        assert arr.shape == grid.shape
        assert arr.dtype == np.float32
        finite = arr[np.isfinite(arr)]
        assert finite.size > 0 and finite.min() > 150.0 and finite.max() < 400.0


def test_prepare_cache_skips_existing_unless_overwrite(tmp_path, goes_dir):
    grid = _grid()
    refs = build_index(Goes19Reader(), goes_dir)[:1]
    cached = prepare_cache(Goes19Reader(), refs, grid, tmp_path / "cache")

    sentinel = np.full(grid.shape, 42.0, dtype=np.float32)
    np.save(cached[0].path, sentinel)

    prepare_cache(Goes19Reader(), refs, grid, tmp_path / "cache")
    np.testing.assert_allclose(load_cached(cached[0]), sentinel)

    prepare_cache(Goes19Reader(), refs, grid, tmp_path / "cache", overwrite=True)
    assert not np.allclose(load_cached(cached[0]), sentinel)


def test_prepare_cache_reports_unreadable_files_without_aborting(tmp_path, goes_dir):
    grid = _grid()
    refs = build_index(Goes19Reader(), goes_dir)
    broken = FrameRef(datetime(2026, 9, 4, 9, 0, tzinfo=timezone.utc),
                      goes_dir / "missing.nc", "goes19")
    cached = prepare_cache(Goes19Reader(), [*refs, broken], grid, tmp_path / "cache")
    assert len(cached) == len(refs)


def test_load_cached_missing_file_raises(tmp_path):
    ref = FrameRef(datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc), tmp_path / "nope.npy",
                   "goes19")
    with pytest.raises(FileNotFoundError):
        load_cached(ref)


# ------------------------------------------------------- orphan cache detection


def _cache_with_index(tmp_path, n=3):
    """A cache tree plus a matching index.json."""
    from sattsr.data.index import save_index

    root = tmp_path / "cache" / "goes19"
    refs = []
    for i in range(n):
        ts = datetime(2026, 2, 10, 6, 10 * i, tzinfo=timezone.utc)
        dest = root / ts.strftime("%Y%m%d") / f"{ts.strftime('%H%M%S')}.npy"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, np.zeros((4, 4), dtype=np.float32))
        refs.append(FrameRef(ts, dest, "goes19"))
    save_index(refs, root / "index.json")
    return root


def test_a_consistent_cache_has_no_orphans(tmp_path):
    from sattsr.data.index import assert_cache_matches_index, find_orphan_cache_files

    root = _cache_with_index(tmp_path)
    assert find_orphan_cache_files(root, "goes19") == []
    assert_cache_matches_index(root, "goes19")          # must not raise


def test_a_stale_nested_directory_is_detected_as_orphaned(tmp_path):
    """Exactly the cache/goes19/goes19/ layout the path bug used to produce."""
    from sattsr.data.index import find_orphan_cache_files

    root = _cache_with_index(tmp_path)
    stale = root / "goes19" / "20260210"
    stale.mkdir(parents=True)
    np.save(stale / "160021.npy", np.zeros((4, 4), dtype=np.float32))

    orphans = find_orphan_cache_files(root, "goes19")
    assert len(orphans) == 1
    assert orphans[0].name == "160021.npy"


def test_assert_cache_matches_index_fails_loudly(tmp_path):
    from sattsr.data.index import assert_cache_matches_index

    root = _cache_with_index(tmp_path)
    np.save(root / "20260210" / "999999.npy", np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="not in index.json"):
        assert_cache_matches_index(root, "goes19")


def test_orphan_check_is_silent_without_an_index(tmp_path):
    """No index means nothing to compare against -- not an error."""
    from sattsr.data.index import find_orphan_cache_files

    root = tmp_path / "cache" / "goes19" / "20260210"
    root.mkdir(parents=True)
    np.save(root / "060000.npy", np.zeros((4, 4), dtype=np.float32))
    assert find_orphan_cache_files(tmp_path / "cache" / "goes19", "goes19") == []
