from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from sattsr.geo.grid import TargetGrid
from sattsr.io.goes import Goes19Reader
from tests.conftest import make_goes_file


def _grid() -> TargetGrid:
    return TargetGrid(lat_min=5.0, lat_max=15.0, lon_min=78.0, lon_max=88.0, resolution_deg=1.0)


def test_timestamp_is_parsed_from_the_abi_filename(tmp_path):
    ts = datetime(2026, 9, 4, 6, 30, 20, tzinfo=timezone.utc)
    path = make_goes_file(tmp_path, ts)
    assert Goes19Reader().timestamp_of(path) == ts


def test_bad_filename_raises(tmp_path):
    bad = tmp_path / "not_an_abi_file.nc"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        Goes19Reader().timestamp_of(bad)


def test_read_returns_kelvin_on_the_target_grid(tmp_path):
    ts = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    path = make_goes_file(tmp_path, ts)
    frame = Goes19Reader().read(path, _grid())
    assert frame.sensor == "goes19"
    assert frame.timestamp == ts
    assert frame.bt.shape == _grid().shape
    assert frame.bt.dtype == np.float32
    finite = frame.bt[np.isfinite(frame.bt)]
    assert finite.size > 0
    assert finite.min() > 200.0 and finite.max() < 300.0


def test_read_marks_off_disk_area_nan(tmp_path):
    path = make_goes_file(tmp_path, datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc))
    far_grid = TargetGrid(lat_min=5.0, lat_max=15.0, lon_min=-100.0, lon_max=-90.0,
                          resolution_deg=1.0)
    assert np.all(np.isnan(Goes19Reader().read(path, far_grid).bt))


def test_discover_finds_and_sorts_files(goes_dir):
    paths = Goes19Reader().discover(goes_dir)
    assert len(paths) == 5
    stamps = [Goes19Reader().timestamp_of(p) for p in paths]
    assert stamps == sorted(stamps)
