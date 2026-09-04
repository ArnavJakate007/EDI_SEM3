from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sattsr.geo.grid import TargetGrid
from sattsr.io.base import Frame
from sattsr.io.writer import read_frames_nc, write_frames_nc


def _grid() -> TargetGrid:
    return TargetGrid(lat_min=10.0, lat_max=15.0, lon_min=70.0, lon_max=75.0, resolution_deg=1.0)


def _frames(grid: TargetGrid, n: int = 3) -> list[Frame]:
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        bt = np.full(grid.shape, 250.0 + i, dtype=np.float32)
        bt[0, 0] = np.nan
        out.append(Frame(t0 + timedelta(minutes=15 * i), bt, "insat3dr", Path("x.h5")))
    return out


def test_round_trip_preserves_values_times_and_flags(tmp_path):
    grid = _grid()
    frames = _frames(grid)
    out = write_frames_nc(tmp_path / "o.nc", frames, grid,
                          synthetic=[False, True, False], model_version="v0.1.0")
    assert out.exists()

    back, flags = read_frames_nc(out)
    assert flags == [False, True, False]
    assert [f.timestamp for f in back] == [f.timestamp for f in frames]
    for a, b in zip(back, frames):
        np.testing.assert_allclose(a.bt, b.bt, equal_nan=True)
        assert a.bt.dtype == np.float32


def test_output_carries_the_synthetic_flag_and_warning(tmp_path):
    grid = _grid()
    out = write_frames_nc(tmp_path / "o.nc", _frames(grid), grid,
                          synthetic=[False, True, False], model_version="v0.1.0")
    with xr.open_dataset(out) as ds:
        assert "synthetic" in ds
        assert ds["synthetic"].attrs["flag_meanings"] == "observed synthesized"
        np.testing.assert_array_equal(ds["synthetic"].values, [0, 1, 0])
        assert "NOT observations" in ds.attrs["comment"]
        assert ds.attrs["model_version"] == "v0.1.0"
        assert ds.attrs["Conventions"].startswith("CF-")
        assert ds["brightness_temperature"].attrs["units"] == "K"


def test_coordinates_match_the_grid(tmp_path):
    grid = _grid()
    out = write_frames_nc(tmp_path / "o.nc", _frames(grid), grid,
                          synthetic=[False] * 3, model_version="v0.1.0")
    with xr.open_dataset(out) as ds:
        np.testing.assert_allclose(ds["lat"].values, grid.lats)
        np.testing.assert_allclose(ds["lon"].values, grid.lons)


def test_mismatched_flag_length_is_rejected(tmp_path):
    grid = _grid()
    with pytest.raises(ValueError):
        write_frames_nc(tmp_path / "o.nc", _frames(grid), grid,
                        synthetic=[False], model_version="v0.1.0")


def test_empty_frame_list_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_frames_nc(tmp_path / "o.nc", [], _grid(), synthetic=[], model_version="v0.1.0")
