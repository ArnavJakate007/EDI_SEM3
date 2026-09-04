from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from sattsr.geo.grid import TargetGrid
from sattsr.io.himawari import HimawariReader
from tests.conftest import make_himawari_file


def test_timestamp_parsed_from_filename(tmp_path):
    ts = datetime(2026, 9, 4, 6, 30, tzinfo=timezone.utc)
    assert HimawariReader().timestamp_of(make_himawari_file(tmp_path, ts)) == ts


def test_read_returns_kelvin_on_target_grid(tmp_path):
    ts = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    grid = TargetGrid(lat_min=10.0, lat_max=20.0, lon_min=75.0, lon_max=85.0, resolution_deg=1.0)
    frame = HimawariReader().read(make_himawari_file(tmp_path, ts), grid)
    assert frame.sensor == "himawari8"
    assert frame.bt.shape == grid.shape
    assert frame.bt.dtype == np.float32
    assert np.isfinite(frame.bt).all()
    assert 200.0 < float(frame.bt.min()) and float(frame.bt.max()) < 320.0
