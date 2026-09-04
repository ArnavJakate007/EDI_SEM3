from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from sattsr.geo.grid import TargetGrid
from sattsr.io.insat import InsatReader
from tests.conftest import make_insat_file


def _grid() -> TargetGrid:
    return TargetGrid(lat_min=10.0, lat_max=20.0, lon_min=70.0, lon_max=80.0, resolution_deg=1.0)


def test_timestamp_parsed_from_mosdac_filename(tmp_path):
    ts = datetime(2026, 9, 4, 6, 30, tzinfo=timezone.utc)
    assert InsatReader().timestamp_of(make_insat_file(tmp_path, ts)) == ts


def test_read_applies_the_temperature_lut(tmp_path):
    ts = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    frame = InsatReader().read(make_insat_file(tmp_path, ts), _grid())
    assert frame.sensor == "insat3dr"
    assert frame.bt.shape == _grid().shape
    assert frame.bt.dtype == np.float32
    finite = frame.bt[np.isfinite(frame.bt)]
    assert finite.size > 0
    assert finite.min() >= 150.0 and finite.max() <= 350.0


def test_read_handles_curvilinear_geolocation(tmp_path):
    ts = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    path = make_insat_file(tmp_path, ts, curvilinear=True)
    frame = InsatReader().read(path, _grid())
    assert frame.bt.shape == _grid().shape
    assert np.isfinite(frame.bt).any()


def test_sensor_label_is_configurable(tmp_path):
    ts = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    frame = InsatReader(sensor="insat3ds").read(make_insat_file(tmp_path, ts), _grid())
    assert frame.sensor == "insat3ds"


def test_bad_filename_raises(tmp_path):
    bad = tmp_path / "random.h5"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        InsatReader().timestamp_of(bad)
