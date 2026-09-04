from __future__ import annotations

import numpy as np
import pytest

from sattsr.config import GridConfig
from sattsr.geo.grid import TargetGrid


def test_grid_shape_and_ordering():
    grid = TargetGrid(lat_min=0.0, lat_max=1.0, lon_min=10.0, lon_max=12.0, resolution_deg=0.5)
    assert grid.shape == (2, 4)
    # latitude descends (north at row 0), longitude ascends
    np.testing.assert_allclose(grid.lats, [0.75, 0.25])
    np.testing.assert_allclose(grid.lons, [10.25, 10.75, 11.25, 11.75])


def test_meshgrid_matches_shape():
    grid = TargetGrid(lat_min=0.0, lat_max=1.0, lon_min=10.0, lon_max=12.0, resolution_deg=0.5)
    lat2d, lon2d = grid.meshgrid()
    assert lat2d.shape == grid.shape
    assert lon2d.shape == grid.shape
    assert lat2d[0, 0] == pytest.approx(0.75)
    assert lon2d[0, -1] == pytest.approx(11.75)


def test_from_config_round_trips():
    grid = TargetGrid.from_config(
        GridConfig(lat_min=-10, lat_max=40, lon_min=60, lon_max=110, resolution_deg=0.05)
    )
    assert grid.shape == (1000, 1000)


def test_degenerate_extent_is_rejected():
    with pytest.raises(ValueError):
        TargetGrid(lat_min=5.0, lat_max=5.0, lon_min=0.0, lon_max=1.0, resolution_deg=0.5)
