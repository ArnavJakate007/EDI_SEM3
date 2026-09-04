from __future__ import annotations

import numpy as np
import pytest

from sattsr.geo.grid import TargetGrid
from sattsr.geo.resample import (
    GeosProjection,
    bilinear_sample,
    regrid_geos,
    regrid_latlon,
    regrid_scattered,
)


def test_bilinear_sample_is_exact_on_integer_positions():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = bilinear_sample(src, np.array([[0.0, 2.0]]), np.array([[1.0, 3.0]]))
    np.testing.assert_allclose(out, [[1.0, 11.0]])


def test_bilinear_sample_interpolates_midpoints():
    src = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    out = bilinear_sample(src, np.array([[0.5]]), np.array([[0.5]]))
    assert out[0, 0] == pytest.approx(15.0)


def test_bilinear_sample_returns_nan_outside_bounds():
    src = np.ones((3, 3), dtype=np.float32)
    out = bilinear_sample(src, np.array([[-1.0, 5.0, np.nan]]), np.array([[1.0, 1.0, 1.0]]))
    assert np.all(np.isnan(out))


def test_regrid_latlon_preserves_a_constant_field():
    src_lats = np.linspace(50.0, 0.0, 51)          # descending on purpose
    src_lons = np.linspace(55.0, 115.0, 61)
    src = np.full((51, 61), 250.0, dtype=np.float32)
    grid = TargetGrid(lat_min=10.0, lat_max=20.0, lon_min=70.0, lon_max=80.0, resolution_deg=1.0)
    out = regrid_latlon(src, src_lats, src_lons, grid)
    assert out.shape == grid.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 250.0, rtol=1e-5)


def test_regrid_latlon_marks_uncovered_area_nan():
    src_lats = np.linspace(20.0, 10.0, 11)
    src_lons = np.linspace(70.0, 80.0, 11)
    src = np.full((11, 11), 250.0, dtype=np.float32)
    grid = TargetGrid(lat_min=30.0, lat_max=40.0, lon_min=70.0, lon_max=80.0, resolution_deg=1.0)
    assert np.all(np.isnan(regrid_latlon(src, src_lats, src_lons, grid)))


def test_regrid_scattered_picks_nearest_neighbour():
    lat2d, lon2d = np.meshgrid(
        np.linspace(20.0, 10.0, 21), np.linspace(70.0, 80.0, 21), indexing="ij"
    )
    src = lat2d.astype(np.float32)
    grid = TargetGrid(lat_min=12.0, lat_max=18.0, lon_min=72.0, lon_max=78.0, resolution_deg=1.0)
    out = regrid_scattered(src, lat2d, lon2d, grid, max_distance_deg=1.0)
    assert out.shape == grid.shape
    lat_t, _ = grid.meshgrid()
    np.testing.assert_allclose(out, lat_t, atol=0.6)


def _fake_geos_projection() -> GeosProjection:
    """A 1000x1000 scan-angle grid spanning +/- 0.08 rad about nadir at 82 E."""
    ang = np.linspace(-0.08, 0.08, 1000)
    return GeosProjection(
        satellite_height=35786023.0,
        longitude_of_origin=82.0,
        semi_major_axis=6378137.0,
        semi_minor_axis=6356752.31414,
        sweep_axis="y",
        x=ang,
        y=-ang,           # y decreases downward, as in real ABI files
    )


def test_regrid_geos_recovers_a_monotonic_ramp():
    proj = _fake_geos_projection()
    src = np.tile(np.arange(1000, dtype=np.float32), (1000, 1))   # value == column index
    grid = TargetGrid(lat_min=10.0, lat_max=20.0, lon_min=75.0, lon_max=85.0, resolution_deg=1.0)
    out = regrid_geos(src, proj, grid)
    assert out.shape == grid.shape
    assert np.isfinite(out).all()
    assert np.all(np.diff(out, axis=1) > 0)      # east -> higher column index


def test_regrid_geos_returns_nan_off_disk():
    proj = _fake_geos_projection()
    src = np.ones((1000, 1000), dtype=np.float32)
    grid = TargetGrid(
        lat_min=-5.0, lat_max=5.0, lon_min=-100.0, lon_max=-90.0, resolution_deg=1.0
    )   # opposite side of the Earth
    assert np.all(np.isnan(regrid_geos(src, proj, grid)))
