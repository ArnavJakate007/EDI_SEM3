"""AHI-L2-FLDK-ISatSS tile handling, against synthetic tiles.

The fixture mirrors the conventions verified against a real downloaded file (see the
`sattsr.io.himawari` module docstring): `Sectorized_CMI` already in Kelvin, ellipsoid
attrs named `semi_major`/`semi_minor`, `sweep_angle_axis` of `y`, and x/y in
microradian. No network or real satellite data required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sattsr.geo.grid import TargetGrid
from sattsr.io.himawari import HimawariReader, is_isatss, isatss_key
from sattsr.io.manifest import scan_raw_files
from tests.conftest import make_himawari_file, make_isatss_scan, make_isatss_tile

T0 = datetime(2019, 7, 1, 2, 0, tzinfo=timezone.utc)


def _grid() -> TargetGrid:
    """A small grid near the sub-satellite point, where the synthetic tiles sit."""
    return TargetGrid(
        lat_min=-2.0, lat_max=2.0, lon_min=139.0, lon_max=143.0, resolution_deg=0.25
    )


# ------------------------------------------------------------------ identification


def test_isatss_tiles_are_recognised(tmp_path):
    tile = make_isatss_tile(tmp_path, T0)
    assert is_isatss(tile)
    assert isatss_key(tile) == ("13", "2019182020000")


def test_gridded_ptree_files_are_not_mistaken_for_isatss(tmp_path):
    gridded = make_himawari_file(tmp_path, T0)
    assert not is_isatss(gridded)
    assert isatss_key(gridded) is None


def test_timestamp_parses_the_13_digit_stamp(tmp_path):
    """ISatSS uses YYYYDDDHHMMSS -- 13 digits, unlike GOES ABI's 14."""
    tile = make_isatss_tile(tmp_path, T0)
    assert HimawariReader().timestamp_of(tile) == T0


def test_both_products_parse_through_one_reader(tmp_path):
    reader = HimawariReader()
    assert reader.timestamp_of(make_isatss_tile(tmp_path, T0)) == T0
    assert reader.timestamp_of(make_himawari_file(tmp_path, T0)) == T0


def test_an_unparseable_name_still_raises(tmp_path):
    bad = tmp_path / "OR_HFD-nonsense.nc"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a Himawari filename"):
        HimawariReader().timestamp_of(bad)


# --------------------------------------------------------------------- grouping


def test_discover_collapses_a_tile_group_to_one_frame(tmp_path):
    """76 tiles of one scan must look like one frame, not 76 duplicate timestamps."""
    make_isatss_scan(tmp_path, T0, n_tiles=5)
    frames = HimawariReader().discover(tmp_path)
    assert len(frames) == 1


def test_discover_keeps_separate_scans_separate(tmp_path):
    for i in range(3):
        make_isatss_scan(tmp_path, T0 + timedelta(minutes=10 * i), n_tiles=4)
    frames = HimawariReader().discover(tmp_path)
    assert len(frames) == 3
    stamps = [HimawariReader().timestamp_of(f) for f in frames]
    assert stamps == sorted(stamps)
    assert stamps[-1] - stamps[0] == timedelta(minutes=20)


def test_discover_ignores_other_channels(tmp_path):
    make_isatss_tile(tmp_path, T0, tile=1, channel=13)
    make_isatss_tile(tmp_path, T0, tile=1, channel=2)
    make_isatss_tile(tmp_path, T0, tile=2, channel=7)
    assert len(HimawariReader(channel=13).discover(tmp_path)) == 1


def test_sibling_tiles_finds_the_whole_scan(tmp_path):
    paths = make_isatss_scan(tmp_path, T0, n_tiles=4)
    make_isatss_scan(tmp_path, T0 + timedelta(minutes=10), n_tiles=4)
    siblings = HimawariReader().sibling_tiles(paths[0])
    assert len(siblings) == 4, "must not pull in the neighbouring scan's tiles"


def test_manifest_reports_one_row_per_scan_not_per_tile(tmp_path):
    """Regression: globbing instead of discover() made every tile its own frame."""
    for i in range(3):
        make_isatss_scan(tmp_path, T0 + timedelta(minutes=10 * i), n_tiles=6)
    report = scan_raw_files(tmp_path, "himawari8")
    assert len(report.rows) == 3
    assert report.duplicates == {}, "tiles must not surface as duplicate timestamps"


# ---------------------------------------------------------------------- reading


def test_read_returns_kelvin_on_the_target_grid(tmp_path):
    paths = make_isatss_scan(tmp_path, T0, n_tiles=3)
    frame = HimawariReader().read(paths[0], _grid())

    assert frame.sensor == "himawari8"
    assert frame.timestamp == T0
    assert frame.bt.shape == _grid().shape
    assert frame.bt.dtype == np.float32

    finite = frame.bt[np.isfinite(frame.bt)]
    assert finite.size > 0, "the synthetic tiles should cover the sub-satellite grid"
    assert finite.min() > 200.0 and finite.max() < 300.0


def test_mosaicking_tiles_covers_more_than_a_single_tile(tmp_path):
    """The whole point of tile assembly: more tiles, more covered target pixels."""
    scan = make_isatss_scan(tmp_path, T0, n_tiles=3)
    wide = TargetGrid(
        lat_min=-2.0, lat_max=2.0, lon_min=139.0, lon_max=147.0, resolution_deg=0.25
    )
    reader = HimawariReader()

    full = reader.read(scan[0], wide).bt
    full_cover = np.count_nonzero(np.isfinite(full))

    solo_dir = tmp_path / "solo"
    make_isatss_tile(solo_dir, T0, tile=1)
    solo = reader.read(next(solo_dir.glob("*.nc")), wide).bt
    solo_cover = np.count_nonzero(np.isfinite(solo))

    assert full_cover > solo_cover, (
        f"mosaic covered {full_cover} px, single tile {solo_cover} px"
    )


def test_a_corrupt_tile_is_skipped_not_fatal(tmp_path):
    scan = make_isatss_scan(tmp_path, T0, n_tiles=3)
    scan[1].write_text("not netcdf", encoding="utf-8")

    frame = HimawariReader().read(scan[0], _grid())
    assert np.count_nonzero(np.isfinite(frame.bt)) > 0, "surviving tiles must still read"


def test_physically_impossible_values_are_rejected(tmp_path):
    """ISatSS has no _FillValue; off-disc decodes to ~70 K and must be dropped."""
    import xarray as xr

    tile = make_isatss_tile(tmp_path, T0)
    with xr.open_dataset(tile) as ds:
        patched = ds.load().copy()
    patched["Sectorized_CMI"].values[:10, :] = 70.0      # the off-disc signature
    tile.unlink()
    patched.to_netcdf(tile)

    bt = HimawariReader().read(tile, _grid()).bt
    finite = bt[np.isfinite(bt)]
    assert finite.size > 0, "the valid rows should still come through"
    assert finite.min() > 100.0, f"a {finite.min():.1f} K pixel survived the gate"
    assert finite.max() < 400.0


def test_gridded_product_still_reads(tmp_path):
    """The JAXA P-Tree fallback path must not regress."""
    gridded = make_himawari_file(tmp_path, T0)
    grid = TargetGrid(
        lat_min=5.0, lat_max=25.0, lon_min=75.0, lon_max=95.0, resolution_deg=1.0
    )
    frame = HimawariReader().read(gridded, grid)
    assert frame.bt.shape == grid.shape
    finite = frame.bt[np.isfinite(frame.bt)]
    assert finite.size > 0 and finite.min() > 200.0


def test_no_intersecting_tile_yields_an_empty_frame_not_a_crash(tmp_path, caplog):
    import logging

    scan = make_isatss_scan(tmp_path, T0, n_tiles=2)
    far = TargetGrid(
        lat_min=10.0, lat_max=20.0, lon_min=-80.0, lon_max=-70.0, resolution_deg=1.0
    )
    with caplog.at_level(logging.WARNING, logger="sattsr.io.himawari"):
        frame = HimawariReader().read(scan[0], far)
    assert frame.bt.shape == far.shape
    assert not np.any(np.isfinite(frame.bt))


def test_projection_uses_the_isatss_attribute_spellings(tmp_path):
    """semi_major/semi_minor and microradian x/y -- getting these wrong misgeolocates."""
    import xarray as xr

    from sattsr.io.himawari import _MICRORAD, _isatss_projection

    tile = make_isatss_tile(tmp_path, T0)
    with xr.open_dataset(tile, mask_and_scale=True) as ds:
        proj = _isatss_projection(ds)
        raw_x = np.asarray(ds["x"].values, dtype=np.float64)

    assert proj.semi_major_axis == pytest.approx(6378137.0)
    assert proj.semi_minor_axis == pytest.approx(6356752.3)
    assert proj.sweep_axis == "y"
    assert proj.longitude_of_origin == pytest.approx(140.7)
    np.testing.assert_allclose(proj.x, raw_x * _MICRORAD, rtol=1e-9)
    assert abs(proj.x).max() < 1.0, "radians, not microradians"
