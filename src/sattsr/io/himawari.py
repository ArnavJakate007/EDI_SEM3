"""Himawari-8/9 AHI Band 13 (10.4 um) gridded reader."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from sattsr.geo.grid import TargetGrid
from sattsr.geo.resample import regrid_latlon
from sattsr.io.base import BaseReader, Frame

_TIME_RE = re.compile(r"_(?P<date>\d{8})_(?P<time>\d{4})_")
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
_BT_NAMES = ("tbb_13", "tbb", "brightness_temperature")


class HimawariReader(BaseReader):
    """Reads the JAXA gridded AHI product, whose `tbb_13` is already in Kelvin."""

    sensor = "himawari8"
    patterns = ("NC_H0*.nc", "*FLDK*.nc")

    def timestamp_of(self, path: Path) -> datetime:
        """Parse `..._YYYYMMDD_HHMM_...` out of the filename."""
        m = _TIME_RE.search(Path(path).name)
        if m is None:
            raise ValueError(f"not a Himawari filename: {Path(path).name}")
        return datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M").replace(
            tzinfo=timezone.utc
        )

    def read(self, path: Path, grid: TargetGrid) -> Frame:
        """Read `tbb_13` and regrid onto `grid`."""
        path = Path(path)
        with xr.open_dataset(path, engine="netcdf4", mask_and_scale=True) as ds:
            var = next((n for n in _BT_NAMES if n in ds), None)
            if var is None:
                raise ValueError(f"no thermal band variable in {path.name}; have {list(ds)}")
            lat_name = next(n for n in _LAT_NAMES if n in ds.coords or n in ds)
            lon_name = next(n for n in _LON_NAMES if n in ds.coords or n in ds)
            bt = np.asarray(ds[var].values, dtype=np.float32).squeeze()
            lats = np.asarray(ds[lat_name].values, dtype=np.float64).squeeze()
            lons = np.asarray(ds[lon_name].values, dtype=np.float64).squeeze()

        bt = np.where(np.isfinite(bt) & (bt > 100.0) & (bt < 400.0), bt, np.nan)
        return Frame(
            timestamp=self.timestamp_of(path),
            bt=regrid_latlon(bt, lats, lons, grid),
            sensor=self.sensor,
            source_path=path,
        )
