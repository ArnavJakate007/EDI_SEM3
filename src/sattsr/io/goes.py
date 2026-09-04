"""GOES-19 ABI Channel 13 (10.3 um) reader."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from sattsr.data.radiometry import planck_radiance_to_bt
from sattsr.geo.grid import TargetGrid
from sattsr.geo.resample import GeosProjection, regrid_geos
from sattsr.io.base import BaseReader, Frame

_START_RE = re.compile(r"_s(?P<stamp>\d{14})_")


class Goes19Reader(BaseReader):
    """Reads ABI L1b `Rad` (or L2 `CMI`) and returns Kelvin on the target grid."""

    sensor = "goes19"
    patterns = ("*C13_G19*.nc", "*C13_G19*.nc4")

    def timestamp_of(self, path: Path) -> datetime:
        """Parse the scan start time out of the ABI filename (YYYYDDDHHMMSSm)."""
        m = _START_RE.search(Path(path).name)
        if m is None:
            raise ValueError(f"not an ABI filename: {Path(path).name}")
        s = m.group("stamp")
        year, doy = int(s[0:4]), int(s[4:7])
        hh, mm, ss = int(s[7:9]), int(s[9:11]), int(s[11:13])
        base = datetime(year, 1, 1, tzinfo=timezone.utc)
        return base + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)

    def read(self, path: Path, grid: TargetGrid) -> Frame:
        """Read one ABI file, convert to Kelvin, and regrid onto `grid`."""
        path = Path(path)
        with xr.open_dataset(path, engine="netcdf4", mask_and_scale=True) as ds:
            if "CMI" in ds:
                bt = np.asarray(ds["CMI"].values, dtype=np.float32)
            else:
                bt = planck_radiance_to_bt(
                    np.asarray(ds["Rad"].values),
                    float(ds["planck_fk1"]),
                    float(ds["planck_fk2"]),
                    float(ds["planck_bc1"]),
                    float(ds["planck_bc2"]),
                )
            attrs = ds["goes_imager_projection"].attrs
            proj = GeosProjection(
                satellite_height=float(attrs["perspective_point_height"]),
                longitude_of_origin=float(attrs["longitude_of_projection_origin"]),
                semi_major_axis=float(attrs["semi_major_axis"]),
                semi_minor_axis=float(attrs["semi_minor_axis"]),
                sweep_axis=str(attrs.get("sweep_angle_axis", "x")),
                x=np.asarray(ds["x"].values, dtype=np.float64),
                y=np.asarray(ds["y"].values, dtype=np.float64),
            )

        bt = np.where(np.isfinite(bt) & (bt > 100.0) & (bt < 400.0), bt, np.nan)
        return Frame(
            timestamp=self.timestamp_of(path),
            bt=regrid_geos(bt, proj, grid),
            sensor=self.sensor,
            source_path=path,
        )
