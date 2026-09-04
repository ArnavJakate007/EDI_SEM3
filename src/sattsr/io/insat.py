"""INSAT-3DR / 3DS TIR1 (10.8 um) HDF5 reader (MOSDAC L1B/L1C)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from sattsr.data.radiometry import apply_lut
from sattsr.geo.grid import TargetGrid
from sattsr.geo.resample import regrid_latlon, regrid_scattered
from sattsr.io.base import BaseReader, Frame

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )
}
_TIME_RE = re.compile(r"_(?P<day>\d{2})(?P<mon>[A-Z]{3})(?P<year>\d{4})_(?P<hhmm>\d{4})")

_COUNT_NAMES = ("IMG_TIR1",)
_LUT_NAMES = ("IMG_TIR1_TEMP",)


class InsatReader(BaseReader):
    """Reads TIR1 counts and converts them through the on-file temperature LUT.

    MOSDAC ships two geolocation styles. L1C (`_MER_`) carries 1-D `Latitude` /
    `Longitude`; L1B carries per-pixel 2-D arrays. Both are handled.
    """

    patterns = ("*.h5", "*.hdf5")

    def __init__(self, sensor: str = "insat3dr") -> None:
        self.sensor = sensor

    def timestamp_of(self, path: Path) -> datetime:
        """Parse `_DDMONYYYY_HHMM` out of the MOSDAC filename."""
        m = _TIME_RE.search(Path(path).name.upper())
        if m is None:
            raise ValueError(f"not a MOSDAC INSAT filename: {Path(path).name}")
        hhmm = m.group("hhmm")
        return datetime(
            int(m.group("year")),
            _MONTHS[m.group("mon")],
            int(m.group("day")),
            int(hhmm[:2]),
            int(hhmm[2:]),
            tzinfo=timezone.utc,
        )

    def read(self, path: Path, grid: TargetGrid) -> Frame:
        """Read TIR1, apply the LUT, and regrid onto `grid`."""
        path = Path(path)
        with h5py.File(path, "r") as f:
            counts = self._first(f, _COUNT_NAMES, path)
            lut = self._first(f, _LUT_NAMES, path)
            lats = np.asarray(f["Latitude"][:], dtype=np.float64).squeeze()
            lons = np.asarray(f["Longitude"][:], dtype=np.float64).squeeze()

        bt = apply_lut(counts, lut)
        bt = np.where(np.isfinite(bt) & (bt > 100.0) & (bt < 400.0), bt, np.nan)

        if lats.ndim == 1 and lons.ndim == 1:
            regridded = regrid_latlon(bt, lats, lons, grid)
        else:
            regridded = regrid_scattered(
                bt, lats, lons, grid, max_distance_deg=max(grid.resolution_deg * 2.0, 0.1)
            )

        return Frame(
            timestamp=self.timestamp_of(path),
            bt=regridded,
            sensor=self.sensor,
            source_path=path,
        )

    @staticmethod
    def _first(handle: h5py.File, names: tuple[str, ...], path: Path) -> np.ndarray:
        for name in names:
            if name in handle:
                return np.asarray(handle[name][:]).squeeze()
        raise ValueError(f"none of {names} present in {path.name}")
