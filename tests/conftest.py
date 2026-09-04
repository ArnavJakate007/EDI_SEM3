from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

FK1, FK2, BC1, BC2 = 1.03413e4, 1.39177e3, 0.20351, 0.99958
SAT_HEIGHT = 35786023.0

INSAT_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _bt_to_radiance(bt: np.ndarray) -> np.ndarray:
    """Forward Planck, so fixtures round-trip through the reader's inversion."""
    return FK1 / (np.exp(FK2 / (BC1 + BC2 * bt)) - 1.0)


def make_goes_file(
    path: Path,
    timestamp: datetime,
    *,
    size: int = 64,
    lon_origin: float = 82.0,
    value_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Path:
    """Write a minimal ABI-L1b-like NetCDF whose filename encodes `timestamp`."""
    ang = np.linspace(-0.06, 0.06, size)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    bt = value_fn(yy, xx) if value_fn else (230.0 + 0.5 * xx + 0.2 * yy).astype(np.float64)

    ds = xr.Dataset(
        {
            "Rad": (("y", "x"), _bt_to_radiance(np.asarray(bt, float)).astype(np.float32)),
            "planck_fk1": ((), np.float32(FK1)),
            "planck_fk2": ((), np.float32(FK2)),
            "planck_bc1": ((), np.float32(BC1)),
            "planck_bc2": ((), np.float32(BC2)),
            "goes_imager_projection": ((), np.int8(0), {
                "grid_mapping_name": "geostationary",
                "perspective_point_height": SAT_HEIGHT,
                "longitude_of_projection_origin": lon_origin,
                "latitude_of_projection_origin": 0.0,
                "semi_major_axis": 6378137.0,
                "semi_minor_axis": 6356752.31414,
                "sweep_angle_axis": "x",
            }),
        },
        coords={"x": ("x", ang, {"units": "rad"}), "y": ("y", -ang, {"units": "rad"})},
    )
    doy = timestamp.timetuple().tm_yday
    stamp = f"{timestamp.year}{doy:03d}{timestamp:%H%M%S}0"
    name = f"OR_ABI-L1b-RadF-M6C13_G19_s{stamp}_e{stamp}_c{stamp}.nc"
    out = path / name
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out


def make_goes_series(root: Path, n: int, *, step_minutes: int = 10, size: int = 64) -> list[Path]:
    """A cadence-regular series whose pattern translates one pixel per step."""
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    paths = []
    for i in range(n):
        paths.append(
            make_goes_file(
                root,
                t0 + timedelta(minutes=step_minutes * i),
                size=size,
                value_fn=lambda yy, xx, s=i: 230.0 + 0.5 * ((xx + s) % size) + 0.2 * yy,
            )
        )
    return paths


def make_himawari_file(path: Path, timestamp: datetime, *, size: int = 32) -> Path:
    """Write a JAXA-style gridded AHI NetCDF with a `tbb_13` variable in Kelvin."""
    lats = np.linspace(30.0, 0.0, size)          # descending
    lons = np.linspace(70.0, 100.0, size)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    tbb = (240.0 + 0.4 * xx + 0.3 * yy).astype(np.float32)
    ds = xr.Dataset(
        {"tbb_13": (("latitude", "longitude"), tbb, {"units": "K"})},
        coords={"latitude": lats, "longitude": lons},
    )
    out = path / f"NC_H08_{timestamp:%Y%m%d_%H%M}_R21_FLDK.02401_02401.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out


def make_insat_file(
    path: Path, timestamp: datetime, *, size: int = 32, curvilinear: bool = False
) -> Path:
    """Write a MOSDAC-style INSAT L1C HDF5 with IMG_TIR1 counts and a temperature LUT."""
    lut = np.linspace(150.0, 350.0, 1024).astype(np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    counts = ((400 + 3 * xx + 2 * yy) % 1024).astype(np.uint16)
    lats = np.linspace(35.0, 5.0, size).astype(np.float32)
    lons = np.linspace(65.0, 95.0, size).astype(np.float32)

    month = INSAT_MONTHS[timestamp.month - 1]
    out = path / (
        f"3DIMG_{timestamp.day:02d}{month}{timestamp.year}_{timestamp:%H%M}_L1C_ASIA_MER.h5"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("IMG_TIR1", data=counts[None, ...])
        f.create_dataset("IMG_TIR1_TEMP", data=lut)
        if curvilinear:
            lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
            f.create_dataset("Latitude", data=lat2d)
            f.create_dataset("Longitude", data=lon2d)
        else:
            f.create_dataset("Latitude", data=lats)
            f.create_dataset("Longitude", data=lons)
    return out


@pytest.fixture
def goes_dir(tmp_path: Path) -> Path:
    """Directory containing five 10-minute GOES-like frames."""
    root = tmp_path / "goes"
    make_goes_series(root, 5)
    return root
