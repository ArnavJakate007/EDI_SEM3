"""CF-compliant NetCDF read/write for frame products.

Every product this project emits goes through here, so the synthetic-frame flag and
the "not an observation" warning are impossible to omit.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sattsr.geo.grid import TargetGrid
from sattsr.io.base import Frame

SYNTHETIC_WARNING = (
    "Frames with synthetic=1 are AI-generated and are NOT observations. "
    "They are decision-support augmentation only."
)


def write_frames_nc(
    path: str | Path,
    frames: Sequence[Frame],
    grid: TargetGrid,
    *,
    synthetic: Sequence[bool],
    model_version: str,
    extra_attrs: dict[str, object] | None = None,
) -> Path:
    """Write a time series of frames as a CF-1.8 NetCDF file."""
    if len(frames) == 0:
        raise ValueError("cannot write an empty frame series")
    if len(synthetic) != len(frames):
        raise ValueError(
            f"synthetic has {len(synthetic)} entries but there are {len(frames)} frames"
        )

    data = np.stack([np.asarray(f.bt, dtype=np.float32) for f in frames])
    if data.shape[1:] != grid.shape:
        raise ValueError(f"frame shape {data.shape[1:]} does not match grid {grid.shape}")

    times = pd.to_datetime(
        [f.timestamp.astimezone(timezone.utc).replace(tzinfo=None) for f in frames]
    )

    ds = xr.Dataset(
        {
            "brightness_temperature": (
                ("time", "lat", "lon"),
                data,
                {
                    "units": "K",
                    "long_name": "Thermal infrared brightness temperature (~10 um)",
                    "standard_name": "toa_brightness_temperature",
                },
            ),
            "synthetic": (
                ("time",),
                np.asarray(synthetic, dtype=np.int8),
                {
                    "long_name": "1 if the frame was synthesized by the model, 0 if observed",
                    "flag_values": np.array([0, 1], dtype=np.int8),
                    "flag_meanings": "observed synthesized",
                },
            ),
        },
        coords={
            "time": times,
            "lat": ("lat", grid.lats, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", grid.lons, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "Temporally super-resolved geostationary thermal imagery",
            "source": frames[0].sensor,
            "model_version": model_version,
            "comment": SYNTHETIC_WARNING,
            "history": f"{datetime.now(timezone.utc).isoformat()} created by sattsr",
        },
    )
    if extra_attrs:
        ds.attrs.update({k: str(v) for k, v in extra_attrs.items()})

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        "brightness_temperature": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
        "time": {"units": "seconds since 1970-01-01T00:00:00", "dtype": "int64"},
    }
    ds.to_netcdf(out, engine="netcdf4", encoding=encoding)
    return out


def read_frames_nc(
    path: str | Path, *, sensor: str | None = None
) -> tuple[list[Frame], list[bool]]:
    """Read a frame-product NetCDF back into Frames plus their synthetic flags."""
    p = Path(path)
    with xr.open_dataset(p, engine="netcdf4", mask_and_scale=True) as ds:
        data = np.asarray(ds["brightness_temperature"].values, dtype=np.float32)
        times = [
            ts.to_pydatetime() for ts in pd.to_datetime(np.asarray(ds["time"].values))
        ]
        flags = (
            [bool(v) for v in np.asarray(ds["synthetic"].values)]
            if "synthetic" in ds
            else [False] * data.shape[0]
        )
        src = sensor or str(ds.attrs.get("source", "unknown"))

    frames = [
        Frame(timestamp=t.replace(tzinfo=timezone.utc), bt=data[i], sensor=src, source_path=p)
        for i, t in enumerate(times)
    ]
    return frames, flags
