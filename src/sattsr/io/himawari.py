"""Himawari-8/9 AHI Band 13 (10.4 um) readers.

Two products are supported, dispatched on the filename so the registry, the config
`Sensor` literal and the manifest all stay unchanged:

AHI-L2-FLDK-ISatSS (default; anonymous on s3://noaa-himawari8)
    `OR_HFD-020-B12-M1C13-T074_GH8_s2019182012000_c....nc`. Verified against a real
    downloaded file: the payload is `Sectorized_CMI`, int16 with scale_factor
    0.064208984375 and add_offset 69.0, whose `standard_name` is
    `brightness_temperature` and `units` is `kelvin` -- it is already BT, so there is
    no Planck inversion here. It carries NO `_FillValue`; off-disc pixels simply
    decode to physically impossible values (~70 K), which the shared 100-400 K gate
    below rejects.

    The grid is the same geostationary fixed grid as GOES ABI, but three details
    differ and all three will silently corrupt the geolocation if missed:
      * the projection variable is `fixedgrid_projection`, and its ellipsoid
        attributes are `semi_major` / `semi_minor`, not GOES's `semi_major_axis` /
        `semi_minor_axis`;
      * `sweep_angle_axis` is `y` (GOES ABI C13 uses `x`);
      * `x` and `y` are in MICRORADIAN, whereas `GeosProjection` expects radian.

    Each scan is split into tiles -- 76 or 88 depending on the slot, despite the
    file's own `number_product_tiles` attribute -- so one frame is a *set* of
    files rather than one file. `discover` groups them and returns a single canonical tile per
    timestamp, and `read` re-joins the siblings -- which keeps the "one manifest row
    is one frame" contract that the rest of the pipeline depends on.

JAXA P-Tree gridded (secondary; needs HIMAWARI_USER / HIMAWARI_PASS)
    `NC_H08_<YYYYMMDD>_<HHMM>_R21_FLDK...nc` carrying `tbb_13` already in Kelvin on a
    regular lat-lon grid. One file is one frame.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from sattsr.geo.grid import TargetGrid
from sattsr.geo.resample import GeosProjection, bilinear_sample, regrid_latlon
from sattsr.io.base import BaseReader, Frame

log = logging.getLogger(__name__)

_TIME_RE = re.compile(r"_(?P<date>\d{8})_(?P<time>\d{4})_")
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
_BT_NAMES = ("tbb_13", "tbb", "brightness_temperature")

#: ISatSS tile filename: channel `M1C13`, tile `T074`, start stamp `s2019182012000`.
#: The stamp is YYYYDDDHHMMSS, optionally followed by GOES ABI's tenths-of-a-second
#: digit -- BOTH spellings occur in the same archive (observed 760 files with 13
#: digits and 64 with 14 in one hour of 2019-07-01), so accept either and ignore the
#: fraction. Matching only 13 silently drops whole scans from the manifest.
#: Anchored on OR_HFD- (Himawari Full Disk). The same directories also carry
#: OR_HR3-* -- the Region-3 target-sector rapid scan: a single tile on a ~2.5 min
#: cadence covering a small box, NOT the full disk. It satisfies every other part
#: of this pattern, so without the anchor those frames would be mixed into the
#: full-disk series and triplets would be built across two different footprints.
_ISATSS_RE = re.compile(
    r"OR_HFD-\d{3}-B\d{2}-M\dC(?P<channel>\d{2})-T(?P<tile>\d{3})"
    r"_G(?P<sat>H\d+)_s(?P<stamp>\d{13,14})_"
)
#: Creation stamp `c2020183161821`, used to prefer a reprocessed tile over the original.
_CREATED_RE = re.compile(r"_c(?P<created>\d{13,14})\.nc$")
_ISATSS_BT_NAME = "Sectorized_CMI"
_ISATSS_PROJ_NAME = "fixedgrid_projection"
#: ISatSS x/y are microradian; GeosProjection wants radian.
_MICRORAD = 1e-6

BT_VALID_MIN, BT_VALID_MAX = 100.0, 400.0


def is_isatss(path: Path) -> bool:
    """True if `path` looks like an AHI-L2-FLDK-ISatSS tile."""
    return _ISATSS_RE.search(Path(path).name) is not None


def isatss_key(path: Path) -> tuple[str, str] | None:
    """`(channel, start_stamp)` identifying which frame a tile belongs to.

    The stamp is truncated to its 13 significant digits so that tiles of the SAME
    scan group together even when the archive spells some of them with the optional
    tenths digit and others without. Grouping on the raw string would split one scan
    into two half-frames.
    """
    m = _ISATSS_RE.search(Path(path).name)
    return (m.group("channel"), m.group("stamp")[:13]) if m else None


def _isatss_timestamp(stamp: str) -> datetime:
    """Parse `YYYYDDDHHMMSS` into UTC, ignoring any trailing tenths digit."""
    year, doy = int(stamp[0:4]), int(stamp[4:7])
    hh, mm, ss = int(stamp[7:9]), int(stamp[9:11]), int(stamp[11:13])
    base = datetime(year, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)


def _clip_physical(bt: np.ndarray) -> np.ndarray:
    """NaN out anything outside the physically plausible BT range."""
    finite = np.isfinite(bt) & (bt > BT_VALID_MIN) & (bt < BT_VALID_MAX)
    return np.where(finite, bt, np.nan)


def _isatss_projection(ds: xr.Dataset) -> GeosProjection:
    """Build a GeosProjection from an ISatSS tile's own coordinates."""
    attrs = ds[_ISATSS_PROJ_NAME].attrs
    return GeosProjection(
        satellite_height=float(attrs["perspective_point_height"]),
        longitude_of_origin=float(attrs["longitude_of_projection_origin"]),
        # ISatSS spells these without the `_axis` suffix GOES uses.
        semi_major_axis=float(attrs.get("semi_major", attrs.get("semi_major_axis"))),
        semi_minor_axis=float(attrs.get("semi_minor", attrs.get("semi_minor_axis"))),
        sweep_axis=str(attrs.get("sweep_angle_axis", "y")),
        x=np.asarray(ds["x"].values, dtype=np.float64) * _MICRORAD,
        y=np.asarray(ds["y"].values, dtype=np.float64) * _MICRORAD,
    )


class HimawariReader(BaseReader):
    """Reads either the ISatSS tiled product or the JAXA gridded product."""

    sensor = "himawari8"
    patterns = ("NC_H0*.nc", "*FLDK*.nc", "OR_HFD-*.nc")

    def __init__(self, channel: int = 13) -> None:
        #: Which AHI channel to keep out of the 16 present in an ISatSS directory.
        self.channel = f"{int(channel):02d}"

    # ------------------------------------------------------------------ discovery

    def timestamp_of(self, path: Path) -> datetime:
        """Parse the scan time from either product's filename."""
        name = Path(path).name
        m = _ISATSS_RE.search(name)
        if m is not None:
            return _isatss_timestamp(m.group("stamp"))

        m = _TIME_RE.search(name)
        if m is None:
            raise ValueError(f"not a Himawari filename: {name}")
        return datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M").replace(
            tzinfo=timezone.utc
        )

    def discover(self, root: Path) -> list[Path]:
        """Find frames, collapsing each ISatSS tile group to one canonical file.

        Without this every one of the 76 tiles would look like a separate frame, and
        the manifest, the cache and the triplet builder would all see 76 duplicate
        timestamps per scan.
        """
        root = Path(root)
        found: set[Path] = set()
        for pattern in self.patterns:
            found.update(p for p in root.rglob(pattern) if p.is_file())

        gridded: list[Path] = []
        groups: dict[tuple[str, str], list[Path]] = {}
        for path in found:
            key = isatss_key(path)
            if key is None:
                gridded.append(path)
                continue
            channel, _ = key
            if channel != self.channel:
                continue
            groups.setdefault(key, []).append(path)

        # The lowest-numbered tile is the stable representative of its scan.
        representatives = [min(paths, key=lambda p: p.name) for paths in groups.values()]

        dated: list[tuple[datetime, Path]] = []
        for path in [*gridded, *representatives]:
            try:
                dated.append((self.timestamp_of(path), path))
            except ValueError:
                continue
        return [p for _, p in sorted(dated, key=lambda kv: (kv[0], str(kv[1])))]

    def sibling_tiles(self, path: Path) -> list[Path]:
        """Every distinct tile of this scan, one file per tile number.

        The archive carries reprocessed copies of some tiles: the same scan and tile
        appears twice, once created in 2019 and once in 2020 (observed on 4 of 10
        slots, up to 24 duplicated tiles in one scan). Keeping both would mosaic the
        same footprint twice and leave which copy wins to filename sort order, so
        deduplicate on tile number and prefer the LATER creation stamp -- a
        reprocessed tile supersedes the original.
        """
        key = isatss_key(path)
        if key is None:
            return [Path(path)]
        parent = Path(path).parent

        best: dict[str, tuple[str, Path]] = {}
        for candidate in parent.glob("OR_HFD-*.nc"):
            if isatss_key(candidate) != key:
                continue
            m = _ISATSS_RE.search(candidate.name)
            tile = m.group("tile")
            created = _CREATED_RE.search(candidate.name)
            stamp = created.group("created")[:13] if created else ""
            if tile not in best or stamp > best[tile][0]:
                best[tile] = (stamp, candidate)

        return sorted(p for _, p in best.values())

    # --------------------------------------------------------------------- reading

    def read(self, path: Path, grid: TargetGrid) -> Frame:
        """Read one frame onto `grid`, mosaicking ISatSS tiles when needed."""
        path = Path(path)
        if is_isatss(path):
            bt = self._read_isatss(path, grid)
        else:
            bt = self._read_gridded(path, grid)
        return Frame(
            timestamp=self.timestamp_of(path),
            bt=bt,
            sensor=self.sensor,
            source_path=path,
        )

    def _read_gridded(self, path: Path, grid: TargetGrid) -> np.ndarray:
        """JAXA P-Tree gridded product: `tbb_13` on a regular lat-lon grid."""
        with xr.open_dataset(path, engine="netcdf4", mask_and_scale=True) as ds:
            var = next((n for n in _BT_NAMES if n in ds), None)
            if var is None:
                raise ValueError(f"no thermal band variable in {path.name}; have {list(ds)}")
            lat_name = next(n for n in _LAT_NAMES if n in ds.coords or n in ds)
            lon_name = next(n for n in _LON_NAMES if n in ds.coords or n in ds)
            bt = np.asarray(ds[var].values, dtype=np.float32).squeeze()
            lats = np.asarray(ds[lat_name].values, dtype=np.float64).squeeze()
            lons = np.asarray(ds[lon_name].values, dtype=np.float64).squeeze()

        return regrid_latlon(_clip_physical(bt), lats, lons, grid)

    def _read_isatss(self, path: Path, grid: TargetGrid) -> np.ndarray:
        """Regrid every tile of this scan onto `grid` and merge them.

        Each tile is regridded independently using its own x/y coordinates, so no
        mosaic arithmetic is needed: `bilinear_sample` already returns NaN outside a
        tile's own extent, and merging is just "first finite value wins". That also
        degrades gracefully when the archive is missing some tiles for a scan.
        """
        tiles = self.sibling_tiles(path)
        out = np.full(grid.shape, np.nan, dtype=np.float32)
        used = 0
        angles: tuple[np.ndarray, np.ndarray] | None = None

        for tile_path in tiles:
            try:
                with xr.open_dataset(tile_path, engine="netcdf4", mask_and_scale=True) as ds:
                    if _ISATSS_BT_NAME not in ds:
                        raise ValueError(
                            f"no {_ISATSS_BT_NAME} in {tile_path.name}; have {list(ds)}"
                        )
                    proj = _isatss_projection(ds)
                    bt = np.asarray(ds[_ISATSS_BT_NAME].values, dtype=np.float32).squeeze()
            except (OSError, ValueError, KeyError) as exc:
                log.warning("skipping tile %s: %s", tile_path.name, exc)
                continue

            # Every tile shares one projection, so the pyproj transform is done once
            # per frame rather than 76 times.
            if angles is None:
                angles = proj.latlon_to_angle(*grid.meshgrid())
            rows, cols = proj.angle_to_index(*angles)

            # Cheap rejection: most of the 76 tiles cover no part of the target grid.
            if not np.any(
                (rows > -1) & (rows < bt.shape[0]) & (cols > -1) & (cols < bt.shape[1])
            ):
                continue

            sampled = bilinear_sample(_clip_physical(bt), rows, cols)
            fill = np.isnan(out) & np.isfinite(sampled)
            out[fill] = sampled[fill]
            used += 1

        if used == 0:
            log.warning(
                "no ISatSS tile of %s intersects the target grid (%.1f..%.1f N, "
                "%.1f..%.1f E)", path.name, grid.lat_min, grid.lat_max,
                grid.lon_min, grid.lon_max,
            )
        else:
            log.debug("mosaicked %d of %d tile(s) for %s", used, len(tiles), path.name)
        return out
