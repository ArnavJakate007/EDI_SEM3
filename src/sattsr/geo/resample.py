"""Resampling of sensor-native grids onto the common TargetGrid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyproj
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from sattsr.geo.grid import TargetGrid


def bilinear_sample(src: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinearly sample `src` at fractional (row, col); NaN where out of bounds."""
    a = np.asarray(src, dtype=np.float64)
    h, w = a.shape
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)

    finite = np.isfinite(r) & np.isfinite(c)
    r_s = np.where(finite, r, -1.0)
    c_s = np.where(finite, c, -1.0)

    # Validity is about the sample position, not the upper neighbour: a position
    # exactly on the last row or column is in bounds, and its (clamped) upper
    # neighbour contributes zero weight.
    valid = finite & (r_s >= 0.0) & (r_s <= h - 1) & (c_s >= 0.0) & (c_s <= w - 1)

    r0 = np.floor(r_s).astype(np.int64)
    c0 = np.floor(c_s).astype(np.int64)
    r1, c1 = r0 + 1, c0 + 1

    r0c, r1c = np.clip(r0, 0, h - 1), np.clip(r1, 0, h - 1)
    c0c, c1c = np.clip(c0, 0, w - 1), np.clip(c1, 0, w - 1)
    dr = np.where(valid, r_s - r0, 0.0)
    dc = np.where(valid, c_s - c0, 0.0)

    top = a[r0c, c0c] * (1.0 - dc) + a[r0c, c1c] * dc
    bot = a[r1c, c0c] * (1.0 - dc) + a[r1c, c1c] * dc
    out = top * (1.0 - dr) + bot * dr
    return np.where(valid, out, np.nan).astype(np.float32)


@dataclass(frozen=True)
class GeosProjection:
    """Geostationary fixed-grid projection parameters read from a sensor file."""

    satellite_height: float
    longitude_of_origin: float
    semi_major_axis: float
    semi_minor_axis: float
    sweep_axis: str
    x: np.ndarray          # scan angle along columns, radians, shape (W,)
    y: np.ndarray          # scan angle along rows, radians, shape (H,)

    def _crs(self) -> pyproj.CRS:
        return pyproj.CRS.from_cf(
            {
                "grid_mapping_name": "geostationary",
                "perspective_point_height": self.satellite_height,
                "longitude_of_projection_origin": self.longitude_of_origin,
                "latitude_of_projection_origin": 0.0,
                "semi_major_axis": self.semi_major_axis,
                "semi_minor_axis": self.semi_minor_axis,
                "sweep_angle_axis": self.sweep_axis,
            }
        )

    def latlon_to_angle(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map geographic coordinates to (y, x) scan angles in radians.

        Split out from `latlon_to_index` because this half depends only on the
        projection *parameters*, not on the x/y arrays. A tiled product whose tiles
        all share one projection can therefore pay for the pyproj transform once
        instead of once per tile.
        """
        tf = pyproj.Transformer.from_crs("EPSG:4326", self._crs(), always_xy=True)
        px, py = tf.transform(np.asarray(lon, float), np.asarray(lat, float))
        with np.errstate(invalid="ignore", divide="ignore"):
            sx = np.asarray(px, float) / self.satellite_height
            sy = np.asarray(py, float) / self.satellite_height
        # pyproj returns +/-inf for points that do not project (off the visible disc)
        sx = np.where(np.isfinite(sx) & (np.abs(sx) < 1e30), sx, np.nan)
        sy = np.where(np.isfinite(sy) & (np.abs(sy) < 1e30), sy, np.nan)
        return sy, sx

    def angle_to_index(self, sy: np.ndarray, sx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert scan angles to fractional (row, col) in this array's own grid."""
        dx = float(self.x[1] - self.x[0])
        dy = float(self.y[1] - self.y[0])
        cols = (sx - float(self.x[0])) / dx
        rows = (sy - float(self.y[0])) / dy
        return rows, cols

    def latlon_to_index(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map geographic coordinates to fractional (row, col) in the sensor array."""
        sy, sx = self.latlon_to_angle(lat, lon)
        return self.angle_to_index(sy, sx)


def regrid_geos(src: np.ndarray, proj: GeosProjection, grid: TargetGrid) -> np.ndarray:
    """Resample a geostationary fixed-grid array onto the target lat-lon grid."""
    lat2d, lon2d = grid.meshgrid()
    rows, cols = proj.latlon_to_index(lat2d, lon2d)
    return bilinear_sample(src, rows, cols)


def regrid_latlon(
    src: np.ndarray, src_lats: np.ndarray, src_lons: np.ndarray, grid: TargetGrid
) -> np.ndarray:
    """Resample an array already on a regular lat-lon grid; either axis may be descending."""
    lat = np.asarray(src_lats, dtype=np.float64)
    lon = np.asarray(src_lons, dtype=np.float64)
    a = np.asarray(src, dtype=np.float64)
    if lat.size > 1 and lat[0] > lat[-1]:
        lat, a = lat[::-1], a[::-1, :]
    if lon.size > 1 and lon[0] > lon[-1]:
        lon, a = lon[::-1], a[:, ::-1]

    interp = RegularGridInterpolator(
        (lat, lon), a, method="linear", bounds_error=False, fill_value=np.nan
    )
    lat2d, lon2d = grid.meshgrid()
    return np.asarray(interp(np.stack([lat2d, lon2d], axis=-1)), dtype=np.float32)


def regrid_scattered(
    src: np.ndarray,
    src_lat2d: np.ndarray,
    src_lon2d: np.ndarray,
    grid: TargetGrid,
    *,
    max_distance_deg: float = 0.1,
) -> np.ndarray:
    """Nearest-neighbour resample of a curvilinear/scattered grid.

    Distances are Euclidean in degrees, which is adequate for a regional grid but
    degrades near the poles. Targets further than `max_distance_deg` become NaN.
    """
    valid = np.isfinite(src) & np.isfinite(src_lat2d) & np.isfinite(src_lon2d)
    if not valid.any():
        return np.full(grid.shape, np.nan, dtype=np.float32)

    pts = np.column_stack([src_lat2d[valid].ravel(), src_lon2d[valid].ravel()])
    vals = np.asarray(src)[valid].ravel().astype(np.float32)
    tree = cKDTree(pts)

    lat2d, lon2d = grid.meshgrid()
    query = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    dist, idx = tree.query(query, k=1, distance_upper_bound=float(max_distance_deg))

    out = np.full(query.shape[0], np.nan, dtype=np.float32)
    hit = np.isfinite(dist) & (idx < vals.size)
    out[hit] = vals[idx[hit]]
    return out.reshape(grid.shape)


def coverage_fraction(frame: np.ndarray) -> float:
    """Fraction of `frame` carrying a usable observation, in [0, 1].

    Regridding onto a fixed lat-lon target leaves NaN wherever the target cell falls
    outside the sensor's disc or the source pixel was flagged. A frame that is mostly
    off-disc is worse than useless for training -- it contributes a tile of zeros and
    a mask of zeros -- so preprocessing gates on this before caching.
    """
    a = np.asarray(frame)
    if a.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(a)) / a.size)
