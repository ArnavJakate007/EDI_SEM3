"""The single common lat-lon grid every sensor is resampled onto."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sattsr.config import GridConfig


@dataclass(frozen=True)
class TargetGrid:
    """Regular lat-lon grid. Cell centres; latitude descending, longitude ascending."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution_deg: float

    def __post_init__(self) -> None:
        if self.lat_max <= self.lat_min or self.lon_max <= self.lon_min:
            raise ValueError("grid extent must be positive in both dimensions")
        if self.resolution_deg <= 0:
            raise ValueError("resolution_deg must be positive")
        if self.shape[0] < 1 or self.shape[1] < 1:
            raise ValueError("grid resolution is coarser than the requested extent")

    @cached_property
    def shape(self) -> tuple[int, int]:
        """(height, width) in pixels."""
        h = int(round((self.lat_max - self.lat_min) / self.resolution_deg))
        w = int(round((self.lon_max - self.lon_min) / self.resolution_deg))
        return h, w

    @cached_property
    def lats(self) -> np.ndarray:
        """Cell-centre latitudes, descending, shape (H,)."""
        h = self.shape[0]
        return (self.lat_max - (np.arange(h) + 0.5) * self.resolution_deg).astype(np.float64)

    @cached_property
    def lons(self) -> np.ndarray:
        """Cell-centre longitudes, ascending, shape (W,)."""
        w = self.shape[1]
        return (self.lon_min + (np.arange(w) + 0.5) * self.resolution_deg).astype(np.float64)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat2d, lon2d), each (H, W)."""
        return np.meshgrid(self.lats, self.lons, indexing="ij")

    @classmethod
    def from_config(cls, cfg: GridConfig) -> TargetGrid:
        """Build a grid from a validated GridConfig."""
        return cls(
            lat_min=cfg.lat_min,
            lat_max=cfg.lat_max,
            lon_min=cfg.lon_min,
            lon_max=cfg.lon_max,
            resolution_deg=cfg.resolution_deg,
        )
