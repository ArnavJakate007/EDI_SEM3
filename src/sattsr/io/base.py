"""Frame container and the reader protocol every sensor implements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from sattsr.geo.grid import TargetGrid


@dataclass(frozen=True)
class Frame:
    """One observed or synthesized thermal-IR frame on the common grid.

    `bt` is float32 brightness temperature in Kelvin, shape (H, W), NaN where invalid.
    """

    timestamp: datetime
    bt: np.ndarray
    sensor: str
    source_path: Path


@runtime_checkable
class Reader(Protocol):
    """Reads one sensor's native files onto the common grid."""

    sensor: str
    patterns: tuple[str, ...]

    def timestamp_of(self, path: Path) -> datetime: ...
    def read(self, path: Path, grid: TargetGrid) -> Frame: ...
    def discover(self, root: Path) -> list[Path]: ...


class BaseReader:
    """Shared file discovery. Subclasses set `sensor` and `patterns`."""

    sensor: str = "unknown"
    patterns: tuple[str, ...] = ()

    def discover(self, root: Path) -> list[Path]:
        """Recursively find every file matching `patterns`, sorted by timestamp."""
        root = Path(root)
        found: set[Path] = set()
        for pattern in self.patterns:
            found.update(root.rglob(pattern))

        dated: list[tuple[datetime, Path]] = []
        for p in found:
            try:
                dated.append((self.timestamp_of(p), p))   # type: ignore[attr-defined]
            except ValueError:
                continue
        return [p for _, p in sorted(dated, key=lambda kv: (kv[0], str(kv[1])))]
