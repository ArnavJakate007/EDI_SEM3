"""Frame discovery, on-disk index, and the regridded frame cache."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sattsr.geo.grid import TargetGrid
from sattsr.io.base import Reader

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameRef:
    """A pointer to one frame: when it was taken and where the bytes live."""

    timestamp: datetime
    path: Path
    sensor: str


def build_index(reader: Reader, root: str | Path) -> list[FrameRef]:
    """Scan `root` for this reader's files and return timestamp-sorted references."""
    refs: list[FrameRef] = []
    for path in reader.discover(Path(root)):
        try:
            refs.append(FrameRef(reader.timestamp_of(path), Path(path), reader.sensor))
        except ValueError:
            log.debug("skipping unparseable file %s", path)
    return sorted(refs, key=lambda r: (r.timestamp, str(r.path)))


def save_index(refs: Sequence[FrameRef], path: str | Path) -> Path:
    """Write an index as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "timestamp": r.timestamp.astimezone(timezone.utc).isoformat(),
            "path": str(r.path),
            "sensor": r.sensor,
        }
        for r in refs
    ]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_index(path: str | Path) -> list[FrameRef]:
    """Read an index written by `save_index`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        FrameRef(datetime.fromisoformat(e["timestamp"]), Path(e["path"]), e["sensor"])
        for e in raw
    ]


def cache_path_for(cache_root: str | Path, ref: FrameRef) -> Path:
    """`<cache_root>/<sensor>/<YYYYMMDD>/<HHMMSS>.npy`.

    A `cache_root` that already ends in the sensor name -- as `configs/goes19.yaml`
    does with `cache/goes19` -- keeps that segment rather than gaining a second one.
    """
    ts = ref.timestamp.astimezone(timezone.utc)
    root = Path(cache_root)
    if root.name != ref.sensor:
        root = root / ref.sensor
    return root / ts.strftime("%Y%m%d") / f"{ts.strftime('%H%M%S')}.npy"


def prepare_cache(
    reader: Reader,
    refs: Iterable[FrameRef],
    grid: TargetGrid,
    cache_root: str | Path,
    *,
    overwrite: bool = False,
    progress: bool = False,
) -> list[FrameRef]:
    """Regrid each referenced file once and cache it as float32 Kelvin `.npy`.

    Files that fail to read are logged and skipped rather than aborting the run --
    partial archive downloads are the norm.
    """
    items: Iterable[FrameRef] = list(refs)
    if progress:
        from tqdm import tqdm

        items = tqdm(list(items), desc="preparing cache")

    out: list[FrameRef] = []
    for ref in items:
        dest = cache_path_for(cache_root, ref)
        if dest.exists() and not overwrite:
            out.append(FrameRef(ref.timestamp, dest, ref.sensor))
            continue
        try:
            frame = reader.read(ref.path, grid)
        except (OSError, ValueError, KeyError) as exc:
            log.warning("skipping %s: %s", ref.path, exc)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, np.asarray(frame.bt, dtype=np.float32))
        out.append(FrameRef(ref.timestamp, dest, ref.sensor))
    return out


def load_cached(ref: FrameRef) -> np.ndarray:
    """Load a cached frame as float32 Kelvin."""
    return np.load(ref.path).astype(np.float32, copy=False)


def timestamp_from_cache_path(path: str | Path) -> datetime:
    """Inverse of `cache_path_for`: recover a frame's timestamp from its cache path.

    Expects `.../<YYYYMMDD>/<HHMMSS>.npy`.
    """
    p = Path(path)
    try:
        stamp = datetime.strptime(f"{p.parent.name}{p.stem}", "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ValueError(f"not a cache frame path: {p}") from exc
    return stamp.replace(tzinfo=timezone.utc)


def scan_cache(cache_root: str | Path, sensor: str) -> list[FrameRef]:
    """Recover frame references by walking a cache tree.

    Prefers nothing: this is the fallback for when `index.json` is absent or stale.
    Unparseable paths are skipped with a debug log rather than raising.
    """
    root = Path(cache_root)
    refs: list[FrameRef] = []
    for path in sorted(root.rglob("*.npy")):
        try:
            refs.append(FrameRef(timestamp_from_cache_path(path), path, sensor))
        except ValueError:
            log.debug("skipping non-frame cache file %s", path)
    return sorted(refs, key=lambda r: (r.timestamp, str(r.path)))


def load_or_scan_index(cache_root: str | Path, sensor: str) -> list[FrameRef]:
    """Frame refs from `<cache_root>/index.json`, falling back to a cache walk."""
    index_path = Path(cache_root) / "index.json"
    if index_path.exists():
        refs = [r for r in load_index(index_path) if r.sensor == sensor]
        if refs:
            return refs
        log.warning("%s has no %s entries; falling back to a cache scan", index_path, sensor)
    return scan_cache(cache_root, sensor)
