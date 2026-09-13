"""End-to-end inference: sensor frames in, flagged NetCDF and a run manifest out.

Input can come from raw sensor files, or from the regridded `.npy` cache. The cache
path matters because `--delete-raw-after-cache` removes raw files as soon as their
frames are cached, so on a disk-constrained machine the cache is usually the only
copy left -- and it is the faster path anyway, since the frames are already on the
target grid and need no reader or regridding.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sattsr import __version__
from sattsr.config import Config
from sattsr.data.index import build_index, load_or_scan_index
from sattsr.geo.grid import TargetGrid
from sattsr.infer.recursive import interpolate_sequence
from sattsr.io.base import Frame
from sattsr.io.registry import get_reader
from sattsr.io.writer import write_frames_nc
from sattsr.models.interpolator import build_model
from sattsr.train.checkpoint import load_checkpoint

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunManifest:
    """Everything the dashboard needs to describe one inference run."""

    run_id: str
    sensor: str
    created_at: datetime
    input_cadence_minutes: float
    output_cadence_minutes: float
    factor: int
    n_original: int
    n_synthetic: int
    output_nc: str
    frames: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: str = ""
    model_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        payload = asdict(self)
        payload["created_at"] = self.created_at.astimezone(timezone.utc).isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunManifest:
        """Inverse of `to_dict`."""
        data = dict(payload)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


def write_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    """Write `manifest.json` into a run directory."""
    out = Path(run_dir) / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return out


def read_manifest(run_dir: str | Path) -> RunManifest:
    """Read `manifest.json` from a run directory."""
    path = Path(run_dir) / "manifest.json"
    return RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def frames_from_cache(
    cache_root: str | Path, sensor: str, *, limit: int | None = None
) -> list[Frame]:
    """Load already-regridded frames straight from the `.npy` cache.

    The cached array is the regridded field itself, so this skips the reader and the
    resampling entirely. It is also the only option once the raw granules have been
    reclaimed by --delete-raw-after-cache.
    """
    refs = load_or_scan_index(cache_root, sensor)
    if limit is not None:
        refs = refs[:limit]

    frames: list[Frame] = []
    for ref in refs:
        try:
            bt = np.load(ref.path).astype(np.float32, copy=False)
        except (OSError, ValueError) as exc:
            log.warning("skipping unreadable cache frame %s: %s", ref.path.name, exc)
            continue
        frames.append(
            Frame(timestamp=ref.timestamp, bt=bt, sensor=sensor, source_path=ref.path)
        )
    return frames


def run_inference(
    config: Config,
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path,
    factor: int,
    device: torch.device,
    limit: int | None = None,
    run_id: str | None = None,
    from_cache: bool = False,
) -> RunManifest:
    """Read a series of sensor frames, densify it, and write a flagged NetCDF product.

    `from_cache=True` sources the regridded `.npy` cache instead of raw granules.
    """
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = TargetGrid.from_config(config.data.grid)

    if from_cache:
        originals = frames_from_cache(input_dir, config.data.sensor, limit=limit)
        source_desc = f"cache {input_dir}"
    else:
        reader = get_reader(config.data.sensor)
        refs = build_index(reader, input_dir)
        if limit is not None:
            refs = refs[:limit]
        originals = [reader.read(ref.path, grid) for ref in refs]
        source_desc = f"raw {input_dir}"

    if len(originals) < 2:
        raise ValueError(
            f"need at least two input frames in {source_desc}, found {len(originals)}"
        )
    log.info("read %d frames from %s", len(originals), source_desc)

    model = build_model(config.model)
    load_checkpoint(checkpoint, model, map_location=str(device))
    model.eval().to(device)

    frames, flags = interpolate_sequence(
        originals, model,
        factor=factor, device=device, norm=config.data.normalization,
        tile_size=config.train.tile_size, tile_overlap=config.train.tile_overlap,
    )

    output_nc = run_dir / "output.nc"
    write_frames_nc(
        output_nc, frames, grid,
        synthetic=flags,
        model_version=__version__,
        extra_attrs={
            "checkpoint": str(checkpoint),
            "interpolation_factor": factor,
            "input_cadence_minutes": config.data.cadence_minutes,
        },
    )

    manifest = RunManifest(
        run_id=run_id or run_dir.name,
        sensor=config.data.sensor,
        created_at=datetime.now(timezone.utc),
        input_cadence_minutes=float(config.data.cadence_minutes),
        output_cadence_minutes=float(config.data.cadence_minutes) / factor,
        factor=int(factor),
        n_original=int(sum(1 for f in flags if not f)),
        n_synthetic=int(sum(flags)),
        output_nc=output_nc.name,
        frames=[
            {
                "index": i,
                "timestamp": frame.timestamp.astimezone(timezone.utc).isoformat(),
                "synthetic": bool(flag),
            }
            for i, (frame, flag) in enumerate(zip(frames, flags, strict=True))
        ],
        checkpoint=str(checkpoint),
    )
    write_manifest(run_dir, manifest)
    return manifest
