"""Manifest-driven, parallel frame preparation into the regridded `.npy` cache.

`sattsr.data.index.prepare_cache` already does the serial, index-driven version of
this. This module is the manifest-driven path used by `scripts/preprocess.py`: it
adds a coverage gate and parallelism, and reuses `cache_path_for` so both paths write
to exactly the same cache layout.

Parallelism uses `concurrent.futures.ProcessPoolExecutor` rather than
`multiprocessing.Pool`: regridding is CPU-bound inside scipy/numpy (the bottleneck
called out in `geo/resample.py`), so it needs processes rather than threads to escape
the GIL, and futures propagate a worker exception attached to the task that raised it,
which matters when one corrupt file in a thousand must be logged by name and skipped.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from sattsr.config import GridConfig
from sattsr.data.index import FrameRef, cache_path_for
from sattsr.geo.grid import TargetGrid
from sattsr.geo.resample import coverage_fraction
from sattsr.io.manifest import ManifestRow
from sattsr.io.registry import get_reader

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrepareTask:
    """One file to regrid and cache. Picklable, so it can cross a process boundary."""

    source: Path
    timestamp: datetime
    sensor: str
    dest: Path
    grid_config: GridConfig
    min_coverage: float
    force: bool


@dataclass(frozen=True)
class PrepareOutcome:
    """What happened to one task."""

    task: PrepareTask
    status: str                     # written | cached | low_coverage | failed
    coverage: float = 0.0
    error: str = ""


@dataclass
class PrepareStats:
    """Aggregate counts for the final summary line."""

    read: int = 0
    written: int = 0
    skipped_cached: int = 0
    skipped_coverage: int = 0
    failed: int = 0
    refs: list[FrameRef] = field(default_factory=list)

    def log_summary(self, sensor: str, cache_root: Path) -> None:
        log.info(
            "%s -> %s: %d frames read, %d written, %d skipped (already cached), "
            "%d skipped (low coverage), %d failed",
            sensor,
            cache_root,
            self.read,
            self.written,
            self.skipped_cached,
            self.skipped_coverage,
            self.failed,
        )


def build_tasks(
    rows: Sequence[ManifestRow],
    *,
    cache_root: str | Path,
    grid_config: GridConfig,
    min_coverage: float = 0.5,
    force: bool = False,
) -> list[PrepareTask]:
    """Turn manifest rows into cache-writing tasks."""
    tasks: list[PrepareTask] = []
    for row in rows:
        ref = FrameRef(row.timestamp_utc, row.path, row.sensor)
        tasks.append(
            PrepareTask(
                source=row.path,
                timestamp=row.timestamp_utc,
                sensor=row.sensor,
                dest=cache_path_for(cache_root, ref),
                grid_config=grid_config,
                min_coverage=float(min_coverage),
                force=bool(force),
            )
        )
    return tasks


def prepare_one(task: PrepareTask) -> PrepareOutcome:
    """Read, regrid, coverage-check and cache one frame.

    Module-level and side-effect-free apart from the write, so `ProcessPoolExecutor`
    can pickle it on Windows' spawn start method.
    """
    if task.dest.exists() and not task.force:
        return PrepareOutcome(task, "cached")

    try:
        reader = get_reader(task.sensor)
        grid = TargetGrid.from_config(task.grid_config)
        frame = reader.read(task.source, grid)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return PrepareOutcome(task, "failed", error=f"{type(exc).__name__}: {exc}")

    coverage = coverage_fraction(frame.bt)
    if coverage < task.min_coverage:
        return PrepareOutcome(task, "low_coverage", coverage=coverage)

    try:
        task.dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(task.dest, np.asarray(frame.bt, dtype=np.float32))
    except OSError as exc:
        return PrepareOutcome(task, "failed", coverage=coverage, error=str(exc))

    return PrepareOutcome(task, "written", coverage=coverage)


def _record(stats: PrepareStats, outcome: PrepareOutcome) -> None:
    """Fold one outcome into the running totals, logging anything abnormal."""
    task = outcome.task
    stats.read += 1
    if outcome.status == "written":
        stats.written += 1
        stats.refs.append(FrameRef(task.timestamp, task.dest, task.sensor))
    elif outcome.status == "cached":
        stats.skipped_cached += 1
        stats.refs.append(FrameRef(task.timestamp, task.dest, task.sensor))
    elif outcome.status == "low_coverage":
        stats.skipped_coverage += 1
        log.warning(
            "skipping %s: coverage %.1f%% below %.1f%% threshold",
            task.source.name,
            outcome.coverage * 100.0,
            task.min_coverage * 100.0,
        )
    else:
        stats.failed += 1
        log.error("failed %s: %s", task.source.name, outcome.error)


def run_prepare(
    tasks: Sequence[PrepareTask], *, workers: int = 1, progress: bool = False
) -> PrepareStats:
    """Execute `tasks`, in a process pool when `workers > 1`.

    A single worker runs in-process, which keeps tracebacks readable and makes the
    unit tests deterministic; the parallel path is otherwise identical.
    """
    stats = PrepareStats()
    if not tasks:
        return stats

    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=len(tasks), desc="preprocess")

    try:
        if workers <= 1:
            for task in tasks:
                _record(stats, prepare_one(task))
                if bar is not None:
                    bar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(prepare_one, t): t for t in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        outcome = future.result()
                    except Exception as exc:            # noqa: BLE001 - worker crash
                        outcome = PrepareOutcome(
                            task, "failed", error=f"worker died: {type(exc).__name__}: {exc}"
                        )
                    _record(stats, outcome)
                    if bar is not None:
                        bar.update(1)
    finally:
        if bar is not None:
            bar.close()

    stats.refs.sort(key=lambda r: (r.timestamp, str(r.path)))
    return stats
