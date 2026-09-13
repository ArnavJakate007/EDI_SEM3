#!/usr/bin/env python
"""Regrid every manifested raw file into the cached .npy frame layout.

Reads data/raw/<sensor>/manifest.csv (from scripts/build_manifest.py), dispatches each
file to its sensor's reader, regrids onto the target grid from the sensor's config,
drops frames whose valid coverage is below --min-coverage, and writes the result to
the cache layout that the dataset and training loop already read.

Idempotent: frames whose .npy already exists are skipped unless --force is passed.

    python scripts/preprocess.py --sensor goes19 --workers 4
    python scripts/preprocess.py --sensor insat3dr --config configs/insat_staggered.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sattsr.config import load_config  # noqa: E402
from sattsr.data.index import FrameRef, load_index, save_index  # noqa: E402
from sattsr.data.normalize import (  # noqa: E402
    PLAUSIBLE_MEDIAN_SHIFT_K,
    RenormRegistry,
)
from sattsr.data.prepare import build_tasks, run_prepare  # noqa: E402
from sattsr.io.manifest import read_manifest  # noqa: E402
from sattsr.io.registry import READERS  # noqa: E402

log = logging.getLogger("preprocess")


def merge_index(index_path: Path, new_refs: list[FrameRef]) -> list[FrameRef]:
    """Fold this run's refs into any existing index instead of replacing it.

    A partial run -- `--limit`, a single day, one sensor of several -- must not
    truncate the index down to just what it happened to touch. Entries whose file has
    since disappeared are dropped, so the index stays an accurate picture of the cache.
    """
    merged: dict[str, FrameRef] = {}
    if index_path.exists():
        for ref in load_index(index_path):
            merged[str(ref.path)] = ref
    for ref in new_refs:
        merged[str(ref.path)] = ref

    alive = [r for r in merged.values() if r.path.exists()]
    dropped = len(merged) - len(alive)
    if dropped:
        log.info("dropped %d index entr(y/ies) whose cache file no longer exists", dropped)
    return sorted(alive, key=lambda r: (r.timestamp, str(r.path)))


def resolve_renorm(
    sensor: str, mode: str, renorm_dir: Path, *, max_shift_k: float = PLAUSIBLE_MEDIAN_SHIFT_K
):
    """Pick the SensorRenorm to bake into this sensor's cache, or None.

    Raises ValueError with actionable text when renormalisation is *required* but
    unavailable OR implausible, so neither a missing registry nor a scene-confounded
    map can be used silently.
    """
    if mode == "off":
        log.info("renormalisation disabled (--renorm off)")
        return None

    try:
        registry = RenormRegistry.load(renorm_dir)
    except FileNotFoundError:
        message = (
            f"no renormalisation registry in {renorm_dir}. Fit one first:\n"
            f"    python scripts/preprocess.py --sensor <reference> --renorm off\n"
            f"    python scripts/fit_renorm.py"
        )
        if mode == "require":
            raise ValueError(f"--renorm require, but {message}") from None
        log.info("no renorm registry in %s; caching un-renormalised frames", renorm_dir)
        return None

    if sensor == registry.reference:
        log.info(
            "%s is the reference sensor; no renormalisation applied (by definition)",
            sensor,
        )
        return None

    renorm = registry.for_sensor(sensor)
    if renorm is None:
        message = (
            f"registry in {renorm_dir} has no map for {sensor!r} "
            f"(it has: {registry.sensors or 'none'}). Fit one with:\n"
            f"    python scripts/fit_renorm.py --sensors {sensor}"
        )
        if mode == "require":
            raise ValueError(f"--renorm require, but {message}")
        log.warning("%s; caching un-renormalised frames", message)
        return None

    for scene, qmap in sorted(renorm.maps.items()):
        s = qmap.summary()
        log.info(
            "  renorm %s [%s]: median %+.2f K, IQR ratio %.3f  (%s)",
            sensor, scene, s["median_shift_k"], s["iqr_ratio"],
            qmap.provenance.describe(),
        )

    bad = renorm.implausible_scenes(max_shift_k=max_shift_k)
    if bad:
        detail = "; ".join(f"[{scene}] {'; '.join(rs)}" for scene, rs in bad.items())
        message = (
            f"the {sensor!r} renorm map fails the {max_shift_k:g} K plausibility "
            f"guard: {detail}. A shift this large between two ~10-11 um window "
            f"channels is a SCENE difference, not a calibration offset -- refit with "
            f"scripts/fit_renorm.py (see each map's provenance block for what it was "
            f"fitted on)."
        )
        if mode == "require":
            raise ValueError(f"--renorm require refuses to use it: {message}")
        log.warning("%s Applying anyway because --renorm auto was requested.", message)

    log.info("applying scene-stratified renorm %s -> %s before caching",
             sensor, registry.reference)
    return renorm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sensor", required=True, choices=sorted(READERS),
        help="Which sensor to preprocess. Selects the reader and the default config.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Config YAML supplying the target grid and cache_root. "
             "Default: configs/<sensor>.yaml. Use an explicit path for the INSAT "
             "scan-mode configs (configs/insat_staggered.yaml, insat_rapidscan.yaml).",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Manifest CSV to read. Default: <raw_root>/manifest.csv from the config.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes. Regridding is CPU-bound, so this scales with "
             "physical cores. Default 1 (in-process, readable tracebacks).",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=0.5,
        help="Minimum fraction of valid (non-NaN) pixels a regridded frame must have "
             "to be cached. Default 0.5. Frames below it are logged and skipped.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-regrid and overwrite frames that are already cached.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N manifest rows (quick smoke test).",
    )
    parser.add_argument(
        "--renorm", choices=("auto", "require", "off"), default="auto",
        help="Apply the fitted cross-sensor renormalisation map before caching. "
             "'auto' (default) applies it when a registry exists and this sensor is "
             "not the reference, and proceeds with a log line when it does not. "
             "'require' fails if no registry or no map for this sensor is present. "
             "'off' skips it entirely. NOTE: renormalisation is baked into the "
             "cached .npy, so changing it means re-running with --force.",
    )
    parser.add_argument(
        "--renorm-dir", type=Path, default=REPO_ROOT / "configs" / "renorm",
        help="Directory holding the registry from scripts/fit_renorm.py "
             "(default: configs/renorm).",
    )
    parser.add_argument(
        "--max-shift-k", type=float, default=PLAUSIBLE_MEDIAN_SHIFT_K,
        help=f"Median-shift plausibility limit in Kelvin (default "
             f"{PLAUSIBLE_MEDIAN_SHIFT_K:g}). --renorm require refuses a map that "
             f"exceeds it; --renorm auto warns loudly and proceeds.",
    )
    parser.add_argument(
        "--no-index", action="store_true",
        help="Skip writing <cache_root>/index.json alongside the cached frames.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    config_path = args.config or REPO_ROOT / "configs" / f"{args.sensor}.yaml"
    if not config_path.exists():
        log.error(
            "no config at %s. Pass --config explicitly (available: %s)",
            config_path,
            ", ".join(sorted(p.name for p in (REPO_ROOT / "configs").glob("*.yaml"))),
        )
        return 2
    cfg = load_config(config_path)

    manifest_path = args.manifest or Path(cfg.data.raw_root) / "manifest.csv"
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not manifest_path.exists():
        log.error(
            "no manifest at %s -- run: python scripts/build_manifest.py --sensor %s",
            manifest_path, args.sensor,
        )
        return 2

    rows = [r for r in read_manifest(manifest_path) if r.sensor == args.sensor]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        log.warning("manifest %s has no rows for sensor %s; nothing to do",
                    manifest_path, args.sensor)
        return 0

    cache_root = Path(cfg.data.cache_root)
    if not cache_root.is_absolute():
        cache_root = REPO_ROOT / cache_root

    try:
        renorm = resolve_renorm(
            args.sensor, args.renorm, args.renorm_dir, max_shift_k=args.max_shift_k
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    log.info("preprocessing %d frame(s) from %s with %d worker(s)",
             len(rows), manifest_path, args.workers)

    tasks = build_tasks(
        rows,
        cache_root=cache_root,
        grid_config=cfg.data.grid,
        min_coverage=args.min_coverage,
        force=args.force,
        renorm=renorm,
    )
    stats = run_prepare(tasks, workers=args.workers, progress=True)
    stats.log_summary(args.sensor, cache_root)

    if stats.refs and not args.no_index:
        refs = merge_index(cache_root / "index.json", stats.refs)
        index_path = save_index(refs, cache_root / "index.json")
        log.info("wrote index of %d cached frame(s) to %s", len(refs), index_path)

    if not stats.refs:
        log.warning("no frames cached -- check --min-coverage and the reader's output")
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
