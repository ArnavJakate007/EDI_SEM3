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
from sattsr.data.index import save_index  # noqa: E402
from sattsr.data.prepare import build_tasks, run_prepare  # noqa: E402
from sattsr.io.manifest import read_manifest  # noqa: E402
from sattsr.io.registry import READERS  # noqa: E402

log = logging.getLogger("preprocess")


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

    log.info("preprocessing %d frame(s) from %s with %d worker(s)",
             len(rows), manifest_path, args.workers)

    tasks = build_tasks(
        rows,
        cache_root=cache_root,
        grid_config=cfg.data.grid,
        min_coverage=args.min_coverage,
        force=args.force,
    )
    stats = run_prepare(tasks, workers=args.workers, progress=True)
    stats.log_summary(args.sensor, cache_root)

    if stats.refs and not args.no_index:
        index_path = save_index(stats.refs, cache_root / "index.json")
        log.info("wrote index of %d cached frame(s) to %s", len(stats.refs), index_path)

    if not stats.refs:
        log.warning("no frames cached -- check --min-coverage and the reader's output")
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
