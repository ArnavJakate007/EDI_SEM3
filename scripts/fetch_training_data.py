#!/usr/bin/env python
"""Fetch a season-spread GOES-19 ABI C13 training set from the NOAA public bucket.

Deliberately samples several days across the year rather than one contiguous block:
the day-wise train/val split needs more than one day, and a model trained on a single
afternoon will not generalise.

This is a thin convenience wrapper with a fixed day list. For an arbitrary range, or
for control over hours and concurrency, use scripts/download_goes19.py instead.

    python scripts/fetch_training_data.py
    python scripts/fetch_training_data.py --dest data/raw/goes19 --jobs 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sattsr.data.download import fetch_goes_c13  # noqa: E402

# Spread across seasons; 16-23 UTC is afternoon over the Americas, when convection peaks.
DAYS = [date(2026, 2, 10), date(2026, 5, 12), date(2026, 6, 15), date(2026, 8, 20)]
HOURS = range(16, 24)

log = logging.getLogger("fetch_training_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dest", type=Path, default=REPO_ROOT / "data" / "raw" / "goes19",
        help="Destination directory (default: data/raw/goes19).",
    )
    parser.add_argument(
        "--jobs", type=int, default=8,
        help="Concurrent download threads. The workload is latency-bound, so this "
             "scales well past the core count. Default 8.",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Stop after this many files per day. Useful for a quick smoke test.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    total = 0
    for day in DAYS:
        paths = fetch_goes_c13(
            args.dest, day=day, hours=HOURS, progress=True,
            max_files=args.max_files, jobs=args.jobs,
        )
        total += len(paths)
        log.info("%s: %d files (running total %d)", day, len(paths), total)

    size_gb = sum(p.stat().st_size for p in args.dest.glob("*.nc")) / 1e9
    log.info("DONE: %d files, %.2f GB in %s", total, size_gb, args.dest)
    log.info("next: python scripts/build_manifest.py --sensor goes19")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
