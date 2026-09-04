"""Fetch a season-spread GOES-19 ABI C13 training set from the NOAA public bucket.

Deliberately samples several days across the year rather than one contiguous block:
the day-wise train/val split needs more than one day, and a model trained on a single
afternoon will not generalise.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

from sattsr.data.download import fetch_goes_c13

# Spread across seasons; 16-23 UTC is afternoon over the Americas, when convection peaks.
DAYS = [date(2026, 2, 10), date(2026, 5, 12), date(2026, 6, 15), date(2026, 8, 20)]
HOURS = range(16, 24)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/goes19")

    total = 0
    for day in DAYS:
        paths = fetch_goes_c13(dest, day=day, hours=HOURS, progress=False)
        total += len(paths)
        print(f"{day}: {len(paths)} files (running total {total})", flush=True)

    size_gb = sum(p.stat().st_size for p in dest.glob("*.nc")) / 1e9
    print(f"DONE: {total} files, {size_gb:.2f} GB in {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
