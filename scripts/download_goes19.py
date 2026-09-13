#!/usr/bin/env python
"""Download GOES-19 ABI Channel 13 NetCDF files from the public NOAA AWS bucket.

The bucket (noaa-goes19) is anonymous-read. Access goes through s3fs with anon=True
rather than boto3 with an UNSIGNED signature: they are equivalent for public buckets,
and s3fs is already a declared and installed dependency, so this adds nothing new.

Resumable (skips files already present at the right size) and retries transient
network failures three times with exponential backoff.

    python scripts/download_goes19.py --start 2026-02-10 --end 2026-02-10
    python scripts/download_goes19.py --start 2026-02-10 --end 2026-02-12 \
        --hours 16-23 --every 3 --max-files 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sattsr.data.download import (  # noqa: E402
    DEFAULT_BUCKET,
    DEFAULT_PRODUCT,
    DownloadStats,
    _filesystem,
    download_keys,
    list_goes_keys,
)
from sattsr.data.mosdac import date_range  # noqa: E402

log = logging.getLogger("download_goes19")


def parse_hours(spec: str) -> list[int]:
    """Parse `0-23`, `16-20`, or `0,6,12` into a list of UTC hours."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        hours = list(range(int(lo), int(hi) + 1))
    else:
        hours = [int(part) for part in spec.split(",") if part.strip()]
    if not hours or not all(0 <= h <= 23 for h in hours):
        raise argparse.ArgumentTypeError(f"invalid hour spec {spec!r}; expected 0-23")
    return hours


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start", required=True, type=date.fromisoformat,
        help="First UTC date to fetch, ISO format (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end", required=True, type=date.fromisoformat,
        help="Last UTC date to fetch, inclusive, ISO format (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "goes19",
        help="Destination directory (default: data/raw/goes19).",
    )
    parser.add_argument(
        "--hours", type=parse_hours, default="0-23",
        help="UTC hours to fetch, as a range '16-23' or a list '0,6,12'. Default all.",
    )
    parser.add_argument(
        "--every", type=int, default=1,
        help="Keep every Nth file. ABI full-disc is 10-minutely, so 3 gives the "
             "30-minute cadence this project targets. Default 1 (keep all).",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Stop after this many files per day. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--bucket", default=DEFAULT_BUCKET, help=f"S3 bucket (default: {DEFAULT_BUCKET}).",
    )
    parser.add_argument(
        "--product", default=DEFAULT_PRODUCT,
        help=f"ABI product level (default: {DEFAULT_PRODUCT}). Use ABI-L2-CMIPF for "
             "the L2 cloud-and-moisture product; the reader handles both Rad and CMI.",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Attempts per file (default 3).",
    )
    parser.add_argument(
        "--jobs", type=int, default=8,
        help="Concurrent download threads. The workload is latency-bound, so "
             "this scales well past the core count. Default 8.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List keys, download nothing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    try:
        days = date_range(args.start, args.end)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    fs = _filesystem()
    total = DownloadStats()

    for day in days:
        keys = list_goes_keys(
            day, args.hours, bucket=args.bucket, product=args.product, fs=fs
        )[:: args.every]
        if args.max_files is not None:
            keys = keys[: args.max_files]

        if args.dry_run:
            log.info("%s: %d key(s) would be fetched", day, len(keys))
            total.found += len(keys)
            continue

        stats = download_keys(
            fs, keys, args.out_dir, progress=True,
            label=f"GOES {day}", attempts=args.retries, jobs=args.jobs,
        )
        stats.log_summary(f"GOES-19 {day}")
        total.found += stats.found
        total.downloaded += stats.downloaded
        total.skipped += stats.skipped
        total.failed += stats.failed

    total.log_summary(f"GOES-19 TOTAL {args.start}..{args.end} -> {args.out_dir}")
    log.info("next: python scripts/build_manifest.py --sensor goes19")
    return 1 if total.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
