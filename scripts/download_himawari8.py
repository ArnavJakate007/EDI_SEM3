#!/usr/bin/env python
"""Download Himawari-8/9 AHI Band 13 files.

WHICH SOURCE, AND WHY -- READ THIS BEFORE RUNNING
=================================================
Two archives carry AHI Band 13, and they are NOT interchangeable for this repo:

  aws (default)   NOAA Open Data mirror, anonymous S3 on noaa-himawari8.
                  Product AHI-L2-FLDK-ISatSS: tiled NetCDF on the SAME geostationary
                  fixed grid as GOES ABI, verified against a real downloaded file.
                  The payload `Sectorized_CMI` is already brightness temperature in
                  Kelvin, so this is close to a drop-in for the GOES reader path.
                  No credentials required.
                  NOT raw AHI-L1b-FLDK: that is Himawari Standard Format, a
                  proprietary binary with no standard Python reader, which would mean
                  writing an HSF decoder from scratch.
                  One scan = 76 tiles, so --every / --max-slots subsample whole
                  10-minute SLOTS, never individual tiles.

  jaxa            JAXA P-Tree, FTP at ftp.ptree.jaxa.jp.
                  Ships the *gridded* product `NC_H08_<YYYYMMDD>_<HHMM>_R21_FLDK...nc`
                  carrying `tbb_13` already in Kelvin on a regular lat-lon grid, which
                  the reader also understands. Kept as a fallback, but it needs a free
                  registered account -- there is no anonymous access. Set
                  HIMAWARI_USER and HIMAWARI_PASS.

Default is aws because it needs no credentials and the reader handles ISatSS directly.

    python scripts/download_himawari8.py --start 2019-07-01 --end 2019-07-01 --hours 1-2 --max-slots 3
    python scripts/download_himawari8.py --source jaxa --start 2026-02-10 --end 2026-02-10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sattsr.data.download import (  # noqa: E402
    HIMAWARI_BUCKETS,
    HIMAWARI_DEFAULT_PRODUCT,
    DownloadStats,
    fetch_himawari_b13_aws,
    fetch_himawari_b13_ptree,
)
from sattsr.data.mosdac import date_range  # noqa: E402

log = logging.getLogger("download_himawari8")

USER_ENV = "HIMAWARI_USER"
PASS_ENV = "HIMAWARI_PASS"


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
    parser.add_argument("--start", required=True, type=date.fromisoformat,
                        help="First UTC date to fetch (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, type=date.fromisoformat,
                        help="Last UTC date to fetch, inclusive (YYYY-MM-DD).")
    parser.add_argument(
        "--source", choices=("aws", "jaxa"), default="aws",
        help="aws = anonymous NOAA mirror, AHI-L2-FLDK-ISatSS tiled NetCDF (default). "
             "jaxa = P-Tree gridded NC_H08_* over FTP, needs HIMAWARI_USER/PASS.",
    )
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "data" / "raw" / "himawari8",
                        help="Destination directory (default: data/raw/himawari8).")
    parser.add_argument("--hours", type=parse_hours, default="0-23",
                        help="UTC hours as a range '0-5' or list '0,6,12'. Default all.")
    parser.add_argument("--every", type=int, default=1,
                        help="Keep every Nth 10-minute SLOT, not every Nth tile "
                             "(AWS source only). Default 1.")
    parser.add_argument("--max-slots", type=int, default=None,
                        help="Stop after this many 10-minute slots per day. Each slot "
                             "is one frame (76 tiles, ~30 MB). Use this, not "
                             "--max-files, to fetch a few complete frames.")
    parser.add_argument("--channel", type=int, default=13,
                        help="AHI channel to fetch (AWS/ISatSS only). Default 13 "
                             "(10.4 um thermal IR).")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Stop after this many files per day (quick smoke test).")
    parser.add_argument("--satellite", choices=sorted(HIMAWARI_BUCKETS), default="himawari8",
                        help="Which spacecraft's AWS bucket to use (aws source only).")
    parser.add_argument("--product", default=HIMAWARI_DEFAULT_PRODUCT,
                        help=f"AWS product prefix (default: {HIMAWARI_DEFAULT_PRODUCT}).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    try:
        days = date_range(args.start, args.end)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    user, password = os.environ.get(USER_ENV), os.environ.get(PASS_ENV)
    if args.source == "jaxa" and (not user or not password):
        log.error(
            "JAXA P-Tree needs credentials and has no anonymous access.\n"
            "  Set %s and %s (free registration at https://www.eorc.jaxa.jp/ptree/),\n"
            "  or re-run with --source aws to use the anonymous NOAA mirror -- but note\n"
            "  the reader caveat above.\n"
            "NEEDS YOUR INPUT: nothing was downloaded.",
            USER_ENV, PASS_ENV,
        )
        return 2

    total = DownloadStats()
    for day in days:
        if args.source == "jaxa":
            stats = fetch_himawari_b13_ptree(
                args.out_dir, day=day, hours=args.hours,
                user=user, password=password,
                max_files=args.max_files, progress=True,
            )
        else:
            stats = fetch_himawari_b13_aws(
                args.out_dir, day=day, hours=args.hours,
                bucket=HIMAWARI_BUCKETS[args.satellite], product=args.product,
                channel=args.channel, max_files=args.max_files,
                max_slots=args.max_slots, every=args.every, progress=True,
            )
        total.found += stats.found
        total.downloaded += stats.downloaded
        total.skipped += stats.skipped
        total.failed += stats.failed

    total.log_summary(f"Himawari TOTAL {args.start}..{args.end} -> {args.out_dir}")
    log.info("next: python scripts/build_manifest.py --sensor himawari8")
    return 1 if total.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
