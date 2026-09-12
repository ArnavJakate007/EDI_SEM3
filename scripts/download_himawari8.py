#!/usr/bin/env python
"""Download Himawari-8/9 AHI Band 13 files.

WHICH SOURCE, AND WHY -- READ THIS BEFORE RUNNING
=================================================
Two archives carry AHI Band 13, and they are NOT interchangeable for this repo:

  jaxa (default)  JAXA P-Tree, FTP at ftp.ptree.jaxa.jp.
                  Ships the *gridded* NetCDF product `NC_H08_<YYYYMMDD>_<HHMM>_R21_
                  FLDK...nc` carrying `tbb_13` already in Kelvin on a regular lat-lon
                  grid. This is exactly what sattsr.io.himawari.HimawariReader parses,
                  so it drops straight into the pipeline.
                  Requires a free registered account -- there is no anonymous access.
                  Set HIMAWARI_USER and HIMAWARI_PASS.

  aws             NOAA Open Data mirror, anonymous S3 (noaa-himawari8/noaa-himawari9).
                  Scriptable with no credentials, which is why it is offered here.
                  BUT it carries AHI L1b/L2 in NOAA's own segmented layout with
                  different variable names -- NOT `tbb_13` on a lat-lon grid. The
                  current HimawariReader will NOT read these files without an added
                  reader branch. Use this only if you intend to extend the reader.

Default is jaxa because it is the one that actually works end-to-end today. If you
have no P-Tree account, the script tells you rather than silently downloading nothing.

    export HIMAWARI_USER=... HIMAWARI_PASS=...
    python scripts/download_himawari8.py --start 2026-02-10 --end 2026-02-10 --hours 0-5
    python scripts/download_himawari8.py --source aws --start 2026-02-10 --end 2026-02-10
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
        "--source", choices=("jaxa", "aws"), default="jaxa",
        help="jaxa = P-Tree gridded NetCDF the reader understands (needs an account). "
             "aws = anonymous NOAA mirror, but the reader cannot parse it yet. "
             "Default: jaxa.",
    )
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "data" / "raw" / "himawari8",
                        help="Destination directory (default: data/raw/himawari8).")
    parser.add_argument("--hours", type=parse_hours, default="0-23",
                        help="UTC hours as a range '0-5' or list '0,6,12'. Default all.")
    parser.add_argument("--every", type=int, default=1,
                        help="Keep every Nth file (AWS source only). Default 1.")
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

    if args.source == "aws":
        log.warning(
            "source=aws: these files are NOT the JAXA gridded product. "
            "sattsr.io.himawari.HimawariReader expects 'tbb_13' on a lat-lon grid and "
            "will reject NOAA's layout until a reader branch is added."
        )

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
                max_files=args.max_files, every=args.every, progress=True,
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
