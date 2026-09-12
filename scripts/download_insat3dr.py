#!/usr/bin/env python
"""Fetch INSAT-3DR/3DS TIR1 files from MOSDAC, or print manual-download instructions.

WHAT THIS SCRIPT CAN AND CANNOT DO -- READ FIRST
================================================
MOSDAC (ISRO) has no stable public API. It is a server-rendered portal behind a login
form, and ordering data is an ASYNCHRONOUS workflow: you place an order and MOSDAC
emails a download link hours later. That cannot be driven reliably from a script
without confirmed, current endpoint details, and the markup has changed between
portal revisions.

So this script does two things:

  --manual-instructions   Prints exactly what to click and where to save the files.
                          THIS IS THE SUPPORTED PATH. Use it.

  (default)               Attempts a session login with MOSDAC_USER / MOSDAC_PASS and
                          then fails with a specific, actionable error. It does not
                          pretend to succeed, and it never silently downloads nothing.

Once files are in data/raw/insat3dr/<scan_mode>/, everything downstream -- manifest
building, preprocessing, the dataset, training -- is identical to the automated
sensors. Nothing knows the files arrived by hand.

    python scripts/download_insat3dr.py --manual-instructions \
        --scan-mode rapid_scan --start 2026-02-10 --end 2026-02-12
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sattsr.data.mosdac import (  # noqa: E402
    SCAN_MODE_PRODUCTS,
    MosdacCredentials,
    MosdacError,
    MosdacSession,
    date_range,
    manual_instructions,
    target_dir,
)

log = logging.getLogger("download_insat3dr")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", required=True, type=date.fromisoformat,
                        help="First UTC date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, type=date.fromisoformat,
                        help="Last UTC date, inclusive (YYYY-MM-DD).")
    parser.add_argument(
        "--scan-mode", required=True, choices=sorted(SCAN_MODE_PRODUCTS),
        help="MOSDAC product stream. 'routine' is the 30-min operational scan, "
             "'staggered' the 15-min interleaved product used for fine-tuning, and "
             "'rapid_scan' the ~4-min sectoral product held out as native "
             "high-cadence ground truth. Files are routed to "
             "data/raw/insat3dr/<scan_mode>/.",
    )
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "data" / "raw" / "insat3dr",
                        help="Raw root; the scan mode becomes a subdirectory of it.")
    parser.add_argument(
        "--manual-instructions", action="store_true",
        help="Print step-by-step manual download instructions and exit. "
             "This is the supported path -- see the module docstring.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    try:
        date_range(args.start, args.end)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    dest = target_dir(args.out_dir, args.scan_mode)
    dest.mkdir(parents=True, exist_ok=True)

    if args.manual_instructions:
        print(manual_instructions(args.out_dir, args.scan_mode, args.start, args.end))
        return 0

    try:
        credentials = MosdacCredentials.from_env()
        session = MosdacSession(credentials)
        session.login()
        paths = session.fetch_range(args.scan_mode, args.start, args.end, args.out_dir)
    except MosdacError as exc:
        log.error("%s", exc)
        log.error(
            "NEEDS YOUR INPUT: nothing was downloaded. Run with "
            "--manual-instructions and follow those steps."
        )
        return 2

    log.info("downloaded %d file(s) to %s", len(paths), dest)
    log.info(
        "next: python scripts/build_manifest.py --sensor insat3dr --raw-root %s",
        args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
