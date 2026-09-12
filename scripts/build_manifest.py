#!/usr/bin/env python
"""Scan data/raw/<sensor>/ and write manifest.csv for the preprocessing stage.

Timestamps are parsed by each sensor's own reader, so the manifest always agrees
with what `sattsr prepare` / `scripts/preprocess.py` will parse from the same names.

    python scripts/build_manifest.py --sensor goes19
    python scripts/build_manifest.py --sensor insat3dr --scan-mode rapid_scan
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sattsr.io.manifest import (  # noqa: E402
    INSAT_SCAN_MODES,
    scan_raw_files,
    write_manifest,
)
from sattsr.io.registry import READERS  # noqa: E402

log = logging.getLogger("build_manifest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sensor",
        required=True,
        choices=sorted(READERS),
        help="Which sensor's raw directory to scan.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Directory to scan. Default: data/raw/<sensor>. "
        "Scanning is recursive, so per-scan-mode subdirectories are picked up.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the CSV. Default: <raw-root>/manifest.csv.",
    )
    parser.add_argument(
        "--scan-mode",
        choices=list(INSAT_SCAN_MODES),
        default=None,
        help="Force a scan mode for every row instead of inferring it from the "
        "directory name. Only meaningful for INSAT; full-disc sensors default "
        "to 'full_disk'.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Log every file as it is manifested."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    raw_root = args.raw_root or REPO_ROOT / "data" / "raw" / args.sensor
    out_path = args.out or raw_root / "manifest.csv"

    report = scan_raw_files(raw_root, args.sensor, scan_mode=args.scan_mode)
    report.log_summary(raw_root)
    write_manifest(report, out_path)

    if not report.rows:
        log.warning("manifest is empty -- nothing to preprocess from %s", raw_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
