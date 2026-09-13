#!/usr/bin/env python
"""Month-at-a-time download -> preprocess -> delete-raw backfill.

Why month-at-a-time
-------------------
Peak disk, not final footprint, is what fills a laptop. Downloading a year of raw
GOES-19 and then preprocessing it would need ~130 GB of raw and ~20 GB of cache
coexisting; doing it one month at a time and deleting each month's raw as soon as its
frames are cached bounds the raw high-water mark to a single month -- a few GB --
however many months are eventually processed.

Each month runs: free-space check -> download -> manifest -> preprocess with
--delete-raw-after-cache -> verify raw actually cleared -> log, then move on. A month
whose raw did not clear is reported rather than silently carried forward.

Day selection
-------------
Rather than 2-3 contiguous whole days per month, this spends the same frame budget on
several separate days, each covering a different 8-hour window, rotating so the
windows tile the full 24 hours across the month. Same download cost and the same
diurnal coverage, but the frames come from several distinct synoptic situations
instead of one -- which is what a flow model actually learns from.

    python scripts/backfill_monthly.py --sensors goes19 --year 2025
    python scripts/backfill_monthly.py --sensors himawari8 --months 2022-01,2022-06
    python scripts/backfill_monthly.py --sensors goes19 --year 2025 --dry-run
"""

from __future__ import annotations

import argparse
import calendar
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sattsr.config import load_config  # noqa: E402

log = logging.getLogger("backfill")

#: Which config drives each sensor's raw/cache roots and target grid.
SENSOR_CONFIG = {"goes19": "goes19.yaml", "himawari8": "himawari.yaml"}

#: Approximate raw megabytes per cached frame, measured on this archive: a GOES ABI
#: C13 granule is ~26 MB, and one Himawari ISatSS scan is ~88 tiles of ~0.35 MB. Used
#: only to size the free-space check, so a rough figure is fine -- but it must never
#: be an UNDERestimate, or the check passes and the disk then fills anyway.
RAW_MB_PER_FRAME = {"goes19": 27.0, "himawari8": 32.0}

#: Rotating 8-hour windows, so a month's sampled days tile the full 24 hours.
HOUR_WINDOWS = [(0, 7), (8, 15), (16, 23)]

RAW_SUFFIXES = (".nc", ".nc4", ".h5")


@dataclass
class MonthResult:
    """What one month actually cost and produced."""

    sensor: str
    year: int
    month: int
    downloaded_gb: float = 0.0
    frames_cached: int = 0
    deleted_gb: float = 0.0
    raw_left_gb: float = 0.0
    free_gb_after: float = 0.0
    skipped: str = ""
    notes: list[str] = field(default_factory=list)


def free_gb(path: Path) -> float:
    """Free space in GB on the volume holding `path`."""
    return shutil.disk_usage(path).free / 1e9


def dir_bytes(path: Path) -> int:
    """Total size of files under `path`, 0 if it does not exist."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def raw_file_count(path: Path) -> int:
    """How many raw granules/tiles are still sitting in `path`."""
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file() and f.suffix in RAW_SUFFIXES)


def day_window_cached(
    cache_root: Path, day: date, lo: int, hi: int, *, min_fraction: float = 0.9
) -> bool:
    """True if this day's hour window is already sufficiently cached.

    Raising --days-per-month re-picks a denser set of days that OVERLAPS the sparser
    set a previous run already fetched (2/month gives [10, 21]; 5/month gives
    [5, 10, 16, 21, 26]). Raw was deleted after caching, so without this check those
    overlapping days would be downloaded again in full -- roughly 40% of the transfer
    on a 2 -> 5 step, which is hours at this link speed.

    A partially-cached day is still re-fetched: `min_fraction` guards against treating
    a day that only half-succeeded as done.
    """
    day_dir = cache_root / day.strftime("%Y%m%d")
    if not day_dir.is_dir():
        return False
    expected = (hi - lo + 1) * 6            # 10-minute cadence
    have = sum(
        1 for f in day_dir.glob("*.npy")
        if f.stem[:2].isdigit() and lo <= int(f.stem[:2]) <= hi
    )
    return have >= expected * min_fraction


def month_days(year: int, month: int, count: int) -> list[int]:
    """Pick `count` days spread across the month, avoiding the edges."""
    last = calendar.monthrange(year, month)[1]
    if count <= 1:
        return [min(15, last)]
    step = last / (count + 1)
    return sorted({max(1, min(last, int(round(step * (i + 1))))) for i in range(count)})


def parse_months(spec: str | None, year: int | None) -> list[tuple[int, int]]:
    """`2022-01,2022-06`, or every month of `--year`."""
    if spec:
        out: list[tuple[int, int]] = []
        for token in spec.split(","):
            token = token.strip()
            if not token:
                continue
            y, m = token.split("-")
            out.append((int(y), int(m)))
        return sorted(set(out))
    if year is None:
        raise ValueError("pass either --year or --months")
    return [(year, m) for m in range(1, 13)]


def run(cmd: list[str], *, label: str) -> tuple[int, str]:
    """Run a child script, returning (exit code, combined output)."""
    log.debug("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        log.warning("%s exited %d", label, proc.returncode)
    return proc.returncode, out


def download_month(
    sensor: str, year: int, month: int, days: list[int], hours_per_day: int, jobs: int,
    *, cache_root: Path | None = None,
) -> int:
    """Fetch this month's chosen days, each on its own rotating hour window.

    Returns how many days were skipped because the cache already covers them.
    """
    skipped = 0
    for i, day in enumerate(days):
        lo, hi = HOUR_WINDOWS[i % len(HOUR_WINDOWS)]
        if hours_per_day != 8:
            hi = lo + hours_per_day - 1
        d = date(year, month, day)

        if cache_root is not None and day_window_cached(cache_root, d, lo, hi):
            log.info("    %s %02d-%02dz already cached; not re-downloading", d, lo, hi)
            skipped += 1
            continue
        if sensor == "goes19":
            cmd = [sys.executable, "scripts/download_goes19.py",
                   "--start", d.isoformat(), "--end", d.isoformat(),
                   "--hours", f"{lo}-{hi}", "--jobs", str(jobs)]
        else:
            cmd = [sys.executable, "scripts/download_himawari8.py",
                   "--start", d.isoformat(), "--end", d.isoformat(),
                   "--hours", f"{lo}-{hi}", "--jobs", str(jobs),
                   "--out-dir", "data/raw/himawari"]
        _, out = run(cmd, label=f"download {sensor} {d} {lo:02d}-{hi:02d}z")
        for line in out.splitlines():
            if "TOTAL" in line or "ERROR" in line:
                log.info("    %s", line.split(": ", 2)[-1])
    return skipped


def process_month(sensor: str, config: Path, workers: int) -> str:
    """Manifest, then preprocess with raw deletion."""
    raw_root = Path(load_config(config).data.raw_root)
    run([sys.executable, "scripts/build_manifest.py", "--sensor", sensor,
         "--raw-root", str(raw_root)], label=f"manifest {sensor}")
    _, out = run(
        [sys.executable, "scripts/preprocess.py", "--sensor", sensor,
         "--config", str(config), "--workers", str(workers),
         "--renorm", "off", "--delete-raw-after-cache"],
        label=f"preprocess {sensor}",
    )
    for line in out.splitlines():
        if "frames read" in line or "reclaimed" in line:
            log.info("    %s", line.split(": ", 2)[-1])
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sensors", default="goes19,himawari8",
        help="Comma-separated sensors to backfill. INSAT-3DR is deliberately not "
             "supported here: it has no automated download path.",
    )
    parser.add_argument("--year", type=int, default=None,
                        help="Backfill all 12 months of this year.")
    parser.add_argument("--months", default=None,
                        help="Explicit months as YYYY-MM,YYYY-MM (overrides --year). "
                             "Needed for Himawari, whose archive has real gaps.")
    parser.add_argument("--days-per-month", type=int, default=3,
                        help="Days sampled per month (default 3). Each gets a "
                             "different 8-hour window so the month tiles 24 hours.")
    parser.add_argument("--hours-per-day", type=int, default=8,
                        help="Hours fetched per sampled day (default 8).")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Concurrent download threads (default 8). The link "
                             "saturates near 8; more only adds contention.")
    parser.add_argument("--workers", type=int, default=6,
                        help="Preprocessing worker processes (default 6).")
    parser.add_argument("--min-free-gb", type=float, default=20.0,
                        help="Refuse to start a month unless this much space is free "
                             "BEYOND that month's expected download (default 20).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and disk projection, fetch nothing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    unknown = [s for s in sensors if s not in SENSOR_CONFIG]
    if unknown:
        log.error("unsupported sensor(s) %s; this tool covers %s only",
                  unknown, sorted(SENSOR_CONFIG))
        return 2

    months = parse_months(args.months, args.year)
    frames_per_day = args.hours_per_day * 6          # 10-minute cadence
    results: list[MonthResult] = []
    start_free = free_gb(REPO_ROOT)
    log.info("starting: %.1f GB free | %d month(s) x %d sensor(s) | "
             "%d day(s)/month x %d h/day",
             start_free, len(months), len(sensors),
             args.days_per_month, args.hours_per_day)

    for year, month in months:
        for sensor in sensors:
            cfg_path = REPO_ROOT / "configs" / SENSOR_CONFIG[sensor]
            cfg = load_config(cfg_path)
            raw_root = REPO_ROOT / cfg.data.raw_root
            cache_root = REPO_ROOT / cfg.data.cache_root
            days = month_days(year, month, args.days_per_month)

            expected_gb = len(days) * frames_per_day * RAW_MB_PER_FRAME[sensor] / 1000.0
            available = free_gb(REPO_ROOT)
            result = MonthResult(sensor, year, month, free_gb_after=available)

            log.info("")
            log.info("=== %04d-%02d %s | days %s | expect ~%.1f GB raw | %.1f GB free ===",
                     year, month, sensor, days, expected_gb, available)

            if available < expected_gb + args.min_free_gb:
                result.skipped = (
                    f"only {available:.1f} GB free; need {expected_gb:.1f} GB "
                    f"+ {args.min_free_gb:.1f} GB margin"
                )
                log.error("STOPPING before %04d-%02d %s: %s", year, month, sensor,
                          result.skipped)
                results.append(result)
                report(results, start_free)
                return 1

            if args.dry_run:
                result.skipped = "dry run"
                results.append(result)
                continue

            before_raw = dir_bytes(raw_root)
            before_cache = len(list(cache_root.rglob("*.npy")))

            n_skipped = download_month(
                sensor, year, month, days, args.hours_per_day, args.jobs,
                cache_root=cache_root,
            )
            if n_skipped:
                result.notes.append(f"{n_skipped}/{len(days)} day(s) already cached")
            after_raw = dir_bytes(raw_root)
            result.downloaded_gb = max(0.0, (after_raw - before_raw) / 1e9)

            process_month(sensor, cfg_path, args.workers)
            result.frames_cached = len(list(cache_root.rglob("*.npy"))) - before_cache

            left_raw = dir_bytes(raw_root)
            result.deleted_gb = max(0.0, (after_raw - left_raw) / 1e9)
            result.raw_left_gb = left_raw / 1e9
            result.free_gb_after = free_gb(REPO_ROOT)

            n_left = raw_file_count(raw_root)
            if n_left:
                result.notes.append(
                    f"{n_left} raw file(s) ({result.raw_left_gb:.2f} GB) kept: these "
                    f"failed preprocessing and stay for retry"
                )

            log.info(
                "--- %04d-%02d %s: +%.2f GB raw -> %d frames cached, %.2f GB reclaimed, "
                "%.2f GB raw left, %.1f GB free",
                year, month, sensor, result.downloaded_gb, result.frames_cached,
                result.deleted_gb, result.raw_left_gb, result.free_gb_after,
            )
            for note in result.notes:
                log.warning("    %s", note)
            results.append(result)

    report(results, start_free)
    return 0


def report(results: list[MonthResult], start_free: float) -> None:
    """Per-month table plus the running disk totals."""
    if not results:
        return
    print("\n" + "=" * 98)
    print("PER-MONTH BACKFILL LOG")
    print("=" * 98)
    print(f"{'month':<9}{'sensor':<12}{'raw DL GB':>10}{'cached':>8}"
          f"{'freed GB':>10}{'raw left':>10}{'free GB':>10}  note")
    print("-" * 98)
    tot_dl = tot_freed = 0.0
    tot_frames = 0
    for r in results:
        note = r.skipped or "; ".join(r.notes)
        print(f"{r.year:04d}-{r.month:02d} {r.sensor:<12}{r.downloaded_gb:>10.2f}"
              f"{r.frames_cached:>8}{r.deleted_gb:>10.2f}{r.raw_left_gb:>10.2f}"
              f"{r.free_gb_after:>10.1f}  {note}")
        tot_dl += r.downloaded_gb
        tot_frames += r.frames_cached
        tot_freed += r.deleted_gb
    print("-" * 98)
    print(f"{'TOTAL':<21}{tot_dl:>10.2f}{tot_frames:>8}{tot_freed:>10.2f}")
    print(f"\nfree space: {start_free:.1f} GB at start -> "
          f"{results[-1].free_gb_after:.1f} GB now "
          f"(net {results[-1].free_gb_after - start_free:+.1f} GB)")


if __name__ == "__main__":
    raise SystemExit(main())
