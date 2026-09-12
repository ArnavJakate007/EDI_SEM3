"""Raw-file discovery and the CSV manifest that preprocessing consumes.

The manifest is a flat, human-inspectable view of what is actually on disk under
`data/raw/<sensor>/`. Timestamps are parsed by each sensor's own reader rather than
by a pattern re-invented here, so the manifest can never disagree with what
`sattsr.io.registry.get_reader` will later parse out of the same filename.

`sattsr.data.index` remains the pipeline's internal source of truth (it carries the
cache locations too); this module is the portable export of the raw side of it.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sattsr.io.registry import get_reader

log = logging.getLogger(__name__)

#: Fixed column contract consumed by `scripts/preprocess.py`. Do not reorder.
MANIFEST_COLUMNS = ("path", "timestamp_utc", "sensor", "scan_mode")

#: Scan modes that INSAT products are split across on MOSDAC.
INSAT_SCAN_MODES = ("routine", "staggered", "rapid_scan")

#: Geostationary full-disc sensors have a single operational scan mode here.
DEFAULT_SCAN_MODE = {"goes19": "full_disk", "himawari8": "full_disk"}

#: Directory spellings that mean a given scan mode, including the ones the existing
#: per-mode configs already use (`data/raw/insat_rapidscan`, `data/raw/insat_staggered`).
_SCAN_MODE_ALIASES = {
    "routine": "routine",
    "staggered": "staggered",
    "insat_staggered": "staggered",
    "rapid_scan": "rapid_scan",
    "rapidscan": "rapid_scan",
    "insat_rapidscan": "rapid_scan",
}


@dataclass(frozen=True)
class ManifestRow:
    """One raw file, ready to be written as a manifest line."""

    path: Path
    timestamp_utc: datetime
    sensor: str
    scan_mode: str

    def as_csv_dict(self, base: Path | None = None) -> dict[str, str]:
        """Render as strings, with POSIX separators so the CSV is platform-portable.

        Paths are written relative to `base` (the manifest's own directory) when
        possible, so the whole `data/raw/<sensor>/` tree stays relocatable and the
        CSV does not bake in one machine's absolute layout.
        """
        path = self.path
        if base is not None:
            try:
                path = path.resolve().relative_to(Path(base).resolve())
            except ValueError:
                path = self.path            # different drive or outside base
        return {
            "path": path.as_posix(),
            "timestamp_utc": self.timestamp_utc.astimezone(timezone.utc).isoformat(),
            "sensor": self.sensor,
            "scan_mode": self.scan_mode,
        }


@dataclass
class ManifestReport:
    """Rows plus everything that went wrong, so nothing is dropped silently."""

    rows: list[ManifestRow] = field(default_factory=list)
    unparsed: list[Path] = field(default_factory=list)
    duplicates: dict[datetime, list[Path]] = field(default_factory=dict)

    def log_summary(self, root: Path) -> None:
        """Log counts, then enumerate every problem file individually."""
        log.info(
            "%s: %d file(s) manifested, %d unparseable, %d duplicated timestamp(s)",
            root,
            len(self.rows),
            len(self.unparsed),
            len(self.duplicates),
        )
        for path in self.unparsed:
            log.warning("excluded (timestamp not parseable): %s", path)
        for ts, paths in sorted(self.duplicates.items()):
            log.warning(
                "duplicate timestamp %s shared by %d files: %s",
                ts.isoformat(),
                len(paths),
                ", ".join(p.name for p in paths),
            )


def infer_scan_mode(path: Path, root: Path, sensor: str, override: str | None = None) -> str:
    """Resolve a file's scan mode from `--scan-mode`, its directory, or the sensor default.

    INSAT files are expected under `data/raw/insat3dr/<scan_mode>/`, but the
    per-mode configs instead use separate roots (`data/raw/insat_rapidscan`), so
    both spellings are recognised.
    """
    if override:
        return override

    candidates = [Path(root).name.lower(), *(p.name.lower() for p in path.parents)]
    for name in candidates:
        if name in _SCAN_MODE_ALIASES:
            return _SCAN_MODE_ALIASES[name]

    return DEFAULT_SCAN_MODE.get(sensor, "routine")


def scan_raw_files(
    root: str | Path, sensor: str, *, scan_mode: str | None = None
) -> ManifestReport:
    """Find every `sensor` file under `root` and parse its timestamp.

    Files whose timestamp cannot be parsed are excluded and recorded in
    `report.unparsed`; files sharing a timestamp are kept and recorded in
    `report.duplicates`. A missing or empty `root` yields an empty report rather
    than raising, so the manifest step is safe to run before anything is downloaded.
    """
    root = Path(root)
    report = ManifestReport()
    if not root.exists():
        log.warning("raw root does not exist: %s", root)
        return report

    reader = get_reader(sensor)
    found: set[Path] = set()
    for pattern in reader.patterns:
        found.update(p for p in root.rglob(pattern) if p.is_file())

    by_timestamp: dict[datetime, list[Path]] = defaultdict(list)
    for path in sorted(found):
        try:
            ts = reader.timestamp_of(path)
        except (ValueError, KeyError, IndexError):
            report.unparsed.append(path)
            continue
        ts = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        by_timestamp[ts].append(path)
        report.rows.append(
            ManifestRow(
                path=path,
                timestamp_utc=ts,
                sensor=sensor,
                scan_mode=infer_scan_mode(path, root, sensor, scan_mode),
            )
        )

    report.rows.sort(key=lambda r: (r.timestamp_utc, r.path.as_posix()))
    report.duplicates = {ts: paths for ts, paths in by_timestamp.items() if len(paths) > 1}
    return report


def write_manifest(report: ManifestReport, out_path: str | Path) -> Path:
    """Write `report.rows` to CSV with the fixed column contract.

    Paths are stored relative to the manifest's own directory where possible; see
    `ManifestRow.as_csv_dict`.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in report.rows:
            writer.writerow(row.as_csv_dict(base=out.parent))
    log.info("wrote %d row(s) to %s", len(report.rows), out)
    return out


def read_manifest(path: str | Path) -> list[ManifestRow]:
    """Read a manifest back, preserving row order.

    Relative paths are resolved against the manifest's own directory, so a moved
    `data/raw/<sensor>/` tree still resolves correctly.
    """
    manifest_path = Path(path)
    base = manifest_path.parent
    rows: list[ManifestRow] = []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            missing = set(MANIFEST_COLUMNS) - set(record)
            if missing:
                raise ValueError(f"{path}: manifest missing column(s) {sorted(missing)}")
            raw_path = Path(record["path"])
            rows.append(
                ManifestRow(
                    path=raw_path if raw_path.is_absolute() else base / raw_path,
                    timestamp_utc=datetime.fromisoformat(record["timestamp_utc"]),
                    sensor=record["sensor"],
                    scan_mode=record["scan_mode"],
                )
            )
    return rows
