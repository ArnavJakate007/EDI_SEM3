"""Manifest building against synthetic raw directories -- no real satellite data."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sattsr.io.manifest import (
    MANIFEST_COLUMNS,
    infer_scan_mode,
    read_manifest,
    scan_raw_files,
    write_manifest,
)
from tests.conftest import make_goes_file, make_himawari_file, make_insat_file

T0 = datetime(2026, 2, 10, 6, 0, tzinfo=timezone.utc)


def _goes_dir(root: Path, n: int = 4) -> Path:
    raw = root / "goes19"
    for i in range(n):
        make_goes_file(raw, T0 + timedelta(minutes=10 * i))
    return raw


def test_manifest_has_the_contract_columns_and_one_row_per_file(tmp_path):
    raw = _goes_dir(tmp_path, n=4)
    report = scan_raw_files(raw, "goes19")
    out = write_manifest(report, tmp_path / "manifest.csv")

    with out.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == MANIFEST_COLUMNS
        rows = list(reader)

    assert len(rows) == 4
    assert {r["sensor"] for r in rows} == {"goes19"}
    assert {r["scan_mode"] for r in rows} == {"full_disk"}


def test_rows_are_sorted_by_timestamp(tmp_path):
    raw = tmp_path / "goes19"
    # Create out of chronological order; the manifest must still come back sorted.
    for minutes in (30, 0, 20, 10):
        make_goes_file(raw, T0 + timedelta(minutes=minutes))

    report = scan_raw_files(raw, "goes19")
    stamps = [r.timestamp_utc for r in report.rows]
    assert stamps == sorted(stamps)
    assert stamps[0] == T0
    assert stamps[-1] == T0 + timedelta(minutes=30)


def test_malformed_filename_is_excluded_and_logged_not_crashed(tmp_path, caplog):
    raw = _goes_dir(tmp_path, n=3)
    bad = raw / "OR_ABI-L1b-RadF-M6C13_G19_NOT_A_TIMESTAMP.nc"
    bad.write_text("not really netcdf", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="sattsr.io.manifest"):
        report = scan_raw_files(raw, "goes19")
        report.log_summary(raw)

    assert len(report.rows) == 3, "the good files must still be manifested"
    assert bad in report.unparsed
    assert bad.name not in {r.path.name for r in report.rows}
    assert any("not parseable" in m for m in caplog.messages)


def test_duplicate_timestamps_are_kept_but_reported(tmp_path, caplog):
    raw = tmp_path / "goes19"
    make_goes_file(raw, T0)
    # Same timestamp, different filename -> same parsed stamp, distinct file.
    original = next(raw.glob("*.nc"))
    duplicate = raw / original.name.replace("_c", "_dup_c")
    duplicate.write_bytes(original.read_bytes())

    with caplog.at_level(logging.WARNING, logger="sattsr.io.manifest"):
        report = scan_raw_files(raw, "goes19")
        report.log_summary(raw)

    assert len(report.rows) == 2, "duplicates are reported, not dropped"
    assert T0 in report.duplicates
    assert any("duplicate timestamp" in m for m in caplog.messages)


def test_empty_and_missing_roots_do_not_raise(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert scan_raw_files(empty, "goes19").rows == []

    report = scan_raw_files(tmp_path / "does_not_exist", "goes19")
    assert report.rows == []
    assert report.unparsed == []


@pytest.mark.parametrize(
    ("directory", "expected"),
    [
        ("rapid_scan", "rapid_scan"),
        ("rapidscan", "rapid_scan"),
        ("insat_rapidscan", "rapid_scan"),
        ("staggered", "staggered"),
        ("insat_staggered", "staggered"),
        ("routine", "routine"),
    ],
)
def test_scan_mode_is_inferred_from_the_directory_name(tmp_path, directory, expected):
    raw = tmp_path / "insat3dr" / directory
    make_insat_file(raw, T0)
    report = scan_raw_files(tmp_path / "insat3dr", "insat3dr")
    assert len(report.rows) == 1
    assert report.rows[0].scan_mode == expected


def test_explicit_scan_mode_overrides_the_directory(tmp_path):
    raw = tmp_path / "insat3dr" / "routine"
    make_insat_file(raw, T0)
    report = scan_raw_files(tmp_path / "insat3dr", "insat3dr", scan_mode="rapid_scan")
    assert report.rows[0].scan_mode == "rapid_scan"


def test_infer_scan_mode_falls_back_to_the_sensor_default(tmp_path):
    assert infer_scan_mode(tmp_path / "f.nc", tmp_path, "goes19") == "full_disk"
    assert infer_scan_mode(tmp_path / "f.nc", tmp_path, "himawari8") == "full_disk"
    assert infer_scan_mode(tmp_path / "f.h5", tmp_path, "insat3dr") == "routine"


def test_himawari_timestamps_parse_through_its_own_reader(tmp_path):
    raw = tmp_path / "himawari8"
    for i in range(3):
        make_himawari_file(raw, T0 + timedelta(minutes=10 * i))
    report = scan_raw_files(raw, "himawari8")
    assert [r.timestamp_utc for r in report.rows] == [
        T0 + timedelta(minutes=10 * i) for i in range(3)
    ]


def test_manifest_round_trips_through_read_manifest(tmp_path):
    raw = _goes_dir(tmp_path, n=3)
    report = scan_raw_files(raw, "goes19")
    out = write_manifest(report, tmp_path / "manifest.csv")

    back = read_manifest(out)
    assert [r.timestamp_utc for r in back] == [r.timestamp_utc for r in report.rows]
    assert [r.sensor for r in back] == ["goes19"] * 3
    assert all(r.path.suffix == ".nc" for r in back)


def test_read_manifest_rejects_a_missing_column(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("path,timestamp_utc\nx.nc,2026-02-10T06:00:00+00:00\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        read_manifest(bad)
