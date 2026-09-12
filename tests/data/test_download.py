"""Retry and resumability logic, exercised without touching the network."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from sattsr.data.download import (
    DownloadStats,
    download_keys,
    should_download,
    with_retries,
)
from sattsr.data.mosdac import (
    MosdacCredentials,
    MosdacError,
    date_range,
    expected_filename,
    manual_instructions,
    target_dir,
)


class FakeFS:
    """Minimal stand-in for s3fs: serves bytes from an in-memory key table."""

    def __init__(self, objects: dict[str, bytes], fail_times: int = 0) -> None:
        self.objects = objects
        self.fail_times = fail_times
        self.get_calls: list[str] = []

    def info(self, key: str) -> dict[str, int]:
        return {"size": len(self.objects[key])}

    def get(self, key: str, dest: str) -> None:
        self.get_calls.append(key)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("transient")
        Path(dest).write_bytes(self.objects[key])


# --------------------------------------------------------------------------- retries


def test_with_retries_returns_on_first_success():
    assert with_retries(lambda: 42, sleep=lambda _: None) == 42


def test_with_retries_recovers_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert with_retries(flaky, attempts=3, sleep=lambda _: None) == "ok"
    assert calls["n"] == 3


def test_with_retries_gives_up_and_reraises():
    with pytest.raises(ConnectionError):
        with_retries(
            lambda: (_ for _ in ()).throw(ConnectionError("always")),
            attempts=3,
            sleep=lambda _: None,
        )


def test_backoff_is_exponential():
    delays: list[float] = []

    def always_fails():
        raise TimeoutError("nope")

    with pytest.raises(TimeoutError):
        with_retries(always_fails, attempts=4, base_delay=1.0, sleep=delays.append)

    assert delays == [1.0, 2.0, 4.0], "expected doubling backoff between attempts"


def test_with_retries_rejects_zero_attempts():
    with pytest.raises(ValueError, match="attempts must be"):
        with_retries(lambda: 1, attempts=0)


# ---------------------------------------------------------------------- resumability


def test_should_download_when_absent(tmp_path):
    assert should_download(tmp_path / "missing.nc") is True


def test_should_download_when_empty(tmp_path):
    empty = tmp_path / "empty.nc"
    empty.touch()
    assert should_download(empty) is True


def test_should_skip_when_size_matches(tmp_path):
    f = tmp_path / "ok.nc"
    f.write_bytes(b"12345")
    assert should_download(f, remote_size=5) is False


def test_should_refetch_on_size_mismatch(tmp_path):
    """A truncated file from an interrupted run must heal, not be mistaken for done."""
    f = tmp_path / "partial.nc"
    f.write_bytes(b"12")
    assert should_download(f, remote_size=5) is True


def test_should_skip_when_remote_size_unknown(tmp_path):
    f = tmp_path / "ok.nc"
    f.write_bytes(b"data")
    assert should_download(f, remote_size=None) is False


# ------------------------------------------------------------------- download_keys


def test_download_keys_counts_and_writes(tmp_path):
    fs = FakeFS({"b/a.nc": b"aaa", "b/c.nc": b"cccc"})
    stats = download_keys(fs, list(fs.objects), tmp_path)

    assert (stats.found, stats.downloaded, stats.skipped, stats.failed) == (2, 2, 0, 0)
    assert (tmp_path / "a.nc").read_bytes() == b"aaa"
    assert len(stats.paths) == 2


def test_download_keys_is_resumable(tmp_path):
    fs = FakeFS({"b/a.nc": b"aaa", "b/c.nc": b"cccc"})
    download_keys(fs, list(fs.objects), tmp_path)
    fs.get_calls.clear()

    again = download_keys(fs, list(fs.objects), tmp_path)
    assert (again.downloaded, again.skipped) == (0, 2)
    assert fs.get_calls == [], "already-complete files must not be refetched"


def test_download_keys_refetches_a_truncated_file(tmp_path):
    fs = FakeFS({"b/a.nc": b"aaaaaaa"})
    (tmp_path / "a.nc").write_bytes(b"aa")          # partial from a killed run

    stats = download_keys(fs, ["b/a.nc"], tmp_path)
    assert stats.downloaded == 1
    assert (tmp_path / "a.nc").read_bytes() == b"aaaaaaa"


def test_download_keys_retries_then_succeeds(tmp_path):
    fs = FakeFS({"b/a.nc": b"aaa"}, fail_times=2)
    stats = download_keys(fs, ["b/a.nc"], tmp_path, attempts=3)
    assert stats.downloaded == 1
    assert stats.failed == 0


def test_download_keys_counts_failure_without_aborting_the_batch(tmp_path):
    class HalfBroken(FakeFS):
        def get(self, key, dest):
            if key.endswith("bad.nc"):
                raise ConnectionError("always down")
            super().get(key, dest)

    fs = HalfBroken({"b/bad.nc": b"x", "b/good.nc": b"yy"})
    stats = download_keys(fs, ["b/bad.nc", "b/good.nc"], tmp_path, attempts=2)

    assert stats.failed == 1
    assert stats.downloaded == 1, "one bad key must not sink the rest of the batch"


def test_stats_summary_is_loggable(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="sattsr.data.download"):
        DownloadStats(found=10, downloaded=7, skipped=2, failed=1).log_summary("GOES")
    assert "10 found in range, 7 downloaded, 2 skipped" in caplog.text


# ------------------------------------------------------------------------- MOSDAC


def test_date_range_is_inclusive():
    assert date_range(date(2026, 2, 10), date(2026, 2, 12)) == [
        date(2026, 2, 10), date(2026, 2, 11), date(2026, 2, 12)
    ]


def test_date_range_rejects_reversed_range():
    with pytest.raises(ValueError, match="precedes"):
        date_range(date(2026, 2, 12), date(2026, 2, 10))


def test_credentials_error_names_the_env_vars(monkeypatch):
    monkeypatch.delenv("MOSDAC_USER", raising=False)
    monkeypatch.delenv("MOSDAC_PASS", raising=False)
    with pytest.raises(MosdacError) as exc:
        MosdacCredentials.from_env()
    assert "MOSDAC_USER" in str(exc.value) and "MOSDAC_PASS" in str(exc.value)


def test_credentials_read_from_env(monkeypatch):
    monkeypatch.setenv("MOSDAC_USER", "someone")
    monkeypatch.setenv("MOSDAC_PASS", "secret")
    creds = MosdacCredentials.from_env()
    assert creds.user == "someone"


def test_target_dir_routes_by_scan_mode(tmp_path):
    assert target_dir(tmp_path, "rapid_scan") == tmp_path / "rapid_scan"


def test_target_dir_rejects_unknown_scan_mode(tmp_path):
    with pytest.raises(ValueError, match="unknown scan mode"):
        target_dir(tmp_path, "nonsense")


def test_expected_filename_round_trips_through_the_insat_reader():
    """The manual instructions must name files the real reader can actually parse."""
    from sattsr.io.insat import InsatReader

    name = expected_filename(date(2026, 2, 10), 6, 30, "routine")
    assert name == "3DIMG_10FEB2026_0630_L1C_ASIA_MER.h5"

    parsed = InsatReader().timestamp_of(Path(name))
    assert (parsed.year, parsed.month, parsed.day) == (2026, 2, 10)
    assert (parsed.hour, parsed.minute) == (6, 30)


def test_rapid_scan_filename_also_parses():
    from sattsr.io.insat import InsatReader

    name = expected_filename(date(2026, 2, 10), 6, 34, "rapid_scan")
    assert "SEC_RAPID" in name
    assert InsatReader().timestamp_of(Path(name)).minute == 34


def test_manual_instructions_name_the_destination_and_a_real_filename(tmp_path):
    text = manual_instructions(tmp_path, "staggered", date(2026, 2, 10), date(2026, 2, 12))
    assert (tmp_path / "staggered").as_posix() in text
    assert "3DIMG_10FEB2026_0600_L1C_ASIA_MER.h5" in text
    assert "build_manifest.py" in text, "must tell the user the next pipeline step"
