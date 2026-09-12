"""Regression tests for GOES-19 cache path construction.

`cache_path_for` used to append `<sensor>` unconditionally, so the shipped
`cache_root: cache/goes19` from `configs/goes19.yaml` produced the nested
`cache/goes19/goes19/<YYYYMMDD>/` seen on disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from sattsr.config import load_config
from sattsr.data.index import FrameRef, cache_path_for

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ref(sensor: str = "goes19") -> FrameRef:
    ts = datetime(2026, 2, 10, 16, 0, 21, tzinfo=timezone.utc)
    return FrameRef(ts, Path(f"raw/{sensor}.nc"), sensor)


def _segments(path: Path) -> list[str]:
    return list(path.parts)


def test_goes19_cache_root_does_not_repeat_the_sensor_segment():
    path = cache_path_for("cache/goes19", _ref())
    assert _segments(path).count("goes19") == 1, f"repeated segment in {path}"
    assert path == Path("cache/goes19/20260210/160021.npy")


def test_shipped_goes19_config_produces_an_unduplicated_cache_path():
    cache_root = load_config(REPO_ROOT / "configs" / "goes19.yaml").data.cache_root
    path = cache_path_for(cache_root, _ref())
    assert _segments(path).count("goes19") == 1, f"repeated segment in {path}"
    assert "goes19/goes19" not in path.as_posix()


@pytest.mark.parametrize("root", ["cache/goes19", "cache/goes19/", "./cache/goes19"])
def test_no_duplication_across_cache_root_spellings(root):
    path = cache_path_for(root, _ref())
    assert _segments(path).count("goes19") == 1, f"repeated segment in {path}"


def test_sensor_segment_is_still_added_when_the_root_does_not_name_it(tmp_path):
    """Guard against fixing the duplication by dropping sensor namespacing entirely."""
    path = cache_path_for(tmp_path / "cache", _ref())
    assert path == tmp_path / "cache" / "goes19" / "20260210" / "160021.npy"


def test_other_sensors_keep_their_own_namespaced_layout(tmp_path):
    path = cache_path_for(tmp_path / "cache" / "himawari", _ref("himawari8"))
    assert path == tmp_path / "cache" / "himawari" / "himawari8" / "20260210" / "160021.npy"
