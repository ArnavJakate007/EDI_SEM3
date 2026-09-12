from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from sattsr.data.index import FrameRef, cache_path_for
from sattsr.data.prune import prune_verified_raw, subsample_index


def _refs(tmp_path: Path, n: int = 6) -> list[FrameRef]:
    t0 = datetime(2026, 9, 4, tzinfo=timezone.utc)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    refs = []
    for i in range(n):
        path = raw / f"f{i}.nc"
        path.write_bytes(b"raw-bytes")
        refs.append(FrameRef(t0 + timedelta(minutes=10 * i), path, "goes19"))
    return refs


def _cache(tmp_path: Path, refs, indices) -> Path:
    cache_root = tmp_path / "cache"
    for i in indices:
        dest = cache_path_for(cache_root, refs[i])
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, np.zeros((4, 4), dtype=np.float32))
    return cache_root


def test_subsample_keeps_every_nth_frame(tmp_path):
    refs = _refs(tmp_path, 10)
    kept = subsample_index(refs, keep_every=3)
    assert len(kept) == 4
    assert [refs.index(r) for r in kept] == [0, 3, 6, 9]


def test_subsample_of_one_is_the_identity(tmp_path):
    refs = _refs(tmp_path, 5)
    assert subsample_index(refs, keep_every=1) == refs


def test_subsample_rejects_a_non_positive_stride(tmp_path):
    with pytest.raises(ValueError):
        subsample_index(_refs(tmp_path, 3), keep_every=0)


def test_dry_run_is_the_default_and_deletes_nothing(tmp_path):
    refs = _refs(tmp_path, 4)
    cache_root = _cache(tmp_path, refs, range(4))
    removed, skipped = prune_verified_raw(refs, cache_root)
    assert len(removed) == 4 and skipped == []
    assert all(r.path.exists() for r in refs), "dry run must not touch the filesystem"


def test_deletion_removes_only_verified_files(tmp_path):
    refs = _refs(tmp_path, 4)
    cache_root = _cache(tmp_path, refs, [0, 2])         # only two are cached
    removed, skipped = prune_verified_raw(refs, cache_root, dry_run=False)

    assert {p.name for p in removed} == {"f0.nc", "f2.nc"}
    assert {p.name for p in skipped} == {"f1.nc", "f3.nc"}
    assert not refs[0].path.exists() and not refs[2].path.exists()
    assert refs[1].path.exists() and refs[3].path.exists()


def test_an_empty_cache_file_does_not_count_as_verified(tmp_path):
    refs = _refs(tmp_path, 2)
    cache_root = tmp_path / "cache"
    dest = cache_path_for(cache_root, refs[0])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"")

    removed, skipped = prune_verified_raw(refs, cache_root, dry_run=False)
    assert removed == []
    assert len(skipped) == 2
    assert all(r.path.exists() for r in refs)


def test_an_already_missing_raw_file_is_skipped_quietly(tmp_path):
    refs = _refs(tmp_path, 2)
    cache_root = _cache(tmp_path, refs, [0, 1])
    refs[0].path.unlink()
    removed, skipped = prune_verified_raw(refs, cache_root, dry_run=False)
    assert [p.name for p in removed] == ["f1.nc"]
    assert [p.name for p in skipped] == ["f0.nc"]
