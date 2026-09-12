"""Bounded storage: temporal subsampling and deletion of verified raw files."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from sattsr.data.index import FrameRef, cache_path_for

log = logging.getLogger(__name__)


def subsample_index(refs: Sequence[FrameRef], *, keep_every: int) -> list[FrameRef]:
    """Keep every `keep_every`-th frame, preserving order."""
    if keep_every < 1:
        raise ValueError(f"keep_every must be >= 1, got {keep_every}")
    return list(refs)[::keep_every]


def prune_verified_raw(
    refs: Sequence[FrameRef], cache_root: str | Path, *, dry_run: bool = True
) -> tuple[list[Path], list[Path]]:
    """Delete raw files whose regridded cache entry exists and is non-empty.

    Defaults to a dry run: nothing is deleted unless `dry_run=False`. A raw file is
    only ever removed after its cache entry has been confirmed present and non-empty,
    so a failed `prepare` can never cascade into data loss.

    Returns (removed, skipped) as raw-file paths.
    """
    removed: list[Path] = []
    skipped: list[Path] = []

    for ref in refs:
        cached = cache_path_for(cache_root, ref)
        verified = cached.is_file() and cached.stat().st_size > 0
        if not verified or not ref.path.exists():
            skipped.append(ref.path)
            continue

        removed.append(ref.path)
        if not dry_run:
            ref.path.unlink()
            log.info("removed verified raw file %s", ref.path)

    return removed, skipped
