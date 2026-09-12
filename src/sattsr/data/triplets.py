"""Construction of (t0, t1, t2) training samples and leak-free train/val splitting."""

from __future__ import annotations

import bisect
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sattsr.data.index import FrameRef


@dataclass(frozen=True)
class Triplet:
    """Two inputs and the real middle frame the model must reproduce."""

    t0: FrameRef
    t1: FrameRef
    t2: FrameRef

    @property
    def span(self) -> timedelta:
        """Elapsed time between the two input frames."""
        return self.t2.timestamp - self.t0.timestamp

    @property
    def t(self) -> float:
        """Normalised temporal position of the target frame, in (0, 1)."""
        total = self.span.total_seconds()
        return (self.t1.timestamp - self.t0.timestamp).total_seconds() / total


def _nearest(times: list[datetime], target: datetime, tolerance: timedelta) -> int | None:
    """Index of the entry closest to `target`, or None if none is within `tolerance`."""
    if not times:
        return None
    i = bisect.bisect_left(times, target)
    best: int | None = None
    best_gap = tolerance
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(times):
            gap = abs(times[j] - target)
            if gap <= best_gap:
                best, best_gap = j, gap
    return best


def build_triplets(
    refs: Sequence[FrameRef], *, step: timedelta, tolerance: timedelta
) -> list[Triplet]:
    """Form every (t, t+step, t+2*step) triplet the series supports."""
    ordered = sorted(refs, key=lambda r: r.timestamp)
    times = [r.timestamp for r in ordered]

    out: list[Triplet] = []
    for i, ref in enumerate(ordered):
        mid = _nearest(times, ref.timestamp + step, tolerance)
        end = _nearest(times, ref.timestamp + 2 * step, tolerance)
        if mid is None or end is None or not (i < mid < end):
            continue
        out.append(Triplet(ref, ordered[mid], ordered[end]))
    return out


def split_triplets(
    triplets: Sequence[Triplet], *, val_fraction: float, seed: int
) -> tuple[list[Triplet], list[Triplet]]:
    """Split by calendar day so no frame can appear on both sides.

    Adjacent triplets share input frames, so a per-sample random split would put a
    validation target into the training set and inflate every reported metric.
    """
    if not triplets:
        return [], []

    by_day: dict[date, list[Triplet]] = {}
    for tri in triplets:
        by_day.setdefault(tri.t0.timestamp.date(), []).append(tri)

    days = sorted(by_day)
    if len(days) == 1:
        return list(triplets), []

    rng = random.Random(seed)
    shuffled = list(days)
    rng.shuffle(shuffled)

    n_val = max(1, round(len(days) * val_fraction))
    n_val = min(n_val, len(days) - 1)          # always leave a training day
    val_days = set(shuffled[:n_val])

    train = [t for d in days if d not in val_days for t in by_day[d]]
    val = [t for d in days if d in val_days for t in by_day[d]]
    return train, val
