from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sattsr.data.index import FrameRef
from sattsr.data.triplets import Triplet, build_triplets, split_triplets

STEP = timedelta(minutes=10)
TOL = timedelta(minutes=2)


def _refs(n: int, *, step: timedelta = STEP, start_day: int = 4) -> list[FrameRef]:
    t0 = datetime(2026, 9, start_day, 0, 0, tzinfo=timezone.utc)
    return [FrameRef(t0 + step * i, Path(f"f{i}.npy"), "goes19") for i in range(n)]


def test_regular_series_yields_overlapping_triplets():
    triplets = build_triplets(_refs(5), step=STEP, tolerance=TOL)
    assert len(triplets) == 3
    assert triplets[0].t0.timestamp == datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    assert triplets[0].t1.timestamp == datetime(2026, 9, 4, 0, 10, tzinfo=timezone.utc)
    assert triplets[0].t2.timestamp == datetime(2026, 9, 4, 0, 20, tzinfo=timezone.utc)


def test_t_is_the_normalised_midpoint():
    triplet = build_triplets(_refs(3), step=STEP, tolerance=TOL)[0]
    assert triplet.t == pytest.approx(0.5)
    assert triplet.span == timedelta(minutes=20)


def test_triplets_never_span_a_gap():
    # 0, 10, 20 ... then a hole ... then 50. Only the contiguous run forms a triplet.
    refs = _refs(6)
    refs = refs[:3] + refs[5:]
    triplets = build_triplets(refs, step=STEP, tolerance=TOL)
    assert len(triplets) == 1
    assert triplets[0].t2.timestamp == datetime(2026, 9, 4, 0, 20, tzinfo=timezone.utc)


def test_a_series_with_no_contiguous_run_yields_nothing():
    refs = _refs(6)
    refs = refs[:2] + refs[4:]          # 0, 10, 40, 50 -- no run of three steps
    assert build_triplets(refs, step=STEP, tolerance=TOL) == []


def test_tolerance_absorbs_jitter():
    t0 = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    refs = [
        FrameRef(t0, Path("a.npy"), "goes19"),
        FrameRef(t0 + timedelta(minutes=11), Path("b.npy"), "goes19"),
        FrameRef(t0 + timedelta(minutes=19), Path("c.npy"), "goes19"),
    ]
    triplets = build_triplets(refs, step=STEP, tolerance=TOL)
    assert len(triplets) == 1
    assert triplets[0].t == pytest.approx(11.0 / 19.0)


def test_empty_and_short_series_yield_nothing():
    assert build_triplets([], step=STEP, tolerance=TOL) == []
    assert build_triplets(_refs(2), step=STEP, tolerance=TOL) == []


def test_split_is_by_day_so_no_frame_appears_in_both_sides():
    triplets = (
        build_triplets(_refs(20, start_day=4), step=STEP, tolerance=TOL)
        + build_triplets(_refs(20, start_day=5), step=STEP, tolerance=TOL)
        + build_triplets(_refs(20, start_day=6), step=STEP, tolerance=TOL)
    )
    train, val = split_triplets(triplets, val_fraction=0.34, seed=0)
    assert train and val

    def days(items: list[Triplet]) -> set:
        return {t.t0.timestamp.date() for t in items} | {t.t2.timestamp.date() for t in items}

    assert days(train).isdisjoint(days(val))
    assert len(train) + len(val) == len(triplets)


def test_split_is_deterministic_for_a_seed():
    triplets = build_triplets(_refs(30), step=STEP, tolerance=TOL)
    a = split_triplets(triplets, val_fraction=0.2, seed=7)
    b = split_triplets(triplets, val_fraction=0.2, seed=7)
    assert a == b


def test_split_keeps_at_least_one_day_on_each_side_when_possible():
    triplets = (
        build_triplets(_refs(5, start_day=4), step=STEP, tolerance=TOL)
        + build_triplets(_refs(5, start_day=5), step=STEP, tolerance=TOL)
    )
    train, val = split_triplets(triplets, val_fraction=0.01, seed=0)
    assert train and val
