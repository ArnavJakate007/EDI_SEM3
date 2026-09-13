"""Sensor-balanced batch sampling."""

from __future__ import annotations

from collections import Counter

import pytest
from torch.utils.data import ConcatDataset

from sattsr.data.sampler import BalancedSensorBatchSampler, sensor_labels

LOPSIDED = ["goes19"] * 120 + ["himawari8"] * 12


class _Tagged:
    """Minimal stand-in for TaggedTripletDataset."""

    def __init__(self, sensor: str, n: int) -> None:
        self.sensor, self._n = sensor, n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):        # pragma: no cover - never sampled here
        return i


def _sampler(labels=LOPSIDED, batch_size=8, **kw):
    return BalancedSensorBatchSampler(labels, batch_size, seed=0, **kw)


# ------------------------------------------------------------------- labels


def test_labels_come_from_a_concat_dataset_in_order():
    ds = ConcatDataset([_Tagged("goes19", 3), _Tagged("himawari8", 2)])
    assert sensor_labels(ds) == ["goes19"] * 3 + ["himawari8"] * 2


def test_labels_from_a_single_tagged_dataset():
    assert sensor_labels(_Tagged("goes19", 4)) == ["goes19"] * 4


def test_an_untagged_dataset_is_rejected_loudly():
    class Bare:
        def __len__(self):
            return 2

    with pytest.raises(TypeError, match="sensor"):
        sensor_labels(Bare())


# ------------------------------------------------------------------ balance


def test_every_batch_holds_an_equal_share_of_each_sensor():
    s = _sampler()
    for batch in s:
        counts = Counter(s.labels[i] for i in batch)
        assert counts["goes19"] == 4 and counts["himawari8"] == 4


def test_the_minority_sensor_is_oversampled_to_parity_over_an_epoch():
    """120 vs 12 triplets must still yield a 50/50 epoch."""
    s = _sampler()
    total = s.composition()
    assert total["goes19"] == total["himawari8"]


def test_uneven_batch_size_splits_deterministically():
    s = _sampler(batch_size=7)
    counts = Counter(s.labels[i] for i in next(iter(s)))
    assert sum(counts.values()) == 7
    assert abs(counts["goes19"] - counts["himawari8"]) == 1


def test_three_sensors_each_get_a_third():
    labels = ["a"] * 30 + ["b"] * 10 + ["c"] * 5
    s = BalancedSensorBatchSampler(labels, 6, seed=0)
    for batch in s:
        assert Counter(s.labels[i] for i in batch) == {"a": 2, "b": 2, "c": 2}


def test_epoch_length_follows_the_largest_sensor():
    """The majority sensor should still be seen once per epoch, not truncated."""
    s = _sampler()
    assert len(s) == pytest.approx(120 / 4, abs=1)


def test_epoch_length_min_undersamples_the_majority_instead():
    s = _sampler(epoch_length="min")
    assert len(s) == pytest.approx(12 / 4, abs=1)


# ------------------------------------------------------------- determinism


def test_the_same_epoch_reproduces_the_same_batches():
    a, b = _sampler(), _sampler()
    a.set_epoch(3)
    b.set_epoch(3)
    assert [list(x) for x in a] == [list(x) for x in b]


def test_different_epochs_draw_different_batches():
    s = _sampler()
    s.set_epoch(0)
    first = [list(x) for x in s]
    s.set_epoch(1)
    assert [list(x) for x in s] != first, "set_epoch must re-roll the shuffle"


def test_every_majority_sample_is_used_within_an_epoch():
    s = _sampler()
    seen = {i for batch in s for i in batch if s.labels[i] == "goes19"}
    assert len(seen) == 120, "the majority sensor should be covered, not subsampled"


def test_batches_are_not_sensor_ordered():
    """Samples are interleaved, so a batch is not goes-block-then-himawari-block."""
    s = _sampler()
    batch = next(iter(s))
    labels = [s.labels[i] for i in batch]
    assert labels != sorted(labels, key=lambda x: x != "goes19")


# ------------------------------------------------------------------ guards


def test_a_zero_batch_size_is_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        _sampler(batch_size=0)


def test_an_unknown_epoch_length_mode_is_rejected():
    with pytest.raises(ValueError, match="epoch_length"):
        _sampler(epoch_length="sideways")


def test_a_single_sensor_still_works():
    s = BalancedSensorBatchSampler(["goes19"] * 20, 4, seed=0)
    assert sum(len(b) for b in s) == 20
