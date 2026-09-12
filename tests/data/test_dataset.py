from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from sattsr.config import NormalizationConfig
from sattsr.data.dataset import TripletDataset
from sattsr.data.index import FrameRef
from sattsr.data.triplets import Triplet

NORM = NormalizationConfig(bt_min=180.0, bt_max=330.0)


def _make_triplets(tmp_path: Path, n: int = 4, size: int = 96, nan_border: int = 0):
    t0 = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    refs = []
    for i in range(n + 2):
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        bt = (240.0 + 0.3 * ((xx + i) % size) + 0.2 * yy).astype(np.float32)
        if nan_border:
            bt[:nan_border, :] = np.nan
            bt[:, :nan_border] = np.nan
        path = tmp_path / f"f{i}.npy"
        np.save(path, bt)
        refs.append(FrameRef(t0 + timedelta(minutes=10 * i), path, "goes19"))
    return [Triplet(refs[i], refs[i + 1], refs[i + 2]) for i in range(n)]


def test_item_shapes_dtypes_and_range(tmp_path):
    ds = TripletDataset(_make_triplets(tmp_path), NORM, tile_size=64, augment=False)
    item = ds[0]
    assert set(item) == {"i0", "i1", "i2", "valid", "t"}
    for key in ("i0", "i1", "i2", "valid"):
        assert item[key].shape == (1, 64, 64)
        assert item[key].dtype == torch.float32
    assert 0.0 <= float(item["i0"].min()) and float(item["i0"].max()) <= 1.0
    assert item["t"].shape == ()
    assert float(item["t"]) == pytest.approx(0.5)


def test_length_matches_triplet_count(tmp_path):
    triplets = _make_triplets(tmp_path, n=5)
    assert len(TripletDataset(triplets, NORM, tile_size=64)) == 5


def test_eval_mode_is_deterministic(tmp_path):
    ds = TripletDataset(_make_triplets(tmp_path), NORM, tile_size=64, augment=False)
    torch.testing.assert_close(ds[0]["i0"], ds[0]["i0"])
    torch.testing.assert_close(ds[1]["i1"], ds[1]["i1"])


def test_set_epoch_changes_the_sampled_tile(tmp_path):
    ds = TripletDataset(_make_triplets(tmp_path), NORM, tile_size=64, augment=True, seed=3)
    ds.set_epoch(0)
    a = ds[0]["i0"].clone()
    ds.set_epoch(1)
    b = ds[0]["i0"].clone()
    assert not torch.allclose(a, b)
    ds.set_epoch(0)
    torch.testing.assert_close(ds[0]["i0"], a)


def test_valid_mask_marks_nan_pixels_and_images_are_zero_filled(tmp_path):
    triplets = _make_triplets(tmp_path, size=96, nan_border=40)
    ds = TripletDataset(triplets, NORM, tile_size=64, augment=False, min_valid_fraction=0.0)
    item = ds[0]
    assert float(item["valid"].min()) == 0.0        # some invalid pixels present
    assert torch.isfinite(item["i0"]).all()         # but the image itself has no NaN
    assert float(item["i0"][item["valid"] == 0].abs().max()) == 0.0


def test_temporal_reversal_augmentation_mirrors_t(tmp_path):
    # An asymmetric triplet: t1 sits 40% of the way through the interval.
    triplets = _make_triplets(tmp_path)
    base = triplets[0]
    skewed = Triplet(
        base.t0,
        FrameRef(base.t0.timestamp + timedelta(minutes=8), base.t1.path, "goes19"),
        base.t2,
    )
    ds = TripletDataset([skewed], NORM, tile_size=64, augment=True, seed=0)
    seen = set()
    for epoch in range(12):
        ds.set_epoch(epoch)
        seen.add(round(float(ds[0]["t"]), 3))
    assert seen == {0.4, 0.6}


def test_dataloader_batches_cleanly(tmp_path):
    ds = TripletDataset(_make_triplets(tmp_path, n=4), NORM, tile_size=64, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch["i0"].shape == (2, 1, 64, 64)
    assert batch["t"].shape == (2,)


def test_tile_larger_than_frame_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        TripletDataset(_make_triplets(tmp_path, size=48), NORM, tile_size=64)[0]
