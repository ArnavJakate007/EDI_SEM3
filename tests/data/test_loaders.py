"""DataLoader construction, split enforcement and mixed-sensor collation.

Builds a tiny synthetic cache tree; no real satellite data is required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sattsr.config import LoaderConfig, NormalizationConfig, load_pretrain_config
from sattsr.data.index import FrameRef
from sattsr.data.loaders import (
    assert_no_split_overlap,
    build_dataloaders,
    frame_times,
    mixed_sensor_collate,
    time_based_split,
)
from sattsr.data.triplets import Triplet

T0 = datetime(2026, 2, 10, 0, 0, tzinfo=timezone.utc)
FRAME = 48          # cached frame edge length
TILE = 32


# ----------------------------------------------------------------- synthetic cache


def write_cache(root: Path, sensor: str, start: datetime, n: int, step_minutes: int) -> None:
    """Write `n` cached frames at `<root>/<YYYYMMDD>/<HHMMSS>.npy`."""
    for i in range(n):
        ts = start + timedelta(minutes=step_minutes * i)
        dest = root / ts.strftime("%Y%m%d") / f"{ts.strftime('%H%M%S')}.npy"
        dest.parent.mkdir(parents=True, exist_ok=True)
        yy, xx = np.meshgrid(np.arange(FRAME), np.arange(FRAME), indexing="ij")
        np.save(dest, (240.0 + 0.3 * ((xx + i) % FRAME) + 0.2 * yy).astype(np.float32))


def write_sensor_config(
    path: Path, sensor: str, cache_root: Path, cadence: float, tolerance: float = 2.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "sensor": sensor,
                    "raw_root": str(cache_root),
                    "cache_root": str(cache_root),
                    "cadence_minutes": cadence,
                    "tolerance_minutes": tolerance,
                    "grid": {
                        "lat_min": 0.0, "lat_max": 10.0,
                        "lon_min": 70.0, "lon_max": 80.0,
                        "resolution_deg": 1.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def plan(tmp_path):
    """A full synthetic project: four caches plus the configs that describe them."""
    configs = tmp_path / "configs"
    caches = tmp_path / "cache"

    write_cache(caches / "goes19", "goes19", T0, n=20, step_minutes=10)
    write_cache(caches / "himawari", "himawari8", T0, n=20, step_minutes=10)
    # Staggered and rapid-scan deliberately live on DIFFERENT days, so the default
    # fixture is leak-free and individual tests can introduce a leak on purpose.
    write_cache(caches / "insat_staggered", "insat3dr",
                T0 + timedelta(days=1), n=16, step_minutes=15)
    write_cache(caches / "insat_rapidscan", "insat3dr",
                T0 + timedelta(days=5), n=16, step_minutes=4)

    write_sensor_config(configs / "goes19.yaml", "goes19", caches / "goes19", 10.0)
    write_sensor_config(configs / "himawari.yaml", "himawari8", caches / "himawari", 10.0)
    write_sensor_config(configs / "insat_staggered.yaml", "insat3dr",
                        caches / "insat_staggered", 15.0, tolerance=3.0)
    write_sensor_config(configs / "insat_rapidscan.yaml", "insat3dr",
                        caches / "insat_rapidscan", 4.0, tolerance=1.0)

    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text(
        yaml.safe_dump(
            {
                "splits": {
                    "pretrain": ["configs/goes19.yaml", "configs/himawari.yaml"],
                    "finetune": ["configs/insat_staggered.yaml"],
                    "rapid_scan_eval": ["configs/insat_rapidscan.yaml"],
                },
                "loader": {
                    "batch_size": 4, "num_workers": 0, "tile_size": TILE,
                    "val_fraction": 0.25, "min_valid_fraction": 0.0,
                    "rapid_scan_guard_minutes": 30.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return load_pretrain_config(train_yaml), tmp_path


# ------------------------------------------------------------------ triplet helpers


def _triplet(base: datetime, i: int, step: int, sensor: str = "insat3dr") -> Triplet:
    def ref(k):
        ts = base + timedelta(minutes=step * k)
        return FrameRef(ts, Path(f"{sensor}/{ts:%H%M%S}.npy"), sensor)

    return Triplet(ref(i), ref(i + 1), ref(i + 2))


# ------------------------------------------------------------- chronological split


def test_train_val_split_is_time_ordered_not_random(plan):
    """Every val frame must come strictly after every train frame."""
    triplets = [_triplet(T0, i, 10, "goes19") for i in range(20)]
    train, val = time_based_split(triplets, 0.25)

    assert train and val
    latest_train = max(t.t2.timestamp for t in train)
    earliest_val = min(t.t0.timestamp for t in val)
    assert earliest_val > latest_train, "val must be the chronological tail, not a shuffle"


def test_split_shares_no_frame_between_train_and_val():
    triplets = [_triplet(T0, i, 10, "goes19") for i in range(20)]
    train, val = time_based_split(triplets, 0.25)
    assert not (frame_times(train) & frame_times(val))


def test_split_is_deterministic():
    triplets = [_triplet(T0, i, 10) for i in range(20)]
    assert time_based_split(triplets, 0.2) == time_based_split(triplets, 0.2)


def test_zero_val_fraction_keeps_everything_in_train():
    triplets = [_triplet(T0, i, 10) for i in range(8)]
    train, val = time_based_split(triplets, 0.0)
    assert len(train) == 8 and val == []


def test_split_rejects_a_nonsensical_fraction():
    with pytest.raises(ValueError, match="val_fraction"):
        time_based_split([_triplet(T0, 0, 10)], 1.5)


def test_empty_split_is_safe():
    assert time_based_split([], 0.15) == ([], [])


# --------------------------------------------------------------- leak enforcement


def test_overlapping_splits_raise():
    shared = [_triplet(T0, i, 15) for i in range(4)]
    with pytest.raises(AssertionError, match="share"):
        assert_no_split_overlap(shared, shared)


def test_disjoint_splits_pass():
    finetune = [_triplet(T0, i, 15) for i in range(4)]
    rapid = [_triplet(T0 + timedelta(days=5), i, 4) for i in range(4)]
    assert_no_split_overlap(finetune, rapid, guard_minutes=30.0)


def test_guard_rejects_near_duplicate_scenes():
    """Frames minutes apart are near-duplicates and leak just as effectively.

    The 2-minute offset against a 4-minute cadence never lands on the 15-minute
    finetune grid, so there is no exact collision -- only near ones.
    """
    finetune = [_triplet(T0, i, 15) for i in range(4)]
    rapid = [_triplet(T0 + timedelta(minutes=2), i, 4) for i in range(4)]

    assert not (frame_times(finetune) & frame_times(rapid)), "fixture must not collide exactly"
    assert_no_split_overlap(finetune, rapid, guard_minutes=0.0)      # exact-only: passes
    with pytest.raises(AssertionError, match="near-duplicate"):
        assert_no_split_overlap(finetune, rapid, guard_minutes=30.0)


def test_build_dataloaders_enforces_the_holdout(plan, monkeypatch):
    """Pointing finetune and rapid-scan at the same cache must stop the run."""
    config, root = plan
    config.splits.rapid_scan_eval = config.splits.finetune
    with pytest.raises(AssertionError, match="rapid_scan|share"):
        build_dataloaders(config, None, repo_root=root)


# ------------------------------------------------------------------- collation


def test_collate_keeps_sensor_tags_as_a_list_not_a_tensor():
    batch = [
        {"i0": torch.zeros(1, 4, 4), "t": torch.tensor(0.5),
         "sensor": "goes19", "scan_mode": "full_disk"},
        {"i0": torch.ones(1, 4, 4), "t": torch.tensor(0.5),
         "sensor": "himawari8", "scan_mode": "full_disk"},
    ]
    out = mixed_sensor_collate(batch)

    assert out["sensor"] == ["goes19", "himawari8"]
    assert isinstance(out["sensor"], list)
    assert out["i0"].shape == (2, 1, 4, 4)
    assert out["t"].shape == (2,)


def test_collate_handles_an_empty_batch():
    assert mixed_sensor_collate([]) == {}


# --------------------------------------------------------------- end-to-end build


def test_build_dataloaders_produces_the_four_named_splits(plan):
    config, root = plan
    loaders = build_dataloaders(config, None, repo_root=root)
    assert set(loaders) == {"train", "val", "finetune", "insat_rapid_scan_eval"}


def test_a_training_batch_has_the_expected_shapes_and_dtypes(plan):
    config, root = plan
    loaders = build_dataloaders(config, None, repo_root=root)
    batch = next(iter(loaders["train"]))

    for key in ("i0", "i1", "i2", "valid"):
        assert batch[key].shape[1:] == (1, TILE, TILE), f"{key} has shape {batch[key].shape}"
        assert batch[key].dtype == torch.float32

    assert batch["t"].dtype == torch.float32
    assert batch["t"].ndim == 1
    assert len(batch["sensor"]) == batch["i0"].shape[0]
    assert float(batch["i0"].min()) >= 0.0 and float(batch["i0"].max()) <= 1.0


def test_pretraining_batches_can_mix_sensors(plan):
    """ConcatDataset over GOES+Himawari must be able to yield both in one batch."""
    config, root = plan
    loaders = build_dataloaders(config, None, repo_root=root)
    seen = set()
    for batch in loaders["train"]:
        seen.update(batch["sensor"])
    assert seen == {"goes19", "himawari8"}


def test_rapid_scan_eval_never_shares_a_frame_with_finetune(plan):
    config, root = plan
    loaders = build_dataloaders(config, None, repo_root=root)

    def times(name):
        loader = loaders[name]
        datasets = getattr(loader.dataset, "datasets", [loader.dataset])
        out = set()
        for tagged in datasets:
            out |= frame_times(tagged.inner.triplets)
        return out

    assert not (times("finetune") & times("insat_rapid_scan_eval"))


def test_val_loader_is_not_shuffled_or_augmented(plan):
    config, root = plan
    loaders = build_dataloaders(config, None, repo_root=root)
    datasets = getattr(loaders["val"].dataset, "datasets", [loaders["val"].dataset])
    assert all(not d.inner.augment for d in datasets), "val must be deterministic"


def test_missing_source_config_is_skipped_not_fatal(plan):
    config, root = plan
    config.splits.pretrain = [*config.splits.pretrain, Path("configs/nope.yaml")]
    loaders = build_dataloaders(config, None, repo_root=root)
    assert "train" in loaders


def test_loader_config_defaults_match_the_pipeline():
    cfg = LoaderConfig()
    assert cfg.val_fraction == 0.15
    assert cfg.tile_size == 256
    assert NormalizationConfig().bt_min == 180.0
