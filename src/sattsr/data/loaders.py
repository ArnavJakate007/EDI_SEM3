"""DataLoader construction for pretraining, fine-tuning, and held-out evaluation.

Triplet formation, tiling and augmentation all come from the existing
`sattsr.data.triplets` and `sattsr.data.dataset`; nothing here reimplements them.
What this module owns is *which frames end up in which split*, and the two
correctness properties that depend on it.

Design choice 1 -- THE TRAIN/VAL SPLIT IS CHRONOLOGICAL, NOT RANDOM
-------------------------------------------------------------------
`time_based_split` takes the LAST `val_fraction` of each sensor's time range as
validation. It is not a random row shuffle, and that is not an incidental default.

Consecutive triplets overlap: triplet i and triplet i+1 share two of their three
frames. Shuffling rows therefore puts a validation *target* frame into a training
triplet, and the model is scored on frames it has already fitted. Val PSNR then
looks excellent and means nothing.

`sattsr.data.triplets.split_triplets` already avoids that by splitting whole calendar
days, but it shuffles which days become validation. The chronological holdout here is
strictly stronger: it also measures whether the model generalises *forward in time*,
which is the only thing that matters for an operational nowcasting product.

Design choice 2 -- RAPID-SCAN FRAMES ARE HELD OUT AND THE SPLIT IS ENFORCED
---------------------------------------------------------------------------
INSAT-3DR rapid-scan is the project's only native high-cadence ground truth, so it is
what validates the central claim that interpolated frames resemble real observations.
If a rapid-scan frame also appears in fine-tuning, that validation is circular and the
whole evaluation is silently void -- with no symptom in any metric.

`assert_no_split_overlap` is therefore called unconditionally inside
`build_dataloaders` and raises rather than warns. `rapid_scan_guard_minutes` extends
the check from exact timestamp collisions to near-duplicate scenes taken minutes apart.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, default_collate

from sattsr.config import LoaderConfig, NormalizationConfig, PretrainConfig, load_config
from sattsr.data.dataset import TripletDataset
from sattsr.data.index import load_or_scan_index
from sattsr.data.triplets import Triplet, build_triplets

log = logging.getLogger(__name__)

#: Non-tensor, per-sample fields that must survive collation as plain lists.
STRING_FIELDS = ("sensor", "scan_mode")


@dataclass(frozen=True)
class SplitSource:
    """One cache tree contributing to one split."""

    split: str
    sensor: str
    scan_mode: str
    cache_root: Path
    norm: NormalizationConfig
    cadence: timedelta
    tolerance: timedelta
    config_path: Path


class TaggedTripletDataset(Dataset):
    """Wraps `TripletDataset`, attaching the sensor and scan-mode tags.

    A pretraining batch mixes GOES and Himawari samples, and the losses and logging
    need to know which is which. `TripletDataset` deliberately yields only tensors, so
    the tags are added here rather than by modifying it.
    """

    def __init__(self, inner: TripletDataset, sensor: str, scan_mode: str) -> None:
        self.inner = inner
        self.sensor = sensor
        self.scan_mode = scan_mode

    def __len__(self) -> int:
        return len(self.inner)

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch re-rolling to the wrapped dataset."""
        self.inner.set_epoch(epoch)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.inner[idx])
        sample["sensor"] = self.sensor
        sample["scan_mode"] = self.scan_mode
        return sample


def mixed_sensor_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate a possibly mixed-sensor batch.

    Tensor fields go through `default_collate`; the string tags are kept as lists,
    because `default_collate` would try to stack them into a tensor and fail.
    """
    if not batch:
        return {}
    tensor_keys = [k for k in batch[0] if k not in STRING_FIELDS]
    collated: dict[str, Any] = default_collate(
        [{k: sample[k] for k in tensor_keys} for sample in batch]
    )
    for field in STRING_FIELDS:
        if field in batch[0]:
            collated[field] = [sample[field] for sample in batch]
    return collated


def time_based_split(
    triplets: Sequence[Triplet], val_fraction: float
) -> tuple[list[Triplet], list[Triplet]]:
    """Chronological holdout: the last `val_fraction` of the time range becomes val.

    See "Design choice 1" in the module docstring. Splitting on the target frame's
    timestamp, and dropping any training triplet whose span crosses the boundary,
    guarantees no frame appears on both sides.
    """
    if not triplets:
        return [], []
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1); got {val_fraction}")
    if val_fraction == 0.0:
        return list(triplets), []

    ordered = sorted(triplets, key=lambda t: t.t1.timestamp)
    if len(ordered) == 1:
        return ordered, []

    # The boundary is the val_fraction quantile of TRIPLET ORDER, not of the
    # wall-clock range. Archive data is clustered -- this project's GOES set is four
    # isolated days spread over six months -- so cutting the wall-clock range at 15%
    # would hand validation every triplet from the last day, a 33% split. Cutting by
    # position keeps the holdout both chronological and the size actually requested.
    cut = int(len(ordered) * (1.0 - val_fraction))
    cut = min(max(cut, 1), len(ordered) - 1)
    boundary = ordered[cut].t0.timestamp

    # Triplets straddling the boundary belong to neither side: dropping them is what
    # guarantees no frame appears in both.
    train = [t for t in ordered if t.t2.timestamp < boundary]
    val = [t for t in ordered if t.t0.timestamp >= boundary]

    if not train or not val:
        train, val = ordered[: cut - 1] or ordered[:1], ordered[cut:]
    return train, val


def frame_times(triplets: Sequence[Triplet]) -> set[datetime]:
    """Every distinct frame timestamp referenced by `triplets`."""
    times: set[datetime] = set()
    for tri in triplets:
        times.update({tri.t0.timestamp, tri.t1.timestamp, tri.t2.timestamp})
    return times


def assert_no_split_overlap(
    finetune: Sequence[Triplet],
    rapid_scan_eval: Sequence[Triplet],
    *,
    guard_minutes: float = 0.0,
    labels: tuple[str, str] = ("finetune", "insat_rapid_scan_eval"),
) -> None:
    """Raise if any frame is shared, or too close in time, between the two splits.

    This is an assertion rather than a warning on purpose: a leak here invalidates
    the project's entire evaluation claim without changing any metric, so it must
    stop the run.
    """
    a, b = frame_times(finetune), frame_times(rapid_scan_eval)
    shared = a & b
    if shared:
        sample = sorted(shared)[:5]
        raise AssertionError(
            f"{labels[0]} and {labels[1]} share {len(shared)} frame timestamp(s); "
            f"the held-out rapid-scan evaluation would be circular. First: "
            f"{[t.isoformat() for t in sample]}"
        )

    if guard_minutes > 0 and a and b:
        guard = timedelta(minutes=guard_minutes)
        ordered_b = sorted(b)
        import bisect

        for ts in sorted(a):
            i = bisect.bisect_left(ordered_b, ts)
            for j in (i - 1, i):
                if 0 <= j < len(ordered_b) and abs(ordered_b[j] - ts) < guard:
                    raise AssertionError(
                        f"{labels[0]} frame {ts.isoformat()} is within {guard_minutes} "
                        f"min of {labels[1]} frame {ordered_b[j].isoformat()}; these are "
                        f"near-duplicate scenes and would leak into the held-out split."
                    )
    log.info(
        "split guard ok: %s (%d frames) and %s (%d frames) are disjoint",
        labels[0], len(a), labels[1], len(b),
    )


def sources_from_config(cfg: PretrainConfig, *, repo_root: Path) -> list[SplitSource]:
    """Resolve every configured split into concrete cache sources."""
    sources: list[SplitSource] = []
    for split, paths in (
        ("pretrain", cfg.splits.pretrain),
        ("finetune", cfg.splits.finetune),
        ("rapid_scan_eval", cfg.splits.rapid_scan_eval),
    ):
        for rel in paths:
            path = Path(rel)
            if not path.is_absolute():
                path = repo_root / path
            if not path.exists():
                log.warning("skipping %s source %s: config not found", split, path)
                continue
            sensor_cfg = load_config(path)
            cache_root = Path(sensor_cfg.data.cache_root)
            if not cache_root.is_absolute():
                cache_root = repo_root / cache_root
            sources.append(
                SplitSource(
                    split=split,
                    sensor=sensor_cfg.data.sensor,
                    scan_mode=_scan_mode_for(split, path),
                    cache_root=cache_root,
                    norm=sensor_cfg.data.normalization,
                    cadence=timedelta(minutes=sensor_cfg.data.cadence_minutes),
                    tolerance=timedelta(minutes=sensor_cfg.data.tolerance_minutes),
                    config_path=path,
                )
            )
    return sources


def _scan_mode_for(split: str, config_path: Path) -> str:
    """Label a source's scan mode from its split and config filename."""
    name = config_path.stem.lower()
    if "rapid" in name:
        return "rapid_scan"
    if "staggered" in name:
        return "staggered"
    return "full_disk" if split == "pretrain" else "routine"


def triplets_for(source: SplitSource) -> list[Triplet]:
    """Cached frames -> triplets at the source's own cadence."""
    refs = load_or_scan_index(source.cache_root, source.sensor)
    if not refs:
        log.warning("no cached frames for %s under %s", source.sensor, source.cache_root)
        return []
    triplets = build_triplets(refs, step=source.cadence, tolerance=source.tolerance)
    log.info(
        "%s [%s/%s]: %d cached frame(s) -> %d triplet(s)",
        source.split, source.sensor, source.scan_mode, len(refs), len(triplets),
    )
    return triplets


def _make_dataset(
    triplets: Sequence[Triplet], source: SplitSource, loader_cfg: LoaderConfig, *, augment: bool
) -> TaggedTripletDataset:
    inner = TripletDataset(
        triplets,
        norm=source.norm,
        tile_size=loader_cfg.tile_size,
        augment=augment,
        min_valid_fraction=loader_cfg.min_valid_fraction,
        seed=loader_cfg.seed,
    )
    return TaggedTripletDataset(inner, sensor=source.sensor, scan_mode=source.scan_mode)


def _loader(dataset: Dataset, loader_cfg: LoaderConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=shuffle,
        num_workers=loader_cfg.num_workers,
        collate_fn=mixed_sensor_collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def build_dataloaders(
    config: PretrainConfig,
    renorm_registry: Any = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, DataLoader]:
    """Build the train / val / finetune / insat_rapid_scan_eval DataLoaders.

    `renorm_registry` is accepted for interface completeness and recorded on the
    returned loaders' datasets, but renormalisation is applied at *preprocessing*
    time, not per batch -- see the note in the Stage 2 summary.

    Raises `AssertionError` if the fine-tune and held-out rapid-scan splits share
    frames; see "Design choice 2" above.
    """
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    loader_cfg = config.loader
    sources = sources_from_config(config, repo_root=repo_root)

    pretrain_train: list[tuple[list[Triplet], SplitSource]] = []
    pretrain_val: list[tuple[list[Triplet], SplitSource]] = []
    finetune: list[tuple[list[Triplet], SplitSource]] = []
    rapid: list[tuple[list[Triplet], SplitSource]] = []

    for source in sources:
        triplets = triplets_for(source)
        if not triplets:
            continue
        if source.split == "pretrain":
            train_part, val_part = time_based_split(triplets, loader_cfg.val_fraction)
            log.info(
                "  %s: %d train / %d val triplet(s) (chronological holdout)",
                source.sensor, len(train_part), len(val_part),
            )
            if train_part:
                pretrain_train.append((train_part, source))
            if val_part:
                pretrain_val.append((val_part, source))
        elif source.split == "finetune":
            finetune.append((triplets, source))
        else:
            rapid.append((triplets, source))

    # Enforce the held-out guarantee before any loader is handed back.
    assert_no_split_overlap(
        [t for parts, _ in finetune for t in parts],
        [t for parts, _ in rapid for t in parts],
        guard_minutes=loader_cfg.rapid_scan_guard_minutes,
    )

    loaders: dict[str, DataLoader] = {}

    def add(name: str, parts: list[tuple[list[Triplet], SplitSource]], *, augment: bool,
            shuffle: bool) -> None:
        datasets = [_make_dataset(t, s, loader_cfg, augment=augment) for t, s in parts]
        if not datasets:
            log.warning("split %r is empty; no loader built for it", name)
            return
        combined: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        loader = _loader(combined, loader_cfg, shuffle=shuffle)
        loader.renorm_registry = renorm_registry        # type: ignore[attr-defined]
        loaders[name] = loader

    add("train", pretrain_train, augment=True, shuffle=True)
    add("val", pretrain_val, augment=False, shuffle=False)
    add("finetune", finetune, augment=True, shuffle=True)
    add("insat_rapid_scan_eval", rapid, augment=False, shuffle=False)

    log.info(
        "built %d loader(s): %s",
        len(loaders),
        ", ".join(f"{k}={len(v.dataset)} sample(s)" for k, v in loaders.items()),
    )
    return loaders
