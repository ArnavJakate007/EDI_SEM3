"""Torch dataset yielding normalised (i0, i1, i2) tile triplets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from sattsr.config import NormalizationConfig
from sattsr.data.index import load_cached
from sattsr.data.radiometry import normalize_bt, valid_mask
from sattsr.data.tiling import TileSpec, extract_tile, tile_positions
from sattsr.data.triplets import Triplet


class TripletDataset(Dataset):
    """Samples one square tile from each triplet, normalised to [0, 1].

    Images are zero-filled where invalid; the accompanying `valid` mask is what the
    loss uses to ignore off-disc pixels. Augmentation (flips, 90-degree rotations and
    temporal reversal) is re-rolled per epoch via `set_epoch`, but is fully determined
    by (seed, epoch, index) so runs stay reproducible.
    """

    def __init__(
        self,
        triplets: Sequence[Triplet],
        norm: NormalizationConfig,
        tile_size: int,
        *,
        augment: bool = True,
        min_valid_fraction: float = 0.5,
        seed: int = 0,
        max_tile_tries: int = 8,
    ) -> None:
        self.triplets = list(triplets)
        self.norm = norm
        self.tile_size = int(tile_size)
        self.augment = augment
        self.min_valid_fraction = float(min_valid_fraction)
        self.seed = int(seed)
        self.max_tile_tries = int(max_tile_tries)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-roll tile choice and augmentation for the next pass."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.triplets)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self._epoch, idx))

    def _pick_tile(
        self, frames: list[np.ndarray], rng: np.random.Generator, idx: int
    ) -> TileSpec:
        specs = tile_positions(frames[0].shape, self.tile_size, overlap=0)
        if not self.augment:
            # Deterministic, but cycled by sample index so a validation pass covers the
            # whole grid instead of scoring the same centre patch over and over.
            return specs[idx % len(specs)]

        best = specs[rng.integers(len(specs))]
        best_score = -1.0
        for _ in range(self.max_tile_tries):
            spec = specs[rng.integers(len(specs))]
            score = float(np.mean([valid_mask(extract_tile(f, spec)).mean() for f in frames]))
            if score >= self.min_valid_fraction:
                return spec
            if score > best_score:
                best, best_score = spec, score
        return best

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        triplet = self.triplets[idx]
        rng = self._rng(idx)

        frames = [
            load_cached(triplet.t0),
            load_cached(triplet.t1),
            load_cached(triplet.t2),
        ]
        if any(f.shape[0] < self.tile_size or f.shape[1] < self.tile_size for f in frames):
            raise ValueError(
                f"tile_size {self.tile_size} exceeds cached frame shape {frames[0].shape}"
            )

        spec = self._pick_tile(frames, rng, idx)
        tiles = [extract_tile(f, spec) for f in frames]
        valid = np.logical_and.reduce([valid_mask(t) for t in tiles]).astype(np.float32)
        images = [normalize_bt(t, self.norm) * valid for t in tiles]
        t_value = float(triplet.t)

        if self.augment:
            if rng.random() < 0.5:
                images = [np.ascontiguousarray(a[:, ::-1]) for a in images]
                valid = np.ascontiguousarray(valid[:, ::-1])
            if rng.random() < 0.5:
                images = [np.ascontiguousarray(a[::-1, :]) for a in images]
                valid = np.ascontiguousarray(valid[::-1, :])
            k = int(rng.integers(4))
            if k:
                images = [np.ascontiguousarray(np.rot90(a, k)) for a in images]
                valid = np.ascontiguousarray(np.rot90(valid, k))
            if rng.random() < 0.5:                       # reverse time
                images = [images[2], images[1], images[0]]
                t_value = 1.0 - t_value

        return {
            "i0": torch.from_numpy(images[0])[None],
            "i1": torch.from_numpy(images[1])[None],
            "i2": torch.from_numpy(images[2])[None],
            "valid": torch.from_numpy(valid)[None],
            "t": torch.tensor(t_value, dtype=torch.float32),
        }
