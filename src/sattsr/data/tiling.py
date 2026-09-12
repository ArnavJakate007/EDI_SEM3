"""Tiled processing of grids larger than the model's receptive window."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    """Top-left corner and edge length of one square tile."""

    row: int
    col: int
    size: int


def _positions(extent: int, size: int, stride: int) -> list[int]:
    starts = list(range(0, max(extent - size, 0) + 1, stride))
    if starts[-1] + size < extent:
        starts.append(extent - size)
    return starts


def tile_positions(shape: tuple[int, int], size: int, overlap: int) -> list[TileSpec]:
    """Cover `shape` with overlapping `size`-square tiles, all fully inside the array."""
    h, w = shape
    if size > h or size > w:
        raise ValueError(f"tile size {size} exceeds array shape {shape}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must be in [0, {size}), got {overlap}")

    stride = size - overlap
    return [
        TileSpec(row=r, col=c, size=size)
        for r in _positions(h, size, stride)
        for c in _positions(w, size, stride)
    ]


def extract_tile(arr: np.ndarray, spec: TileSpec) -> np.ndarray:
    """Copy one tile out of `arr`."""
    return np.asarray(arr)[spec.row:spec.row + spec.size, spec.col:spec.col + spec.size].copy()


def hann_weight(size: int, overlap: int) -> np.ndarray:
    """Separable Hann taper over the overlap band; strictly positive everywhere.

    The taper is offset by one sample so the outermost row and column keep a small
    non-zero weight, which avoids a divide-by-zero at the array border.
    """
    if overlap <= 0:
        return np.ones((size, size), dtype=np.float32)

    ramp = np.hanning(2 * overlap + 3)[1:overlap + 2]     # rises from >0 to exactly 1
    profile = np.ones(size, dtype=np.float64)
    profile[:overlap + 1] = ramp
    profile[size - overlap - 1:] = ramp[::-1]
    return np.outer(profile, profile).astype(np.float32)


def reassemble(
    tiles: Sequence[np.ndarray],
    specs: Sequence[TileSpec],
    shape: tuple[int, int],
    overlap: int,
) -> np.ndarray:
    """Blend overlapping tiles back into one array using Hann weights."""
    if len(tiles) != len(specs):
        raise ValueError(f"{len(tiles)} tiles but {len(specs)} specs")

    acc = np.zeros(shape, dtype=np.float64)
    wsum = np.zeros(shape, dtype=np.float64)
    cache: dict[int, np.ndarray] = {}

    for tile, spec in zip(tiles, specs, strict=True):
        w = cache.setdefault(spec.size, hann_weight(spec.size, overlap))
        sl = (slice(spec.row, spec.row + spec.size), slice(spec.col, spec.col + spec.size))
        acc[sl] += np.asarray(tile, dtype=np.float64) * w
        wsum[sl] += w

    return np.divide(acc, np.maximum(wsum, 1e-8)).astype(np.float32)


def pad_to_multiple(
    arr: np.ndarray, multiple: int, *, min_size: int = 0
) -> tuple[np.ndarray, tuple[int, int]]:
    """Edge-pad so both dimensions are >= `min_size` and divisible by `multiple`.

    Returns the padded array and the original (H, W) so `crop_to` can undo it.
    """
    a = np.asarray(arr)
    h, w = a.shape[-2], a.shape[-1]
    target_h = max(min_size, ((h + multiple - 1) // multiple) * multiple)
    target_w = max(min_size, ((w + multiple - 1) // multiple) * multiple)
    target_h = ((target_h + multiple - 1) // multiple) * multiple
    target_w = ((target_w + multiple - 1) // multiple) * multiple

    if (target_h, target_w) == (h, w):
        return a, (h, w)
    pad = ((0, target_h - h), (0, target_w - w))
    return np.pad(a, pad, mode="edge"), (h, w)


def crop_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Undo `pad_to_multiple`."""
    return np.asarray(arr)[..., : shape[0], : shape[1]]
