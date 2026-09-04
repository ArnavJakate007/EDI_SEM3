"""Full-grid tiled prediction and recursive temporal bisection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from sattsr.config import NormalizationConfig
from sattsr.data.radiometry import denormalize_bt, normalize_bt
from sattsr.data.tiling import crop_to, extract_tile, pad_to_multiple, reassemble, tile_positions
from sattsr.io.base import Frame
from sattsr.models.interpolator import FrameInterpolator


@torch.no_grad()
def predict_midframe(
    model: FrameInterpolator,
    i0: np.ndarray,
    i2: np.ndarray,
    *,
    device: torch.device,
    norm: NormalizationConfig,
    tile_size: int = 256,
    tile_overlap: int = 32,
    t: float = 0.5,
    batch_size: int = 8,
) -> np.ndarray:
    """Synthesize the frame at normalised time `t` over an arbitrarily large grid."""
    a = np.asarray(i0, dtype=np.float32)
    b = np.asarray(i2, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    multiple = max(model.size_multiple, 1)
    effective_tile = min(tile_size, min(a.shape))
    effective_tile = max((effective_tile // multiple) * multiple, multiple)
    # Cap the overlap at a quarter of the tile: `hann_weight` builds its taper from
    # both edges, and a wider band would make the two ramps collide.
    effective_overlap = max(min(tile_overlap, effective_tile // 4), 0)

    a_pad, original = pad_to_multiple(a, multiple, min_size=effective_tile)
    b_pad, _ = pad_to_multiple(b, multiple, min_size=effective_tile)

    specs = tile_positions(a_pad.shape, effective_tile, effective_overlap)
    tiles_a = [normalize_bt(extract_tile(a_pad, s), norm) for s in specs]
    tiles_b = [normalize_bt(extract_tile(b_pad, s), norm) for s in specs]

    model.eval().to(device)
    predictions: list[np.ndarray] = []
    for start in range(0, len(specs), batch_size):
        chunk = slice(start, start + batch_size)
        x0 = torch.from_numpy(np.stack(tiles_a[chunk]))[:, None].to(device)
        x2 = torch.from_numpy(np.stack(tiles_b[chunk]))[:, None].to(device)
        tt = torch.full((x0.shape[0],), float(t), device=device, dtype=x0.dtype)
        pred = model(x0, x2, tt)["pred"].float().cpu().numpy()[:, 0]
        predictions.extend(pred)

    merged = reassemble(predictions, specs, a_pad.shape, effective_overlap)
    out = denormalize_bt(crop_to(merged, original), norm)

    both_invalid = ~np.isfinite(a) & ~np.isfinite(b)
    return np.where(both_invalid, np.nan, out).astype(np.float32)


def bisect_pair(
    model: FrameInterpolator,
    i0: np.ndarray,
    i2: np.ndarray,
    *,
    factor: int,
    device: torch.device,
    norm: NormalizationConfig,
    tile_size: int = 256,
    tile_overlap: int = 32,
    batch_size: int = 8,
) -> list[np.ndarray]:
    """Insert `factor - 1` frames between two observations by repeated bisection."""
    if factor < 2 or factor & (factor - 1):
        raise ValueError(f"factor must be a power of two >= 2, got {factor}")

    kwargs = {
        "device": device,
        "norm": norm,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "batch_size": batch_size,
    }

    def _recurse(a: np.ndarray, b: np.ndarray, depth: int) -> list[np.ndarray]:
        if depth == 0:
            return []
        mid = predict_midframe(model, a, b, t=0.5, **kwargs)
        return [*_recurse(a, mid, depth - 1), mid, *_recurse(mid, b, depth - 1)]

    return _recurse(i0, i2, factor.bit_length() - 1)


def interpolate_sequence(
    frames: Sequence[Frame],
    model: FrameInterpolator,
    *,
    factor: int,
    device: torch.device,
    norm: NormalizationConfig,
    tile_size: int = 256,
    tile_overlap: int = 32,
    batch_size: int = 8,
) -> tuple[list[Frame], list[bool]]:
    """Densify a whole series, returning the interleaved frames and their synthetic flags."""
    ordered = sorted(frames, key=lambda f: f.timestamp)
    if len(ordered) < 2:
        raise ValueError("need at least two frames to interpolate between")

    out: list[Frame] = []
    flags: list[bool] = []

    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        out.append(left)
        flags.append(False)

        middles = bisect_pair(
            model, left.bt, right.bt, factor=factor, device=device, norm=norm,
            tile_size=tile_size, tile_overlap=tile_overlap, batch_size=batch_size,
        )
        gap = (right.timestamp - left.timestamp) / factor
        for k, bt in enumerate(middles, start=1):
            out.append(
                Frame(
                    timestamp=left.timestamp + gap * k,
                    bt=bt,
                    sensor=left.sensor,
                    source_path=left.source_path,
                )
            )
            flags.append(True)

    out.append(ordered[-1])
    flags.append(False)
    return out, flags
