from __future__ import annotations

import numpy as np
import pytest

from sattsr.data.tiling import (
    TileSpec,
    crop_to,
    extract_tile,
    hann_weight,
    pad_to_multiple,
    reassemble,
    tile_positions,
)


def test_tile_positions_cover_the_whole_array():
    specs = tile_positions((300, 500), size=128, overlap=32)
    covered = np.zeros((300, 500), dtype=bool)
    for s in specs:
        covered[s.row:s.row + s.size, s.col:s.col + s.size] = True
    assert covered.all()
    assert all(s.row + s.size <= 300 and s.col + s.size <= 500 for s in specs)


def test_tile_positions_exact_fit_has_no_duplicates():
    specs = tile_positions((256, 256), size=128, overlap=0)
    assert len(specs) == 4
    assert len({(s.row, s.col) for s in specs}) == 4


def test_tile_larger_than_array_is_rejected():
    with pytest.raises(ValueError):
        tile_positions((64, 64), size=128, overlap=16)


def test_overlap_not_smaller_than_size_is_rejected():
    with pytest.raises(ValueError):
        tile_positions((256, 256), size=64, overlap=64)


def test_extract_tile_returns_the_right_window():
    arr = np.arange(100).reshape(10, 10).astype(np.float32)
    tile = extract_tile(arr, TileSpec(row=2, col=3, size=4))
    np.testing.assert_array_equal(tile, arr[2:6, 3:7])


def test_hann_weight_is_strictly_positive():
    w = hann_weight(64, 16)
    assert w.shape == (64, 64)
    assert w.min() > 0.0
    assert w.max() == pytest.approx(1.0)


def test_reassembly_reconstructs_a_smooth_field_exactly():
    rng = np.random.default_rng(0)
    full = rng.normal(250.0, 5.0, size=(300, 500)).astype(np.float32)
    specs = tile_positions(full.shape, size=128, overlap=32)
    tiles = [extract_tile(full, s) for s in specs]
    out = reassemble(tiles, specs, full.shape, overlap=32)
    assert out.shape == full.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, full, atol=1e-3)


def test_reassembly_has_no_visible_seam():
    # a constant field must come back constant, including across tile boundaries
    full = np.full((300, 500), 250.0, dtype=np.float32)
    specs = tile_positions(full.shape, size=128, overlap=32)
    out = reassemble([extract_tile(full, s) for s in specs], specs, full.shape, overlap=32)
    assert float(np.abs(out - 250.0).max()) < 1e-3


def test_pad_to_multiple_and_crop_round_trip():
    arr = np.arange(35 * 53).reshape(35, 53).astype(np.float32)
    padded, original = pad_to_multiple(arr, 32, min_size=64)
    assert padded.shape[0] % 32 == 0 and padded.shape[1] % 32 == 0
    assert padded.shape[0] >= 64 and padded.shape[1] >= 64
    assert original == (35, 53)
    np.testing.assert_array_equal(crop_to(padded, original), arr)


def test_pad_to_multiple_leaves_conforming_arrays_alone():
    arr = np.zeros((64, 128), dtype=np.float32)
    padded, original = pad_to_multiple(arr, 32, min_size=64)
    assert padded.shape == (64, 128)
    assert original == (64, 128)
