from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from sattsr.config import ModelConfig, NormalizationConfig
from sattsr.infer.recursive import bisect_pair, interpolate_sequence, predict_midframe
from sattsr.io.base import Frame
from sattsr.models.interpolator import build_model

NORM = NormalizationConfig(bt_min=180.0, bt_max=330.0)
CFG = ModelConfig(base_channels=8, scales=[2, 1], use_raft_init=False)
DEVICE = torch.device("cpu")


def _field(size: int = 96, shift: int = 0) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return (250.0 + 20.0 * np.sin((xx + shift) * 0.3) * np.cos(yy * 0.25)).astype(np.float32)


def _frames(n: int = 3, size: int = 96) -> list[Frame]:
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    return [
        Frame(t0 + timedelta(minutes=30 * i), _field(size, shift=4 * i), "insat3dr", Path("x.h5"))
        for i in range(n)
    ]


def test_predict_midframe_returns_kelvin_on_the_full_grid():
    model = build_model(CFG).eval()
    out = predict_midframe(model, _field(), _field(shift=8), device=DEVICE, norm=NORM,
                           tile_size=32, tile_overlap=8)
    assert out.shape == (96, 96)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert 180.0 <= float(out.min()) and float(out.max()) <= 330.0


def test_predict_midframe_handles_a_grid_that_is_not_tile_aligned():
    model = build_model(CFG).eval()
    out = predict_midframe(model, _field(70), _field(70, shift=6), device=DEVICE, norm=NORM,
                           tile_size=32, tile_overlap=8)
    assert out.shape == (70, 70)


def test_predict_midframe_handles_a_grid_smaller_than_one_tile():
    model = build_model(CFG).eval()
    out = predict_midframe(model, _field(24), _field(24, shift=2), device=DEVICE, norm=NORM,
                           tile_size=32, tile_overlap=8)
    assert out.shape == (24, 24)


def test_predict_midframe_preserves_pixels_invalid_in_both_inputs():
    model = build_model(CFG).eval()
    a, b = _field(), _field(shift=8)
    a[:10, :10] = np.nan
    b[:10, :10] = np.nan
    a[20:24, 20:24] = np.nan            # invalid in only one input -> must be filled
    out = predict_midframe(model, a, b, device=DEVICE, norm=NORM, tile_size=32, tile_overlap=8)
    assert np.isnan(out[:10, :10]).all()
    assert np.isfinite(out[20:24, 20:24]).all()


def test_predict_midframe_is_deterministic():
    model = build_model(CFG).eval()
    a, b = _field(), _field(shift=8)
    first = predict_midframe(model, a, b, device=DEVICE, norm=NORM, tile_size=32, tile_overlap=8)
    second = predict_midframe(model, a, b, device=DEVICE, norm=NORM, tile_size=32, tile_overlap=8)
    np.testing.assert_allclose(first, second)


def test_bisect_pair_frame_counts():
    model = build_model(CFG).eval()
    a, b = _field(), _field(shift=8)
    kwargs = {"device": DEVICE, "norm": NORM, "tile_size": 32, "tile_overlap": 8}
    assert len(bisect_pair(model, a, b, factor=2, **kwargs)) == 1
    assert len(bisect_pair(model, a, b, factor=4, **kwargs)) == 3
    assert len(bisect_pair(model, a, b, factor=8, **kwargs)) == 7


def test_bisect_pair_rejects_non_power_of_two_factors():
    model = build_model(CFG).eval()
    with pytest.raises(ValueError):
        bisect_pair(model, _field(), _field(shift=8), factor=3, device=DEVICE, norm=NORM,
                    tile_size=32, tile_overlap=8)


def test_interpolate_sequence_interleaves_and_flags_correctly():
    model = build_model(CFG).eval()
    frames = _frames(3)
    out, flags = interpolate_sequence(frames, model, factor=2, device=DEVICE, norm=NORM,
                                      tile_size=32, tile_overlap=8)
    assert len(out) == 5                       # 3 originals + 2 midpoints
    assert flags == [False, True, False, True, False]
    assert [f.timestamp for f in out] == sorted(f.timestamp for f in out)
    assert out[1].timestamp - out[0].timestamp == timedelta(minutes=15)


def test_interpolate_sequence_at_factor_four_reaches_seven_and_a_half_minutes():
    model = build_model(CFG).eval()
    out, flags = interpolate_sequence(_frames(2), model, factor=4, device=DEVICE, norm=NORM,
                                      tile_size=32, tile_overlap=8)
    assert len(out) == 5
    assert flags == [False, True, True, True, False]
    assert out[1].timestamp - out[0].timestamp == timedelta(seconds=450)   # 7.5 minutes


def test_interpolate_sequence_marks_every_synthetic_frame():
    model = build_model(CFG).eval()
    out, flags = interpolate_sequence(_frames(4), model, factor=2, device=DEVICE, norm=NORM,
                                      tile_size=32, tile_overlap=8)
    assert sum(flags) == 3
    assert all(f.sensor == "insat3dr" for f in out)


def test_interpolate_sequence_needs_at_least_two_frames():
    model = build_model(CFG).eval()
    with pytest.raises(ValueError):
        interpolate_sequence(_frames(1), model, factor=2, device=DEVICE, norm=NORM,
                             tile_size=32, tile_overlap=8)
