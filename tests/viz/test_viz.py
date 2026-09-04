from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sattsr.geo.grid import TargetGrid
from sattsr.io.base import Frame
from sattsr.io.writer import write_frames_nc
from sattsr.viz.colormap import NODATA_RGB, bt_to_rgb
from sattsr.viz.frames import make_animation, render_run_animations, render_run_frames, save_png

# --------------------------------------------------------------------------- colormap


def test_output_shape_and_dtype():
    rgb = bt_to_rgb(np.full((4, 5), 250.0, dtype=np.float32))
    assert rgb.shape == (4, 5, 3)
    assert rgb.dtype == np.uint8


def test_warm_scenes_render_darker_than_cold_ones():
    warm = bt_to_rgb(np.full((2, 2), 300.0, dtype=np.float32))
    cold = bt_to_rgb(np.full((2, 2), 240.0, dtype=np.float32))
    assert int(warm.sum()) < int(cold.sum())


def test_very_cold_tops_are_coloured_not_grey():
    core = bt_to_rgb(np.full((2, 2), 200.0, dtype=np.float32))[0, 0]
    assert int(core.max()) - int(core.min()) > 40, "deep convection must be visibly coloured"


def test_nan_renders_as_the_nodata_colour():
    rgb = bt_to_rgb(np.array([[np.nan, 250.0]], dtype=np.float32))
    assert tuple(int(v) for v in rgb[0, 0]) == NODATA_RGB


def test_values_outside_the_range_are_clamped_not_wrapped():
    low = bt_to_rgb(np.array([[100.0]], dtype=np.float32))
    high = bt_to_rgb(np.array([[400.0]], dtype=np.float32))
    assert np.array_equal(low, bt_to_rgb(np.array([[180.0]], dtype=np.float32)))
    assert np.array_equal(high, bt_to_rgb(np.array([[310.0]], dtype=np.float32)))


def test_mapping_is_monotonic_in_the_grey_band():
    values = np.array([[300.0, 280.0, 260.0, 240.0]], dtype=np.float32)
    brightness = bt_to_rgb(values)[0].sum(axis=-1)
    assert np.all(np.diff(brightness) > 0)


# --------------------------------------------------------------------------- frames


def _grid() -> TargetGrid:
    return TargetGrid(lat_min=10.0, lat_max=14.0, lon_min=70.0, lon_max=74.0,
                      resolution_deg=0.25)


def _run(tmp_path: Path, n: int = 5) -> Path:
    grid = _grid()
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    frames, flags = [], []
    for i in range(n):
        bt = np.full(grid.shape, 240.0 + 4.0 * i, dtype=np.float32)
        bt[0, 0] = np.nan
        frames.append(Frame(t0 + timedelta(minutes=15 * i), bt, "insat3dr", Path("x.h5")))
        flags.append(i % 2 == 1)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_frames_nc(run_dir / "output.nc", frames, grid, synthetic=flags, model_version="test")
    return run_dir


def test_save_png_writes_a_readable_image(tmp_path):
    out = save_png(np.full((8, 8), 250.0, dtype=np.float32), tmp_path / "f.png")
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (8, 8)
        assert img.mode == "RGB"


def test_render_run_frames_writes_one_png_per_frame(tmp_path):
    run_dir = _run(tmp_path, n=5)
    paths = render_run_frames(run_dir)
    assert len(paths) == 5
    assert [p.name for p in paths] == ["000.png", "001.png", "002.png", "003.png", "004.png"]
    assert all(p.exists() for p in paths)
    assert all(p.parent == run_dir / "frames" for p in paths)


def test_render_run_frames_requires_an_output_netcdf(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        render_run_frames(empty)


def test_make_animation_produces_a_multi_frame_gif(tmp_path):
    pngs = [save_png(np.full((8, 8), 240.0 + 10 * i, dtype=np.float32), tmp_path / f"{i}.png")
            for i in range(4)]
    gif = make_animation(pngs, tmp_path / "anim.gif", fps=4)
    assert gif.exists()
    with Image.open(gif) as img:
        assert img.n_frames == 4


def test_make_animation_rejects_an_empty_sequence(tmp_path):
    with pytest.raises(ValueError):
        make_animation([], tmp_path / "anim.gif")


def test_render_run_animations_writes_both_required_animations(tmp_path):
    run_dir = _run(tmp_path, n=5)
    render_run_frames(run_dir)
    animations = render_run_animations(run_dir)

    assert set(animations) == {"original", "interpolated"}
    assert animations["original"].exists() and animations["interpolated"].exists()
    with Image.open(animations["original"]) as img:
        assert img.n_frames == 3          # observed frames only
    with Image.open(animations["interpolated"]) as img:
        assert img.n_frames == 5          # observed plus synthesized
