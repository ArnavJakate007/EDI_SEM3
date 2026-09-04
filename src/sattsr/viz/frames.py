"""PNG frame export and time-lapse animation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from sattsr.io.writer import read_frames_nc
from sattsr.viz.colormap import bt_to_rgb


def save_png(
    bt: np.ndarray, path: str | Path, *, vmin: float = 180.0, vmax: float = 310.0
) -> Path:
    """Write one brightness-temperature array as a colour PNG."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bt_to_rgb(bt, vmin=vmin, vmax=vmax), mode="RGB").save(out)
    return out


def render_run_frames(
    run_dir: str | Path, *, vmin: float = 180.0, vmax: float = 310.0
) -> list[Path]:
    """Render every frame in a run's `output.nc` to `frames/NNN.png`."""
    root = Path(run_dir)
    nc_path = root / "output.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"no output.nc in {root}")

    frames, _ = read_frames_nc(nc_path)
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    return [
        save_png(frame.bt, frames_dir / f"{i:03d}.png", vmin=vmin, vmax=vmax)
        for i, frame in enumerate(frames)
    ]


def make_animation(png_paths: Sequence[str | Path], out_path: str | Path, *, fps: int = 6) -> Path:
    """Combine PNGs into a looping animated GIF."""
    paths = [Path(p) for p in png_paths]
    if not paths:
        raise ValueError("cannot build an animation from zero frames")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in paths]
    try:
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=max(int(1000 / max(fps, 1)), 20),
            loop=0,
            optimize=False,
        )
    finally:
        for image in images:
            image.close()
    return out


def render_run_animations(run_dir: str | Path, *, fps: int = 6) -> dict[str, Path]:
    """Build the two animations the dashboard compares side by side.

    `original` contains only observed frames, at the sensor's native cadence.
    `interpolated` contains every frame, observed and synthesized.
    """
    root = Path(run_dir)
    frames, flags = read_frames_nc(root / "output.nc")
    frames_dir = root / "frames"
    if not frames_dir.exists():
        render_run_frames(root)

    all_pngs = [frames_dir / f"{i:03d}.png" for i in range(len(frames))]
    observed = [p for p, synthetic in zip(all_pngs, flags, strict=True) if not synthetic]

    animations_dir = root / "animations"
    return {
        "original": make_animation(observed, animations_dir / "original.gif", fps=fps),
        "interpolated": make_animation(all_pngs, animations_dir / "interpolated.gif", fps=fps),
    }
