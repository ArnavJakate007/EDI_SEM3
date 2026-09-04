"""Brightness-temperature to colour, using a conventional IR enhancement."""

from __future__ import annotations

import numpy as np

NODATA_RGB: tuple[int, int, int] = (20, 20, 30)

# (brightness temperature K, RGB). Warm scenes dark, cold cloud tops bright,
# deep convection below ~235 K in colour so cores are visible in an animation.
IR_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (180.0, (255, 0, 255)),
    (195.0, (255, 90, 0)),
    (205.0, (255, 240, 0)),
    (215.0, (0, 220, 80)),
    (225.0, (0, 220, 255)),
    (235.0, (255, 255, 255)),
    (250.0, (200, 200, 200)),
    (273.0, (110, 110, 110)),
    (310.0, (0, 0, 0)),
)


def bt_to_rgb(bt: np.ndarray, *, vmin: float = 180.0, vmax: float = 310.0) -> np.ndarray:
    """Render Kelvin as an (H, W, 3) uint8 image; NaN becomes NODATA_RGB."""
    a = np.asarray(bt, dtype=np.float64)
    finite = np.isfinite(a)
    clamped = np.clip(np.where(finite, a, vmin), vmin, vmax)

    knots = np.array([k for k, _ in IR_STOPS], dtype=np.float64)
    colours = np.array([c for _, c in IR_STOPS], dtype=np.float64)

    channels = [np.interp(clamped, knots, colours[:, i]) for i in range(3)]
    rgb = np.stack(channels, axis=-1)
    rgb = np.where(finite[..., None], rgb, np.array(NODATA_RGB, dtype=np.float64))
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
