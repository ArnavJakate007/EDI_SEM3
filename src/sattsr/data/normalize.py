"""Cross-sensor radiometric renormalisation by quantile matching.

`sattsr.data.radiometry.match_statistics` already corrects a *linear* mean/scale
offset between sensors. That is the right first-order correction, but ABI C13
(10.3 um), AHI B13 (10.4 um) and INSAT TIR1 (10.8 um) differ in spectral response
function, so the offset between them is not constant across the temperature range:
cold convective tops and warm surface scenes are biased by different amounts.

Quantile matching handles that -- it maps a source sensor's BT distribution onto the
reference sensor's, monotonically, at every temperature. This module is additive:
`match_statistics` remains for the cheap linear case, and a `QuantileMap` is the
distribution-aware upgrade fitted from real cached frames by `scripts/fit_renorm.py`.

Model range
-----------
`to_model_range` / `from_model_range` default to **[0, 1]**, not the [-1, 1] some
RIFE reference implementations use. That is deliberate and load-bearing here:
`TripletDataset` already emits [0, 1] via `normalize_bt`, and
`FrameInterpolator.forward` clamps its prediction with `torch.clamp(..., 0.0, 1.0)`.
Emitting [-1, 1] would have the model silently clip away its entire negative half.
Pass `lo=-1.0, hi=1.0` explicitly if you change that clamp too.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sattsr.config import NormalizationConfig

log = logging.getLogger(__name__)

DEFAULT_N_QUANTILES = 256
_MIN_SAMPLES = 32


@dataclass(frozen=True)
class QuantileMap:
    """Monotone BT -> BT mapping from one sensor onto a reference sensor."""

    sensor: str
    reference: str
    quantiles: np.ndarray
    source_values: np.ndarray
    target_values: np.ndarray
    n_source_samples: int = 0
    n_target_samples: int = 0

    def apply(self, bt: np.ndarray) -> np.ndarray:
        """Map `bt` (Kelvin) onto the reference sensor's scale, preserving NaN.

        Values outside the fitted range are clamped to the endpoint mapping rather
        than extrapolated, so an unusually cold pixel cannot be mapped to a physically
        impossible temperature.
        """
        a = np.asarray(bt, dtype=np.float32)
        finite = np.isfinite(a)
        out = np.full(a.shape, np.nan, dtype=np.float32)
        if np.any(finite):
            out[finite] = np.interp(
                a[finite], self.source_values, self.target_values
            ).astype(np.float32)
        return out

    def summary(self) -> dict[str, float]:
        """Median shift and IQR before/after, for the sanity check at fit time."""
        q = self.quantiles
        median = float(np.interp(0.5, q, self.source_values))
        median_ref = float(np.interp(0.5, q, self.target_values))
        src_iqr = float(
            np.interp(0.75, q, self.source_values) - np.interp(0.25, q, self.source_values)
        )
        dst_iqr = float(
            np.interp(0.75, q, self.target_values) - np.interp(0.25, q, self.target_values)
        )
        return {
            "median_source_k": median,
            "median_reference_k": median_ref,
            "median_shift_k": median_ref - median,
            "iqr_source_k": src_iqr,
            "iqr_reference_k": dst_iqr,
            "iqr_ratio": dst_iqr / src_iqr if src_iqr > 1e-6 else float("nan"),
        }

    def to_dict(self) -> dict:
        return {
            "sensor": self.sensor,
            "reference": self.reference,
            "quantiles": [float(v) for v in self.quantiles],
            "source_values": [float(v) for v in self.source_values],
            "target_values": [float(v) for v in self.target_values],
            "n_source_samples": int(self.n_source_samples),
            "n_target_samples": int(self.n_target_samples),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> QuantileMap:
        return cls(
            sensor=payload["sensor"],
            reference=payload["reference"],
            quantiles=np.asarray(payload["quantiles"], dtype=np.float64),
            source_values=np.asarray(payload["source_values"], dtype=np.float32),
            target_values=np.asarray(payload["target_values"], dtype=np.float32),
            n_source_samples=int(payload.get("n_source_samples", 0)),
            n_target_samples=int(payload.get("n_target_samples", 0)),
        )


def _finite(samples: np.ndarray) -> np.ndarray:
    a = np.asarray(samples, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def fit_quantile_map(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    *,
    sensor: str,
    reference: str,
    n_quantiles: int = DEFAULT_N_QUANTILES,
) -> QuantileMap:
    """Fit a monotone quantile mapping from `source_samples` onto `target_samples`.

    Both arrays are pooled brightness temperatures in Kelvin; NaN is dropped. The
    fitted breakpoints are forced non-decreasing so `np.interp` in `apply` stays
    well-defined even on a degenerate (near-constant) sample.
    """
    src, dst = _finite(source_samples), _finite(target_samples)
    if src.size < _MIN_SAMPLES or dst.size < _MIN_SAMPLES:
        raise ValueError(
            f"need at least {_MIN_SAMPLES} valid samples per side to fit {sensor!r} "
            f"-> {reference!r}; got {src.size} and {dst.size}"
        )

    quantiles = np.linspace(0.0, 1.0, int(n_quantiles))
    source_values = np.maximum.accumulate(np.quantile(src, quantiles)).astype(np.float32)
    target_values = np.maximum.accumulate(np.quantile(dst, quantiles)).astype(np.float32)

    return QuantileMap(
        sensor=sensor,
        reference=reference,
        quantiles=quantiles,
        source_values=source_values,
        target_values=target_values,
        n_source_samples=int(src.size),
        n_target_samples=int(dst.size),
    )


class RenormRegistry:
    """Sensor -> QuantileMap, persisted as one JSON file per sensor."""

    def __init__(self, reference: str, maps: dict[str, QuantileMap] | None = None) -> None:
        self.reference = reference
        self.maps: dict[str, QuantileMap] = dict(maps or {})

    def __contains__(self, sensor: str) -> bool:
        return sensor in self.maps

    def __len__(self) -> int:
        return len(self.maps)

    def add(self, qmap: QuantileMap) -> None:
        """Register (or replace) the map for one sensor."""
        if qmap.reference != self.reference:
            raise ValueError(
                f"map for {qmap.sensor!r} targets {qmap.reference!r}, "
                f"but this registry's reference is {self.reference!r}"
            )
        self.maps[qmap.sensor] = qmap

    def apply(self, bt: np.ndarray, sensor: str) -> np.ndarray:
        """Renormalise `bt` onto the reference scale.

        The reference sensor, and any sensor with no fitted map, passes through
        unchanged -- a missing map must never silently zero or distort data.
        """
        if sensor == self.reference:
            return np.asarray(bt, dtype=np.float32)
        qmap = self.maps.get(sensor)
        if qmap is None:
            log.debug("no renorm map for %s; passing through unchanged", sensor)
            return np.asarray(bt, dtype=np.float32)
        return qmap.apply(bt)

    def save(self, directory: str | Path) -> Path:
        """Write one `<sensor>.json` per map, plus a `registry.json` index."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        for sensor, qmap in self.maps.items():
            (out / f"{sensor}.json").write_text(
                json.dumps(qmap.to_dict(), indent=2), encoding="utf-8"
            )
        (out / "registry.json").write_text(
            json.dumps({"reference": self.reference, "sensors": sorted(self.maps)}, indent=2),
            encoding="utf-8",
        )
        log.info("saved %d renorm map(s) to %s", len(self.maps), out)
        return out

    @classmethod
    def load(cls, directory: str | Path) -> RenormRegistry:
        """Read a registry written by `save`."""
        src = Path(directory)
        index_path = src / "registry.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"no registry.json in {src}; run scripts/fit_renorm.py first"
            )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        registry = cls(reference=index["reference"])
        for sensor in index.get("sensors", []):
            payload = json.loads((src / f"{sensor}.json").read_text(encoding="utf-8"))
            registry.add(QuantileMap.from_dict(payload))
        return registry


def to_model_range(
    bt: np.ndarray, cfg: NormalizationConfig, *, lo: float = 0.0, hi: float = 1.0
) -> np.ndarray:
    """Kelvin -> model range (default [0, 1]; see the module docstring before changing).

    Out-of-range temperatures are clipped and NaN becomes `lo`, matching
    `radiometry.normalize_bt` so the two cannot drift apart.
    """
    a = np.asarray(bt, dtype=np.float32)
    unit = (a - np.float32(cfg.bt_min)) / np.float32(cfg.data_range)
    unit = np.clip(unit, 0.0, 1.0)
    scaled = unit * np.float32(hi - lo) + np.float32(lo)
    return np.where(np.isfinite(scaled), scaled, np.float32(lo)).astype(np.float32)


def from_model_range(
    x: np.ndarray, cfg: NormalizationConfig, *, lo: float = 0.0, hi: float = 1.0
) -> np.ndarray:
    """Model range -> Kelvin. Exact inverse of `to_model_range` within the clip band."""
    a = np.asarray(x, dtype=np.float32)
    unit = (a - np.float32(lo)) / np.float32(hi - lo)
    return (unit * np.float32(cfg.data_range) + np.float32(cfg.bt_min)).astype(np.float32)
