"""Cross-sensor radiometric renormalisation by scene-stratified quantile matching.

`sattsr.data.radiometry.match_statistics` already corrects a *linear* mean/scale
offset between sensors. ABI C13 (10.3 um), AHI B13 (10.4 um) and INSAT TIR1 (10.8 um)
differ in spectral response, so the offset between them is not constant across the
temperature range, and quantile matching handles that.

WHY THE FIT IS STRATIFIED BY SCENE, NOT BY REGION
--------------------------------------------------
A quantile map between two sensors is only a *calibration* correction if both
sensors' samples describe the same physical scene population. Fitting one pooled map
from whatever each sensor happens to look at makes it a *climatology* correction
wearing a calibration costume -- and nothing about the fit reveals the difference.

That is not hypothetical here. GOES-19 sits at -75.0 deg and Himawari-8 at +140.7 deg,
144.3 deg apart. Their usable-viewing-angle disks do not overlap at all: at a 80 deg
satellite-zenith cutoff the intersection is empty, and any intersection at all
requires going past 85 deg, where both sensors are at extreme limb. There is
therefore NO shared sky to co-calibrate on, and a naive pooled fit between the two
recovers "Americas convection vs Indian Ocean convection", not "ABI vs AHI".

So instead of a geographic box, samples are stratified by scene type and a map is
fitted per category. Clear warm scenes are the anchor: their brightness temperature
is close to the surface/sea-surface temperature, which is far more climatologically
stable and far less sensor-geometry dependent than cloud-top temperature.

The classifier is deliberately crude -- a BT threshold plus a local-texture test --
because the point is stratification, not cloud science.

LIMITATION worth knowing: without a land/sea mask, `clear_warm` mixes clear ocean
with clear land. Clear land has a large diurnal and seasonal amplitude that clear
ocean does not, so it is a weaker anchor than true clear-sky ocean. `classify_scene`
accepts an optional `land_mask` so a real mask can be supplied later.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sattsr.config import NormalizationConfig

log = logging.getLogger(__name__)

DEFAULT_N_QUANTILES = 256
_MIN_SAMPLES = 32

#: Scene categories, coldest to warmest.
CONVECTIVE = "convective"
STRATIFORM = "stratiform"
CLEAR_WARM = "clear_warm"
POOLED = "all"
SCENE_CATEGORIES = (CONVECTIVE, STRATIFORM, CLEAR_WARM)

#: Deep convective cloud tops reach the tropopause; ~235 K is the long-standing
#: threshold used for cold-cloud/convective area in IR window channels.
CONVECTIVE_MAX_K = 235.0
#: Above this a pixel is very unlikely to be cloud-topped at 10-11 um.
CLEAR_MIN_K = 280.0
#: Local standard deviation below which a neighbourhood counts as uniform. Cloud
#: fields are texturally busy at this scale; clear surface is not.
UNIFORM_STD_K = 2.0

#: A real inter-calibration offset between two ~10-11 um window channels should be
#: small. GSICS-style inter-calibration of geostationary IR window channels is
#: routinely within about 1 K, and spectral-response differences across 10.3/10.4/10.8
#: um account for at most a couple more on an identical scene. 5 K therefore leaves
#: generous headroom while still catching a scene-confounded fit -- the disjoint-disk
#: GOES/Himawari fit that motivated this module came out at ~18 K.
PLAUSIBLE_MEDIAN_SHIFT_K = 5.0
#: A calibration correction should not rescale the spread much either.
PLAUSIBLE_IQR_RATIO = (0.75, 1.33)


def _local_std(bt: np.ndarray, window: int = 5) -> np.ndarray:
    """Standard deviation in a `window`x`window` neighbourhood, NaN-aware.

    Uses summed-area tables so it stays O(N) regardless of window size.
    """
    a = np.asarray(bt, dtype=np.float64)
    valid = np.isfinite(a)
    filled = np.where(valid, a, 0.0)

    pad = window // 2
    def _boxsum(x: np.ndarray) -> np.ndarray:
        padded = np.pad(x, pad, mode="edge")
        c = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)))
        h, w = x.shape
        return (c[window:window + h, window:window + w] - c[0:h, window:window + w]
                - c[window:window + h, 0:w] + c[0:h, 0:w])

    n = _boxsum(valid.astype(np.float64))
    s = _boxsum(filled)
    s2 = _boxsum(filled * filled)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = np.maximum(s2 / n - mean * mean, 0.0)
    return np.where(n > 1, np.sqrt(var), np.nan)


def classify_scene(
    bt: np.ndarray,
    *,
    land_mask: np.ndarray | None = None,
    window: int = 5,
) -> np.ndarray:
    """Label every pixel with a scene category, or "" where invalid.

    Cheap and deliberately unsophisticated:
      * `convective`  -- BT below `CONVECTIVE_MAX_K` (cold, high cloud top)
      * `clear_warm`  -- BT above `CLEAR_MIN_K` AND locally uniform
      * `stratiform`  -- everything else valid (mid-level / textured cloud)

    `land_mask` (True over land) excludes land from `clear_warm`, which upgrades it
    from "clear surface" to the much more stable "clear-sky ocean". Without a mask,
    clear land is included and the anchor is correspondingly weaker.
    """
    a = np.asarray(bt, dtype=np.float32)
    out = np.full(a.shape, "", dtype=object)
    valid = np.isfinite(a)
    if not valid.any():
        return out

    uniform = _local_std(a, window=window) < UNIFORM_STD_K
    clear = valid & (a > CLEAR_MIN_K) & uniform
    if land_mask is not None:
        clear &= ~np.asarray(land_mask, dtype=bool)

    out[valid] = STRATIFORM
    out[valid & (a < CONVECTIVE_MAX_K)] = CONVECTIVE
    out[clear] = CLEAR_WARM
    return out


def scene_samples(
    bt: np.ndarray, *, land_mask: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Split one frame's valid BT values by scene category."""
    labels = classify_scene(bt, land_mask=land_mask)
    a = np.asarray(bt, dtype=np.float32)
    return {
        scene: a[(labels == scene) & np.isfinite(a)]
        for scene in SCENE_CATEGORIES
    }


@dataclass(frozen=True)
class Provenance:
    """What a fitted map was actually made from, so a saved map is auditable.

    Without this, today's disjoint-disk mismatch is invisible to anyone reading
    `configs/renorm/*.json` six months from now.
    """

    scene: str = POOLED
    n_source_frames: int = 0
    n_target_frames: int = 0
    source_bounds: dict[str, float] = field(default_factory=dict)
    target_bounds: dict[str, float] = field(default_factory=dict)
    source_dates: tuple[str, str] | None = None
    target_dates: tuple[str, str] | None = None
    fitted_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "n_source_frames": self.n_source_frames,
            "n_target_frames": self.n_target_frames,
            "source_bounds": self.source_bounds,
            "target_bounds": self.target_bounds,
            "source_dates": list(self.source_dates) if self.source_dates else None,
            "target_dates": list(self.target_dates) if self.target_dates else None,
            "fitted_at": self.fitted_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> Provenance:
        d = d or {}
        sd, td = d.get("source_dates"), d.get("target_dates")
        return cls(
            scene=d.get("scene", POOLED),
            n_source_frames=int(d.get("n_source_frames", 0)),
            n_target_frames=int(d.get("n_target_frames", 0)),
            source_bounds=d.get("source_bounds", {}) or {},
            target_bounds=d.get("target_bounds", {}) or {},
            source_dates=tuple(sd) if sd else None,
            target_dates=tuple(td) if td else None,
            fitted_at=d.get("fitted_at", ""),
            notes=d.get("notes", ""),
        )

    def describe(self) -> str:
        def box(b):
            if not b:
                return "unknown extent"
            return (f"lat {b.get('lat_min'):.1f}..{b.get('lat_max'):.1f}, "
                    f"lon {b.get('lon_min'):.1f}..{b.get('lon_max'):.1f}")
        src = f"{self.n_source_frames} frame(s), {box(self.source_bounds)}"
        dst = f"{self.n_target_frames} frame(s), {box(self.target_bounds)}"
        return f"scene={self.scene}; source: {src}; reference: {dst}"


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
    scene: str = POOLED
    provenance: Provenance = field(default_factory=Provenance)

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

    def implausibility(
        self, *, max_shift_k: float = PLAUSIBLE_MEDIAN_SHIFT_K
    ) -> list[str]:
        """Reasons this map does not look like a sensor-calibration correction.

        Empty list means it passes. See `PLAUSIBLE_MEDIAN_SHIFT_K` for the rationale
        behind the threshold.
        """
        s = self.summary()
        reasons: list[str] = []
        shift = s["median_shift_k"]
        if abs(shift) > max_shift_k:
            reasons.append(
                f"median shift {shift:+.2f} K exceeds the {max_shift_k:g} K plausibility "
                f"limit for two ~10-11 um window channels"
            )
        ratio = s["iqr_ratio"]
        lo, hi = PLAUSIBLE_IQR_RATIO
        if np.isfinite(ratio) and not (lo <= ratio <= hi):
            reasons.append(
                f"IQR ratio {ratio:.3f} is outside {lo}-{hi}; the two sample "
                f"populations have different spread, which a calibration offset "
                f"cannot explain"
            )
        return reasons

    def is_plausible(self, *, max_shift_k: float = PLAUSIBLE_MEDIAN_SHIFT_K) -> bool:
        """True if this map looks like calibration rather than a scene difference."""
        return not self.implausibility(max_shift_k=max_shift_k)

    def to_dict(self) -> dict:
        return {
            "sensor": self.sensor,
            "reference": self.reference,
            "scene": self.scene,
            "quantiles": [float(v) for v in self.quantiles],
            "source_values": [float(v) for v in self.source_values],
            "target_values": [float(v) for v in self.target_values],
            "n_source_samples": int(self.n_source_samples),
            "n_target_samples": int(self.n_target_samples),
            "summary": self.summary(),
            "implausibility": self.implausibility(),
            "provenance": self.provenance.to_dict(),
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
            scene=payload.get("scene", POOLED),
            provenance=Provenance.from_dict(payload.get("provenance")),
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
    scene: str = POOLED,
    provenance: Provenance | None = None,
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
            f"-> {reference!r} [{scene}]; got {src.size} and {dst.size}"
        )

    quantiles = np.linspace(0.0, 1.0, int(n_quantiles))
    source_values = np.maximum.accumulate(np.quantile(src, quantiles)).astype(np.float32)
    target_values = np.maximum.accumulate(np.quantile(dst, quantiles)).astype(np.float32)

    if provenance is None:
        provenance = Provenance(scene=scene)
    provenance = Provenance(
        scene=scene,
        n_source_frames=provenance.n_source_frames,
        n_target_frames=provenance.n_target_frames,
        source_bounds=provenance.source_bounds,
        target_bounds=provenance.target_bounds,
        source_dates=provenance.source_dates,
        target_dates=provenance.target_dates,
        fitted_at=provenance.fitted_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=provenance.notes,
    )

    return QuantileMap(
        sensor=sensor,
        reference=reference,
        quantiles=quantiles,
        source_values=source_values,
        target_values=target_values,
        n_source_samples=int(src.size),
        n_target_samples=int(dst.size),
        scene=scene,
        provenance=provenance,
    )


@dataclass
class SensorRenorm:
    """Every scene map for one sensor, applied per pixel.

    Picklable (plain data plus numpy arrays), so `PrepareTask` can carry it across
    the preprocessing process pool.
    """

    sensor: str
    reference: str
    maps: dict[str, QuantileMap] = field(default_factory=dict)

    def apply(self, bt: np.ndarray, *, land_mask: np.ndarray | None = None) -> np.ndarray:
        """Renormalise each pixel with the map for the scene it belongs to.

        Pixels whose category has no fitted map fall back to the pooled map, and then
        to passing through unchanged -- never to NaN or zero.
        """
        a = np.asarray(bt, dtype=np.float32)
        if not self.maps:
            return a

        pooled = self.maps.get(POOLED)
        if set(self.maps) == {POOLED}:
            return pooled.apply(a)

        labels = classify_scene(a, land_mask=land_mask)
        out = a.copy()
        for scene in SCENE_CATEGORIES:
            qmap = self.maps.get(scene) or pooled
            if qmap is None:
                continue
            sel = (labels == scene) & np.isfinite(a)
            if sel.any():
                out[sel] = qmap.apply(a[sel])
        return out

    def implausible_scenes(
        self, *, max_shift_k: float = PLAUSIBLE_MEDIAN_SHIFT_K
    ) -> dict[str, list[str]]:
        """Scene -> reasons, for every map that fails the plausibility guard."""
        return {
            scene: reasons
            for scene, qmap in sorted(self.maps.items())
            if (reasons := qmap.implausibility(max_shift_k=max_shift_k))
        }

    def is_plausible(self, *, max_shift_k: float = PLAUSIBLE_MEDIAN_SHIFT_K) -> bool:
        """True only if every scene map passes."""
        return not self.implausible_scenes(max_shift_k=max_shift_k)


class RenormRegistry:
    """sensor -> scene -> QuantileMap, persisted as one JSON file per sensor."""

    def __init__(
        self, reference: str, maps: dict[str, dict[str, QuantileMap]] | None = None
    ) -> None:
        self.reference = reference
        self.maps: dict[str, dict[str, QuantileMap]] = {
            k: dict(v) for k, v in (maps or {}).items()
        }

    def __contains__(self, sensor: str) -> bool:
        return sensor in self.maps

    def __len__(self) -> int:
        return len(self.maps)

    @property
    def sensors(self) -> list[str]:
        return sorted(self.maps)

    def add(self, qmap: QuantileMap) -> None:
        """Register (or replace) one sensor/scene map."""
        if qmap.reference != self.reference:
            raise ValueError(
                f"map for {qmap.sensor!r} targets {qmap.reference!r}, "
                f"but this registry's reference is {self.reference!r}"
            )
        self.maps.setdefault(qmap.sensor, {})[qmap.scene] = qmap

    def for_sensor(self, sensor: str) -> SensorRenorm | None:
        """All of one sensor's scene maps, ready to apply. None if untracked."""
        scenes = self.maps.get(sensor)
        if not scenes:
            return None
        return SensorRenorm(sensor=sensor, reference=self.reference, maps=dict(scenes))

    def apply(self, bt: np.ndarray, sensor: str) -> np.ndarray:
        """Renormalise `bt` onto the reference scale.

        The reference sensor, and any sensor with no fitted map, passes through
        unchanged -- a missing map must never silently zero or distort data.
        """
        if sensor == self.reference:
            return np.asarray(bt, dtype=np.float32)
        renorm = self.for_sensor(sensor)
        if renorm is None:
            log.debug("no renorm map for %s; passing through unchanged", sensor)
            return np.asarray(bt, dtype=np.float32)
        return renorm.apply(bt)

    def save(self, directory: str | Path) -> Path:
        """Write one `<sensor>.json` per sensor, plus a `registry.json` index."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        for sensor, scenes in self.maps.items():
            payload = {
                "sensor": sensor,
                "reference": self.reference,
                "scenes": {s: m.to_dict() for s, m in sorted(scenes.items())},
            }
            (out / f"{sensor}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        (out / "registry.json").write_text(
            json.dumps(
                {
                    "reference": self.reference,
                    "sensors": sorted(self.maps),
                    "scenes_per_sensor": {
                        s: sorted(v) for s, v in sorted(self.maps.items())
                    },
                    "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("saved renorm maps for %d sensor(s) to %s", len(self.maps), out)
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
            for scene_payload in payload.get("scenes", {}).values():
                registry.add(QuantileMap.from_dict(scene_payload))
        return registry


def to_model_range(
    bt: np.ndarray, cfg: NormalizationConfig, *, lo: float = 0.0, hi: float = 1.0
) -> np.ndarray:
    """Kelvin -> model range (default [0, 1]; see the note below before changing).

    The default is [0, 1], NOT the [-1, 1] some RIFE implementations use, because
    `TripletDataset` emits [0, 1] and `FrameInterpolator.forward` clamps its
    prediction to [0, 1]; emitting [-1, 1] would silently clip the negative half away.

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
