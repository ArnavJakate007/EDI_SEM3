#!/usr/bin/env python
"""Fit cross-sensor radiometric renormalisation maps from cached frames.

Fits a SEPARATE quantile map per scene category (convective / stratiform /
clear_warm) rather than one pooled map per sensor. See the module docstring of
`sattsr.data.normalize` for why: GOES-19 (-75.0 deg) and Himawari-8 (+140.7 deg)
have no overlapping usable-viewing-angle disk at all, so a single pooled fit between
them recovers a regional-climate difference, not a sensor-calibration difference.
Stratifying by scene removes most of that confound; clear warm scenes are the
strongest anchor because their BT tracks the surface rather than cloud tops.

Every fitted map is checked against a plausibility guard and records its own
provenance (frames, geographic bounds, date range) into the saved JSON, so a map can
be audited later instead of being taken on trust.

    python scripts/fit_renorm.py
    python scripts/fit_renorm.py --sensors insat3dr --n-samples-per-sensor 400
    python scripts/fit_renorm.py --scenes clear_warm        # anchor scene only
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sattsr.config import load_config  # noqa: E402
from sattsr.data.index import assert_cache_matches_index, load_or_scan_index  # noqa: E402
from sattsr.data.normalize import (  # noqa: E402
    PLAUSIBLE_MEDIAN_SHIFT_K,
    SCENE_CATEGORIES,
    Provenance,
    RenormRegistry,
    fit_quantile_map,
    scene_samples,
)

log = logging.getLogger("fit_renorm")

#: Which config supplies each sensor's cache_root. Paths come from configs, never
#: hard-coded here.
SENSOR_CONFIGS = {
    "goes19": "goes19.yaml",
    "himawari8": "himawari.yaml",
    "insat3dr": "insat_staggered.yaml",
    "insat3ds": "insat.yaml",
}


def cached_frames(cache_root: Path, sensor: str = "") -> list[Path]:
    """Every cached .npy frame under `cache_root`, sorted for reproducibility.

    Fails loudly on orphans -- files present on disk but absent from index.json.
    Pooling BT samples from stale frames produces a renorm map that is quietly wrong.
    """
    if not cache_root.exists():
        return []
    assert_cache_matches_index(cache_root, sensor)
    return sorted(p for p in cache_root.rglob("*.npy") if p.is_file())


def pool_by_scene(
    frames: list[Path], n_frames: int, *, rng: random.Random, per_frame: int = 20_000
) -> dict[str, np.ndarray]:
    """Pool valid BT values per scene category from a random subset of `frames`.

    Sampling is random rather than the first N chronologically: the first N frames of
    a cache are one contiguous few hours over one region, which is not a
    representative draw from the sensor's BT distribution.
    """
    buckets: dict[str, list[np.ndarray]] = {s: [] for s in SCENE_CATEGORIES}
    if not frames:
        return {s: np.empty(0, dtype=np.float32) for s in SCENE_CATEGORIES}

    chosen = frames if len(frames) <= n_frames else rng.sample(frames, n_frames)
    for path in chosen:
        try:
            arr = np.load(path).astype(np.float32, copy=False)
        except (OSError, ValueError) as exc:
            log.warning("skipping unreadable cache file %s: %s", path.name, exc)
            continue
        for scene, values in scene_samples(arr).items():
            if values.size > per_frame:
                idx = rng.sample(range(values.size), per_frame)
                values = values[np.asarray(idx)]
            if values.size:
                buckets[scene].append(values)

    return {
        s: (np.concatenate(v) if v else np.empty(0, dtype=np.float32))
        for s, v in buckets.items()
    }


def sensor_context(sensor: str, configs_dir: Path) -> tuple[Path | None, dict, str]:
    """(cache_root, lat/lon bounds, config name) for one sensor."""
    name = SENSOR_CONFIGS.get(sensor)
    if name is None:
        return None, {}, ""
    path = configs_dir / name
    if not path.exists():
        log.warning("no config %s for sensor %s", path, sensor)
        return None, {}, ""
    cfg = load_config(path)
    root = Path(cfg.data.cache_root)
    root = root if root.is_absolute() else REPO_ROOT / root
    g = cfg.data.grid
    bounds = {
        "lat_min": float(g.lat_min), "lat_max": float(g.lat_max),
        "lon_min": float(g.lon_min), "lon_max": float(g.lon_max),
    }
    return root, bounds, name


def date_span(cache_root: Path, sensor: str) -> tuple[str, str] | None:
    """First and last cached frame timestamp, for provenance."""
    try:
        refs = load_or_scan_index(cache_root, sensor)
    except (OSError, ValueError):
        return None
    if not refs:
        return None
    stamps = sorted(r.timestamp for r in refs)
    return stamps[0].isoformat(), stamps[-1].isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reference-sensor", default="goes19", choices=sorted(SENSOR_CONFIGS),
        help="Sensor whose radiometry every other sensor is mapped onto. Default goes19.",
    )
    parser.add_argument(
        "--sensors", default=None,
        help="Comma-separated sensors to fit. Default: every non-reference sensor "
             "that has cached frames.",
    )
    parser.add_argument(
        "--scenes", default=",".join(SCENE_CATEGORIES),
        help=f"Comma-separated scene categories to fit. Default: all of "
             f"{','.join(SCENE_CATEGORIES)}. 'clear_warm' alone is the most "
             f"defensible anchor, since its BT tracks the surface rather than "
             f"cloud tops.",
    )
    parser.add_argument(
        "--n-samples-per-sensor", type=int, default=200,
        help="How many cached frames to pool BT samples from, per sensor. Frames are "
             "sampled randomly, not taken in chronological order. Default 200.",
    )
    parser.add_argument(
        "--n-quantiles", type=int, default=256,
        help="Number of quantile breakpoints in the fitted map. Default 256.",
    )
    parser.add_argument(
        "--max-shift-k", type=float, default=PLAUSIBLE_MEDIAN_SHIFT_K,
        help=f"Median-shift plausibility limit in Kelvin (default "
             f"{PLAUSIBLE_MEDIAN_SHIFT_K:g}). Beyond this a fit is almost certainly "
             f"scene-confounded rather than a calibration correction.",
    )
    parser.add_argument(
        "--configs-dir", type=Path, default=REPO_ROOT / "configs",
        help="Directory holding the per-sensor config YAMLs.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "configs" / "renorm",
        help="Where to write the registry. Default configs/renorm/.",
    )
    parser.add_argument(
        "--note", default="",
        help="Free-text note recorded in every fitted map's provenance.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = random.Random(args.seed)
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    ref = args.reference_sensor
    ref_root, ref_bounds, _ = sensor_context(ref, args.configs_dir)
    try:
        ref_frames = cached_frames(ref_root, ref) if ref_root else []
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    if not ref_frames:
        log.error(
            "reference sensor %s has no cached frames under %s -- run "
            "scripts/preprocess.py --sensor %s first", ref, ref_root, ref,
        )
        return 2

    if len(ref_frames) < args.n_samples_per_sensor:
        log.warning("reference %s has only %d cached frame(s) (< %d requested); using all",
                    ref, len(ref_frames), args.n_samples_per_sensor)
    ref_by_scene = pool_by_scene(ref_frames, args.n_samples_per_sensor, rng=rng)
    ref_dates = date_span(ref_root, ref)
    log.info("reference %s: %d frame(s); samples per scene: %s", ref, len(ref_frames),
             {s: int(v.size) for s, v in ref_by_scene.items()})

    targets = (
        [s.strip() for s in args.sensors.split(",") if s.strip()]
        if args.sensors
        else [s for s in sorted(SENSOR_CONFIGS) if s != ref]
    )

    registry = RenormRegistry(reference=ref)
    flagged: list[str] = []

    for sensor in targets:
        if sensor == ref:
            continue
        root, bounds, _ = sensor_context(sensor, args.configs_dir)
        try:
            frames = cached_frames(root, sensor) if root else []
        except ValueError as exc:
            log.error("skipping %s: %s", sensor, exc)
            continue
        if not frames:
            log.warning("skipping %s: no cached frames under %s", sensor, root)
            continue
        if len(frames) < args.n_samples_per_sensor:
            log.warning("%s has only %d cached frame(s) (< %d requested); using all",
                        sensor, len(frames), args.n_samples_per_sensor)

        by_scene = pool_by_scene(frames, args.n_samples_per_sensor, rng=rng)
        provenance = Provenance(
            n_source_frames=len(frames),
            n_target_frames=len(ref_frames),
            source_bounds=bounds,
            target_bounds=ref_bounds,
            source_dates=date_span(root, sensor),
            target_dates=ref_dates,
            notes=args.note,
        )
        log.info("\n%s -> %s", sensor, ref)

        for scene in scenes:
            src, dst = by_scene.get(scene), ref_by_scene.get(scene)
            if src is None or dst is None:
                log.warning("  [%s] unknown scene category; skipped", scene)
                continue
            try:
                qmap = fit_quantile_map(
                    src, dst, sensor=sensor, reference=ref,
                    n_quantiles=args.n_quantiles, scene=scene, provenance=provenance,
                )
            except ValueError as exc:
                log.warning("  [%s] not fitted: %s", scene, exc)
                continue

            registry.add(qmap)
            s = qmap.summary()
            log.info(
                "  [%-11s] %8d vs %8d samples | median %.2f -> %.2f K (%+.2f K) | "
                "IQR ratio %.3f",
                scene, src.size, dst.size, s["median_source_k"],
                s["median_reference_k"], s["median_shift_k"], s["iqr_ratio"],
            )
            for reason in qmap.implausibility(max_shift_k=args.max_shift_k):
                log.warning("      IMPLAUSIBLE: %s", reason)
                flagged.append(f"{sensor}/{scene}")

    if not registry.maps:
        log.error("no maps fitted; nothing written")
        return 1

    registry.save(args.out_dir)
    log.info("wrote maps for %d sensor(s) to %s", len(registry.maps), args.out_dir)

    if flagged:
        log.warning(
            "\n%d map(s) failed the %g K plausibility guard: %s\n"
            "These look like SCENE differences, not sensor calibration. "
            "preprocess.py --renorm require will refuse to use them; pass "
            "--renorm auto to apply anyway (and see each map's provenance block).",
            len(set(flagged)), args.max_shift_k, ", ".join(sorted(set(flagged))),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
