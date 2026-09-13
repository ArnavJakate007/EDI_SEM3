#!/usr/bin/env python
"""Fit cross-sensor radiometric renormalisation maps from cached frames.

Pools brightness-temperature samples from a random sample of cached frames for the
reference sensor and each target sensor, fits a monotone quantile mapping onto the
reference, and saves the registry to configs/renorm/.

The fitted summary stats are printed deliberately: a renorm map that is subtly wrong
does not fail here, it quietly poisons fine-tuning three steps later. A median shift
of a few Kelvin between ABI C13 and INSAT TIR1 is expected; tens of Kelvin, or an
IQR ratio far from 1, means something upstream is wrong.

    python scripts/fit_renorm.py
    python scripts/fit_renorm.py --sensors insat3dr --n-samples-per-sensor 400
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
from sattsr.data.index import assert_cache_matches_index  # noqa: E402
from sattsr.data.normalize import RenormRegistry, fit_quantile_map  # noqa: E402

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
    Pooling BT samples from stale frames produces a renorm map that is quietly wrong,
    and a wrong map poisons fine-tuning several steps later with no visible symptom.
    """
    if not cache_root.exists():
        return []
    assert_cache_matches_index(cache_root, sensor)
    return sorted(p for p in cache_root.rglob("*.npy") if p.is_file())


def pool_samples(
    frames: list[Path], n_frames: int, *, rng: random.Random, per_frame: int = 20_000
) -> np.ndarray:
    """Pool valid BT values from a random subset of `frames`.

    Sampling is random rather than the first N chronologically: the first N frames of
    a cache are one contiguous few hours over one region, which is not a
    representative draw from the sensor's BT distribution.
    """
    if not frames:
        return np.empty(0, dtype=np.float32)

    chosen = frames if len(frames) <= n_frames else rng.sample(frames, n_frames)
    parts: list[np.ndarray] = []
    for path in chosen:
        try:
            arr = np.load(path).astype(np.float32, copy=False).ravel()
        except (OSError, ValueError) as exc:
            log.warning("skipping unreadable cache file %s: %s", path.name, exc)
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size > per_frame:
            idx = rng.sample(range(arr.size), per_frame)
            arr = arr[np.asarray(idx)]
        parts.append(arr)

    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def resolve_cache_root(sensor: str, configs_dir: Path) -> Path | None:
    """Read a sensor's cache_root out of its config."""
    name = SENSOR_CONFIGS.get(sensor)
    if name is None:
        return None
    path = configs_dir / name
    if not path.exists():
        log.warning("no config %s for sensor %s", path, sensor)
        return None
    root = Path(load_config(path).data.cache_root)
    return root if root.is_absolute() else REPO_ROOT / root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--n-samples-per-sensor", type=int, default=200,
        help="How many cached frames to pool BT samples from, per sensor. Frames are "
             "sampled randomly, not taken in chronological order. Default 200. "
             "A sensor with fewer cached frames uses what it has and warns.",
    )
    parser.add_argument(
        "--n-quantiles", type=int, default=256,
        help="Number of quantile breakpoints in the fitted map. Default 256.",
    )
    parser.add_argument(
        "--configs-dir", type=Path, default=REPO_ROOT / "configs",
        help="Directory holding the per-sensor config YAMLs.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "configs" / "renorm",
        help="Where to write the registry. Default configs/renorm/.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = random.Random(args.seed)

    ref_root = resolve_cache_root(args.reference_sensor, args.configs_dir)
    try:
        ref_frames = cached_frames(ref_root, args.reference_sensor) if ref_root else []
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    if not ref_frames:
        log.error(
            "reference sensor %s has no cached frames under %s -- run "
            "scripts/preprocess.py --sensor %s first",
            args.reference_sensor, ref_root, args.reference_sensor,
        )
        return 2

    if len(ref_frames) < args.n_samples_per_sensor:
        log.warning(
            "reference %s has only %d cached frame(s), fewer than the requested %d; "
            "using all of them",
            args.reference_sensor, len(ref_frames), args.n_samples_per_sensor,
        )
    ref_samples = pool_samples(ref_frames, args.n_samples_per_sensor, rng=rng)
    log.info(
        "reference %s: %d frame(s) available, %d BT sample(s) pooled",
        args.reference_sensor, len(ref_frames), ref_samples.size,
    )

    if args.sensors:
        targets = [s.strip() for s in args.sensors.split(",") if s.strip()]
    else:
        targets = [s for s in sorted(SENSOR_CONFIGS) if s != args.reference_sensor]

    registry = RenormRegistry(reference=args.reference_sensor)
    for sensor in targets:
        if sensor == args.reference_sensor:
            continue
        root = resolve_cache_root(sensor, args.configs_dir)
        try:
            frames = cached_frames(root, sensor) if root else []
        except ValueError as exc:
            log.error("skipping %s: %s", sensor, exc)
            continue
        if not frames:
            log.warning("skipping %s: no cached frames under %s", sensor, root)
            continue
        if len(frames) < args.n_samples_per_sensor:
            log.warning(
                "%s has only %d cached frame(s), fewer than the requested %d; "
                "using all of them", sensor, len(frames), args.n_samples_per_sensor,
            )

        samples = pool_samples(frames, args.n_samples_per_sensor, rng=rng)
        try:
            qmap = fit_quantile_map(
                samples, ref_samples,
                sensor=sensor, reference=args.reference_sensor,
                n_quantiles=args.n_quantiles,
            )
        except ValueError as exc:
            log.warning("skipping %s: %s", sensor, exc)
            continue

        registry.add(qmap)
        summary = qmap.summary()
        log.info(
            "fitted %s -> %s from %d frame(s) / %d sample(s):",
            sensor, args.reference_sensor, len(frames), samples.size,
        )
        log.info(
            "    median %.2f K -> %.2f K  (shift %+.2f K)",
            summary["median_source_k"], summary["median_reference_k"],
            summary["median_shift_k"],
        )
        log.info(
            "    IQR %.2f K -> %.2f K  (ratio %.3f)",
            summary["iqr_source_k"], summary["iqr_reference_k"], summary["iqr_ratio"],
        )
        if abs(summary["median_shift_k"]) > 20.0:
            log.warning(
                "    median shift for %s exceeds 20 K -- verify the reader's "
                "calibration before trusting this map", sensor,
            )

    if not registry.maps:
        log.error("no maps fitted; nothing written")
        return 1

    registry.save(args.out_dir)
    log.info("wrote %d map(s) to %s", len(registry.maps), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
