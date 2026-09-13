#!/usr/bin/env python
"""Score the non-learned baselines, and optionally a checkpoint, on the same split.

The number that matters for "is this model doing anything yet" is not its absolute
PSNR -- thermal-IR imagery is smooth, so even a naive average of the two neighbours
scores well. It is the MARGIN over that average. This prints both, computed with the
identical masked-PSNR function the training loop uses, so the two are comparable.

    python scripts/eval_baseline.py --split val
    python scripts/eval_baseline.py --split val --checkpoint checkpoints/pretrain/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sattsr.config import load_pretrain_config  # noqa: E402
from sattsr.data.loaders import build_dataloaders  # noqa: E402
from sattsr.models.interpolator import build_model  # noqa: E402
from sattsr.train.checkpoint import load_checkpoint  # noqa: E402
from sattsr.train.loop import _batch_psnr  # noqa: E402

log = logging.getLogger("eval_baseline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml")
    parser.add_argument(
        "--split", default="val",
        help="Which loader to score: train, val, finetune, insat_rapid_scan_eval.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Also score this checkpoint's model on the same batches.",
    )
    parser.add_argument("--device", default="auto", help="auto (default), cpu, or cuda.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_pretrain_config(args.config)
    loaders = build_dataloaders(config, None, repo_root=REPO_ROOT)
    if args.split not in loaders:
        log.error("no %r loader; available: %s", args.split, sorted(loaders))
        return 2
    loader = loaders[args.split]

    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    model = None
    if args.checkpoint is not None:
        model = build_model(config.model).to(device).eval()
        payload = load_checkpoint(args.checkpoint, model, map_location=str(device))
        log.info("scoring checkpoint %s (epoch %s)", args.checkpoint, payload.get("epoch"))

    totals: dict[str, float] = defaultdict(float)
    per_sensor: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sensor_n: Counter[str] = Counter()
    seen = 0

    with torch.no_grad():
        for batch in loader:
            i0 = batch["i0"].to(device)
            i1 = batch["i1"].to(device)
            i2 = batch["i2"].to(device)
            t = batch["t"].to(device).view(-1, 1, 1, 1)
            mask = batch.get("valid")
            mask = mask.to(device) if mask is not None else None
            n = int(i0.shape[0])

            # Two non-learned references. `t` is not always 0.5, so the time-weighted
            # blend is the fairer of the two.
            mid = 0.5 * (i0 + i2)
            blend = (1.0 - t) * i0 + t * i2
            scores = {
                "naive_mid": _batch_psnr(mid, i1, mask),
                "linear_blend_t": _batch_psnr(blend, i1, mask),
                "copy_frame0": _batch_psnr(i0, i1, mask),
            }
            if model is not None:
                pred = model(i0, i2, batch["t"].to(device))["pred"]
                scores["model"] = _batch_psnr(pred, i1, mask)

            for key, value in scores.items():
                totals[key] += value * n
            for sensor in batch.get("sensor", []):
                sensor_n[sensor] += 1
                for key, value in scores.items():
                    per_sensor[sensor][key] += value
            seen += n

    if seen == 0:
        log.error("split %r produced no samples", args.split)
        return 1

    print(f"\n{args.split}: {seen} sample(s)")
    print(f"{'method':<18} {'PSNR (dB)':>10}")
    print("-" * 30)
    order = ["copy_frame0", "naive_mid", "linear_blend_t", "model"]
    for key in order:
        if key in totals:
            print(f"{key:<18} {totals[key] / seen:>10.2f}")

    if "model" in totals:
        margin = (totals["model"] - totals["linear_blend_t"]) / seen
        print(f"\nmodel margin over the time-weighted blend: {margin:+.2f} dB")
        if margin <= 0.0:
            print("  -> the model is NOT yet beating a naive average. More data or "
                  "more epochs needed before any quality claim.")

    if len(sensor_n) > 1:
        print("\nper sensor:")
        for sensor, n in sorted(sensor_n.items()):
            row = "  ".join(
                f"{k}={per_sensor[sensor][k] / n:.2f}" for k in order if k in totals
            )
            print(f"  {sensor:<12} n={n:<4} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
