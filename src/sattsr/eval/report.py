"""Scoring the model against the baselines, and writing the comparison report."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sattsr.config import NormalizationConfig
from sattsr.data.index import load_cached
from sattsr.data.triplets import Triplet
from sattsr.eval.baselines import run_baseline
from sattsr.eval.events import EventCategory, EventThresholds, category_counts, classify_event
from sattsr.eval.metrics import metric_suite
from sattsr.eval.motion import motion_suite
from sattsr.eval.plots import category_rows, margin, write_category_charts
from sattsr.infer.recursive import predict_midframe
from sattsr.models.interpolator import FrameInterpolator

METRIC_KEYS: tuple[str, ...] = (
    "mse", "rmse", "psnr", "ssim", "fsim", "displacement_error", "csi", "pod", "far",
)

CATEGORY_DISCLAIMER = (
    "Event categories are assigned by a documented heuristic on brightness temperature "
    "and inter-frame flow vorticity, not by a validated meteorological classifier. They "
    "exist so convective and cyclonic performance is reported separately rather than "
    "averaged away."
)

DEFAULT_METHODS: tuple[str, ...] = ("model", "linear", "farneback")
_BASELINE_NAMES = frozenset({"linear", "farneback"})


@dataclass(frozen=True)
class SampleResult:
    """One method's score on one triplet."""

    timestamp: datetime
    category: EventCategory
    method: str
    metrics: dict[str, float]


def _clean(value: float | None) -> float | None:
    """JSON has no NaN or Infinity literals; browsers reject them."""
    if value is None:
        return None
    return float(value) if math.isfinite(float(value)) else None


def _subsample(triplets: Sequence[Triplet], limit: int | None) -> list[Triplet]:
    """Take `limit` triplets spread EVENLY across the series, not the first N.

    A contiguous prefix is one weather situation over a few hours, so it collapses the
    event breakdown: a 12-triplet prefix of this archive scored 12 stratiform and zero
    convective, clear or cyclonic, which makes the per-category comparison vacuous.
    Even spacing samples the whole time range, so the categories populate in roughly
    the proportion the archive actually contains.
    """
    ordered = sorted(triplets, key=lambda t: t.t1.timestamp)
    if limit is None or limit >= len(ordered):
        return ordered
    if limit <= 0:
        return []
    step = len(ordered) / limit
    return [ordered[min(len(ordered) - 1, int(i * step))] for i in range(limit)]


def evaluate_triplets(
    model: FrameInterpolator,
    triplets: Sequence[Triplet],
    *,
    device: torch.device,
    norm: NormalizationConfig,
    tile_size: int = 256,
    tile_overlap: int = 32,
    methods: Sequence[str] = DEFAULT_METHODS,
    limit: int | None = None,
    thresholds: EventThresholds = EventThresholds(),
    progress: bool = False,
) -> list[SampleResult]:
    """Score every requested method against the real middle frame of each triplet."""
    unknown = [m for m in methods if m != "model" and m not in _BASELINE_NAMES]
    if unknown:
        raise KeyError(f"unknown evaluation method(s): {unknown}")

    items: Sequence[Triplet] = _subsample(triplets, limit)
    iterator: Any = items
    if progress:
        from tqdm import tqdm

        iterator = tqdm(list(items), desc="evaluating")

    results: list[SampleResult] = []
    for triplet in iterator:
        i0 = load_cached(triplet.t0)
        truth = load_cached(triplet.t1)
        i2 = load_cached(triplet.t2)
        category = classify_event(
            np.stack([i0, truth, i2]), data_range=norm.data_range, thresholds=thresholds
        )

        for method in methods:
            if method == "model":
                pred = predict_midframe(
                    model, i0, i2, device=device, norm=norm, tile_size=tile_size,
                    tile_overlap=tile_overlap, t=float(triplet.t),
                )
            else:
                pred = run_baseline(method, i0, i2, float(triplet.t), data_range=norm.data_range)

            metrics = metric_suite(pred, truth, data_range=norm.data_range)
            metrics.update(motion_suite(i0, pred, truth, data_range=norm.data_range))
            results.append(SampleResult(triplet.t1.timestamp, category, method, metrics))
    return results


def _mean(values: list[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _summarise(results: Sequence[SampleResult]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for key, value in r.metrics.items():
            buckets[r.method][key].append(value)
    return {
        method: {key: _mean(values) for key, values in metrics.items()}
        for method, metrics in buckets.items()
    }


def aggregate(results: Sequence[SampleResult]) -> dict[str, Any]:
    """Mean of every metric, overall and per event category."""
    if not results:
        return {"n_samples": 0, "methods": [], "counts": category_counts([]),
                "overall": {}, "by_category": {}}

    per_timestamp: dict[datetime, EventCategory] = {r.timestamp: r.category for r in results}
    by_category: dict[str, dict[str, dict[str, float]]] = {}
    for category in EventCategory:
        subset = [r for r in results if r.category == category]
        if subset:
            by_category[category.value] = _summarise(subset)

    return {
        "n_samples": len(per_timestamp),
        "methods": sorted({r.method for r in results}),
        "counts": category_counts(per_timestamp.values()),
        "overall": _summarise(results),
        "by_category": by_category,
    }


def _sanitise(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _sanitise(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_sanitise(v) for v in node]
    if isinstance(node, bool):
        return node
    if isinstance(node, (int, float, np.floating, np.integer)):
        return _clean(float(node))
    return node


#: Metrics the model-vs-Farneback verdict is computed for. The answer is not the
#: same across them -- PSNR and MSE reward pixel-exactness, which a flow-warped copy
#: of a real frame is very good at, while FSIM weights phase/gradient structure, which
#: is what actually degrades when a warp tears a cloud field. Reporting only one would
#: pick the conclusion.
VERDICT_METRICS = ("psnr", "ssim", "mse", "fsim")


def head_to_head(
    summary: dict[str, Any], *, metric: str = "psnr"
) -> dict[str, Any]:
    """Model vs Farneback per event category, decided rather than left implicit.

    The project's premise is that classical optical flow fails specifically on fast,
    non-linear cloud motion even when it is competitive on average. A pooled number
    cannot show that either way, and three similar per-category numbers make the
    reader do the subtraction. This records the margin and the verdict directly, so
    "does it beat Farneback on the cases that matter" is answerable by lookup.
    """
    rows = category_rows(summary, metric)
    per_category = {}
    for row in rows:
        m = margin(row["values"], metric)
        per_category[row["category"]] = {
            "n": row["n"],
            "metric": metric,
            "model": row["values"].get("model"),
            "farneback": row["values"].get("farneback"),
            "linear": row["values"].get("linear"),
            "margin_vs_farneback": m,
            "beats_farneback": None if m is None else bool(m > 0),
        }

    overall = summary.get("overall", {}) or {}
    pooled = margin(
        {k: (overall.get(k) or {}).get(metric) for k in ("model", "farneback", "linear")},
        metric,
    )
    wins = [c for c, v in per_category.items() if v["beats_farneback"]]
    return {
        "metric": metric,
        "pooled_margin_vs_farneback": pooled,
        "pooled_beats_farneback": None if pooled is None else bool(pooled > 0),
        "categories_won": sorted(wins),
        "by_category": per_category,
    }


def write_report(
    path: str | Path,
    results: Sequence[SampleResult],
    *,
    run_id: str,
    config_summary: dict[str, Any],
    notes: str | None = None,
) -> dict[str, Any]:
    """Write the comparison report as browser-safe JSON and return it."""
    report: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config_summary,
        "notes": notes,
        "metric_keys": list(METRIC_KEYS),
        "category_disclaimer": CATEGORY_DISCLAIMER,
        "summary": _sanitise(aggregate(results)),
        "head_to_head": _sanitise(head_to_head(aggregate(results))),
        "head_to_head_by_metric": _sanitise(
            {m: head_to_head(aggregate(results), metric=m) for m in VERDICT_METRICS}
        ),
        "samples": [
            {
                "timestamp": r.timestamp.astimezone(timezone.utc).isoformat(),
                "category": r.category.value,
                "method": r.method,
                "metrics": {k: _clean(v) for k, v in r.metrics.items()},
            }
            for r in results
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_category_charts(report, out.parent)
    return report


def load_report(path: str | Path) -> dict[str, Any]:
    """Read a report written by `write_report`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
