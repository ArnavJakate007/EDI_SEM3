from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from sattsr.config import ModelConfig, NormalizationConfig
from sattsr.data.index import FrameRef
from sattsr.data.triplets import Triplet
from sattsr.eval.events import EventCategory
from sattsr.eval.report import (
    METRIC_KEYS,
    SampleResult,
    aggregate,
    evaluate_triplets,
    load_report,
    write_report,
)
from sattsr.models.interpolator import build_model

NORM = NormalizationConfig(bt_min=180.0, bt_max=330.0)
CFG = ModelConfig(base_channels=8, scales=[2, 1], use_raft_init=False)


def _result(method: str, category: EventCategory, psnr: float, ssim: float) -> SampleResult:
    return SampleResult(
        timestamp=datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc),
        category=category,
        method=method,
        metrics={"psnr": psnr, "ssim": ssim, "csi": float("nan")},
    )


def _triplets(tmp_path: Path, n: int = 2, size: int = 64) -> list[Triplet]:
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    refs = []
    for i in range(n + 2):
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        bt = (255.0 + 20.0 * np.sin((xx + 3 * i) * 0.3) * np.cos(yy * 0.25)).astype(np.float32)
        p = tmp_path / f"f{i}.npy"
        np.save(p, bt)
        refs.append(FrameRef(t0 + timedelta(minutes=30 * i), p, "insat3dr"))
    return [Triplet(refs[i], refs[i + 1], refs[i + 2]) for i in range(n)]


def test_aggregate_averages_per_method_and_category():
    results = [
        _result("model", EventCategory.CLEAR, 30.0, 0.9),
        _result("model", EventCategory.CONVECTIVE, 20.0, 0.7),
        _result("linear", EventCategory.CLEAR, 25.0, 0.8),
        _result("linear", EventCategory.CONVECTIVE, 15.0, 0.5),
    ]
    summary = aggregate(results)
    assert summary["n_samples"] == 1                    # unique timestamps, not rows
    assert sorted(summary["methods"]) == ["linear", "model"]
    assert summary["overall"]["model"]["psnr"] == pytest.approx(25.0)
    assert summary["by_category"]["convective"]["model"]["psnr"] == pytest.approx(20.0)


def test_aggregate_ignores_nan_metrics():
    results = [
        _result("model", EventCategory.CLEAR, 30.0, 0.9),
        _result("model", EventCategory.CLEAR, float("nan"), 0.7),
    ]
    summary = aggregate(results)
    assert summary["overall"]["model"]["psnr"] == pytest.approx(30.0)
    assert summary["overall"]["model"]["ssim"] == pytest.approx(0.8)


def test_aggregate_of_nothing_is_empty_not_a_crash():
    summary = aggregate([])
    assert summary["n_samples"] == 0
    assert summary["overall"] == {}


def test_write_report_round_trips_and_carries_provenance(tmp_path):
    results = [_result("model", EventCategory.CLEAR, 30.0, 0.9)]
    report = write_report(tmp_path / "report.json", results, run_id="run-1",
                          config_summary={"sensor": "insat3dr"}, notes="fine-tuned")
    assert (tmp_path / "report.json").exists()
    assert report["run_id"] == "run-1"

    back = load_report(tmp_path / "report.json")
    assert back["run_id"] == "run-1"
    assert back["config"]["sensor"] == "insat3dr"
    assert back["notes"] == "fine-tuned"
    assert back["summary"]["overall"]["model"]["psnr"] == pytest.approx(30.0)
    assert len(back["samples"]) == 1
    assert "heuristic" in back["category_disclaimer"].lower()


def test_report_json_is_plain_and_has_no_nan_literals(tmp_path):
    results = [_result("model", EventCategory.CLEAR, float("nan"), 0.9)]
    write_report(tmp_path / "report.json", results, run_id="r", config_summary={})
    text = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert "NaN" not in text                     # JSON.parse in the browser would choke
    assert json.loads(text)["samples"][0]["metrics"]["psnr"] is None


def test_evaluate_triplets_scores_every_method(tmp_path):
    model = build_model(CFG).eval()
    results = evaluate_triplets(
        model, _triplets(tmp_path), device=torch.device("cpu"), norm=NORM,
        tile_size=32, tile_overlap=8,
    )
    methods = {r.method for r in results}
    assert methods == {"model", "linear", "farneback"}
    assert len(results) == 2 * 3
    for r in results:
        assert set(METRIC_KEYS).issubset(r.metrics)
        assert isinstance(r.category, EventCategory)


def test_evaluate_triplets_honours_limit_and_method_selection(tmp_path):
    model = build_model(CFG).eval()
    results = evaluate_triplets(
        model, _triplets(tmp_path, n=4), device=torch.device("cpu"), norm=NORM,
        tile_size=32, tile_overlap=8, methods=("model", "linear"), limit=2,
    )
    assert len(results) == 2 * 2
    assert {r.method for r in results} == {"model", "linear"}


def test_evaluate_triplets_rejects_an_unknown_method(tmp_path):
    model = build_model(CFG).eval()
    with pytest.raises(KeyError):
        evaluate_triplets(model, _triplets(tmp_path), device=torch.device("cpu"), norm=NORM,
                          tile_size=32, tile_overlap=8, methods=("model", "magic"))
