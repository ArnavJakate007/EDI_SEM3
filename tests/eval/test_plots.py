"""Event-stratified charts and the model-vs-Farneback verdict."""

from __future__ import annotations

import json

import pytest

from sattsr.eval.plots import (
    LOWER_IS_BETTER,
    category_rows,
    grouped_chart,
    margin,
    margin_chart,
    write_category_charts,
)
from sattsr.eval.report import head_to_head


def _summary(**overrides):
    base = {
        "counts": {"clear": 4, "stratiform": 10, "convective": 6, "cyclonic": 0},
        "overall": {
            "model": {"psnr": 38.0, "mse": 3.0},
            "farneback": {"psnr": 37.0, "mse": 4.0},
            "linear": {"psnr": 33.0, "mse": 9.0},
        },
        "by_category": {
            "clear": {
                "model": {"psnr": 40.0, "mse": 2.0},
                "farneback": {"psnr": 41.0, "mse": 1.5},
                "linear": {"psnr": 36.0, "mse": 5.0},
            },
            "stratiform": {
                "model": {"psnr": 38.0, "mse": 3.0},
                "farneback": {"psnr": 39.0, "mse": 2.5},
                "linear": {"psnr": 33.0, "mse": 9.0},
            },
            "convective": {
                "model": {"psnr": 36.0, "mse": 5.0},
                "farneback": {"psnr": 34.0, "mse": 7.0},
                "linear": {"psnr": 30.0, "mse": 14.0},
            },
            "cyclonic": {},
        },
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------- rows


def test_category_rows_skips_categories_with_no_samples():
    rows = category_rows(_summary(), "psnr")
    assert [r["category"] for r in rows] == ["clear", "stratiform", "convective"]
    assert all(r["n"] > 0 for r in rows)


def test_category_rows_carries_every_method():
    row = category_rows(_summary(), "psnr")[0]
    assert set(row["values"]) == {"model", "farneback", "linear"}


# ------------------------------------------------------------------- margin


def test_margin_is_positive_when_the_model_wins_on_psnr():
    assert margin({"model": 36.0, "farneback": 34.0}, "psnr") == pytest.approx(2.0)


def test_margin_sign_flips_for_lower_is_better_metrics():
    """MSE down is good, so a smaller model MSE must read as a POSITIVE margin."""
    assert "mse" in LOWER_IS_BETTER
    assert margin({"model": 5.0, "farneback": 7.0}, "mse") == pytest.approx(2.0)
    assert margin({"model": 7.0, "farneback": 5.0}, "mse") == pytest.approx(-2.0)


def test_margin_is_none_without_a_farneback_reference():
    assert margin({"model": 36.0, "farneback": None}, "psnr") is None


# ------------------------------------------------------------------ verdict


def test_head_to_head_decides_each_category():
    h = head_to_head(_summary())
    assert h["by_category"]["convective"]["beats_farneback"] is True
    assert h["by_category"]["stratiform"]["beats_farneback"] is False
    assert h["categories_won"] == ["convective"]


def test_head_to_head_reports_the_pooled_verdict_separately():
    """Pooled and per-category can disagree -- that is the whole point."""
    h = head_to_head(_summary())
    assert h["pooled_beats_farneback"] is True
    assert h["pooled_margin_vs_farneback"] == pytest.approx(1.0)
    assert "convective" in h["categories_won"]


def test_a_model_can_lose_pooled_yet_win_a_category():
    s = _summary(overall={
        "model": {"psnr": 36.0}, "farneback": {"psnr": 38.0}, "linear": {"psnr": 33.0},
    })
    h = head_to_head(s)
    assert h["pooled_beats_farneback"] is False
    assert h["categories_won"] == ["convective"], (
        "losing on average must not hide a category win"
    )


def test_head_to_head_omits_empty_categories():
    assert "cyclonic" not in head_to_head(_summary())["by_category"]


# ------------------------------------------------------------------- charts


def test_margin_chart_is_valid_svg_with_one_row_per_category():
    svg = margin_chart(_summary())
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    for cat in ("clear", "stratiform", "convective"):
        assert f">{cat}<" in svg
    assert "cyclonic" not in svg


def test_margin_chart_colours_wins_and_losses_differently():
    from sattsr.eval.plots import LOSS_COLOUR, WIN_COLOUR

    svg = margin_chart(_summary())
    assert WIN_COLOUR in svg, "the convective win should be drawn as a win"
    assert LOSS_COLOUR in svg, "the stratiform loss should be drawn as a loss"


def test_margin_chart_labels_the_signed_margin():
    svg = margin_chart(_summary())
    assert "+2.00" in svg and "-1.00" in svg


def test_charts_degrade_gracefully_with_no_scored_categories():
    empty = {"counts": {}, "overall": {}, "by_category": {}}
    assert "no scored categories" in margin_chart(empty)
    assert "no scored categories" in grouped_chart(empty)


def test_grouped_chart_shows_every_method():
    svg = grouped_chart(_summary())
    for method in ("model", "farneback", "linear"):
        assert method in svg


def test_write_category_charts_emits_both_files(tmp_path):
    paths = write_category_charts({"summary": _summary()}, tmp_path)
    assert set(paths) == {"category_margin", "category_metric"}
    for p in paths.values():
        assert p.exists() and p.read_text(encoding="utf-8").startswith("<svg")


def test_charts_escape_category_names(tmp_path):
    nasty = _summary()
    nasty["counts"]["<script>"] = 3
    nasty["by_category"]["<script>"] = {"model": {"psnr": 1.0}, "farneback": {"psnr": 1.0}}
    svg = margin_chart(nasty)
    assert "<script>" not in svg and "&lt;script&gt;" in svg
