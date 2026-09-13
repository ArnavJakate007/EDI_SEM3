"""SVG charts for the evaluation report.

SVG rather than matplotlib: the only consumer is the dashboard, which is a browser,
so SVG renders natively and stays sharp at any zoom. It also keeps matplotlib out of
the dependency list for what amounts to two bar charts.

The charts are built around one question -- does the model beat classical optical flow
on the events that matter? A grouped bar chart of three similar numbers makes that
comparison something you have to work out. So the primary chart is the MARGIN of model
minus Farneback per event category, diverging around zero: bars right of the axis mean
the model wins that category, bars left mean it loses, and the length is how much.
The absolute values are kept in a second chart for context.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

#: Consistent colours across both charts.
METHOD_COLOURS = {
    "model": "#2f6fed",
    "farneback": "#e07b39",
    "linear": "#8a8f98",
}
WIN_COLOUR = "#1a9850"
LOSS_COLOUR = "#d73027"
AXIS = "#c8ccd2"
TEXT = "#2b2f36"
MUTED = "#6b7280"

#: Metrics where a LOWER value is better, so the margin sign has to flip.
LOWER_IS_BETTER = {"mse", "rmse", "far", "displacement_error"}


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def category_rows(
    summary: dict[str, Any], metric: str, *, methods: Sequence[str] = ("model", "farneback", "linear")
) -> list[dict[str, Any]]:
    """Per-category values for `metric`, skipping categories with no samples."""
    counts = summary.get("counts", {}) or {}
    rows: list[dict[str, Any]] = []
    for category, per_method in (summary.get("by_category") or {}).items():
        n = int(_finite(counts.get(category, 0)) or 0)
        if not per_method or n == 0:
            continue
        values = {
            m: _finite((per_method.get(m) or {}).get(metric))
            for m in methods
        }
        if values.get("model") is None:
            continue
        rows.append({"category": category, "n": n, "values": values})
    return rows


def margin(values: dict[str, float | None], metric: str) -> float | None:
    """Model minus Farneback, signed so positive always means the model is better."""
    model, ref = values.get("model"), values.get("farneback")
    if model is None or ref is None:
        return None
    delta = model - ref
    return -delta if metric in LOWER_IS_BETTER else delta


def _svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui,-apple-system,'
        f'Segoe UI,Roboto,sans-serif" font-size="12">{body}</svg>'
    )


def margin_chart(summary: dict[str, Any], *, metric: str = "psnr", width: int = 680) -> str:
    """Diverging bars of model-minus-Farneback per event category.

    This is the chart that answers the project's central question at a glance.
    """
    rows = category_rows(summary, metric)
    if not rows:
        return _svg(width, 60,
                    f'<text x="12" y="34" fill="{MUTED}">no scored categories yet</text>')

    row_h, top, left, right = 38, 52, 150, 28
    height = top + row_h * len(rows) + 34
    plot_w = width - left - right
    mid = left + plot_w / 2

    span = max(
        (abs(margin(r["values"], metric) or 0.0) for r in rows), default=1.0
    ) or 1.0
    span *= 1.25

    parts = [
        f'<text x="12" y="20" font-weight="600" fill="{TEXT}">'
        f'Model vs Farneback by event type &#8212; {escape(metric.upper())}</text>',
        f'<text x="12" y="38" fill="{MUTED}">bar to the right = model wins that '
        f'category; length = margin</text>',
        f'<line x1="{mid}" y1="{top - 8}" x2="{mid}" y2="{top + row_h * len(rows)}" '
        f'stroke="{AXIS}" stroke-width="1"/>',
    ]

    for i, row in enumerate(rows):
        y = top + i * row_h
        cy = y + row_h / 2
        m = margin(row["values"], metric)
        parts.append(
            f'<text x="12" y="{cy + 4}" fill="{TEXT}">{escape(row["category"])}'
            f'<tspan fill="{MUTED}"> (n={row["n"]})</tspan></text>'
        )
        if m is None:
            parts.append(f'<text x="{mid + 8}" y="{cy + 4}" fill="{MUTED}">no Farneback</text>')
            continue
        w = abs(m) / span * (plot_w / 2)
        x = mid if m >= 0 else mid - w
        colour = WIN_COLOUR if m >= 0 else LOSS_COLOUR
        parts.append(
            f'<rect x="{x:.1f}" y="{y + 9}" width="{max(w, 1):.1f}" height="{row_h - 18}" '
            f'rx="2" fill="{colour}" opacity="0.85"/>'
        )
        label_x = mid + w + 8 if m >= 0 else mid - w - 8
        anchor = "start" if m >= 0 else "end"
        parts.append(
            f'<text x="{label_x:.1f}" y="{cy + 4}" text-anchor="{anchor}" '
            f'fill="{colour}" font-weight="600">{m:+.2f}</text>'
        )

    parts.append(
        f'<text x="{mid}" y="{top + row_h * len(rows) + 22}" text-anchor="middle" '
        f'fill="{MUTED}">0 = parity with Farneback</text>'
    )
    return _svg(width, height, "".join(parts))


def grouped_chart(
    summary: dict[str, Any], *, metric: str = "psnr", width: int = 680,
    methods: Sequence[str] = ("model", "farneback", "linear"),
) -> str:
    """Absolute per-category values for each method, for context under the margins."""
    rows = category_rows(summary, metric, methods=methods)
    if not rows:
        return _svg(width, 60,
                    f'<text x="12" y="34" fill="{MUTED}">no scored categories yet</text>')

    bar_h, gap, top, left, right = 14, 6, 56, 150, 70
    group_h = len(methods) * (bar_h + gap) + 12
    height = top + group_h * len(rows) + 16
    plot_w = width - left - right

    every = [v for r in rows for v in r["values"].values() if v is not None]
    lo, hi = (min(every), max(every)) if every else (0.0, 1.0)
    if metric in LOWER_IS_BETTER:
        lo = 0.0
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad
    rng = hi - lo or 1.0

    parts = [
        f'<text x="12" y="20" font-weight="600" fill="{TEXT}">'
        f'{escape(metric.upper())} by event type</text>',
        f'<text x="12" y="38" fill="{MUTED}">'
        + "  ".join(
            f'<tspan fill="{METHOD_COLOURS.get(m, MUTED)}">&#9632;</tspan> {escape(m)}'
            for m in methods
        )
        + "</text>",
    ]

    for i, row in enumerate(rows):
        gy = top + i * group_h
        parts.append(
            f'<text x="12" y="{gy + 12}" fill="{TEXT}">{escape(row["category"])}'
            f'<tspan fill="{MUTED}"> (n={row["n"]})</tspan></text>'
        )
        for j, method in enumerate(methods):
            v = row["values"].get(method)
            y = gy + j * (bar_h + gap)
            if v is None:
                parts.append(f'<text x="{left}" y="{y + bar_h - 2}" fill="{MUTED}">n/a</text>')
                continue
            w = max((v - lo) / rng * plot_w, 1.0)
            parts.append(
                f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="2" '
                f'fill="{METHOD_COLOURS.get(method, MUTED)}" opacity="0.85"/>'
                f'<text x="{left + w + 6:.1f}" y="{y + bar_h - 2}" fill="{TEXT}">{v:.2f}</text>'
            )
    return _svg(width, height, "".join(parts))


def write_category_charts(
    report: dict[str, Any], out_dir: str | Path, *, metric: str = "psnr"
) -> dict[str, Path]:
    """Write both charts beside the report. Returns the paths written."""
    summary = report.get("summary", {}) or {}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name, svg in (
        ("category_margin", margin_chart(summary, metric=metric)),
        ("category_metric", grouped_chart(summary, metric=metric)),
    ):
        path = out / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        written[name] = path
    return written
