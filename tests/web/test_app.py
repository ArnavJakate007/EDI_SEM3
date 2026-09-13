from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sattsr.geo.grid import TargetGrid
from sattsr.infer.pipeline import RunManifest, write_manifest
from sattsr.io.base import Frame
from sattsr.io.writer import write_frames_nc
from sattsr.viz.frames import render_run_animations, render_run_frames
from web.app import create_app

STATIC = Path(__file__).resolve().parents[2] / "web" / "static"

REQUIRED_IDS = [
    "run-select", "summary", "player-original", "player-interpolated", "timeline",
    "play-toggle", "speed", "synthetic-badge", "frame-label", "chart-overall",
    "chart-category", "metrics-table", "download-nc",
]


def _build_run(root: Path, run_id: str, *, n: int = 5, with_report: bool = True) -> Path:
    grid = TargetGrid(lat_min=10.0, lat_max=14.0, lon_min=70.0, lon_max=74.0,
                      resolution_deg=0.5)
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    frames, flags = [], []
    for i in range(n):
        frames.append(
            Frame(t0 + timedelta(minutes=15 * i),
                  np.full(grid.shape, 245.0 + i, dtype=np.float32), "insat3dr", Path("x.h5"))
        )
        flags.append(i % 2 == 1)

    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_frames_nc(run_dir / "output.nc", frames, grid, synthetic=flags, model_version="t")
    render_run_frames(run_dir)
    render_run_animations(run_dir, fps=4)

    write_manifest(run_dir, RunManifest(
        run_id=run_id, sensor="insat3dr", created_at=t0,
        input_cadence_minutes=30.0, output_cadence_minutes=15.0, factor=2,
        n_original=sum(1 for f in flags if not f), n_synthetic=sum(flags),
        output_nc="output.nc",
        frames=[{"index": i, "timestamp": f.timestamp.isoformat(), "synthetic": bool(s)}
                for i, (f, s) in enumerate(zip(frames, flags))],
        checkpoint="best.pt", model_version="0.1.0",
    ))
    if with_report:
        (run_dir / "report.json").write_text(
            json.dumps({"run_id": run_id,
                        "summary": {"overall": {"model": {"psnr": 31.0}}},
                        "samples": []}),
            encoding="utf-8",
        )
    return run_dir


@pytest.fixture
def client(tmp_path) -> TestClient:
    runs = tmp_path / "runs"
    _build_run(runs, "insat-demo")
    _build_run(runs, "goes-demo", with_report=False)
    return TestClient(create_app(runs))


# --------------------------------------------------------------------------- api


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_runs_listing_summarises_every_run(client):
    payload = client.get("/api/runs").json()
    assert {r["run_id"] for r in payload} == {"insat-demo", "goes-demo"}
    demo = next(r for r in payload if r["run_id"] == "insat-demo")
    assert demo["n_original"] == 3
    assert demo["n_synthetic"] == 2
    assert demo["has_report"] is True
    assert next(r for r in payload if r["run_id"] == "goes-demo")["has_report"] is False


def test_run_detail_returns_the_manifest(client):
    payload = client.get("/api/runs/insat-demo").json()
    assert payload["factor"] == 2
    assert len(payload["frames"]) == 5
    assert payload["frames"][1]["synthetic"] is True


def test_missing_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_frame_png_is_served(client):
    response = client.get("/api/runs/insat-demo/frames/2.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_out_of_range_frame_is_404(client):
    assert client.get("/api/runs/insat-demo/frames/99.png").status_code == 404


def test_both_animations_are_served(client):
    for kind in ("original", "interpolated"):
        response = client.get(f"/api/runs/insat-demo/animations/{kind}.gif")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/gif"


def test_unknown_animation_kind_is_rejected(client):
    assert client.get("/api/runs/insat-demo/animations/sideways.gif").status_code == 404


def test_report_is_served_when_present_and_404_when_not(client):
    assert client.get("/api/runs/insat-demo/report").json()["run_id"] == "insat-demo"
    assert client.get("/api/runs/goes-demo/report").status_code == 404


def test_netcdf_download(client):
    response = client.get("/api/runs/insat-demo/download")
    assert response.status_code == 200
    assert response.content[:3] in (b"CDF", b"\x89HD")


def test_path_traversal_in_run_id_is_rejected(client):
    for hostile in ("..", "../secrets", "..%2F..%2Fetc", "a/b"):
        assert client.get(f"/api/runs/{hostile}").status_code in (400, 404)


def test_missing_runs_directory_yields_an_empty_listing(tmp_path):
    client = TestClient(create_app(tmp_path / "does-not-exist"))
    assert client.get("/api/runs").json() == []


# --------------------------------------------------------------------------- static


def test_every_static_asset_exists():
    for name in ("index.html", "styles.css", "app.js"):
        assert (STATIC / name).is_file(), name


@pytest.mark.parametrize("element_id", REQUIRED_IDS)
def test_html_declares_every_required_element(element_id):
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert f'id="{element_id}"' in html


def test_html_loads_the_stylesheet_the_script_and_chartjs():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "styles.css" in html
    assert "app.js" in html
    assert "chart.js" in html.lower()


def test_script_talks_to_the_documented_endpoints():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for endpoint in ("/api/runs", "/frames/", "/report", "/download"):
        assert endpoint in js


def test_dashboard_is_served_at_the_root(tmp_path):
    client = TestClient(create_app(tmp_path / "runs"))
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Temporal Super Resolution" in response.text


def test_api_routes_still_win_over_the_static_mount(tmp_path):
    client = TestClient(create_app(tmp_path / "runs"))
    assert client.get("/api/health").json()["status"] == "ok"


# --------------------------------------- event-stratified endpoints


def _run_with_verdict(tmp_path):
    """A run directory whose report carries the head-to-head breakdown."""
    import json

    from sattsr.eval.plots import write_category_charts

    run = tmp_path / "r1"
    run.mkdir(parents=True)
    summary = {
        "counts": {"stratiform": 5, "convective": 4},
        "overall": {"model": {"psnr": 38.0}, "farneback": {"psnr": 37.0},
                    "linear": {"psnr": 33.0}},
        "by_category": {
            "stratiform": {"model": {"psnr": 38.0}, "farneback": {"psnr": 39.0},
                           "linear": {"psnr": 33.0}},
            "convective": {"model": {"psnr": 36.0}, "farneback": {"psnr": 34.0},
                           "linear": {"psnr": 30.0}},
        },
    }
    from sattsr.eval.report import head_to_head

    report = {"run_id": "r1", "summary": summary, "head_to_head": head_to_head(summary)}
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    write_category_charts(report, run)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "r1", "sensor": "goes19",
        "created_at": "2026-01-01T00:00:00+00:00",
        "input_cadence_minutes": 10.0, "output_cadence_minutes": 5.0,
        "factor": 2, "n_original": 3, "n_synthetic": 2,
        "output_nc": "output.nc", "frames": [], "checkpoint": "best.pt",
        "model_version": "0.1.0",
    }), encoding="utf-8")
    return run


def test_head_to_head_endpoint_serves_the_verdict(tmp_path):
    from fastapi.testclient import TestClient

    from web.app import create_app

    _run_with_verdict(tmp_path)
    client = TestClient(create_app(tmp_path))
    body = client.get("/api/runs/r1/head-to-head").json()
    assert body["categories_won"] == ["convective"]
    assert body["by_category"]["stratiform"]["beats_farneback"] is False


def test_charts_are_served_as_svg(tmp_path):
    from fastapi.testclient import TestClient

    from web.app import create_app

    _run_with_verdict(tmp_path)
    client = TestClient(create_app(tmp_path))
    for name in ("category_margin", "category_metric"):
        r = client.get(f"/api/runs/r1/charts/{name}.svg")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/svg+xml")


def test_unknown_chart_name_is_rejected(tmp_path):
    from fastapi.testclient import TestClient

    from web.app import create_app

    _run_with_verdict(tmp_path)
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/runs/r1/charts/evil.svg").status_code == 404


def test_runs_listing_flags_chart_availability(tmp_path):
    from fastapi.testclient import TestClient

    from web.app import create_app

    _run_with_verdict(tmp_path)
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/runs").json()[0]["has_charts"] is True


def test_a_report_without_the_verdict_says_to_rerun(tmp_path):
    import json

    from fastapi.testclient import TestClient

    from web.app import create_app

    run = _run_with_verdict(tmp_path)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    report.pop("head_to_head")
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")

    client = TestClient(create_app(tmp_path))
    r = client.get("/api/runs/r1/head-to-head")
    assert r.status_code == 404
    assert "re-run evaluate" in r.json()["detail"]
