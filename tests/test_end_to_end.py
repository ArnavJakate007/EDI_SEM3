from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from sattsr.config import Config
from sattsr.data.index import build_index, prepare_cache
from sattsr.data.triplets import build_triplets
from sattsr.eval.report import evaluate_triplets, write_report
from sattsr.geo.grid import TargetGrid
from sattsr.infer.pipeline import run_inference
from sattsr.io.registry import get_reader
from sattsr.io.writer import read_frames_nc
from sattsr.models.interpolator import build_model
from sattsr.train.checkpoint import save_checkpoint
from sattsr.viz.frames import render_run_animations, render_run_frames
from tests.conftest import make_goes_series
from web.app import create_app


@pytest.mark.slow
def test_raw_files_to_dashboard(tmp_path):
    """Every stage of the pipeline, wired together, on synthetic GOES-like input."""
    config = Config.model_validate({
        "data": {
            "sensor": "goes19",
            "raw_root": str(tmp_path / "raw"),
            "cache_root": str(tmp_path / "cache"),
            "cadence_minutes": 30,
            "tolerance_minutes": 5,
            "grid": {"lat_min": 5.0, "lat_max": 13.0, "lon_min": 78.0,
                     "lon_max": 86.0, "resolution_deg": 0.25},
        },
        "model": {"base_channels": 8, "scales": [2, 1], "use_raft_init": False},
        "train": {"tile_size": 32, "tile_overlap": 8, "batch_size": 2, "epochs": 1,
                  "amp": False, "num_workers": 0},
    })
    make_goes_series(tmp_path / "raw", 6, step_minutes=30, size=48)

    # 1. ingest and cache
    reader = get_reader(config.data.sensor)
    grid = TargetGrid.from_config(config.data.grid)
    refs = prepare_cache(reader, build_index(reader, config.data.raw_root), grid,
                         config.data.cache_root)
    assert len(refs) == 6

    # 2. triplets
    triplets = build_triplets(refs, step=timedelta(minutes=30), tolerance=timedelta(minutes=5))
    assert len(triplets) == 4

    # 3. a (randomly initialised) model standing in for a trained one
    checkpoint = save_checkpoint(tmp_path / "m.pt", model=build_model(config.model),
                                 epoch=0, best_metric=0.0, config=config)

    # 4. inference -> flagged NetCDF
    run_dir = tmp_path / "runs" / "e2e"
    manifest = run_inference(config, input_dir=config.data.raw_root, output_dir=run_dir,
                             checkpoint=checkpoint, factor=4, device=torch.device("cpu"),
                             limit=3)
    assert manifest.output_cadence_minutes == pytest.approx(7.5)

    frames, flags = read_frames_nc(run_dir / "output.nc")
    assert len(frames) == 9                     # 3 observed + 6 synthesized
    assert sum(flags) == 6
    assert all(np.isfinite(f.bt).any() for f in frames)

    # 5. report
    model = build_model(config.model).eval()
    results = evaluate_triplets(model, triplets, device=torch.device("cpu"),
                                norm=config.data.normalization, tile_size=32, tile_overlap=8,
                                limit=2)
    report = write_report(run_dir / "report.json", results, run_id="e2e",
                          config_summary={"sensor": config.data.sensor})
    assert set(report["summary"]["methods"]) == {"model", "linear", "farneback"}

    # 6. visuals
    render_run_frames(run_dir)
    render_run_animations(run_dir, fps=4)

    # 7. dashboard serves all of it
    client = TestClient(create_app(tmp_path / "runs"))
    assert client.get("/api/runs").json()[0]["run_id"] == "e2e"
    assert client.get("/api/runs/e2e/frames/1.png").status_code == 200
    assert client.get("/api/runs/e2e/animations/original.gif").status_code == 200
    assert client.get("/api/runs/e2e/animations/interpolated.gif").status_code == 200
    assert client.get("/api/runs/e2e/report").status_code == 200
    assert client.get("/api/runs/e2e/download").status_code == 200
