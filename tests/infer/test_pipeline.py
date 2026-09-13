from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from sattsr.config import Config, ModelConfig
from sattsr.infer.pipeline import RunManifest, read_manifest, run_inference, write_manifest
from sattsr.io.writer import read_frames_nc
from sattsr.models.interpolator import build_model
from sattsr.train.checkpoint import save_checkpoint
from tests.conftest import make_goes_series

CFG = ModelConfig(base_channels=8, scales=[2, 1], use_raft_init=False)


def _config(tmp_path) -> Config:
    return Config.model_validate(
        {
            "data": {
                "sensor": "goes19",
                "raw_root": str(tmp_path / "raw"),
                "cache_root": str(tmp_path / "cache"),
                "cadence_minutes": 30,
                "tolerance_minutes": 5,
                "grid": {"lat_min": 5.0, "lat_max": 13.0, "lon_min": 78.0,
                         "lon_max": 86.0, "resolution_deg": 0.25},
            },
            "model": CFG.model_dump(),
            "train": {"tile_size": 32, "tile_overlap": 8},
        }
    )


def _checkpoint(tmp_path):
    return save_checkpoint(tmp_path / "best.pt", model=build_model(CFG), epoch=0, best_metric=0.0)


def test_manifest_round_trips(tmp_path):
    manifest = RunManifest(
        run_id="r1", sensor="goes19", created_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        input_cadence_minutes=30.0, output_cadence_minutes=15.0, factor=2,
        n_original=3, n_synthetic=2, output_nc="output.nc",
        frames=[{"index": 0, "timestamp": "2026-09-04T06:00:00+00:00", "synthetic": False}],
        checkpoint="best.pt", model_version="0.1.0",
    )
    write_manifest(tmp_path, manifest)
    assert (tmp_path / "manifest.json").exists()
    assert read_manifest(tmp_path) == manifest


def test_run_inference_produces_a_netcdf_and_a_manifest(tmp_path):
    raw = tmp_path / "raw"
    make_goes_series(raw, 3, step_minutes=30, size=48)
    out_dir = tmp_path / "runs" / "demo"

    manifest = run_inference(
        _config(tmp_path), input_dir=raw, output_dir=out_dir,
        checkpoint=_checkpoint(tmp_path), factor=2, device=torch.device("cpu"),
    )

    assert manifest.n_original == 3
    assert manifest.n_synthetic == 2
    assert manifest.output_cadence_minutes == pytest.approx(15.0)
    assert (out_dir / "output.nc").exists()
    assert (out_dir / "manifest.json").exists()

    frames, flags = read_frames_nc(out_dir / "output.nc")
    assert len(frames) == 5
    assert flags == [False, True, False, True, False]
    assert all(np.isfinite(f.bt).any() for f in frames)


def test_run_inference_at_factor_four(tmp_path):
    raw = tmp_path / "raw"
    make_goes_series(raw, 2, step_minutes=30, size=48)
    manifest = run_inference(
        _config(tmp_path), input_dir=raw, output_dir=tmp_path / "runs" / "f4",
        checkpoint=_checkpoint(tmp_path), factor=4, device=torch.device("cpu"),
    )
    assert manifest.n_synthetic == 3
    assert manifest.output_cadence_minutes == pytest.approx(7.5)


def test_manifest_frame_entries_match_the_netcdf(tmp_path):
    raw = tmp_path / "raw"
    make_goes_series(raw, 3, step_minutes=30, size=48)
    out_dir = tmp_path / "runs" / "demo"
    run_inference(
        _config(tmp_path), input_dir=raw, output_dir=out_dir,
        checkpoint=_checkpoint(tmp_path), factor=2, device=torch.device("cpu"),
    )
    payload = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [f["synthetic"] for f in payload["frames"]] == [False, True, False, True, False]
    assert [f["index"] for f in payload["frames"]] == [0, 1, 2, 3, 4]


def test_run_inference_needs_at_least_two_input_frames(tmp_path):
    raw = tmp_path / "raw"
    make_goes_series(raw, 1, step_minutes=30, size=48)
    with pytest.raises(ValueError):
        run_inference(_config(tmp_path), input_dir=raw, output_dir=tmp_path / "runs" / "x",
                      checkpoint=_checkpoint(tmp_path), factor=2, device=torch.device("cpu"))


def test_limit_truncates_the_input_series(tmp_path):
    raw = tmp_path / "raw"
    make_goes_series(raw, 5, step_minutes=30, size=48)
    manifest = run_inference(
        _config(tmp_path), input_dir=raw, output_dir=tmp_path / "runs" / "lim",
        checkpoint=_checkpoint(tmp_path), factor=2, device=torch.device("cpu"), limit=3,
    )
    assert manifest.n_original == 3


# ------------------------------------------------- cache-sourced inference


def _write_cache_frames(root, n=4, size=32):
    """A small regridded .npy cache with an index, as preprocess.py leaves behind."""
    from datetime import datetime, timedelta, timezone

    import numpy as np

    from sattsr.data.index import FrameRef, save_index

    t0 = datetime(2025, 6, 15, 0, 0, tzinfo=timezone.utc)
    refs = []
    for i in range(n):
        ts = t0 + timedelta(minutes=10 * i)
        dest = root / ts.strftime("%Y%m%d") / f"{ts.strftime('%H%M%S')}.npy"
        dest.parent.mkdir(parents=True, exist_ok=True)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        np.save(dest, (250.0 + 0.3 * ((xx + i) % size) + 0.2 * yy).astype(np.float32))
        refs.append(FrameRef(ts, dest, "goes19"))
    save_index(refs, root / "index.json")
    return refs


def test_frames_from_cache_reads_regridded_frames(tmp_path):
    """The cache is often the only copy left once raw has been reclaimed."""
    import numpy as np

    from sattsr.infer.pipeline import frames_from_cache

    refs = _write_cache_frames(tmp_path / "cache", n=5)
    frames = frames_from_cache(tmp_path / "cache", "goes19")

    assert len(frames) == 5
    assert [f.timestamp for f in frames] == [r.timestamp for r in refs]
    assert all(f.sensor == "goes19" for f in frames)
    assert frames[0].bt.dtype == np.float32
    assert frames[0].bt.shape == (32, 32)


def test_frames_from_cache_honours_limit(tmp_path):
    from sattsr.infer.pipeline import frames_from_cache

    _write_cache_frames(tmp_path / "cache", n=6)
    assert len(frames_from_cache(tmp_path / "cache", "goes19", limit=3)) == 3


def test_frames_from_cache_skips_an_unreadable_frame(tmp_path):
    from sattsr.infer.pipeline import frames_from_cache

    refs = _write_cache_frames(tmp_path / "cache", n=4)
    refs[1].path.write_text("not a npy", encoding="utf-8")
    assert len(frames_from_cache(tmp_path / "cache", "goes19")) == 3


def test_frames_from_cache_is_empty_for_an_empty_cache(tmp_path):
    from sattsr.infer.pipeline import frames_from_cache

    (tmp_path / "cache").mkdir()
    assert frames_from_cache(tmp_path / "cache", "goes19") == []
