from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from typer.testing import CliRunner

from sattsr.cli import app, resolve_device
from sattsr.config import ModelConfig
from sattsr.models.interpolator import build_model
from sattsr.train.checkpoint import save_checkpoint
from tests.conftest import make_goes_series

runner = CliRunner()
CFG = ModelConfig(base_channels=8, scales=[2, 1], use_raft_init=False)

CONFIG_YAML = """
data:
  sensor: goes19
  raw_root: {raw}
  cache_root: {cache}
  cadence_minutes: 30
  tolerance_minutes: 5
  grid:
    lat_min: 5.0
    lat_max: 13.0
    lon_min: 78.0
    lon_max: 86.0
    resolution_deg: 0.25
model:
  base_channels: 8
  scales: [2, 1]
  use_raft_init: false
train:
  tile_size: 32
  tile_overlap: 8
  batch_size: 2
  epochs: 1
  amp: false
  num_workers: 0
  checkpoint_dir: {ckpt}
"""


def _write_config(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    make_goes_series(raw, 6, step_minutes=30, size=48)
    path = tmp_path / "config.yaml"
    path.write_text(
        CONFIG_YAML.format(
            raw=(tmp_path / "raw").as_posix(),
            cache=(tmp_path / "cache").as_posix(),
            ckpt=(tmp_path / "ckpt").as_posix(),
        ),
        encoding="utf-8",
    )
    return path


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("prepare", "train", "finetune", "evaluate", "infer", "render", "serve",
                    "prune"):
        assert command in result.stdout


def test_resolve_device_accepts_auto_and_explicit():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_prepare_builds_the_cache(tmp_path):
    config = _write_config(tmp_path)
    result = runner.invoke(app, ["prepare", "--config", str(config)])
    assert result.exit_code == 0, result.output
    cached = list((tmp_path / "cache").rglob("*.npy"))
    assert len(cached) == 6
    assert np.load(cached[0]).shape == (32, 32)


def test_train_writes_a_checkpoint(tmp_path):
    config = _write_config(tmp_path)
    assert runner.invoke(app, ["prepare", "--config", str(config)]).exit_code == 0
    result = runner.invoke(app, ["train", "--config", str(config), "--device", "cpu"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "ckpt" / "best.pt").exists()


def test_infer_render_and_evaluate_produce_a_complete_run(tmp_path):
    config = _write_config(tmp_path)
    checkpoint = save_checkpoint(tmp_path / "m.pt", model=build_model(CFG), epoch=0,
                                 best_metric=0.0)
    run_dir = tmp_path / "runs" / "smoke"

    assert runner.invoke(app, ["prepare", "--config", str(config)]).exit_code == 0

    result = runner.invoke(app, [
        "infer", "--config", str(config), "--checkpoint", str(checkpoint),
        "--input", str(tmp_path / "raw"), "--out", str(run_dir),
        "--factor", "2", "--device", "cpu", "--limit", "3",
    ])
    assert result.exit_code == 0, result.output
    assert (run_dir / "output.nc").exists()

    assert runner.invoke(app, ["render", "--run", str(run_dir)]).exit_code == 0
    assert (run_dir / "frames" / "000.png").exists()
    assert (run_dir / "animations" / "original.gif").exists()
    assert (run_dir / "animations" / "interpolated.gif").exists()

    result = runner.invoke(app, [
        "evaluate", "--config", str(config), "--checkpoint", str(checkpoint),
        "--out", str(run_dir / "report.json"), "--device", "cpu", "--limit", "2",
    ])
    assert result.exit_code == 0, result.output
    assert (run_dir / "report.json").exists()


def test_prune_is_a_dry_run_without_yes(tmp_path):
    config = _write_config(tmp_path)
    assert runner.invoke(app, ["prepare", "--config", str(config)]).exit_code == 0
    before = len(list((tmp_path / "raw").glob("*.nc")))
    result = runner.invoke(app, ["prune", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "would remove" in result.output
    assert len(list((tmp_path / "raw").glob("*.nc"))) == before


def test_missing_config_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["prepare", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
