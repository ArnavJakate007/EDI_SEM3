from __future__ import annotations

from pathlib import Path

import pytest

from sattsr.config import load_config, load_pretrain_config
from sattsr.geo.grid import TargetGrid

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

#: configs/train.yaml is the multi-sensor training plan, not a per-sensor config;
#: it validates against PretrainConfig instead and is checked separately below.
PLAN_CONFIGS = {"train.yaml"}
CONFIGS = sorted(p for p in CONFIG_DIR.glob("*.yaml") if p.name not in PLAN_CONFIGS)


def test_configs_directory_is_not_empty():
    assert CONFIGS, "no shipped configs found"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_shipped_config_validates(path):
    cfg = load_config(path)
    grid = TargetGrid.from_config(cfg.data.grid)
    assert grid.shape[0] > 0 and grid.shape[1] > 0
    assert cfg.data.cadence_minutes > 0
    assert cfg.train.tile_size % 32 == 0, "tile size must clear the coarsest model scale"
    assert 0 < cfg.train.tile_overlap < cfg.train.tile_size


def test_the_required_insat_validation_configs_exist():
    names = {p.name for p in CONFIGS}
    assert {"insat.yaml", "insat_rapidscan.yaml", "insat_staggered.yaml"} <= names


def test_insat_validation_configs_use_their_documented_cadences():
    rapid = load_config(CONFIG_DIR / "insat_rapidscan.yaml")
    staggered = load_config(CONFIG_DIR / "insat_staggered.yaml")
    # Native rapid-scan cadence is 4 min 30 s, not a clean 4 minutes.
    assert rapid.data.cadence_minutes == pytest.approx(4.5)
    assert staggered.data.cadence_minutes == pytest.approx(15.0)
    assert rapid.data.sensor.startswith("insat")
    assert staggered.data.sensor.startswith("insat")


def test_the_training_plan_validates_and_holds_out_rapid_scan():
    """configs/train.yaml must keep rapid-scan out of every training split."""
    plan = load_pretrain_config(CONFIG_DIR / "train.yaml")

    assert plan.splits.pretrain, "pretraining needs at least one sensor"
    assert plan.splits.rapid_scan_eval, "the held-out rapid-scan split must be populated"

    training_sources = {Path(p).name for p in [*plan.splits.pretrain, *plan.splits.finetune]}
    holdout_sources = {Path(p).name for p in plan.splits.rapid_scan_eval}
    assert not (training_sources & holdout_sources), (
        "a config feeding both training and the held-out rapid-scan split would make "
        "the evaluation circular"
    )
    assert 0.0 < plan.loader.val_fraction < 1.0
    assert plan.loader.tile_size % 32 == 0


def test_every_plan_referenced_config_exists():
    plan = load_pretrain_config(CONFIG_DIR / "train.yaml")
    repo_root = CONFIG_DIR.parent
    for group in (plan.splits.pretrain, plan.splits.finetune, plan.splits.rapid_scan_eval):
        for rel in group:
            assert (repo_root / rel).exists(), f"train.yaml references missing config {rel}"
