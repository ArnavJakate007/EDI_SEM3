from __future__ import annotations

from pathlib import Path

import pytest

from sattsr.config import load_config
from sattsr.geo.grid import TargetGrid

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


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
    assert rapid.data.cadence_minutes == pytest.approx(4.0)
    assert staggered.data.cadence_minutes == pytest.approx(15.0)
    assert rapid.data.sensor.startswith("insat")
    assert staggered.data.sensor.startswith("insat")
