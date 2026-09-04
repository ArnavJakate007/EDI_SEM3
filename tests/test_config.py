from __future__ import annotations

import pytest

from sattsr.config import Config, load_config


def test_load_config_applies_defaults(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "data:\n"
        "  sensor: goes19\n"
        "  raw_root: data/raw/goes19\n"
        "  cache_root: cache/goes19\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    assert cfg.data.sensor == "goes19"
    assert cfg.data.grid.resolution_deg == pytest.approx(0.05)
    assert cfg.data.normalization.data_range == pytest.approx(150.0)
    assert cfg.model.scales == [4, 2, 1]
    assert cfg.train.tile_size == 256


def test_load_config_overrides_nested_values(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "data:\n"
        "  sensor: insat3dr\n"
        "  raw_root: data/raw/insat\n"
        "  cache_root: cache/insat\n"
        "  cadence_minutes: 30\n"
        "  normalization:\n"
        "    bt_min: 190\n"
        "    bt_max: 320\n"
        "train:\n"
        "  freeze_prefixes: ['flow_net']\n"
        "  epochs: 5\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.data.cadence_minutes == pytest.approx(30.0)
    assert cfg.data.normalization.data_range == pytest.approx(130.0)
    assert cfg.train.freeze_prefixes == ["flow_net"]
    assert cfg.train.epochs == 5


def test_unknown_sensor_is_rejected(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "data:\n  sensor: sentinel2\n  raw_root: x\n  cache_root: y\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_config(cfg_path)
