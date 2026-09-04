"""Typed configuration for the sattsr pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Sensor = Literal["goes19", "himawari8", "insat3dr", "insat3ds"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GridConfig(_Base):
    """Common lat-lon target grid. Defaults cover the INSAT Indian-region disc."""

    lat_min: float = -10.0
    lat_max: float = 40.0
    lon_min: float = 60.0
    lon_max: float = 110.0
    resolution_deg: float = Field(0.05, gt=0.0)


class NormalizationConfig(_Base):
    """Brightness-temperature range used to map Kelvin onto [0, 1]."""

    bt_min: float = 180.0
    bt_max: float = 330.0

    @property
    def data_range(self) -> float:
        """Width of the BT range in Kelvin, used as PSNR/SSIM data_range."""
        return self.bt_max - self.bt_min


class DataConfig(_Base):
    sensor: Sensor
    raw_root: Path
    cache_root: Path
    grid: GridConfig = GridConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    cadence_minutes: float = 10.0
    tolerance_minutes: float = 2.0


class ModelConfig(_Base):
    base_channels: int = 64
    scales: list[int] = [4, 2, 1]
    use_raft_init: bool = True
    flow_channels: int = 96
    flow_radius: int = 3
    flow_iters: int = 4


class LossConfig(_Base):
    w_recon: float = 1.0
    w_ssim: float = 0.25
    w_smooth: float = 0.05
    w_consistency: float = 0.10
    w_radiometric: float = 0.05


class TrainConfig(_Base):
    tile_size: int = 256
    tile_overlap: int = 32
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-5
    epochs: int = 40
    amp: bool = True
    num_workers: int = 4
    seed: int = 1337
    checkpoint_dir: Path = Path("checkpoints")
    val_fraction: float = 0.1
    freeze_prefixes: list[str] = []
    lr_head_multiplier: float = 10.0


class Config(_Base):
    data: DataConfig
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)
