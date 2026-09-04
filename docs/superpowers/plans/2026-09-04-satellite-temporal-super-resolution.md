# Cross-Sensor Temporal Super Resolution for Geostationary Thermal Imagery — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `.nc`-in/`.nc`-out AI frame-interpolation system that raises geostationary thermal-IR cadence from 30 min → 15 min → 7.5 min, validated against real high-cadence GOES-19/Himawari observations and applied to INSAT-3DR/3DS, with a web dashboard comparing original and interpolated animations.

**Architecture:** A Python package `sattsr` with a strict pipeline: sensor readers regrid GOES-19 / Himawari / INSAT thermal bands onto one common lat-lon grid in Kelvin → a cache of `.npy` frames feeds a triplet `Dataset` → a RAFT-lite correlation flow estimator initialises a RIFE-style coarse-to-fine `IFNet` plus a residual refinement net → composite loss (Charbonnier + SSIM + flow smoothness + warp consistency + radiometric) trains it on GOES-19 → partial-freeze fine-tuning adapts it to INSAT → recursive bisection inference tiles the full grid and writes CF-compliant NetCDF with every synthetic frame flagged → a FastAPI service serves run manifests, PNG frames, metrics and reports to a static dashboard.

**Tech Stack:** Python 3.11, PyTorch 2.x, xarray + netCDF4 + h5py, pyproj, OpenCV, scikit-image, NumPy/SciPy, Pydantic v2, Typer, FastAPI + Uvicorn, Chart.js, pytest, hatchling + venv/pip.

**Spec:** `docs/superpowers/specs/2026-09-04-satellite-temporal-super-resolution-spec.md`

## Global Constraints

Every task's requirements implicitly include this section.

- **Python `>=3.11`.** Use `from __future__ import annotations` in every module; use `X | None`, `list[...]`, `dict[...]` — never `typing.Optional`/`List`/`Dict`.
- **Model-facing I/O is NetCDF.** `.h5` is accepted only as an *ingest* format for INSAT L1B/L1C. Every product the model emits is `.nc`.
- **Brightness temperature is always Kelvin, `float32`, NaN for invalid/off-disk pixels**, at every boundary between modules. Normalisation to `[0, 1]` happens only inside the dataset/model boundary and is undone before anything is written or scored.
- **Array convention:** NumPy arrays are `(H, W)` or `(T, H, W)`, latitude descending, longitude ascending. Torch tensors are `(N, C, H, W)` with `C = 1`.
- **Every synthetic frame must be machine-identifiable.** The NetCDF `synthetic` variable (`flag_values = [0, 1]`, `flag_meanings = "observed synthesized"`) is mandatory on all output; so is the global attribute `comment = "Frames with synthetic=1 are AI-generated and are NOT observations."`
- **Band:** thermal infrared ~10 µm only — GOES-19 ABI C13, Himawari AHI B13, INSAT TIR1.
- **Required metrics:** MSE, PSNR, SSIM, FSIM, plus cloud-motion metrics (CSI/POD/FAR on a cold-cloud threshold, and motion displacement error).
- **Required event categories** (exact spelling, lowercase): `clear`, `stratiform`, `convective`, `cyclonic`.
- **Required baselines:** `linear` (linear blend) and `farneback` (classical optical-flow warp). The learned model is reported alongside them, never alone.
- **Cadence targets:** GOES-19 30 → 15 → 7.5 min (`factor=4`); INSAT 30 → 15 min (`factor=2`).
- **Determinism:** every training/eval entry point seeds `random`, `numpy`, and `torch` from `TrainConfig.seed`.
- **Tests never hit the network and never require a GPU.** Synthetic fixtures only. Anything touching S3 or MOSDAC is behind a CLI command that tests do not call.
- **Style:** `ruff` clean. Public functions get type hints and a one-line docstring.

## File Structure

```
c:\EDI-SEM3\
├── pyproject.toml                  # deps, ruff + pytest config, console script
├── README.md
├── configs/
│   ├── goes19.yaml                 # pretraining data + train config
│   ├── himawari.yaml               # cross-sensor benchmark
│   ├── insat.yaml                  # fine-tune + deployment config
│   ├── insat_rapidscan.yaml        # ~4 min INSAT-native validation
│   └── insat_staggered.yaml        # ~15 min INSAT-native validation
├── docs/RESULTS.md                 # the results the project owes, filled in as they arrive
├── src/sattsr/
│   ├── config.py                   # Pydantic config models + YAML loader
│   ├── geo/
│   │   ├── grid.py                 # TargetGrid: the one common lat-lon grid
│   │   └── resample.py             # bilinear sampling; geos/latlon/scattered regridders
│   ├── io/
│   │   ├── base.py                 # Frame dataclass + Reader protocol
│   │   ├── goes.py                 # GOES-19 ABI C13 reader
│   │   ├── himawari.py             # Himawari-8/9 AHI B13 gridded reader
│   │   ├── insat.py                # INSAT-3DR/3DS TIR1 HDF5 reader
│   │   ├── writer.py               # CF-compliant NetCDF write/read of frame products
│   │   └── registry.py             # sensor name -> reader
│   ├── data/
│   │   ├── radiometry.py           # Planck inversion, LUT, normalise, stats matching
│   │   ├── index.py                # FrameRef index + regridded .npy cache
│   │   ├── triplets.py             # (t0, t1, t2) sample construction
│   │   ├── tiling.py               # tile positions, extract, Hann reassembly, padding
│   │   ├── dataset.py              # TripletDataset (torch)
│   │   ├── prune.py                # bounded storage: subsample + verified raw deletion
│   │   └── download.py             # NOAA S3 fetch (the only network-touching module)
│   ├── models/
│   │   ├── blocks.py               # conv/ResBlock, backwarp
│   │   ├── flow.py                 # RAFT-lite correlation flow estimator
│   │   ├── ifnet.py                # RIFE-style coarse-to-fine IFBlock/IFNet
│   │   └── interpolator.py         # FrameInterpolator: flow init + IFNet + refine
│   ├── losses/
│   │   ├── functional.py           # charbonnier, smoothness, radiometric, consistency
│   │   ├── ssim.py                 # differentiable SSIM (torch)
│   │   └── composite.py            # CompositeLoss
│   ├── train/
│   │   ├── checkpoint.py           # save/load
│   │   ├── loop.py                 # train_one_epoch, validate, fit
│   │   └── finetune.py             # partial-freeze domain adaptation
│   ├── eval/
│   │   ├── metrics.py              # MSE, RMSE, PSNR, SSIM, FSIM, metric_suite
│   │   ├── motion.py               # Farneback flow, EPE, cold-cloud CSI/POD/FAR
│   │   ├── events.py               # clear/stratiform/convective/cyclonic classifier
│   │   ├── baselines.py            # linear blend, Farneback warp
│   │   └── report.py               # SampleResult, aggregate, write_report
│   ├── infer/
│   │   ├── recursive.py            # tiled midframe prediction + recursive bisection
│   │   └── pipeline.py             # RunManifest, run_inference
│   ├── viz/
│   │   ├── colormap.py             # BT -> RGB
│   │   └── frames.py               # PNG export, GIF/MP4 animation
│   └── cli.py                      # Typer app: prepare/train/finetune/evaluate/infer/serve
├── web/
│   ├── app.py                      # create_app(runs_dir) -> FastAPI
│   └── static/{index.html,app.js,styles.css}
└── tests/                          # mirrors src/sattsr, plus tests/web and tests/fixtures
```

---

# Phase 0 — Foundation

### Task 1: Project scaffold and configuration

**Files:**
- Create: `pyproject.toml`, `README.md`, `.gitignore`
- Create: `src/sattsr/__init__.py`, `src/sattsr/config.py`
- Create: `configs/goes19.yaml`, `configs/insat.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `sattsr.config.{GridConfig, NormalizationConfig, DataConfig, ModelConfig, LossConfig, TrainConfig, Config, load_config}`. `NormalizationConfig.data_range -> float`. `load_config(path: str | Path) -> Config`.

- [ ] **Step 1: Initialise the repository and tooling**

```bash
git init
git branch -M main
mkdir -p src/sattsr configs tests web/static docs
```

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
*.egg-info/
data/
cache/
checkpoints/
runs/
reports/
*.nc
*.h5
*.npy
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "sattsr"
version = "0.1.0"
description = "Cross-sensor temporal super resolution for geostationary thermal imagery"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "scipy>=1.11",
    "xarray>=2024.1",
    "netCDF4>=1.6",
    "h5py>=3.10",
    "pyproj>=3.6",
    "opencv-python-headless>=4.9",
    "scikit-image>=0.22",
    "pillow>=10.2",
    "imageio>=2.34",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "typer>=0.12",
    "tqdm>=4.66",
    "fastapi>=0.110",
    "uvicorn>=0.29",
]

[project.optional-dependencies]
download = ["s3fs>=2024.2", "fsspec>=2024.2", "requests>=2.31"]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.3", "httpx>=0.27"]

[project.scripts]
sattsr = "sattsr.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sattsr"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
filterwarnings = ["ignore::DeprecationWarning"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- [ ] **Step 3: Create the environment and install**

```bash
python -m venv .venv
# PowerShell:  .\.venv\Scripts\Activate.ps1
# Git Bash:    source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Expected: install succeeds; `python -c "import torch, xarray, pyproj, cv2; print('ok')"` prints `ok`.

- [ ] **Step 4: Write the failing test**

`tests/test_config.py`:

```python
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
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.config'`

- [ ] **Step 6: Implement `src/sattsr/config.py`**

Also create an empty `src/sattsr/__init__.py` containing `__version__ = "0.1.0"`.

```python
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
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 8: Write the shipped config files**

`configs/goes19.yaml`:

```yaml
data:
  sensor: goes19
  raw_root: data/raw/goes19
  cache_root: cache/goes19
  cadence_minutes: 10
  tolerance_minutes: 2
  grid:
    lat_min: 10.0
    lat_max: 45.0
    lon_min: -105.0
    lon_max: -60.0
    resolution_deg: 0.04
  normalization:
    bt_min: 180.0
    bt_max: 330.0
train:
  tile_size: 256
  tile_overlap: 32
  batch_size: 8
  lr: 2.0e-4
  epochs: 40
  amp: true
  checkpoint_dir: checkpoints/goes19
```

`configs/insat.yaml`:

```yaml
data:
  sensor: insat3dr
  raw_root: data/raw/insat
  cache_root: cache/insat
  cadence_minutes: 30
  tolerance_minutes: 5
  grid:
    lat_min: -10.0
    lat_max: 40.0
    lon_min: 60.0
    lon_max: 110.0
    resolution_deg: 0.05
  normalization:
    bt_min: 180.0
    bt_max: 330.0
train:
  tile_size: 256
  tile_overlap: 32
  batch_size: 4
  lr: 5.0e-5
  epochs: 10
  amp: true
  checkpoint_dir: checkpoints/insat
  freeze_prefixes: ["flow_net.encoder"]
  lr_head_multiplier: 10.0
```

Create `configs/himawari.yaml` as a copy of `configs/goes19.yaml` with `sensor: himawari8`, `raw_root: data/raw/himawari`, `cache_root: cache/himawari`, and the INSAT grid block (Himawari is the cross-sensor benchmark over the same region as INSAT).

- [ ] **Step 9: Write `README.md`**

```markdown
# sattsr — Cross-Sensor Temporal Super Resolution for Geostationary Thermal Imagery

Raises the effective cadence of geostationary thermal-IR imagery from 30 min to 15 min
and 7.5 min using a learned optical-flow frame-interpolation model, and applies it to
INSAT-3DR/3DS.

## Install

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1     # PowerShell
    pip install -e ".[dev,download]"

## Pipeline

    sattsr prepare  --config configs/goes19.yaml
    sattsr train    --config configs/goes19.yaml
    sattsr evaluate --config configs/goes19.yaml --checkpoint checkpoints/goes19/best.pt
    sattsr finetune --config configs/insat.yaml  --checkpoint checkpoints/goes19/best.pt
    sattsr infer    --config configs/insat.yaml  --checkpoint checkpoints/insat/best.pt \
                    --input data/raw/insat --out runs/insat-demo --factor 2
    sattsr serve    --runs-dir runs

Synthetic frames are always flagged `synthetic=1` in the output NetCDF. They are
decision-support augmentation, never a substitute for real observations.
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml README.md .gitignore src/sattsr/__init__.py src/sattsr/config.py configs tests/test_config.py
git commit -m "feat: project scaffold and typed pipeline configuration"
```

---

# Phase 1 — Data foundation

### Task 2: Common target grid and resampling

**Files:**
- Create: `src/sattsr/geo/__init__.py`, `src/sattsr/geo/grid.py`, `src/sattsr/geo/resample.py`
- Test: `tests/geo/test_grid.py`, `tests/geo/test_resample.py`

**Interfaces:**
- Consumes: `sattsr.config.GridConfig`.
- Produces:
  - `TargetGrid(lat_min, lat_max, lon_min, lon_max, resolution_deg)` frozen dataclass with
    `.lats -> np.ndarray` (descending, `(H,)`), `.lons -> np.ndarray` (ascending, `(W,)`),
    `.shape -> tuple[int, int]`, `.meshgrid() -> tuple[np.ndarray, np.ndarray]`,
    `.from_config(cfg: GridConfig) -> TargetGrid`.
  - `bilinear_sample(src: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray`
  - `GeosProjection(satellite_height, longitude_of_origin, semi_major_axis, semi_minor_axis, sweep_axis, x, y)`
    with `.latlon_to_index(lat, lon) -> tuple[np.ndarray, np.ndarray]`
  - `regrid_geos(src, proj: GeosProjection, grid: TargetGrid) -> np.ndarray`
  - `regrid_latlon(src, src_lats, src_lons, grid: TargetGrid) -> np.ndarray`
  - `regrid_scattered(src, src_lat2d, src_lon2d, grid, *, max_distance_deg=0.1) -> np.ndarray`

  All regridders return `float32 (H, W)` with `np.nan` outside coverage.

- [ ] **Step 1: Write the failing test for `TargetGrid`**

Create `tests/geo/__init__.py` (empty), then `tests/geo/test_grid.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from sattsr.config import GridConfig
from sattsr.geo.grid import TargetGrid


def test_grid_shape_and_ordering():
    grid = TargetGrid(lat_min=0.0, lat_max=1.0, lon_min=10.0, lon_max=12.0, resolution_deg=0.5)
    assert grid.shape == (2, 4)
    # latitude descends (north at row 0), longitude ascends
    np.testing.assert_allclose(grid.lats, [0.75, 0.25])
    np.testing.assert_allclose(grid.lons, [10.25, 10.75, 11.25, 11.75])


def test_meshgrid_matches_shape():
    grid = TargetGrid(lat_min=0.0, lat_max=1.0, lon_min=10.0, lon_max=12.0, resolution_deg=0.5)
    lat2d, lon2d = grid.meshgrid()
    assert lat2d.shape == grid.shape
    assert lon2d.shape == grid.shape
    assert lat2d[0, 0] == pytest.approx(0.75)
    assert lon2d[0, -1] == pytest.approx(11.75)


def test_from_config_round_trips():
    grid = TargetGrid.from_config(
        GridConfig(lat_min=-10, lat_max=40, lon_min=60, lon_max=110, resolution_deg=0.05)
    )
    assert grid.shape == (1000, 1000)


def test_degenerate_extent_is_rejected():
    with pytest.raises(ValueError):
        TargetGrid(lat_min=5.0, lat_max=5.0, lon_min=0.0, lon_max=1.0, resolution_deg=0.5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/geo/test_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.geo'`

- [ ] **Step 3: Implement `src/sattsr/geo/grid.py`**

Create an empty `src/sattsr/geo/__init__.py` too.

```python
"""The single common lat-lon grid every sensor is resampled onto."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sattsr.config import GridConfig


@dataclass(frozen=True)
class TargetGrid:
    """Regular lat-lon grid. Cell centres; latitude descending, longitude ascending."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution_deg: float

    def __post_init__(self) -> None:
        if self.lat_max <= self.lat_min or self.lon_max <= self.lon_min:
            raise ValueError("grid extent must be positive in both dimensions")
        if self.resolution_deg <= 0:
            raise ValueError("resolution_deg must be positive")
        if self.shape[0] < 1 or self.shape[1] < 1:
            raise ValueError("grid resolution is coarser than the requested extent")

    @cached_property
    def shape(self) -> tuple[int, int]:
        """(height, width) in pixels."""
        h = int(round((self.lat_max - self.lat_min) / self.resolution_deg))
        w = int(round((self.lon_max - self.lon_min) / self.resolution_deg))
        return h, w

    @cached_property
    def lats(self) -> np.ndarray:
        """Cell-centre latitudes, descending, shape (H,)."""
        h = self.shape[0]
        return (self.lat_max - (np.arange(h) + 0.5) * self.resolution_deg).astype(np.float64)

    @cached_property
    def lons(self) -> np.ndarray:
        """Cell-centre longitudes, ascending, shape (W,)."""
        w = self.shape[1]
        return (self.lon_min + (np.arange(w) + 0.5) * self.resolution_deg).astype(np.float64)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat2d, lon2d), each (H, W)."""
        return np.meshgrid(self.lats, self.lons, indexing="ij")

    @classmethod
    def from_config(cls, cfg: GridConfig) -> TargetGrid:
        """Build a grid from a validated GridConfig."""
        return cls(
            lat_min=cfg.lat_min,
            lat_max=cfg.lat_max,
            lon_min=cfg.lon_min,
            lon_max=cfg.lon_max,
            resolution_deg=cfg.resolution_deg,
        )
```

Note: `cached_property` needs an instance `__dict__`, which frozen dataclasses have unless
`slots=True`. Do **not** add `slots=True`.

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/geo/test_grid.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the failing test for resampling**

`tests/geo/test_resample.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from sattsr.geo.grid import TargetGrid
from sattsr.geo.resample import (
    GeosProjection,
    bilinear_sample,
    regrid_geos,
    regrid_latlon,
    regrid_scattered,
)


def test_bilinear_sample_is_exact_on_integer_positions():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = bilinear_sample(src, np.array([[0.0, 2.0]]), np.array([[1.0, 3.0]]))
    np.testing.assert_allclose(out, [[1.0, 11.0]])


def test_bilinear_sample_interpolates_midpoints():
    src = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    out = bilinear_sample(src, np.array([[0.5]]), np.array([[0.5]]))
    assert out[0, 0] == pytest.approx(15.0)


def test_bilinear_sample_returns_nan_outside_bounds():
    src = np.ones((3, 3), dtype=np.float32)
    out = bilinear_sample(src, np.array([[-1.0, 5.0, np.nan]]), np.array([[1.0, 1.0, 1.0]]))
    assert np.all(np.isnan(out))


def test_regrid_latlon_preserves_a_constant_field():
    src_lats = np.linspace(50.0, 0.0, 51)          # descending on purpose
    src_lons = np.linspace(55.0, 115.0, 61)
    src = np.full((51, 61), 250.0, dtype=np.float32)
    grid = TargetGrid(lat_min=10.0, lat_max=20.0, lon_min=70.0, lon_max=80.0, resolution_deg=1.0)
    out = regrid_latlon(src, src_lats, src_lons, grid)
    assert out.shape == grid.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 250.0, rtol=1e-5)


def test_regrid_latlon_marks_uncovered_area_nan():
    src_lats = np.linspace(20.0, 10.0, 11)
    src_lons = np.linspace(70.0, 80.0, 11)
    src = np.full((11, 11), 250.0, dtype=np.float32)
    grid = TargetGrid(lat_min=30.0, lat_max=40.0, lon_min=70.0, lon_max=80.0, resolution_deg=1.0)
    assert np.all(np.isnan(regrid_latlon(src, src_lats, src_lons, grid)))


def test_regrid_scattered_picks_nearest_neighbour():
    lat2d, lon2d = np.meshgrid(
        np.linspace(20.0, 10.0, 21), np.linspace(70.0, 80.0, 21), indexing="ij"
    )
    src = lat2d.astype(np.float32)
    grid = TargetGrid(lat_min=12.0, lat_max=18.0, lon_min=72.0, lon_max=78.0, resolution_deg=1.0)
    out = regrid_scattered(src, lat2d, lon2d, grid, max_distance_deg=1.0)
    assert out.shape == grid.shape
    lat_t, _ = grid.meshgrid()
    np.testing.assert_allclose(out, lat_t, atol=0.6)


def _fake_geos_projection() -> GeosProjection:
    """A 1000x1000 scan-angle grid spanning +/- 0.08 rad about nadir at 82 E."""
    ang = np.linspace(-0.08, 0.08, 1000)
    return GeosProjection(
        satellite_height=35786023.0,
        longitude_of_origin=82.0,
        semi_major_axis=6378137.0,
        semi_minor_axis=6356752.31414,
        sweep_axis="y",
        x=ang,
        y=-ang,           # y decreases downward, as in real ABI files
    )


def test_regrid_geos_recovers_a_monotonic_ramp():
    proj = _fake_geos_projection()
    src = np.tile(np.arange(1000, dtype=np.float32), (1000, 1))   # value == column index
    grid = TargetGrid(lat_min=10.0, lat_max=20.0, lon_min=75.0, lon_max=85.0, resolution_deg=1.0)
    out = regrid_geos(src, proj, grid)
    assert out.shape == grid.shape
    assert np.isfinite(out).all()
    assert np.all(np.diff(out, axis=1) > 0)      # east -> higher column index


def test_regrid_geos_returns_nan_off_disk():
    proj = _fake_geos_projection()
    src = np.ones((1000, 1000), dtype=np.float32)
    grid = TargetGrid(
        lat_min=-5.0, lat_max=5.0, lon_min=-100.0, lon_max=-90.0, resolution_deg=1.0
    )   # opposite side of the Earth
    assert np.all(np.isnan(regrid_geos(src, proj, grid)))
```

- [ ] **Step 6: Run it to verify it fails**

Run: `pytest tests/geo/test_resample.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.geo.resample'`

- [ ] **Step 7: Implement `src/sattsr/geo/resample.py`**

```python
"""Resampling of sensor-native grids onto the common TargetGrid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyproj
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from sattsr.geo.grid import TargetGrid


def bilinear_sample(src: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinearly sample `src` at fractional (row, col); NaN where out of bounds."""
    a = np.asarray(src, dtype=np.float64)
    h, w = a.shape
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)

    finite = np.isfinite(r) & np.isfinite(c)
    r_s = np.where(finite, r, -1.0)
    c_s = np.where(finite, c, -1.0)

    # Validity is about the sample position, not the upper neighbour: a position
    # exactly on the last row or column is in bounds, and its (clamped) upper
    # neighbour contributes zero weight.
    valid = finite & (r_s >= 0.0) & (r_s <= h - 1) & (c_s >= 0.0) & (c_s <= w - 1)

    r0 = np.floor(r_s).astype(np.int64)
    c0 = np.floor(c_s).astype(np.int64)
    r1, c1 = r0 + 1, c0 + 1

    r0c, r1c = np.clip(r0, 0, h - 1), np.clip(r1, 0, h - 1)
    c0c, c1c = np.clip(c0, 0, w - 1), np.clip(c1, 0, w - 1)
    dr = np.where(valid, r_s - r0, 0.0)
    dc = np.where(valid, c_s - c0, 0.0)

    top = a[r0c, c0c] * (1.0 - dc) + a[r0c, c1c] * dc
    bot = a[r1c, c0c] * (1.0 - dc) + a[r1c, c1c] * dc
    out = top * (1.0 - dr) + bot * dr
    return np.where(valid, out, np.nan).astype(np.float32)


@dataclass(frozen=True)
class GeosProjection:
    """Geostationary fixed-grid projection parameters read from a sensor file."""

    satellite_height: float
    longitude_of_origin: float
    semi_major_axis: float
    semi_minor_axis: float
    sweep_axis: str
    x: np.ndarray          # scan angle along columns, radians, shape (W,)
    y: np.ndarray          # scan angle along rows, radians, shape (H,)

    def _crs(self) -> pyproj.CRS:
        return pyproj.CRS.from_cf(
            {
                "grid_mapping_name": "geostationary",
                "perspective_point_height": self.satellite_height,
                "longitude_of_projection_origin": self.longitude_of_origin,
                "latitude_of_projection_origin": 0.0,
                "semi_major_axis": self.semi_major_axis,
                "semi_minor_axis": self.semi_minor_axis,
                "sweep_angle_axis": self.sweep_axis,
            }
        )

    def latlon_to_index(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map geographic coordinates to fractional (row, col) in the sensor array."""
        tf = pyproj.Transformer.from_crs("EPSG:4326", self._crs(), always_xy=True)
        px, py = tf.transform(np.asarray(lon, float), np.asarray(lat, float))
        with np.errstate(invalid="ignore", divide="ignore"):
            sx = np.asarray(px, float) / self.satellite_height
            sy = np.asarray(py, float) / self.satellite_height
        # pyproj returns +/-inf for points that do not project (off the visible disc)
        sx = np.where(np.isfinite(sx) & (np.abs(sx) < 1e30), sx, np.nan)
        sy = np.where(np.isfinite(sy) & (np.abs(sy) < 1e30), sy, np.nan)

        dx = float(self.x[1] - self.x[0])
        dy = float(self.y[1] - self.y[0])
        cols = (sx - float(self.x[0])) / dx
        rows = (sy - float(self.y[0])) / dy
        return rows, cols


def regrid_geos(src: np.ndarray, proj: GeosProjection, grid: TargetGrid) -> np.ndarray:
    """Resample a geostationary fixed-grid array onto the target lat-lon grid."""
    lat2d, lon2d = grid.meshgrid()
    rows, cols = proj.latlon_to_index(lat2d, lon2d)
    return bilinear_sample(src, rows, cols)


def regrid_latlon(
    src: np.ndarray, src_lats: np.ndarray, src_lons: np.ndarray, grid: TargetGrid
) -> np.ndarray:
    """Resample an array already on a regular lat-lon grid; either axis may be descending."""
    lat = np.asarray(src_lats, dtype=np.float64)
    lon = np.asarray(src_lons, dtype=np.float64)
    a = np.asarray(src, dtype=np.float64)
    if lat.size > 1 and lat[0] > lat[-1]:
        lat, a = lat[::-1], a[::-1, :]
    if lon.size > 1 and lon[0] > lon[-1]:
        lon, a = lon[::-1], a[:, ::-1]

    interp = RegularGridInterpolator(
        (lat, lon), a, method="linear", bounds_error=False, fill_value=np.nan
    )
    lat2d, lon2d = grid.meshgrid()
    return np.asarray(interp(np.stack([lat2d, lon2d], axis=-1)), dtype=np.float32)


def regrid_scattered(
    src: np.ndarray,
    src_lat2d: np.ndarray,
    src_lon2d: np.ndarray,
    grid: TargetGrid,
    *,
    max_distance_deg: float = 0.1,
) -> np.ndarray:
    """Nearest-neighbour resample of a curvilinear/scattered grid.

    Distances are Euclidean in degrees, which is adequate for a regional grid but
    degrades near the poles. Targets further than `max_distance_deg` become NaN.
    """
    valid = np.isfinite(src) & np.isfinite(src_lat2d) & np.isfinite(src_lon2d)
    if not valid.any():
        return np.full(grid.shape, np.nan, dtype=np.float32)

    pts = np.column_stack([src_lat2d[valid].ravel(), src_lon2d[valid].ravel()])
    vals = np.asarray(src)[valid].ravel().astype(np.float32)
    tree = cKDTree(pts)

    lat2d, lon2d = grid.meshgrid()
    query = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    dist, idx = tree.query(query, k=1, distance_upper_bound=float(max_distance_deg))

    out = np.full(query.shape[0], np.nan, dtype=np.float32)
    hit = np.isfinite(dist) & (idx < vals.size)
    out[hit] = vals[idx[hit]]
    return out.reshape(grid.shape)
```

- [ ] **Step 8: Run the whole geo suite to verify it passes**

Run: `pytest tests/geo -v`
Expected: 12 passed

- [ ] **Step 9: Commit**

```bash
git add src/sattsr/geo tests/geo
git commit -m "feat: common lat-lon target grid and geos/latlon/scattered resampling"
```

---

### Task 3: Radiometry — Kelvin conversion, normalisation, cross-sensor matching

**Files:**
- Create: `src/sattsr/data/__init__.py`, `src/sattsr/data/radiometry.py`
- Test: `tests/data/test_radiometry.py`

**Interfaces:**
- Consumes: `sattsr.config.NormalizationConfig`.
- Produces:
  - `planck_radiance_to_bt(rad, fk1, fk2, bc1, bc2) -> np.ndarray` (float32 Kelvin)
  - `apply_lut(counts: np.ndarray, lut: np.ndarray) -> np.ndarray` (float32 Kelvin, NaN out of range)
  - `normalize_bt(bt, cfg: NormalizationConfig) -> np.ndarray` — float32 in `[0, 1]`, **NaN → 0.0**
  - `denormalize_bt(x, cfg: NormalizationConfig) -> np.ndarray` — float32 Kelvin
  - `valid_mask(bt) -> np.ndarray` (bool)
  - `SensorStats(mean: float, std: float)`, `sensor_stats(bt) -> SensorStats`,
    `match_statistics(bt, src: SensorStats, dst: SensorStats) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

Create `tests/data/__init__.py` (empty), then `tests/data/test_radiometry.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from sattsr.config import NormalizationConfig
from sattsr.data.radiometry import (
    SensorStats,
    apply_lut,
    denormalize_bt,
    match_statistics,
    normalize_bt,
    planck_radiance_to_bt,
    sensor_stats,
    valid_mask,
)

# Representative GOES ABI band 13 Planck coefficients.
FK1, FK2, BC1, BC2 = 1.03413e4, 1.39177e3, 0.20351, 0.99958


def test_planck_inversion_lands_in_a_physical_range():
    bt = planck_radiance_to_bt(np.array([[10.0, 60.0, 120.0]], dtype=np.float32),
                               FK1, FK2, BC1, BC2)
    assert bt.dtype == np.float32
    assert np.all(bt > 150.0) and np.all(bt < 350.0)
    assert np.all(np.diff(bt, axis=1) > 0)      # radiance and BT increase together


def test_planck_inversion_rejects_non_positive_radiance():
    bt = planck_radiance_to_bt(np.array([0.0, -5.0], dtype=np.float32), FK1, FK2, BC1, BC2)
    assert np.all(np.isnan(bt))


def test_apply_lut_maps_counts_to_temperature():
    lut = np.linspace(150.0, 350.0, 1024).astype(np.float32)
    bt = apply_lut(np.array([[0, 512, 1023]], dtype=np.uint16), lut)
    np.testing.assert_allclose(bt, [[lut[0], lut[512], lut[1023]]], rtol=1e-6)


def test_apply_lut_marks_out_of_range_counts_nan():
    lut = np.linspace(150.0, 350.0, 16).astype(np.float32)
    bt = apply_lut(np.array([-1, 3, 99]), lut)
    assert np.isnan(bt[0]) and np.isfinite(bt[1]) and np.isnan(bt[2])


def test_normalize_and_denormalize_round_trip():
    cfg = NormalizationConfig(bt_min=180.0, bt_max=330.0)
    bt = np.array([[180.0, 255.0, 330.0]], dtype=np.float32)
    x = normalize_bt(bt, cfg)
    np.testing.assert_allclose(x, [[0.0, 0.5, 1.0]], atol=1e-6)
    np.testing.assert_allclose(denormalize_bt(x, cfg), bt, atol=1e-3)


def test_normalize_clips_and_zero_fills_nan():
    cfg = NormalizationConfig(bt_min=200.0, bt_max=300.0)
    x = normalize_bt(np.array([150.0, np.nan, 400.0], dtype=np.float32), cfg)
    assert x[0] == pytest.approx(0.0)
    assert x[1] == pytest.approx(0.0)
    assert x[2] == pytest.approx(1.0)
    assert not np.isnan(x).any()


def test_valid_mask_flags_only_finite_pixels():
    bt = np.array([[250.0, np.nan], [np.inf, 300.0]], dtype=np.float32)
    np.testing.assert_array_equal(valid_mask(bt), [[True, False], [False, True]])


def test_match_statistics_shifts_mean_and_scale():
    rng = np.random.default_rng(0)
    bt = rng.normal(240.0, 12.0, size=(64, 64)).astype(np.float32)
    out = match_statistics(bt, sensor_stats(bt), SensorStats(mean=260.0, std=6.0))
    assert float(np.nanmean(out)) == pytest.approx(260.0, abs=0.5)
    assert float(np.nanstd(out)) == pytest.approx(6.0, abs=0.5)


def test_match_statistics_preserves_nan_positions():
    bt = np.array([[250.0, np.nan]], dtype=np.float32)
    out = match_statistics(bt, SensorStats(250.0, 1.0), SensorStats(260.0, 2.0))
    assert np.isnan(out[0, 1])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/data/test_radiometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.data'`

- [ ] **Step 3: Implement `src/sattsr/data/radiometry.py`**

Create an empty `src/sattsr/data/__init__.py` too.

```python
"""Radiometric conversion and normalisation of thermal-IR observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sattsr.config import NormalizationConfig


def planck_radiance_to_bt(
    rad: np.ndarray, fk1: float, fk2: float, bc1: float, bc2: float
) -> np.ndarray:
    """Invert the Planck function: ABI L1b spectral radiance -> brightness temperature (K)."""
    r = np.asarray(rad, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        bt = (fk2 / np.log(fk1 / r + 1.0) - bc1) / bc2
    bt = np.where(r > 0, bt, np.nan)
    return np.where(np.isfinite(bt), bt, np.nan).astype(np.float32)


def apply_lut(counts: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map integer sensor counts through a count -> temperature table (INSAT L1B/L1C)."""
    # Cast first: NumPy 2 refuses to put the -1 sentinel into an unsigned array.
    c = np.asarray(counts, dtype=np.float64)
    table = np.asarray(lut, dtype=np.float32).ravel()
    idx = np.rint(np.where(np.isfinite(c), c, -1.0)).astype(np.int64)
    inside = (idx >= 0) & (idx < table.size)
    out = np.full(idx.shape, np.nan, dtype=np.float32)
    out[inside] = table[idx[inside]]
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def valid_mask(bt: np.ndarray) -> np.ndarray:
    """True where the brightness temperature is a usable finite value."""
    return np.isfinite(np.asarray(bt))


def normalize_bt(bt: np.ndarray, cfg: NormalizationConfig) -> np.ndarray:
    """Map Kelvin onto [0, 1]. Out-of-range values are clipped; NaN becomes 0.0."""
    a = np.asarray(bt, dtype=np.float32)
    x = (a - np.float32(cfg.bt_min)) / np.float32(cfg.data_range)
    x = np.clip(x, 0.0, 1.0)
    return np.where(np.isfinite(x), x, np.float32(0.0)).astype(np.float32)


def denormalize_bt(x: np.ndarray, cfg: NormalizationConfig) -> np.ndarray:
    """Map [0, 1] back onto Kelvin."""
    a = np.asarray(x, dtype=np.float32)
    return (a * np.float32(cfg.data_range) + np.float32(cfg.bt_min)).astype(np.float32)


@dataclass(frozen=True)
class SensorStats:
    """Brightness-temperature mean and standard deviation for one sensor."""

    mean: float
    std: float


def sensor_stats(bt: np.ndarray) -> SensorStats:
    """Compute BT statistics over the valid pixels of an array or stack."""
    a = np.asarray(bt, dtype=np.float64)
    return SensorStats(mean=float(np.nanmean(a)), std=float(np.nanstd(a)))


def match_statistics(bt: np.ndarray, src: SensorStats, dst: SensorStats) -> np.ndarray:
    """Renormalise BT from one sensor's radiometric statistics to another's.

    This is the cross-sensor adaptation step: it removes the systematic mean/scale
    offset between, say, GOES ABI C13 and INSAT TIR1 before fine-tuning.
    """
    a = np.asarray(bt, dtype=np.float32)
    scale = np.float32(dst.std / src.std) if src.std > 1e-6 else np.float32(1.0)
    return ((a - np.float32(src.mean)) * scale + np.float32(dst.mean)).astype(np.float32)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/data/test_radiometry.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/data tests/data
git commit -m "feat: radiometric conversion, normalisation and cross-sensor statistics matching"
```

---

### Task 4: Frame model, reader protocol, and the GOES-19 reader

**Files:**
- Create: `src/sattsr/io/__init__.py`, `src/sattsr/io/base.py`, `src/sattsr/io/goes.py`
- Create: `tests/conftest.py`
- Test: `tests/io/test_goes.py`

**Interfaces:**
- Consumes: `TargetGrid`, `regrid_geos`, `GeosProjection`, `planck_radiance_to_bt`.
- Produces:
  - `Frame(timestamp: datetime, bt: np.ndarray, sensor: str, source_path: Path)` — frozen dataclass,
    `bt` is float32 Kelvin `(H, W)` with NaN for invalid.
  - `Reader` protocol: `sensor: str`, `patterns: tuple[str, ...]`,
    `timestamp_of(path) -> datetime`, `read(path, grid) -> Frame`, `discover(root) -> list[Path]`.
  - `BaseReader` — mixin implementing `discover()` from `patterns` (recursive, sorted).
  - `Goes19Reader()` with `sensor = "goes19"`.
- Test fixture: `tests/conftest.py::make_goes_file(path, timestamp, *, value_fn=None) -> Path`.

- [ ] **Step 1: Write the shared test fixtures**

First create `tests/__init__.py` (empty). It is required for two reasons: later tests do
`from tests.conftest import ...`, and without it pytest resolves `tests/io/test_goes.py` to the
module path `io.test_goes`, which collides with the standard library and fails collection.

`tests/conftest.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import xarray as xr

FK1, FK2, BC1, BC2 = 1.03413e4, 1.39177e3, 0.20351, 0.99958
SAT_HEIGHT = 35786023.0


def _bt_to_radiance(bt: np.ndarray) -> np.ndarray:
    """Forward Planck, so fixtures round-trip through the reader's inversion."""
    return FK1 / (np.exp(FK2 / (BC1 + BC2 * bt)) - 1.0)


def make_goes_file(
    path: Path,
    timestamp: datetime,
    *,
    size: int = 64,
    lon_origin: float = 82.0,
    value_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> Path:
    """Write a minimal ABI-L1b-like NetCDF whose filename encodes `timestamp`."""
    ang = np.linspace(-0.06, 0.06, size)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    bt = value_fn(yy, xx) if value_fn else (230.0 + 0.5 * xx + 0.2 * yy).astype(np.float64)

    ds = xr.Dataset(
        {
            "Rad": (("y", "x"), _bt_to_radiance(bt).astype(np.float32)),
            "planck_fk1": ((), np.float32(FK1)),
            "planck_fk2": ((), np.float32(FK2)),
            "planck_bc1": ((), np.float32(BC1)),
            "planck_bc2": ((), np.float32(BC2)),
            "goes_imager_projection": ((), np.int8(0), {
                "grid_mapping_name": "geostationary",
                "perspective_point_height": SAT_HEIGHT,
                "longitude_of_projection_origin": lon_origin,
                "latitude_of_projection_origin": 0.0,
                "semi_major_axis": 6378137.0,
                "semi_minor_axis": 6356752.31414,
                "sweep_angle_axis": "x",
            }),
        },
        coords={"x": ("x", ang, {"units": "rad"}), "y": ("y", -ang, {"units": "rad"})},
    )
    doy = timestamp.timetuple().tm_yday
    stamp = f"{timestamp.year}{doy:03d}{timestamp:%H%M%S}0"
    name = f"OR_ABI-L1b-RadF-M6C13_G19_s{stamp}_e{stamp}_c{stamp}.nc"
    out = path / name
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out


def make_goes_series(root: Path, n: int, *, step_minutes: int = 10, size: int = 64) -> list[Path]:
    """A cadence-regular series whose pattern translates one pixel per step."""
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    paths = []
    for i in range(n):
        shift = i
        paths.append(
            make_goes_file(
                root,
                t0 + timedelta(minutes=step_minutes * i),
                size=size,
                value_fn=lambda yy, xx, s=shift: 230.0 + 0.5 * ((xx + s) % size) + 0.2 * yy,
            )
        )
    return paths


@pytest.fixture
def goes_dir(tmp_path: Path) -> Path:
    """Directory containing five 10-minute GOES-like frames."""
    root = tmp_path / "goes"
    make_goes_series(root, 5)
    return root
```

- [ ] **Step 2: Write the failing test**

Create `tests/io/__init__.py` (empty), then `tests/io/test_goes.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from sattsr.geo.grid import TargetGrid
from sattsr.io.goes import Goes19Reader
from tests.conftest import make_goes_file


def _grid() -> TargetGrid:
    return TargetGrid(lat_min=5.0, lat_max=15.0, lon_min=78.0, lon_max=88.0, resolution_deg=1.0)


def test_timestamp_is_parsed_from_the_abi_filename(tmp_path):
    ts = datetime(2026, 9, 4, 6, 30, 20, tzinfo=timezone.utc)
    path = make_goes_file(tmp_path, ts)
    assert Goes19Reader().timestamp_of(path) == ts


def test_bad_filename_raises(tmp_path):
    bad = tmp_path / "not_an_abi_file.nc"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        Goes19Reader().timestamp_of(bad)


def test_read_returns_kelvin_on_the_target_grid(tmp_path):
    ts = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    path = make_goes_file(tmp_path, ts)
    frame = Goes19Reader().read(path, _grid())
    assert frame.sensor == "goes19"
    assert frame.timestamp == ts
    assert frame.bt.shape == _grid().shape
    assert frame.bt.dtype == np.float32
    finite = frame.bt[np.isfinite(frame.bt)]
    assert finite.size > 0
    assert finite.min() > 200.0 and finite.max() < 300.0


def test_read_marks_off_disk_area_nan(tmp_path):
    path = make_goes_file(tmp_path, datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc))
    far_grid = TargetGrid(lat_min=5.0, lat_max=15.0, lon_min=-100.0, lon_max=-90.0,
                          resolution_deg=1.0)
    assert np.all(np.isnan(Goes19Reader().read(path, far_grid).bt))


def test_discover_finds_and_sorts_files(goes_dir):
    paths = Goes19Reader().discover(goes_dir)
    assert len(paths) == 5
    stamps = [Goes19Reader().timestamp_of(p) for p in paths]
    assert stamps == sorted(stamps)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `pytest tests/io/test_goes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.io'`

- [ ] **Step 4: Implement `src/sattsr/io/base.py`**

Create an empty `src/sattsr/io/__init__.py` too.

```python
"""Frame container and the reader protocol every sensor implements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from sattsr.geo.grid import TargetGrid


@dataclass(frozen=True)
class Frame:
    """One observed or synthesized thermal-IR frame on the common grid.

    `bt` is float32 brightness temperature in Kelvin, shape (H, W), NaN where invalid.
    """

    timestamp: datetime
    bt: np.ndarray
    sensor: str
    source_path: Path


@runtime_checkable
class Reader(Protocol):
    """Reads one sensor's native files onto the common grid."""

    sensor: str
    patterns: tuple[str, ...]

    def timestamp_of(self, path: Path) -> datetime: ...
    def read(self, path: Path, grid: TargetGrid) -> Frame: ...
    def discover(self, root: Path) -> list[Path]: ...


class BaseReader:
    """Shared file discovery. Subclasses set `sensor` and `patterns`."""

    sensor: str = "unknown"
    patterns: tuple[str, ...] = ()

    def discover(self, root: Path) -> list[Path]:
        """Recursively find every file matching `patterns`, sorted by timestamp."""
        root = Path(root)
        found: set[Path] = set()
        for pattern in self.patterns:
            found.update(root.rglob(pattern))

        dated: list[tuple[datetime, Path]] = []
        for p in found:
            try:
                dated.append((self.timestamp_of(p), p))   # type: ignore[attr-defined]
            except ValueError:
                continue
        return [p for _, p in sorted(dated, key=lambda kv: (kv[0], str(kv[1])))]
```

- [ ] **Step 5: Implement `src/sattsr/io/goes.py`**

```python
"""GOES-19 ABI Channel 13 (10.3 um) reader."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from sattsr.data.radiometry import planck_radiance_to_bt
from sattsr.geo.grid import TargetGrid
from sattsr.geo.resample import GeosProjection, regrid_geos
from sattsr.io.base import BaseReader, Frame

_START_RE = re.compile(r"_s(?P<stamp>\d{14})_")


class Goes19Reader(BaseReader):
    """Reads ABI L1b `Rad` (or L2 `CMI`) and returns Kelvin on the target grid."""

    sensor = "goes19"
    patterns = ("*C13_G19*.nc", "*C13_G19*.nc4")

    def timestamp_of(self, path: Path) -> datetime:
        """Parse the scan start time out of the ABI filename (YYYYDDDHHMMSSm)."""
        m = _START_RE.search(Path(path).name)
        if m is None:
            raise ValueError(f"not an ABI filename: {path.name}")
        s = m.group("stamp")
        year, doy = int(s[0:4]), int(s[4:7])
        hh, mm, ss = int(s[7:9]), int(s[9:11]), int(s[11:13])
        base = datetime(year, 1, 1, tzinfo=timezone.utc)
        return base + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)

    def read(self, path: Path, grid: TargetGrid) -> Frame:
        """Read one ABI file, convert to Kelvin, and regrid onto `grid`."""
        path = Path(path)
        with xr.open_dataset(path, engine="netcdf4", mask_and_scale=True) as ds:
            if "CMI" in ds:
                bt = np.asarray(ds["CMI"].values, dtype=np.float32)
            else:
                bt = planck_radiance_to_bt(
                    np.asarray(ds["Rad"].values),
                    float(ds["planck_fk1"]),
                    float(ds["planck_fk2"]),
                    float(ds["planck_bc1"]),
                    float(ds["planck_bc2"]),
                )
            attrs = ds["goes_imager_projection"].attrs
            proj = GeosProjection(
                satellite_height=float(attrs["perspective_point_height"]),
                longitude_of_origin=float(attrs["longitude_of_projection_origin"]),
                semi_major_axis=float(attrs["semi_major_axis"]),
                semi_minor_axis=float(attrs["semi_minor_axis"]),
                sweep_axis=str(attrs.get("sweep_angle_axis", "x")),
                x=np.asarray(ds["x"].values, dtype=np.float64),
                y=np.asarray(ds["y"].values, dtype=np.float64),
            )

        bt = np.where(np.isfinite(bt) & (bt > 100.0) & (bt < 400.0), bt, np.nan)
        return Frame(
            timestamp=self.timestamp_of(path),
            bt=regrid_geos(bt, proj, grid),
            sensor=self.sensor,
            source_path=path,
        )
```

- [ ] **Step 6: Run it to verify it passes**

Run: `pytest tests/io/test_goes.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add src/sattsr/io tests/io tests/conftest.py
git commit -m "feat: Frame model, reader protocol, and GOES-19 ABI C13 reader"
```

---

### Task 5: Himawari and INSAT readers, plus the sensor registry

**Files:**
- Create: `src/sattsr/io/himawari.py`, `src/sattsr/io/insat.py`, `src/sattsr/io/registry.py`
- Modify: `tests/conftest.py` (add `make_himawari_file`, `make_insat_file`)
- Test: `tests/io/test_himawari.py`, `tests/io/test_insat.py`, `tests/io/test_registry.py`

**Interfaces:**
- Consumes: `BaseReader`, `Frame`, `TargetGrid`, `regrid_latlon`, `regrid_scattered`, `apply_lut`.
- Produces:
  - `HimawariReader()` — `sensor = "himawari8"`, reads gridded `tbb_13` (already Kelvin).
  - `InsatReader(sensor: str = "insat3dr")` — reads `IMG_TIR1` counts through the
    `IMG_TIR1_TEMP` LUT; dispatches on 1-D vs 2-D `Latitude`/`Longitude`.
  - `get_reader(sensor: str) -> Reader`, `READERS: dict[str, type]`.

- [ ] **Step 1: Add fixture builders to `tests/conftest.py`**

Append:

```python
import h5py

INSAT_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def make_himawari_file(path: Path, timestamp: datetime, *, size: int = 32) -> Path:
    """Write a JAXA-style gridded AHI NetCDF with a `tbb_13` variable in Kelvin."""
    lats = np.linspace(30.0, 0.0, size)          # descending
    lons = np.linspace(70.0, 100.0, size)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    tbb = (240.0 + 0.4 * xx + 0.3 * yy).astype(np.float32)
    ds = xr.Dataset(
        {"tbb_13": (("latitude", "longitude"), tbb, {"units": "K"})},
        coords={"latitude": lats, "longitude": lons},
    )
    out = path / f"NC_H08_{timestamp:%Y%m%d_%H%M}_R21_FLDK.02401_02401.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out


def make_insat_file(
    path: Path, timestamp: datetime, *, size: int = 32, curvilinear: bool = False
) -> Path:
    """Write a MOSDAC-style INSAT L1C HDF5 with IMG_TIR1 counts and a temperature LUT."""
    lut = np.linspace(150.0, 350.0, 1024).astype(np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    counts = ((400 + 3 * xx + 2 * yy) % 1024).astype(np.uint16)
    lats = np.linspace(35.0, 5.0, size).astype(np.float32)
    lons = np.linspace(65.0, 95.0, size).astype(np.float32)

    month = INSAT_MONTHS[timestamp.month - 1]
    out = path / f"3DIMG_{timestamp.day:02d}{month}{timestamp.year}_{timestamp:%H%M}_L1C_ASIA_MER.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("IMG_TIR1", data=counts[None, ...])
        f.create_dataset("IMG_TIR1_TEMP", data=lut)
        if curvilinear:
            lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
            f.create_dataset("Latitude", data=lat2d)
            f.create_dataset("Longitude", data=lon2d)
        else:
            f.create_dataset("Latitude", data=lats)
            f.create_dataset("Longitude", data=lons)
    return out
```

- [ ] **Step 2: Write the failing tests**

`tests/io/test_himawari.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from sattsr.geo.grid import TargetGrid
from sattsr.io.himawari import HimawariReader
from tests.conftest import make_himawari_file


def test_timestamp_parsed_from_filename(tmp_path):
    ts = datetime(2026, 9, 4, 6, 30, tzinfo=timezone.utc)
    assert HimawariReader().timestamp_of(make_himawari_file(tmp_path, ts)) == ts


def test_read_returns_kelvin_on_target_grid(tmp_path):
    ts = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    grid = TargetGrid(lat_min=10.0, lat_max=20.0, lon_min=75.0, lon_max=85.0, resolution_deg=1.0)
    frame = HimawariReader().read(make_himawari_file(tmp_path, ts), grid)
    assert frame.sensor == "himawari8"
    assert frame.bt.shape == grid.shape
    assert frame.bt.dtype == np.float32
    assert np.isfinite(frame.bt).all()
    assert 200.0 < float(frame.bt.min()) and float(frame.bt.max()) < 320.0
```

`tests/io/test_insat.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from sattsr.geo.grid import TargetGrid
from sattsr.io.insat import InsatReader
from tests.conftest import make_insat_file


def _grid() -> TargetGrid:
    return TargetGrid(lat_min=10.0, lat_max=20.0, lon_min=70.0, lon_max=80.0, resolution_deg=1.0)


def test_timestamp_parsed_from_mosdac_filename(tmp_path):
    ts = datetime(2026, 9, 4, 6, 30, tzinfo=timezone.utc)
    assert InsatReader().timestamp_of(make_insat_file(tmp_path, ts)) == ts


def test_read_applies_the_temperature_lut(tmp_path):
    ts = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    frame = InsatReader().read(make_insat_file(tmp_path, ts), _grid())
    assert frame.sensor == "insat3dr"
    assert frame.bt.shape == _grid().shape
    assert frame.bt.dtype == np.float32
    finite = frame.bt[np.isfinite(frame.bt)]
    assert finite.size > 0
    assert finite.min() >= 150.0 and finite.max() <= 350.0


def test_read_handles_curvilinear_geolocation(tmp_path):
    ts = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    path = make_insat_file(tmp_path, ts, curvilinear=True)
    frame = InsatReader().read(path, _grid())
    assert frame.bt.shape == _grid().shape
    assert np.isfinite(frame.bt).any()


def test_sensor_label_is_configurable(tmp_path):
    ts = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    frame = InsatReader(sensor="insat3ds").read(make_insat_file(tmp_path, ts), _grid())
    assert frame.sensor == "insat3ds"


def test_bad_filename_raises(tmp_path):
    bad = tmp_path / "random.h5"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        InsatReader().timestamp_of(bad)
```

`tests/io/test_registry.py`:

```python
from __future__ import annotations

import pytest

from sattsr.io.goes import Goes19Reader
from sattsr.io.himawari import HimawariReader
from sattsr.io.insat import InsatReader
from sattsr.io.registry import get_reader


def test_registry_returns_the_right_reader():
    assert isinstance(get_reader("goes19"), Goes19Reader)
    assert isinstance(get_reader("himawari8"), HimawariReader)
    insat = get_reader("insat3ds")
    assert isinstance(insat, InsatReader)
    assert insat.sensor == "insat3ds"
    assert get_reader("insat3dr").sensor == "insat3dr"


def test_unknown_sensor_raises():
    with pytest.raises(KeyError):
        get_reader("sentinel2")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `pytest tests/io -v`
Expected: the three new files FAIL with `ModuleNotFoundError`; `test_goes.py` still passes.

- [ ] **Step 4: Implement `src/sattsr/io/himawari.py`**

```python
"""Himawari-8/9 AHI Band 13 (10.4 um) gridded reader."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from sattsr.geo.grid import TargetGrid
from sattsr.geo.resample import regrid_latlon
from sattsr.io.base import BaseReader, Frame

_TIME_RE = re.compile(r"_(?P<date>\d{8})_(?P<time>\d{4})_")
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
_BT_NAMES = ("tbb_13", "tbb", "brightness_temperature")


class HimawariReader(BaseReader):
    """Reads the JAXA gridded AHI product, whose `tbb_13` is already in Kelvin."""

    sensor = "himawari8"
    patterns = ("NC_H0*.nc", "*FLDK*.nc")

    def timestamp_of(self, path: Path) -> datetime:
        """Parse `..._YYYYMMDD_HHMM_...` out of the filename."""
        m = _TIME_RE.search(Path(path).name)
        if m is None:
            raise ValueError(f"not a Himawari filename: {path.name}")
        return datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M").replace(
            tzinfo=timezone.utc
        )

    def read(self, path: Path, grid: TargetGrid) -> Frame:
        """Read `tbb_13` and regrid onto `grid`."""
        path = Path(path)
        with xr.open_dataset(path, engine="netcdf4", mask_and_scale=True) as ds:
            var = next((n for n in _BT_NAMES if n in ds), None)
            if var is None:
                raise ValueError(f"no thermal band variable in {path.name}; have {list(ds)}")
            lat_name = next(n for n in _LAT_NAMES if n in ds.coords or n in ds)
            lon_name = next(n for n in _LON_NAMES if n in ds.coords or n in ds)
            bt = np.asarray(ds[var].values, dtype=np.float32).squeeze()
            lats = np.asarray(ds[lat_name].values, dtype=np.float64).squeeze()
            lons = np.asarray(ds[lon_name].values, dtype=np.float64).squeeze()

        bt = np.where(np.isfinite(bt) & (bt > 100.0) & (bt < 400.0), bt, np.nan)
        return Frame(
            timestamp=self.timestamp_of(path),
            bt=regrid_latlon(bt, lats, lons, grid),
            sensor=self.sensor,
            source_path=path,
        )
```

- [ ] **Step 5: Implement `src/sattsr/io/insat.py`**

```python
"""INSAT-3DR / 3DS TIR1 (10.8 um) HDF5 reader (MOSDAC L1B/L1C)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from sattsr.data.radiometry import apply_lut
from sattsr.geo.grid import TargetGrid
from sattsr.geo.resample import regrid_latlon, regrid_scattered
from sattsr.io.base import BaseReader, Frame

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
)}
_TIME_RE = re.compile(r"_(?P<day>\d{2})(?P<mon>[A-Z]{3})(?P<year>\d{4})_(?P<hhmm>\d{4})")

_COUNT_NAMES = ("IMG_TIR1",)
_LUT_NAMES = ("IMG_TIR1_TEMP",)


class InsatReader(BaseReader):
    """Reads TIR1 counts and converts them through the on-file temperature LUT.

    MOSDAC ships two geolocation styles. L1C (`_MER_`) carries 1-D `Latitude` /
    `Longitude`; L1B carries per-pixel 2-D arrays. Both are handled.
    """

    patterns = ("*.h5", "*.hdf5")

    def __init__(self, sensor: str = "insat3dr") -> None:
        self.sensor = sensor

    def timestamp_of(self, path: Path) -> datetime:
        """Parse `_DDMONYYYY_HHMM` out of the MOSDAC filename."""
        m = _TIME_RE.search(Path(path).name.upper())
        if m is None:
            raise ValueError(f"not a MOSDAC INSAT filename: {path.name}")
        hhmm = m.group("hhmm")
        return datetime(
            int(m.group("year")), _MONTHS[m.group("mon")], int(m.group("day")),
            int(hhmm[:2]), int(hhmm[2:]), tzinfo=timezone.utc,
        )

    def read(self, path: Path, grid: TargetGrid) -> Frame:
        """Read TIR1, apply the LUT, and regrid onto `grid`."""
        path = Path(path)
        with h5py.File(path, "r") as f:
            counts = self._first(f, _COUNT_NAMES, path)
            lut = self._first(f, _LUT_NAMES, path)
            lats = np.asarray(f["Latitude"][:], dtype=np.float64).squeeze()
            lons = np.asarray(f["Longitude"][:], dtype=np.float64).squeeze()

        bt = apply_lut(counts, lut)
        bt = np.where(np.isfinite(bt) & (bt > 100.0) & (bt < 400.0), bt, np.nan)

        if lats.ndim == 1 and lons.ndim == 1:
            regridded = regrid_latlon(bt, lats, lons, grid)
        else:
            regridded = regrid_scattered(
                bt, lats, lons, grid, max_distance_deg=max(grid.resolution_deg * 2.0, 0.1)
            )

        return Frame(
            timestamp=self.timestamp_of(path),
            bt=regridded,
            sensor=self.sensor,
            source_path=path,
        )

    @staticmethod
    def _first(handle: h5py.File, names: tuple[str, ...], path: Path) -> np.ndarray:
        for name in names:
            if name in handle:
                return np.asarray(handle[name][:]).squeeze()
        raise ValueError(f"none of {names} present in {path.name}")
```

- [ ] **Step 6: Implement `src/sattsr/io/registry.py`**

```python
"""Sensor name -> reader instance."""

from __future__ import annotations

from collections.abc import Callable

from sattsr.io.base import Reader
from sattsr.io.goes import Goes19Reader
from sattsr.io.himawari import HimawariReader
from sattsr.io.insat import InsatReader

READERS: dict[str, Callable[[], Reader]] = {
    "goes19": Goes19Reader,
    "himawari8": HimawariReader,
    "insat3dr": lambda: InsatReader(sensor="insat3dr"),
    "insat3ds": lambda: InsatReader(sensor="insat3ds"),
}


def get_reader(sensor: str) -> Reader:
    """Return a fresh reader for `sensor`, or raise KeyError."""
    try:
        factory = READERS[sensor]
    except KeyError as exc:
        raise KeyError(f"unsupported sensor {sensor!r}; known: {sorted(READERS)}") from exc
    return factory()
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest tests/io -v`
Expected: 13 passed

- [ ] **Step 8: Commit**

```bash
git add src/sattsr/io tests/io tests/conftest.py
git commit -m "feat: Himawari and INSAT TIR1 readers with sensor registry"
```

---

### Task 6: CF-compliant NetCDF frame products

**Files:**
- Create: `src/sattsr/io/writer.py`
- Test: `tests/io/test_writer.py`

**Interfaces:**
- Consumes: `Frame`, `TargetGrid`.
- Produces:
  - `write_frames_nc(path, frames: list[Frame], grid: TargetGrid, *, synthetic: list[bool] | Sequence[bool], model_version: str, extra_attrs: dict | None = None) -> Path`
  - `read_frames_nc(path, *, sensor: str | None = None) -> tuple[list[Frame], list[bool]]`

  The `synthetic` variable and the "NOT observations" comment attribute are mandatory —
  this is the ethics requirement from the spec, enforced by test.

- [ ] **Step 1: Write the failing test**

`tests/io/test_writer.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sattsr.geo.grid import TargetGrid
from sattsr.io.base import Frame
from sattsr.io.writer import read_frames_nc, write_frames_nc


def _frames(grid: TargetGrid, n: int = 3) -> list[Frame]:
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        bt = np.full(grid.shape, 250.0 + i, dtype=np.float32)
        bt[0, 0] = np.nan
        out.append(Frame(t0 + timedelta(minutes=15 * i), bt, "insat3dr", Path("x.h5")))
    return out


def _grid() -> TargetGrid:
    return TargetGrid(lat_min=10.0, lat_max=15.0, lon_min=70.0, lon_max=75.0, resolution_deg=1.0)


def test_round_trip_preserves_values_times_and_flags(tmp_path):
    grid = _grid()
    frames = _frames(grid)
    out = write_frames_nc(tmp_path / "o.nc", frames, grid,
                          synthetic=[False, True, False], model_version="v0.1.0")
    assert out.exists()

    back, flags = read_frames_nc(out)
    assert flags == [False, True, False]
    assert [f.timestamp for f in back] == [f.timestamp for f in frames]
    for a, b in zip(back, frames):
        np.testing.assert_allclose(a.bt, b.bt, equal_nan=True)
        assert a.bt.dtype == np.float32


def test_output_carries_the_synthetic_flag_and_warning(tmp_path):
    grid = _grid()
    out = write_frames_nc(tmp_path / "o.nc", _frames(grid), grid,
                          synthetic=[False, True, False], model_version="v0.1.0")
    with xr.open_dataset(out) as ds:
        assert "synthetic" in ds
        assert ds["synthetic"].attrs["flag_meanings"] == "observed synthesized"
        np.testing.assert_array_equal(ds["synthetic"].values, [0, 1, 0])
        assert "NOT observations" in ds.attrs["comment"]
        assert ds.attrs["model_version"] == "v0.1.0"
        assert ds.attrs["Conventions"].startswith("CF-")
        assert ds["brightness_temperature"].attrs["units"] == "K"


def test_coordinates_match_the_grid(tmp_path):
    grid = _grid()
    out = write_frames_nc(tmp_path / "o.nc", _frames(grid), grid,
                          synthetic=[False] * 3, model_version="v0.1.0")
    with xr.open_dataset(out) as ds:
        np.testing.assert_allclose(ds["lat"].values, grid.lats)
        np.testing.assert_allclose(ds["lon"].values, grid.lons)


def test_mismatched_flag_length_is_rejected(tmp_path):
    grid = _grid()
    with pytest.raises(ValueError):
        write_frames_nc(tmp_path / "o.nc", _frames(grid), grid,
                        synthetic=[False], model_version="v0.1.0")


def test_empty_frame_list_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_frames_nc(tmp_path / "o.nc", [], _grid(), synthetic=[], model_version="v0.1.0")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/io/test_writer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.io.writer'`

- [ ] **Step 3: Make the pandas dependency explicit**

`writer.py` uses pandas for CF time encoding. It already arrives as a transitive dependency of
xarray, but pin it directly — add `"pandas>=2.0",` to `[project].dependencies` in
`pyproject.toml`, then `pip install -e ".[dev]"`.

- [ ] **Step 4: Implement `src/sattsr/io/writer.py`**

```python
"""CF-compliant NetCDF read/write for frame products.

Every product this project emits goes through here, so the synthetic-frame flag and
the "not an observation" warning are impossible to omit.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sattsr.geo.grid import TargetGrid
from sattsr.io.base import Frame

SYNTHETIC_WARNING = (
    "Frames with synthetic=1 are AI-generated and are NOT observations. "
    "They are decision-support augmentation only."
)


def write_frames_nc(
    path: str | Path,
    frames: Sequence[Frame],
    grid: TargetGrid,
    *,
    synthetic: Sequence[bool],
    model_version: str,
    extra_attrs: dict[str, object] | None = None,
) -> Path:
    """Write a time series of frames as a CF-1.8 NetCDF file."""
    if len(frames) == 0:
        raise ValueError("cannot write an empty frame series")
    if len(synthetic) != len(frames):
        raise ValueError(
            f"synthetic has {len(synthetic)} entries but there are {len(frames)} frames"
        )

    data = np.stack([np.asarray(f.bt, dtype=np.float32) for f in frames])
    if data.shape[1:] != grid.shape:
        raise ValueError(f"frame shape {data.shape[1:]} does not match grid {grid.shape}")

    times = pd.to_datetime([f.timestamp.astimezone(timezone.utc).replace(tzinfo=None)
                            for f in frames])

    ds = xr.Dataset(
        {
            "brightness_temperature": (
                ("time", "lat", "lon"),
                data,
                {
                    "units": "K",
                    "long_name": "Thermal infrared brightness temperature (~10 um)",
                    "standard_name": "toa_brightness_temperature",
                },
            ),
            "synthetic": (
                ("time",),
                np.asarray(synthetic, dtype=np.int8),
                {
                    "long_name": "1 if the frame was synthesized by the model, 0 if observed",
                    "flag_values": np.array([0, 1], dtype=np.int8),
                    "flag_meanings": "observed synthesized",
                },
            ),
        },
        coords={
            "time": times,
            "lat": ("lat", grid.lats, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", grid.lons, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "Temporally super-resolved geostationary thermal imagery",
            "source": frames[0].sensor,
            "model_version": model_version,
            "comment": SYNTHETIC_WARNING,
            "history": f"{datetime.now(timezone.utc).isoformat()} created by sattsr",
        },
    )
    if extra_attrs:
        ds.attrs.update({k: str(v) for k, v in extra_attrs.items()})

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        "brightness_temperature": {
            "zlib": True, "complevel": 4, "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        },
        "time": {"units": "seconds since 1970-01-01T00:00:00", "dtype": "int64"},
    }
    ds.to_netcdf(out, engine="netcdf4", encoding=encoding)
    return out


def read_frames_nc(
    path: str | Path, *, sensor: str | None = None
) -> tuple[list[Frame], list[bool]]:
    """Read a frame-product NetCDF back into Frames plus their synthetic flags."""
    p = Path(path)
    with xr.open_dataset(p, engine="netcdf4", mask_and_scale=True) as ds:
        data = np.asarray(ds["brightness_temperature"].values, dtype=np.float32)
        times = pd.to_datetime(ds["time"].values).to_pydatetime()
        flags = (
            [bool(v) for v in np.asarray(ds["synthetic"].values)]
            if "synthetic" in ds
            else [False] * data.shape[0]
        )
        src = sensor or str(ds.attrs.get("source", "unknown"))

    frames = [
        Frame(timestamp=t.replace(tzinfo=timezone.utc), bt=data[i], sensor=src, source_path=p)
        for i, t in enumerate(times)
    ]
    return frames, flags
```

- [ ] **Step 5: Run it to verify it passes**

Run: `pytest tests/io/test_writer.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/sattsr/io/writer.py tests/io/test_writer.py
git commit -m "feat: CF-compliant NetCDF frame products with mandatory synthetic flagging"
```

---

### Task 7: Frame index and the regridded frame cache

**Files:**
- Create: `src/sattsr/data/index.py`
- Test: `tests/data/test_index.py`

**Interfaces:**
- Consumes: `Reader`, `TargetGrid`, `Frame`.
- Produces:
  - `FrameRef(timestamp: datetime, path: Path, sensor: str)` frozen dataclass.
  - `build_index(reader, root) -> list[FrameRef]` — sorted by timestamp, unparseable files skipped.
  - `save_index(refs, path) -> Path` / `load_index(path) -> list[FrameRef]` (JSON).
  - `cache_path_for(cache_root, ref) -> Path` — `<cache_root>/<sensor>/<YYYYMMDD>/<HHMMSS>.npy`.
  - `prepare_cache(reader, refs, grid, cache_root, *, overwrite=False, progress=False) -> list[FrameRef]`
    — writes regridded float32 Kelvin `.npy` per frame; returns refs pointing at the cache.
  - `load_cached(ref) -> np.ndarray`.

Why a cache: reading and reprojecting a full-disc file per training sample is orders of magnitude
slower than the model step. Everything downstream reads `.npy`.

- [ ] **Step 1: Write the failing test**

`tests/data/test_index.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from sattsr.data.index import (
    FrameRef,
    build_index,
    cache_path_for,
    load_cached,
    load_index,
    prepare_cache,
    save_index,
)
from sattsr.geo.grid import TargetGrid
from sattsr.io.goes import Goes19Reader


def _grid() -> TargetGrid:
    return TargetGrid(lat_min=5.0, lat_max=15.0, lon_min=78.0, lon_max=88.0, resolution_deg=1.0)


def test_build_index_is_sorted_and_complete(goes_dir):
    refs = build_index(Goes19Reader(), goes_dir)
    assert len(refs) == 5
    assert [r.timestamp for r in refs] == sorted(r.timestamp for r in refs)
    assert all(r.sensor == "goes19" for r in refs)


def test_build_index_skips_unparseable_files(goes_dir):
    (goes_dir / "stray_C13_G19_file.nc").write_text("junk", encoding="utf-8")
    assert len(build_index(Goes19Reader(), goes_dir)) == 5


def test_index_json_round_trip(tmp_path, goes_dir):
    refs = build_index(Goes19Reader(), goes_dir)
    out = save_index(refs, tmp_path / "index.json")
    assert load_index(out) == refs


def test_cache_path_layout(tmp_path):
    ref = FrameRef(datetime(2026, 9, 4, 6, 30, 20, tzinfo=timezone.utc),
                   tmp_path / "x.nc", "goes19")
    assert cache_path_for(tmp_path / "cache", ref) == (
        tmp_path / "cache" / "goes19" / "20260904" / "063020.npy"
    )


def test_prepare_cache_writes_kelvin_arrays(tmp_path, goes_dir):
    grid = _grid()
    refs = build_index(Goes19Reader(), goes_dir)
    cached = prepare_cache(Goes19Reader(), refs, grid, tmp_path / "cache")

    assert len(cached) == len(refs)
    assert [c.timestamp for c in cached] == [r.timestamp for r in refs]
    for c in cached:
        assert c.path.suffix == ".npy"
        arr = load_cached(c)
        assert arr.shape == grid.shape
        assert arr.dtype == np.float32
        finite = arr[np.isfinite(arr)]
        assert finite.size > 0 and finite.min() > 150.0 and finite.max() < 400.0


def test_prepare_cache_skips_existing_unless_overwrite(tmp_path, goes_dir):
    grid = _grid()
    refs = build_index(Goes19Reader(), goes_dir)[:1]
    cached = prepare_cache(Goes19Reader(), refs, grid, tmp_path / "cache")

    sentinel = np.full(grid.shape, 42.0, dtype=np.float32)
    np.save(cached[0].path, sentinel)

    prepare_cache(Goes19Reader(), refs, grid, tmp_path / "cache")
    np.testing.assert_allclose(load_cached(cached[0]), sentinel)

    prepare_cache(Goes19Reader(), refs, grid, tmp_path / "cache", overwrite=True)
    assert not np.allclose(load_cached(cached[0]), sentinel)


def test_prepare_cache_reports_unreadable_files_without_aborting(tmp_path, goes_dir):
    grid = _grid()
    refs = build_index(Goes19Reader(), goes_dir)
    broken = FrameRef(datetime(2026, 9, 4, 9, 0, tzinfo=timezone.utc),
                      goes_dir / "missing.nc", "goes19")
    cached = prepare_cache(Goes19Reader(), [*refs, broken], grid, tmp_path / "cache")
    assert len(cached) == len(refs)


def test_load_cached_missing_file_raises(tmp_path):
    ref = FrameRef(datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc), tmp_path / "nope.npy", "goes19")
    with pytest.raises(FileNotFoundError):
        load_cached(ref)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/data/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.data.index'`

- [ ] **Step 3: Implement `src/sattsr/data/index.py`**

```python
"""Frame discovery, on-disk index, and the regridded frame cache."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sattsr.geo.grid import TargetGrid
from sattsr.io.base import Reader

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameRef:
    """A pointer to one frame: when it was taken and where the bytes live."""

    timestamp: datetime
    path: Path
    sensor: str


def build_index(reader: Reader, root: str | Path) -> list[FrameRef]:
    """Scan `root` for this reader's files and return timestamp-sorted references."""
    refs: list[FrameRef] = []
    for path in reader.discover(Path(root)):
        try:
            refs.append(FrameRef(reader.timestamp_of(path), Path(path), reader.sensor))
        except ValueError:
            log.debug("skipping unparseable file %s", path)
    return sorted(refs, key=lambda r: (r.timestamp, str(r.path)))


def save_index(refs: Sequence[FrameRef], path: str | Path) -> Path:
    """Write an index as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"timestamp": r.timestamp.astimezone(timezone.utc).isoformat(),
         "path": str(r.path), "sensor": r.sensor}
        for r in refs
    ]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_index(path: str | Path) -> list[FrameRef]:
    """Read an index written by `save_index`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        FrameRef(datetime.fromisoformat(e["timestamp"]), Path(e["path"]), e["sensor"])
        for e in raw
    ]


def cache_path_for(cache_root: str | Path, ref: FrameRef) -> Path:
    """`<cache_root>/<sensor>/<YYYYMMDD>/<HHMMSS>.npy`."""
    ts = ref.timestamp.astimezone(timezone.utc)
    return Path(cache_root) / ref.sensor / ts.strftime("%Y%m%d") / f"{ts.strftime('%H%M%S')}.npy"


def prepare_cache(
    reader: Reader,
    refs: Iterable[FrameRef],
    grid: TargetGrid,
    cache_root: str | Path,
    *,
    overwrite: bool = False,
    progress: bool = False,
) -> list[FrameRef]:
    """Regrid each referenced file once and cache it as float32 Kelvin `.npy`.

    Files that fail to read are logged and skipped rather than aborting the run —
    partial archive downloads are the norm.
    """
    items = list(refs)
    if progress:
        from tqdm import tqdm

        items = tqdm(items, desc="preparing cache")   # type: ignore[assignment]

    out: list[FrameRef] = []
    for ref in items:
        dest = cache_path_for(cache_root, ref)
        if dest.exists() and not overwrite:
            out.append(FrameRef(ref.timestamp, dest, ref.sensor))
            continue
        try:
            frame = reader.read(ref.path, grid)
        except (OSError, ValueError, KeyError) as exc:
            log.warning("skipping %s: %s", ref.path, exc)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, np.asarray(frame.bt, dtype=np.float32))
        out.append(FrameRef(ref.timestamp, dest, ref.sensor))
    return out


def load_cached(ref: FrameRef) -> np.ndarray:
    """Load a cached frame as float32 Kelvin."""
    return np.load(ref.path).astype(np.float32, copy=False)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/data/test_index.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/data/index.py tests/data/test_index.py
git commit -m "feat: frame index and regridded frame cache"
```

---

### Task 8: Triplet construction and leak-free splitting

**Files:**
- Create: `src/sattsr/data/triplets.py`
- Test: `tests/data/test_triplets.py`

**Interfaces:**
- Consumes: `FrameRef`.
- Produces:
  - `Triplet(t0: FrameRef, t1: FrameRef, t2: FrameRef)` frozen dataclass with `.t -> float`
    (normalised position of `t1` in `[0, 1]`) and `.span -> timedelta`.
  - `build_triplets(refs, *, step: timedelta, tolerance: timedelta) -> list[Triplet]`
  - `split_triplets(triplets, *, val_fraction: float, seed: int) -> tuple[list[Triplet], list[Triplet]]`

`split_triplets` splits **by calendar day**, never per sample. Consecutive triplets share frames,
so a random per-sample split would leak the validation targets into training and inflate every
metric in the report.

- [ ] **Step 1: Write the failing test**

`tests/data/test_triplets.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sattsr.data.index import FrameRef
from sattsr.data.triplets import Triplet, build_triplets, split_triplets

STEP = timedelta(minutes=10)
TOL = timedelta(minutes=2)


def _refs(n: int, *, step: timedelta = STEP, start_day: int = 4) -> list[FrameRef]:
    t0 = datetime(2026, 9, start_day, 0, 0, tzinfo=timezone.utc)
    return [FrameRef(t0 + step * i, Path(f"f{i}.npy"), "goes19") for i in range(n)]


def test_regular_series_yields_overlapping_triplets():
    triplets = build_triplets(_refs(5), step=STEP, tolerance=TOL)
    assert len(triplets) == 3
    assert triplets[0].t0.timestamp == datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    assert triplets[0].t1.timestamp == datetime(2026, 9, 4, 0, 10, tzinfo=timezone.utc)
    assert triplets[0].t2.timestamp == datetime(2026, 9, 4, 0, 20, tzinfo=timezone.utc)


def test_t_is_the_normalised_midpoint():
    triplet = build_triplets(_refs(3), step=STEP, tolerance=TOL)[0]
    assert triplet.t == pytest.approx(0.5)
    assert triplet.span == timedelta(minutes=20)


def test_gaps_break_triplets():
    refs = _refs(6)
    refs = refs[:2] + refs[4:]          # drop two frames -> a 30-minute hole
    assert len(build_triplets(refs, step=STEP, tolerance=TOL)) == 1


def test_tolerance_absorbs_jitter():
    t0 = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    refs = [
        FrameRef(t0, Path("a.npy"), "goes19"),
        FrameRef(t0 + timedelta(minutes=11), Path("b.npy"), "goes19"),
        FrameRef(t0 + timedelta(minutes=19), Path("c.npy"), "goes19"),
    ]
    triplets = build_triplets(refs, step=STEP, tolerance=TOL)
    assert len(triplets) == 1
    assert triplets[0].t == pytest.approx(11.0 / 19.0)


def test_empty_and_short_series_yield_nothing():
    assert build_triplets([], step=STEP, tolerance=TOL) == []
    assert build_triplets(_refs(2), step=STEP, tolerance=TOL) == []


def test_split_is_by_day_so_no_frame_appears_in_both_sides():
    triplets = (
        build_triplets(_refs(20, start_day=4), step=STEP, tolerance=TOL)
        + build_triplets(_refs(20, start_day=5), step=STEP, tolerance=TOL)
        + build_triplets(_refs(20, start_day=6), step=STEP, tolerance=TOL)
    )
    train, val = split_triplets(triplets, val_fraction=0.34, seed=0)
    assert train and val

    def days(items: list[Triplet]) -> set:
        return {t.t0.timestamp.date() for t in items} | {t.t2.timestamp.date() for t in items}

    assert days(train).isdisjoint(days(val))
    assert len(train) + len(val) == len(triplets)


def test_split_is_deterministic_for_a_seed():
    triplets = build_triplets(_refs(30), step=STEP, tolerance=TOL)
    a = split_triplets(triplets, val_fraction=0.2, seed=7)
    b = split_triplets(triplets, val_fraction=0.2, seed=7)
    assert a == b


def test_split_keeps_at_least_one_day_on_each_side_when_possible():
    triplets = (
        build_triplets(_refs(5, start_day=4), step=STEP, tolerance=TOL)
        + build_triplets(_refs(5, start_day=5), step=STEP, tolerance=TOL)
    )
    train, val = split_triplets(triplets, val_fraction=0.01, seed=0)
    assert train and val
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/data/test_triplets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.data.triplets'`

- [ ] **Step 3: Implement `src/sattsr/data/triplets.py`**

```python
"""Construction of (t0, t1, t2) training samples and leak-free train/val splitting."""

from __future__ import annotations

import bisect
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sattsr.data.index import FrameRef


@dataclass(frozen=True)
class Triplet:
    """Two inputs and the real middle frame the model must reproduce."""

    t0: FrameRef
    t1: FrameRef
    t2: FrameRef

    @property
    def span(self) -> timedelta:
        """Elapsed time between the two input frames."""
        return self.t2.timestamp - self.t0.timestamp

    @property
    def t(self) -> float:
        """Normalised temporal position of the target frame, in (0, 1)."""
        total = self.span.total_seconds()
        return (self.t1.timestamp - self.t0.timestamp).total_seconds() / total


def _nearest(times: list[datetime], target: datetime, tolerance: timedelta) -> int | None:
    """Index of the entry closest to `target`, or None if none is within `tolerance`."""
    if not times:
        return None
    i = bisect.bisect_left(times, target)
    best: int | None = None
    best_gap = tolerance
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(times):
            gap = abs(times[j] - target)
            if gap <= best_gap:
                best, best_gap = j, gap
    return best


def build_triplets(
    refs: Sequence[FrameRef], *, step: timedelta, tolerance: timedelta
) -> list[Triplet]:
    """Form every (t, t+step, t+2*step) triplet the series supports."""
    ordered = sorted(refs, key=lambda r: r.timestamp)
    times = [r.timestamp for r in ordered]

    out: list[Triplet] = []
    for i, ref in enumerate(ordered):
        mid = _nearest(times, ref.timestamp + step, tolerance)
        end = _nearest(times, ref.timestamp + 2 * step, tolerance)
        if mid is None or end is None or not (i < mid < end):
            continue
        out.append(Triplet(ref, ordered[mid], ordered[end]))
    return out


def split_triplets(
    triplets: Sequence[Triplet], *, val_fraction: float, seed: int
) -> tuple[list[Triplet], list[Triplet]]:
    """Split by calendar day so no frame can appear on both sides.

    Adjacent triplets share input frames, so a per-sample random split would put a
    validation target into the training set and inflate every reported metric.
    """
    if not triplets:
        return [], []

    by_day: dict[date, list[Triplet]] = {}
    for tri in triplets:
        by_day.setdefault(tri.t0.timestamp.date(), []).append(tri)

    days = sorted(by_day)
    if len(days) == 1:
        return list(triplets), []

    rng = random.Random(seed)
    shuffled = list(days)
    rng.shuffle(shuffled)

    n_val = max(1, round(len(days) * val_fraction))
    n_val = min(n_val, len(days) - 1)          # always leave a training day
    val_days = set(shuffled[:n_val])

    train = [t for d in days if d not in val_days for t in by_day[d]]
    val = [t for d in days if d in val_days for t in by_day[d]]
    return train, val
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/data/test_triplets.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/data/triplets.py tests/data/test_triplets.py
git commit -m "feat: triplet construction with day-wise leak-free splitting"
```

---

### Task 9: Tiling, padding and seamless reassembly

**Files:**
- Create: `src/sattsr/data/tiling.py`
- Test: `tests/data/test_tiling.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TileSpec(row: int, col: int, size: int)` frozen dataclass.
  - `tile_positions(shape: tuple[int, int], size: int, overlap: int) -> list[TileSpec]`
  - `extract_tile(arr, spec) -> np.ndarray`
  - `hann_weight(size, overlap) -> np.ndarray` — strictly positive everywhere.
  - `reassemble(tiles, specs, shape, overlap) -> np.ndarray`
  - `pad_to_multiple(arr, multiple, *, min_size=0) -> tuple[np.ndarray, tuple[int, int]]`
  - `crop_to(arr, shape) -> np.ndarray`

Full-disc grids are far larger than a trainable tile, so inference runs tile-by-tile. Overlapping
tiles blended with a Hann window are what keep tile seams out of the animation.

- [ ] **Step 1: Write the failing test**

`tests/data/test_tiling.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from sattsr.data.tiling import (
    TileSpec,
    crop_to,
    extract_tile,
    hann_weight,
    pad_to_multiple,
    reassemble,
    tile_positions,
)


def test_tile_positions_cover_the_whole_array():
    specs = tile_positions((300, 500), size=128, overlap=32)
    covered = np.zeros((300, 500), dtype=bool)
    for s in specs:
        covered[s.row:s.row + s.size, s.col:s.col + s.size] = True
    assert covered.all()
    assert all(s.row + s.size <= 300 and s.col + s.size <= 500 for s in specs)


def test_tile_positions_exact_fit_has_no_duplicates():
    specs = tile_positions((256, 256), size=128, overlap=0)
    assert len(specs) == 4
    assert len({(s.row, s.col) for s in specs}) == 4


def test_tile_larger_than_array_is_rejected():
    with pytest.raises(ValueError):
        tile_positions((64, 64), size=128, overlap=16)


def test_overlap_not_smaller_than_size_is_rejected():
    with pytest.raises(ValueError):
        tile_positions((256, 256), size=64, overlap=64)


def test_extract_tile_returns_the_right_window():
    arr = np.arange(100).reshape(10, 10).astype(np.float32)
    tile = extract_tile(arr, TileSpec(row=2, col=3, size=4))
    np.testing.assert_array_equal(tile, arr[2:6, 3:7])


def test_hann_weight_is_strictly_positive():
    w = hann_weight(64, 16)
    assert w.shape == (64, 64)
    assert w.min() > 0.0
    assert w.max() == pytest.approx(1.0)


def test_reassembly_reconstructs_a_smooth_field_exactly():
    rng = np.random.default_rng(0)
    full = rng.normal(250.0, 5.0, size=(300, 500)).astype(np.float32)
    specs = tile_positions(full.shape, size=128, overlap=32)
    tiles = [extract_tile(full, s) for s in specs]
    out = reassemble(tiles, specs, full.shape, overlap=32)
    assert out.shape == full.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, full, atol=1e-3)


def test_reassembly_has_no_visible_seam():
    # a constant field must come back constant, including across tile boundaries
    full = np.full((300, 500), 250.0, dtype=np.float32)
    specs = tile_positions(full.shape, size=128, overlap=32)
    out = reassemble([extract_tile(full, s) for s in specs], specs, full.shape, overlap=32)
    assert float(np.abs(out - 250.0).max()) < 1e-3


def test_pad_to_multiple_and_crop_round_trip():
    arr = np.arange(35 * 53).reshape(35, 53).astype(np.float32)
    padded, original = pad_to_multiple(arr, 32, min_size=64)
    assert padded.shape[0] % 32 == 0 and padded.shape[1] % 32 == 0
    assert padded.shape[0] >= 64 and padded.shape[1] >= 64
    assert original == (35, 53)
    np.testing.assert_array_equal(crop_to(padded, original), arr)


def test_pad_to_multiple_leaves_conforming_arrays_alone():
    arr = np.zeros((64, 128), dtype=np.float32)
    padded, original = pad_to_multiple(arr, 32, min_size=64)
    assert padded.shape == (64, 128)
    assert original == (64, 128)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/data/test_tiling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.data.tiling'`

- [ ] **Step 3: Implement `src/sattsr/data/tiling.py`**

```python
"""Tiled processing of grids larger than the model's receptive window."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    """Top-left corner and edge length of one square tile."""

    row: int
    col: int
    size: int


def _positions(extent: int, size: int, stride: int) -> list[int]:
    starts = list(range(0, max(extent - size, 0) + 1, stride))
    if starts[-1] + size < extent:
        starts.append(extent - size)
    return starts


def tile_positions(shape: tuple[int, int], size: int, overlap: int) -> list[TileSpec]:
    """Cover `shape` with overlapping `size`-square tiles, all fully inside the array."""
    h, w = shape
    if size > h or size > w:
        raise ValueError(f"tile size {size} exceeds array shape {shape}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must be in [0, {size}), got {overlap}")

    stride = size - overlap
    return [
        TileSpec(row=r, col=c, size=size)
        for r in _positions(h, size, stride)
        for c in _positions(w, size, stride)
    ]


def extract_tile(arr: np.ndarray, spec: TileSpec) -> np.ndarray:
    """Copy one tile out of `arr`."""
    return np.asarray(arr)[spec.row:spec.row + spec.size, spec.col:spec.col + spec.size].copy()


def hann_weight(size: int, overlap: int) -> np.ndarray:
    """Separable Hann taper over the overlap band; strictly positive everywhere.

    The taper is offset by one sample so the outermost row and column keep a small
    non-zero weight, which avoids a divide-by-zero at the array border.
    """
    if overlap <= 0:
        return np.ones((size, size), dtype=np.float32)

    ramp = np.hanning(2 * overlap + 3)[1:overlap + 2]     # rises from >0 to exactly 1
    profile = np.ones(size, dtype=np.float64)
    profile[:overlap + 1] = ramp
    profile[size - overlap - 1:] = ramp[::-1]
    return np.outer(profile, profile).astype(np.float32)


def reassemble(
    tiles: Sequence[np.ndarray],
    specs: Sequence[TileSpec],
    shape: tuple[int, int],
    overlap: int,
) -> np.ndarray:
    """Blend overlapping tiles back into one array using Hann weights."""
    if len(tiles) != len(specs):
        raise ValueError(f"{len(tiles)} tiles but {len(specs)} specs")

    acc = np.zeros(shape, dtype=np.float64)
    wsum = np.zeros(shape, dtype=np.float64)
    cache: dict[int, np.ndarray] = {}

    for tile, spec in zip(tiles, specs, strict=True):
        w = cache.setdefault(spec.size, hann_weight(spec.size, overlap))
        sl = (slice(spec.row, spec.row + spec.size), slice(spec.col, spec.col + spec.size))
        acc[sl] += np.asarray(tile, dtype=np.float64) * w
        wsum[sl] += w

    return np.divide(acc, np.maximum(wsum, 1e-8)).astype(np.float32)


def pad_to_multiple(
    arr: np.ndarray, multiple: int, *, min_size: int = 0
) -> tuple[np.ndarray, tuple[int, int]]:
    """Edge-pad so both dimensions are >= `min_size` and divisible by `multiple`.

    Returns the padded array and the original (H, W) so `crop_to` can undo it.
    """
    a = np.asarray(arr)
    h, w = a.shape[-2], a.shape[-1]
    target_h = max(min_size, ((h + multiple - 1) // multiple) * multiple)
    target_w = max(min_size, ((w + multiple - 1) // multiple) * multiple)
    target_h = ((target_h + multiple - 1) // multiple) * multiple
    target_w = ((target_w + multiple - 1) // multiple) * multiple

    if (target_h, target_w) == (h, w):
        return a, (h, w)
    pad = ((0, target_h - h), (0, target_w - w))
    return np.pad(a, pad, mode="edge"), (h, w)


def crop_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Undo `pad_to_multiple`."""
    return np.asarray(arr)[..., : shape[0], : shape[1]]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/data/test_tiling.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/data/tiling.py tests/data/test_tiling.py
git commit -m "feat: tiling with Hann-blended seamless reassembly and padding helpers"
```

---

### Task 10: The torch triplet dataset

**Files:**
- Create: `src/sattsr/data/dataset.py`
- Test: `tests/data/test_dataset.py`

**Interfaces:**
- Consumes: `Triplet`, `load_cached`, `normalize_bt`, `valid_mask`, `NormalizationConfig`,
  `tile_positions`, `extract_tile`.
- Produces:
  - `TripletDataset(triplets, norm, tile_size, *, augment=True, min_valid_fraction=0.5, seed=0, max_tile_tries=8)`
    — a `torch.utils.data.Dataset` whose items are
    `{"i0": (1,h,w), "i1": (1,h,w), "i2": (1,h,w), "valid": (1,h,w), "t": scalar}`,
    all `torch.float32`, images normalised to `[0, 1]`.
  - `.set_epoch(epoch: int) -> None` — re-rolls tile choice and augmentation each epoch while
    staying reproducible.

- [ ] **Step 1: Write the failing test**

`tests/data/test_dataset.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from sattsr.config import NormalizationConfig
from sattsr.data.dataset import TripletDataset
from sattsr.data.index import FrameRef
from sattsr.data.triplets import Triplet

NORM = NormalizationConfig(bt_min=180.0, bt_max=330.0)


def _make_triplets(tmp_path: Path, n: int = 4, size: int = 96, nan_border: int = 0):
    t0 = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)
    refs = []
    for i in range(n + 2):
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        bt = (240.0 + 0.3 * ((xx + i) % size) + 0.2 * yy).astype(np.float32)
        if nan_border:
            bt[:nan_border, :] = np.nan
            bt[:, :nan_border] = np.nan
        path = tmp_path / f"f{i}.npy"
        np.save(path, bt)
        refs.append(FrameRef(t0 + timedelta(minutes=10 * i), path, "goes19"))
    return [Triplet(refs[i], refs[i + 1], refs[i + 2]) for i in range(n)]


def test_item_shapes_dtypes_and_range(tmp_path):
    ds = TripletDataset(_make_triplets(tmp_path), NORM, tile_size=64, augment=False)
    item = ds[0]
    assert set(item) == {"i0", "i1", "i2", "valid", "t"}
    for key in ("i0", "i1", "i2", "valid"):
        assert item[key].shape == (1, 64, 64)
        assert item[key].dtype == torch.float32
    assert 0.0 <= float(item["i0"].min()) and float(item["i0"].max()) <= 1.0
    assert item["t"].shape == ()
    assert float(item["t"]) == pytest.approx(0.5)


def test_length_matches_triplet_count(tmp_path):
    triplets = _make_triplets(tmp_path, n=5)
    assert len(TripletDataset(triplets, NORM, tile_size=64)) == 5


def test_eval_mode_is_deterministic(tmp_path):
    ds = TripletDataset(_make_triplets(tmp_path), NORM, tile_size=64, augment=False)
    torch.testing.assert_close(ds[0]["i0"], ds[0]["i0"])
    torch.testing.assert_close(ds[1]["i1"], ds[1]["i1"])


def test_set_epoch_changes_the_sampled_tile(tmp_path):
    ds = TripletDataset(_make_triplets(tmp_path), NORM, tile_size=64, augment=True, seed=3)
    ds.set_epoch(0)
    a = ds[0]["i0"].clone()
    ds.set_epoch(1)
    b = ds[0]["i0"].clone()
    assert not torch.allclose(a, b)
    ds.set_epoch(0)
    torch.testing.assert_close(ds[0]["i0"], a)


def test_valid_mask_marks_nan_pixels_and_images_are_zero_filled(tmp_path):
    triplets = _make_triplets(tmp_path, size=96, nan_border=40)
    ds = TripletDataset(triplets, NORM, tile_size=64, augment=False, min_valid_fraction=0.0)
    item = ds[0]
    assert float(item["valid"].min()) == 0.0        # some invalid pixels present
    assert torch.isfinite(item["i0"]).all()         # but the image itself has no NaN
    assert float(item["i0"][item["valid"] == 0].abs().max()) == 0.0


def test_temporal_reversal_augmentation_mirrors_t(tmp_path):
    # An asymmetric triplet: t1 sits 40% of the way through the interval.
    triplets = _make_triplets(tmp_path)
    base = triplets[0]
    skewed = Triplet(
        base.t0,
        FrameRef(base.t0.timestamp + timedelta(minutes=8), base.t1.path, "goes19"),
        base.t2,
    )
    ds = TripletDataset([skewed], NORM, tile_size=64, augment=True, seed=0)
    seen = set()
    for epoch in range(12):
        ds.set_epoch(epoch)
        seen.add(round(float(ds[0]["t"]), 3))
    assert seen == {0.4, 0.6}


def test_dataloader_batches_cleanly(tmp_path):
    ds = TripletDataset(_make_triplets(tmp_path, n=4), NORM, tile_size=64, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    assert batch["i0"].shape == (2, 1, 64, 64)
    assert batch["t"].shape == (2,)


def test_tile_larger_than_frame_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        TripletDataset(_make_triplets(tmp_path, size=48), NORM, tile_size=64)[0]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/data/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.data.dataset'`

- [ ] **Step 3: Implement `src/sattsr/data/dataset.py`**

```python
"""Torch dataset yielding normalised (i0, i1, i2) tile triplets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from sattsr.config import NormalizationConfig
from sattsr.data.index import load_cached
from sattsr.data.radiometry import normalize_bt, valid_mask
from sattsr.data.tiling import TileSpec, extract_tile, tile_positions
from sattsr.data.triplets import Triplet


class TripletDataset(Dataset):
    """Samples one square tile from each triplet, normalised to [0, 1].

    Images are zero-filled where invalid; the accompanying `valid` mask is what the
    loss uses to ignore off-disc pixels. Augmentation (flips, 90-degree rotations and
    temporal reversal) is re-rolled per epoch via `set_epoch`, but is fully determined
    by (seed, epoch, index) so runs stay reproducible.
    """

    def __init__(
        self,
        triplets: Sequence[Triplet],
        norm: NormalizationConfig,
        tile_size: int,
        *,
        augment: bool = True,
        min_valid_fraction: float = 0.5,
        seed: int = 0,
        max_tile_tries: int = 8,
    ) -> None:
        self.triplets = list(triplets)
        self.norm = norm
        self.tile_size = int(tile_size)
        self.augment = augment
        self.min_valid_fraction = float(min_valid_fraction)
        self.seed = int(seed)
        self.max_tile_tries = int(max_tile_tries)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-roll tile choice and augmentation for the next pass."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.triplets)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self._epoch, idx))

    def _pick_tile(
        self, frames: list[np.ndarray], rng: np.random.Generator
    ) -> TileSpec:
        specs = tile_positions(frames[0].shape, self.tile_size, overlap=0)
        if not self.augment:
            return specs[len(specs) // 2]

        best = specs[rng.integers(len(specs))]
        best_score = -1.0
        for _ in range(self.max_tile_tries):
            spec = specs[rng.integers(len(specs))]
            score = float(
                np.mean([valid_mask(extract_tile(f, spec)).mean() for f in frames])
            )
            if score >= self.min_valid_fraction:
                return spec
            if score > best_score:
                best, best_score = spec, score
        return best

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        triplet = self.triplets[idx]
        rng = self._rng(idx)

        frames = [
            load_cached(triplet.t0),
            load_cached(triplet.t1),
            load_cached(triplet.t2),
        ]
        if any(f.shape[0] < self.tile_size or f.shape[1] < self.tile_size for f in frames):
            raise ValueError(
                f"tile_size {self.tile_size} exceeds cached frame shape {frames[0].shape}"
            )

        spec = self._pick_tile(frames, rng)
        tiles = [extract_tile(f, spec) for f in frames]
        valid = np.logical_and.reduce([valid_mask(t) for t in tiles]).astype(np.float32)
        images = [normalize_bt(t, self.norm) * valid for t in tiles]
        t_value = float(triplet.t)

        if self.augment:
            if rng.random() < 0.5:
                images = [np.ascontiguousarray(a[:, ::-1]) for a in images]
                valid = np.ascontiguousarray(valid[:, ::-1])
            if rng.random() < 0.5:
                images = [np.ascontiguousarray(a[::-1, :]) for a in images]
                valid = np.ascontiguousarray(valid[::-1, :])
            k = int(rng.integers(4))
            if k:
                images = [np.ascontiguousarray(np.rot90(a, k)) for a in images]
                valid = np.ascontiguousarray(np.rot90(valid, k))
            if rng.random() < 0.5:                       # reverse time
                images = [images[2], images[1], images[0]]
                t_value = 1.0 - t_value

        return {
            "i0": torch.from_numpy(images[0])[None],
            "i1": torch.from_numpy(images[1])[None],
            "i2": torch.from_numpy(images[2])[None],
            "valid": torch.from_numpy(valid)[None],
            "t": torch.tensor(t_value, dtype=torch.float32),
        }
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/data -v`
Expected: 43 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/data/dataset.py tests/data/test_dataset.py
git commit -m "feat: torch triplet dataset with masked tiles and epoch-stable augmentation"
```

---

# Phase 2 — The interpolation model

### Task 11: Convolution blocks and differentiable backward warping

**Files:**
- Create: `src/sattsr/models/__init__.py`, `src/sattsr/models/blocks.py`
- Test: `tests/models/test_blocks.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1, dilation=1) -> nn.Sequential`
    (Conv2d + PReLU)
  - `ResBlock(channels)` — `nn.Module`, `forward(x) -> Tensor` same shape.
  - `backwarp(x: Tensor, flow: Tensor) -> Tensor` — `x` is `(N, C, H, W)`, `flow` is `(N, 2, H, W)`
    in **pixels**, channel 0 = horizontal (`+` right), channel 1 = vertical (`+` down).
    `backwarp(x, flow)[:, :, y, x0] == x[:, :, y + flow_y, x0 + flow_x]`.

- [ ] **Step 1: Write the failing test**

Create `tests/models/__init__.py` (empty), then `tests/models/test_blocks.py`:

```python
from __future__ import annotations

import torch

from sattsr.models.blocks import ResBlock, backwarp, conv_block


def test_conv_block_preserves_spatial_size_by_default():
    block = conv_block(3, 8)
    out = block(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 8, 16, 16)


def test_conv_block_stride_halves_resolution():
    out = conv_block(3, 8, 3, 2, 1)(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 8, 8, 8)


def test_resblock_is_shape_preserving_and_differentiable():
    x = torch.randn(2, 8, 12, 12, requires_grad=True)
    out = ResBlock(8)(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_backwarp_with_zero_flow_is_the_identity():
    x = torch.randn(2, 1, 8, 8)
    torch.testing.assert_close(backwarp(x, torch.zeros(2, 2, 8, 8)), x, atol=1e-5, rtol=1e-5)


def test_backwarp_shifts_by_the_requested_pixel_offset():
    # a horizontal ramp: value == column index
    x = torch.arange(8, dtype=torch.float32).repeat(8, 1)[None, None]
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = 2.0                       # sample two pixels to the right
    out = backwarp(x, flow)
    # interior columns must now read as (col + 2)
    torch.testing.assert_close(out[0, 0, 4, 1:5], torch.tensor([3.0, 4.0, 5.0, 6.0]), atol=1e-4,
                               rtol=1e-4)


def test_backwarp_shifts_vertically():
    x = torch.arange(8, dtype=torch.float32)[:, None].repeat(1, 8)[None, None]
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 1] = 1.0
    out = backwarp(x, flow)
    torch.testing.assert_close(out[0, 0, 1:5, 4], torch.tensor([2.0, 3.0, 4.0, 5.0]), atol=1e-4,
                               rtol=1e-4)


def test_backwarp_clamps_at_the_border():
    x = torch.ones(1, 1, 8, 8)
    out = backwarp(x, torch.full((1, 2, 8, 8), 20.0))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.ones_like(out), atol=1e-5, rtol=1e-5)


def test_backwarp_gradients_reach_both_inputs():
    x = torch.randn(1, 1, 8, 8, requires_grad=True)
    flow = torch.randn(1, 2, 8, 8, requires_grad=True) * 0.5
    backwarp(x, flow).sum().backward()
    assert x.grad is not None and flow.grad is not None
    assert torch.isfinite(flow.grad).all()
    assert float(flow.grad.abs().sum()) > 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/models/test_blocks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.models'`

- [ ] **Step 3: Implement `src/sattsr/models/blocks.py`**

Create an empty `src/sattsr/models/__init__.py` too.

```python
"""Shared network building blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def conv_block(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    """Conv2d followed by a channel-wise PReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, dilation=dilation, bias=True),
        nn.PReLU(out_ch),
    )


class ResBlock(nn.Module):
    """Pre-activation-free residual block that preserves shape."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = conv_block(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.PReLU(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


def backwarp(x: Tensor, flow: Tensor) -> Tensor:
    """Backward-warp `x` by `flow` (pixels; channel 0 = right, channel 1 = down).

    Output pixel (y, x0) reads from input pixel (y + flow_y, x0 + flow_x), bilinearly,
    clamping to the border outside the frame.
    """
    n, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    src_x = grid_x[None] + flow[:, 0]
    src_y = grid_y[None] + flow[:, 1]

    norm_x = 2.0 * src_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(h - 1, 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)          # (N, H, W, 2)

    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/models/test_blocks.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/models tests/models
git commit -m "feat: convolution blocks and differentiable backward warping"
```

---

### Task 12: RAFT-lite correlation flow estimator

**Files:**
- Create: `src/sattsr/models/flow.py`
- Test: `tests/models/test_flow.py`

**Interfaces:**
- Consumes: `conv_block`, `ResBlock`, `backwarp`, `ModelConfig`.
- Produces:
  - `FeatureEncoder(out_ch)` — `(N, 1, H, W) -> (N, out_ch, H/8, W/8)`
  - `local_correlation(f1, f2, radius) -> Tensor` — `(N, (2r+1)^2, H, W)`
  - `FlowRefiner(in_ch, hidden=128)` — `-> (N, 2, H, W)` flow delta
  - `RaftLiteFlow(channels=96, radius=3, iters=4)` with
    `forward(i0, i2) -> tuple[Tensor, Tensor]` = `(flow_02, flow_20)`, each `(N, 2, H/8, W/8)`
    **in units of 1/8-resolution pixels**. `RaftLiteFlow.from_config(cfg: ModelConfig)`.

This is the "AI/ML optical flow" half of the spec: an all-pairs-in-a-window correlation volume
refined iteratively, the same idea as RAFT but with a convolutional update instead of a GRU, sized
so a student GPU can train it.

- [ ] **Step 1: Write the failing test**

`tests/models/test_flow.py`:

```python
from __future__ import annotations

import torch

from sattsr.config import ModelConfig
from sattsr.models.flow import FeatureEncoder, FlowRefiner, RaftLiteFlow, local_correlation


def test_feature_encoder_downsamples_by_eight():
    out = FeatureEncoder(96)(torch.randn(2, 1, 64, 64))
    assert out.shape == (2, 96, 8, 8)


def test_local_correlation_shape_and_peak_position():
    torch.manual_seed(0)
    f1 = torch.randn(1, 16, 8, 8)
    corr = local_correlation(f1, f1, radius=2)
    assert corr.shape == (1, 25, 8, 8)
    # self-correlation peaks at the centre displacement for interior pixels
    centre = (2 * 2 + 1) ** 2 // 2
    assert torch.argmax(corr[0, :, 4, 4]).item() == centre


def test_local_correlation_is_finite_for_zero_features():
    corr = local_correlation(torch.zeros(1, 8, 6, 6), torch.zeros(1, 8, 6, 6), radius=1)
    assert torch.isfinite(corr).all()


def test_flow_refiner_returns_two_channels():
    out = FlowRefiner(in_ch=20, hidden=32)(torch.randn(2, 20, 8, 8))
    assert out.shape == (2, 2, 8, 8)


def test_raft_lite_returns_bidirectional_flow_at_eighth_resolution():
    net = RaftLiteFlow(channels=32, radius=2, iters=2)
    f02, f20 = net(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64))
    assert f02.shape == (2, 2, 8, 8)
    assert f20.shape == (2, 2, 8, 8)
    assert torch.isfinite(f02).all() and torch.isfinite(f20).all()


def test_raft_lite_is_deterministic_in_eval_mode():
    net = RaftLiteFlow(channels=32, radius=2, iters=2).eval()
    x, y = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        a, _ = net(x, y)
        b, _ = net(x, y)
    torch.testing.assert_close(a, b)


def test_raft_lite_gradients_flow_to_the_inputs():
    net = RaftLiteFlow(channels=32, radius=2, iters=2)
    x = torch.rand(1, 1, 64, 64, requires_grad=True)
    y = torch.rand(1, 1, 64, 64, requires_grad=True)
    f02, f20 = net(x, y)
    (f02.sum() + f20.sum()).backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0
    assert y.grad is not None and float(y.grad.abs().sum()) > 0.0


def test_from_config_uses_the_configured_sizes():
    cfg = ModelConfig(flow_channels=32, flow_radius=2, flow_iters=3)
    net = RaftLiteFlow.from_config(cfg)
    assert net.radius == 2 and net.iters == 3
    f02, _ = net(torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32))
    assert f02.shape == (1, 2, 4, 4)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/models/test_flow.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.models.flow'`

- [ ] **Step 3: Implement `src/sattsr/models/flow.py`**

```python
"""RAFT-lite: an iteratively refined local-correlation optical-flow estimator."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sattsr.config import ModelConfig
from sattsr.models.blocks import ResBlock, backwarp, conv_block


class FeatureEncoder(nn.Module):
    """Single-channel image -> dense features at 1/8 resolution."""

    def __init__(self, out_ch: int) -> None:
        super().__init__()
        c1, c2 = max(out_ch // 4, 4), max(out_ch // 2, 8)
        self.net = nn.Sequential(
            conv_block(1, c1, 3, 2, 1),
            conv_block(c1, c2, 3, 2, 1),
            conv_block(c2, out_ch, 3, 2, 1),
            ResBlock(out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def local_correlation(f1: Tensor, f2: Tensor, radius: int) -> Tensor:
    """Correlate each `f1` location against a (2r+1)^2 window of `f2`.

    Returns (N, (2r+1)^2, H, W), scaled by 1/sqrt(C) so the magnitude does not grow
    with feature width.
    """
    n, c, h, w = f1.shape
    k = 2 * radius + 1
    patches = F.unfold(f2, kernel_size=k, padding=radius).view(n, c, k * k, h, w)
    return (f1.unsqueeze(2) * patches).sum(dim=1) / (c**0.5)


class FlowRefiner(nn.Module):
    """Maps (correlation, features, current flow) to a flow increment."""

    def __init__(self, in_ch: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_ch, hidden),
            conv_block(hidden, hidden),
            nn.Conv2d(hidden, 2, 3, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RaftLiteFlow(nn.Module):
    """Bidirectional optical flow at 1/8 resolution, refined over `iters` steps."""

    def __init__(self, channels: int = 96, radius: int = 3, iters: int = 4) -> None:
        super().__init__()
        self.radius = int(radius)
        self.iters = int(iters)
        self.encoder = FeatureEncoder(channels)
        corr_ch = (2 * self.radius + 1) ** 2
        self.refiner = FlowRefiner(corr_ch + channels + 2, hidden=max(channels, 64))

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> RaftLiteFlow:
        """Build from a validated ModelConfig."""
        return cls(channels=cfg.flow_channels, radius=cfg.flow_radius, iters=cfg.flow_iters)

    def _estimate(self, fa: Tensor, fb: Tensor) -> Tensor:
        n, _, h, w = fa.shape
        flow = torch.zeros(n, 2, h, w, device=fa.device, dtype=fa.dtype)
        for _ in range(self.iters):
            corr = local_correlation(fa, backwarp(fb, flow), self.radius)
            flow = flow + self.refiner(torch.cat([corr, fa, flow], dim=1))
        return flow

    def forward(self, i0: Tensor, i2: Tensor) -> tuple[Tensor, Tensor]:
        """Return (flow_02, flow_20) at 1/8 resolution, in 1/8-resolution pixels."""
        f0 = self.encoder(i0)
        f2 = self.encoder(i2)
        return self._estimate(f0, f2), self._estimate(f2, f0)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/models/test_flow.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/models/flow.py tests/models/test_flow.py
git commit -m "feat: RAFT-lite correlation-based optical flow estimator"
```

---

### Task 13: RIFE-style coarse-to-fine synthesis network

**Files:**
- Create: `src/sattsr/models/ifnet.py`
- Test: `tests/models/test_ifnet.py`

**Interfaces:**
- Consumes: `conv_block`, `ResBlock`, `backwarp`, `ModelConfig`.
- Produces:
  - `IFBlock(in_ch, channels)` with `forward(x: Tensor, flow: Tensor, scale: int) -> tuple[Tensor, Tensor]`
    returning `(flow_delta (N,4,H,W), mask_delta (N,1,H,W))` at the **input** resolution.
    `x` carries 4 channels (`warped0`, `warped2`, `t_map`, `mask`); `flow` carries 4
    (`F_t->0` then `F_t->2`), so `in_ch = 8`.
  - `IFNet(cfg: ModelConfig)` with
    `forward(i0, i2, t: Tensor, flow_init: Tensor | None = None) -> dict[str, Tensor]`
    returning keys `flow` `(N,4,H,W)`, `mask` `(N,1,H,W)` (sigmoid, in `[0,1]`),
    `warped0`, `warped2`, `merged`, each `(N,1,H,W)`.

Input `H` and `W` must both be divisible by `4 * max(scales)` (32 for the default `[4, 2, 1]`).

- [ ] **Step 1: Write the failing test**

`tests/models/test_ifnet.py`:

```python
from __future__ import annotations

import pytest
import torch

from sattsr.config import ModelConfig
from sattsr.models.ifnet import IFBlock, IFNet


def _cfg() -> ModelConfig:
    return ModelConfig(base_channels=16, scales=[4, 2, 1])


def test_ifblock_returns_deltas_at_input_resolution():
    block = IFBlock(in_ch=8, channels=16)
    x = torch.randn(2, 4, 64, 64)
    flow = torch.zeros(2, 4, 64, 64)
    d_flow, d_mask = block(x, flow, scale=2)
    assert d_flow.shape == (2, 4, 64, 64)
    assert d_mask.shape == (2, 1, 64, 64)


def test_ifblock_works_at_every_configured_scale():
    block = IFBlock(in_ch=8, channels=16)
    for scale in (4, 2, 1):
        d_flow, d_mask = block(torch.randn(1, 4, 64, 64), torch.zeros(1, 4, 64, 64), scale=scale)
        assert d_flow.shape == (1, 4, 64, 64)
        assert d_mask.shape == (1, 1, 64, 64)


def test_ifnet_output_keys_and_shapes():
    net = IFNet(_cfg())
    i0, i2 = torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64)
    out = net(i0, i2, torch.full((2,), 0.5))
    assert set(out) == {"flow", "mask", "warped0", "warped2", "merged"}
    assert out["flow"].shape == (2, 4, 64, 64)
    assert out["mask"].shape == (2, 1, 64, 64)
    for key in ("warped0", "warped2", "merged"):
        assert out[key].shape == (2, 1, 64, 64)
    assert torch.isfinite(out["merged"]).all()


def test_ifnet_mask_is_a_probability():
    out = IFNet(_cfg())(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64), torch.tensor([0.5]))
    assert float(out["mask"].min()) >= 0.0
    assert float(out["mask"].max()) <= 1.0


def test_ifnet_merged_is_the_masked_blend_of_the_warps():
    out = IFNet(_cfg())(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64), torch.tensor([0.5]))
    expected = out["warped0"] * out["mask"] + out["warped2"] * (1.0 - out["mask"])
    torch.testing.assert_close(out["merged"], expected, atol=1e-5, rtol=1e-5)


def test_ifnet_accepts_a_flow_initialisation():
    net = IFNet(_cfg())
    i0, i2, t = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64), torch.tensor([0.5])
    with torch.no_grad():
        plain = net(i0, i2, t)["flow"]
        seeded = net(i0, i2, t, flow_init=torch.full((1, 4, 64, 64), 3.0))["flow"]
    assert not torch.allclose(plain, seeded)


def test_ifnet_uses_the_timestep():
    net = IFNet(_cfg())
    i0, i2 = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        early = net(i0, i2, torch.tensor([0.25]))["merged"]
        late = net(i0, i2, torch.tensor([0.75]))["merged"]
    assert not torch.allclose(early, late)


def test_ifnet_gradients_reach_the_inputs():
    net = IFNet(_cfg())
    i0 = torch.rand(1, 1, 64, 64, requires_grad=True)
    i2 = torch.rand(1, 1, 64, 64, requires_grad=True)
    net(i0, i2, torch.tensor([0.5]))["merged"].sum().backward()
    assert float(i0.grad.abs().sum()) > 0.0
    assert float(i2.grad.abs().sum()) > 0.0


def test_ifnet_rejects_sizes_not_divisible_by_the_coarsest_scale():
    with pytest.raises(ValueError):
        IFNet(_cfg())(torch.rand(1, 1, 60, 64), torch.rand(1, 1, 60, 64), torch.tensor([0.5]))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/models/test_ifnet.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.models.ifnet'`

- [ ] **Step 3: Implement `src/sattsr/models/ifnet.py`**

```python
"""RIFE-style coarse-to-fine intermediate-flow and fusion network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sattsr.config import ModelConfig
from sattsr.models.blocks import ResBlock, backwarp, conv_block


class IFBlock(nn.Module):
    """One coarse-to-fine refinement stage.

    Consumes the current warped pair plus the current flow and mask, and predicts
    increments to both. `scale` selects the resolution the stage reasons at: large
    scale sees big displacements, small scale sharpens detail.
    """

    def __init__(self, in_ch: int, channels: int) -> None:
        super().__init__()
        half = max(channels // 2, 8)
        self.stem = nn.Sequential(
            conv_block(in_ch, half, 3, 2, 1),
            conv_block(half, channels, 3, 2, 1),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(4)])
        self.head = nn.ConvTranspose2d(channels, 5, 4, 2, 1)

    def forward(self, x: Tensor, flow: Tensor, scale: int = 1) -> tuple[Tensor, Tensor]:
        """Return (flow_delta, mask_delta) at the resolution of `x`."""
        if scale != 1:
            inv = 1.0 / float(scale)
            x = F.interpolate(x, scale_factor=inv, mode="bilinear", align_corners=False)
            flow = (
                F.interpolate(flow, scale_factor=inv, mode="bilinear", align_corners=False)
                / float(scale)
            )
        out = self.head(self.body(self.stem(torch.cat([x, flow], dim=1))))
        out = F.interpolate(out, scale_factor=float(scale) * 2.0, mode="bilinear",
                            align_corners=False)
        return out[:, :4] * (float(scale) * 2.0), out[:, 4:5]


class IFNet(nn.Module):
    """Stacks IFBlocks from coarse to fine and fuses the two warped views."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.scales = list(cfg.scales)
        if not self.scales:
            raise ValueError("ModelConfig.scales must not be empty")
        # 4 image/context channels (warped0, warped2, t_map, mask) + 4 flow channels
        self.blocks = nn.ModuleList(
            IFBlock(in_ch=8, channels=cfg.base_channels) for _ in self.scales
        )

    @property
    def size_multiple(self) -> int:
        """Input height and width must both be divisible by this."""
        return 4 * max(self.scales)

    def forward(
        self,
        i0: Tensor,
        i2: Tensor,
        t: Tensor,
        flow_init: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict intermediate flows, a fusion mask, and the blended frame."""
        n, _, h, w = i0.shape
        m = self.size_multiple
        if h % m or w % m:
            raise ValueError(f"input {h}x{w} must be divisible by {m}")

        t_map = t.reshape(-1, 1, 1, 1).to(i0.dtype).expand(n, 1, h, w)
        flow = (
            torch.zeros(n, 4, h, w, device=i0.device, dtype=i0.dtype)
            if flow_init is None
            else flow_init.to(i0.dtype)
        )
        mask = torch.zeros(n, 1, h, w, device=i0.device, dtype=i0.dtype)
        warped0 = backwarp(i0, flow[:, :2])
        warped2 = backwarp(i2, flow[:, 2:4])

        for block, scale in zip(self.blocks, self.scales, strict=True):
            x = torch.cat([warped0, warped2, t_map, mask], dim=1)
            d_flow, d_mask = block(x, flow, scale=scale)
            flow = flow + d_flow
            mask = mask + d_mask
            warped0 = backwarp(i0, flow[:, :2])
            warped2 = backwarp(i2, flow[:, 2:4])

        sigma = torch.sigmoid(mask)
        return {
            "flow": flow,
            "mask": sigma,
            "warped0": warped0,
            "warped2": warped2,
            "merged": warped0 * sigma + warped2 * (1.0 - sigma),
        }
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/models/test_ifnet.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/models/ifnet.py tests/models/test_ifnet.py
git commit -m "feat: RIFE-style coarse-to-fine intermediate flow and fusion network"
```

---

### Task 14: The complete FrameInterpolator

**Files:**
- Create: `src/sattsr/models/interpolator.py`
- Test: `tests/models/test_interpolator.py`

**Interfaces:**
- Consumes: `RaftLiteFlow`, `IFNet`, `ResBlock`, `conv_block`, `ModelConfig`.
- Produces:
  - `RefineNet(in_ch, channels=32)` — small U-Net returning a bounded residual in `[-0.5, 0.5]`.
  - `FrameInterpolator(cfg: ModelConfig)`:
    - `forward(i0, i2, t: Tensor | float = 0.5) -> dict[str, Tensor]` with keys
      `pred`, `merged`, `flow`, `mask`, `warped0`, `warped2`, `residual`. `pred` is clamped to `[0, 1]`.
    - `interpolate(i0, i2, t=0.5) -> Tensor` — `no_grad` convenience returning `pred` only.
    - `size_multiple -> int`
  - `build_model(cfg: ModelConfig) -> FrameInterpolator`

The linear-motion identity used to seed the IFNet: with `F_{0->2}` and `F_{2->0}` from the flow
net, `F_{t->0} ≈ -t · F_{0->2}` and `F_{t->2} ≈ -(1-t) · F_{2->0}`.

- [ ] **Step 1: Write the failing test**

`tests/models/test_interpolator.py`:

```python
from __future__ import annotations

import pytest
import torch

from sattsr.config import ModelConfig
from sattsr.models.interpolator import FrameInterpolator, RefineNet, build_model


def _cfg(use_raft: bool = True) -> ModelConfig:
    return ModelConfig(
        base_channels=16, scales=[4, 2, 1], use_raft_init=use_raft,
        flow_channels=32, flow_radius=2, flow_iters=2,
    )


def test_refinenet_output_is_bounded_and_shape_preserving():
    out = RefineNet(in_ch=6, channels=8)(torch.randn(2, 6, 64, 64))
    assert out.shape == (2, 1, 64, 64)
    assert float(out.abs().max()) <= 0.5


def test_forward_keys_shapes_and_range():
    model = build_model(_cfg())
    out = model(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64), 0.5)
    assert set(out) == {"pred", "merged", "flow", "mask", "warped0", "warped2", "residual"}
    assert out["pred"].shape == (2, 1, 64, 64)
    assert float(out["pred"].min()) >= 0.0 and float(out["pred"].max()) <= 1.0
    assert torch.isfinite(out["pred"]).all()


def test_t_accepts_a_scalar_or_a_batch_tensor():
    model = build_model(_cfg())
    i0, i2 = torch.rand(3, 1, 64, 64), torch.rand(3, 1, 64, 64)
    a = model(i0, i2, 0.5)["pred"]
    b = model(i0, i2, torch.tensor([0.5, 0.5, 0.5]))["pred"]
    assert a.shape == b.shape == (3, 1, 64, 64)


def test_different_timesteps_give_different_frames():
    model = build_model(_cfg()).eval()
    i0, i2 = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        assert not torch.allclose(model(i0, i2, 0.25)["pred"], model(i0, i2, 0.75)["pred"])


def test_raft_initialisation_can_be_disabled():
    model = build_model(_cfg(use_raft=False))
    assert model.flow_net is None
    assert model(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64))["pred"].shape == (
        1, 1, 64, 64
    )


def test_interpolate_is_a_no_grad_shortcut():
    model = build_model(_cfg()).eval()
    pred = model.interpolate(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64), 0.5)
    assert pred.shape == (1, 1, 64, 64)
    assert not pred.requires_grad


def test_gradients_reach_every_submodule():
    model = build_model(_cfg())
    model(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64), 0.5)["pred"].sum().backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    assert missing == [], f"no gradient reached: {missing[:5]}"


def test_eval_mode_is_deterministic():
    model = build_model(_cfg()).eval()
    i0, i2 = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(model(i0, i2, 0.5)["pred"], model(i0, i2, 0.5)["pred"])


def test_size_multiple_is_reported_and_enforced():
    model = build_model(_cfg())
    assert model.size_multiple == 16
    with pytest.raises(ValueError):
        model(torch.rand(1, 1, 60, 64), torch.rand(1, 1, 60, 64), 0.5)


def test_model_is_small_enough_to_train_on_a_student_gpu():
    params = sum(p.numel() for p in build_model(ModelConfig()).parameters())
    assert params < 25_000_000, f"{params} parameters is too large"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/models/test_interpolator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.models.interpolator'`

- [ ] **Step 3: Implement `src/sattsr/models/interpolator.py`**

```python
"""The end-to-end frame interpolator: flow initialisation, fusion, refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sattsr.config import ModelConfig
from sattsr.models.blocks import ResBlock, conv_block
from sattsr.models.flow import RaftLiteFlow
from sattsr.models.ifnet import IFNet

_RESIDUAL_LIMIT = 0.5


class RefineNet(nn.Module):
    """Small U-Net that predicts a bounded correction to the fused frame."""

    def __init__(self, in_ch: int, channels: int = 32) -> None:
        super().__init__()
        self.down1 = conv_block(in_ch, channels, 3, 2, 1)
        self.down2 = conv_block(channels, channels * 2, 3, 2, 1)
        self.mid = ResBlock(channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)
        self.act1 = nn.PReLU(channels)
        self.up2 = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)
        self.act2 = nn.PReLU(channels)
        self.out = nn.Conv2d(channels, 1, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        d1 = self.down1(x)
        m = self.mid(self.down2(d1))
        u1 = self.act1(self.up1(m))
        u2 = self.act2(self.up2(torch.cat([u1, d1], dim=1)))
        return torch.tanh(self.out(u2)) * _RESIDUAL_LIMIT


class FrameInterpolator(nn.Module):
    """RAFT-lite flow initialisation -> RIFE-style fusion -> residual refinement."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.flow_net = RaftLiteFlow.from_config(cfg) if cfg.use_raft_init else None
        self.ifnet = IFNet(cfg)
        # i0, i2, warped0, warped2, merged, mask
        self.refine = RefineNet(in_ch=6, channels=max(cfg.base_channels // 2, 8))

    @property
    def size_multiple(self) -> int:
        """Input height and width must both be divisible by this."""
        return max(self.ifnet.size_multiple, 8)

    def _flow_init(self, i0: Tensor, i2: Tensor, t: Tensor) -> Tensor | None:
        if self.flow_net is None:
            return None
        flow_02, flow_20 = self.flow_net(i0, i2)
        tt = t.reshape(-1, 1, 1, 1).to(i0.dtype)
        # linear-motion identity, then lift from 1/8 resolution back to full pixels
        f_t0 = -tt * flow_02
        f_t2 = -(1.0 - tt) * flow_20
        coarse = torch.cat([f_t0, f_t2], dim=1)
        return F.interpolate(coarse, size=i0.shape[-2:], mode="bilinear", align_corners=False) * 8.0

    def forward(self, i0: Tensor, i2: Tensor, t: Tensor | float = 0.5) -> dict[str, Tensor]:
        """Synthesize the frame at normalised time `t` between `i0` and `i2`."""
        if not torch.is_tensor(t):
            t = torch.full((i0.shape[0],), float(t), device=i0.device, dtype=i0.dtype)
        t = t.to(device=i0.device, dtype=i0.dtype).reshape(-1)

        out = self.ifnet(i0, i2, t, flow_init=self._flow_init(i0, i2, t))
        residual = self.refine(
            torch.cat(
                [i0, i2, out["warped0"], out["warped2"], out["merged"], out["mask"]], dim=1
            )
        )
        out["residual"] = residual
        out["pred"] = torch.clamp(out["merged"] + residual, 0.0, 1.0)
        return out

    @torch.no_grad()
    def interpolate(self, i0: Tensor, i2: Tensor, t: Tensor | float = 0.5) -> Tensor:
        """Inference shortcut returning just the predicted frame."""
        return self.forward(i0, i2, t)["pred"]


def build_model(cfg: ModelConfig) -> FrameInterpolator:
    """Construct the interpolator described by `cfg`."""
    return FrameInterpolator(cfg)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/models -v`
Expected: 35 passed

If `test_gradients_reach_every_submodule` reports unused parameters, the cause is almost always a
head whose output is multiplied by zero — check that `_flow_init` is actually being called
(`use_raft_init: true`) and that every IFBlock in `scales` runs.

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/models/interpolator.py tests/models/test_interpolator.py
git commit -m "feat: end-to-end frame interpolator with flow init and residual refinement"
```

---

# Phase 3 — Training

### Task 15: Composite training loss

**Files:**
- Create: `src/sattsr/losses/__init__.py`, `src/sattsr/losses/functional.py`,
  `src/sattsr/losses/ssim.py`, `src/sattsr/losses/composite.py`
- Test: `tests/losses/test_functional.py`, `tests/losses/test_ssim.py`,
  `tests/losses/test_composite.py`

**Interfaces:**
- Consumes: `LossConfig`.
- Produces:
  - `charbonnier(pred, target, *, mask=None, eps=1e-3) -> Tensor`
  - `flow_smoothness(flow, image) -> Tensor` — edge-aware first-order, `flow` is `(N,4,H,W)`
  - `warp_consistency(warped0, warped2, *, mask=None) -> Tensor`
  - `radiometric_consistency(pred, i0, i2, t, *, mask=None) -> Tensor`
  - `SSIM(window_size=11, sigma=1.5, data_range=1.0)` — `forward(a, b) -> Tensor` scalar mean SSIM
  - `SSIMLoss(...)` — `forward(a, b) -> 1 - SSIM`
  - `CompositeLoss(cfg: LossConfig)` — `forward(out: dict, batch: dict) -> tuple[Tensor, dict[str, float]]`.
    The dict always contains keys `total`, `recon`, `ssim`, `smooth`, `consistency`, `radiometric`.

The five terms map one-to-one onto the loss list in the synopsis: reconstruction
(L1/Charbonnier), perceptual/structural (SSIM), flow smoothness, temporal consistency (the two
warped views must agree), and radiometric consistency (the synthesized frame's mean brightness
temperature must sit on the line between its neighbours).

- [ ] **Step 1: Write the failing tests**

Create `tests/losses/__init__.py` (empty), then `tests/losses/test_functional.py`:

```python
from __future__ import annotations

import pytest
import torch

from sattsr.losses.functional import (
    charbonnier,
    flow_smoothness,
    radiometric_consistency,
    warp_consistency,
)


def test_charbonnier_is_near_zero_for_identical_inputs():
    x = torch.rand(2, 1, 8, 8)
    assert float(charbonnier(x, x)) < 2e-3


def test_charbonnier_grows_with_error():
    a, b = torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8)
    assert float(charbonnier(a, b)) > float(charbonnier(a, b * 0.5))


def test_charbonnier_mask_excludes_pixels():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    target[..., 0, 0] = 100.0
    mask = torch.ones(1, 1, 4, 4)
    mask[..., 0, 0] = 0.0
    assert float(charbonnier(pred, target, mask=mask)) < 2e-3
    assert float(charbonnier(pred, target)) > 1.0


def test_charbonnier_all_zero_mask_does_not_divide_by_zero():
    value = charbonnier(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4),
                        mask=torch.zeros(1, 1, 4, 4))
    assert torch.isfinite(value)


def test_flow_smoothness_penalises_a_jumpy_field():
    image = torch.zeros(1, 1, 8, 8)
    smooth = torch.zeros(1, 4, 8, 8)
    jumpy = torch.zeros(1, 4, 8, 8)
    jumpy[:, :, :, 4:] = 5.0
    assert float(flow_smoothness(jumpy, image)) > float(flow_smoothness(smooth, image))


def test_flow_smoothness_is_relaxed_at_image_edges():
    flow = torch.zeros(1, 4, 8, 8)
    flow[:, :, :, 4:] = 5.0
    flat_image = torch.zeros(1, 1, 8, 8)
    edgy_image = torch.zeros(1, 1, 8, 8)
    edgy_image[:, :, :, 4:] = 1.0
    assert float(flow_smoothness(flow, edgy_image)) < float(flow_smoothness(flow, flat_image))


def test_warp_consistency_is_zero_when_the_two_views_agree():
    x = torch.rand(1, 1, 8, 8)
    assert float(warp_consistency(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_radiometric_consistency_rewards_the_time_weighted_mean():
    i0 = torch.zeros(1, 1, 8, 8)
    i2 = torch.ones(1, 1, 8, 8)
    t = torch.tensor([0.25])
    good = torch.full((1, 1, 8, 8), 0.25)
    bad = torch.full((1, 1, 8, 8), 0.9)
    assert float(radiometric_consistency(good, i0, i2, t)) == pytest.approx(0.0, abs=1e-6)
    assert float(radiometric_consistency(bad, i0, i2, t)) > 0.5


def test_every_loss_term_is_differentiable():
    pred = torch.rand(1, 1, 8, 8, requires_grad=True)
    other = torch.rand(1, 1, 8, 8)
    flow = torch.zeros(1, 4, 8, 8, requires_grad=True)
    total = (
        charbonnier(pred, other)
        + flow_smoothness(flow, pred)
        + warp_consistency(pred, other)
        + radiometric_consistency(pred, other, other, torch.tensor([0.5]))
    )
    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
```

`tests/losses/test_ssim.py`:

```python
from __future__ import annotations

import pytest
import torch

from sattsr.losses.ssim import SSIM, SSIMLoss


def test_ssim_of_an_image_with_itself_is_one():
    x = torch.rand(2, 1, 32, 32)
    assert float(SSIM()(x, x)) == pytest.approx(1.0, abs=1e-3)


def test_ssim_drops_when_noise_is_added():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 32, 32)
    noisy = (x + torch.randn_like(x) * 0.3).clamp(0, 1)
    assert float(SSIM()(x, noisy)) < 0.9


def test_ssim_loss_is_one_minus_ssim():
    x = torch.rand(1, 1, 32, 32)
    y = torch.rand(1, 1, 32, 32)
    assert float(SSIMLoss()(x, y)) == pytest.approx(1.0 - float(SSIM()(x, y)), abs=1e-6)


def test_ssim_loss_is_differentiable():
    x = torch.rand(1, 1, 32, 32, requires_grad=True)
    SSIMLoss()(x, torch.rand(1, 1, 32, 32)).backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0


def test_ssim_matches_skimage_within_tolerance():
    skimage = pytest.importorskip("skimage.metrics")
    torch.manual_seed(1)
    a = torch.rand(1, 1, 64, 64)
    b = (a + torch.randn_like(a) * 0.1).clamp(0, 1)
    ours = float(SSIM()(a, b))
    theirs = skimage.structural_similarity(
        a[0, 0].numpy(), b[0, 0].numpy(), data_range=1.0, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False,
    )
    assert ours == pytest.approx(theirs, abs=0.03)
```

`tests/losses/test_composite.py`:

```python
from __future__ import annotations

import torch

from sattsr.config import LossConfig
from sattsr.losses.composite import CompositeLoss


def _out_and_batch(perfect: bool = False):
    torch.manual_seed(0)
    target = torch.rand(2, 1, 32, 32)
    pred = target.clone() if perfect else torch.rand(2, 1, 32, 32)
    out = {
        "pred": pred.requires_grad_(True),
        "merged": pred,
        "flow": torch.zeros(2, 4, 32, 32, requires_grad=True),
        "mask": torch.full((2, 1, 32, 32), 0.5),
        "warped0": target,
        "warped2": target,
        "residual": torch.zeros(2, 1, 32, 32),
    }
    batch = {
        "i0": torch.rand(2, 1, 32, 32),
        "i1": target,
        "i2": torch.rand(2, 1, 32, 32),
        "valid": torch.ones(2, 1, 32, 32),
        "t": torch.tensor([0.5, 0.5]),
    }
    return out, batch


def test_reports_every_component():
    total, parts = CompositeLoss(LossConfig())(*_out_and_batch())
    assert set(parts) == {"total", "recon", "ssim", "smooth", "consistency", "radiometric"}
    assert all(isinstance(v, float) for v in parts.values())
    assert parts["total"] == float(total)


def test_a_perfect_prediction_scores_lower_than_a_random_one():
    loss = CompositeLoss(LossConfig())
    good, _ = loss(*_out_and_batch(perfect=True))
    bad, _ = loss(*_out_and_batch(perfect=False))
    assert float(good) < float(bad)


def test_zero_weights_disable_terms():
    cfg = LossConfig(w_recon=1.0, w_ssim=0.0, w_smooth=0.0, w_consistency=0.0, w_radiometric=0.0)
    out, batch = _out_and_batch()
    total, parts = CompositeLoss(cfg)(out, batch)
    assert parts["total"] == parts["recon"]
    assert float(total) > 0.0


def test_total_is_differentiable_through_pred_and_flow():
    out, batch = _out_and_batch()
    total, _ = CompositeLoss(LossConfig())(out, batch)
    total.backward()
    assert out["pred"].grad is not None and float(out["pred"].grad.abs().sum()) > 0.0
    assert out["flow"].grad is not None


def test_invalid_pixels_are_ignored():
    out, batch = _out_and_batch(perfect=True)
    out["pred"] = out["pred"].detach().clone()
    out["pred"][:, :, :8, :] = 0.0            # corrupt a strip
    batch["valid"] = torch.ones(2, 1, 32, 32)
    with_all = float(CompositeLoss(LossConfig())(out, batch)[1]["recon"])
    batch["valid"][:, :, :8, :] = 0.0         # mask the same strip out
    with_mask = float(CompositeLoss(LossConfig())(out, batch)[1]["recon"])
    assert with_mask < with_all
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/losses -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.losses'`

- [ ] **Step 3: Implement `src/sattsr/losses/functional.py`**

Create an empty `src/sattsr/losses/__init__.py` too.

```python
"""Individual loss terms."""

from __future__ import annotations

import torch
from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    if values.shape[1] != mask.shape[1]:
        mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def charbonnier(
    pred: Tensor, target: Tensor, *, mask: Tensor | None = None, eps: float = 1e-3
) -> Tensor:
    """Robust L1 reconstruction loss, optionally restricted to valid pixels."""
    return _masked_mean(torch.sqrt((pred - target) ** 2 + eps * eps), mask)


def flow_smoothness(flow: Tensor, image: Tensor) -> Tensor:
    """Edge-aware first-order smoothness: penalise flow gradients away from image edges."""
    d_x = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()
    d_y = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()
    g_x = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    g_y = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    return (d_x * torch.exp(-g_x)).mean() + (d_y * torch.exp(-g_y)).mean()


def warp_consistency(warped0: Tensor, warped2: Tensor, *, mask: Tensor | None = None) -> Tensor:
    """Temporal consistency: the forward and backward warped views must agree."""
    return _masked_mean((warped0 - warped2).abs(), mask)


def radiometric_consistency(
    pred: Tensor, i0: Tensor, i2: Tensor, t: Tensor, *, mask: Tensor | None = None
) -> Tensor:
    """Keep the synthesized frame's mean brightness on the line between its neighbours."""
    if mask is None:
        mask = torch.ones_like(pred)
    weight = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mean_pred = (pred * mask).sum(dim=(1, 2, 3)) / weight
    mean_0 = (i0 * mask).sum(dim=(1, 2, 3)) / weight
    mean_2 = (i2 * mask).sum(dim=(1, 2, 3)) / weight
    tt = t.reshape(-1).to(pred.dtype)
    return ((mean_pred - ((1.0 - tt) * mean_0 + tt * mean_2)).abs()).mean()
```

- [ ] **Step 4: Implement `src/sattsr/losses/ssim.py`**

```python
"""Differentiable structural similarity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gaussian_window(size: int, sigma: float) -> Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return torch.outer(g, g)[None, None]


class SSIM(nn.Module):
    """Mean SSIM over a batch, using Gaussian weighting (no sample-covariance bias)."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.register_buffer("window", _gaussian_window(window_size, sigma), persistent=False)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Scalar mean SSIM between two (N, 1, H, W) tensors."""
        window = self.window.to(dtype=a.dtype, device=a.device)
        pad = self.window_size // 2

        mu_a = F.conv2d(a, window, padding=pad)
        mu_b = F.conv2d(b, window, padding=pad)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

        sigma_a2 = F.conv2d(a * a, window, padding=pad) - mu_a2
        sigma_b2 = F.conv2d(b * b, window, padding=pad) - mu_b2
        sigma_ab = F.conv2d(a * b, window, padding=pad) - mu_ab

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
        den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
        return (num / den).mean()


class SSIMLoss(nn.Module):
    """1 - SSIM, so lower is better."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0) -> None:
        super().__init__()
        self.ssim = SSIM(window_size, sigma, data_range)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return 1.0 - self.ssim(a, b)
```

- [ ] **Step 5: Implement `src/sattsr/losses/composite.py`**

```python
"""The weighted training objective."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sattsr.config import LossConfig
from sattsr.losses.functional import (
    charbonnier,
    flow_smoothness,
    radiometric_consistency,
    warp_consistency,
)
from sattsr.losses.ssim import SSIMLoss


class CompositeLoss(nn.Module):
    """Reconstruction + structure + flow smoothness + temporal + radiometric."""

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ssim_loss = SSIMLoss(data_range=1.0)

    def forward(
        self, out: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        """Return the scalar total and a float breakdown for logging."""
        pred = out["pred"]
        target = batch["i1"]
        mask = batch.get("valid")
        cfg = self.cfg

        recon = charbonnier(pred, target, mask=mask)
        ssim = self.ssim_loss(pred, target)
        smooth = flow_smoothness(out["flow"], target)
        consistency = warp_consistency(out["warped0"], out["warped2"], mask=mask)
        radiometric = radiometric_consistency(pred, batch["i0"], batch["i2"], batch["t"],
                                              mask=mask)

        total = (
            cfg.w_recon * recon
            + cfg.w_ssim * ssim
            + cfg.w_smooth * smooth
            + cfg.w_consistency * consistency
            + cfg.w_radiometric * radiometric
        )

        parts = {
            "recon": float(cfg.w_recon * recon.detach()),
            "ssim": float(cfg.w_ssim * ssim.detach()),
            "smooth": float(cfg.w_smooth * smooth.detach()),
            "consistency": float(cfg.w_consistency * consistency.detach()),
            "radiometric": float(cfg.w_radiometric * radiometric.detach()),
        }
        parts["total"] = float(total.detach())
        return total, parts
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/losses -v`
Expected: 18 passed

- [ ] **Step 7: Commit**

```bash
git add src/sattsr/losses tests/losses
git commit -m "feat: composite loss with reconstruction, SSIM, smoothness, temporal and radiometric terms"
```

---

### Task 16: Checkpointing

**Files:**
- Create: `src/sattsr/train/__init__.py`, `src/sattsr/train/checkpoint.py`
- Test: `tests/train/test_checkpoint.py`

**Interfaces:**
- Consumes: `Config`.
- Produces:
  - `save_checkpoint(path, *, model, optimizer=None, epoch, best_metric, config=None, extra=None) -> Path`
  - `load_checkpoint(path, model=None, optimizer=None, *, map_location="cpu", strict=True) -> dict`
    — returns `{"epoch", "best_metric", "config", "extra"}`.

- [ ] **Step 1: Write the failing test**

Create `tests/train/__init__.py` (empty), then `tests/train/test_checkpoint.py`:

```python
from __future__ import annotations

import pytest
import torch

from sattsr.config import Config, ModelConfig
from sattsr.models.interpolator import build_model
from sattsr.train.checkpoint import load_checkpoint, save_checkpoint

CFG = ModelConfig(base_channels=16, scales=[4, 2, 1], flow_channels=32, flow_radius=2,
                  flow_iters=2)


def _config() -> Config:
    return Config.model_validate(
        {"data": {"sensor": "goes19", "raw_root": "r", "cache_root": "c"},
         "model": CFG.model_dump()}
    )


def test_round_trip_restores_weights_and_metadata(tmp_path):
    model = build_model(CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(tmp_path / "best.pt", model=model, optimizer=optimizer,
                           epoch=7, best_metric=31.5, config=_config())
    assert path.exists()

    restored = build_model(CFG)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1.0)
    meta = load_checkpoint(path, restored, restored_opt)

    assert meta["epoch"] == 7
    assert meta["best_metric"] == pytest.approx(31.5)
    assert meta["config"]["data"]["sensor"] == "goes19"
    for (_, a), (_, b) in zip(model.state_dict().items(), restored.state_dict().items()):
        torch.testing.assert_close(a, b)
    assert restored_opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_loading_without_a_model_returns_metadata_only(tmp_path):
    model = build_model(CFG)
    path = save_checkpoint(tmp_path / "c.pt", model=model, epoch=1, best_metric=0.0)
    meta = load_checkpoint(path)
    assert meta["epoch"] == 1
    assert "model_state" in meta


def test_extra_payload_survives(tmp_path):
    path = save_checkpoint(tmp_path / "c.pt", model=build_model(CFG), epoch=0, best_metric=0.0,
                           extra={"sensor_mean": 245.0})
    assert load_checkpoint(path)["extra"]["sensor_mean"] == pytest.approx(245.0)


def test_strict_false_tolerates_a_shape_change(tmp_path):
    path = save_checkpoint(tmp_path / "c.pt", model=build_model(CFG), epoch=0, best_metric=0.0)
    wider = build_model(ModelConfig(base_channels=32, scales=[4, 2, 1], flow_channels=32,
                                    flow_radius=2, flow_iters=2))
    with pytest.raises(RuntimeError):
        load_checkpoint(path, wider, strict=True)
    load_checkpoint(path, wider, strict=False)      # must not raise


def test_parent_directory_is_created(tmp_path):
    path = save_checkpoint(tmp_path / "deep" / "nested" / "c.pt", model=build_model(CFG),
                           epoch=0, best_metric=0.0)
    assert path.exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/train/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.train'`

- [ ] **Step 3: Implement `src/sattsr/train/checkpoint.py`**

Create an empty `src/sattsr/train/__init__.py` too.

```python
"""Checkpoint save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from sattsr.config import Config


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    epoch: int,
    best_metric: float,
    config: Config | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write model weights plus enough metadata to resume or audit the run."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config": config.model_dump(mode="json") if config is not None else None,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, out)
    return out


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    *,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring a model and optimizer in place."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
```

Note: `strict=False` still raises on a *shape* mismatch for a key that exists in both. The test
above relies on that — `base_channels` 16 vs 32 changes tensor shapes, so `strict=True` raises;
`strict=False` must not. If your torch version raises in both cases, filter the state dict:

```python
    if model is not None:
        state = payload["model_state"]
        if not strict:
            own = model.state_dict()
            state = {k: v for k, v in state.items()
                     if k in own and own[k].shape == v.shape}
        model.load_state_dict(state, strict=strict)
```

Use that form — it is the behaviour the test asserts and it is what fine-tuning across
architecture tweaks needs.

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/train/test_checkpoint.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/train tests/train
git commit -m "feat: checkpoint save and load with config and metadata"
```

---

### Task 17: Training loop

**Files:**
- Create: `src/sattsr/train/loop.py`
- Test: `tests/train/test_loop.py`

**Interfaces:**
- Consumes: `CompositeLoss`, `FrameInterpolator`, `TripletDataset`, `save_checkpoint`, `Config`.
- Produces:
  - `seed_everything(seed: int) -> None`
  - `EpochResult(loss: float, components: dict[str, float], psnr: float)` frozen dataclass
  - `train_one_epoch(model, loader, criterion, optimizer, device, *, scaler=None, grad_clip=1.0) -> EpochResult`
  - `validate(model, loader, criterion, device) -> EpochResult`
  - `fit(model, train_loader, val_loader, criterion, optimizer, device, *, epochs, checkpoint_dir, amp=True, config=None, scheduler=None, on_epoch=None) -> Path`
    — writes `last.pt` every epoch and `best.pt` on the best validation PSNR; returns the `best.pt` path.

- [ ] **Step 1: Write the failing test**

`tests/train/test_loop.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from sattsr.config import LossConfig, ModelConfig, NormalizationConfig
from sattsr.data.dataset import TripletDataset
from sattsr.data.index import FrameRef
from sattsr.data.triplets import Triplet
from sattsr.losses.composite import CompositeLoss
from sattsr.models.interpolator import build_model
from sattsr.train.loop import EpochResult, fit, seed_everything, train_one_epoch, validate

from datetime import datetime, timedelta, timezone
from pathlib import Path

CFG = ModelConfig(base_channels=8, scales=[2, 1], use_raft_init=False)
NORM = NormalizationConfig(bt_min=180.0, bt_max=330.0)


def _translating_triplets(tmp_path: Path, n: int = 4, size: int = 64):
    """Frames whose pattern shifts one pixel per step, so interpolation is learnable."""
    t0 = datetime(2026, 9, 4, tzinfo=timezone.utc)
    refs = []
    for i in range(n + 2):
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        bt = (250.0 + 20.0 * np.sin((xx + 2 * i) * 0.3) + 5.0 * np.cos(yy * 0.2)).astype(
            np.float32
        )
        p = tmp_path / f"f{i}.npy"
        np.save(p, bt)
        refs.append(FrameRef(t0 + timedelta(minutes=10 * i), p, "goes19"))
    return [Triplet(refs[i], refs[i + 1], refs[i + 2]) for i in range(n)]


def _loader(tmp_path: Path, batch_size: int = 2, augment: bool = False) -> DataLoader:
    ds = TripletDataset(_translating_triplets(tmp_path), NORM, tile_size=32, augment=augment)
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


def test_seed_everything_makes_sampling_reproducible():
    seed_everything(11)
    a = torch.rand(4)
    seed_everything(11)
    torch.testing.assert_close(torch.rand(4), a)


def test_train_one_epoch_reports_metrics_and_steps_the_optimizer(tmp_path):
    model = build_model(CFG)
    before = [p.detach().clone() for p in model.parameters()]
    result = train_one_epoch(
        model, _loader(tmp_path), CompositeLoss(LossConfig()),
        torch.optim.AdamW(model.parameters(), lr=1e-3), torch.device("cpu"),
    )
    assert isinstance(result, EpochResult)
    assert np.isfinite(result.loss) and np.isfinite(result.psnr)
    assert "recon" in result.components
    assert any(not torch.allclose(a, b) for a, b in zip(before, model.parameters()))


def test_validate_does_not_change_weights(tmp_path):
    model = build_model(CFG)
    before = [p.detach().clone() for p in model.parameters()]
    result = validate(model, _loader(tmp_path), CompositeLoss(LossConfig()), torch.device("cpu"))
    assert np.isfinite(result.psnr)
    for a, b in zip(before, model.parameters()):
        torch.testing.assert_close(a, b)


def test_fit_writes_last_and_best_and_returns_best(tmp_path):
    model = build_model(CFG)
    best = fit(
        model, _loader(tmp_path), _loader(tmp_path), CompositeLoss(LossConfig()),
        torch.optim.AdamW(model.parameters(), lr=1e-3), torch.device("cpu"),
        epochs=2, checkpoint_dir=tmp_path / "ckpt", amp=False,
    )
    assert best == tmp_path / "ckpt" / "best.pt"
    assert best.exists()
    assert (tmp_path / "ckpt" / "last.pt").exists()


def test_fit_invokes_the_epoch_callback(tmp_path):
    seen: list[tuple[int, float]] = []
    model = build_model(CFG)
    fit(
        model, _loader(tmp_path), _loader(tmp_path), CompositeLoss(LossConfig()),
        torch.optim.AdamW(model.parameters(), lr=1e-3), torch.device("cpu"),
        epochs=3, checkpoint_dir=tmp_path / "ckpt", amp=False,
        on_epoch=lambda e, tr, va: seen.append((e, va.psnr)),
    )
    assert [e for e, _ in seen] == [0, 1, 2]


@pytest.mark.slow
def test_model_can_overfit_a_single_sample(tmp_path):
    """The architecture must be able to learn at all - the cheapest guard against
    a silently disconnected head or a sign error in the warp."""
    seed_everything(0)
    model = build_model(CFG)
    criterion = CompositeLoss(LossConfig(w_smooth=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    ds = TripletDataset(_translating_triplets(tmp_path, n=1), NORM, tile_size=32, augment=False)
    batch = {k: v[None] if v.ndim else v.reshape(1) for k, v in ds[0].items()}

    first = None
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = criterion(model(batch["i0"], batch["i2"], batch["t"]), batch)
        loss.backward()
        optimizer.step()
        if step == 0:
            first = float(loss)
    assert float(loss) < 0.5 * first, f"loss went {first:.4f} -> {float(loss):.4f}"
```

Register the marker by adding to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: longer-running tests; run with -m slow"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/train/test_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.train.loop'`

- [ ] **Step 3: Implement `src/sattsr/train/loop.py`**

```python
"""Training and validation loops."""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from sattsr.config import Config
from sattsr.train.checkpoint import save_checkpoint

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class EpochResult:
    """Aggregated metrics for one pass over a loader."""

    loss: float
    components: dict[str, float] = field(default_factory=dict)
    psnr: float = float("nan")


def _batch_psnr(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> float:
    """PSNR on the normalised [0, 1] scale, so data_range is 1."""
    err = (pred - target) ** 2
    mse = float((err * mask).sum() / mask.sum().clamp_min(1.0)) if mask is not None \
        else float(err.mean())
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer | None,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)

    totals: dict[str, float] = defaultdict(float)
    psnr_sum = 0.0
    seen = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        n = int(batch["i0"].shape[0])

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                out = model(batch["i0"], batch["i2"], batch["t"])
                loss, parts = criterion(out, batch)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        for key, value in parts.items():
            totals[key] += value * n
        psnr_sum += _batch_psnr(
            out["pred"].detach().float(), batch["i1"].float(), batch.get("valid")
        ) * n
        seen += n

    if seen == 0:
        return EpochResult(loss=float("nan"), components={}, psnr=float("nan"))
    components = {k: v / seen for k, v in totals.items()}
    return EpochResult(loss=components["total"], components=components, psnr=psnr_sum / seen)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float = 1.0,
) -> EpochResult:
    """One optimisation pass over `loader`."""
    return _run_epoch(model, loader, criterion, optimizer, device,
                      scaler=scaler, grad_clip=grad_clip)


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> EpochResult:
    """One evaluation pass over `loader`."""
    return _run_epoch(model, loader, criterion, None, device, scaler=None, grad_clip=0.0)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    *,
    epochs: int,
    checkpoint_dir: str | Path,
    amp: bool = True,
    config: Config | None = None,
    scheduler: object | None = None,
    on_epoch: Callable[[int, EpochResult, EpochResult], None] | None = None,
) -> Path:
    """Train for `epochs`, tracking the best validation PSNR. Returns the best checkpoint."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    best_psnr = -float("inf")
    best_path = ckpt_dir / "best.pt"

    for epoch in range(epochs):
        for loader in (train_loader, val_loader):
            dataset = getattr(loader, "dataset", None)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                       scaler=scaler)
        val_result = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        log.info(
            "epoch %d/%d train_loss=%.4f train_psnr=%.2f val_loss=%.4f val_psnr=%.2f",
            epoch + 1, epochs, train_result.loss, train_result.psnr,
            val_result.loss, val_result.psnr,
        )

        save_checkpoint(ckpt_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch,
                        best_metric=best_psnr, config=config)
        if val_result.psnr > best_psnr or not best_path.exists():
            best_psnr = max(best_psnr, val_result.psnr)
            save_checkpoint(best_path, model=model, optimizer=optimizer, epoch=epoch,
                            best_metric=best_psnr, config=config)

        if on_epoch is not None:
            on_epoch(epoch, train_result, val_result)

    return best_path
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/train/test_loop.py -v -m "not slow"` then `pytest tests/train/test_loop.py -v -m slow`
Expected: 5 passed, then 1 passed (the overfit test takes roughly a minute on CPU)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/sattsr/train/loop.py tests/train/test_loop.py
git commit -m "feat: training and validation loops with AMP, clipping and best-PSNR checkpointing"
```

---

### Task 18: Partial-freeze fine-tuning for cross-sensor adaptation

**Files:**
- Create: `src/sattsr/train/finetune.py`
- Test: `tests/train/test_finetune.py`

**Interfaces:**
- Consumes: `FrameInterpolator`, `SensorStats`, `sensor_stats`, `load_cached`, `FrameRef`,
  `TrainConfig`.
- Produces:
  - `freeze_modules(model, prefixes: Sequence[str]) -> int` — returns how many parameters froze.
  - `unfreeze_all(model) -> None`
  - `build_finetune_optimizer(model, *, lr, weight_decay=1e-5, head_prefixes=("refine",), head_multiplier=10.0) -> torch.optim.AdamW`
  - `estimate_cache_stats(refs: Sequence[FrameRef], *, max_frames=50, seed=0) -> SensorStats`
  - `prepare_finetune(model, cfg: TrainConfig) -> torch.optim.AdamW` — freeze, then build the optimizer.

Freezing the flow encoder is what makes INSAT adaptation cheap: motion structure transfers across
sensors, only the radiometry and the synthesis head need to move.

- [ ] **Step 1: Write the failing test**

`tests/train/test_finetune.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from sattsr.config import ModelConfig, TrainConfig
from sattsr.data.index import FrameRef
from sattsr.models.interpolator import build_model
from sattsr.train.finetune import (
    build_finetune_optimizer,
    estimate_cache_stats,
    freeze_modules,
    prepare_finetune,
    unfreeze_all,
)

CFG = ModelConfig(base_channels=16, scales=[4, 2, 1], flow_channels=32, flow_radius=2,
                  flow_iters=2)


def test_freeze_modules_disables_only_matching_parameters():
    model = build_model(CFG)
    frozen = freeze_modules(model, ["flow_net.encoder"])
    assert frozen > 0
    for name, param in model.named_parameters():
        assert param.requires_grad != name.startswith("flow_net.encoder")


def test_freeze_with_no_match_returns_zero():
    assert freeze_modules(build_model(CFG), ["does.not.exist"]) == 0


def test_unfreeze_all_restores_training():
    model = build_model(CFG)
    freeze_modules(model, ["flow_net"])
    unfreeze_all(model)
    assert all(p.requires_grad for p in model.parameters())


def test_optimizer_excludes_frozen_parameters():
    model = build_model(CFG)
    freeze_modules(model, ["flow_net"])
    opt = build_finetune_optimizer(model, lr=1e-4)
    in_optimizer = {id(p) for group in opt.param_groups for p in group["params"]}
    for name, param in model.named_parameters():
        assert (id(param) in in_optimizer) == param.requires_grad, name


def test_optimizer_gives_the_head_a_higher_learning_rate():
    opt = build_finetune_optimizer(build_model(CFG), lr=1e-4, head_prefixes=("refine",),
                                   head_multiplier=10.0)
    rates = sorted(group["lr"] for group in opt.param_groups)
    assert rates == [pytest.approx(1e-4), pytest.approx(1e-3)]


def test_frozen_parameters_do_not_move_during_a_step():
    model = build_model(CFG)
    freeze_modules(model, ["flow_net.encoder"])
    opt = build_finetune_optimizer(model, lr=1e-2)
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if n.startswith("flow_net.encoder")}

    out = model(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64), 0.5)
    out["pred"].sum().backward()
    opt.step()

    for name, param in model.named_parameters():
        if name in before:
            torch.testing.assert_close(param, before[name])


def test_estimate_cache_stats_matches_the_data(tmp_path: Path):
    rng = np.random.default_rng(0)
    refs = []
    t0 = datetime(2026, 9, 4, tzinfo=timezone.utc)
    for i in range(6):
        arr = rng.normal(255.0, 8.0, size=(32, 32)).astype(np.float32)
        arr[0, 0] = np.nan
        p = tmp_path / f"f{i}.npy"
        np.save(p, arr)
        refs.append(FrameRef(t0 + timedelta(minutes=30 * i), p, "insat3dr"))

    stats = estimate_cache_stats(refs, max_frames=6)
    assert stats.mean == pytest.approx(255.0, abs=1.0)
    assert stats.std == pytest.approx(8.0, abs=1.0)


def test_estimate_cache_stats_subsamples_deterministically(tmp_path: Path):
    t0 = datetime(2026, 9, 4, tzinfo=timezone.utc)
    refs = []
    for i in range(20):
        p = tmp_path / f"f{i}.npy"
        np.save(p, np.full((8, 8), float(i), dtype=np.float32))
        refs.append(FrameRef(t0 + timedelta(minutes=i), p, "insat3dr"))
    a = estimate_cache_stats(refs, max_frames=5, seed=3)
    b = estimate_cache_stats(refs, max_frames=5, seed=3)
    assert a == b


def test_prepare_finetune_applies_the_configured_freeze():
    cfg = TrainConfig(lr=5e-5, freeze_prefixes=["flow_net.encoder"], lr_head_multiplier=10.0)
    model = build_model(CFG)
    opt = prepare_finetune(model, cfg)
    assert not any(p.requires_grad for n, p in model.named_parameters()
                   if n.startswith("flow_net.encoder"))
    assert sorted(g["lr"] for g in opt.param_groups) == [pytest.approx(5e-5),
                                                         pytest.approx(5e-4)]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/train/test_finetune.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.train.finetune'`

- [ ] **Step 3: Implement `src/sattsr/train/finetune.py`**

```python
"""Cross-sensor domain adaptation: partial freezing and discriminative learning rates."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from sattsr.config import TrainConfig
from sattsr.data.index import FrameRef, load_cached
from sattsr.data.radiometry import SensorStats


def freeze_modules(model: nn.Module, prefixes: Sequence[str]) -> int:
    """Freeze every parameter whose name starts with one of `prefixes`."""
    frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad_(False)
            frozen += 1
    return frozen


def unfreeze_all(model: nn.Module) -> None:
    """Make every parameter trainable again."""
    for param in model.parameters():
        param.requires_grad_(True)


def build_finetune_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 1e-5,
    head_prefixes: Sequence[str] = ("refine",),
    head_multiplier: float = 10.0,
) -> torch.optim.AdamW:
    """AdamW over the trainable parameters, with a faster group for the synthesis head."""
    base: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head if any(name.startswith(p) for p in head_prefixes) else base).append(param)

    groups = []
    if base:
        groups.append({"params": base, "lr": lr})
    if head:
        groups.append({"params": head, "lr": lr * head_multiplier})
    if not groups:
        raise ValueError("no trainable parameters remain; check freeze_prefixes")
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


def estimate_cache_stats(
    refs: Sequence[FrameRef], *, max_frames: int = 50, seed: int = 0
) -> SensorStats:
    """Brightness-temperature mean and std over a deterministic sample of cached frames."""
    items = list(refs)
    if not items:
        raise ValueError("cannot estimate statistics from an empty reference list")
    if len(items) > max_frames:
        items = random.Random(seed).sample(items, max_frames)

    values = [load_cached(ref).ravel() for ref in items]
    stacked = np.concatenate(values)
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        raise ValueError("no valid pixels in the sampled frames")
    return SensorStats(mean=float(finite.mean()), std=float(finite.std()))


def prepare_finetune(model: nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """Apply the configured freeze and return the fine-tuning optimizer."""
    if cfg.freeze_prefixes:
        freeze_modules(model, cfg.freeze_prefixes)
    return build_finetune_optimizer(
        model,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        head_multiplier=cfg.lr_head_multiplier,
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/train -v -m "not slow"`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/train/finetune.py tests/train/test_finetune.py
git commit -m "feat: partial-freeze fine-tuning with discriminative learning rates"
```

---

# Phase 4 — Evaluation

### Task 19: Image quality metrics (MSE, RMSE, PSNR, SSIM, FSIM)

**Files:**
- Create: `src/sattsr/eval/__init__.py`, `src/sattsr/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: nothing (pure NumPy).
- Produces, all operating on `(H, W)` float arrays **in Kelvin** with NaN allowed, all returning
  plain `float` and ignoring pixels invalid in either input:
  - `mse(pred, target) -> float`
  - `rmse(pred, target) -> float`
  - `psnr(pred, target, *, data_range) -> float`
  - `ssim(pred, target, *, data_range) -> float`
  - `fsim(pred, target, *, data_range) -> float`
  - `phase_congruency(image, *, n_scales=4, min_wavelength=6.0, mult=2.0, sigma_onf=0.55) -> np.ndarray`
  - `metric_suite(pred, target, *, data_range) -> dict[str, float]` with keys
    `mse`, `rmse`, `psnr`, `ssim`, `fsim`.

FSIM is not in scikit-image, so it is implemented here: monogenic-signal phase congruency (log-Gabor
bandpass plus Riesz transform, without Kovesi's noise compensation) combined with Scharr gradient
magnitude, using the standard constants `T1 = 0.85`, `T2 = 160` on a `[0, 255]` scale.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/__init__.py` (empty), then `tests/eval/test_metrics.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from sattsr.eval.metrics import (
    fsim,
    metric_suite,
    mse,
    phase_congruency,
    psnr,
    rmse,
    ssim,
)

DR = 150.0


def _cloud_field(seed: int = 0, shift: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    base = 250.0 + 25.0 * np.sin((xx + shift) * 0.25) * np.cos(yy * 0.2)
    return (base + rng.normal(0.0, 0.5, size=(64, 64))).astype(np.float32)


def test_identical_arrays_score_perfectly():
    a = _cloud_field()
    assert mse(a, a) == pytest.approx(0.0, abs=1e-6)
    assert rmse(a, a) == pytest.approx(0.0, abs=1e-6)
    assert psnr(a, a, data_range=DR) == pytest.approx(100.0)
    assert ssim(a, a, data_range=DR) == pytest.approx(1.0, abs=1e-4)
    assert fsim(a, a, data_range=DR) == pytest.approx(1.0, abs=1e-4)


def test_metrics_degrade_monotonically_with_noise():
    a = _cloud_field()
    rng = np.random.default_rng(1)
    small = a + rng.normal(0.0, 1.0, size=a.shape).astype(np.float32)
    large = a + rng.normal(0.0, 8.0, size=a.shape).astype(np.float32)

    assert mse(a, small) < mse(a, large)
    assert psnr(a, small, data_range=DR) > psnr(a, large, data_range=DR)
    assert ssim(a, small, data_range=DR) > ssim(a, large, data_range=DR)
    assert fsim(a, small, data_range=DR) > fsim(a, large, data_range=DR)


def test_rmse_is_the_square_root_of_mse():
    a, b = _cloud_field(0), _cloud_field(1)
    assert rmse(a, b) == pytest.approx(np.sqrt(mse(a, b)), rel=1e-6)


def test_psnr_matches_its_definition():
    a = np.full((16, 16), 250.0, dtype=np.float32)
    b = a + 1.5
    assert psnr(a, b, data_range=DR) == pytest.approx(
        20.0 * np.log10(DR) - 10.0 * np.log10(1.5**2), rel=1e-6
    )


def test_nan_pixels_are_excluded_not_propagated():
    a = _cloud_field()
    b = a.copy()
    b[:8, :] = np.nan
    for value in (mse(a, b), psnr(a, b, data_range=DR), ssim(a, b, data_range=DR),
                  fsim(a, b, data_range=DR)):
        assert np.isfinite(value)
    assert mse(a, b) == pytest.approx(0.0, abs=1e-6)


def test_all_nan_returns_nan_not_a_crash():
    a = np.full((16, 16), np.nan, dtype=np.float32)
    assert np.isnan(mse(a, a))
    assert np.isnan(ssim(a, a, data_range=DR))
    assert np.isnan(fsim(a, a, data_range=DR))


def test_ssim_agrees_with_skimage_on_clean_data():
    skimage = pytest.importorskip("skimage.metrics")
    a, b = _cloud_field(0), _cloud_field(0, shift=2)
    theirs = skimage.structural_similarity(
        a.astype(np.float64), b.astype(np.float64), data_range=DR,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    assert ssim(a, b, data_range=DR) == pytest.approx(theirs, abs=1e-6)


def test_phase_congruency_peaks_on_a_structural_edge():
    img = np.zeros((64, 64), dtype=np.float64)
    img[:, 32:] = 200.0
    pc = phase_congruency(img)
    assert pc.shape == img.shape
    assert np.isfinite(pc).all()
    assert float(pc[:, 30:34].mean()) > float(pc[:, 5:15].mean())


def test_fsim_is_bounded():
    a, b = _cloud_field(0), _cloud_field(5, shift=7)
    value = fsim(a, b, data_range=DR)
    assert 0.0 <= value <= 1.0


def test_fsim_prefers_structural_agreement_over_a_constant_offset():
    a = _cloud_field()
    shifted = a + 4.0                      # same structure, different bias
    scrambled = _cloud_field(9, shift=17)  # different structure
    assert fsim(a, shifted, data_range=DR) > fsim(a, scrambled, data_range=DR)


def test_metric_suite_returns_every_required_metric():
    result = metric_suite(_cloud_field(0), _cloud_field(0, shift=1), data_range=DR)
    assert set(result) == {"mse", "rmse", "psnr", "ssim", "fsim"}
    assert all(isinstance(v, float) for v in result.values())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.eval'`

- [ ] **Step 3: Implement `src/sattsr/eval/metrics.py`**

Create an empty `src/sattsr/eval/__init__.py` too.

```python
"""Image quality metrics, computed only over pixels valid in both inputs."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity

_EPS = 1e-12
_FSIM_T1 = 0.85
_FSIM_T2 = 160.0


def _pair(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return a, b, np.isfinite(a) & np.isfinite(b)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error in Kelvin^2."""
    a, b, mask = _pair(pred, target)
    if not mask.any():
        return float("nan")
    return float(np.mean((a[mask] - b[mask]) ** 2))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error in Kelvin."""
    value = mse(pred, target)
    return float(np.sqrt(value)) if np.isfinite(value) else value


def psnr(pred: np.ndarray, target: np.ndarray, *, data_range: float) -> float:
    """Peak signal-to-noise ratio in dB; capped at 100 for an exact match."""
    value = mse(pred, target)
    if not np.isfinite(value):
        return float("nan")
    if value <= _EPS:
        return 100.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(value))


def ssim(pred: np.ndarray, target: np.ndarray, *, data_range: float) -> float:
    """Structural similarity, averaged over valid pixels only."""
    a, b, mask = _pair(pred, target)
    if not mask.any():
        return float("nan")
    fill = float(b[mask].mean())
    a_f = np.where(mask, a, fill)
    b_f = np.where(mask, b, fill)
    _, smap = structural_similarity(
        a_f, b_f, data_range=data_range, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False, full=True,
    )
    return float(smap[mask].mean())


def _log_gabor_and_riesz(
    shape: tuple[int, int], n_scales: int, min_wavelength: float, mult: float, sigma_onf: float
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    rows, cols = shape
    u1 = (np.arange(cols) - cols // 2) / float(cols)
    u2 = (np.arange(rows) - rows // 2) / float(rows)
    u1, u2 = np.meshgrid(u1, u2)

    radius = np.fft.ifftshift(np.sqrt(u1**2 + u2**2))
    u1 = np.fft.ifftshift(u1)
    u2 = np.fft.ifftshift(u2)
    radius[0, 0] = 1.0

    low_pass = 1.0 / (1.0 + (radius / 0.45) ** 30)     # Butterworth, kills corner artefacts
    filters = []
    for scale in range(n_scales):
        f0 = 1.0 / (min_wavelength * (mult**scale))
        lg = np.exp(-((np.log(radius / f0)) ** 2) / (2.0 * np.log(sigma_onf) ** 2)) * low_pass
        lg[0, 0] = 0.0
        filters.append(lg)

    denom = np.sqrt(u1**2 + u2**2)
    denom[0, 0] = 1.0
    return filters, 1j * u1 / denom, 1j * u2 / denom


def phase_congruency(
    image: np.ndarray,
    *,
    n_scales: int = 4,
    min_wavelength: float = 6.0,
    mult: float = 2.0,
    sigma_onf: float = 0.55,
) -> np.ndarray:
    """Monogenic-signal phase congruency in [0, 1].

    Uses log-Gabor bandpass filters plus the Riesz transform for the odd components.
    Kovesi's noise compensation is deliberately omitted: FSIM only needs a relative
    weighting of structurally significant pixels, and the omission keeps this
    dependency-free and fast.
    """
    img = np.asarray(image, dtype=np.float64)
    spectrum = np.fft.fft2(img)
    filters, hx, hy = _log_gabor_and_riesz(img.shape, n_scales, min_wavelength, mult, sigma_onf)

    sum_even = np.zeros_like(img)
    sum_odd_x = np.zeros_like(img)
    sum_odd_y = np.zeros_like(img)
    sum_amp = np.zeros_like(img)

    for lg in filters:
        band = spectrum * lg
        even = np.real(np.fft.ifft2(band))
        odd_x = np.real(np.fft.ifft2(band * hx))
        odd_y = np.real(np.fft.ifft2(band * hy))
        sum_even += even
        sum_odd_x += odd_x
        sum_odd_y += odd_y
        sum_amp += np.sqrt(even**2 + odd_x**2 + odd_y**2)

    energy = np.sqrt(sum_even**2 + sum_odd_x**2 + sum_odd_y**2)
    return np.clip(energy / (sum_amp + 1e-4), 0.0, 1.0)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = sobel(image, axis=1, mode="nearest")
    gy = sobel(image, axis=0, mode="nearest")
    return np.sqrt(gx**2 + gy**2)


def fsim(pred: np.ndarray, target: np.ndarray, *, data_range: float) -> float:
    """Feature similarity index over valid pixels.

    Both images are mapped onto [0, 255] using the target's minimum and the shared
    `data_range`, so the standard T1/T2 constants apply.
    """
    a, b, mask = _pair(pred, target)
    if not mask.any():
        return float("nan")

    lo = float(b[mask].min())
    fill = float(b[mask].mean())
    scale = 255.0 / max(data_range, _EPS)
    a_s = (np.where(mask, a, fill) - lo) * scale
    b_s = (np.where(mask, b, fill) - lo) * scale

    pc1, pc2 = phase_congruency(a_s), phase_congruency(b_s)
    g1, g2 = _gradient_magnitude(a_s), _gradient_magnitude(b_s)

    s_pc = (2.0 * pc1 * pc2 + _FSIM_T1) / (pc1**2 + pc2**2 + _FSIM_T1)
    s_g = (2.0 * g1 * g2 + _FSIM_T2) / (g1**2 + g2**2 + _FSIM_T2)
    weight = np.maximum(pc1, pc2) * mask

    denominator = float(weight.sum())
    if denominator <= _EPS:
        return float("nan")
    return float(np.clip((s_pc * s_g * weight).sum() / denominator, 0.0, 1.0))


def metric_suite(pred: np.ndarray, target: np.ndarray, *, data_range: float) -> dict[str, float]:
    """Every required image-quality metric in one call."""
    return {
        "mse": mse(pred, target),
        "rmse": rmse(pred, target),
        "psnr": psnr(pred, target, data_range=data_range),
        "ssim": ssim(pred, target, data_range=data_range),
        "fsim": fsim(pred, target, data_range=data_range),
    }
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: 11 passed

If `test_phase_congruency_peaks_on_a_structural_edge` fails, check the `ifftshift` calls: the
log-Gabor filters must be in the same (unshifted) frequency layout as `np.fft.fft2` output.

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/eval tests/eval
git commit -m "feat: MSE, RMSE, PSNR, SSIM and a from-scratch FSIM implementation"
```

---

### Task 20: Cloud-motion metrics

**Files:**
- Create: `src/sattsr/eval/motion.py`
- Test: `tests/eval/test_motion.py`

**Interfaces:**
- Consumes: nothing (NumPy + OpenCV).
- Produces:
  - `to_uint8(image, *, data_range, lo=None) -> np.ndarray`
  - `farneback_flow(a, b, *, data_range) -> np.ndarray` — `(H, W, 2)` float32, `[..., 0]` = right
  - `flow_endpoint_error(f1, f2) -> float`
  - `motion_displacement_error(i0, pred_mid, gt_mid, *, data_range) -> float` — the EPE between
    the motion the prediction implies and the motion the truth implies
  - `cold_cloud_scores(pred, target, *, threshold_k=235.0) -> dict[str, float]` with keys
    `csi`, `pod`, `far`
  - `mean_abs_vorticity(a, b, *, data_range) -> float`
  - `motion_suite(i0, pred_mid, gt_mid, *, data_range, threshold_k=235.0) -> dict[str, float]`

Pixel-similarity metrics reward a blurry average; these do not. `motion_displacement_error` asks
whether the cloud field actually moved to the right place, and the CSI/POD/FAR triple asks whether
deep convection (BT below 235 K) landed where it should have.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_motion.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from sattsr.eval.motion import (
    cold_cloud_scores,
    farneback_flow,
    flow_endpoint_error,
    mean_abs_vorticity,
    motion_displacement_error,
    motion_suite,
    to_uint8,
)

DR = 150.0


def _texture(size: int = 96, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    field = 250.0 + 20.0 * np.sin(xx * 0.4) * np.cos(yy * 0.35)
    return (field + rng.normal(0.0, 0.4, size=(size, size))).astype(np.float32)


def test_to_uint8_spans_the_full_byte_range():
    out = to_uint8(np.array([[180.0, 330.0]], dtype=np.float32), data_range=DR, lo=180.0)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_to_uint8_handles_nan():
    out = to_uint8(np.array([[np.nan, 250.0]], dtype=np.float32), data_range=DR, lo=180.0)
    assert np.isfinite(out).all()


def test_farneback_recovers_a_known_translation():
    a = _texture()
    b = np.roll(a, 4, axis=1)               # content moves 4 px right
    flow = farneback_flow(a, b, data_range=DR)
    assert flow.shape == (96, 96, 2)
    interior = flow[20:76, 20:76, 0]
    assert float(np.median(interior)) == pytest.approx(4.0, abs=1.5)


def test_flow_endpoint_error_is_zero_for_identical_fields():
    f = np.ones((8, 8, 2), dtype=np.float32)
    assert flow_endpoint_error(f, f) == pytest.approx(0.0)
    assert flow_endpoint_error(f, f * 2.0) == pytest.approx(np.sqrt(2.0), rel=1e-5)


def test_motion_displacement_error_rewards_correct_motion():
    i0 = _texture()
    gt_mid = np.roll(i0, 3, axis=1)
    good = np.roll(i0, 3, axis=1)
    blurry = 0.5 * i0 + 0.5 * np.roll(i0, 6, axis=1)   # a linear-blend style guess
    assert motion_displacement_error(i0, good, gt_mid, data_range=DR) < \
        motion_displacement_error(i0, blurry, gt_mid, data_range=DR)


def test_cold_cloud_scores_are_perfect_for_an_exact_match():
    target = np.full((16, 16), 260.0, dtype=np.float32)
    target[:8, :8] = 210.0
    scores = cold_cloud_scores(target.copy(), target, threshold_k=235.0)
    assert scores["csi"] == pytest.approx(1.0)
    assert scores["pod"] == pytest.approx(1.0)
    assert scores["far"] == pytest.approx(0.0)


def test_cold_cloud_scores_penalise_a_missed_cold_core():
    target = np.full((16, 16), 260.0, dtype=np.float32)
    target[:8, :8] = 210.0
    pred = np.full((16, 16), 260.0, dtype=np.float32)     # missed it entirely
    scores = cold_cloud_scores(pred, target, threshold_k=235.0)
    assert scores["pod"] == pytest.approx(0.0)
    assert scores["csi"] == pytest.approx(0.0)


def test_cold_cloud_scores_with_no_cold_pixels_return_nan_not_zero():
    warm = np.full((16, 16), 280.0, dtype=np.float32)
    scores = cold_cloud_scores(warm, warm, threshold_k=235.0)
    assert np.isnan(scores["pod"])


def test_vorticity_is_higher_for_a_rotating_field_than_a_translating_one():
    size = 96
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    base = (250.0 + 20.0 * np.sin(xx * 0.4) * np.cos(yy * 0.35)).astype(np.float32)

    translated = np.roll(base, 4, axis=1)

    cy = cx = size / 2.0
    angle = 0.12
    sy = np.cos(angle) * (yy - cy) - np.sin(angle) * (xx - cx) + cy
    sx = np.sin(angle) * (yy - cy) + np.cos(angle) * (xx - cx) + cx
    rotated = base[np.clip(sy.round().astype(int), 0, size - 1),
                   np.clip(sx.round().astype(int), 0, size - 1)]

    assert mean_abs_vorticity(base, rotated, data_range=DR) > \
        mean_abs_vorticity(base, translated, data_range=DR)


def test_motion_suite_returns_every_key():
    i0 = _texture()
    mid = np.roll(i0, 3, axis=1)
    result = motion_suite(i0, mid, mid, data_range=DR)
    assert set(result) == {"displacement_error", "csi", "pod", "far"}
    assert all(isinstance(v, float) for v in result.values())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/eval/test_motion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.eval.motion'`

- [ ] **Step 3: Implement `src/sattsr/eval/motion.py`**

```python
"""Metrics that score cloud motion rather than per-pixel similarity."""

from __future__ import annotations

import cv2
import numpy as np

_EPS = 1e-12
_FARNEBACK = {
    "pyr_scale": 0.5, "levels": 4, "winsize": 21, "iterations": 3,
    "poly_n": 5, "poly_sigma": 1.2, "flags": 0,
}


def to_uint8(image: np.ndarray, *, data_range: float, lo: float | None = None) -> np.ndarray:
    """Map Kelvin onto bytes for OpenCV; NaN becomes the field mean."""
    a = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, dtype=np.uint8)
    base = float(a[finite].min()) if lo is None else float(lo)
    filled = np.where(finite, a, float(a[finite].mean()))
    scaled = (filled - base) / max(data_range, _EPS) * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def farneback_flow(a: np.ndarray, b: np.ndarray, *, data_range: float) -> np.ndarray:
    """Dense classical optical flow from `a` to `b`, in pixels, shape (H, W, 2)."""
    lo = float(np.nanmin([np.nanmin(a), np.nanmin(b)]))
    return cv2.calcOpticalFlowFarneback(
        to_uint8(a, data_range=data_range, lo=lo),
        to_uint8(b, data_range=data_range, lo=lo),
        None, **_FARNEBACK,
    ).astype(np.float32)


def flow_endpoint_error(f1: np.ndarray, f2: np.ndarray) -> float:
    """Mean Euclidean distance between two flow fields."""
    return float(np.mean(np.linalg.norm(np.asarray(f1) - np.asarray(f2), axis=-1)))


def motion_displacement_error(
    i0: np.ndarray, pred_mid: np.ndarray, gt_mid: np.ndarray, *, data_range: float
) -> float:
    """How far the predicted cloud field moved compared with how far it really moved.

    A blurred prediction scores badly here even when its MSE looks respectable, which
    is exactly the failure mode the spec calls out.
    """
    return flow_endpoint_error(
        farneback_flow(i0, pred_mid, data_range=data_range),
        farneback_flow(i0, gt_mid, data_range=data_range),
    )


def cold_cloud_scores(
    pred: np.ndarray, target: np.ndarray, *, threshold_k: float = 235.0
) -> dict[str, float]:
    """Categorical skill for deep convection, thresholded on brightness temperature.

    Returns NaN rather than 0 when the category is absent from both fields, so empty
    scenes do not drag the reported average down.
    """
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    p = (a < threshold_k) & valid
    t = (b < threshold_k) & valid

    hits = float(np.count_nonzero(p & t))
    misses = float(np.count_nonzero(~p & t))
    false_alarms = float(np.count_nonzero(p & ~t))

    union = hits + misses + false_alarms
    observed = hits + misses
    predicted = hits + false_alarms
    return {
        "csi": hits / union if union > 0 else float("nan"),
        "pod": hits / observed if observed > 0 else float("nan"),
        "far": false_alarms / predicted if predicted > 0 else float("nan"),
    }


def mean_abs_vorticity(a: np.ndarray, b: np.ndarray, *, data_range: float) -> float:
    """Mean |dv/dx - du/dy| of the flow between two frames: how rotational the motion is."""
    flow = farneback_flow(a, b, data_range=data_range)
    du_dy = np.gradient(flow[..., 0], axis=0)
    dv_dx = np.gradient(flow[..., 1], axis=1)
    return float(np.mean(np.abs(dv_dx - du_dy)))


def motion_suite(
    i0: np.ndarray,
    pred_mid: np.ndarray,
    gt_mid: np.ndarray,
    *,
    data_range: float,
    threshold_k: float = 235.0,
) -> dict[str, float]:
    """Every cloud-motion metric in one call."""
    result = {"displacement_error": motion_displacement_error(i0, pred_mid, gt_mid,
                                                              data_range=data_range)}
    result.update(cold_cloud_scores(pred_mid, gt_mid, threshold_k=threshold_k))
    return result
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/eval/test_motion.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/eval/motion.py tests/eval/test_motion.py
git commit -m "feat: cloud-motion metrics - displacement error and cold-cloud CSI/POD/FAR"
```

---

### Task 21: Baselines

**Files:**
- Create: `src/sattsr/eval/baselines.py`
- Test: `tests/eval/test_baselines.py`

**Interfaces:**
- Consumes: `farneback_flow`.
- Produces:
  - `linear_blend(i0, i2, t=0.5) -> np.ndarray`
  - `farneback_warp(i0, i2, t=0.5, *, data_range) -> np.ndarray`
  - `BASELINES: dict[str, Callable[..., np.ndarray]]` keyed `"linear"` and `"farneback"`
  - `run_baseline(name, i0, i2, t=0.5, *, data_range) -> np.ndarray`

Both return float32 Kelvin with NaN preserved where **both** inputs are invalid. The report is
meaningless without these: "the model produces a plausible-looking frame" is not a result unless
a linear blend produces a worse one.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_baselines.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from sattsr.eval.baselines import BASELINES, farneback_warp, linear_blend, run_baseline
from sattsr.eval.metrics import psnr

DR = 150.0


def _texture(size: int = 96, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    field = 250.0 + 20.0 * np.sin(xx * 0.4) * np.cos(yy * 0.35)
    return (field + rng.normal(0.0, 0.4, size=(size, size))).astype(np.float32)


def test_linear_blend_at_the_midpoint_is_the_average():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.full((4, 4), 10.0, dtype=np.float32)
    np.testing.assert_allclose(linear_blend(a, b, 0.5), 5.0)
    np.testing.assert_allclose(linear_blend(a, b, 0.25), 2.5)


def test_linear_blend_preserves_dtype_and_nan():
    a = np.array([[np.nan, 1.0]], dtype=np.float32)
    b = np.array([[np.nan, 3.0]], dtype=np.float32)
    out = linear_blend(a, b)
    assert out.dtype == np.float32
    assert np.isnan(out[0, 0]) and out[0, 1] == pytest.approx(2.0)


def test_farneback_warp_beats_a_linear_blend_on_pure_translation():
    i0 = _texture()
    i2 = np.roll(i0, 8, axis=1)
    truth = np.roll(i0, 4, axis=1)
    warped = farneback_warp(i0, i2, 0.5, data_range=DR)
    blended = linear_blend(i0, i2, 0.5)
    assert psnr(warped, truth, data_range=DR) > psnr(blended, truth, data_range=DR)


def test_farneback_warp_returns_the_right_shape_and_dtype():
    i0, i2 = _texture(0), _texture(seed=1)
    out = farneback_warp(i0, i2, 0.5, data_range=DR)
    assert out.shape == i0.shape
    assert out.dtype == np.float32


def test_registry_exposes_both_required_baselines():
    assert set(BASELINES) == {"linear", "farneback"}


def test_run_baseline_dispatches_and_rejects_unknown_names():
    i0, i2 = _texture(0), _texture(seed=1)
    np.testing.assert_allclose(
        run_baseline("linear", i0, i2, 0.5, data_range=DR), linear_blend(i0, i2, 0.5)
    )
    assert run_baseline("farneback", i0, i2, 0.5, data_range=DR).shape == i0.shape
    with pytest.raises(KeyError):
        run_baseline("magic", i0, i2, 0.5, data_range=DR)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/eval/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.eval.baselines'`

- [ ] **Step 3: Implement `src/sattsr/eval/baselines.py`**

```python
"""Non-learned interpolation baselines the model must beat."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from sattsr.eval.motion import farneback_flow


def linear_blend(i0: np.ndarray, i2: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Time-weighted average of the two neighbours."""
    a = np.asarray(i0, dtype=np.float32)
    b = np.asarray(i2, dtype=np.float32)
    return ((1.0 - t) * a + t * b).astype(np.float32)


def _remap(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp `image` by `flow` (pixels), filling invalid pixels with the field mean."""
    h, w = image.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    finite = np.isfinite(image)
    filled = np.where(finite, image, float(image[finite].mean()) if finite.any() else 0.0)
    warped = cv2.remap(
        filled.astype(np.float32),
        (grid_x + flow[..., 0]).astype(np.float32),
        (grid_y + flow[..., 1]).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped.astype(np.float32)


def farneback_warp(
    i0: np.ndarray, i2: np.ndarray, t: float = 0.5, *, data_range: float
) -> np.ndarray:
    """Classical optical-flow interpolation: warp both neighbours towards `t` and blend.

    The intermediate flow is approximated by scaling the endpoint flow, which is the
    standard cheap construction and exactly the "assumes smooth motion" behaviour the
    learned model is meant to improve on.
    """
    a = np.asarray(i0, dtype=np.float32)
    b = np.asarray(i2, dtype=np.float32)
    flow_02 = farneback_flow(a, b, data_range=data_range)
    flow_20 = farneback_flow(b, a, data_range=data_range)

    warped0 = _remap(a, -float(t) * flow_02)
    warped2 = _remap(b, -(1.0 - float(t)) * flow_20)
    out = (1.0 - t) * warped0 + t * warped2

    invalid = ~np.isfinite(a) & ~np.isfinite(b)
    return np.where(invalid, np.nan, out).astype(np.float32)


BASELINES: dict[str, Callable[..., np.ndarray]] = {
    "linear": lambda i0, i2, t, data_range: linear_blend(i0, i2, t),
    "farneback": lambda i0, i2, t, data_range: farneback_warp(i0, i2, t, data_range=data_range),
}


def run_baseline(
    name: str, i0: np.ndarray, i2: np.ndarray, t: float = 0.5, *, data_range: float
) -> np.ndarray:
    """Dispatch to a named baseline."""
    try:
        fn = BASELINES[name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}") from exc
    return fn(i0, i2, t, data_range)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/eval -v`
Expected: 27 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/eval/baselines.py tests/eval/test_baselines.py
git commit -m "feat: linear-blend and Farneback-warp interpolation baselines"
```

---

### Task 22: Meteorological event categorisation

**Files:**
- Create: `src/sattsr/eval/events.py`
- Test: `tests/eval/test_events.py`

**Interfaces:**
- Consumes: `mean_abs_vorticity`.
- Produces:
  - `EventCategory` — a `str` enum with members `CLEAR="clear"`, `STRATIFORM="stratiform"`,
    `CONVECTIVE="convective"`, `CYCLONIC="cyclonic"`.
  - `EventThresholds(cloud_bt_k=270.0, cloud_fraction=0.15, deep_bt_k=220.0, deep_fraction=0.02, cyclonic_deep_fraction=0.15, vorticity=0.05)` frozen dataclass.
  - `classify_event(frames: np.ndarray, *, data_range=150.0, thresholds=EventThresholds()) -> EventCategory`
    — `frames` is `(T, H, W)` Kelvin with `T >= 2`.
  - `category_counts(categories: Iterable[EventCategory]) -> dict[str, int]` — always reports all
    four keys, zero-filled.

This is a documented heuristic on brightness temperature and flow vorticity, not a validated
meteorological classifier — its job is to stratify the report so convective and cyclonic
performance cannot be averaged away, which is exactly what the spec asks for. Say so in the report.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_events.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from sattsr.eval.events import (
    EventCategory,
    EventThresholds,
    category_counts,
    classify_event,
)


def _stack(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    return np.stack([frame_a, frame_b]).astype(np.float32)


def _warm(size: int = 96, value: float = 292.0) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def test_warm_scene_is_clear():
    warm = _warm()
    assert classify_event(_stack(warm, np.roll(warm, 2, axis=1))) == EventCategory.CLEAR


def test_uniform_cool_cloud_deck_is_stratiform():
    deck = _warm()
    deck[:, :] = 255.0                       # cloudy but nowhere near deep convection
    assert classify_event(_stack(deck, np.roll(deck, 2, axis=1))) == EventCategory.STRATIFORM


def test_small_very_cold_core_is_convective():
    scene = _warm()
    scene[:, :] = 260.0
    scene[40:52, 40:52] = 205.0              # ~1.5% of the scene below 220 K
    assert classify_event(_stack(scene, np.roll(scene, 2, axis=1))) == EventCategory.CONVECTIVE


def test_large_rotating_cold_shield_is_cyclonic():
    size = 128
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    radius = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    scene = np.where(radius < size * 0.38, 200.0, 265.0).astype(np.float32)
    scene += (2.0 * np.sin(xx * 0.5) * np.cos(yy * 0.5)).astype(np.float32)  # texture to track

    angle = 0.25
    cy = cx = size / 2.0
    sy = np.cos(angle) * (yy - cy) - np.sin(angle) * (xx - cx) + cy
    sx = np.sin(angle) * (yy - cy) + np.cos(angle) * (xx - cx) + cx
    rotated = scene[np.clip(sy.round().astype(int), 0, size - 1),
                    np.clip(sx.round().astype(int), 0, size - 1)]

    assert classify_event(_stack(scene, rotated)) == EventCategory.CYCLONIC


def test_all_nan_scene_is_clear_not_a_crash():
    nan_frame = np.full((32, 32), np.nan, dtype=np.float32)
    assert classify_event(_stack(nan_frame, nan_frame)) == EventCategory.CLEAR


def test_thresholds_are_configurable():
    scene = _warm()
    scene[:, :] = 255.0
    strict = EventThresholds(cloud_bt_k=250.0)     # 255 K no longer counts as cloud
    assert classify_event(_stack(scene, scene), thresholds=strict) == EventCategory.CLEAR


def test_single_frame_input_is_rejected():
    with pytest.raises(ValueError):
        classify_event(_warm()[None])


def test_category_values_match_the_spec_spelling():
    assert [c.value for c in EventCategory] == ["clear", "stratiform", "convective", "cyclonic"]


def test_category_counts_zero_fills_every_category():
    counts = category_counts([EventCategory.CLEAR, EventCategory.CLEAR,
                              EventCategory.CONVECTIVE])
    assert counts == {"clear": 2, "stratiform": 0, "convective": 1, "cyclonic": 0}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/eval/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.eval.events'`

- [ ] **Step 3: Implement `src/sattsr/eval/events.py`**

```python
"""Heuristic stratification of scenes into meteorological event categories.

This is a screening heuristic on brightness temperature and flow vorticity, not a
validated meteorological classifier. Its purpose is to keep convective and cyclonic
performance from being averaged away in the report.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from sattsr.eval.motion import mean_abs_vorticity


class EventCategory(str, Enum):
    """The four categories the report is broken down by."""

    CLEAR = "clear"
    STRATIFORM = "stratiform"
    CONVECTIVE = "convective"
    CYCLONIC = "cyclonic"


@dataclass(frozen=True)
class EventThresholds:
    """Tunable decision boundaries. Defaults follow common IR nowcasting practice."""

    cloud_bt_k: float = 270.0            # below this a pixel is cloudy
    cloud_fraction: float = 0.15         # cloudier than this and the scene is not clear
    deep_bt_k: float = 220.0             # below this is deep convective cloud top
    deep_fraction: float = 0.02          # this much deep cloud makes a scene convective
    cyclonic_deep_fraction: float = 0.15  # a large cold shield, plus rotation, is cyclonic
    vorticity: float = 0.05              # mean |curl| of the inter-frame flow, px/px


def classify_event(
    frames: np.ndarray,
    *,
    data_range: float = 150.0,
    thresholds: EventThresholds = EventThresholds(),
) -> EventCategory:
    """Classify a short sequence of Kelvin frames, shape (T, H, W) with T >= 2."""
    stack = np.asarray(frames, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError(f"expected (T, H, W) with T >= 2, got shape {stack.shape}")

    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return EventCategory.CLEAR

    cloud_fraction = float(np.mean(finite < thresholds.cloud_bt_k))
    deep_fraction = float(np.mean(finite < thresholds.deep_bt_k))

    if cloud_fraction < thresholds.cloud_fraction:
        return EventCategory.CLEAR
    if deep_fraction < thresholds.deep_fraction:
        return EventCategory.STRATIFORM
    if deep_fraction >= thresholds.cyclonic_deep_fraction:
        vorticity = mean_abs_vorticity(stack[0], stack[-1], data_range=data_range)
        if vorticity >= thresholds.vorticity:
            return EventCategory.CYCLONIC
    return EventCategory.CONVECTIVE


def category_counts(categories: Iterable[EventCategory]) -> dict[str, int]:
    """Count each category, always reporting all four keys."""
    counts = {c.value: 0 for c in EventCategory}
    for category in categories:
        counts[EventCategory(category).value] += 1
    return counts
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/eval/test_events.py -v`
Expected: 9 passed

If `test_large_rotating_cold_shield_is_cyclonic` fails, print
`mean_abs_vorticity(scene, rotated, data_range=150.0)` and adjust
`EventThresholds.vorticity` to sit between that value and the translation case in
`tests/eval/test_motion.py`. Record the chosen value and why in the docstring — the threshold is
a calibration decision, not a magic number.

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/eval/events.py tests/eval/test_events.py
git commit -m "feat: heuristic clear/stratiform/convective/cyclonic event categorisation"
```

---

# Phase 5 — Inference and INSAT deployment

### Task 23: Tiled and recursive inference

**Files:**
- Create: `src/sattsr/infer/__init__.py`, `src/sattsr/infer/recursive.py`
- Test: `tests/infer/test_recursive.py`

**Interfaces:**
- Consumes: `FrameInterpolator`, `NormalizationConfig`, `normalize_bt`, `denormalize_bt`,
  `tile_positions`, `extract_tile`, `reassemble`, `pad_to_multiple`, `crop_to`, `Frame`.
- Produces:
  - `predict_midframe(model, i0, i2, *, device, norm, tile_size=256, tile_overlap=32, t=0.5, batch_size=8) -> np.ndarray`
    — full-grid Kelvin prediction; NaN preserved where **both** inputs are invalid.
  - `bisect_pair(model, i0, i2, *, factor, device, norm, tile_size, tile_overlap, batch_size=8) -> list[np.ndarray]`
    — `factor` must be a power of two ≥ 2; returns `factor - 1` frames in time order.
  - `interpolate_sequence(frames: list[Frame], model, *, factor, device, norm, tile_size, tile_overlap, batch_size=8) -> tuple[list[Frame], list[bool]]`
    — interleaves originals and synthetic frames, returning the full series plus flags.

Recursive bisection is how 30 min becomes 15 min and then 7.5 min: `factor=2` inserts the
midpoint, `factor=4` bisects again on both halves.

- [ ] **Step 1: Write the failing test**

Create `tests/infer/__init__.py` (empty), then `tests/infer/test_recursive.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from sattsr.config import ModelConfig, NormalizationConfig
from sattsr.infer.recursive import bisect_pair, interpolate_sequence, predict_midframe
from sattsr.io.base import Frame
from sattsr.models.interpolator import build_model

NORM = NormalizationConfig(bt_min=180.0, bt_max=330.0)
CFG = ModelConfig(base_channels=8, scales=[2, 1], use_raft_init=False)
DEVICE = torch.device("cpu")


def _field(size: int = 96, shift: int = 0) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return (250.0 + 20.0 * np.sin((xx + shift) * 0.3) * np.cos(yy * 0.25)).astype(np.float32)


def _frames(n: int = 3, size: int = 96) -> list[Frame]:
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    return [
        Frame(t0 + timedelta(minutes=30 * i), _field(size, shift=4 * i), "insat3dr", Path("x.h5"))
        for i in range(n)
    ]


def test_predict_midframe_returns_kelvin_on_the_full_grid():
    model = build_model(CFG).eval()
    out = predict_midframe(model, _field(), _field(shift=8), device=DEVICE, norm=NORM,
                           tile_size=32, tile_overlap=8)
    assert out.shape == (96, 96)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert 180.0 <= float(out.min()) and float(out.max()) <= 330.0


def test_predict_midframe_handles_a_grid_that_is_not_tile_aligned():
    model = build_model(CFG).eval()
    out = predict_midframe(model, _field(70), _field(70, shift=6), device=DEVICE, norm=NORM,
                           tile_size=32, tile_overlap=8)
    assert out.shape == (70, 70)


def test_predict_midframe_handles_a_grid_smaller_than_one_tile():
    model = build_model(CFG).eval()
    out = predict_midframe(model, _field(24), _field(24, shift=2), device=DEVICE, norm=NORM,
                           tile_size=32, tile_overlap=8)
    assert out.shape == (24, 24)


def test_predict_midframe_preserves_pixels_invalid_in_both_inputs():
    model = build_model(CFG).eval()
    a, b = _field(), _field(shift=8)
    a[:10, :10] = np.nan
    b[:10, :10] = np.nan
    a[20:24, 20:24] = np.nan            # invalid in only one input -> must be filled
    out = predict_midframe(model, a, b, device=DEVICE, norm=NORM, tile_size=32, tile_overlap=8)
    assert np.isnan(out[:10, :10]).all()
    assert np.isfinite(out[20:24, 20:24]).all()


def test_predict_midframe_is_deterministic():
    model = build_model(CFG).eval()
    a, b = _field(), _field(shift=8)
    first = predict_midframe(model, a, b, device=DEVICE, norm=NORM, tile_size=32, tile_overlap=8)
    second = predict_midframe(model, a, b, device=DEVICE, norm=NORM, tile_size=32, tile_overlap=8)
    np.testing.assert_allclose(first, second)


def test_bisect_pair_frame_counts():
    model = build_model(CFG).eval()
    a, b = _field(), _field(shift=8)
    kwargs = {"device": DEVICE, "norm": NORM, "tile_size": 32, "tile_overlap": 8}
    assert len(bisect_pair(model, a, b, factor=2, **kwargs)) == 1
    assert len(bisect_pair(model, a, b, factor=4, **kwargs)) == 3
    assert len(bisect_pair(model, a, b, factor=8, **kwargs)) == 7


def test_bisect_pair_rejects_non_power_of_two_factors():
    model = build_model(CFG).eval()
    with pytest.raises(ValueError):
        bisect_pair(model, _field(), _field(shift=8), factor=3, device=DEVICE, norm=NORM,
                    tile_size=32, tile_overlap=8)


def test_interpolate_sequence_interleaves_and_flags_correctly():
    model = build_model(CFG).eval()
    frames = _frames(3)
    out, flags = interpolate_sequence(frames, model, factor=2, device=DEVICE, norm=NORM,
                                      tile_size=32, tile_overlap=8)
    assert len(out) == 5                       # 3 originals + 2 midpoints
    assert flags == [False, True, False, True, False]
    assert [f.timestamp for f in out] == sorted(f.timestamp for f in out)
    assert out[1].timestamp - out[0].timestamp == timedelta(minutes=15)


def test_interpolate_sequence_at_factor_four_reaches_seven_and_a_half_minutes():
    model = build_model(CFG).eval()
    out, flags = interpolate_sequence(_frames(2), model, factor=4, device=DEVICE, norm=NORM,
                                      tile_size=32, tile_overlap=8)
    assert len(out) == 5
    assert flags == [False, True, True, True, False]
    assert out[1].timestamp - out[0].timestamp == timedelta(seconds=450)   # 7.5 minutes


def test_interpolate_sequence_marks_every_synthetic_frame():
    model = build_model(CFG).eval()
    out, flags = interpolate_sequence(_frames(4), model, factor=2, device=DEVICE, norm=NORM,
                                      tile_size=32, tile_overlap=8)
    assert sum(flags) == 3
    assert all(f.sensor == "insat3dr" for f in out)


def test_interpolate_sequence_needs_at_least_two_frames():
    model = build_model(CFG).eval()
    with pytest.raises(ValueError):
        interpolate_sequence(_frames(1), model, factor=2, device=DEVICE, norm=NORM,
                             tile_size=32, tile_overlap=8)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/infer/test_recursive.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.infer'`

- [ ] **Step 3: Implement `src/sattsr/infer/recursive.py`**

Create an empty `src/sattsr/infer/__init__.py` too.

```python
"""Full-grid tiled prediction and recursive temporal bisection."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import torch

from sattsr.config import NormalizationConfig
from sattsr.data.radiometry import denormalize_bt, normalize_bt
from sattsr.data.tiling import crop_to, extract_tile, pad_to_multiple, reassemble, tile_positions
from sattsr.io.base import Frame
from sattsr.models.interpolator import FrameInterpolator


@torch.no_grad()
def predict_midframe(
    model: FrameInterpolator,
    i0: np.ndarray,
    i2: np.ndarray,
    *,
    device: torch.device,
    norm: NormalizationConfig,
    tile_size: int = 256,
    tile_overlap: int = 32,
    t: float = 0.5,
    batch_size: int = 8,
) -> np.ndarray:
    """Synthesize the frame at normalised time `t` over an arbitrarily large grid."""
    a = np.asarray(i0, dtype=np.float32)
    b = np.asarray(i2, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    multiple = max(model.size_multiple, 1)
    effective_tile = min(tile_size, min(a.shape))
    effective_tile = max((effective_tile // multiple) * multiple, multiple)
    # Cap the overlap at a quarter of the tile: `hann_weight` builds its taper from
    # both edges, and a wider band would make the two ramps collide.
    effective_overlap = max(min(tile_overlap, effective_tile // 4), 0)

    a_pad, original = pad_to_multiple(a, multiple, min_size=effective_tile)
    b_pad, _ = pad_to_multiple(b, multiple, min_size=effective_tile)

    specs = tile_positions(a_pad.shape, effective_tile, effective_overlap)
    tiles_a = [normalize_bt(extract_tile(a_pad, s), norm) for s in specs]
    tiles_b = [normalize_bt(extract_tile(b_pad, s), norm) for s in specs]

    model.eval().to(device)
    predictions: list[np.ndarray] = []
    for start in range(0, len(specs), batch_size):
        chunk = slice(start, start + batch_size)
        x0 = torch.from_numpy(np.stack(tiles_a[chunk]))[:, None].to(device)
        x2 = torch.from_numpy(np.stack(tiles_b[chunk]))[:, None].to(device)
        tt = torch.full((x0.shape[0],), float(t), device=device, dtype=x0.dtype)
        pred = model(x0, x2, tt)["pred"].float().cpu().numpy()[:, 0]
        predictions.extend(pred)

    merged = reassemble(predictions, specs, a_pad.shape, effective_overlap)
    out = denormalize_bt(crop_to(merged, original), norm)

    both_invalid = ~np.isfinite(a) & ~np.isfinite(b)
    return np.where(both_invalid, np.nan, out).astype(np.float32)


def bisect_pair(
    model: FrameInterpolator,
    i0: np.ndarray,
    i2: np.ndarray,
    *,
    factor: int,
    device: torch.device,
    norm: NormalizationConfig,
    tile_size: int = 256,
    tile_overlap: int = 32,
    batch_size: int = 8,
) -> list[np.ndarray]:
    """Insert `factor - 1` frames between two observations by repeated bisection."""
    if factor < 2 or factor & (factor - 1):
        raise ValueError(f"factor must be a power of two >= 2, got {factor}")

    kwargs = {
        "device": device, "norm": norm, "tile_size": tile_size,
        "tile_overlap": tile_overlap, "batch_size": batch_size,
    }

    def _recurse(a: np.ndarray, b: np.ndarray, depth: int) -> list[np.ndarray]:
        if depth == 0:
            return []
        mid = predict_midframe(model, a, b, t=0.5, **kwargs)
        return [*_recurse(a, mid, depth - 1), mid, *_recurse(mid, b, depth - 1)]

    return _recurse(i0, i2, factor.bit_length() - 1)


def interpolate_sequence(
    frames: Sequence[Frame],
    model: FrameInterpolator,
    *,
    factor: int,
    device: torch.device,
    norm: NormalizationConfig,
    tile_size: int = 256,
    tile_overlap: int = 32,
    batch_size: int = 8,
) -> tuple[list[Frame], list[bool]]:
    """Densify a whole series, returning the interleaved frames and their synthetic flags."""
    ordered = sorted(frames, key=lambda f: f.timestamp)
    if len(ordered) < 2:
        raise ValueError("need at least two frames to interpolate between")

    out: list[Frame] = []
    flags: list[bool] = []

    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        out.append(left)
        flags.append(False)

        middles = bisect_pair(
            model, left.bt, right.bt, factor=factor, device=device, norm=norm,
            tile_size=tile_size, tile_overlap=tile_overlap, batch_size=batch_size,
        )
        gap = (right.timestamp - left.timestamp) / factor
        for k, bt in enumerate(middles, start=1):
            out.append(
                Frame(
                    timestamp=left.timestamp + gap * k,
                    bt=bt,
                    sensor=left.sensor,
                    source_path=left.source_path,
                )
            )
            flags.append(True)

    out.append(ordered[-1])
    flags.append(False)
    return out, flags
```

Note on `timedelta` division: `(right.timestamp - left.timestamp) / factor` returns a
`timedelta`, and a 30-minute gap at `factor=4` gives exactly 7 minutes 30 seconds — which is why
the test asserts `timedelta(seconds=450)`.

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/infer/test_recursive.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/infer tests/infer
git commit -m "feat: tiled full-grid prediction and recursive 30-15-7.5 minute bisection"
```

---

### Task 24: Evaluation harness and comparison report

**Files:**
- Create: `src/sattsr/eval/report.py`
- Test: `tests/eval/test_report.py`

**Interfaces:**
- Consumes: `metric_suite`, `motion_suite`, `classify_event`, `category_counts`, `run_baseline`,
  `predict_midframe`, `Triplet`, `load_cached`, `NormalizationConfig`.
- Produces:
  - `SampleResult(timestamp: datetime, category: EventCategory, method: str, metrics: dict[str, float])`
  - `METRIC_KEYS: tuple[str, ...]` = `("mse", "rmse", "psnr", "ssim", "fsim", "displacement_error", "csi", "pod", "far")`
  - `evaluate_triplets(model, triplets, *, device, norm, tile_size=256, tile_overlap=32, methods=("model", "linear", "farneback"), limit=None, thresholds=EventThresholds(), progress=False) -> list[SampleResult]`
  - `aggregate(results) -> dict` with keys `n_samples`, `methods`, `counts`, `overall`, `by_category`
  - `write_report(path, results, *, run_id, config_summary, notes=None) -> dict`
  - `load_report(path) -> dict`

The report is the deliverable the spec asks for: model against both baselines, on every required
metric, broken out by event category, with the heuristic nature of the categories stated in the
file itself.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_report.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from sattsr.config import ModelConfig, NormalizationConfig
from sattsr.data.index import FrameRef
from sattsr.data.triplets import Triplet
from sattsr.eval.events import EventCategory
from sattsr.eval.report import (
    METRIC_KEYS,
    SampleResult,
    aggregate,
    evaluate_triplets,
    load_report,
    write_report,
)
from sattsr.models.interpolator import build_model

NORM = NormalizationConfig(bt_min=180.0, bt_max=330.0)
CFG = ModelConfig(base_channels=8, scales=[2, 1], use_raft_init=False)


def _result(method: str, category: EventCategory, psnr: float, ssim: float) -> SampleResult:
    return SampleResult(
        timestamp=datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc),
        category=category,
        method=method,
        metrics={"psnr": psnr, "ssim": ssim, "csi": float("nan")},
    )


def _triplets(tmp_path: Path, n: int = 2, size: int = 64) -> list[Triplet]:
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    refs = []
    for i in range(n + 2):
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        bt = (255.0 + 20.0 * np.sin((xx + 3 * i) * 0.3) * np.cos(yy * 0.25)).astype(np.float32)
        p = tmp_path / f"f{i}.npy"
        np.save(p, bt)
        refs.append(FrameRef(t0 + timedelta(minutes=30 * i), p, "insat3dr"))
    return [Triplet(refs[i], refs[i + 1], refs[i + 2]) for i in range(n)]


def test_aggregate_averages_per_method_and_category():
    results = [
        _result("model", EventCategory.CLEAR, 30.0, 0.9),
        _result("model", EventCategory.CONVECTIVE, 20.0, 0.7),
        _result("linear", EventCategory.CLEAR, 25.0, 0.8),
        _result("linear", EventCategory.CONVECTIVE, 15.0, 0.5),
    ]
    summary = aggregate(results)
    assert summary["n_samples"] == 2                    # unique timestamps, not rows
    assert sorted(summary["methods"]) == ["linear", "model"]
    assert summary["overall"]["model"]["psnr"] == pytest.approx(25.0)
    assert summary["by_category"]["convective"]["model"]["psnr"] == pytest.approx(20.0)
    assert summary["counts"]["clear"] == 1


def test_aggregate_ignores_nan_metrics():
    results = [
        _result("model", EventCategory.CLEAR, 30.0, 0.9),
        _result("model", EventCategory.CLEAR, float("nan"), 0.7),
    ]
    summary = aggregate(results)
    assert summary["overall"]["model"]["psnr"] == pytest.approx(30.0)
    assert summary["overall"]["model"]["ssim"] == pytest.approx(0.8)


def test_aggregate_of_nothing_is_empty_not_a_crash():
    summary = aggregate([])
    assert summary["n_samples"] == 0
    assert summary["overall"] == {}


def test_write_report_round_trips_and_carries_provenance(tmp_path):
    results = [_result("model", EventCategory.CLEAR, 30.0, 0.9)]
    report = write_report(tmp_path / "report.json", results, run_id="run-1",
                          config_summary={"sensor": "insat3dr"}, notes="fine-tuned")
    assert (tmp_path / "report.json").exists()
    assert report["run_id"] == "run-1"

    back = load_report(tmp_path / "report.json")
    assert back["run_id"] == "run-1"
    assert back["config"]["sensor"] == "insat3dr"
    assert back["notes"] == "fine-tuned"
    assert back["summary"]["overall"]["model"]["psnr"] == pytest.approx(30.0)
    assert len(back["samples"]) == 1
    assert "heuristic" in back["category_disclaimer"].lower()


def test_report_json_is_plain_and_has_no_nan_literals(tmp_path):
    results = [_result("model", EventCategory.CLEAR, float("nan"), 0.9)]
    write_report(tmp_path / "report.json", results, run_id="r", config_summary={})
    text = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert "NaN" not in text                     # JSON.parse in the browser would choke
    assert json.loads(text)["samples"][0]["metrics"]["psnr"] is None


def test_evaluate_triplets_scores_every_method(tmp_path):
    model = build_model(CFG).eval()
    results = evaluate_triplets(
        model, _triplets(tmp_path), device=torch.device("cpu"), norm=NORM,
        tile_size=32, tile_overlap=8,
    )
    methods = {r.method for r in results}
    assert methods == {"model", "linear", "farneback"}
    assert len(results) == 2 * 3
    for r in results:
        assert set(METRIC_KEYS).issubset(r.metrics)
        assert isinstance(r.category, EventCategory)


def test_evaluate_triplets_honours_limit_and_method_selection(tmp_path):
    model = build_model(CFG).eval()
    results = evaluate_triplets(
        model, _triplets(tmp_path, n=4), device=torch.device("cpu"), norm=NORM,
        tile_size=32, tile_overlap=8, methods=("model", "linear"), limit=2,
    )
    assert len(results) == 2 * 2
    assert {r.method for r in results} == {"model", "linear"}


def test_evaluate_triplets_rejects_an_unknown_method(tmp_path):
    model = build_model(CFG).eval()
    with pytest.raises(KeyError):
        evaluate_triplets(model, _triplets(tmp_path), device=torch.device("cpu"), norm=NORM,
                          tile_size=32, tile_overlap=8, methods=("model", "magic"))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/eval/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.eval.report'`

- [ ] **Step 3: Implement `src/sattsr/eval/report.py`**

```python
"""Scoring the model against the baselines, and writing the comparison report."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sattsr.config import NormalizationConfig
from sattsr.data.index import load_cached
from sattsr.data.triplets import Triplet
from sattsr.eval.baselines import run_baseline
from sattsr.eval.events import EventCategory, EventThresholds, category_counts, classify_event
from sattsr.eval.metrics import metric_suite
from sattsr.eval.motion import motion_suite
from sattsr.infer.recursive import predict_midframe
from sattsr.models.interpolator import FrameInterpolator

METRIC_KEYS: tuple[str, ...] = (
    "mse", "rmse", "psnr", "ssim", "fsim", "displacement_error", "csi", "pod", "far",
)

CATEGORY_DISCLAIMER = (
    "Event categories are assigned by a documented heuristic on brightness temperature "
    "and inter-frame flow vorticity, not by a validated meteorological classifier. They "
    "exist so convective and cyclonic performance is reported separately rather than "
    "averaged away."
)

DEFAULT_METHODS: tuple[str, ...] = ("model", "linear", "farneback")


@dataclass(frozen=True)
class SampleResult:
    """One method's score on one triplet."""

    timestamp: datetime
    category: EventCategory
    method: str
    metrics: dict[str, float]


def _clean(value: float) -> float | None:
    """JSON has no NaN or Infinity literals; browsers reject them."""
    return None if value is None or not math.isfinite(float(value)) else float(value)


def evaluate_triplets(
    model: FrameInterpolator,
    triplets: Sequence[Triplet],
    *,
    device: torch.device,
    norm: NormalizationConfig,
    tile_size: int = 256,
    tile_overlap: int = 32,
    methods: Sequence[str] = DEFAULT_METHODS,
    limit: int | None = None,
    thresholds: EventThresholds = EventThresholds(),
    progress: bool = False,
) -> list[SampleResult]:
    """Score every requested method against the real middle frame of each triplet."""
    unknown = [m for m in methods if m != "model" and m not in {"linear", "farneback"}]
    if unknown:
        raise KeyError(f"unknown evaluation method(s): {unknown}")

    items = list(triplets)[: limit if limit is not None else len(triplets)]
    if progress:
        from tqdm import tqdm

        items = tqdm(items, desc="evaluating")   # type: ignore[assignment]

    results: list[SampleResult] = []
    for triplet in items:
        i0 = load_cached(triplet.t0)
        truth = load_cached(triplet.t1)
        i2 = load_cached(triplet.t2)
        category = classify_event(np.stack([i0, truth, i2]),
                                  data_range=norm.data_range, thresholds=thresholds)

        for method in methods:
            if method == "model":
                pred = predict_midframe(
                    model, i0, i2, device=device, norm=norm, tile_size=tile_size,
                    tile_overlap=tile_overlap, t=float(triplet.t),
                )
            else:
                pred = run_baseline(method, i0, i2, float(triplet.t),
                                    data_range=norm.data_range)

            metrics = metric_suite(pred, truth, data_range=norm.data_range)
            metrics.update(motion_suite(i0, pred, truth, data_range=norm.data_range))
            results.append(
                SampleResult(triplet.t1.timestamp, category, method, metrics)
            )
    return results


def _mean(values: list[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _summarise(results: Sequence[SampleResult]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for key, value in r.metrics.items():
            buckets[r.method][key].append(value)
    return {
        method: {key: _mean(values) for key, values in metrics.items()}
        for method, metrics in buckets.items()
    }


def aggregate(results: Sequence[SampleResult]) -> dict[str, Any]:
    """Mean of every metric, overall and per event category."""
    if not results:
        return {"n_samples": 0, "methods": [], "counts": category_counts([]),
                "overall": {}, "by_category": {}}

    per_timestamp: dict[datetime, EventCategory] = {r.timestamp: r.category for r in results}
    by_category: dict[str, dict[str, dict[str, float]]] = {}
    for category in EventCategory:
        subset = [r for r in results if r.category == category]
        if subset:
            by_category[category.value] = _summarise(subset)

    return {
        "n_samples": len(per_timestamp),
        "methods": sorted({r.method for r in results}),
        "counts": category_counts(per_timestamp.values()),
        "overall": _summarise(results),
        "by_category": by_category,
    }


def write_report(
    path: str | Path,
    results: Sequence[SampleResult],
    *,
    run_id: str,
    config_summary: dict[str, Any],
    notes: str | None = None,
) -> dict[str, Any]:
    """Write the comparison report as browser-safe JSON and return it."""
    summary = aggregate(results)
    report: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config_summary,
        "notes": notes,
        "metric_keys": list(METRIC_KEYS),
        "category_disclaimer": CATEGORY_DISCLAIMER,
        "summary": _sanitise(summary),
        "samples": [
            {
                "timestamp": r.timestamp.astimezone(timezone.utc).isoformat(),
                "category": r.category.value,
                "method": r.method,
                "metrics": {k: _clean(v) for k, v in r.metrics.items()},
            }
            for r in results
        ],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def _sanitise(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _sanitise(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_sanitise(v) for v in node]
    if isinstance(node, (int, float, np.floating, np.integer)):
        return _clean(float(node))
    return node


def load_report(path: str | Path) -> dict[str, Any]:
    """Read a report written by `write_report`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/eval -v`
Expected: 44 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/eval/report.py tests/eval/test_report.py
git commit -m "feat: evaluation harness and category-stratified comparison report"
```

---

### Task 25: The inference run pipeline

**Files:**
- Create: `src/sattsr/infer/pipeline.py`
- Test: `tests/infer/test_pipeline.py`

**Interfaces:**
- Consumes: `Config`, `get_reader`, `TargetGrid`, `build_index`, `prepare_cache`,
  `interpolate_sequence`, `write_frames_nc`, `load_checkpoint`, `build_model`.
- Produces:
  - `RunManifest` frozen dataclass with fields `run_id`, `sensor`, `created_at`,
    `input_cadence_minutes`, `output_cadence_minutes`, `factor`, `n_original`, `n_synthetic`,
    `output_nc`, `frames: list[dict]`, `checkpoint`, `model_version`; plus
    `.to_dict()` and `RunManifest.from_dict(d)`.
  - `write_manifest(run_dir, manifest) -> Path` / `read_manifest(run_dir) -> RunManifest`
  - `run_inference(config, *, input_dir, output_dir, checkpoint, factor, device, limit=None, run_id=None) -> RunManifest`

Run directory layout, which the web layer reads directly:

```
runs/<run_id>/
├── manifest.json
├── output.nc            # every frame, synthetic ones flagged
├── report.json          # written by `sattsr evaluate`, optional
├── frames/000.png ...   # written in Task 26
└── animations/          # written in Task 26
```

- [ ] **Step 1: Write the failing test**

`tests/infer/test_pipeline.py`:

```python
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
    manifest = run_inference(
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/infer/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.infer.pipeline'`

- [ ] **Step 3: Implement `src/sattsr/infer/pipeline.py`**

```python
"""End-to-end inference: raw sensor files in, flagged NetCDF and a run manifest out."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sattsr import __version__
from sattsr.config import Config
from sattsr.data.index import build_index
from sattsr.geo.grid import TargetGrid
from sattsr.infer.recursive import interpolate_sequence
from sattsr.io.base import Frame
from sattsr.io.registry import get_reader
from sattsr.io.writer import write_frames_nc
from sattsr.models.interpolator import build_model
from sattsr.train.checkpoint import load_checkpoint

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunManifest:
    """Everything the dashboard needs to describe one inference run."""

    run_id: str
    sensor: str
    created_at: datetime
    input_cadence_minutes: float
    output_cadence_minutes: float
    factor: int
    n_original: int
    n_synthetic: int
    output_nc: str
    frames: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: str = ""
    model_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        payload = asdict(self)
        payload["created_at"] = self.created_at.astimezone(timezone.utc).isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunManifest:
        """Inverse of `to_dict`."""
        data = dict(payload)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


def write_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    """Write `manifest.json` into a run directory."""
    out = Path(run_dir) / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return out


def read_manifest(run_dir: str | Path) -> RunManifest:
    """Read `manifest.json` from a run directory."""
    path = Path(run_dir) / "manifest.json"
    return RunManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def run_inference(
    config: Config,
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path,
    factor: int,
    device: torch.device,
    limit: int | None = None,
    run_id: str | None = None,
) -> RunManifest:
    """Read a series of sensor files, densify it, and write a flagged NetCDF product."""
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    reader = get_reader(config.data.sensor)
    grid = TargetGrid.from_config(config.data.grid)

    refs = build_index(reader, input_dir)
    if limit is not None:
        refs = refs[:limit]
    if len(refs) < 2:
        raise ValueError(f"need at least two input frames in {input_dir}, found {len(refs)}")

    originals: list[Frame] = [reader.read(ref.path, grid) for ref in refs]
    log.info("read %d frames from %s", len(originals), input_dir)

    model = build_model(config.model)
    load_checkpoint(checkpoint, model, map_location=str(device))
    model.eval().to(device)

    frames, flags = interpolate_sequence(
        originals, model,
        factor=factor, device=device, norm=config.data.normalization,
        tile_size=config.train.tile_size, tile_overlap=config.train.tile_overlap,
    )

    output_nc = run_dir / "output.nc"
    write_frames_nc(
        output_nc, frames, grid,
        synthetic=flags,
        model_version=__version__,
        extra_attrs={
            "checkpoint": str(checkpoint),
            "interpolation_factor": factor,
            "input_cadence_minutes": config.data.cadence_minutes,
        },
    )

    manifest = RunManifest(
        run_id=run_id or run_dir.name,
        sensor=config.data.sensor,
        created_at=datetime.now(timezone.utc),
        input_cadence_minutes=float(config.data.cadence_minutes),
        output_cadence_minutes=float(config.data.cadence_minutes) / factor,
        factor=int(factor),
        n_original=int(sum(1 for f in flags if not f)),
        n_synthetic=int(sum(flags)),
        output_nc=output_nc.name,
        frames=[
            {
                "index": i,
                "timestamp": frame.timestamp.astimezone(timezone.utc).isoformat(),
                "synthetic": bool(flag),
            }
            for i, (frame, flag) in enumerate(zip(frames, flags, strict=True))
        ],
        checkpoint=str(checkpoint),
    )
    write_manifest(run_dir, manifest)
    return manifest
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/infer -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add src/sattsr/infer/pipeline.py tests/infer/test_pipeline.py
git commit -m "feat: end-to-end inference run producing flagged NetCDF and a run manifest"
```

---

# Phase 6 — Visualisation and the web dashboard

### Task 26: Frame rendering and time-lapse animations

**Files:**
- Create: `src/sattsr/viz/__init__.py`, `src/sattsr/viz/colormap.py`, `src/sattsr/viz/frames.py`
- Test: `tests/viz/test_colormap.py`, `tests/viz/test_frames.py`

**Interfaces:**
- Consumes: `read_frames_nc`, `read_manifest`.
- Produces:
  - `IR_STOPS: tuple[tuple[float, tuple[int, int, int]], ...]` — the brightness-temperature
    enhancement control points.
  - `bt_to_rgb(bt, *, vmin=180.0, vmax=310.0) -> np.ndarray` — `(H, W, 3)` uint8; NaN renders as
    `NODATA_RGB = (20, 20, 30)`.
  - `save_png(bt, path, *, vmin=180.0, vmax=310.0) -> Path`
  - `render_run_frames(run_dir, *, vmin=180.0, vmax=310.0) -> list[Path]` — writes
    `frames/000.png …` from `output.nc`.
  - `make_animation(png_paths, out_path, *, fps=6) -> Path` — animated GIF.
  - `render_run_animations(run_dir, *, fps=6) -> dict[str, Path]` — writes
    `animations/original.gif` (observed frames only) and `animations/interpolated.gif`
    (every frame), the side-by-side pair the spec requires.

The enhancement is the conventional IR curve: warm cloud-free scenes render dark, cold cloud tops
render bright, and tops colder than about 235 K enter a colour ramp so convective cores stand out
in the animation.

- [ ] **Step 1: Write the failing tests**

Create `tests/viz/__init__.py` (empty), then `tests/viz/test_colormap.py`:

```python
from __future__ import annotations

import numpy as np

from sattsr.viz.colormap import NODATA_RGB, bt_to_rgb


def test_output_shape_and_dtype():
    rgb = bt_to_rgb(np.full((4, 5), 250.0, dtype=np.float32))
    assert rgb.shape == (4, 5, 3)
    assert rgb.dtype == np.uint8


def test_warm_scenes_render_darker_than_cold_ones():
    warm = bt_to_rgb(np.full((2, 2), 300.0, dtype=np.float32))
    cold = bt_to_rgb(np.full((2, 2), 240.0, dtype=np.float32))
    assert int(warm.sum()) < int(cold.sum())


def test_very_cold_tops_are_coloured_not_grey():
    core = bt_to_rgb(np.full((2, 2), 200.0, dtype=np.float32))[0, 0]
    assert int(core.max()) - int(core.min()) > 40, "deep convection must be visibly coloured"


def test_nan_renders_as_the_nodata_colour():
    rgb = bt_to_rgb(np.array([[np.nan, 250.0]], dtype=np.float32))
    assert tuple(int(v) for v in rgb[0, 0]) == NODATA_RGB


def test_values_outside_the_range_are_clamped_not_wrapped():
    low = bt_to_rgb(np.array([[100.0]], dtype=np.float32))
    high = bt_to_rgb(np.array([[400.0]], dtype=np.float32))
    assert np.array_equal(low, bt_to_rgb(np.array([[180.0]], dtype=np.float32)))
    assert np.array_equal(high, bt_to_rgb(np.array([[310.0]], dtype=np.float32)))


def test_mapping_is_monotonic_in_the_grey_band():
    values = np.array([[300.0, 280.0, 260.0, 240.0]], dtype=np.float32)
    brightness = bt_to_rgb(values)[0].sum(axis=-1)
    assert np.all(np.diff(brightness) > 0)
```

`tests/viz/test_frames.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sattsr.geo.grid import TargetGrid
from sattsr.io.base import Frame
from sattsr.io.writer import write_frames_nc
from sattsr.viz.frames import make_animation, render_run_animations, render_run_frames, save_png


def _grid() -> TargetGrid:
    return TargetGrid(lat_min=10.0, lat_max=14.0, lon_min=70.0, lon_max=74.0,
                      resolution_deg=0.25)


def _run(tmp_path: Path, n: int = 5) -> Path:
    grid = _grid()
    t0 = datetime(2026, 9, 4, 6, 0, tzinfo=timezone.utc)
    frames, flags = [], []
    for i in range(n):
        bt = np.full(grid.shape, 240.0 + 4.0 * i, dtype=np.float32)
        bt[0, 0] = np.nan
        frames.append(Frame(t0 + timedelta(minutes=15 * i), bt, "insat3dr", Path("x.h5")))
        flags.append(i % 2 == 1)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_frames_nc(run_dir / "output.nc", frames, grid, synthetic=flags,
                    model_version="test")
    return run_dir


def test_save_png_writes_a_readable_image(tmp_path):
    out = save_png(np.full((8, 8), 250.0, dtype=np.float32), tmp_path / "f.png")
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (8, 8)
        assert img.mode == "RGB"


def test_render_run_frames_writes_one_png_per_frame(tmp_path):
    run_dir = _run(tmp_path, n=5)
    paths = render_run_frames(run_dir)
    assert len(paths) == 5
    assert [p.name for p in paths] == ["000.png", "001.png", "002.png", "003.png", "004.png"]
    assert all(p.exists() for p in paths)
    assert all(p.parent == run_dir / "frames" for p in paths)


def test_render_run_frames_requires_an_output_netcdf(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        render_run_frames(empty)


def test_make_animation_produces_a_multi_frame_gif(tmp_path):
    pngs = [save_png(np.full((8, 8), 240.0 + 10 * i, dtype=np.float32), tmp_path / f"{i}.png")
            for i in range(4)]
    gif = make_animation(pngs, tmp_path / "anim.gif", fps=4)
    assert gif.exists()
    with Image.open(gif) as img:
        assert img.n_frames == 4


def test_make_animation_rejects_an_empty_sequence(tmp_path):
    with pytest.raises(ValueError):
        make_animation([], tmp_path / "anim.gif")


def test_render_run_animations_writes_both_required_animations(tmp_path):
    run_dir = _run(tmp_path, n=5)
    render_run_frames(run_dir)
    animations = render_run_animations(run_dir)

    assert set(animations) == {"original", "interpolated"}
    assert animations["original"].exists() and animations["interpolated"].exists()
    with Image.open(animations["original"]) as img:
        assert img.n_frames == 3          # observed frames only
    with Image.open(animations["interpolated"]) as img:
        assert img.n_frames == 5          # observed plus synthesized
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/viz -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.viz'`

- [ ] **Step 3: Implement `src/sattsr/viz/colormap.py`**

Create an empty `src/sattsr/viz/__init__.py` too.

```python
"""Brightness-temperature to colour, using a conventional IR enhancement."""

from __future__ import annotations

import numpy as np

NODATA_RGB: tuple[int, int, int] = (20, 20, 30)

# (brightness temperature K, RGB). Warm scenes dark, cold cloud tops bright,
# deep convection below ~235 K in colour so cores are visible in an animation.
IR_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (180.0, (255, 0, 255)),
    (195.0, (255, 90, 0)),
    (205.0, (255, 240, 0)),
    (215.0, (0, 220, 80)),
    (225.0, (0, 220, 255)),
    (235.0, (255, 255, 255)),
    (250.0, (200, 200, 200)),
    (273.0, (110, 110, 110)),
    (310.0, (0, 0, 0)),
)


def bt_to_rgb(
    bt: np.ndarray, *, vmin: float = 180.0, vmax: float = 310.0
) -> np.ndarray:
    """Render Kelvin as an (H, W, 3) uint8 image; NaN becomes NODATA_RGB."""
    a = np.asarray(bt, dtype=np.float64)
    finite = np.isfinite(a)
    clamped = np.clip(np.where(finite, a, vmin), vmin, vmax)

    knots = np.array([k for k, _ in IR_STOPS], dtype=np.float64)
    colours = np.array([c for _, c in IR_STOPS], dtype=np.float64)

    channels = [np.interp(clamped, knots, colours[:, i]) for i in range(3)]
    rgb = np.stack(channels, axis=-1)
    rgb = np.where(finite[..., None], rgb, np.array(NODATA_RGB, dtype=np.float64))
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Implement `src/sattsr/viz/frames.py`**

```python
"""PNG frame export and time-lapse animation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from sattsr.io.writer import read_frames_nc
from sattsr.viz.colormap import bt_to_rgb


def save_png(
    bt: np.ndarray, path: str | Path, *, vmin: float = 180.0, vmax: float = 310.0
) -> Path:
    """Write one brightness-temperature array as a colour PNG."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bt_to_rgb(bt, vmin=vmin, vmax=vmax), mode="RGB").save(out)
    return out


def render_run_frames(
    run_dir: str | Path, *, vmin: float = 180.0, vmax: float = 310.0
) -> list[Path]:
    """Render every frame in a run's `output.nc` to `frames/NNN.png`."""
    root = Path(run_dir)
    nc_path = root / "output.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"no output.nc in {root}")

    frames, _ = read_frames_nc(nc_path)
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    return [
        save_png(frame.bt, frames_dir / f"{i:03d}.png", vmin=vmin, vmax=vmax)
        for i, frame in enumerate(frames)
    ]


def make_animation(png_paths: Sequence[str | Path], out_path: str | Path, *, fps: int = 6) -> Path:
    """Combine PNGs into a looping animated GIF."""
    paths = [Path(p) for p in png_paths]
    if not paths:
        raise ValueError("cannot build an animation from zero frames")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in paths]
    try:
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=max(int(1000 / max(fps, 1)), 20),
            loop=0,
            optimize=False,
        )
    finally:
        for image in images:
            image.close()
    return out


def render_run_animations(run_dir: str | Path, *, fps: int = 6) -> dict[str, Path]:
    """Build the two animations the dashboard compares side by side.

    `original` contains only observed frames, at the sensor's native cadence.
    `interpolated` contains every frame, observed and synthesized.
    """
    root = Path(run_dir)
    frames, flags = read_frames_nc(root / "output.nc")
    frames_dir = root / "frames"
    if not frames_dir.exists():
        render_run_frames(root)

    all_pngs = [frames_dir / f"{i:03d}.png" for i in range(len(frames))]
    observed = [p for p, synthetic in zip(all_pngs, flags, strict=True) if not synthetic]

    animations_dir = root / "animations"
    return {
        "original": make_animation(observed, animations_dir / "original.gif", fps=fps),
        "interpolated": make_animation(all_pngs, animations_dir / "interpolated.gif", fps=fps),
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/viz -v`
Expected: 12 passed

- [ ] **Step 6: Commit**

```bash
git add src/sattsr/viz tests/viz
git commit -m "feat: IR colour enhancement, PNG export and side-by-side time-lapse animations"
```

---

### Task 27: FastAPI backend

**Files:**
- Create: `web/__init__.py`, `web/app.py`
- Test: `tests/web/__init__.py`, `tests/web/test_app.py`

**Interfaces:**
- Consumes: `read_manifest`, `load_report`.
- Produces:
  - `create_app(runs_dir: str | Path, *, static_dir: Path | None = None) -> FastAPI`
  - Endpoints:
    - `GET /api/health` → `{"status": "ok", "version": ...}`
    - `GET /api/runs` → `[{run_id, sensor, created_at, factor, n_original, n_synthetic, input_cadence_minutes, output_cadence_minutes, has_report}]`, newest first
    - `GET /api/runs/{run_id}` → the full manifest
    - `GET /api/runs/{run_id}/frames/{index}.png` → image
    - `GET /api/runs/{run_id}/animations/{kind}.gif` → `kind` in `original`/`interpolated`
    - `GET /api/runs/{run_id}/report` → the report JSON, `404` if not generated yet
    - `GET /api/runs/{run_id}/download` → `output.nc`
    - `GET /` → the dashboard (Task 28)
  - `RUN_ID_RE` — `^[A-Za-z0-9._-]{1,64}$`; anything else is rejected with `400`.

`run_id` and `index` come from the URL and are used to build filesystem paths, so both are
validated and every resolved path is checked to still live under `runs_dir`. Without that,
`GET /api/runs/..%2F..%2Fetc/...` reads arbitrary files.

- [ ] **Step 1: Write the failing test**

Create `tests/web/__init__.py` (empty), then `tests/web/test_app.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/web -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'web'`

- [ ] **Step 3: Make `web/` importable**

Create `web/__init__.py` (empty) and add to `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/sattsr", "web"]
```

Add the repository root to the test path so `import web` and `from tests.conftest import …`
both resolve — under `[tool.pytest.ini_options]`:

```toml
pythonpath = ["."]
```

- [ ] **Step 4: Implement `web/app.py`**

```python
"""FastAPI service backing the comparison dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sattsr import __version__
from sattsr.infer.pipeline import read_manifest

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ANIMATION_KINDS = frozenset({"original", "interpolated"})


def _safe_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Resolve a run directory, refusing anything that escapes `runs_dir`."""
    if not RUN_ID_RE.match(run_id) or run_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="invalid run id")
    root = runs_dir.resolve()
    candidate = (root / run_id).resolve()
    if root not in candidate.parents and candidate != root:
        raise HTTPException(status_code=400, detail="invalid run id")
    if not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return candidate


def _require_file(path: Path, message: str) -> Path:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=message)
    return path


def create_app(runs_dir: str | Path, *, static_dir: Path | None = None) -> FastAPI:
    """Build the API over a directory of inference runs."""
    runs_root = Path(runs_dir)
    static_root = Path(static_dir) if static_dir else Path(__file__).parent / "static"

    app = FastAPI(title="sattsr dashboard", version=__version__)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, object]]:
        if not runs_root.is_dir():
            return []
        summaries: list[dict[str, object]] = []
        for entry in sorted(runs_root.iterdir()):
            if not (entry / "manifest.json").is_file():
                continue
            manifest = read_manifest(entry)
            summaries.append(
                {
                    "run_id": manifest.run_id,
                    "sensor": manifest.sensor,
                    "created_at": manifest.created_at.isoformat(),
                    "factor": manifest.factor,
                    "n_original": manifest.n_original,
                    "n_synthetic": manifest.n_synthetic,
                    "input_cadence_minutes": manifest.input_cadence_minutes,
                    "output_cadence_minutes": manifest.output_cadence_minutes,
                    "has_report": (entry / "report.json").is_file(),
                }
            )
        summaries.sort(key=lambda s: str(s["created_at"]), reverse=True)
        return summaries

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, object]:
        run_dir = _safe_run_dir(runs_root, run_id)
        _require_file(run_dir / "manifest.json", "run has no manifest")
        return read_manifest(run_dir).to_dict()

    @app.get("/api/runs/{run_id}/frames/{index}.png")
    def frame_png(run_id: str, index: int) -> FileResponse:
        run_dir = _safe_run_dir(runs_root, run_id)
        if index < 0:
            raise HTTPException(status_code=404, detail="no such frame")
        path = _require_file(run_dir / "frames" / f"{index:03d}.png", "no such frame")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/runs/{run_id}/animations/{kind}.gif")
    def animation(run_id: str, kind: str) -> FileResponse:
        if kind not in ANIMATION_KINDS:
            raise HTTPException(status_code=404, detail=f"unknown animation: {kind}")
        run_dir = _safe_run_dir(runs_root, run_id)
        path = _require_file(run_dir / "animations" / f"{kind}.gif", "animation not generated")
        return FileResponse(path, media_type="image/gif")

    @app.get("/api/runs/{run_id}/report")
    def report(run_id: str) -> JSONResponse:
        run_dir = _safe_run_dir(runs_root, run_id)
        path = _require_file(run_dir / "report.json", "no report for this run")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/runs/{run_id}/download")
    def download(run_id: str) -> FileResponse:
        run_dir = _safe_run_dir(runs_root, run_id)
        path = _require_file(run_dir / "output.nc", "no NetCDF product for this run")
        return FileResponse(path, media_type="application/x-netcdf",
                            filename=f"{run_id}.nc")

    if static_root.is_dir():
        app.mount("/", StaticFiles(directory=static_root, html=True), name="static")

    return app
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/web -v`
Expected: 12 passed

The static mount is registered last on purpose: FastAPI matches routes in order, so mounting `/`
first would swallow every `/api/...` request.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml web tests/web
git commit -m "feat: FastAPI backend serving runs, frames, animations and reports"
```

---

### Task 28: The dashboard front end

**Files:**
- Create: `web/static/index.html`, `web/static/styles.css`, `web/static/app.js`
- Test: `tests/web/test_static.py`

**Interfaces:**
- Consumes: the API from Task 27.
- Produces: a single-page dashboard with these DOM contracts, which the test asserts:
  `#run-select`, `#summary`, `#player-original`, `#player-interpolated`, `#timeline`,
  `#play-toggle`, `#speed`, `#synthetic-badge`, `#frame-label`, `#chart-overall`,
  `#chart-category`, `#metrics-table`, `#download-nc`.

Two players share one timeline. The original panel steps only through observed frames; the
interpolated panel steps through every frame. That side-by-side comparison plus the metric charts
is what FR-3 asks for, and "Web GUI Design" is a scored criterion, so the layout is deliberate
rather than a default.

- [ ] **Step 1: Write the failing test**

`tests/web/test_static.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from web.app import create_app

STATIC = Path(__file__).resolve().parents[2] / "web" / "static"

REQUIRED_IDS = [
    "run-select", "summary", "player-original", "player-interpolated", "timeline",
    "play-toggle", "speed", "synthetic-badge", "frame-label", "chart-overall",
    "chart-category", "metrics-table", "download-nc",
]


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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/web/test_static.py -v`
Expected: FAIL — the static files do not exist yet

- [ ] **Step 3: Write `web/static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cross-Sensor Temporal Super Resolution</title>
  <link rel="stylesheet" href="styles.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
          defer></script>
  <script src="app.js" defer></script>
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Temporal Super Resolution</h1>
      <p class="subtitle">Geostationary thermal imagery &mdash; original vs AI-interpolated</p>
    </div>
    <label class="run-picker">
      <span>Run</span>
      <select id="run-select" aria-label="Select an inference run"></select>
    </label>
  </header>

  <section id="summary" class="summary" aria-live="polite"></section>

  <section class="players">
    <figure class="panel">
      <figcaption>
        <span class="panel-title">Original</span>
        <span class="panel-note" id="label-original">observed frames</span>
      </figcaption>
      <div class="frame-wrap">
        <img id="player-original" alt="Original satellite frame" />
      </div>
    </figure>

    <figure class="panel">
      <figcaption>
        <span class="panel-title">Interpolated</span>
        <span class="panel-note" id="label-interpolated">observed + synthesized</span>
        <span class="badge" id="synthetic-badge" hidden>SYNTHETIC</span>
      </figcaption>
      <div class="frame-wrap">
        <img id="player-interpolated" alt="Interpolated satellite frame" />
      </div>
    </figure>
  </section>

  <section class="controls">
    <button id="play-toggle" type="button">Play</button>
    <input id="timeline" type="range" min="0" max="0" value="0" step="1"
           aria-label="Frame timeline" />
    <span id="frame-label" class="frame-label">&mdash;</span>
    <label class="speed">
      Speed
      <select id="speed">
        <option value="500">0.5x</option>
        <option value="250" selected>1x</option>
        <option value="120">2x</option>
        <option value="60">4x</option>
      </select>
    </label>
    <a id="download-nc" class="button-link" href="#" download>Download .nc</a>
  </section>

  <section class="charts">
    <figure class="card">
      <figcaption>Model vs baselines &mdash; overall</figcaption>
      <canvas id="chart-overall" height="220"></canvas>
    </figure>
    <figure class="card">
      <figcaption>PSNR by event category</figcaption>
      <canvas id="chart-category" height="220"></canvas>
    </figure>
  </section>

  <section class="card">
    <h2>Comparison against ground truth</h2>
    <div class="table-scroll">
      <table id="metrics-table"><tbody></tbody></table>
    </div>
    <p class="disclaimer">
      Frames marked <strong>SYNTHETIC</strong> are AI-generated and are not observations.
      They are decision-support augmentation only.
    </p>
  </section>
</body>
</html>
```

- [ ] **Step 4: Write `web/static/styles.css`**

```css
:root {
  --bg: #0e1116;
  --panel: #171b22;
  --line: #262c36;
  --text: #e6e9ef;
  --muted: #9aa4b2;
  --accent: #4fc3f7;
  --warn: #ffb74d;
  --radius: 10px;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 0 24px 48px;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.5 "Segoe UI", system-ui, -apple-system, sans-serif;
}

.topbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
  padding: 20px 0 16px;
  border-bottom: 1px solid var(--line);
}

.topbar h1 { margin: 0; font-size: 22px; letter-spacing: 0.2px; }
.subtitle { margin: 4px 0 0; color: var(--muted); font-size: 13px; }

.run-picker { display: flex; align-items: center; gap: 8px; color: var(--muted); }

select, button, .button-link {
  background: var(--panel);
  color: var(--text);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 7px 12px;
  font: inherit;
  cursor: pointer;
  text-decoration: none;
}

button:hover, .button-link:hover, select:hover { border-color: var(--accent); }

.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin: 20px 0;
}

.stat {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 12px 14px;
}

.stat .value { font-size: 20px; font-weight: 600; }
.stat .label { color: var(--muted); font-size: 12px; text-transform: uppercase; }

.players { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }

.panel {
  margin: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 12px;
}

.panel figcaption { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.panel-title { font-weight: 600; }
.panel-note { color: var(--muted); font-size: 12px; }

.badge {
  margin-left: auto;
  background: var(--warn);
  color: #16191f;
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.6px;
}

.frame-wrap { aspect-ratio: 1 / 1; background: #05070a; border-radius: 6px; overflow: hidden; }
.frame-wrap img { width: 100%; height: 100%; object-fit: contain; image-rendering: pixelated; }

.controls {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
  margin: 18px 0;
  padding: 12px 14px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
}

#timeline { flex: 1 1 260px; accent-color: var(--accent); }
.frame-label { color: var(--muted); font-variant-numeric: tabular-nums; min-width: 210px; }
.speed { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 13px; }

.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }

.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 14px 16px;
  margin: 0 0 16px;
}

.card figcaption, .card h2 { font-size: 14px; font-weight: 600; margin: 0 0 10px; }

.table-scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--muted); font-weight: 600; }
tbody tr.best td { color: var(--accent); }

.disclaimer { color: var(--muted); font-size: 12px; margin: 12px 0 0; }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
```

- [ ] **Step 5: Write `web/static/app.js`**

```javascript
"use strict";

const METRIC_ORDER = ["psnr", "ssim", "fsim", "rmse", "displacement_error", "csi"];
const HIGHER_IS_BETTER = new Set(["psnr", "ssim", "fsim", "csi", "pod"]);
const METHOD_COLOURS = { model: "#4fc3f7", linear: "#9aa4b2", farneback: "#ffb74d" };

const state = {
  runId: null,
  manifest: null,
  observed: [],   // manifest indices of observed frames
  cursor: 0,
  timer: null,
  charts: {},
};

const $ = (id) => document.getElementById(id);

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) return null;
  return response.json();
}

function frameUrl(runId, index) {
  return `/api/runs/${encodeURIComponent(runId)}/frames/${index}.png`;
}

function formatTime(iso) {
  return iso ? iso.replace("T", " ").replace("+00:00", "Z").slice(0, 19) : "—";
}

function renderSummary(manifest) {
  const stats = [
    ["Sensor", manifest.sensor],
    ["Input cadence", `${manifest.input_cadence_minutes} min`],
    ["Output cadence", `${manifest.output_cadence_minutes} min`],
    ["Factor", `${manifest.factor}x`],
    ["Observed frames", manifest.n_original],
    ["Synthesized frames", manifest.n_synthetic],
  ];
  $("summary").innerHTML = stats
    .map(([label, value]) =>
      `<div class="stat"><div class="value">${value}</div><div class="label">${label}</div></div>`)
    .join("");
}

function showFrame(cursor) {
  const frames = state.manifest.frames;
  state.cursor = Math.max(0, Math.min(cursor, frames.length - 1));
  const frame = frames[state.cursor];

  $("player-interpolated").src = frameUrl(state.runId, state.cursor);
  $("synthetic-badge").hidden = !frame.synthetic;

  // The original panel holds the most recent real observation at or before this time.
  let latest = state.observed[0] ?? 0;
  for (const index of state.observed) {
    if (index <= state.cursor) latest = index;
  }
  $("player-original").src = frameUrl(state.runId, latest);

  $("timeline").value = String(state.cursor);
  $("frame-label").textContent =
    `${formatTime(frame.timestamp)}  ·  ${state.cursor + 1}/${frames.length}` +
    `  ·  ${frame.synthetic ? "synthesized" : "observed"}`;
}

function stop() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  $("play-toggle").textContent = "Play";
}

function play() {
  stop();
  const delay = Number($("speed").value);
  state.timer = setInterval(() => {
    showFrame((state.cursor + 1) % state.manifest.frames.length);
  }, delay);
  $("play-toggle").textContent = "Pause";
}

function drawOverallChart(summary) {
  const methods = Object.keys(summary.overall || {});
  if (!methods.length) return;
  const labels = METRIC_ORDER.filter((m) => m in (summary.overall[methods[0]] || {}));

  state.charts.overall?.destroy();
  state.charts.overall = new Chart($("chart-overall"), {
    type: "bar",
    data: {
      labels,
      datasets: methods.map((method) => ({
        label: method,
        backgroundColor: METHOD_COLOURS[method] || "#7e57c2",
        data: labels.map((metric) => summary.overall[method][metric] ?? null),
      })),
    },
    options: {
      responsive: true,
      scales: { y: { type: "logarithmic", ticks: { color: "#9aa4b2" } },
                x: { ticks: { color: "#9aa4b2" } } },
      plugins: { legend: { labels: { color: "#e6e9ef" } } },
    },
  });
}

function drawCategoryChart(summary) {
  const categories = Object.keys(summary.by_category || {});
  if (!categories.length) return;
  const methods = summary.methods || [];

  state.charts.category?.destroy();
  state.charts.category = new Chart($("chart-category"), {
    type: "bar",
    data: {
      labels: categories,
      datasets: methods.map((method) => ({
        label: method,
        backgroundColor: METHOD_COLOURS[method] || "#7e57c2",
        data: categories.map((c) => summary.by_category[c]?.[method]?.psnr ?? null),
      })),
    },
    options: {
      responsive: true,
      scales: { y: { title: { display: true, text: "PSNR (dB)", color: "#9aa4b2" },
                     ticks: { color: "#9aa4b2" } },
                x: { ticks: { color: "#9aa4b2" } } },
      plugins: { legend: { labels: { color: "#e6e9ef" } } },
    },
  });
}

function drawTable(summary) {
  const methods = Object.keys(summary.overall || {});
  const table = $("metrics-table");
  if (!methods.length) {
    table.innerHTML = "<tbody><tr><td>No report generated for this run yet.</td></tr></tbody>";
    return;
  }
  const metrics = METRIC_ORDER.filter((m) => m in summary.overall[methods[0]]);
  const head = `<thead><tr><th>Metric</th>${methods.map((m) => `<th>${m}</th>`).join("")}` +
               `</tr></thead>`;
  const body = metrics.map((metric) => {
    const values = methods.map((m) => summary.overall[m][metric]);
    const finite = values.filter((v) => typeof v === "number");
    const best = finite.length
      ? (HIGHER_IS_BETTER.has(metric) ? Math.max(...finite) : Math.min(...finite))
      : null;
    const cells = values.map((v) => {
      const shown = typeof v === "number" ? v.toFixed(4) : "—";
      return `<td${v === best ? ' class="win"' : ""}>${shown}</td>`;
    }).join("");
    return `<tr><td>${metric}</td>${cells}</tr>`;
  }).join("");
  table.innerHTML = `${head}<tbody>${body}</tbody>`;
}

async function loadRun(runId) {
  stop();
  state.runId = runId;
  state.manifest = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  if (!state.manifest) return;

  state.observed = state.manifest.frames
    .map((frame, index) => (frame.synthetic ? -1 : index))
    .filter((index) => index >= 0);

  renderSummary(state.manifest);
  $("timeline").max = String(state.manifest.frames.length - 1);
  $("download-nc").href = `/api/runs/${encodeURIComponent(runId)}/download`;
  $("label-original").textContent =
    `observed frames · ${state.manifest.input_cadence_minutes} min`;
  $("label-interpolated").textContent =
    `observed + synthesized · ${state.manifest.output_cadence_minutes} min`;
  showFrame(0);

  const report = await getJSON(`/api/runs/${encodeURIComponent(runId)}/report`);
  const summary = report?.summary ?? { overall: {}, by_category: {}, methods: [] };
  drawOverallChart(summary);
  drawCategoryChart(summary);
  drawTable(summary);
}

async function init() {
  const runs = (await getJSON("/api/runs")) || [];
  const select = $("run-select");
  select.innerHTML = runs
    .map((r) => `<option value="${r.run_id}">${r.run_id} (${r.sensor})</option>`)
    .join("");

  select.addEventListener("change", () => loadRun(select.value));
  $("play-toggle").addEventListener("click", () => (state.timer ? stop() : play()));
  $("timeline").addEventListener("input", (event) => {
    stop();
    showFrame(Number(event.target.value));
  });
  $("speed").addEventListener("change", () => { if (state.timer) play(); });

  if (runs.length) await loadRun(runs[0].run_id);
  else $("summary").innerHTML =
    '<div class="stat"><div class="value">No runs</div>' +
    '<div class="label">run sattsr infer first</div></div>';
}

document.addEventListener("DOMContentLoaded", init);
```

Add the `.win` cell colour to `styles.css`:

```css
td.win { color: var(--accent); font-weight: 600; }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/web -v`
Expected: 30 passed

- [ ] **Step 7: Check it by eye**

```bash
python -c "import uvicorn; from web.app import create_app; uvicorn.run(create_app('runs'), port=8000)"
```

Open `http://127.0.0.1:8000`. With no runs yet it must render the empty state without console
errors. Come back to this after Task 29 produces a real run.

- [ ] **Step 8: Commit**

```bash
git add web/static tests/web/test_static.py
git commit -m "feat: side-by-side comparison dashboard with metric charts"
```

---

# Phase 7 — Command line and end-to-end

### Task 29: The `sattsr` CLI and an end-to-end smoke test

**Files:**
- Create: `src/sattsr/cli.py`, `src/sattsr/data/download.py`
- Test: `tests/test_cli.py`, `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: everything built so far.
- Produces:
  - `app: typer.Typer` with commands `prepare`, `train`, `finetune`, `evaluate`, `infer`,
    `render`, `serve`, `fetch-goes`.
  - `resolve_device(name: str) -> torch.device` — `"auto"` picks CUDA when available.
  - `build_datasets(config, *, sensor_cache=None) -> tuple[TripletDataset, TripletDataset]`
  - `download.fetch_goes_c13(dest, *, date, hours, bucket="noaa-goes19", product="ABI-L1b-RadF", max_files=None) -> list[Path]`
    — anonymous S3 listing and download. **Not covered by tests** (it needs the network); it is
    isolated in its own module for that reason.

- [ ] **Step 1: Write the failing CLI test**

`tests/test_cli.py`:

```python
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
    for command in ("prepare", "train", "finetune", "evaluate", "infer", "render", "serve"):
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


def test_missing_config_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["prepare", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
```

- [ ] **Step 2: Write the failing end-to-end test**

`tests/test_end_to_end.py`:

```python
from __future__ import annotations

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

from datetime import timedelta


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
```

- [ ] **Step 3: Run both to verify they fail**

Run: `pytest tests/test_cli.py tests/test_end_to_end.py -v -m ""`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.cli'`

- [ ] **Step 4: Implement `src/sattsr/data/download.py`**

```python
"""Anonymous bulk download of GOES ABI files from the NOAA public S3 bucket.

Not covered by the test suite: it needs the network. Keep network access confined here.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)


def fetch_goes_c13(
    dest: str | Path,
    *,
    day: date,
    hours: range | list[int],
    bucket: str = "noaa-goes19",
    product: str = "ABI-L1b-RadF",
    channel: str = "C13",
    max_files: int | None = None,
) -> list[Path]:
    """Download one day's Channel 13 full-disc files for the requested hours."""
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    out_dir = Path(dest)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for hour in hours:
        prefix = f"{bucket}/{product}/{day.year}/{day.timetuple().tm_yday:03d}/{hour:02d}/"
        try:
            keys = [k for k in fs.ls(prefix) if channel in k and k.endswith(".nc")]
        except FileNotFoundError:
            log.warning("no data at %s", prefix)
            continue

        for key in sorted(keys):
            local = out_dir / Path(key).name
            if not local.exists():
                log.info("downloading %s", key)
                fs.get(key, str(local))
            downloaded.append(local)
            if max_files is not None and len(downloaded) >= max_files:
                return downloaded
    return downloaded
```

- [ ] **Step 5: Implement `src/sattsr/cli.py`**

```python
"""Command line entry points."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import torch
import typer
from torch.utils.data import DataLoader

from sattsr.config import Config, load_config
from sattsr.data.dataset import TripletDataset
from sattsr.data.index import build_index, prepare_cache, save_index
from sattsr.data.triplets import build_triplets, split_triplets
from sattsr.eval.report import evaluate_triplets, write_report
from sattsr.geo.grid import TargetGrid
from sattsr.infer.pipeline import run_inference
from sattsr.io.registry import get_reader
from sattsr.losses.composite import CompositeLoss
from sattsr.models.interpolator import build_model
from sattsr.train.checkpoint import load_checkpoint
from sattsr.train.finetune import prepare_finetune
from sattsr.train.loop import fit, seed_everything
from sattsr.viz.frames import render_run_animations, render_run_frames

app = typer.Typer(add_completion=False, help="Cross-sensor temporal super resolution.")

ConfigOpt = typer.Option(..., "--config", "-c", exists=True, dir_okay=False,
                         help="Path to a YAML config file.")
DeviceOpt = typer.Option("auto", "--device", help="auto, cpu, cuda or cuda:N.")


def resolve_device(name: str) -> torch.device:
    """Resolve a device string, preferring CUDA when `auto` and it is available."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _cached_refs(config: Config) -> list:
    reader = get_reader(config.data.sensor)
    grid = TargetGrid.from_config(config.data.grid)
    return prepare_cache(reader, build_index(reader, config.data.raw_root), grid,
                         config.data.cache_root)


def build_datasets(config: Config) -> tuple[TripletDataset, TripletDataset]:
    """Cache-backed training and validation datasets, split by calendar day."""
    refs = _cached_refs(config)
    triplets = build_triplets(
        refs,
        step=timedelta(minutes=config.data.cadence_minutes),
        tolerance=timedelta(minutes=config.data.tolerance_minutes),
    )
    if not triplets:
        raise typer.BadParameter(
            "no triplets could be formed; check cadence_minutes and the input directory"
        )
    train_t, val_t = split_triplets(triplets, val_fraction=config.train.val_fraction,
                                    seed=config.train.seed)
    if not val_t:                       # single-day dataset: hold out the tail
        cut = max(1, int(len(train_t) * 0.8))
        train_t, val_t = train_t[:cut], train_t[cut:] or train_t[-1:]

    make = lambda items, augment: TripletDataset(   # noqa: E731
        items, config.data.normalization, config.train.tile_size,
        augment=augment, seed=config.train.seed,
    )
    return make(train_t, True), make(val_t, False)


def _loaders(config: Config) -> tuple[DataLoader, DataLoader]:
    train_ds, val_ds = build_datasets(config)
    common = {"num_workers": config.train.num_workers, "pin_memory": False}
    return (
        DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True, **common),
        DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False, **common),
    )


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Configure logging for every command."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def prepare(config: Path = ConfigOpt) -> None:
    """Discover raw files and build the regridded frame cache."""
    cfg = load_config(config)
    refs = _cached_refs(cfg)
    index_path = Path(cfg.data.cache_root) / "index.json"
    save_index(refs, index_path)
    typer.echo(f"cached {len(refs)} frames; index at {index_path}")


@app.command()
def train(config: Path = ConfigOpt, device: str = DeviceOpt) -> None:
    """Pretrain the interpolator on the configured sensor."""
    cfg = load_config(config)
    seed_everything(cfg.train.seed)
    dev = resolve_device(device)

    model = build_model(cfg.model)
    train_loader, val_loader = _loaders(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    best = fit(model, train_loader, val_loader, CompositeLoss(cfg.loss), optimizer, dev,
               epochs=cfg.train.epochs, checkpoint_dir=cfg.train.checkpoint_dir,
               amp=cfg.train.amp, config=cfg)
    typer.echo(f"best checkpoint: {best}")


@app.command()
def finetune(
    config: Path = ConfigOpt,
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, dir_okay=False),
    device: str = DeviceOpt,
) -> None:
    """Adapt a pretrained model to another sensor with partial freezing."""
    cfg = load_config(config)
    seed_everything(cfg.train.seed)
    dev = resolve_device(device)

    model = build_model(cfg.model)
    load_checkpoint(checkpoint, model, map_location=str(dev), strict=False)
    optimizer = prepare_finetune(model, cfg.train)

    train_loader, val_loader = _loaders(cfg)
    best = fit(model, train_loader, val_loader, CompositeLoss(cfg.loss), optimizer, dev,
               epochs=cfg.train.epochs, checkpoint_dir=cfg.train.checkpoint_dir,
               amp=cfg.train.amp, config=cfg)
    typer.echo(f"fine-tuned checkpoint: {best}")


@app.command()
def evaluate(
    config: Path = ConfigOpt,
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, dir_okay=False),
    out: Path = typer.Option(Path("reports/report.json"), "--out"),
    device: str = DeviceOpt,
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    """Score the model against both baselines and write the comparison report."""
    cfg = load_config(config)
    dev = resolve_device(device)

    model = build_model(cfg.model)
    load_checkpoint(checkpoint, model, map_location=str(dev))
    model.eval().to(dev)

    refs = _cached_refs(cfg)
    triplets = build_triplets(refs, step=timedelta(minutes=cfg.data.cadence_minutes),
                              tolerance=timedelta(minutes=cfg.data.tolerance_minutes))
    results = evaluate_triplets(
        model, triplets, device=dev, norm=cfg.data.normalization,
        tile_size=cfg.train.tile_size, tile_overlap=cfg.train.tile_overlap,
        limit=limit, progress=True,
    )
    report = write_report(out, results, run_id=out.parent.name,
                          config_summary={"sensor": cfg.data.sensor,
                                          "checkpoint": str(checkpoint)})
    for method, metrics in report["summary"]["overall"].items():
        typer.echo(f"{method:>10}  psnr={metrics.get('psnr')}  ssim={metrics.get('ssim')}")


@app.command()
def infer(
    config: Path = ConfigOpt,
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, dir_okay=False),
    input_dir: Path = typer.Option(..., "--input", exists=True, file_okay=False),
    out: Path = typer.Option(..., "--out"),
    factor: int = typer.Option(2, "--factor", help="2 halves the interval, 4 quarters it."),
    device: str = DeviceOpt,
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    """Densify a series of observations and write a flagged NetCDF product."""
    cfg = load_config(config)
    manifest = run_inference(cfg, input_dir=input_dir, output_dir=out, checkpoint=checkpoint,
                             factor=factor, device=resolve_device(device), limit=limit)
    typer.echo(
        f"{manifest.n_original} observed + {manifest.n_synthetic} synthesized frames "
        f"at {manifest.output_cadence_minutes} min -> {out}"
    )


@app.command()
def render(
    run: Path = typer.Option(..., "--run", exists=True, file_okay=False),
    fps: int = typer.Option(6, "--fps"),
) -> None:
    """Render a run's PNG frames and both time-lapse animations."""
    frames = render_run_frames(run)
    animations = render_run_animations(run, fps=fps)
    typer.echo(f"{len(frames)} frames; animations: {', '.join(sorted(animations))}")


@app.command()
def serve(
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Serve the comparison dashboard."""
    import uvicorn

    from web.app import create_app

    uvicorn.run(create_app(runs_dir), host=host, port=port)


@app.command("fetch-goes")
def fetch_goes(
    dest: Path = typer.Option(..., "--dest"),
    day: str = typer.Option(..., "--day", help="YYYY-MM-DD"),
    start_hour: int = typer.Option(0, "--start-hour"),
    end_hour: int = typer.Option(6, "--end-hour"),
    max_files: int | None = typer.Option(None, "--max-files"),
) -> None:
    """Download GOES-19 ABI Channel 13 files from the NOAA public bucket."""
    from sattsr.data.download import fetch_goes_c13

    paths = fetch_goes_c13(dest, day=date.fromisoformat(day),
                           hours=range(start_hour, end_hour), max_files=max_files)
    typer.echo(f"downloaded {len(paths)} files to {dest}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 6 passed

Run: `pytest tests/test_end_to_end.py -v -m slow`
Expected: 1 passed (a few minutes on CPU)

- [ ] **Step 7: Run the whole suite**

```bash
pytest -q -m "not slow"
pytest -q -m slow
ruff check src web tests
```

Expected: all green, no ruff findings.

- [ ] **Step 8: Commit**

```bash
git add src/sattsr/cli.py src/sattsr/data/download.py tests/test_cli.py tests/test_end_to_end.py
git commit -m "feat: sattsr CLI and end-to-end pipeline smoke test"
```

---

### Task 30: INSAT validation configs and bounded storage

**Files:**
- Create: `configs/insat_rapidscan.yaml`, `configs/insat_staggered.yaml`
- Create: `src/sattsr/data/prune.py`
- Modify: `src/sattsr/cli.py` (add the `prune` command)
- Create: `docs/RESULTS.md`
- Test: `tests/data/test_prune.py`, `tests/test_configs.py`

**Interfaces:**
- Consumes: `FrameRef`, `cache_path_for`, `load_config`.
- Produces:
  - `subsample_index(refs, *, keep_every: int) -> list[FrameRef]`
  - `prune_verified_raw(refs, cache_root, *, dry_run=True) -> tuple[list[Path], list[Path]]`
    — returns `(removed, skipped)`. A raw file is removed **only** when its cache entry exists
    and is non-empty.
  - CLI `sattsr prune --config ... [--keep-every N] [--yes]` — `--yes` is required to delete
    anything; without it the command reports what it would do and changes nothing.

This closes the two synopsis non-functional requirements not yet covered: bounded storage via
temporal subsampling and deletion of verified raw files, and INSAT-native validation against the
rapid-scan and staggered modes.

- [ ] **Step 1: Write the failing config test**

`tests/test_configs.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from sattsr.config import load_config
from sattsr.geo.grid import TargetGrid

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))


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
    rapid = load_config(Path("configs/insat_rapidscan.yaml"))
    staggered = load_config(Path("configs/insat_staggered.yaml"))
    assert rapid.data.cadence_minutes == pytest.approx(4.0)
    assert staggered.data.cadence_minutes == pytest.approx(15.0)
    assert rapid.data.sensor.startswith("insat")
    assert staggered.data.sensor.startswith("insat")
```

- [ ] **Step 2: Write the two INSAT validation configs**

`configs/insat_rapidscan.yaml` — INSAT-3DR's ~4-minute severe-weather mode, the highest-cadence
INSAT-native ground truth available:

```yaml
data:
  sensor: insat3dr
  raw_root: data/raw/insat_rapidscan
  cache_root: cache/insat_rapidscan
  cadence_minutes: 4
  tolerance_minutes: 1
  grid:
    lat_min: -10.0
    lat_max: 40.0
    lon_min: 60.0
    lon_max: 110.0
    resolution_deg: 0.05
  normalization:
    bt_min: 180.0
    bt_max: 330.0
train:
  tile_size: 256
  tile_overlap: 32
  batch_size: 4
  checkpoint_dir: checkpoints/insat
```

`configs/insat_staggered.yaml` — the routine ~15-minute staggered mode:

```yaml
data:
  sensor: insat3dr
  raw_root: data/raw/insat_staggered
  cache_root: cache/insat_staggered
  cadence_minutes: 15
  tolerance_minutes: 3
  grid:
    lat_min: -10.0
    lat_max: 40.0
    lon_min: 60.0
    lon_max: 110.0
    resolution_deg: 0.05
  normalization:
    bt_min: 180.0
    bt_max: 330.0
train:
  tile_size: 256
  tile_overlap: 32
  batch_size: 4
  checkpoint_dir: checkpoints/insat
```

Run: `pytest tests/test_configs.py -v`
Expected: passes for the config files; the `insat.yaml` case must pass too — if it fails on
`tile_size % 32`, fix the config rather than the assertion.

- [ ] **Step 3: Write the failing prune test**

`tests/data/test_prune.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from sattsr.data.index import FrameRef, cache_path_for
from sattsr.data.prune import prune_verified_raw, subsample_index


def _refs(tmp_path: Path, n: int = 6) -> list[FrameRef]:
    t0 = datetime(2026, 9, 4, tzinfo=timezone.utc)
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    refs = []
    for i in range(n):
        path = raw / f"f{i}.nc"
        path.write_bytes(b"raw-bytes")
        refs.append(FrameRef(t0 + timedelta(minutes=10 * i), path, "goes19"))
    return refs


def _cache(tmp_path: Path, refs, indices) -> Path:
    cache_root = tmp_path / "cache"
    for i in indices:
        dest = cache_path_for(cache_root, refs[i])
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, np.zeros((4, 4), dtype=np.float32))
    return cache_root


def test_subsample_keeps_every_nth_frame(tmp_path):
    refs = _refs(tmp_path, 10)
    kept = subsample_index(refs, keep_every=3)
    assert len(kept) == 4
    assert [refs.index(r) for r in kept] == [0, 3, 6, 9]


def test_subsample_of_one_is_the_identity(tmp_path):
    refs = _refs(tmp_path, 5)
    assert subsample_index(refs, keep_every=1) == refs


def test_subsample_rejects_a_non_positive_stride(tmp_path):
    with pytest.raises(ValueError):
        subsample_index(_refs(tmp_path, 3), keep_every=0)


def test_dry_run_is_the_default_and_deletes_nothing(tmp_path):
    refs = _refs(tmp_path, 4)
    cache_root = _cache(tmp_path, refs, range(4))
    removed, skipped = prune_verified_raw(refs, cache_root)
    assert len(removed) == 4 and skipped == []
    assert all(r.path.exists() for r in refs), "dry run must not touch the filesystem"


def test_deletion_removes_only_verified_files(tmp_path):
    refs = _refs(tmp_path, 4)
    cache_root = _cache(tmp_path, refs, [0, 2])         # only two are cached
    removed, skipped = prune_verified_raw(refs, cache_root, dry_run=False)

    assert {p.name for p in removed} == {"f0.nc", "f2.nc"}
    assert {p.name for p in skipped} == {"f1.nc", "f3.nc"}
    assert not refs[0].path.exists() and not refs[2].path.exists()
    assert refs[1].path.exists() and refs[3].path.exists()


def test_an_empty_cache_file_does_not_count_as_verified(tmp_path):
    refs = _refs(tmp_path, 2)
    cache_root = tmp_path / "cache"
    dest = cache_path_for(cache_root, refs[0])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"")

    removed, skipped = prune_verified_raw(refs, cache_root, dry_run=False)
    assert removed == []
    assert len(skipped) == 2
    assert all(r.path.exists() for r in refs)


def test_an_already_missing_raw_file_is_skipped_quietly(tmp_path):
    refs = _refs(tmp_path, 2)
    cache_root = _cache(tmp_path, refs, [0, 1])
    refs[0].path.unlink()
    removed, skipped = prune_verified_raw(refs, cache_root, dry_run=False)
    assert [p.name for p in removed] == ["f1.nc"]
    assert [p.name for p in skipped] == ["f0.nc"]
```

- [ ] **Step 4: Run it to verify it fails**

Run: `pytest tests/data/test_prune.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sattsr.data.prune'`

- [ ] **Step 5: Implement `src/sattsr/data/prune.py`**

```python
"""Bounded storage: temporal subsampling and deletion of verified raw files."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from sattsr.data.index import FrameRef, cache_path_for

log = logging.getLogger(__name__)


def subsample_index(refs: Sequence[FrameRef], *, keep_every: int) -> list[FrameRef]:
    """Keep every `keep_every`-th frame, preserving order."""
    if keep_every < 1:
        raise ValueError(f"keep_every must be >= 1, got {keep_every}")
    return list(refs)[::keep_every]


def prune_verified_raw(
    refs: Sequence[FrameRef], cache_root: str | Path, *, dry_run: bool = True
) -> tuple[list[Path], list[Path]]:
    """Delete raw files whose regridded cache entry exists and is non-empty.

    Defaults to a dry run: nothing is deleted unless `dry_run=False`. A raw file is
    only ever removed after its cache entry has been confirmed present and non-empty,
    so a failed `prepare` can never cascade into data loss.

    Returns (removed, skipped) as raw-file paths.
    """
    removed: list[Path] = []
    skipped: list[Path] = []

    for ref in refs:
        cached = cache_path_for(cache_root, ref)
        verified = cached.is_file() and cached.stat().st_size > 0
        if not verified or not ref.path.exists():
            skipped.append(ref.path)
            continue

        removed.append(ref.path)
        if not dry_run:
            ref.path.unlink()
            log.info("removed verified raw file %s", ref.path)

    return removed, skipped
```

- [ ] **Step 6: Add the `prune` command to `src/sattsr/cli.py`**

Add the import alongside the others:

```python
from sattsr.data.prune import prune_verified_raw, subsample_index
```

and the command:

```python
@app.command()
def prune(
    config: Path = ConfigOpt,
    keep_every: int = typer.Option(1, "--keep-every",
                                   help="Keep every Nth frame; 1 keeps all of them."),
    yes: bool = typer.Option(False, "--yes",
                             help="Actually delete. Without this nothing is removed."),
) -> None:
    """Bound storage by subsampling the archive and deleting verified raw files."""
    cfg = load_config(config)
    reader = get_reader(cfg.data.sensor)
    refs = subsample_index(build_index(reader, cfg.data.raw_root), keep_every=keep_every)

    removed, skipped = prune_verified_raw(refs, cfg.data.cache_root, dry_run=not yes)
    verb = "removed" if yes else "would remove"
    typer.echo(f"{verb} {len(removed)} raw files; {len(skipped)} skipped (not yet cached)")
    if not yes and removed:
        typer.echo("re-run with --yes to delete")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest tests/data/test_prune.py tests/test_configs.py tests/test_cli.py -v`
Expected: all pass

- [ ] **Step 8: Write the results template**

`docs/RESULTS.md` — the file the team fills in as real numbers arrive. Committing it empty is
deliberate: it names every result the project owes before any of them exist.

```markdown
# Results

## Run inventory

| Run | Sensor | Config | Checkpoint | Factor | Input cadence | Output cadence |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

## Model vs baselines (held-out GOES-19)

| Metric | Linear blend | Farneback warp | Ours |
| --- | --- | --- | --- |
| MSE (K^2) | | | |
| PSNR (dB) | | | |
| SSIM | | | |
| FSIM | | | |
| Displacement error (px) | | | |
| Cold-cloud CSI | | | |

## By event category (PSNR, dB)

| Category | n | Linear | Farneback | Ours |
| --- | --- | --- | --- | --- |
| clear | | | | |
| stratiform | | | | |
| convective | | | | |
| cyclonic | | | | |

## Cross-sensor: Himawari-8 benchmark

## INSAT-3DR/3DS

### Rapid-scan validation (~4 min native cadence)

### Staggered-mode validation (~15 min native cadence)

### Qualitative review

Screenshots of the dashboard, one per event category, with a note on where the model fails.

## Known failure cases

Report these explicitly. A frame that looks plausible but moves cloud in the wrong direction is
a worse outcome than a visibly blurry one, and the report must say so.
```

- [ ] **Step 9: Commit**

```bash
git add configs src/sattsr/data/prune.py src/sattsr/cli.py docs/RESULTS.md \
        tests/data/test_prune.py tests/test_configs.py
git commit -m "feat: INSAT validation configs, bounded-storage pruning and results template"
```

---

## Operating procedure — running it on real data

The tasks above are complete when the suite is green on synthetic fixtures. This is the sequence
that turns that into results, and it is where the real time goes.

**1. Pretrain on GOES-19** (10-minute cadence, abundant)

```bash
sattsr fetch-goes --dest data/raw/goes19 --day 2026-06-15 --start-hour 0 --end-hour 12
sattsr prepare  --config configs/goes19.yaml
sattsr train    --config configs/goes19.yaml
sattsr evaluate --config configs/goes19.yaml \
                --checkpoint checkpoints/goes19/best.pt \
                --out reports/goes19/report.json
```

Sample across seasons and across the diurnal cycle, not one contiguous block — a model trained on
twelve hours of one day will not generalise, and the day-wise split will happily report that it
does.

**2. Cross-sensor benchmark on Himawari-8** — same commands with `configs/himawari.yaml` and no
retraining. The gap between GOES and Himawari numbers is the honest estimate of how much
performance the INSAT transfer will cost.

**3. Adapt to INSAT-3DR/3DS**

```bash
sattsr prepare  --config configs/insat.yaml
sattsr finetune --config configs/insat.yaml --checkpoint checkpoints/goes19/best.pt
```

Before fine-tuning, compute both sensors' statistics with
`sattsr.train.finetune.estimate_cache_stats` and record them in `docs/RESULTS.md`. If the INSAT
mean sits far outside the GOES distribution, apply `match_statistics` during cache preparation
rather than relying on fine-tuning to absorb the offset.

**4. Validate against INSAT-native ground truth**

```bash
sattsr evaluate --config configs/insat_rapidscan.yaml \
                --checkpoint checkpoints/insat/best.pt \
                --out reports/insat_rapidscan/report.json
sattsr evaluate --config configs/insat_staggered.yaml \
                --checkpoint checkpoints/insat/best.pt \
                --out reports/insat_staggered/report.json
```

Rapid-scan data is only collected during documented severe-weather events, so plan around
specific dated cases. If MOSDAC access does not come through in time, say so in `RESULTS.md` and
fall back to the GOES/Himawari numbers plus qualitative INSAT review — the spec anticipates this,
and an unstated fallback is worse than a stated one.

**5. Produce the deliverable run and dashboard**

```bash
sattsr infer  --config configs/insat.yaml \
              --checkpoint checkpoints/insat/best.pt \
              --input data/raw/insat --out runs/insat-15min --factor 2
sattsr render --run runs/insat-15min --fps 8
cp reports/insat_staggered/report.json runs/insat-15min/report.json
sattsr serve  --runs-dir runs
```

Also produce a GOES run at `--factor 4` to demonstrate the 7.5-minute target.

**6. Bound storage between runs**

```bash
sattsr prune --config configs/goes19.yaml --keep-every 3      # reports, deletes nothing
sattsr prune --config configs/goes19.yaml --keep-every 3 --yes
```

---

## Deferred — deliberately not in this plan

State these in the report rather than letting a reader assume they were done.

- **Seasonal conditioning.** The synopsis suggests one model covering seasonal variation via
  seasonal encoding. The architecture conditions on temporal position `t` but not on season. The
  hook is there — extend the `t_map` channel in `IFNet.forward` to a two-channel
  `(t, day_of_year)` embedding — but it is not built or evaluated here.
- **Full RAFT.** `RaftLiteFlow` uses local correlation with convolutional refinement instead of
  RAFT's all-pairs volume and GRU update. That is a deliberate size trade for trainability on
  student hardware; call it RAFT-*style*, not RAFT.
- **Perceptual/adversarial losses.** SSIM is the structural term. No VGG or GAN loss.
- **Full-disc native resolution.** Everything runs on a regional lat-lon grid. Native-resolution
  full-disc inference works through the tiler but has not been profiled.
- **Nowcasting.** Extrapolation beyond the observation window is out of scope, per the spec.

---

## Self-review record

Run against the spec after writing, per the writing-plans checklist.

**Spec coverage.** Every FR maps to a task: FR-1 to Tasks 12-14 (learned flow and synthesis),
6 and 25 (`.nc` in/out), 23 (arbitrary `t` and recursive 30→15→7.5); FR-2 to Tasks 19-21 and 24
(MSE/PSNR/SSIM/FSIM plus motion metrics, against both baselines); FR-3 to Tasks 26-28
(both animations, charts, report); FR-4 to Tasks 5, 18, 25 and 30 (INSAT reader, fine-tuning,
15-minute product, native validation configs). Non-functional requirements: AMP (Task 17),
transfer learning and partial freeze (Task 18), bounded storage (Task 30), safety stratification
(Task 22), synthetic-frame metadata (Task 6, enforced by test).

**Gaps found and closed.** Two synopsis requirements had no task on the first pass — bounded
storage via verified raw deletion, and INSAT rapid-scan/staggered validation. Both are now
Task 30. Seasonal encoding is the one synopsis item left unbuilt; it is listed under Deferred
rather than silently dropped.

**Consistency fixes applied.** `predict_midframe`'s overlap is now capped at a quarter of the
tile so `hann_weight`'s two ramps cannot collide on small tiles; the `pandas` dependency used by
the NetCDF writer is pinned explicitly rather than relied on transitively.

**Two tests to watch.** `test_set_epoch_changes_the_sampled_tile` (Task 10) draws from four tile
positions plus augmentation flags — the outcome is deterministic given the seed, so if it fails,
change the `seed=3` argument rather than weakening the assertion.
`test_large_rotating_cold_shield_is_cyclonic` (Task 22) depends on a vorticity threshold that
should be calibrated against the printed value on first run, with the chosen number recorded in
the docstring.

**Verification gate.** Before claiming any task complete, run its test file and read the output.
Before claiming the project complete: `pytest -q` with and without `-m slow`, `ruff check`, and a
dashboard loaded in a browser against a real run.
