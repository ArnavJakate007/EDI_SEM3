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
from sattsr.data.index import (
    FrameRef,
    build_index,
    load_or_scan_index,
    prepare_cache,
    save_index,
)
from sattsr.data.prune import prune_verified_raw, subsample_index
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


log = logging.getLogger(__name__)


def _cached_refs(config: Config, *, progress: bool = False) -> list[FrameRef]:
    """Frame references for this sensor, preferring the cache over the raw tree.

    The cache is consulted first because `--delete-raw-after-cache` reclaims raw
    granules as soon as their frames are cached: on a disk-constrained machine the
    cache is usually the only copy left, and enumerating from raw would report an
    empty dataset for a cache holding thousands of frames. Raw is still scanned (and
    anything missing regridded) when the cache has nothing, which is the first-run case.
    """
    cached = load_or_scan_index(config.data.cache_root, config.data.sensor)
    if cached:
        log.info("using %d cached frame(s) from %s", len(cached), config.data.cache_root)
        return cached

    reader = get_reader(config.data.sensor)
    grid = TargetGrid.from_config(config.data.grid)
    return prepare_cache(
        reader, build_index(reader, config.data.raw_root), grid, config.data.cache_root,
        progress=progress,
    )


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
    train_t, val_t = split_triplets(
        triplets, val_fraction=config.train.val_fraction, seed=config.train.seed
    )
    if not val_t:                       # single-day dataset: hold out the tail
        cut = max(1, int(len(train_t) * 0.8))
        train_t, val_t = train_t[:cut], train_t[cut:] or train_t[-1:]

    def make(items: list, augment: bool) -> TripletDataset:
        return TripletDataset(
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
    refs = _cached_refs(cfg, progress=True)
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
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
    triplets = build_triplets(
        refs,
        step=timedelta(minutes=cfg.data.cadence_minutes),
        tolerance=timedelta(minutes=cfg.data.tolerance_minutes),
    )
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
    from_cache: bool = typer.Option(
        False, "--from-cache",
        help="Read regridded frames from the .npy cache instead of raw granules. "
             "Required once --delete-raw-after-cache has reclaimed the raw files, "
             "and faster regardless since no reader or regridding is needed.",
    ),
) -> None:
    """Densify a series of observations and write a flagged NetCDF product."""
    cfg = load_config(config)
    manifest = run_inference(cfg, input_dir=input_dir, output_dir=out, checkpoint=checkpoint,
                             factor=factor, device=resolve_device(device), limit=limit,
                             from_cache=from_cache)
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


@app.command("fetch-goes")
def fetch_goes(
    dest: Path = typer.Option(..., "--dest"),
    day: str = typer.Option(..., "--day", help="YYYY-MM-DD"),
    start_hour: int = typer.Option(0, "--start-hour"),
    end_hour: int = typer.Option(6, "--end-hour"),
    every: int = typer.Option(1, "--every", help="Keep every Nth file (3 -> 30 min cadence)."),
    max_files: int | None = typer.Option(None, "--max-files"),
) -> None:
    """Download GOES-19 ABI Channel 13 files from the NOAA public bucket."""
    from sattsr.data.download import fetch_goes_c13

    paths = fetch_goes_c13(dest, day=date.fromisoformat(day),
                           hours=range(start_hour, end_hour), every=every,
                           max_files=max_files, progress=True)
    typer.echo(f"downloaded {len(paths)} files to {dest}")


if __name__ == "__main__":
    app()
