"""Prepare the GOES-19 cache and pretrain the interpolator on it.

Mirrors what `sattsr prepare` + `sattsr train` will do once the CLI lands; kept as a
script so training can start before the remaining pipeline tasks are finished.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import timedelta
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sattsr.config import load_config
from sattsr.data.dataset import TripletDataset
from sattsr.data.index import build_index, prepare_cache, save_index
from sattsr.data.triplets import build_triplets, split_triplets
from sattsr.geo.grid import TargetGrid
from sattsr.io.registry import get_reader
from sattsr.losses.composite import CompositeLoss
from sattsr.models.interpolator import build_model
from sattsr.train.loop import fit, seed_everything

log = logging.getLogger("train_goes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/goes19.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.tile_size is not None:
        cfg.train.tile_size = args.tile_size
    cfg.train.num_workers = args.num_workers

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )
    log.info("device=%s", device)
    seed_everything(cfg.train.seed)

    reader = get_reader(cfg.data.sensor)
    grid = TargetGrid.from_config(cfg.data.grid)

    refs = build_index(reader, cfg.data.raw_root)
    log.info("found %d raw files", len(refs))
    if args.skip_prepare:
        from sattsr.data.index import cache_path_for

        refs = [
            type(r)(r.timestamp, cache_path_for(cfg.data.cache_root, r), r.sensor)
            for r in refs
        ]
        refs = [r for r in refs if r.path.exists()]
    else:
        t0 = time.time()
        refs = prepare_cache(reader, refs, grid, cfg.data.cache_root, progress=True)
        log.info("cached %d frames in %.0fs", len(refs), time.time() - t0)
    save_index(refs, Path(cfg.data.cache_root) / "index.json")

    triplets = build_triplets(
        refs,
        step=timedelta(minutes=cfg.data.cadence_minutes),
        tolerance=timedelta(minutes=cfg.data.tolerance_minutes),
    )
    train_t, val_t = split_triplets(
        triplets, val_fraction=cfg.train.val_fraction, seed=cfg.train.seed
    )
    if not val_t:
        cut = max(1, int(len(train_t) * 0.8))
        train_t, val_t = train_t[:cut], train_t[cut:] or train_t[-1:]
    log.info("triplets: %d total -> %d train / %d val", len(triplets), len(train_t), len(val_t))

    def make(items, augment):
        return TripletDataset(
            items, cfg.data.normalization, cfg.train.tile_size,
            augment=augment, seed=cfg.train.seed,
        )

    common = {"num_workers": cfg.train.num_workers, "pin_memory": device.type == "cuda",
              "persistent_workers": cfg.train.num_workers > 0}
    train_loader = DataLoader(make(train_t, True), batch_size=cfg.train.batch_size,
                              shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(make(val_t, False), batch_size=cfg.train.batch_size,
                            shuffle=False, **common)

    model = build_model(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model parameters: %.2fM", n_params / 1e6)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    history: list[dict[str, float]] = []

    def on_epoch(epoch, tr, va):
        history.append({
            "epoch": epoch, "train_loss": tr.loss, "train_psnr": tr.psnr,
            "val_loss": va.loss, "val_psnr": va.psnr,
        })
        Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.train.checkpoint_dir) / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch {epoch + 1}/{cfg.train.epochs} "
            f"train_loss={tr.loss:.4f} train_psnr={tr.psnr:.2f} "
            f"val_loss={va.loss:.4f} val_psnr={va.psnr:.2f}",
            flush=True,
        )

    best = fit(
        model, train_loader, val_loader, CompositeLoss(cfg.loss), optimizer, device,
        epochs=cfg.train.epochs, checkpoint_dir=cfg.train.checkpoint_dir,
        amp=cfg.train.amp, config=cfg, scheduler=scheduler, on_epoch=on_epoch,
    )
    best_psnr = max((h["val_psnr"] for h in history), default=float("nan"))
    print(f"DONE best checkpoint: {best}  best val PSNR: {best_psnr:.2f} dB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
