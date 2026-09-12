#!/usr/bin/env python
"""Multi-sensor pretraining on the high-cadence sensors (GOES-19 + Himawari-8).

Consumes the "train" and "val" DataLoaders from `sattsr.data.loaders`, so the
chronological holdout and the rapid-scan leak guard are enforced before a single
gradient step is taken.

    python src/sattsr/train/pretrain.py --config configs/train.yaml --epochs 1
    python src/sattsr/train/pretrain.py --config configs/train.yaml \
        --resume-from checkpoints/pretrain/last.pt

Choices worth knowing about
---------------------------
LR schedule: cosine annealing over the full epoch budget. A single long pretraining
run has no plateau signal to key a step schedule off, and cosine needs no tuning
beyond the epoch count that is already configured.

Logging: a CSV at `log_csv`, not TensorBoard. TensorBoard is not installed and would
be a new dependency for this one purpose; the CSV is greppable, diffable, survives in
git, and loads in one pandas call. Every loss component is logged separately for both
train and val, because the total alone cannot tell you which term is misbehaving.

Checkpoints: `checkpoints/pretrain/`, not `checkpoints/goes19/`. This run covers GOES
*and* Himawari, so labelling it with one sensor would misdescribe it. The per-sensor
configs keep their own `checkpoints/<sensor>/` for single-sensor runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sattsr.config import PretrainConfig, load_pretrain_config  # noqa: E402
from sattsr.data.loaders import build_dataloaders  # noqa: E402
from sattsr.data.normalize import RenormRegistry  # noqa: E402
from sattsr.losses.composite import CompositeLoss  # noqa: E402
from sattsr.models.interpolator import build_model  # noqa: E402
from sattsr.train.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from sattsr.train.loop import EpochResult, seed_everything, train_one_epoch, validate  # noqa: E402

log = logging.getLogger("pretrain")

CSV_BASE_COLUMNS = ("epoch", "lr", "seconds", "train_loss", "train_psnr",
                    "val_loss", "val_psnr")


def loss_component_names(config: PretrainConfig) -> list[str]:
    """Component names CompositeLoss reports, derived from the configured weights.

    LossConfig names its weights `w_recon`, `w_ssim`, ...; CompositeLoss reports the
    components as `recon`, `ssim`, ... plus `total`. Deriving one from the other keeps
    the CSV header in step with the loss if a term is ever added.
    """
    names = [f[2:] for f in config.loss.model_dump() if f.startswith("w_")]
    return sorted(names) + ["total"]


def config_hash(config: PretrainConfig) -> str:
    """Stable hash of the config, so a checkpoint identifies the run that made it."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_device(requested: str) -> torch.device:
    """Pick the compute device, falling back cleanly when CUDA is absent."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def load_renorm(config: PretrainConfig, repo_root: Path) -> RenormRegistry | None:
    """Load the renormalisation registry if one has been fitted."""
    if config.renorm_dir is None:
        return None
    path = Path(config.renorm_dir)
    if not path.is_absolute():
        path = repo_root / path
    try:
        registry = RenormRegistry.load(path)
    except FileNotFoundError:
        log.warning(
            "no renorm registry at %s -- continuing without cross-sensor "
            "renormalisation (run scripts/fit_renorm.py to create it)", path,
        )
        return None
    log.info("loaded renorm registry (%d map(s), reference %s)", len(registry),
             registry.reference)
    return registry


class CsvLogger:
    """Append one row per epoch, with every loss component broken out."""

    def __init__(self, path: Path, component_names: list[str]) -> None:
        self.path = path
        self.columns = list(CSV_BASE_COLUMNS)
        for split in ("train", "val"):
            self.columns += [f"{split}_{name}" for name in component_names]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(self.columns)

    def append(
        self, epoch: int, lr: float, seconds: float, train: EpochResult, val: EpochResult
    ) -> None:
        row = {
            "epoch": epoch, "lr": lr, "seconds": round(seconds, 2),
            "train_loss": train.loss, "train_psnr": train.psnr,
            "val_loss": val.loss, "val_psnr": val.psnr,
        }
        for name, value in train.components.items():
            row[f"train_{name}"] = value
        for name, value in val.components.items():
            row[f"val_{name}"] = value
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([row.get(c, "") for c in self.columns])


def log_epoch(epoch: int, epochs: int, lr: float, seconds: float,
              train: EpochResult, val: EpochResult) -> None:
    """Per-epoch summary, with each loss component named individually."""
    log.info(
        "epoch %d/%d  lr=%.2e  %.1fs  train_loss=%.4f train_psnr=%.2f  "
        "val_loss=%.4f val_psnr=%.2f",
        epoch + 1, epochs, lr, seconds, train.loss, train.psnr, val.loss, val.psnr,
    )
    for name in sorted(set(train.components) | set(val.components)):
        if name == "total":
            continue
        log.info(
            "    %-13s train=%.5f  val=%.5f",
            name, train.components.get(name, float("nan")),
            val.components.get(name, float("nan")),
        )


def run(
    config: PretrainConfig,
    *,
    epochs: int | None = None,
    device_name: str = "auto",
    resume_from: Path | None = None,
    repo_root: Path | None = None,
    limit_batches: int | None = None,
) -> Path:
    """Run pretraining and return the path to the best checkpoint."""
    repo_root = repo_root or REPO_ROOT
    epochs = epochs if epochs is not None else config.train.epochs
    seed_everything(config.train.seed)

    device = resolve_device(device_name)
    log.info("device=%s epochs=%d", device, epochs)

    registry = load_renorm(config, repo_root)
    loaders = build_dataloaders(config, registry, repo_root=repo_root)
    if "train" not in loaders:
        raise RuntimeError(
            "no training data. Run scripts/preprocess.py for the sensors named in "
            f"{[str(p) for p in config.splits.pretrain]} first."
        )
    train_loader: DataLoader = loaders["train"]
    val_loader: DataLoader = loaders.get("val") or train_loader
    if "val" not in loaders:
        log.warning("no validation split available; validating on the training split")

    model = build_model(config.model).to(device)
    criterion = CompositeLoss(config.loss).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    # T_max is the CONFIGURED epoch budget, not the `epochs` argument. --epochs 1 then
    # means "run one epoch of the planned schedule" rather than "anneal fully in one
    # epoch", which is what a smoke test and a resumed run both need.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.train.epochs, 1)
    )

    start_epoch, best_psnr = 0, -float("inf")
    if resume_from is not None:
        payload = load_checkpoint(resume_from, model, optimizer, map_location=str(device))
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_psnr = float(payload.get("best_metric", -float("inf")))

        # Without this the schedule restarts from its first step, so a resumed run
        # trains at the wrong learning rate for the rest of its life.
        scheduler_state = payload.get("extra", {}).get("scheduler_state")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        else:
            for _ in range(start_epoch):
                scheduler.step()
            log.warning("checkpoint carried no scheduler state; fast-forwarded %d step(s)",
                        start_epoch)

        log.info("resumed from %s at epoch %d (best_psnr=%.2f, lr=%.2e)",
                 resume_from, start_epoch, best_psnr,
                 optimizer.param_groups[0]["lr"])
        if payload.get("extra", {}).get("config_hash") not in (None, config_hash(config)):
            log.warning(
                "checkpoint was written with a different config than the one loaded; "
                "optimizer state may not correspond to this schedule"
            )

    ckpt_dir = Path(config.train.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = repo_root / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    use_amp = bool(config.train.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    log.info("mixed precision: %s", "on" if use_amp else "off (fp32)")

    csv_logger = None
    if config.log_csv is not None:
        csv_path = Path(config.log_csv)
        if not csv_path.is_absolute():
            csv_path = repo_root / csv_path
        csv_logger = CsvLogger(csv_path, loss_component_names(config))
        log.info("logging metrics to %s", csv_path)

    if limit_batches is not None:
        train_loader = _truncate(train_loader, limit_batches)
        val_loader = _truncate(val_loader, limit_batches)

    for epoch in range(start_epoch, start_epoch + epochs):
        started = time.perf_counter()
        for loader in (train_loader, val_loader):
            dataset = getattr(loader, "dataset", None)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_clip=1.0,
        )
        val_result = validate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        seconds = time.perf_counter() - started

        log_epoch(epoch, start_epoch + epochs, lr, seconds, train_result, val_result)
        if csv_logger is not None:
            csv_logger.append(epoch, lr, seconds, train_result, val_result)

        extra = {
            "config_hash": config_hash(config),
            "sensors": "pretrain",
            "scheduler_state": scheduler.state_dict(),
        }
        improved = val_result.psnr > best_psnr or not best_path.exists()
        if improved:
            best_psnr = max(best_psnr, val_result.psnr)

        # last.pt must carry the updated best, or resuming from it would reset the
        # high-water mark and let the next epoch overwrite a better best.pt.
        save_checkpoint(ckpt_dir / "last.pt", model=model, optimizer=optimizer,
                        epoch=epoch, best_metric=best_psnr, config=config, extra=extra)
        if improved:
            save_checkpoint(best_path, model=model, optimizer=optimizer, epoch=epoch,
                            best_metric=best_psnr, config=config, extra=extra)
            log.info("    new best val_psnr=%.2f -> %s", best_psnr, best_path)

    log.info("done. best checkpoint: %s (val_psnr=%.2f)", best_path, best_psnr)
    return best_path


class _TruncatedLoader:
    """Yields at most `n` batches. Used by --limit-batches for smoke tests."""

    def __init__(self, loader: DataLoader, n: int) -> None:
        self.loader, self.n = loader, n
        self.dataset = loader.dataset

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.n:
                return
            yield batch

    def __len__(self) -> int:
        return min(self.n, len(self.loader))


def _truncate(loader: DataLoader, n: int):
    return _TruncatedLoader(loader, n)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "train.yaml",
        help="Training plan YAML (default: configs/train.yaml). Supplies the split "
             "sources, loader settings, model/loss hyperparameters and LR schedule.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the config's epoch count. Useful for a one-epoch smoke test.",
    )
    parser.add_argument(
        "--device", default="auto", help="auto (default), cpu, or cuda.",
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Checkpoint to restore model and optimizer state from before training.",
    )
    parser.add_argument(
        "--limit-batches", type=int, default=None,
        help="Process at most N batches per epoch. For smoke tests, not real runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if not args.config.exists():
        log.error("no config at %s", args.config)
        return 2

    config = load_pretrain_config(args.config)
    try:
        run(config, epochs=args.epochs, device_name=args.device,
            resume_from=args.resume_from, limit_batches=args.limit_batches)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Re-exported for tests and callers that want the pieces rather than the CLI.
__all__ = ["CsvLogger", "config_hash", "loss_component_names", "main",
           "resolve_device", "run"]
