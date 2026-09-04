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
    if mask is not None:
        mse = float((err * mask).sum() / mask.sum().clamp_min(1.0))
    else:
        mse = float(err.mean())
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
                assert optimizer is not None
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
        psnr_sum += (
            _batch_psnr(out["pred"].detach().float(), batch["i1"].float(), batch.get("valid")) * n
        )
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
    return _run_epoch(
        model, loader, criterion, optimizer, device, scaler=scaler, grad_clip=grad_clip
    )


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

        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )
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
