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
    """Load a checkpoint, optionally restoring a model and optimizer in place.

    With `strict=False` any tensor whose name or shape does not line up is dropped,
    which is what cross-architecture fine-tuning needs.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        state = payload["model_state"]
        if not strict:
            own = model.state_dict()
            state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
