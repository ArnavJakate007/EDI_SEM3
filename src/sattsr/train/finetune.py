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

    stacked = np.concatenate([load_cached(ref).ravel() for ref in items])
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
