from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from sattsr.config import Config, LossConfig, ModelConfig, NormalizationConfig, TrainConfig
from sattsr.data.dataset import TripletDataset
from sattsr.data.index import FrameRef
from sattsr.data.triplets import Triplet
from sattsr.losses.composite import CompositeLoss
from sattsr.models.interpolator import build_model
from sattsr.train.checkpoint import load_checkpoint, save_checkpoint
from sattsr.train.finetune import (
    build_finetune_optimizer,
    estimate_cache_stats,
    freeze_modules,
    prepare_finetune,
    unfreeze_all,
)
from sattsr.train.loop import EpochResult, fit, seed_everything, train_one_epoch, validate

CKPT_CFG = ModelConfig(base_channels=16, scales=[4, 2, 1], flow_channels=32, flow_radius=2,
                       flow_iters=2)
LOOP_CFG = ModelConfig(base_channels=8, scales=[2, 1], use_raft_init=False)
NORM = NormalizationConfig(bt_min=180.0, bt_max=330.0)


def _config() -> Config:
    return Config.model_validate(
        {"data": {"sensor": "goes19", "raw_root": "r", "cache_root": "c"},
         "model": CKPT_CFG.model_dump()}
    )


# --------------------------------------------------------------------------- checkpoint


def test_round_trip_restores_weights_and_metadata(tmp_path):
    model = build_model(CKPT_CFG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(tmp_path / "best.pt", model=model, optimizer=optimizer,
                           epoch=7, best_metric=31.5, config=_config())
    assert path.exists()

    restored = build_model(CKPT_CFG)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1.0)
    meta = load_checkpoint(path, restored, restored_opt)

    assert meta["epoch"] == 7
    assert meta["best_metric"] == pytest.approx(31.5)
    assert meta["config"]["data"]["sensor"] == "goes19"
    for (_, a), (_, b) in zip(model.state_dict().items(), restored.state_dict().items()):
        torch.testing.assert_close(a, b)
    assert restored_opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_loading_without_a_model_returns_metadata_only(tmp_path):
    path = save_checkpoint(tmp_path / "c.pt", model=build_model(CKPT_CFG), epoch=1,
                           best_metric=0.0)
    meta = load_checkpoint(path)
    assert meta["epoch"] == 1
    assert "model_state" in meta


def test_extra_payload_survives(tmp_path):
    path = save_checkpoint(tmp_path / "c.pt", model=build_model(CKPT_CFG), epoch=0,
                           best_metric=0.0, extra={"sensor_mean": 245.0})
    assert load_checkpoint(path)["extra"]["sensor_mean"] == pytest.approx(245.0)


def test_strict_false_tolerates_a_shape_change(tmp_path):
    path = save_checkpoint(tmp_path / "c.pt", model=build_model(CKPT_CFG), epoch=0,
                           best_metric=0.0)
    wider = build_model(ModelConfig(base_channels=32, scales=[4, 2, 1], flow_channels=32,
                                    flow_radius=2, flow_iters=2))
    with pytest.raises(RuntimeError):
        load_checkpoint(path, wider, strict=True)
    load_checkpoint(path, wider, strict=False)      # must not raise


def test_parent_directory_is_created(tmp_path):
    path = save_checkpoint(tmp_path / "deep" / "nested" / "c.pt", model=build_model(CKPT_CFG),
                           epoch=0, best_metric=0.0)
    assert path.exists()


# --------------------------------------------------------------------------- loop


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
    model = build_model(LOOP_CFG)
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
    model = build_model(LOOP_CFG)
    before = [p.detach().clone() for p in model.parameters()]
    result = validate(model, _loader(tmp_path), CompositeLoss(LossConfig()), torch.device("cpu"))
    assert np.isfinite(result.psnr)
    for a, b in zip(before, model.parameters()):
        torch.testing.assert_close(a, b)


def test_fit_writes_last_and_best_and_returns_best(tmp_path):
    model = build_model(LOOP_CFG)
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
    model = build_model(LOOP_CFG)
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
    model = build_model(LOOP_CFG)
    criterion = CompositeLoss(LossConfig(w_smooth=0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    ds = TripletDataset(_translating_triplets(tmp_path, n=1), NORM, tile_size=32, augment=False)
    item = ds[0]
    batch = {k: (v[None] if v.ndim else v.reshape(1)) for k, v in item.items()}

    first = None
    loss_value = float("nan")
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = criterion(model(batch["i0"], batch["i2"], batch["t"]), batch)
        loss.backward()
        optimizer.step()
        loss_value = float(loss)
        if step == 0:
            first = loss_value
    assert first is not None
    assert loss_value < 0.5 * first, f"loss went {first:.4f} -> {loss_value:.4f}"


# --------------------------------------------------------------------------- finetune


def test_freeze_modules_disables_only_matching_parameters():
    model = build_model(CKPT_CFG)
    frozen = freeze_modules(model, ["flow_net.encoder"])
    assert frozen > 0
    for name, param in model.named_parameters():
        assert param.requires_grad != name.startswith("flow_net.encoder")


def test_freeze_with_no_match_returns_zero():
    assert freeze_modules(build_model(CKPT_CFG), ["does.not.exist"]) == 0


def test_unfreeze_all_restores_training():
    model = build_model(CKPT_CFG)
    freeze_modules(model, ["flow_net"])
    unfreeze_all(model)
    assert all(p.requires_grad for p in model.parameters())


def test_optimizer_excludes_frozen_parameters():
    model = build_model(CKPT_CFG)
    freeze_modules(model, ["flow_net"])
    opt = build_finetune_optimizer(model, lr=1e-4)
    in_optimizer = {id(p) for group in opt.param_groups for p in group["params"]}
    for name, param in model.named_parameters():
        assert (id(param) in in_optimizer) == param.requires_grad, name


def test_optimizer_gives_the_head_a_higher_learning_rate():
    opt = build_finetune_optimizer(build_model(CKPT_CFG), lr=1e-4, head_prefixes=("refine",),
                                   head_multiplier=10.0)
    rates = sorted(group["lr"] for group in opt.param_groups)
    assert rates == [pytest.approx(1e-4), pytest.approx(1e-3)]


def test_frozen_parameters_do_not_move_during_a_step():
    model = build_model(CKPT_CFG)
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
    model = build_model(CKPT_CFG)
    opt = prepare_finetune(model, cfg)
    assert not any(p.requires_grad for n, p in model.named_parameters()
                   if n.startswith("flow_net.encoder"))
    assert sorted(g["lr"] for g in opt.param_groups) == [pytest.approx(5e-5),
                                                         pytest.approx(5e-4)]
