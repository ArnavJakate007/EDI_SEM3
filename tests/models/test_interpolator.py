from __future__ import annotations

import pytest
import torch

from sattsr.config import ModelConfig
from sattsr.models.interpolator import RefineNet, build_model


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
