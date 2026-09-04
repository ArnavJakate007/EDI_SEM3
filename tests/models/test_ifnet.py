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
