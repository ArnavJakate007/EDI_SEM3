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
