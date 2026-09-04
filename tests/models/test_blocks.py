from __future__ import annotations

import torch

from sattsr.models.blocks import ResBlock, backwarp, conv_block


def test_conv_block_preserves_spatial_size_by_default():
    block = conv_block(3, 8)
    out = block(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 8, 16, 16)


def test_conv_block_stride_halves_resolution():
    out = conv_block(3, 8, 3, 2, 1)(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 8, 8, 8)


def test_resblock_is_shape_preserving_and_differentiable():
    x = torch.randn(2, 8, 12, 12, requires_grad=True)
    out = ResBlock(8)(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_backwarp_with_zero_flow_is_the_identity():
    x = torch.randn(2, 1, 8, 8)
    torch.testing.assert_close(backwarp(x, torch.zeros(2, 2, 8, 8)), x, atol=1e-5, rtol=1e-5)


def test_backwarp_shifts_by_the_requested_pixel_offset():
    # a horizontal ramp: value == column index
    x = torch.arange(8, dtype=torch.float32).repeat(8, 1)[None, None]
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = 2.0                       # sample two pixels to the right
    out = backwarp(x, flow)
    torch.testing.assert_close(
        out[0, 0, 4, 1:5], torch.tensor([3.0, 4.0, 5.0, 6.0]), atol=1e-4, rtol=1e-4
    )


def test_backwarp_shifts_vertically():
    x = torch.arange(8, dtype=torch.float32)[:, None].repeat(1, 8)[None, None]
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 1] = 1.0
    out = backwarp(x, flow)
    torch.testing.assert_close(
        out[0, 0, 1:5, 4], torch.tensor([2.0, 3.0, 4.0, 5.0]), atol=1e-4, rtol=1e-4
    )


def test_backwarp_clamps_at_the_border():
    x = torch.ones(1, 1, 8, 8)
    out = backwarp(x, torch.full((1, 2, 8, 8), 20.0))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.ones_like(out), atol=1e-5, rtol=1e-5)


def test_backwarp_gradients_reach_both_inputs():
    x = torch.randn(1, 1, 8, 8, requires_grad=True)
    # must be a leaf tensor, or .grad is never populated
    flow = (torch.randn(1, 2, 8, 8) * 0.5).requires_grad_(True)
    backwarp(x, flow).sum().backward()
    assert x.grad is not None and flow.grad is not None
    assert torch.isfinite(flow.grad).all()
    assert float(flow.grad.abs().sum()) > 0.0
