from __future__ import annotations

import pytest
import torch

from sattsr.config import LossConfig
from sattsr.losses.composite import CompositeLoss
from sattsr.losses.functional import (
    charbonnier,
    flow_smoothness,
    radiometric_consistency,
    warp_consistency,
)
from sattsr.losses.ssim import SSIM, SSIMLoss

# --------------------------------------------------------------------------- functional


def test_charbonnier_is_near_zero_for_identical_inputs():
    x = torch.rand(2, 1, 8, 8)
    assert float(charbonnier(x, x)) < 2e-3


def test_charbonnier_grows_with_error():
    a, b = torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8)
    assert float(charbonnier(a, b)) > float(charbonnier(a, b * 0.5))


def test_charbonnier_mask_excludes_pixels():
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    target[..., 0, 0] = 100.0
    mask = torch.ones(1, 1, 4, 4)
    mask[..., 0, 0] = 0.0
    assert float(charbonnier(pred, target, mask=mask)) < 2e-3
    assert float(charbonnier(pred, target)) > 1.0


def test_charbonnier_all_zero_mask_does_not_divide_by_zero():
    value = charbonnier(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4),
                        mask=torch.zeros(1, 1, 4, 4))
    assert torch.isfinite(value)


def test_flow_smoothness_penalises_a_jumpy_field():
    image = torch.zeros(1, 1, 8, 8)
    smooth = torch.zeros(1, 4, 8, 8)
    jumpy = torch.zeros(1, 4, 8, 8)
    jumpy[:, :, :, 4:] = 5.0
    assert float(flow_smoothness(jumpy, image)) > float(flow_smoothness(smooth, image))


def test_flow_smoothness_is_relaxed_at_image_edges():
    flow = torch.zeros(1, 4, 8, 8)
    flow[:, :, :, 4:] = 5.0
    flat_image = torch.zeros(1, 1, 8, 8)
    edgy_image = torch.zeros(1, 1, 8, 8)
    edgy_image[:, :, :, 4:] = 1.0
    assert float(flow_smoothness(flow, edgy_image)) < float(flow_smoothness(flow, flat_image))


def test_warp_consistency_is_zero_when_the_two_views_agree():
    x = torch.rand(1, 1, 8, 8)
    assert float(warp_consistency(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_radiometric_consistency_rewards_the_time_weighted_mean():
    i0 = torch.zeros(1, 1, 8, 8)
    i2 = torch.ones(1, 1, 8, 8)
    t = torch.tensor([0.25])
    good = torch.full((1, 1, 8, 8), 0.25)
    bad = torch.full((1, 1, 8, 8), 0.9)
    assert float(radiometric_consistency(good, i0, i2, t)) == pytest.approx(0.0, abs=1e-6)
    assert float(radiometric_consistency(bad, i0, i2, t)) > 0.5


def test_every_loss_term_is_differentiable():
    pred = torch.rand(1, 1, 8, 8, requires_grad=True)
    other = torch.rand(1, 1, 8, 8)
    flow = torch.zeros(1, 4, 8, 8, requires_grad=True)
    total = (
        charbonnier(pred, other)
        + flow_smoothness(flow, pred)
        + warp_consistency(pred, other)
        + radiometric_consistency(pred, other, other, torch.tensor([0.5]))
    )
    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


# --------------------------------------------------------------------------- ssim


def test_ssim_of_an_image_with_itself_is_one():
    x = torch.rand(2, 1, 32, 32)
    assert float(SSIM()(x, x)) == pytest.approx(1.0, abs=1e-3)


def test_ssim_drops_when_noise_is_added():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 32, 32)
    noisy = (x + torch.randn_like(x) * 0.3).clamp(0, 1)
    assert float(SSIM()(x, noisy)) < 0.9


def test_ssim_loss_is_one_minus_ssim():
    x = torch.rand(1, 1, 32, 32)
    y = torch.rand(1, 1, 32, 32)
    assert float(SSIMLoss()(x, y)) == pytest.approx(1.0 - float(SSIM()(x, y)), abs=1e-6)


def test_ssim_loss_is_differentiable():
    x = torch.rand(1, 1, 32, 32, requires_grad=True)
    SSIMLoss()(x, torch.rand(1, 1, 32, 32)).backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0


def test_ssim_matches_skimage_within_tolerance():
    skimage = pytest.importorskip("skimage.metrics")
    torch.manual_seed(1)
    a = torch.rand(1, 1, 64, 64)
    b = (a + torch.randn_like(a) * 0.1).clamp(0, 1)
    ours = float(SSIM()(a, b))
    theirs = skimage.structural_similarity(
        a[0, 0].numpy(), b[0, 0].numpy(), data_range=1.0, gaussian_weights=True,
        sigma=1.5, use_sample_covariance=False,
    )
    assert ours == pytest.approx(theirs, abs=0.03)


# --------------------------------------------------------------------------- composite


def _out_and_batch(perfect: bool = False):
    torch.manual_seed(0)
    target = torch.rand(2, 1, 32, 32)
    pred = target.clone() if perfect else torch.rand(2, 1, 32, 32)
    out = {
        "pred": pred.requires_grad_(True),
        "merged": pred,
        "flow": torch.zeros(2, 4, 32, 32, requires_grad=True),
        "mask": torch.full((2, 1, 32, 32), 0.5),
        "warped0": target,
        "warped2": target,
        "residual": torch.zeros(2, 1, 32, 32),
    }
    batch = {
        "i0": torch.rand(2, 1, 32, 32),
        "i1": target,
        "i2": torch.rand(2, 1, 32, 32),
        "valid": torch.ones(2, 1, 32, 32),
        "t": torch.tensor([0.5, 0.5]),
    }
    return out, batch


def test_reports_every_component():
    total, parts = CompositeLoss(LossConfig())(*_out_and_batch())
    assert set(parts) == {"total", "recon", "ssim", "smooth", "consistency", "radiometric"}
    assert all(isinstance(v, float) for v in parts.values())
    assert parts["total"] == float(total)


def test_a_perfect_prediction_scores_lower_than_a_random_one():
    loss = CompositeLoss(LossConfig())
    good, _ = loss(*_out_and_batch(perfect=True))
    bad, _ = loss(*_out_and_batch(perfect=False))
    assert float(good) < float(bad)


def test_zero_weights_disable_terms():
    cfg = LossConfig(w_recon=1.0, w_ssim=0.0, w_smooth=0.0, w_consistency=0.0, w_radiometric=0.0)
    out, batch = _out_and_batch()
    total, parts = CompositeLoss(cfg)(out, batch)
    assert parts["total"] == pytest.approx(parts["recon"])
    assert float(total) > 0.0


def test_total_is_differentiable_through_pred_and_flow():
    out, batch = _out_and_batch()
    total, _ = CompositeLoss(LossConfig())(out, batch)
    total.backward()
    assert out["pred"].grad is not None and float(out["pred"].grad.abs().sum()) > 0.0
    assert out["flow"].grad is not None


def test_invalid_pixels_are_ignored():
    out, batch = _out_and_batch(perfect=True)
    out["pred"] = out["pred"].detach().clone()
    out["merged"] = out["pred"]
    out["pred"][:, :, :8, :] = 0.0            # corrupt a strip
    batch["valid"] = torch.ones(2, 1, 32, 32)
    with_all = float(CompositeLoss(LossConfig())(out, batch)[1]["recon"])
    batch["valid"][:, :, :8, :] = 0.0         # mask the same strip out
    with_mask = float(CompositeLoss(LossConfig())(out, batch)[1]["recon"])
    assert with_mask < with_all
