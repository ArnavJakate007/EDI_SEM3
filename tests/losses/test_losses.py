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
from sattsr.losses.ssim import MSSSIM, SSIM, MSSSIMLoss, SSIMLoss

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


# ------------------------------------------------------------------------- MS-SSIM


def test_ms_ssim_of_an_image_with_itself_is_one():
    a = torch.rand(2, 1, 64, 64)
    assert float(MSSSIM()(a, a)) == pytest.approx(1.0, abs=1e-4)


def test_ms_ssim_drops_when_noise_is_added():
    torch.manual_seed(0)
    a = torch.rand(2, 1, 64, 64)
    noisy = (a + 0.3 * torch.randn_like(a)).clamp(0, 1)
    assert float(MSSSIM()(a, noisy)) < float(MSSSIM()(a, a))


def test_ms_ssim_loss_is_one_minus_ms_ssim():
    torch.manual_seed(1)
    a, b = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)
    assert float(MSSSIMLoss()(a, b)) == pytest.approx(1.0 - float(MSSSIM()(a, b)), abs=1e-6)


def test_ms_ssim_loss_is_differentiable():
    a = torch.rand(1, 1, 64, 64, requires_grad=True)
    MSSSIMLoss()(a, torch.rand(1, 1, 64, 64)).backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0.0


@pytest.mark.parametrize(("size", "expected_levels"), [(32, 2), (64, 3), (256, 5)])
def test_ms_ssim_adapts_its_pyramid_to_small_inputs(size, expected_levels):
    """A 32x32 tile must not error just because five scales do not fit in it."""
    metric = MSSSIM()
    assert metric.usable_levels(size, size) == expected_levels
    a = torch.rand(1, 1, size, size)
    assert float(metric(a, a)) == pytest.approx(1.0, abs=1e-4)


def test_ms_ssim_is_more_scale_sensitive_than_single_scale_ssim():
    """Blurring destroys fine detail; MS-SSIM should notice at least as much."""
    torch.manual_seed(2)
    a = torch.rand(1, 1, 64, 64)
    blurred = torch.nn.functional.avg_pool2d(a, 4)
    blurred = torch.nn.functional.interpolate(blurred, size=(64, 64), mode="nearest")
    assert float(MSSSIM()(a, blurred)) < 1.0
    assert float(SSIM()(a, blurred)) < 1.0


# ------------------------------------------------------- composite weight sensitivity


def test_changing_a_nonzero_weight_changes_the_total():
    """Not just zeroing a term -- the weighting must actually scale it."""
    out, batch = _out_and_batch()
    base = LossConfig(w_recon=1.0, w_ssim=0.25, w_smooth=0.05,
                      w_consistency=0.10, w_radiometric=0.05)
    heavier = base.model_copy(update={"w_ssim": 0.75})

    total_base, parts_base = CompositeLoss(base)(out, batch)
    total_heavy, parts_heavy = CompositeLoss(heavier)(out, batch)

    assert float(total_heavy) != pytest.approx(float(total_base))
    assert parts_heavy["ssim"] == pytest.approx(3.0 * parts_base["ssim"], rel=1e-5)


def _out_and_batch_all_terms_active():
    """Like `_out_and_batch`, but with every loss term genuinely non-zero.

    The plain fixture uses a zero flow field and identical warped views, which makes
    the smoothness and consistency terms exactly 0 -- so those two weights cannot
    move the total and nothing exercises them.
    """
    out, batch = _out_and_batch()
    torch.manual_seed(3)
    out["flow"] = torch.randn(2, 4, 32, 32, requires_grad=True)
    out["warped0"] = torch.rand(2, 1, 32, 32)
    out["warped2"] = torch.rand(2, 1, 32, 32)
    return out, batch


def test_the_active_fixture_really_activates_every_term():
    _, parts = CompositeLoss(LossConfig())(*_out_and_batch_all_terms_active())
    for name in ("recon", "ssim", "smooth", "consistency", "radiometric"):
        assert parts[name] > 0.0, f"{name} is inert in this fixture"


@pytest.mark.parametrize(
    "field", ["w_recon", "w_ssim", "w_smooth", "w_consistency", "w_radiometric"]
)
def test_every_weight_independently_moves_the_total(field):
    out, batch = _out_and_batch_all_terms_active()
    base = LossConfig()
    bumped = base.model_copy(update={field: getattr(base, field) + 0.5})
    total_base, _ = CompositeLoss(base)(out, batch)
    total_bumped, _ = CompositeLoss(bumped)(out, batch)
    assert float(total_bumped) > float(total_base), f"{field} has no effect on the total"


def test_every_component_is_finite_and_non_negative():
    """A NaN in any term silently poisons the whole run; catch it here."""
    for perfect in (True, False):
        _, parts = CompositeLoss(LossConfig())(*_out_and_batch(perfect=perfect))
        for name, value in parts.items():
            assert value == value, f"{name} is NaN"
            assert abs(value) != float("inf"), f"{name} is infinite"
            assert value >= 0.0, f"{name} is negative ({value})"


# ------------------------------------------- mixed-precision numerical safety


def test_ssim_computes_in_float32_even_for_half_inputs():
    """SSIM must not evaluate its variance terms in float16."""
    a = torch.rand(1, 1, 32, 32, dtype=torch.float16)
    b = torch.rand(1, 1, 32, 32, dtype=torch.float16)
    assert SSIM()(a, b).dtype == torch.float32
    assert MSSSIM()(a, b).dtype == torch.float32


def test_ssim_gradient_is_finite_on_a_nearly_constant_half_input():
    """The exact failure mode that silently froze a 40-epoch AMP run.

    Thermal-IR imagery is smooth, so E[x^2] and E[x]^2 are nearly equal over a window.
    Subtracting them in float16 is catastrophic cancellation, and the resulting
    near-zero (or negative) denominator makes the backward pass emit NaN. Those NaN
    gradients made GradScaler skip every optimizer step, so the model never updated
    while train/val loss still looked plausible.
    """
    base = torch.full((1, 1, 32, 32), 0.5)
    a = (base + 1e-4 * torch.randn(1, 1, 32, 32)).half().float().requires_grad_(True)
    target = (base + 1e-4 * torch.randn(1, 1, 32, 32)).half().float()

    loss = SSIMLoss(data_range=1.0)(a, target)
    loss.backward()
    assert torch.isfinite(loss), "SSIM went non-finite on a smooth input"
    assert a.grad is not None and torch.isfinite(a.grad).all(), "SSIM produced NaN grads"


def test_ssim_of_a_perfectly_constant_pair_is_finite():
    """Zero variance on both sides is the worst case for the denominator."""
    a = torch.full((1, 1, 24, 24), 0.3, requires_grad=True)
    b = torch.full((1, 1, 24, 24), 0.3)
    value = SSIM()(a, b)
    value.backward()
    assert torch.isfinite(value) and value > 0.99
    assert torch.isfinite(a.grad).all()


def test_composite_loss_gradients_are_finite_under_autocast():
    """End-to-end guard: every parameter must receive a usable gradient under AMP."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sattsr.config import ModelConfig
    from sattsr.models.interpolator import build_model

    torch.manual_seed(0)
    model = build_model(
        ModelConfig(base_channels=16, scales=[2, 1], flow_channels=16,
                    flow_radius=2, flow_iters=2)
    ).to(device)
    # A smooth, low-contrast scene -- the condition that triggers the cancellation.
    i0 = (0.5 + 0.01 * torch.randn(2, 1, 64, 64)).clamp(0, 1).to(device)
    i2 = (0.5 + 0.01 * torch.randn(2, 1, 64, 64)).clamp(0, 1).to(device)
    batch = {
        "i0": i0, "i2": i2, "i1": 0.5 * (i0 + i2),
        "valid": torch.ones(2, 1, 64, 64, device=device),
        "t": torch.tensor([0.5, 0.5], device=device),
    }

    with torch.autocast(device_type=device, enabled=True):
        out = model(batch["i0"], batch["i2"], batch["t"])
        loss, _ = CompositeLoss(LossConfig())(out, batch)
    loss.float().backward()

    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"{len(bad)} parameter(s) got non-finite gradients under autocast: {bad[:5]}"
