"""Can the architecture learn at all?

The standard sanity check before trusting any real training run: drive the full
stack -- FrameInterpolator -> CompositeLoss -> AdamW -- at a single synthetic triplet
for a fixed number of steps and require the loss to fall substantially. A model that
cannot overfit one batch will never fit a dataset, and this catches a dead branch,
a detached graph or a sign error in minutes rather than after an overnight run.

Named `test_overfit.py` rather than `test_rife.py` because the repo's top-level model
lives in `models/interpolator.py` (`FrameInterpolator`) -- there is no `rife.py` to
mirror, and `tests/models/test_interpolator.py` already unit-tests that module.
"""

from __future__ import annotations

import pytest
import torch

from sattsr.config import LossConfig, ModelConfig
from sattsr.losses.composite import CompositeLoss
from sattsr.models.interpolator import build_model

SIZE = 64
STEPS = 100


def _small_model_config() -> ModelConfig:
    """A deliberately small model, so the test runs on CPU in seconds."""
    return ModelConfig(
        base_channels=16, scales=[2, 1], use_raft_init=True,
        flow_channels=16, flow_radius=2, flow_iters=2,
    )


def _translating_triplet(shift: int = 4):
    """A learnable triplet: one pattern translating at a constant rate.

    i1 sits exactly halfway between i0 and i2, so a correct flow estimate plus a
    correct warp reproduces it. If the model cannot fit this, the failure is
    architectural, not a data problem.
    """
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    base = (
        0.5
        + 0.25 * torch.sin(xx.float() / 7.0)
        + 0.2 * torch.cos(yy.float() / 9.0)
    ).clamp(0, 1)

    def rolled(by: int):
        return torch.roll(base, shifts=by, dims=1)[None, None].float()

    return {
        "i0": rolled(0),
        "i1": rolled(shift // 2),
        "i2": rolled(shift),
        "valid": torch.ones(1, 1, SIZE, SIZE),
        "t": torch.tensor([0.5]),
    }


@pytest.mark.slow
def test_model_overfits_a_single_batch():
    """Loss must fall substantially over 100 steps on one repeated triplet."""
    torch.manual_seed(0)
    batch = _translating_triplet()
    model = build_model(_small_model_config())
    criterion = CompositeLoss(LossConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    losses: list[float] = []
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["i0"], batch["i2"], batch["t"])
        loss, _ = criterion(out, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    first, last = losses[0], losses[-1]
    best = min(losses)
    print(f"\noverfit: first={first:.5f} last={last:.5f} best={best:.5f} "
          f"drop={(1 - last / first) * 100:.1f}%")

    assert all(loss == loss for loss in losses), "loss went NaN"
    assert last < first * 0.6, (
        f"loss only fell from {first:.5f} to {last:.5f} in {STEPS} steps; "
        f"the architecture may not be learning"
    )


@pytest.mark.slow
def test_reconstruction_term_specifically_improves():
    """The total could fall while the term that actually matters does not."""
    torch.manual_seed(0)
    batch = _translating_triplet()
    model = build_model(_small_model_config())
    criterion = CompositeLoss(LossConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    first_recon = last_recon = None
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["i0"], batch["i2"], batch["t"])
        loss, parts = criterion(out, batch)
        loss.backward()
        optimizer.step()
        if step == 0:
            first_recon = parts["recon"]
        last_recon = parts["recon"]

    print(f"\nrecon: first={first_recon:.5f} last={last_recon:.5f}")
    assert last_recon < first_recon * 0.7


def test_forward_preserves_shape_on_a_synthetic_batch():
    """Cheap shape contract check; runs without the 100-step loop."""
    batch = _translating_triplet()
    model = build_model(_small_model_config()).eval()
    with torch.no_grad():
        out = model(batch["i0"], batch["i2"], batch["t"])

    assert out["pred"].shape == batch["i1"].shape == (1, 1, SIZE, SIZE)
    assert out["pred"].dtype == torch.float32
    assert float(out["pred"].min()) >= 0.0 and float(out["pred"].max()) <= 1.0
    assert out["mask"].shape == (1, 1, SIZE, SIZE)
    assert out["flow"].shape[1] == 4, "bidirectional flow: two 2-channel fields"


def test_a_batch_of_two_runs_and_keeps_samples_independent():
    batch = _translating_triplet()
    model = build_model(_small_model_config()).eval()
    doubled = {k: torch.cat([v, v]) if v.ndim > 0 else v for k, v in batch.items()}
    doubled["t"] = torch.tensor([0.25, 0.75])

    with torch.no_grad():
        out = model(doubled["i0"], doubled["i2"], doubled["t"])

    assert out["pred"].shape == (2, 1, SIZE, SIZE)
    assert not torch.allclose(out["pred"][0], out["pred"][1]), (
        "different t values must give different frames"
    )
