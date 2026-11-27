"""Guards on decisions that were measured, not guessed - and once silently reverted.

Commit 74f40d1 reverted most of the 2026-11-19 pipeline audit by rewriting three
files from a stale snapshot instead of patching them. Nothing caught it: the unit
tests all passed, because each tested a function in isolation and none of them
started the trainer or checked a default.

Every assertion here corresponds to something that reversion undid. They are
deliberately blunt - a default value, an exception, an exit code - because the
failure mode was not subtle logic but a whole decision quietly disappearing.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# Importing the trainer pulls in model.finetune -> dated_seva -> seva, so any test
# that touches TrainConfig or Trainer needs SEVA present. Guarding those on torch
# alone made them fail, rather than skip, on a machine with torch and no SEVA.
needs_torch = pytest.mark.skipif(not _have("torch"), reason="needs torch")
needs_seva = pytest.mark.skipif(not _have("seva.model"),
                                reason="needs SEVA installed (scripts/setup_seva.sh)")


# ------------------------------------------------------------------- the entry point

@pytest.mark.skipif(not _have("seva.model"), reason="needs SEVA installed")
def test_the_trainer_cli_still_starts():
    """``--help`` exits zero. The single cheapest check there is.

    The trainer was unable to even parse its arguments for six commits, because
    probes.py imports a name from loss.py that the reversion had deleted. No unit
    test touched the entry point, so nothing noticed until a GPU job was queued.
    """
    done = subprocess.run([sys.executable, str(REPO / "scripts/train_dated_seva.py"), "--help"],
                          capture_output=True, text=True, timeout=600, cwd=REPO)
    assert done.returncode == 0, (
        f"train_dated_seva.py --help exited {done.returncode}:\n"
        f"{(done.stderr or done.stdout)[-2000:]}")
    assert "--dataset" in done.stdout


# ---------------------------------------------------------------- training precision

@needs_torch
def test_the_loss_defaults_to_fp32():
    """fp32 is the default because bf16 was measured to fail.

    Same-seed bf16 runs moved their weights in unrelated directions and a fixed
    sample's loss stalled after a few steps, while fp32 fell steadily
    (docs/findings.md, 2026-11-19). Flipping this back to bf16 costs nothing
    visible and quietly stops the model learning.

    loss.py imports only torch, so this half of the check runs everywhere - which
    is where it earns its keep, since this is the constant that was reverted.
    """
    from timemachine.training.loss import TRAIN_AUTOCAST_DTYPE

    assert TRAIN_AUTOCAST_DTYPE is None, "fp32 means autocast off, i.e. None"


@needs_seva
def test_the_trainer_defaults_to_fp32():
    from timemachine.training.trainer import TrainConfig

    assert TrainConfig.autocast == "fp32"


@needs_torch
def test_fp32_is_reachable_by_name():
    from timemachine.training.loss import AUTOCAST_DTYPES

    assert AUTOCAST_DTYPES["fp32"] is None
    assert set(AUTOCAST_DTYPES) >= {"fp32", "bf16", "fp16"}


@needs_torch
def test_the_autocast_helper_exists_and_disables_for_fp32():
    """probes.py imports this; deleting it broke every import of the trainer."""
    import torch

    from timemachine.training.loss import autocast

    with autocast(None):
        assert not torch.is_autocast_enabled()


# ------------------------------------------------------------------- the rank loss

@needs_torch
def test_rank_weight_without_detach_is_refused():
    """The guard on a bug that cannot be seen in a loss curve.

    Date FiLM reads the dates of the *last* forward pass when gradient
    checkpointing recomputes a block, so a reference pass that keeps its gradient
    makes the true-date gradients recompute against wrong-era dates.
    """
    from timemachine.training.loss import LossConfig

    with pytest.raises(ValueError, match="rank_detach"):
        LossConfig(rank_weight=0.1, rank_detach=False)

    LossConfig(rank_weight=0.1, rank_detach=True)        # must not raise


@needs_torch
def test_rank_reference_is_validated():
    from timemachine.training.loss import LossConfig

    for reference in ("wrong", "unknown"):
        LossConfig(rank_reference=reference)
    with pytest.raises(ValueError, match="rank_reference"):
        LossConfig(rank_reference="sideways")


@needs_torch
def test_the_reference_pass_runs_before_the_true_date_pass():
    """Order matters under gradient checkpointing, so it is asserted, not assumed.

    A fake model records the dates of each forward pass. The reference must come
    first, leaving the true dates as the ones a recomputed block will see.
    """
    import torch
    from torch import nn

    from timemachine.training.loss import LossConfig, training_loss

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
            self.calls: list[dict] = []

        def forward(self, x, t, cond, num_frames=None, **kw):
            self.calls.append({k: (v.clone() if torch.is_tensor(v) else v)
                               for k, v in kw.items()})
            return x * self.scale

    views, channels, size = 4, 4, 8
    model = Recorder()
    latents = torch.randn(views, channels, size, size)
    cond = {"replace": torch.zeros(views, channels + 1, size, size)}
    uncond = {"replace": torch.zeros(views, channels + 1, size, size)}
    interval = torch.tensor([[1900.0, 1900.0], [1910.0, 1910.0],
                             [2010.0, 2010.0], [2015.0, 2015.0]])
    is_input = torch.tensor([True, False, True, False])
    cfg = LossConfig(uncond_prob=0.0, date_dropout=0.0, date_jitter=False,
                     rank_weight=0.1, rank_detach=True, rank_prob=1.0, rank_min_sigma=0.0)

    loss, info = training_loss(model, torch.linspace(0.002, 80, 1000), latents, cond, uncond,
                               interval, cfg, torch.Generator().manual_seed(0), is_input=is_input)
    assert len(model.calls) == 2, "the rank loss needs a reference pass and a true-date pass"
    # With rank_reference="wrong" the reference sees swapped eras, the true pass does not.
    reference, true_pass = model.calls
    assert not torch.equal(reference["date"], true_pass["date"])
    assert torch.equal(true_pass["date"].sort().values, interval[:, 0].sort().values), \
        "the true dates must be the dates of the LAST pass"
    assert "rank_loss" in info


# ------------------------------------------------------------------- render probes

@needs_seva
def test_render_probes_pad_to_sevas_frame_count():
    """SEVA runs 21 frames per forward pass; unpadded renders were the pre-audit bug."""
    from timemachine.training.trainer import TrainConfig

    assert TrainConfig.render_frames == 21


# --------------------------------------------------------------- checkpoint safety

@needs_seva
def test_resuming_checks_the_training_precision():
    """Resuming a bf16 checkpoint into an fp32 run mixes precisions in one run.

    Asserted against the source because constructing a Trainer needs SEVA weights
    and a GPU; what matters is that the guard has not been deleted again.
    """
    import inspect

    from timemachine.training.trainer import Trainer

    source = inspect.getsource(Trainer.resume)
    assert "autocast" in source, "the precision guard is missing from Trainer.resume"
    assert "grad_norm_history" in source, "the spike guard's history is not being restored"


@needs_seva
def test_checkpoints_record_the_precision_they_were_trained_at():
    import inspect

    from timemachine.training.trainer import Trainer

    assert "autocast" in inspect.getsource(Trainer.save)


# --------------------------------------------------------- the appearance pathway

@needs_seva
def test_the_appearance_pathway_survived_the_repair():
    """The repair must add appearance conditioning without removing the audit."""
    from timemachine.training.loss import LossConfig, draw_appearance_validity   # noqa: F401
    from timemachine.training.trainer import TrainConfig

    assert hasattr(LossConfig, "appearance_dropout")
    assert hasattr(TrainConfig, "lr_appearance")
    assert hasattr(TrainConfig, "appearance_stats_samples")


@needs_torch
def test_both_rank_passes_see_the_same_appearance():
    """The rank loss compares dates, so the appearance must be identical in both.

    If the validity were drawn per pass, the comparison would partly measure the
    appearance being dropped rather than the date being wrong, and the rank signal
    would be noise.
    """
    import torch
    from torch import nn

    from timemachine.training.loss import LossConfig, training_loss

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
            self.calls: list[dict] = []

        def forward(self, x, t, cond, num_frames=None, **kw):
            self.calls.append({k: (v.clone() if torch.is_tensor(v) else v)
                               for k, v in kw.items()})
            return x * self.scale

    views, channels, size, features = 4, 4, 8, 14
    model = Recorder()
    cfg = LossConfig(uncond_prob=0.0, date_dropout=0.0, date_jitter=False,
                     appearance_dropout=0.0, rank_weight=0.1, rank_detach=True,
                     rank_prob=1.0, rank_min_sigma=0.0)
    loss, info = training_loss(
        model, torch.linspace(0.002, 80, 1000), torch.randn(views, channels, size, size),
        {"replace": torch.zeros(views, channels + 1, size, size)},
        {"replace": torch.zeros(views, channels + 1, size, size)},
        torch.tensor([[1900.0, 1900.0], [1910.0, 1910.0], [2010.0, 2010.0], [2015.0, 2015.0]]),
        cfg, torch.Generator().manual_seed(0),
        is_input=torch.tensor([True, False, True, False]),
        appearance=torch.randn(views, features))

    assert len(model.calls) == 2
    reference, true_pass = model.calls
    assert torch.equal(reference["appearance"], true_pass["appearance"])
    assert torch.equal(reference["appearance_valid"], true_pass["appearance_valid"])
    assert info["appearance_known"] is True
