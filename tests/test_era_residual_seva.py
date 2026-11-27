"""Era-residualized appearance: the wiring through the model, config, trainer and CLI.

These need SEVA's real ResBlock (the FiLM hooks find blocks by type), and they
import it **without** an importorskip: on a machine without SEVA this file errors
rather than reporting green. A skip is indistinguishable from a pass, and this
suite has already hidden four bugs that way (CLAUDE.md). The encoder-level tests,
which need no SEVA, are in ``test_era_residual.py`` and run everywhere.
"""
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from seva.modules import layers
from torch import nn

from timemachine.model.dated_seva import DatedSGMWrapper
from timemachine.model.finetune import (
    FinetuneConfig, load_trainable_state_dict, prepare_for_finetuning, trainable_state_dict,
)
from timemachine.training.trainer import TrainConfig, Trainer

from .test_era_residual import era_data

REPO = Path(__file__).resolve().parents[1]
FEATURES = 14
CHANNELS = 32


class TinyResSeva(nn.Module):
    """SEVA's interface with two real ResBlocks, as in tests/test_appearance_training.py."""

    model_channels = 8

    def __init__(self):
        super().__init__()
        self.time_embed = nn.Linear(1, 4 * self.model_channels)
        self.input_blocks = nn.ModuleList([
            layers.ResBlock(CHANNELS, 4 * self.model_channels, CHANNELS, 6, 0.0)])
        self.middle_block = layers.ResBlock(CHANNELS, 4 * self.model_channels, CHANNELS, 6, 0.0)
        self.output_blocks = nn.ModuleList()
        # Projects back to the latent width, or the loss cannot subtract output from noise.
        self.head = nn.Conv2d(CHANNELS, CHANNELS - 2, 3, padding=1)

    def forward(self, x, t, y, dense_y, num_frames=None):
        emb = self.time_embed(t[:, None].float())
        h = self.input_blocks[0](x, emb, dense_y)
        return self.head(self.middle_block(h, emb, dense_y))


def tiny_model(remove_era=False, appearance="film"):
    # Seeded so every model shares one frozen base, as real runs share SEVA's weights;
    # otherwise two "identical" models differ before any checkpoint is loaded.
    torch.manual_seed(1234)
    model = DatedSGMWrapper(TinyResSeva())
    cfg = FinetuneConfig(lora_rank=0, date_injection="film", appearance_injection=appearance,
                         appearance_remove_era=remove_era, film_dim=32, appearance_dim=32,
                         grad_checkpointing=False)
    return model, cfg, prepare_for_finetuning(model, cfg)


def forward_args(n_views=4):
    torch.manual_seed(1)
    return (torch.randn(n_views, CHANNELS - 2, 4, 4),
            torch.tensor([10, 10, 500, 500][:n_views]),
            {"crossattn": torch.zeros(n_views, 1, 8),
             "dense_vector": torch.randn(n_views, 6, 4, 4),
             "concat": torch.zeros(n_views, 2, 4, 4)})


def trained(model, groups, seed=0):
    """Non-zero FiLM layers, so appearance actually moves the output."""
    torch.manual_seed(seed)
    for p in groups["film"] + groups["appearance"]:
        nn.init.normal_(p, std=0.05)
    return model


# ------------------------------------------------------------------ config

def test_remove_era_needs_an_appearance_pathway():
    with pytest.raises(ValueError, match="appearance"):
        FinetuneConfig(appearance_injection="none", appearance_remove_era=True)
    assert FinetuneConfig().appearance_remove_era is False


def test_the_flag_off_leaves_the_encoder_exactly_as_before():
    """Arm 2 (date + appearance FiLM) must be unchanged by this feature existing."""
    torch.manual_seed(0)
    model, _, groups = tiny_model(remove_era=False)
    enc = model.date_module.appearance_encoder
    assert not enc.removes_era
    # Nothing new is trainable: the buffer is not a parameter.
    assert all(p is not enc.era_directions for p in groups["appearance"])
    assert "era_directions" not in dict(enc.named_parameters())

    trained(model, groups)
    x, t, c = forward_args()
    look = torch.randn(4, FEATURES)
    before = model(x, t, c, date=torch.full((4,), 1900.), appearance=look, num_frames=4)
    enc.set_era_directions(torch.zeros(0, FEATURES))       # explicit k = 0
    after = model(x, t, c, date=torch.full((4,), 1900.), appearance=look, num_frames=4)
    assert torch.equal(before, after)


def test_prepare_does_not_fit_anything_by_itself():
    """Fitting needs data, which only the trainer has; a fresh model is k = 0 until then."""
    model, _, _ = tiny_model(remove_era=True)
    assert not model.date_module.appearance_encoder.removes_era


# ------------------------------------------------------------------ through the model

def test_era_moves_do_not_reach_the_model_output():
    model, _, groups = tiny_model(remove_era=True)
    stats, years = era_data()
    enc = model.date_module.appearance_encoder
    enc.fit_normalization(stats)
    from timemachine.model.appearance_film import fit_era_directions
    fit_era_directions(enc, stats, years)
    trained(model, groups)

    x, t, c = forward_args()
    look = stats[:4]
    era_step = (torch.randn(4, enc.era_directions.shape[0]) * 3) @ enc.era_directions * enc.scale
    date = torch.full((4,), 1900.)
    run = lambda a: model(x, t, c, date=date, appearance=a, num_frames=4)
    torch.testing.assert_close(run(look + era_step), run(look), atol=1e-5, rtol=1e-4)
    assert not torch.allclose(run(look), run(stats[4:8]))  # other appearance still matters


# ------------------------------------------------------------------ checkpoints

@pytest.mark.parametrize("k", [1, 6])
def test_a_model_checkpoint_with_era_directions_loads_into_a_fresh_model(k):
    """The path load_finetuned and Trainer.resume take: build fresh (k = 0), then load."""
    model, _, groups = tiny_model(remove_era=True)
    enc = model.date_module.appearance_encoder
    enc.fit_normalization(torch.randn(50, FEATURES) * 3 + 1)
    enc.set_era_directions(torch.randn(k, FEATURES))
    trained(model, groups)
    state = trainable_state_dict(model)
    assert state["appearance"]["encoder"]["era_directions"].shape == (k, FEATURES)

    restored, _, _ = tiny_model(remove_era=True)
    load_trainable_state_dict(restored, state)
    assert torch.equal(restored.date_module.appearance_encoder.era_directions, enc.era_directions)
    x, t, c = forward_args()
    look, date = torch.randn(4, FEATURES), torch.full((4,), 1900.)
    torch.testing.assert_close(restored(x, t, c, date=date, appearance=look, num_frames=4),
                               model(x, t, c, date=date, appearance=look, num_frames=4))


def test_an_arm_two_checkpoint_loads_into_an_era_model_and_vice_versa():
    plain, _, _ = tiny_model(remove_era=False)
    state = trainable_state_dict(plain)
    del state["appearance"]["encoder"]["era_directions"]      # as written before this change
    era_model, _, _ = tiny_model(remove_era=True)
    era_model.date_module.appearance_encoder.set_era_directions(torch.randn(3, FEATURES))
    load_trainable_state_dict(era_model, state)
    assert not era_model.date_module.appearance_encoder.removes_era

    era_model.date_module.appearance_encoder.set_era_directions(torch.randn(2, FEATURES))
    load_trainable_state_dict(plain, trainable_state_dict(era_model))
    assert plain.date_module.appearance_encoder.era_directions.shape == (2, FEATURES)


# ------------------------------------------------------------------ the trainer

class StatsDataset:
    """Samples of eight views, drawn from the synthetic era data."""

    def __init__(self):
        self.stats, self.years = era_data(n=1600)

    def __len__(self):
        return 200

    def __getitem__(self, i):
        rows = list(range(8 * i, 8 * i + 8))
        return {"appearance": self.stats[rows], "date_mid": torch.tensor(self.years[rows]),
                "file_keys": [f"v{r}" for r in rows]}


def test_the_trainer_fits_normalization_and_era_from_one_draw(tmp_path):
    model, _, _ = tiny_model(remove_era=True)
    enc = model.date_module.appearance_encoder
    cfg = TrainConfig(dataset="unused", run_dir=str(tmp_path), appearance_stats_samples=64,
                      era_stats_samples=150, seed=0)
    fake = SimpleNamespace(cfg=cfg,
                           dataset=StatsDataset(), run_dir=tmp_path)
    report = Trainer.fit_era_residual(fake, enc)

    assert enc.removes_era and report["dims_removed"] == enc.era_directions.shape[0]
    assert report["n_rows"] == 150 * 8                        # the larger of the two sample counts
    assert not torch.equal(enc.scale, torch.ones(FEATURES))   # normalization fitted, same draw
    written = json.loads((tmp_path / "era_directions.json").read_text())
    assert written["dims_removed"] == report["dims_removed"]
    np.testing.assert_allclose(written["directions"], enc.era_directions.numpy(), atol=1e-6)


def test_train_config_records_the_era_sample_count():
    cfg = TrainConfig(dataset="unused", run_dir="unused")
    assert cfg.era_stats_samples >= cfg.appearance_stats_samples


# ------------------------------------------------------------------ the real entry points

@pytest.fixture(scope="module")
def train_help() -> str:
    done = subprocess.run([sys.executable, str(REPO / "scripts/train_dated_seva.py"), "--help"],
                          capture_output=True, text=True, timeout=300, cwd=REPO)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def test_the_trainer_cli_offers_the_flag(train_help):
    assert "--appearance-remove-era" in train_help
    assert "--era-stats-samples" in train_help


def _sbatch_train_flags(path: Path) -> set[str]:
    text = path.read_text().replace("\\\n", " ")
    command = next(line for line in text.splitlines() if "train_dated_seva.py" in line
                   and not line.lstrip().startswith("#"))
    return set(re.findall(r"(?<!\S)--[\w-]+", command))


def test_every_flag_the_class_job_passes_exists(train_help):
    """The job script once passed six flags the trainer does not have, and would have
    died in argparse after queueing. Checked against the real --help, not a list."""
    offered = set(re.findall(r"--[\w-]+", train_help))
    passed = _sbatch_train_flags(REPO / "slurm/train_dated_seva_class.sbatch")
    assert "--appearance-remove-era" in passed
    assert passed <= offered, sorted(passed - offered)


def test_the_class_job_supplies_the_required_arguments():
    passed = _sbatch_train_flags(REPO / "slurm/train_dated_seva_class.sbatch")
    assert {"--dataset", "--run-name"} <= passed
