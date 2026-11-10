"""CPU tests for the training pieces: conditioning layout, loss, dates, probes."""
import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("seva.eval", reason="SEVA not installed (scripts/setup_seva.sh)")

from timemachine.training.conditioning import build_conditions, camera_rays  # noqa: E402
from timemachine.training.loss import (  # noqa: E402
    LossConfig, draw_dates, noise_prediction_loss, predict_noise, training_loss,
)
from timemachine.training.probes import RenderProbeSpec, select_probe_rows, swapped_years  # noqa: E402

T, C, H = 5, 4, 6
IS_INPUT = torch.tensor([True, True, False, False, False])


def conditions():
    latents = torch.randn(T, C, H, H)
    clip = torch.randn(1024)
    plucker = torch.randn(T, 6, H, H)
    return latents, *build_conditions(latents, clip, plucker, IS_INPUT)


def test_conditions_match_seva_layout():
    latents, cond, uncond = conditions()
    assert cond["crossattn"].shape == (T, 1, 1024) and cond["concat"].shape == (T, 7, H, H)
    assert torch.equal(cond["concat"][:, 0, 0, 0], IS_INPUT.float())
    assert torch.equal(cond["replace"][IS_INPUT, :C], latents[IS_INPUT])
    assert torch.all(cond["replace"][IS_INPUT, C] == 1) and torch.all(cond["replace"][~IS_INPUT] == 0)
    assert torch.all(uncond["crossattn"] == 0) and torch.all(uncond["replace"] == 0)
    assert torch.all(uncond["concat"][:, 0] == 0)
    assert torch.equal(uncond["dense_vector"], cond["dense_vector"])


class Recorder(nn.Module):
    """Stand-in network: remembers its inputs, predicts zero noise."""

    def forward(self, x, t, c, num_frames=None, date=None, date_valid=None):
        self.x, self.t, self.c, self.date = x, t, c, date
        return torch.zeros_like(x)


def test_predict_noise_replaces_inputs_and_scales_like_seva():
    latents, cond, _ = conditions()
    net, sigma = Recorder(), torch.full((T,), 3.0)
    noisy = latents + 3.0 * torch.randn_like(latents)
    _, replaced = predict_noise(net, noisy, sigma, torch.full((T,), 500), cond)
    c_in = 1 / (3.0 ** 2 + 1) ** 0.5
    torch.testing.assert_close(net.x[IS_INPUT], latents[IS_INPUT] * c_in)
    torch.testing.assert_close(net.x[~IS_INPUT], noisy[~IS_INPUT] * c_in)
    assert torch.equal(replaced, IS_INPUT) and "replace" not in net.c and "replace" in cond


def test_loss_covers_targets_only_unless_unconditional():
    latents, cond, uncond = conditions()
    sigmas, noise = torch.linspace(0.1, 50, 1000), torch.randn(T, C, H, H)
    expected_cond = noise[~IS_INPUT].pow(2).mean()          # zero prediction -> mean noise^2
    torch.testing.assert_close(noise_prediction_loss(Recorder(), sigmas, latents, cond, 10, noise), expected_cond)
    torch.testing.assert_close(noise_prediction_loss(Recorder(), sigmas, latents, uncond, 10, noise),
                               noise.pow(2).mean())


def test_dates_are_drawn_inside_intervals_and_dropped_together():
    interval = torch.tensor([[1890.0, 1900.0], [2015.0, 2016.0], [1950.0, 1951.0]])
    gen = torch.Generator().manual_seed(0)
    for _ in range(20):
        year, valid = draw_dates(interval, LossConfig(date_dropout=0.5), gen)
        assert torch.all((year >= interval[:, 0]) & (year < interval[:, 1]))
        assert valid.all() or not valid.any()
    year, _ = draw_dates(interval, LossConfig(date_jitter=False, date_dropout=0.0))
    torch.testing.assert_close(year, interval.mean(1))


def test_training_loss_passes_dates():
    latents, cond, uncond = conditions()
    net = Recorder()
    interval = torch.tensor([[1890.0, 1900.0]] * T)
    loss, info = training_loss(net, torch.linspace(0.1, 50, 1000), latents, cond, uncond, interval,
                               LossConfig(uncond_prob=0.0, date_dropout=0.0), torch.Generator().manual_seed(0))
    assert loss.ndim == 0 and not info["uncond"] and info["dates_known"]
    assert net.date.shape == (T,) and torch.all((net.date >= 1890) & (net.date < 1900))


def test_camera_rays_shape():
    c2w = torch.eye(4).repeat(3, 1, 1)
    c2w[1:, 0, 3] = torch.tensor([1.0, 2.0])
    K = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]).repeat(3, 1, 1)
    assert camera_rays(c2w, K, 1, 64, 64).shape == (3, 6, 8, 8)


def test_swapped_years_moves_targets_only():
    year = torch.tensor([2010.0, 1880.0, 1900.0, 2021.0])
    is_input = torch.tensor([True, True, False, False])
    assert swapped_years(year, is_input).tolist() == [2010.0, 1880.0, 2015.0, 1890.0]


class FakeDataset:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.dates = np.array([1880.0] * 6 + [2015.0] * 10 + [2021.0] * 6)
        self.camera_feats = rng.normal(size=(len(self.dates), 6))
        import pandas as pd
        self.views = pd.DataFrame({"file_key": [f"f{i}" for i in range(len(self.dates))]})


def test_probe_rows_come_from_the_requested_eras():
    ds = FakeDataset()
    spec = RenderProbeSpec("fire", target_range=(2019.3, 2025), input_range=(2010, 2019.28),
                           render_years=(2015, 2021), n_inputs=3, n_targets=4)
    rows = select_probe_rows(ds, spec)
    assert len(rows) == 7 and len(set(rows)) == 7
    assert np.all(ds.dates[rows[:3]] == 2015) and np.all(ds.dates[rows[3:]] == 2021)
    pinned = RenderProbeSpec("pin", (0, 1), (0, 1), (2000,), file_keys=("f3", "f0"))
    assert select_probe_rows(ds, pinned).tolist() == [3, 0]
    assert RenderProbeSpec.from_dict({"name": "x", "target_range": [1, 2], "input_range": [0, 1],
                                      "render_years": [1900, None]}).render_years == (1900, None)


def test_level_weights():
    from timemachine.training.loss import level_weight
    assert level_weight(10.0, LossConfig()) == 1.0
    assert level_weight(10.0, LossConfig(weighting="edm")) == 101.0
    assert level_weight(0.1, LossConfig(weighting="min_snr")) == pytest.approx(0.05)
    assert level_weight(2.0, LossConfig(weighting="min_snr")) == 1.0
    with pytest.raises(ValueError):
        LossConfig(weighting="nope")


def test_weighted_training_loss_reports_raw_mse():
    latents, cond, uncond = conditions()
    sigmas = torch.full((1000,), 3.0)
    loss, info = training_loss(Recorder(), sigmas, latents, cond, uncond, None,
                               LossConfig(weighting="edm"), torch.Generator().manual_seed(0))
    assert float(loss) == pytest.approx(10.0 * info["eps_mse"], rel=1e-5)


def test_divergence_guard_watches_loss_and_high_noise_error(monkeypatch, tmp_path):
    """Training stops when the probe loss or its high-noise error rises too far above its first value."""
    import types

    from timemachine.training import trainer as tr

    t = object.__new__(tr.Trainer)          # no model or data needed for the guard
    t.cfg = tr.TrainConfig(dataset="unused", run_dir=str(tmp_path), abort_loss_ratio=3.0, abort_x0_ratio=1.5)
    t.run_dir, t.step, t.sigmas, t.probe_baseline = tmp_path, 0, None, {}
    t.model = types.SimpleNamespace(eval=lambda: None, train=lambda: None)
    stats = {"timestep_rms": 0.3, "date_rms_2015": 0.0, "date_rms_unknown": 0.0, "date_rms_spread": 0.0}
    monkeypatch.setattr(tr, "finetune_stats", lambda model: stats)
    results = iter([{"loss_true": 0.020, "x0_mse_high_noise": 0.10},     # first values
                    {"loss_true": 0.059, "x0_mse_high_noise": 0.14},     # below both limits: keep going
                    {"loss_true": 0.021, "x0_mse_high_noise": 0.16}])    # colour drift: stop
    t.gap_probe = lambda model, sigmas: next(results)
    assert t.run_gap_probe() is None
    assert t.run_gap_probe() is None
    assert "x0_mse_high_noise" in t.run_gap_probe()
    results = iter([{"loss_true": 0.061, "x0_mse_high_noise": 0.10}])   # denoising loss above 3x: stop
    assert "loss_true" in t.run_gap_probe()
