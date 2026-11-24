"""CPU tests for one MVGenMaster training step (``timemachine.mvgen.step``), on small fakes of the model.

Run from MVGenMaster's checkout, in its environment (see ``timemachine/mvgen/__init__.py``). The fake
model has the parts a step uses: a UNet with MVGenMaster's call, a VAE, a config, the real DDPM
scheduler, ``dates()`` and ``date_stats()``. MVGenMaster's camera and prior functions are replaced by
ones that return recognisable tensors of the right shapes, except in the last section, which runs the
real ones and compares the step with a line-by-line transcription of the authors' ``train.py``.
"""
import json
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

my_diffusers = pytest.importorskip("my_diffusers", reason="runs in MVGenMaster's environment, from its checkout")

# Under pytest the repo root is on the import path (its conftest.py), so ``src`` is the repo's own ``src``
# package and hides MVGenMaster's ``src`` folder, which ``timemachine.mvgen`` imports from. Let the package
# search that folder too.
import src  # noqa: E402

MVGEN_SRC = str(Path(my_diffusers.__file__).resolve().parents[1] / "src")
if MVGEN_SRC not in list(src.__path__):
    src.__path__.append(MVGEN_SRC)

import einops  # noqa: E402
import numpy as np  # noqa: E402
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # noqa: E402
from easydict import EasyDict  # noqa: E402
from my_diffusers.schedulers import DDPMScheduler  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from src.modules.camera import get_camera_embedding  # noqa: E402
from src.modules.position_encoding import get_3d_priors  # noqa: E402
from src.modules.schedulers import get_diffusion_scheduler  # noqa: E402

from timemachine.mvgen import step  # noqa: E402
from timemachine.mvgen.step import (  # noqa: E402
    LossConfig, autocast, build_inputs, cross_date_drops, cross_date_views, date_embedding_size, depth_for_priors,
    encode_images, predict, training_loss, v_prediction_loss,
)

N_VIEWS, N_REF, SIZE, LATENT = 6, 2, 16, 2          # views, references, image side, latent side (image / 8)
IS_REF = torch.arange(N_VIEWS) < N_REF
# References from 2000 and 2001. Targets: between them; near one of them; two that no reference is near.
DATE_MID = torch.tensor([2000.0, 2001.0, 2000.5, 2001.8, 2010.0, 1950.0])
CROSS_DATE = torch.tensor([False, False, False, False, True, True])
# With a period edge at 2001.5, the target of 2001.8 has a reference within a year but none in its period.
EDGE = (2001.5,)
CROSS_DATE_WITH_EDGE = torch.tensor([False, False, False, True, True, True])
SCALING = 0.5
# MVGenMaster's training scheduler: SD 2.1's, with its "snr_rescale" betas and zero terminal SNR.
SCHEDULER = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "snr_rescale",
             "snr_rescale": 4, "rescale_betas_zero_snr": True, "prediction_type": "v_prediction", "clip_sample": False}
# What MVGenMaster's camera and prior functions read from the released model's config.yaml.
CONFIG = {"camera_longest_side": 5.0, "normalize_extrinsic": True, "normalize_t": False, "fourier_embedding": False,
          "fourier_embed_dim": 18}
NO_DROPOUT = dict(date_dropout=0.0, camera_dropout=0.0, prior_dropout=0.0, prior_view_dropout=0.0, precision="fp32")


class FakeUNet(nn.Module):
    """MVGenMaster's UNet call on a 1x1 convolution of the latents; remembers what it was given."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1)
        self.dates = None                # (year, valid) while FakeMVGen.dates() is open
        self.seen = {}

    def forward(self, sample, timesteps, encoder_hidden_states, add_inputs=None, class_labels=None, coords=None,
                return_dict=True):
        out = self.conv(sample.flatten(0, 1))
        self.seen = dict(sample=sample, timesteps=timesteps, encoder_hidden_states=encoder_hidden_states,
                         add_inputs=add_inputs, class_labels=class_labels, coords=coords, dates=self.dates,
                         return_dict=return_dict, dtype=out.dtype)
        return (out,)


class FakeVAE(nn.Module):
    """A VAE whose posterior mean is the 8x8 average of the image (4th channel: its grey value)."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=SCALING)
        self.logvar = nn.Parameter(torch.tensor(-2.0))

    def encode(self, images):
        mean = F.avg_pool2d(images, 8)
        mean = torch.cat([mean, mean.mean(1, keepdim=True)], dim=1)
        moments = torch.cat([mean, self.logvar.expand_as(mean)], dim=1)
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(moments))


class FakeMVGen:
    """The parts of ``timemachine.mvgen.model.MVGen`` that a training step uses."""

    def __init__(self, prior_type="3dpe+pixel"):
        self.unet, self.vae = FakeUNet(), FakeVAE()
        self.config = EasyDict(CONFIG, model_cfg=EasyDict(prior_type=prior_type, depth_mid_times=4))
        self.scheduler = DDPMScheduler.from_config(SCHEDULER)
        self.stats = None               # what date_stats() reports: None, as a model without a date conditioner

    def date_stats(self):
        return self.stats

    @contextmanager
    def dates(self, year, valid=None, uncond="same"):
        self.unet.dates = (year, valid)
        try:
            yield
        finally:
            self.unet.dates = None


def fake_camera_embedding(K, w2c, b, f, h, w, config):
    return torch.full((b, f, 6, h, w), 2.0)


def fake_3d_priors(config, depth, K, w2c, cond_num, nframe, device, colors=None, latents=None, vae=None,
                   prior_type="3dpe"):
    """Recognisable priors of the released model's shapes. Like the real function, it changes ``depth`` in place."""
    h, w = depth.shape[-2:]
    depth.clamp_(max=1.0)
    if prior_type == "3dpe":
        return torch.full((nframe, 193, h // 8, w // 8), 3.0)
    return [torch.full((nframe, 193, h // 8, w // 8), 3.0), torch.full((nframe, 4, h, w), 4.0)]


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(step, "get_camera_embedding", fake_camera_embedding)
    monkeypatch.setattr(step, "get_3d_priors", fake_3d_priors)
    return FakeMVGen()


def make_sample(size=SIZE):
    gen = torch.Generator().manual_seed(0)
    K = torch.tensor([[size * 1.2, 0, size / 2], [0, size * 1.2, size / 2], [0, 0, 1]]).repeat(N_VIEWS, 1, 1)
    w2c = torch.eye(4).repeat(N_VIEWS, 1, 1)
    w2c[:, 0, 3] = torch.linspace(0, 1.0, N_VIEWS)
    depth = torch.rand(N_VIEWS, 1, size, size, generator=gen) + 1.0
    depth[:, :, : size // 4] = 40.0                         # a far band (sky), beyond 4x the median depth
    depth[N_REF:] = 0
    return {"images": torch.rand(N_VIEWS, 3, size, size, generator=gen) * 2 - 1, "K": K, "w2c": w2c, "depth": depth,
            "n_ref": N_REF, "date": torch.stack([DATE_MID - 0.5, DATE_MID + 0.5], dim=1), "date_mid": DATE_MID,
            "rows": torch.arange(N_VIEWS), "file_keys": [f"view_{j}.jpg" for j in range(N_VIEWS)]}


def seeded(seed=0):
    return torch.Generator().manual_seed(seed)


def is_zero(prior):
    """Bool [F]: which views of a prior are all zeros."""
    return prior.flatten(1).abs().amax(1) == 0


# ---- configuration and precision

@pytest.mark.parametrize("field, value", [
    ("precision", "fp16"), ("date_views", "inputs"), ("cross_date_pixels", "zero"), ("cross_date_gap", -1.0),
    ("class_label", -1), ("date_dropout", 1.5), ("camera_dropout", -0.1), ("prior_dropout", 2.0),
    ("prior_view_dropout", -1.0)])
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        LossConfig(**{field: value})


def test_config_defaults_are_the_authors_rates_and_bf16():
    cfg = LossConfig()
    assert (cfg.camera_dropout, cfg.prior_dropout, cfg.prior_view_dropout, cfg.date_dropout) == (0.1, 0.1, 0.15, 0.1)
    assert (cfg.precision, cfg.class_label, cfg.date_views, cfg.cross_date_pixels) == ("bf16", 0, "all", "keep")


def test_autocast_contexts():
    with autocast("bf16", "cpu"):
        assert torch.is_autocast_enabled("cpu") and torch.get_autocast_dtype("cpu") == torch.bfloat16
        with autocast("fp32", "cpu"):                      # fp32 also switches off an outer autocast
            assert not torch.is_autocast_enabled("cpu")
    with pytest.raises(ValueError):
        autocast("fp16")


def test_forward_runs_in_the_chosen_precision_and_the_loss_is_fp32(model):
    sample = make_sample()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    noise = torch.randn(latents.shape, generator=seeded())
    losses = {}
    for precision in ("fp32", "bf16"):
        cfg = LossConfig(precision=precision)
        loss, per_view = v_prediction_loss(model, latents, inputs, 400, noise, None, None, cfg, N_REF)
        assert model.unet.seen["dtype"] == (torch.bfloat16 if precision == "bf16" else torch.float32)
        assert loss.dtype == per_view.dtype == torch.float32
        losses[precision] = loss
    torch.testing.assert_close(losses["bf16"], losses["fp32"], rtol=0.05, atol=0)
    model.unet.zero_grad()
    losses["bf16"].backward()
    grad = model.unet.conv.weight.grad
    assert grad.dtype == torch.float32 and torch.isfinite(grad).all() and grad.abs().sum() > 0
    timesteps = torch.tensor([400])
    pred = predict(model, latents, timesteps, inputs, 0, None, None, "bf16")
    assert model.unet.seen["dtype"] == torch.bfloat16 and pred.dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):      # fp32 stays fp32 inside someone else's autocast
        pred = predict(model, latents, timesteps, inputs, 0, None, None, "fp32")
    assert model.unet.seen["dtype"] == torch.float32 and pred.dtype == torch.float32


# ---- latents

def test_encode_images_scales_the_posterior_and_can_be_seeded(model):
    images = make_sample()["images"]
    mean = F.avg_pool2d(images, 8)
    mode = encode_images(model, images, sample_posterior=False)
    assert mode.shape == (1, N_VIEWS, 4, LATENT, LATENT) and mode.dtype == torch.float32 and not mode.requires_grad
    torch.testing.assert_close(mode[0, :, :3], mean * SCALING)
    torch.testing.assert_close(mode, encode_images(model, images, sample_posterior=False))
    drawn = encode_images(model, images, generator=seeded(1))
    assert not torch.equal(drawn, mode)
    assert torch.equal(drawn, encode_images(model, images, generator=seeded(1)))
    assert not torch.equal(drawn, encode_images(model, images, generator=seeded(2)))


# ---- conditioning tensors

def test_inputs_layout(model):
    sample = make_sample()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    assert inputs.add_inputs.shape == (1, N_VIEWS, 7, SIZE, SIZE)
    assert inputs.latent_masks.shape == (1, N_VIEWS, 1, LATENT, LATENT)
    assert torch.equal(inputs.add_inputs[0, :, 0, 0, 0], (~IS_REF).float())          # mask: 0 reference, 1 target
    assert torch.equal(inputs.latent_masks[0, :, 0, 0, 0], (~IS_REF).float())
    assert torch.all(inputs.add_inputs[:, :, 1:] == 2.0)                             # then the 6 camera channels
    coordinates, pixels = inputs.coords
    assert coordinates.shape == (N_VIEWS, 193, LATENT, LATENT) and pixels.shape == (N_VIEWS, 4, SIZE, SIZE)
    assert torch.all(coordinates == 3.0) and torch.all(pixels == 4.0)
    assert not any(t.requires_grad for t in (inputs.add_inputs, inputs.latent_masks, coordinates, pixels))


def test_build_inputs_never_changes_the_sample(model):
    sample = make_sample()
    before = {k: v.clone() for k, v in sample.items() if isinstance(v, torch.Tensor)}
    latents = encode_images(model, sample["images"], sample_posterior=False)
    everything = torch.ones(N_VIEWS, dtype=torch.bool)
    build_inputs(model, sample, latents, dropped_prior_views=everything, dropped_pixel_views=everything)
    assert all(torch.equal(sample[k], v) for k, v in before.items())       # depth too: the fake clamps its copy
    assert everything.all()                                                # the callers' masks are not edited


def test_dropped_camera_and_priors_are_zeros_of_the_same_shapes(model):
    sample = make_sample()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents, drop_camera=True)
    assert torch.all(inputs.add_inputs[:, :, 1:] == 0) and inputs.add_inputs.shape == (1, N_VIEWS, 7, SIZE, SIZE)
    assert torch.equal(inputs.add_inputs[0, :, 0, 0, 0], (~IS_REF).float())           # the mask channel stays
    assert torch.all(inputs.coords[0] == 3.0) and torch.all(inputs.coords[1] == 4.0)

    inputs = build_inputs(model, sample, latents, drop_priors=True)
    coordinates, pixels = inputs.coords
    assert coordinates.shape == (N_VIEWS, 193, LATENT, LATENT) and pixels.shape == (N_VIEWS, 4, SIZE, SIZE)
    assert is_zero(coordinates).all() and is_zero(pixels).all()                       # the references' as well
    assert torch.all(inputs.add_inputs[:, :, 1:] == 2.0)


def test_view_masks_zero_the_right_views_and_never_a_reference(model):
    sample = make_sample()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    views = torch.tensor([True, False, False, True, False, True])                     # view 0 is a reference
    coordinates, pixels = build_inputs(model, sample, latents, dropped_prior_views=views).coords
    expected = views & ~IS_REF
    assert torch.equal(is_zero(coordinates), expected) and torch.equal(is_zero(pixels), expected)

    coordinates, pixels = build_inputs(model, sample, latents, dropped_pixel_views=views).coords
    assert torch.equal(is_zero(pixels), expected) and not is_zero(coordinates).any()

    with pytest.raises(ValueError, match="view mask"):
        build_inputs(model, sample, latents, dropped_prior_views=views[:-1])
    with pytest.raises(ValueError, match="view mask"):
        build_inputs(model, sample, latents, dropped_pixel_views=views.float())


def test_a_single_prior_tensor_is_handled_and_has_no_pixels_to_drop(monkeypatch):
    monkeypatch.setattr(step, "get_camera_embedding", fake_camera_embedding)
    monkeypatch.setattr(step, "get_3d_priors", fake_3d_priors)
    model, sample = FakeMVGen(prior_type="3dpe"), make_sample()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    coords = build_inputs(model, sample, latents, dropped_prior_views=CROSS_DATE).coords
    assert isinstance(coords, torch.Tensor) and torch.equal(is_zero(coords), CROSS_DATE)
    with pytest.raises(ValueError, match="no warped-pixel prior"):
        build_inputs(model, sample, latents, dropped_pixel_views=CROSS_DATE)
    with pytest.raises(ValueError, match="no warped-pixel prior"):                    # also when all priors go
        build_inputs(model, sample, latents, drop_priors=True, dropped_pixel_views=CROSS_DATE)


def test_samples_must_be_single_view_sets_with_references_and_targets(model):
    sample = make_sample()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    with pytest.raises(ValueError, match="one view set"):
        build_inputs(model, {**sample, "images": sample["images"][None]}, latents)
    for n_ref in (0, N_VIEWS):
        with pytest.raises(ValueError, match="references and targets"):
            training_loss(model, {**sample, "n_ref": n_ref}, LossConfig(**NO_DROPOUT), seeded())


def test_cross_date_views():
    assert torch.equal(cross_date_views(DATE_MID, N_REF, 1.0), CROSS_DATE)
    assert torch.equal(cross_date_views(DATE_MID, N_REF, 0.1), ~IS_REF)                # every target is apart
    assert not cross_date_views(DATE_MID, N_REF, 100.0).any()
    assert not cross_date_views(torch.tensor([2000.0, 2001.0, 2002.0]), 2, 1.0).any()  # exactly the gap: near


def test_cross_date_views_count_a_reference_only_in_the_targets_period():
    assert torch.equal(cross_date_views(DATE_MID, N_REF, 1.0, EDGE), CROSS_DATE_WITH_EDGE)
    # however wide the gap: a reference on the other side of an edge does not show the target's state
    assert torch.equal(cross_date_views(DATE_MID, N_REF, 100.0, EDGE), torch.tensor([False] * 3 + [True] * 2 + [False]))
    # Notre-Dame: a target of May 2019 with references of late 2018 (across the fire) and of 2024 (same period,
    # but years away) has no reference that shows it; one reference of June 2019 is enough
    fire = (1970.0, 2019.2877)
    assert cross_date_views(torch.tensor([2018.9, 2024.0, 2019.37]), 2, 1.0, fire).tolist() == [False, False, True]
    assert cross_date_views(torch.tensor([2018.9, 2024.0, 2019.37]), 2, 1.0).tolist() == [False, False, False]
    assert cross_date_views(torch.tensor([2018.9, 2019.45, 2019.37]), 2, 1.0, fire).tolist() == [False, False, False]
    # a date on an edge belongs to the later period, also in a sample's float32 dates
    on_edge = torch.tensor([2019.2877, 2019.2877 - 0.01, 2019.2877 + 0.01], dtype=torch.float32)
    assert cross_date_views(on_edge[[1, 0]], 1, 1.0, fire).tolist() == [False, True]
    assert cross_date_views(on_edge[[2, 0]], 1, 1.0, fire).tolist() == [False, False]
    assert cross_date_views(on_edge[[0, 0]].double(), 1, 1.0, fire).tolist() == [False, False]


def test_cross_date_drops_names_what_each_setting_takes():
    def drops(setting):
        return cross_date_drops(DATE_MID, N_REF, LossConfig(cross_date_pixels=setting), EDGE)

    assert drops("keep") == (None, None)
    prior_views, pixel_views = drops("drop")
    assert prior_views is None and torch.equal(pixel_views, CROSS_DATE_WITH_EDGE)
    prior_views, pixel_views = drops("drop_all")
    assert pixel_views is None and torch.equal(prior_views, CROSS_DATE_WITH_EDGE)
    wide = cross_date_drops(DATE_MID, N_REF, LossConfig(cross_date_pixels="drop", cross_date_gap=20.0))[1]
    assert torch.equal(wide, DATE_MID == 1950.0)                                       # the gap is the config's


# ---- the UNet call and the loss

def test_predict_passes_the_conditioning_and_sets_dates_only_for_the_forward_pass(model):
    sample = make_sample()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    year, valid, timesteps = DATE_MID.clone(), ~IS_REF, torch.tensor([7])
    pred = predict(model, latents, timesteps, inputs, 1, year, valid, "fp32")
    seen = model.unet.seen
    assert pred.shape == latents.shape
    torch.testing.assert_close(pred[0], model.unet.conv(latents[0]))                  # rows are views, in order
    assert seen["sample"] is latents and seen["timesteps"] is timesteps and seen["encoder_hidden_states"] is None
    assert seen["add_inputs"] is inputs.add_inputs and seen["coords"] is inputs.coords and seen["return_dict"] is False
    assert seen["class_labels"].dtype == torch.long and seen["class_labels"].tolist() == [1]
    assert torch.equal(seen["dates"][0], year) and torch.equal(seen["dates"][1], valid)
    assert model.unet.dates is None                                                   # cleared after the pass

    predict(model, latents, timesteps, inputs, 0, None, None, "fp32")                 # no dates: none are set
    assert model.unet.seen["dates"] is None


def test_references_stay_clean_and_targets_are_noised(model):
    sample = make_sample()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    noise = torch.randn(latents.shape, generator=seeded())
    timesteps = torch.tensor([600])
    v_prediction_loss(model, latents, inputs, 600, noise, None, None, LossConfig(**NO_DROPOUT), N_REF)
    seen = model.unet.seen["sample"]
    assert torch.equal(seen[:, :N_REF], latents[:, :N_REF])
    torch.testing.assert_close(seen[:, N_REF:], model.scheduler.add_noise(latents, noise, timesteps)[:, N_REF:])
    assert not torch.isclose(seen[:, N_REF:], latents[:, N_REF:]).all()
    assert model.unet.seen["timesteps"].tolist() == [600]


def test_loss_is_the_velocity_error_on_the_targets_only(model):
    sample = make_sample()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    noise = torch.randn(latents.shape, generator=seeded())
    timesteps = torch.tensor([300])
    cfg = LossConfig(**NO_DROPOUT)
    loss, per_view = v_prediction_loss(model, latents, inputs, 300, noise, None, None, cfg, N_REF)

    # the reference recipe (scripts/mvgen/grad_precision.py)
    noisy = model.scheduler.add_noise(latents, noise, timesteps)
    noisy = latents * (1 - inputs.latent_masks) + noisy * inputs.latent_masks
    target = model.scheduler.get_velocity(latents, noise, timesteps)
    pred = model.unet.conv(noisy[0])[None]
    torch.testing.assert_close(loss, F.mse_loss(pred[:, N_REF:], target[:, N_REF:]))
    assert per_view.shape == (N_VIEWS - N_REF,)
    torch.testing.assert_close(per_view, (pred - target)[0, N_REF:].pow(2).flatten(1).mean(1))
    assert not torch.isclose(loss, F.mse_loss(pred, target))                          # the references do not count

    model.unet.zero_grad()
    loss.backward()
    assert model.unet.conv.weight.grad.abs().sum() > 0


def test_loss_refuses_a_scheduler_that_does_not_predict_the_velocity(model):
    sample = make_sample()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    model.scheduler = DDPMScheduler.from_config({**SCHEDULER, "prediction_type": "epsilon"})
    with pytest.raises(ValueError, match="velocity"):
        v_prediction_loss(model, latents, inputs, 300, torch.zeros_like(latents), None, None, LossConfig(), N_REF)


# ---- the stochastic training step

def test_training_loss_without_dropout_gives_everything_and_true_dates(model):
    sample = make_sample()
    loss, info = training_loss(model, sample, LossConfig(**NO_DROPOUT, class_label=1), seeded())
    seen = model.unet.seen
    year, valid = seen["dates"]
    assert valid.all() and torch.all((year >= sample["date"][:, 0]) & (year < sample["date"][:, 1]))
    assert not torch.equal(year, DATE_MID)                                             # drawn inside the interval
    assert torch.all(seen["add_inputs"][:, :, 1:] == 2.0)
    assert torch.all(seen["coords"][0] == 3.0) and torch.all(seen["coords"][1] == 4.0)
    assert seen["class_labels"].tolist() == [1]
    assert info == {"timestep": info["timestep"], "loss": float(loss), "camera_dropped": False, "priors_dropped": False,
                    "prior_views_dropped": 0, "pixel_views_dropped": 0, "dates_known": True, "date_rms": None,
                    "date_temb_ratio": None}
    assert json.loads(json.dumps(info)) == info and loss.requires_grad

    training_loss(model, sample, LossConfig(**NO_DROPOUT, date_jitter=False), seeded())
    torch.testing.assert_close(model.unet.seen["dates"][0], DATE_MID)


def test_date_views_targets_hides_the_references_dates(model):
    _, info = training_loss(model, make_sample(), LossConfig(**NO_DROPOUT, date_views="targets"), seeded())
    assert torch.equal(model.unet.seen["dates"][1], ~IS_REF) and info["dates_known"]


@pytest.mark.parametrize("date_views", ["all", "targets"])
def test_date_dropout_marks_every_view_unknown(model, date_views):
    cfg = LossConfig(**{**NO_DROPOUT, "date_dropout": 1.0}, date_views=date_views)
    _, info = training_loss(model, make_sample(), cfg, seeded())
    assert not model.unet.seen["dates"][1].any() and not info["dates_known"]


def test_cross_date_pixels_drop_zeroes_only_the_pixel_prior_of_far_dated_targets(model):
    sample = make_sample()
    _, info = training_loss(model, sample, LossConfig(**NO_DROPOUT, cross_date_pixels="drop"), seeded())
    coordinates, pixels = model.unet.seen["coords"]
    assert torch.equal(is_zero(pixels), CROSS_DATE) and torch.all(pixels[~CROSS_DATE] == 4.0)
    assert torch.all(coordinates == 3.0)                                               # never the coordinate prior
    assert info["pixel_views_dropped"] == 2 and info["prior_views_dropped"] == 0

    _, info = training_loss(model, sample, LossConfig(**NO_DROPOUT, cross_date_pixels="drop", cross_date_gap=20.0),
                            seeded())
    assert torch.equal(is_zero(model.unet.seen["coords"][1]), DATE_MID == 1950.0) and info["pixel_views_dropped"] == 1

    _, info = training_loss(model, sample, LossConfig(**NO_DROPOUT), seeded())         # "keep", the default
    assert not is_zero(model.unet.seen["coords"][1]).any() and info["pixel_views_dropped"] == 0

    # with the run's period edges, a target whose only near references lie in another period is cross-date too
    _, info = training_loss(model, sample, LossConfig(**NO_DROPOUT, cross_date_pixels="drop"), seeded(), EDGE)
    assert torch.equal(is_zero(model.unet.seen["coords"][1]), CROSS_DATE_WITH_EDGE) and info["pixel_views_dropped"] == 3
    assert torch.all(model.unet.seen["coords"][0] == 3.0)


def test_cross_date_pixels_drop_all_zeroes_both_priors_of_cross_date_targets(model):
    """The state the authors' per-view prior dropout produced: a view's priors are zeroed together."""
    sample = make_sample()
    _, info = training_loss(model, sample, LossConfig(**NO_DROPOUT, cross_date_pixels="drop_all"), seeded())
    coordinates, pixels = model.unet.seen["coords"]
    assert torch.equal(is_zero(coordinates), CROSS_DATE) and torch.equal(is_zero(pixels), CROSS_DATE)
    assert torch.all(coordinates[~CROSS_DATE] == 3.0) and torch.all(pixels[~CROSS_DATE] == 4.0)
    assert info["prior_views_dropped"] == 2 and info["pixel_views_dropped"] == 0

    _, info = training_loss(model, sample, LossConfig(**NO_DROPOUT, cross_date_pixels="drop_all"), seeded(), EDGE)
    coordinates, pixels = model.unet.seen["coords"]
    assert torch.equal(is_zero(coordinates), CROSS_DATE_WITH_EDGE)
    assert torch.equal(is_zero(pixels), CROSS_DATE_WITH_EDGE) and info["prior_views_dropped"] == 3

    # together with the random per-view dropout: a target loses its priors for either reason
    cfg = LossConfig(**{**NO_DROPOUT, "prior_view_dropout": 0.5}, cross_date_pixels="drop_all")
    counts = set()
    for seed in range(12):
        _, info = training_loss(model, sample, cfg, seeded(seed))
        dropped = is_zero(model.unet.seen["coords"][0])
        assert torch.equal(dropped, is_zero(model.unet.seen["coords"][1])) and dropped[CROSS_DATE].all()
        assert not dropped[:N_REF].any() and info["prior_views_dropped"] == int(dropped.sum())
        counts.add(int(dropped.sum()))
    assert min(counts) == 2 and max(counts) > 2


def test_training_loss_reports_the_size_of_the_date_embedding(model):
    model.stats = {"date_rms": 0.3, "temb_rms": 0.15}
    _, info = training_loss(model, make_sample(), LossConfig(**NO_DROPOUT), seeded())
    assert info["date_rms"] == 0.3 and info["date_temb_ratio"] == pytest.approx(2.0)
    assert date_embedding_size(model) == {"date_rms": 0.3, "date_temb_ratio": pytest.approx(2.0)}
    model.stats = {"date_rms": torch.tensor(0.0), "temb_rms": torch.tensor(0.2)}       # zero init; tensors are accepted
    size = date_embedding_size(model)
    assert size == {"date_rms": 0.0, "date_temb_ratio": 0.0} and json.loads(json.dumps(size)) == size
    model.stats = None                                                                  # an undated pass
    assert date_embedding_size(model) == {"date_rms": None, "date_temb_ratio": None}


def test_dropout_at_rate_one_zeroes_what_it_names(model):
    sample = make_sample()
    _, info = training_loss(model, sample, LossConfig(**{**NO_DROPOUT, "camera_dropout": 1.0}), seeded())
    add_inputs = model.unet.seen["add_inputs"]
    assert torch.all(add_inputs[:, :, 1:] == 0) and torch.equal(add_inputs[0, :, 0, 0, 0], (~IS_REF).float())
    assert info["camera_dropped"] and not info["priors_dropped"]

    cfg = LossConfig(**{**NO_DROPOUT, "prior_dropout": 1.0, "prior_view_dropout": 1.0}, cross_date_pixels="drop")
    _, info = training_loss(model, sample, cfg, seeded())
    coordinates, pixels = model.unet.seen["coords"]
    assert coordinates.shape == (N_VIEWS, 193, LATENT, LATENT) and pixels.shape == (N_VIEWS, 4, SIZE, SIZE)
    assert is_zero(coordinates).all() and is_zero(pixels).all()
    assert info["priors_dropped"] and info["prior_views_dropped"] == 0 and info["pixel_views_dropped"] == 0

    _, info = training_loss(model, sample, LossConfig(**{**NO_DROPOUT, "prior_view_dropout": 1.0}), seeded())
    coordinates, pixels = model.unet.seen["coords"]
    assert torch.equal(is_zero(coordinates), ~IS_REF) and torch.equal(is_zero(pixels), ~IS_REF)
    assert info["prior_views_dropped"] == N_VIEWS - N_REF and not info["priors_dropped"]


def test_draws_follow_the_configured_rates(model):
    sample, gen, n = make_sample(), seeded(), 400
    cfg = LossConfig(date_dropout=0.5, camera_dropout=0.25, prior_dropout=0.5, prior_view_dropout=0.5, precision="fp32")
    infos = [training_loss(model, sample, cfg, gen)[1] for _ in range(n)]
    timesteps = torch.tensor([i["timestep"] for i in infos], dtype=torch.float)
    assert timesteps.min() >= 0 and timesteps.max() <= 999
    assert timesteps.min() < 50 and timesteps.max() > 950 and 440 < timesteps.mean() < 560       # uniform over 0..999

    def share(key):
        return sum(bool(i[key]) for i in infos) / n

    assert 0.18 < share("camera_dropped") < 0.32 and 0.42 < share("priors_dropped") < 0.58
    assert 0.42 < share("dates_known") < 0.58
    kept = [i["prior_views_dropped"] for i in infos if not i["priors_dropped"]]
    assert 0.42 < sum(kept) / (len(kept) * (N_VIEWS - N_REF)) < 0.58
    both = sum(i["camera_dropped"] and i["priors_dropped"] for i in infos) / n                    # independent draws
    assert 0.07 < both < 0.18


def test_training_loss_is_deterministic_with_a_seeded_generator(model):
    sample, cfg = make_sample(), LossConfig(precision="fp32", cross_date_pixels="drop")

    def run(seed):
        loss, info = training_loss(model, sample, cfg, seeded(seed))
        seen = model.unet.seen
        return loss, info, seen["sample"], seen["dates"][0], seen["add_inputs"], seen["coords"][1]

    first, again, other = run(11), run(11), run(12)
    assert first[1] == again[1] and all(torch.equal(a, b) for a, b in zip(first[2:], again[2:]))
    assert torch.equal(first[0], again[0])
    assert first[0] != other[0] and not torch.equal(first[2], other[2]) and not torch.equal(first[3], other[3])


# ---- MVGenMaster's own camera and prior functions (not replaced)

class DepthOnAnotherDevice:
    """Depth as the authors' training loop holds it: on the CPU while the model is on a GPU.

    ``get_3d_priors`` calls ``depth.to(device)`` once per prior, which there copies the tensor to the
    GPU, so each prior contracts far depth in a copy of its own. A CPU test has no second device;
    this hands out the same fresh copies. It is the reference for "contracted once" that does not
    rely on the float64 route ``depth_for_priors`` takes.
    """

    def __init__(self, depth):
        self.depth, self.shape = depth, depth.shape

    def to(self, device):
        return self.depth.clone()


def real_priors(model, sample, latents, depth):
    return get_3d_priors(model.config, depth, sample["K"], sample["w2c"], sample["n_ref"],
                         nframe=sample["images"].shape[0], device="cpu", colors=sample["images"], latents=latents,
                         vae=model.vae, prior_type="3dpe+pixel")


def test_depth_for_priors_is_a_private_float64_copy():
    depth = make_sample()["depth"]
    for given in (depth, depth.double()):                   # float64 input: still a copy, not the caller's tensor
        private = depth_for_priors(given, "cpu")
        assert private.dtype == torch.float64 and torch.equal(private, depth.double())
        private.zero_()
        assert given.abs().sum() > 0


def test_build_inputs_matches_the_authors_recipe_with_the_real_camera_and_prior_functions():
    size = 64
    model, sample = FakeMVGen(), make_sample(size)
    f, depth_before = N_VIEWS, sample["depth"].clone()
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    assert torch.equal(sample["depth"], depth_before)

    # as the authors' train.py builds them: every prior from its own copy of the depth, far depth contracted once
    camera = get_camera_embedding(sample["K"], sample["w2c"], 1, f, size, size, config=model.config)
    coordinates, pixels = real_priors(model, sample, latents, DepthOnAnotherDevice(sample["depth"]))
    assert camera.shape == (1, f, 6, size, size)
    assert coordinates.shape == (f, 193, size // 8, size // 8) and pixels.shape == (f, 4, size, size)
    assert torch.equal(inputs.add_inputs[:, :, 1:], camera)
    assert torch.equal(inputs.add_inputs[0, :, 0, 0, 0], (~IS_REF).float())
    assert torch.equal(inputs.coords[0], coordinates) and torch.equal(inputs.coords[1], pixels)
    assert not is_zero(coordinates).any() and not is_zero(pixels[:N_REF]).any()

    # what build_inputs must not do: hand over float32 depth on the device (the authors' demo script). The prior
    # code then edits the caller's tensor, and the warped pixels are computed from far depth contracted twice.
    shared = sample["depth"].clone()
    coordinates_demo, pixels_demo = real_priors(model, sample, latents, shared)
    assert not torch.equal(shared, depth_before)
    assert torch.equal(coordinates_demo, coordinates) and not torch.equal(pixels_demo, pixels)

    dropped = build_inputs(model, sample, latents, drop_camera=True, drop_priors=True,
                           dropped_pixel_views=CROSS_DATE)
    assert dropped.add_inputs.shape == inputs.add_inputs.shape and torch.all(dropped.add_inputs[:, :, 1:] == 0)
    assert [c.shape for c in dropped.coords] == [c.shape for c in inputs.coords]
    assert all(is_zero(c).all() for c in dropped.coords)
    again = build_inputs(model, sample, latents)           # the same sample gives the same inputs every time
    assert torch.equal(again.coords[0], coordinates) and torch.equal(again.coords[1], pixels)


# ---- the authors' training step, transcribed

class RecordingUNet(nn.Module):
    """MVGenMaster's UNet call; the output depends on every input, and the inputs are remembered."""

    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.w_latents = nn.Parameter(torch.randn(4, 4, generator=gen) * 0.5)
        self.w_add = nn.Parameter(torch.randn(4, 7, generator=gen) * 0.3)
        self.w_coordinates = nn.Parameter(torch.randn(4, 193, generator=gen) * 0.05)
        self.w_pixels = nn.Parameter(torch.randn(4, 4, generator=gen) * 0.3)
        self.w_domain = nn.Parameter(torch.randn(3, 4, generator=gen) * 0.1)
        self.seen = None

    def forward(self, sample, timesteps, encoder_hidden_states, add_inputs=None, class_labels=None, coords=None,
                return_dict=True):
        coordinates, pixels = coords
        self.seen = {"sample": sample.clone(), "timesteps": timesteps.clone(), "add_inputs": add_inputs.clone(),
                     "class_labels": class_labels.clone(), "coordinates": coordinates.clone(),
                     "pixels": pixels.clone()}
        f = sample.shape[1]
        latents = sample.flatten(0, 1).float()
        small = latents.shape[-2:]
        add = F.adaptive_avg_pool2d(add_inputs.flatten(0, 1).float(), small)
        out = (torch.einsum("oc,bchw->bohw", self.w_latents, latents) + torch.einsum("oc,bchw->bohw", self.w_add, add)
               + torch.einsum("oc,bchw->bohw", self.w_coordinates, coordinates.float())
               + torch.einsum("oc,bchw->bohw", self.w_pixels, F.adaptive_avg_pool2d(pixels.float(), small)))
        out = out + self.w_domain[class_labels].repeat_interleave(f, dim=0)[:, :, None, None]
        return (out * (1 + timesteps.float().repeat_interleave(f)[:, None, None, None] / 1000),)


def authors_step(batch, config, vae, unet, noise_scheduler, device, latents, *, random_cond_num, timesteps, noise,
                 drop_camera=False, drop_coords=False, cmask=None):
    """One optimisation step of ``external/MVGenMaster/train.py``, transcribed line by line (lines 692-865; each
    line here names the line it copies).

    The random draws are arguments, so that both implementations get the same number of references,
    timestep, noise and dropout decisions; so are the latents (lines 712-713 encode them with the
    VAE). Left out: branches the released config does not take (no camera embedding, noise offset,
    input perturbation, other prior types, epsilon prediction) and ``dynamic_nframe``. ``batch`` holds
    ``(b f)`` tensors as the authors' data loader gives them. Returns the loss and what the UNet received.
    """
    f = config.nframe
    image = batch["image"].to(device)                                                                    # 692
    image = einops.rearrange(image, "(b f) c h w -> b f c h w", f=f)                                      # 693
    b, f, _, h, w = image.shape                                                                          # 710
    _, _, _, latent_h, latent_w = latents.shape                                                          # 714
    random_masks = torch.ones((b, f, 1, h, w), device=device, dtype=latents.dtype)                       # 717
    random_latent_masks = torch.ones((b, f, 1, latent_h, latent_w), device=device, dtype=latents.dtype)   # 718
    random_masks[:, :random_cond_num] = 0                                                                # 723
    random_latent_masks[:, :random_cond_num] = 0                                                         # 724
    if drop_camera:                                                    # 736: random.random() < cfg_training_rate
        add_channel = np.sum(config.model_cfg.additional_in_channels) - 1                                # 737
        if "pixel" in config.model_cfg.get("prior_type", "3dpe"):                                        # 743
            add_channel -= 4                                                                             # 744
        camera_embedding = torch.zeros((b, f, add_channel, h, w), device=device, dtype=latents.dtype)     # 747
    else:
        camera_embedding = get_camera_embedding(batch["intrinsic"].to(device), batch["extrinsic"].to(device),
                                                b, f, h, w, config=config).to(device=device, dtype=latents.dtype)  # 749
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)                                 # 769
    noisy_latents = latents * (1 - random_latent_masks) + noisy_latents * random_latent_masks            # 771
    assert noise_scheduler.config.prediction_type == "v_prediction"                                      # 783
    target = noise_scheduler.get_velocity(latents, noise, timesteps)                                     # 784
    inputs = noisy_latents                                                                               # 789
    add_inputs = torch.cat([random_masks, camera_embedding], dim=2)                                      # 793
    assert config.model_cfg.enable_depth and config.model_cfg.priors3d                                   # 795-796
    if drop_coords:                                                    # 797: random.random() < cfg_training_rate
        assert config.model_cfg.prior_type == "3dpe+pixel"                                               # 798
        coords = [torch.zeros((b * f, 192 + 1, latent_h, latent_w), device=device, dtype=latents.dtype),
                  torch.zeros((b * f, 4, h, w), device=device, dtype=latents.dtype)]                     # 799-801
    else:
        coords = get_3d_priors(config, batch["depth"], batch["intrinsic"], batch["extrinsic"],
                               random_cond_num, nframe=f, device=device, colors=batch["image"],
                               latents=latents, vae=vae, prior_type=config.model_cfg.get("prior_type", "3dpe"))  # 815
        if cmask is not None:                                          # 820-824: cmask is drawn there, at coords_cfg
            cmask = cmask.astype(np.float64).reshape(b, f, 1, 1, 1)                # 825 (here on a copy of the mask)
            cmask[:, :random_cond_num] = 1                                                               # 826
            cmask = torch.tensor(cmask, device=device, dtype=torch.float32)                              # 827
            for j in range(len(coords)):                                                                 # 830
                coords[j] = einops.rearrange(coords[j], "(b f) c h w -> b f c h w", f=f)                  # 831
                coords[j] = coords[j] * cmask                                                            # 832
                coords[j] = einops.rearrange(coords[j], "b f c h w -> (b f) c h w")                       # 833
    domain_dict = config.model_cfg.get("domain_dict", None)                                              # 851
    tags = batch["tag"][::f]                                                                             # 853
    class_labels = [domain_dict.get(tag, domain_dict["others"]) for tag in tags]                         # 854
    class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)                           # 855
    model_pred = unet(inputs, timesteps, None, add_inputs=add_inputs, class_labels=class_labels, coords=coords,
                      return_dict=False)[0]                            # 860-861 (there under the precision's autocast)
    model_pred = einops.rearrange(model_pred, "(b f) c h w -> b f c h w", f=f)                            # 863
    loss = F.mse_loss(model_pred.float()[:, random_cond_num:], target.float()[:, random_cond_num:],
                      reduction="mean")                                                                  # 865
    return loss, dict(unet.seen)


@pytest.fixture(scope="module")
def released():
    """A fake model with the released model's ``config.yaml`` and the training scheduler the authors build from it."""
    path = Path("check_points/pretrained_model/config.yaml")
    if not path.exists():
        pytest.skip("needs the released model's config.yaml: run from MVGenMaster's checkout")
    model = FakeMVGen()
    model.config = EasyDict(OmegaConf.load(path))
    model.scheduler = get_diffusion_scheduler(model.config, name="DDPM")
    model.unet = RecordingUNet()
    return model


def arc_view_set(n_views, n_ref, size=64):
    """Cameras on an arc that look at a wall 4 units away; the references' depth has a far band ("sky") at 60."""
    gen = torch.Generator().manual_seed(1)
    images = F.interpolate(torch.randn(n_views, 3, 8, 8, generator=gen), size=size, mode="bilinear")
    images = (images * 0.6).clamp(-1, 1)
    K = torch.tensor([[70.0, 0, size / 2], [0, 70.0, size / 2], [0, 0, 1]]).repeat(n_views, 1, 1)
    w2c = torch.eye(4).repeat(n_views, 1, 1)
    for j in range(n_views):
        angle = math.radians(-12 + 3.5 * j)
        c, s = math.cos(angle), math.sin(angle)
        rotation = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        centre = torch.tensor([4 * math.sin(angle), 0.05 * j, 4 - 4 * math.cos(angle)])
        w2c[j, :3, :3] = rotation.T
        w2c[j, :3, 3] = -rotation.T @ centre
    depth = torch.zeros(n_views, 1, size, size)
    depth[:n_ref] = 4 + 0.3 * F.interpolate(torch.randn(n_ref, 1, 8, 8, generator=gen), size=size, mode="bilinear")
    depth[:n_ref, :, : size // 4] = 60.0
    return {"images": images, "K": K, "w2c": w2c, "depth": depth, "n_ref": n_ref}


# What is dropped: the arguments of the authors' step, and those of build_inputs for the same decisions. In the
# last case the authors' mask also names view 1, a reference, which their line 826 and our build_inputs both keep.
DROPOUT_CASES = {
    "nothing": ({}, {}),
    "camera": ({"drop_camera": True}, {"drop_camera": True}),
    "all priors": ({"drop_coords": True}, {"drop_priors": True}),
    "priors of views 1, 4 and 6": (
        {"cmask": np.array([1, 0, 1, 1, 0, 1, 0, 1])},
        {"dropped_prior_views": torch.tensor([False, True, False, False, True, False, True, False])}),
}


@pytest.mark.parametrize("dropped", list(DROPOUT_CASES))
@pytest.mark.parametrize("timestep", [20, 500, 999])
@pytest.mark.parametrize("tag, class_label", [("co3dv2", 0), ("megascenes", 1)])
def test_step_feeds_the_unet_what_the_authors_training_step_feeds_it(released, dropped, timestep, tag, class_label):
    """``build_inputs`` + ``v_prediction_loss`` against the transcription of ``train.py``: every tensor the UNet
    receives is identical, the warped-pixel prior included, and so is the loss."""
    config, unet = released.config, released.unet
    n_ref = 3
    sample = arc_view_set(config.nframe, n_ref)
    latents = encode_images(released, sample["images"], sample_posterior=False)
    noise = torch.randn(latents.shape, generator=seeded(timestep))
    theirs_kwargs, ours_kwargs = DROPOUT_CASES[dropped]

    with torch.no_grad():
        inputs = build_inputs(released, sample, latents, **ours_kwargs)
        cfg = LossConfig(precision="fp32", class_label=class_label)
        ours_loss, _ = v_prediction_loss(released, latents, inputs, timestep, noise, None, None, cfg, n_ref)
        ours = dict(unet.seen)
        batch = {"image": sample["images"].clone(), "intrinsic": sample["K"].clone(),
                 "extrinsic": sample["w2c"].clone(), "depth": DepthOnAnotherDevice(sample["depth"].clone()),
                 "tag": [tag] * config.nframe}
        theirs_loss, theirs = authors_step(batch, config, released.vae, unet, released.scheduler, torch.device("cpu"),
                                           latents, random_cond_num=n_ref, timesteps=torch.tensor([timestep]),
                                           noise=noise, **theirs_kwargs)

    assert set(ours) == set(theirs)
    for name in ours:
        assert ours[name].shape == theirs[name].shape and ours[name].dtype == theirs[name].dtype, name
        assert torch.equal(ours[name], theirs[name]), name
    assert ours["class_labels"].tolist() == [class_label]
    # the same numbers summed in another order (mean of the views' means, against one mean over all targets)
    torch.testing.assert_close(ours_loss, theirs_loss, rtol=1e-6, atol=0)
    assert ours_loss > 0
    if dropped == "nothing":            # the far band is in play: from depth shared between the priors, pixels move
        shared = real_priors(released, sample, latents, sample["depth"].clone())
        assert torch.equal(shared[0], ours["coordinates"]) and not torch.equal(shared[1], ours["pixels"])
