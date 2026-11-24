"""CPU tests for the MVGenMaster trainer, its probes and its command lines, on small stand-ins.

Run from MVGenMaster's checkout, in its environment (see ``timemachine/mvgen/__init__.py``). Nothing
of the released model is loaded:

* ``TinyUNet`` has MVGenMaster's UNet call around two of its real ResBlocks (so the real date
  conditioner attaches) and one linear layer that LoRA adapts; like MVGenMaster's blocks it only
  checkpoints in ``train()`` mode. The adapters, the training step and the checkpoints are the real
  code (``timemachine.mvgen.model`` / ``.step``), with MVGenMaster's camera and prior functions
  replaced by ones that return recognisable tensors.
* ``FakePipeline`` stands in for the sampling pipeline: the same call and a guidance-doubled batch
  through the same UNet.
* The dataset is a folder of 24 tiny views written to ``tmp_path``.

The tests at the end fake nothing but the weights: a small UNet of MVGenMaster's own class runs
through the trainer and the probes with MVGenMaster's camera rays, 3D priors and sampling pipeline.
"""
import ast
import importlib
import importlib.util
import json
import sys
import unicodedata
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

my_diffusers = pytest.importorskip("my_diffusers", reason="runs in MVGenMaster's environment, from its checkout")

# Under pytest the repo root is on the import path (its conftest.py), so ``src`` is the repo's own ``src``
# package and hides MVGenMaster's ``src`` folder, which ``timemachine.mvgen`` imports from. Let the package
# search that folder too.
import src  # noqa: E402

MVGEN_SRC = str(Path(my_diffusers.__file__).resolve().parents[1] / "src")
if MVGEN_SRC not in list(src.__path__):
    src.__path__.append(MVGEN_SRC)

from diffusers import AutoencoderKL  # noqa: E402
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # noqa: E402
from easydict import EasyDict  # noqa: E402
import my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview as pipeline_module  # noqa: E402
from my_diffusers.models import UNet2DConditionModel  # noqa: E402
from my_diffusers.models.resnet import ResnetBlock2D  # noqa: E402
from my_diffusers.schedulers import DDPMScheduler  # noqa: E402
from PIL import Image  # noqa: E402
from src.modules.camera import get_camera_embedding  # noqa: E402
from src.modules.extra_encoder import ExConvEncoder2  # noqa: E402
from src.modules.position_encoding import get_3d_priors  # noqa: E402

from timemachine.data.holdout import HoldoutConfig  # noqa: E402
from timemachine.mvgen import model as mvgen_model  # noqa: E402
from timemachine.mvgen import step  # noqa: E402
from timemachine.mvgen import trainer as trainer_module  # noqa: E402
from timemachine.mvgen.data import SCENE_FILE, MVGenScene, ViewSamplingConfig, depth_path  # noqa: E402
from timemachine.mvgen.date import dated_blocks  # noqa: E402
from timemachine.mvgen.model import FinetuneConfig, MVGen, prepare_for_finetuning, trainable_state_dict  # noqa: E402
from timemachine.mvgen.probes import (  # noqa: E402
    DateLossProbe, RenderProbe, check_view_sets, date_loss_summary, describe_views, heldout_view_sets,
    parse_render_date, psnr, read_keys_file, read_rows_file, render_views, representative_years, rows_of_file_keys,
    scene_view_sets, swapped_years, unet_precision, view_priors_dropped,
)
from timemachine.mvgen.step import LossConfig, build_inputs, encode_images, predict  # noqa: E402
from timemachine.mvgen.trainer import (  # noqa: E402
    RESUME_MAY_CHANGE, SPIKE_GUARD_MIN_HISTORY, TrainConfig, Trainer, WeightEMA, checkpoint_to_resume, describe_pinned,
    flat_config, lr_factor, mode_dependent_modules, parse_scale, resume_differences,
)

REPO = Path(__file__).resolve().parents[1]
N, SIZE, LATENT, WIDTH = 24, 32, 4, 16      # views in the folder; crop side; latent side; timestep-embedding width
EDGES = (1970.0, 2019.2877)                 # Notre-Dame's periods: old | modern | after the fire
SWAP_YEARS = (1890.0, 2015.0, 2021.0)
# Rows 0-7 of the folder are old (1900 + 0.5 i), 8-15 modern (2012 + 0.1 i), 16-23 after the fire (2020 + 0.1 i).
MODERN_FROM_MODERN = (8, 9, 10, 11, 12, 13)         # 3 references, 3 targets
AFTER_FROM_MODERN = (8, 9, 10, 16, 17, 18)
# MVGenMaster's training scheduler, and what its camera and prior functions read from the model's config.
SCHEDULER = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "snr_rescale",
             "snr_rescale": 4, "rescale_betas_zero_snr": True, "prediction_type": "v_prediction", "clip_sample": False}
CONFIG = {"camera_longest_side": 5.0, "normalize_extrinsic": True, "normalize_t": False, "fourier_embedding": False,
          "fourier_embed_dim": 18}


# ----------------------------------------------------------------------------- stand-ins
class TinyUNet(nn.Module):
    """MVGenMaster's UNet call around two real ResBlocks and a linear layer for LoRA; views do not interact.

    ``seen`` records every forward pass: the mode, whether gradients were on, and the conditioning.
    """

    def __init__(self, channels: int = 8):
        super().__init__()
        self.time_embedding = nn.Linear(1, WIDTH)
        self.conv_in = nn.Conv2d(4, channels, 1)
        self.resnets = nn.ModuleList(ResnetBlock2D(in_channels=channels, temb_channels=WIDTH, groups=4)
                                     for _ in range(2))
        self.transformer = nn.Module()
        self.transformer.attn1 = nn.Module()
        self.transformer.attn1.to_q = nn.Linear(channels, channels)
        self.conv_out = nn.Conv2d(channels, 4, 1)
        self.gradient_checkpointing = False
        self.seen = []

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward(self, sample, timestep, encoder_hidden_states=None, add_inputs=None, class_labels=None, coords=None,
                return_dict=True, **unused):
        batch, views = sample.shape[:2]
        self.seen.append(dict(training=self.training, grad=torch.is_grad_enabled(), sample=sample,
                              add_inputs=add_inputs, coords=coords, class_labels=class_labels))
        t = torch.as_tensor(timestep, dtype=torch.float32).reshape(-1).expand(batch)
        temb = self.time_embedding(t[:, None] / 1000.0).repeat_interleave(views, dim=0)
        conditioning = add_inputs.flatten(0, 1).mean(dim=(1, 2, 3)) + sum(c.mean(dim=(1, 2, 3)) for c in coords)
        hidden = self.conv_in(sample.flatten(0, 1)) + 0.1 * conditioning.view(-1, 1, 1, 1)
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:       # as MVGenMaster's blocks decide
                hidden = checkpoint(resnet, hidden, temb, use_reentrant=False)
            else:
                hidden = resnet(hidden, temb)
        hidden = hidden + self.transformer.attn1.to_q(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return (self.conv_out(hidden),)


class FakeVAE(nn.Module):
    """A VAE whose posterior mean is the 8x8 average of the image (4th channel: its grey value)."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=0.5)
        self.logvar = nn.Parameter(torch.tensor(-2.0))

    def encode(self, images):
        mean = F.avg_pool2d(images, 8)
        mean = torch.cat([mean, mean.mean(1, keepdim=True)], dim=1)
        moments = torch.cat([mean, self.logvar.expand_as(mean)], dim=1)
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(moments))


class FakePipeline:
    """MVGenMaster's sampling call: a guidance batch (unconditional half first) through the model's UNet.

    Like the real one it edits the depth it is given in place, and it draws from the global random
    state, which the trainer must shield training from. ``calls`` records its arguments.
    """

    def __init__(self, model: MVGen):
        self.model, self.calls, self.progress_bar = model, [], {}

    def set_progress_bar_config(self, **kwargs) -> None:
        self.progress_bar = kwargs

    def __call__(self, *, images, nframe, cond_num, key_rescale, height, width, intrinsics, extrinsics,
                 num_inference_steps, guidance_scale, output_type, config, tag, class_label, depth, vae, generator):
        self.calls.append(dict(images=images.clone(), nframe=nframe, cond_num=cond_num, steps=num_inference_steps,
                               guidance=guidance_scale, output_type=output_type, config_nframe=config.nframe, tag=tag,
                               class_label=class_label, depth=depth.clone(), K=intrinsics, w2c=extrinsics, vae=vae))
        depth.clamp_(max=1.0)
        torch.rand(3)
        latents = torch.randn((1, nframe, 4, height // 8, width // 8), generator=generator)
        masks = torch.ones(1, nframe, 1, height, width)
        masks[:, :cond_num] = 0
        camera = torch.full((1, nframe, 6, height, width), 2.0)
        add_inputs = torch.cat([torch.cat([masks, torch.zeros_like(camera)], 2), torch.cat([masks, camera], 2)])
        priors = [torch.full((nframe, 193, height // 8, width // 8), 3.0), torch.full((nframe, 4, height, width), 4.0)]
        coords = [torch.cat([torch.zeros_like(prior), prior]) for prior in priors]
        labels = torch.full((1,), class_label, dtype=torch.long)
        for i in range(num_inference_steps):
            pred = self.model.unet(torch.cat([latents] * 2), torch.tensor(900 - i), add_inputs=add_inputs,
                                   encoder_hidden_states=None, class_labels=labels, coords=coords, return_dict=False,
                                   cond_num=cond_num, key_rescale=key_rescale)[0]
            uncond, cond = pred.chunk(2)
            latents = latents - 0.5 * (uncond + guidance_scale * (cond - uncond))[None]
        frames = torch.sigmoid(F.interpolate(latents[0, :, :3], scale_factor=8)).permute(0, 2, 3, 1)
        return SimpleNamespace(images=frames.numpy())


class TinyMVGen(MVGen):
    """An ``MVGen`` whose sampling pipeline is the fake one; ``pipelines`` keeps those it handed out."""

    def sampling_pipeline(self) -> FakePipeline:
        self.pipelines = [*getattr(self, "pipelines", []), FakePipeline(self)]
        return self.pipelines[-1]


def tiny_model() -> TinyMVGen:
    """A fresh stand-in for the released model: frozen, in eval mode, no adapters yet."""
    torch.manual_seed(0)
    unet, vae = TinyUNet().requires_grad_(False).eval(), FakeVAE().requires_grad_(False).eval()
    config = EasyDict(CONFIG, model_cfg=EasyDict(prior_type="3dpe+pixel", depth_mid_times=4))
    return TinyMVGen(unet=unet, vae=vae, config=config, scheduler=DDPMScheduler.from_config(SCHEDULER))


def adapted_model(trained: bool = True, **finetune) -> TinyMVGen:
    """A stand-in with adapters; ``trained``: their zero-initialised layers are made non-zero."""
    model = tiny_model()
    prepare_for_finetuning(model, FinetuneConfig(**{"date_freqs": 4, "date_hidden": 16, **finetune}))
    if trained:
        torch.manual_seed(1)
        nn.init.normal_(model.date.embedding.mlp[-1].weight, std=0.5)
    return model


def fake_camera_embedding(K, w2c, b, f, h, w, config):
    return torch.full((b, f, 6, h, w), 2.0)


def fake_3d_priors(config, depth, K, w2c, cond_num, nframe, device, colors=None, latents=None, vae=None,
                   prior_type="3dpe"):
    """Recognisable priors of the released model's shapes: coordinates 3.0, warped pixels 4.0."""
    h, w = depth.shape[-2:]
    return [torch.full((nframe, 193, h // 8, w // 8), 3.0), torch.full((nframe, 4, h, w), 4.0)]


@pytest.fixture(autouse=True)
def fake_conditioning(monkeypatch):
    monkeypatch.setattr(step, "get_camera_embedding", fake_camera_embedding)
    monkeypatch.setattr(step, "get_3d_priors", fake_3d_priors)


def write_folder(root: Path, size: int = SIZE, camera_extent: float = 0.2 * (N - 1)) -> Path:
    """A dataset folder of 24 views for crops of side ``size``: plain images (red = 10 x row), 1.5 times as wide
    as high, cameras 0.2 apart along x that look along z, and a constant depth map per view.

    Rows 0-7 are old, 8-15 modern, 16-23 after the fire; every view is usable, has reliable depth and
    shares 100 points with every other view. ``camera_extent`` is what ``scene.json`` gives as the
    extent of all the scene's cameras: by default that of the 24 views (4.6, under MVGenMaster's limit
    of 5, so the scene-scale factor is 1); a larger one stands for cameras that are not in the table.
    """
    (root / "images").mkdir(parents=True)
    (root / "depth").mkdir()
    date_mid = np.concatenate([1900 + 0.5 * np.arange(8), 2012 + 0.1 * np.arange(8), 2020 + 0.1 * np.arange(8)])
    rows = []
    for i in range(N):
        path = root / "images" / f"view_{i:02d}.png"
        Image.fromarray(np.full((size, 3 * size // 2, 3), (10 * i, 100, 200), dtype=np.uint8)).save(path)
        rows.append(dict(file_key=path.name, image_path=str(path), image_width=3 * size // 2, image_height=size,
                         camera_width=3 * size, camera_height=2 * size, fx=2.5 * size, fy=2.5 * size, cx=1.5 * size,
                         cy=float(size), k1=0.0, k2=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0, tx=-0.2 * i, ty=0.0, tz=0.0,
                         date_lo=date_mid[i] - 0.25, date_hi=date_mid[i] + 0.25, date_mid=date_mid[i], usable=True))
        np.save(depth_path(root, i), np.full((size, size), 1.0 + 0.125 * i, dtype=np.float16))
    views = pd.DataFrame(rows)
    views.to_parquet(root / "views.parquet", index=False)
    covis = np.full((N, N), 100, dtype=np.uint16)
    np.fill_diagonal(covis, 400)
    np.save(root / "covisibility.npy", covis)
    pd.DataFrame({"row": np.arange(N), "file_key": views["file_key"], "n_points": 100, "scale": 1.0,
                  "median_rel_error": 0.05, "ok": True}).to_parquet(root / "depth_stats.parquet", index=False)
    (root / SCENE_FILE).write_text(json.dumps({
        "scene_id": 0, "model": 0, "n_registered_cameras": N, "camera_min": [0.0, 0.0, 0.0],
        "camera_max": [camera_extent, 0.0, 0.0], "camera_extent": camera_extent}))
    return root


@pytest.fixture
def folder(tmp_path) -> Path:
    return write_folder(tmp_path / "scene")


@pytest.fixture
def scene(folder) -> MVGenScene:
    return MVGenScene(folder, resolution=SIZE)


def config(folder: Path, run_dir: Path, **changes) -> TrainConfig:
    """A three-step run on the folder with everything optional switched off; ``changes`` override fields."""
    base = dict(dataset=str(folder), run_dir=str(run_dir), resolution=SIZE, steps=3, grad_accum=2, lr_date=1e-2,
                lr_lora=1e-2, warmup_steps=0, lr_schedule="constant", ema_decay=0.0, num_workers=0, log_every=1,
                probe_every=0, render_every=0, save_every=0, latest_every=0, probe_samples=2,
                probe_timesteps=(200, 800), render_steps=2, loss=LossConfig(precision="fp32"),
                finetune=FinetuneConfig(date_freqs=4, date_hidden=16, lora_rank=2))
    return TrainConfig(**{**base, **changes})


def watch_training(monkeypatch, stop_after: int | None = None) -> list[list[int]]:
    """Record the rows of every sample the trainer's loss is given; raise once ``stop_after`` samples were seen."""
    seen, real = [], trainer_module.training_loss

    def watched(model, sample, *args):
        if len(seen) == stop_after:
            raise RuntimeError("job killed")
        seen.append(sample["rows"].tolist())
        return real(model, sample, *args)

    monkeypatch.setattr(trainer_module, "training_loss", watched)
    return seen


def read_log(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (Path(run_dir) / "log.jsonl").read_text().splitlines()]


def flat_weights(model: MVGen) -> torch.Tensor:
    """Every trainable weight of the model in one vector."""
    state = trainable_state_dict(model)
    return torch.cat([state[part][key].flatten() for part in ("date", "lora") for key in sorted(state[part])])


def record_dates(model: MVGen) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Remember the (year, valid) the date embedding is asked for at every UNet forward pass."""
    seen = []
    model.date.embedding.register_forward_hook(
        lambda module, args, output: seen.append((args[0].clone(), args[1].clone())))
    return seen


def is_zero(prior: torch.Tensor) -> torch.Tensor:
    """Bool per view: is its part of a prior all zeros?"""
    return prior.flatten(1).abs().amax(1) == 0


def load_script(name: str):
    """A script of ``scripts/mvgen`` as a module (registered by name, as its dataclasses need)."""
    spec = importlib.util.spec_from_file_location(f"mvgen_{name}_script", REPO / "scripts" / "mvgen" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- configuration and small parts
@pytest.mark.parametrize("changes", [
    dict(steps=0), dict(grad_accum=0), dict(log_every=0), dict(lr_date=0.0), dict(lr_lora=-1e-4),
    dict(lr_schedule="linear"), dict(warmup_steps=-1), dict(lr_min_ratio=1.5), dict(grad_clip_factor=1.0),
    dict(ema_decay=1.0), dict(ema_decay=-0.1), dict(probe_every=-1), dict(save_every=-5), dict(probe_samples=0),
    dict(probe_timesteps=()), dict(probe_timesteps=(-1,)), dict(render_dates=()), dict(render_dates=("yesterday",)),
    dict(render_seeds=()), dict(render_steps=0), dict(render_uncond_dates="none"),
    dict(swap_years=(1890.0, 2015.0)), dict(swap_years=(2015.0, 1890.0, 2021.0)),
    dict(pinned_rows=((0, 1, 2),)), dict(pinned_rows=((0, 1, 2, 2),)), dict(render_rows=((0, 1, -2, 3),)),
    dict(holdout=HoldoutConfig(scenes=(25512,))),
    dict(pinned_rows=(MODERN_FROM_MODERN,), holdout=HoldoutConfig(fraction=0.1)),
    dict(scale="world"), dict(scale=0.0), dict(scale=-0.5), dict(scale=float("nan")), dict(scale=None),
])
def test_config_rejects_invalid_values(changes, tmp_path):
    with pytest.raises(ValueError):
        config(tmp_path, tmp_path / "run", **changes)


def test_config_normalises_sequences_and_keeps_the_plans_defaults(tmp_path):
    cfg = TrainConfig(dataset="d", run_dir="r", pinned_rows=[[8, 9, 10, 11]], render_dates=["true", "2015", 1890],
                      swap_years=[1890, 2015, 2021], probe_timesteps=[50, 950])
    assert cfg.pinned_rows == ((8, 9, 10, 11),) and cfg.render_dates == ("true", 2015.0, 1890.0)
    assert cfg.swap_years == SWAP_YEARS and cfg.probe_timesteps == (50, 950)
    assert json.loads(json.dumps(asdict(cfg)))["sampling"]["period_edges"] == list(EDGES)
    default = TrainConfig(dataset="d", run_dir="r")
    assert (default.lr_schedule, default.grad_accum, default.loss.precision) == ("cosine", 4, "bf16")
    assert default.sampling.period_edges == EDGES and default.finetune.grad_checkpointing
    assert TrainConfig(dataset="d", run_dir="r", grad_clip_factor=None).grad_clip_factor is None
    assert default.scale == "scene"                     # one factor per scene, as in the authors' training


def test_scale_settings_are_parsed():
    assert [parse_scale(value) for value in ("scene", "sample", "0.5", 0.0868, 1, "2e-1")] == \
        ["scene", "sample", 0.5, 0.0868, 1.0, 0.2]
    assert isinstance(parse_scale(1), float)
    for bad in ("world", "", "0", "-1", "nan", "inf", 0, None):
        with pytest.raises(ValueError, match="scale"):
            parse_scale(bad)
    assert TrainConfig(dataset="d", run_dir="r", scale="0.25").scale == 0.25
    assert json.loads(json.dumps(asdict(TrainConfig(dataset="d", run_dir="r", scale="sample"))))["scale"] == "sample"


def test_lr_factor():
    assert [lr_factor(s, 4, "constant", 100, 0.1) for s in (0, 1, 3, 4, 99)] == [0.25, 0.5, 1.0, 1.0, 1.0]
    assert lr_factor(0, 0, "cosine", 10, 0.1) == 1.0                                # no warm-up
    assert lr_factor(4, 4, "cosine", 104, 0.1) == 1.0                               # the decay starts after it
    assert lr_factor(54, 4, "cosine", 104, 0.1) == pytest.approx(0.55)              # half way
    assert lr_factor(104, 4, "cosine", 104, 0.1) == pytest.approx(0.1)
    assert lr_factor(500, 4, "cosine", 104, 0.1) == pytest.approx(0.1)              # and stays at the floor
    factors = [lr_factor(s, 4, "cosine", 104, 0.1) for s in range(4, 105)]
    assert all(a >= b for a, b in zip(factors, factors[1:]))


def test_weight_ema_warms_up_and_swaps_weights_in_and_out():
    param = nn.Parameter(torch.zeros(3))
    ema = WeightEMA([param], decay=0.5)
    expected = torch.zeros(3)
    for n in range(1, 12):
        with torch.no_grad():
            param.fill_(float(n))
        ema.update()
        decay = min(0.5, (1 + n) / (10 + n))                 # 2/11, 3/12, ... reaches 0.5 at the 8th update
        expected = decay * expected + (1 - decay) * n
        torch.testing.assert_close(ema.shadow[0], expected)
    assert ema.updates == 11 and min(0.5, 2 / 11) < 0.2 and min(0.5, 9 / 18) == 0.5
    with ema.applied():
        torch.testing.assert_close(param.data, expected)
    assert torch.equal(param.data, torch.full((3,), 11.0)) and param.requires_grad
    with pytest.raises(RuntimeError, match="probe failed"), ema.applied():      # the raw weights come back anyway
        raise RuntimeError("probe failed")
    assert torch.equal(param.data, torch.full((3,), 11.0))
    ema.reset(updates=4)
    assert torch.equal(ema.shadow[0], param.data) and ema.updates == 4


def test_mode_dependent_modules_finds_what_train_mode_would_change():
    inert = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.0), nn.GroupNorm(1, 2), nn.SiLU())
    assert mode_dependent_modules(inert) == []
    assert mode_dependent_modules(tiny_model().unet) == []      # MVGenMaster's ResBlocks: dropout rate 0
    active = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.1), nn.BatchNorm1d(2), nn.Dropout2d(0.2))
    assert mode_dependent_modules(active) == ["1", "2", "3"]


def test_trainer_refuses_a_unet_that_train_mode_would_change(folder, tmp_path):
    model = tiny_model()
    model.unet.resnets[0].dropout.p = 0.1
    with pytest.raises(ValueError, match="train\\(\\) mode"):
        Trainer(config(folder, tmp_path / "run"), model=model)
    unchecked = config(folder, tmp_path / "run2", finetune=FinetuneConfig(date_freqs=4, date_hidden=16,
                                                                          grad_checkpointing=False))
    model = tiny_model()
    model.unet.resnets[0].dropout.p = 0.1
    Trainer(unchecked, model=model)                             # stays in eval(): the dropout is never active


def test_trainer_needs_something_to_train(folder, tmp_path):
    with pytest.raises(ValueError, match="nothing to train"):
        Trainer(config(folder, tmp_path / "run", finetune=FinetuneConfig(date=False)), model=tiny_model())


# ----------------------------------------------------------------------------- the training loop
def test_three_steps_accumulate_log_and_save(folder, tmp_path, monkeypatch):
    cfg = config(folder, tmp_path / "run")
    trainer = Trainer(cfg, model=tiny_model())
    unet = trainer.model.unet
    frozen = {name: p.clone() for name, p in unet.named_parameters() if not p.requires_grad}
    start = flat_weights(trainer.model)
    events, real_loss = [], trainer_module.training_loss

    def counted_loss(*args):
        events.append("sample")
        return real_loss(*args)

    monkeypatch.setattr(trainer_module, "training_loss", counted_loss)
    trainer.optimizer.register_step_post_hook(lambda optimizer, args, kwargs: events.append("step"))
    trainer.train()

    assert events == ["sample", "sample", "step"] * 3 and trainer.step == 3     # one step per grad_accum samples
    assert len(unet.seen) == 6 and all(call["training"] and call["grad"] for call in unet.seen)
    assert not unet.training                                                    # left in eval() mode
    assert not torch.equal(flat_weights(trainer.model), start)
    assert all(p.abs().sum() > 0 for p in trainer.params)                       # the zero-initialised layers too
    assert all(torch.equal(p, frozen[name]) for name, p in unet.named_parameters() if name in frozen)

    log = read_log(cfg.run_dir)
    assert [(r["step"], r["samples"]) for r in log] == [(1, 2), (2, 4), (3, 6)]
    for record in log:
        assert record["loss"] > 0 and record["grad_norm_median"] > 0 and record["clipped_fraction"] == 0.0
        assert record["lr_date"] == record["lr_lora"] == 1e-2 and record["skipped_steps"] == 0
    run_dir = Path(cfg.run_dir)
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["latest.pt", "step_000003.pt"]
    saved_config = json.loads((run_dir / "config.json").read_text())
    assert saved_config["grad_accum"] == 2 and (saved_config["scale"], saved_config["scale_factor"]) == ("scene", 1.0)
    assert {name: value for name, value in saved_config.items() if name != "scale_factor"} == \
        json.loads(json.dumps(asdict(cfg)))
    assert json.loads((run_dir / "probes.json").read_text())["date_loss"] == {}
    ckpt = torch.load(run_dir / "checkpoints" / "step_000003.pt", weights_only=False)
    assert ckpt["step"] == 3 and ckpt["precision"] == "fp32" and ckpt["weights"] == "raw"
    assert (ckpt["scale"], ckpt["scale_factor"]) == ("scene", 1.0) and ckpt["train_config"] == asdict(cfg)
    assert ckpt["train_config"]["loss"]["precision"] == "fp32" and "optimizer" not in ckpt
    assert FinetuneConfig(**ckpt["finetune_config"]) == cfg.finetune
    torch.testing.assert_close(ckpt["trainable"]["date"]["embedding.mlp.2.weight"],
                               trainer.model.date.embedding.mlp[-1].weight.detach())


def test_first_update_uses_the_mean_gradient_of_the_accumulated_samples(folder, tmp_path):
    """Two pinned samples, grad_accum 2, one step of plain AdamW: the update is that of the two losses' mean."""
    cfg = config(folder, tmp_path / "run", steps=1, pinned_rows=(MODERN_FROM_MODERN, AFTER_FROM_MODERN), seed=3,
                 finetune=FinetuneConfig(date_freqs=4, date_hidden=16))
    trainer = Trainer(cfg, model=tiny_model())
    trainer.train()

    twin = Trainer(config(folder, tmp_path / "twin", steps=1, pinned_rows=cfg.pinned_rows, seed=3,
                          finetune=cfg.finetune), model=tiny_model())
    twin.model.unet.train()
    losses = [step.training_loss(twin.model, sample, cfg.loss, twin.generator)[0] for sample in twin.pinned]
    (sum(losses) / 2).backward()
    twin.optimizer.step()
    torch.testing.assert_close(flat_weights(trainer.model), flat_weights(twin.model), rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("checkpointing", [True, False])
def test_unet_is_in_train_mode_exactly_when_checkpointing(folder, tmp_path, checkpointing):
    """MVGenMaster's blocks checkpoint only in train() mode: backward() then runs every ResBlock a second time."""
    finetune = FinetuneConfig(date_freqs=4, date_hidden=16, grad_checkpointing=checkpointing)
    trainer = Trainer(config(folder, tmp_path / "run", steps=2, grad_accum=1, finetune=finetune), model=tiny_model())
    unet, calls = trainer.model.unet, []
    unet.resnets[0].register_forward_pre_hook(lambda module, args: calls.append(1))
    trainer.train()
    assert [call["training"] for call in unet.seen] == [checkpointing] * 2
    assert len(calls) == (4 if checkpointing else 2)
    assert all(p.abs().sum() > 0 for p in trainer.params)


def test_spike_guard_clips_only_above_a_multiple_of_the_median_norm(folder, tmp_path):
    trainer = Trainer(config(folder, tmp_path / "run", grad_clip_factor=10.0), model=tiny_model())
    assert trainer.clip_limit() == float("inf")
    trainer.grad_norm_history.extend([2.0] * (SPIKE_GUARD_MIN_HISTORY - 1))
    assert trainer.clip_limit() == float("inf")                 # not enough history yet
    trainer.grad_norm_history.append(4.0)
    assert trainer.clip_limit() == 20.0
    unguarded = Trainer(config(folder, tmp_path / "run2", grad_clip_factor=None), model=tiny_model())
    unguarded.grad_norm_history.extend([2.0] * 100)
    assert unguarded.clip_limit() == float("inf")


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
def test_a_spike_is_clipped_and_a_non_finite_gradient_skips_the_update(folder, tmp_path, monkeypatch):
    cfg = config(folder, tmp_path / "run", steps=2, grad_accum=1, grad_clip_factor=2.0)
    trainer = Trainer(cfg, model=tiny_model())
    trainer.grad_norm_history.extend([1e-9] * SPIKE_GUARD_MIN_HISTORY)         # any real gradient is a spike now
    trainer.train()
    assert [r["clipped_fraction"] for r in read_log(cfg.run_dir)] == [1.0, 1.0]

    cfg = config(folder, tmp_path / "nan", steps=2, grad_accum=1)
    trainer = Trainer(cfg, model=tiny_model())
    start, real = flat_weights(trainer.model), trainer_module.training_loss
    monkeypatch.setattr(trainer_module, "training_loss",
                        lambda *args: (real(*args)[0] * float("nan"), real(*args)[1]))
    trainer.train()
    assert trainer.step == 2 and torch.equal(flat_weights(trainer.model), start)
    assert [r["skipped_steps"] for r in read_log(cfg.run_dir)] == [1, 1]
    assert all(r["grad_norm_median"] is None for r in read_log(cfg.run_dir))


def test_ema_is_what_probes_and_checkpoints_use(folder, tmp_path):
    cfg = config(folder, tmp_path / "run", steps=3, grad_accum=1, ema_decay=0.5, probe_every=1,
                 pinned_rows=(MODERN_FROM_MODERN,))
    trainer = Trainer(cfg, model=tiny_model())
    weight = trainer.model.date.embedding.mlp[-1].weight
    raw, probed = [], []
    trainer.optimizer.register_step_post_hook(lambda optimizer, args, kwargs: raw.append(weight.detach().clone()))

    class WeightReader:
        """Stands in for the date-loss probe: notes the weights, mode and gradient switch it runs under."""
        last_losses = {}

        def __call__(self, model):
            probed.append((weight.detach().clone(), model.unet.training, torch.is_grad_enabled()))
            return {"loss_true": 1.0, "change": 0.0, "rel_gap_swapped": 0.0, "rel_gap_unknown": 0.0,
                    "date_rms": 0.0, "date_temb_ratio": 0.0}

    trainer.loss_probes = {"pinned": WeightReader()}
    trainer.train()

    expected, averages = torch.zeros_like(weight), []
    for n, value in enumerate(raw, start=1):
        decay = min(0.5, (1 + n) / (10 + n))
        expected = decay * expected + (1 - decay) * value
        averages.append(expected)
    assert len(probed) == 4 and torch.count_nonzero(probed[0][0]) == 0          # step 0: still the zero init
    for (seen, training, grad), average in zip(probed[1:], averages):
        torch.testing.assert_close(seen, average)
        assert not training and not grad
    assert torch.equal(weight.detach(), raw[-1]) and trainer.model.unet.training is False   # raw weights are back
    assert not torch.allclose(raw[-1], averages[-1])

    for name in ("step_000003.pt", "latest.pt"):
        ckpt = torch.load(Path(cfg.run_dir) / "checkpoints" / name, weights_only=False)
        torch.testing.assert_close(ckpt["trainable"]["date"]["embedding.mlp.2.weight"], averages[-1])
        assert ckpt["weights"] == "ema"
    assert torch.equal(ckpt["raw_trainable"]["date"]["embedding.mlp.2.weight"], raw[-1]) and ckpt["ema_updates"] == 3


# ----------------------------------------------------------------------------- pinned view sets
def test_pinned_rows_cycle_in_order(folder, tmp_path, monkeypatch):
    sets = (MODERN_FROM_MODERN, AFTER_FROM_MODERN)
    seen = watch_training(monkeypatch)
    one = Trainer(config(folder, tmp_path / "one", steps=3, grad_accum=1, pinned_rows=sets), model=tiny_model())
    one.train()
    assert seen == [list(sets[0]), list(sets[1]), list(sets[0])]
    seen.clear()
    Trainer(config(folder, tmp_path / "two", steps=2, grad_accum=2, pinned_rows=sets), model=tiny_model()).train()
    assert seen == [list(sets[0]), list(sets[1])] * 2                           # every step sees both
    seen.clear()
    three = (*sets, (0, 1, 2, 3))
    Trainer(config(folder, tmp_path / "three", steps=2, grad_accum=2, pinned_rows=three), model=tiny_model()).train()
    assert seen == [list(three[0]), list(three[1]), list(three[2]), list(three[0])]

    pinned = json.loads((tmp_path / "one" / "pinned.json").read_text())
    assert [p["rows"] for p in pinned] == [list(rows) for rows in sets] and pinned[0]["n_ref"] == 3
    assert pinned[1]["file_keys"] == [f"view_{row:02d}.png" for row in sets[1]]
    assert pinned[1]["years"][3:] == [2020.0, 2020.1, 2020.2] and pinned[0]["not_usable"] == {}
    assert pinned[0]["periods"] == [1] * 6 and pinned[1]["periods"] == [1] * 3 + [2] * 3
    assert pinned[0]["references_without_reliable_depth"] == []
    assert [s["n_ref"] for s in one.pinned] == [3, 3] and one.pinned[1]["rows"].tolist() == list(sets[1])


def test_a_pinned_run_probes_and_renders_its_own_view_sets_and_needs_no_depth_table(folder, tmp_path, capsys):
    (folder / "depth_stats.parquet").unlink()                   # only the references' depth maps are needed
    sets = (AFTER_FROM_MODERN, MODERN_FROM_MODERN)
    trainer = Trainer(config(folder, tmp_path / "run", pinned_rows=sets, probe_every=5, render_every=5),
                      model=tiny_model())
    printed = capsys.readouterr().out
    assert "pinned view set 0, references without reliable depth: ['view_08.png'" in printed
    # every pinned set's rows, dates and periods are printed at start-up
    assert ("[train] pinned view set 0: references rows [8, 9, 10] dated [2012.0, 2012.1, 2012.2] (1970 to 2019.2877); "
            "targets rows [16, 17, 18] dated [2020.0, 2020.1, 2020.2] (from 2019.2877)") in printed
    assert "[train] pinned view set 1: references rows [8, 9, 10]" in printed
    assert list(trainer.loss_probes) == ["pinned"] and trainer.loss_probes["pinned"].rows == [list(s) for s in sets]
    assert [(p.name, p.rows) for p in trainer.render_probes] == [("set0", list(sets[0])), ("set1", list(sets[1]))]
    # swapped dates: the median date of the pinned targets of each period; no pinned target is old
    assert trainer.swap_years == pytest.approx((1901.75, 2012.4, 2020.1))
    assert trainer.loss_probes["pinned"].swapped[0].tolist() == pytest.approx([2012.0, 2012.1, 2012.2] + [2012.4] * 3)
    assert trainer.loss_probes["pinned"].swapped[1].tolist() == pytest.approx([2012.0, 2012.1, 2012.2] + [2020.1] * 3)
    records = json.loads((tmp_path / "run" / "probes.json").read_text())
    assert records["swap_years"] == pytest.approx([1901.75, 2012.4, 2020.1])
    assert records["date_loss"]["pinned"][0]["swapped_years"] == pytest.approx([2012.4] * 3)
    assert [r["name"] for r in records["renders"]] == ["set0", "set1"]

    depth_path(folder, 9).unlink()
    with pytest.raises(FileNotFoundError, match="prepare_depth"):
        Trainer(config(folder, tmp_path / "run2", pinned_rows=sets), model=tiny_model())
    with pytest.raises(FileNotFoundError, match="depth_stats"):          # sampling the scene needs the table
        Trainer(config(folder, tmp_path / "run3"), model=tiny_model())
    with pytest.raises(ValueError, match="not in the view table"):
        Trainer(config(folder, tmp_path / "run4", pinned_rows=((0, 1, 2, N),)), model=tiny_model())


def test_describe_pinned_names_every_period_of_the_references_and_of_the_targets():
    views = {"rows": [3, 9, 17, 16, 4], "n_ref": 3, "years": [1901.5, 2012.1, 2020.1, 2020.0, 1902.0],
             "periods": [0, 1, 2, 2, 0]}
    assert describe_pinned(views, EDGES) == (
        "references rows [3, 9, 17] dated [1901.5, 2012.1, 2020.1] (before 1970, 1970 to 2019.2877, from 2019.2877); "
        "targets rows [16, 4] dated [2020.0, 1902.0] (from 2019.2877, before 1970)")
    assert describe_pinned({**views, "periods": [0] * 5}, ()).count("(all)") == 2


def test_scene_scale_reaches_training_probes_renders_and_records(tmp_path, capsys):
    folder = write_folder(tmp_path / "wide", camera_extent=10.0)        # all the scene's cameras span 10 units
    cfg = config(folder, tmp_path / "run", steps=1, grad_accum=1, pinned_rows=(AFTER_FROM_MODERN,), probe_every=1,
                 render_every=1, render_dates=("true",), render_seeds=(0,))
    trainer = Trainer(cfg, model=tiny_model())
    assert (trainer.scene.scale, trainer.scene.scale_factor) == ("scene", 0.5)      # 5 / 10: the authors' rule
    assert "[train] scene scale 'scene': factor 0.5 on camera translations and depth" in capsys.readouterr().out
    unscaled = MVGenScene(folder, resolution=SIZE, scale=1.0).make_sample(np.array(AFTER_FROM_MODERN), 3)
    assert unscaled["w2c"][:, 0, 3].abs().max() > 1 and unscaled["depth"][:3].min() > 1
    for sample in (trainer.pinned[0], trainer.loss_probes["pinned"].samples[0], trainer.render_probes[0].sample):
        assert sample["scale"] == 0.5
        torch.testing.assert_close(sample["w2c"][:, :3, 3], unscaled["w2c"][:, :3, 3] * 0.5)
        torch.testing.assert_close(sample["depth"], unscaled["depth"] * 0.5)
        assert torch.equal(sample["w2c"][:, :3, :3], unscaled["w2c"][:, :3, :3])
    trainer.train()

    run_dir = Path(cfg.run_dir)
    assert json.loads((run_dir / "config.json").read_text())["scale_factor"] == 0.5
    ckpt = torch.load(run_dir / "checkpoints" / "latest.pt", weights_only=False)
    assert (ckpt["scale"], ckpt["scale_factor"], ckpt["train_config"]["scale"]) == ("scene", 0.5, "scene")
    records = json.loads((run_dir / "probes.json").read_text())
    assert records["date_loss"]["pinned"][0]["scale_factor"] == 0.5
    assert (records["renders"][0]["scale"], records["renders"][0]["scale_factor"]) == ("scene", 0.5)
    rendered = json.loads((run_dir / "renders" / "step_000001" / "set0.json").read_text())
    assert (rendered["scale"], rendered["scale_factor"]) == ("scene", 0.5)
    assert json.loads((run_dir / "probes" / "pinned_baseline.json").read_text())["scale_factors"] == [0.5]
    call = trainer.model.pipelines[-1].calls[0]                         # what the sampling pipeline was given
    torch.testing.assert_close(call["w2c"][:, :3, 3], unscaled["w2c"][:, :3, 3] * 0.5)
    torch.testing.assert_close(call["depth"].float(), unscaled["depth"] * 0.5)

    # a constant factor, and the demo's rule per sample (these six cameras span 2 units: left alone)
    for scale, factor, said in ((0.25, 0.25, "factor 0.25 on"), ("sample", None, "a factor per sample")):
        other = Trainer(config(folder, tmp_path / f"run_{scale}", scale=scale, pinned_rows=(AFTER_FROM_MODERN,)),
                        model=tiny_model())
        assert other.scene.scale_factor == factor and said in capsys.readouterr().out
        assert other.pinned[0]["scale"] == (factor or 1.0)
        assert torch.load(other.save(), weights_only=False)["scale_factor"] == factor
        assert json.loads((Path(other.cfg.run_dir) / "config.json").read_text())["scale_factor"] == factor
    with pytest.raises(ValueError, match="opened with scale 'sample', the run's is 'scene'"):
        Trainer(config(folder, tmp_path / "run_x"), model=tiny_model(),
                scene=MVGenScene(folder, resolution=SIZE, scale="sample"))
    (folder / SCENE_FILE).unlink()                                      # "scene" needs the extent of all cameras
    with pytest.raises(FileNotFoundError, match=SCENE_FILE):
        Trainer(config(folder, tmp_path / "run_y"), model=tiny_model())


def test_log_lines_carry_the_size_of_the_date_embedding(folder, tmp_path, capsys):
    cfg = config(folder, tmp_path / "run", steps=3, grad_accum=1, pinned_rows=(AFTER_FROM_MODERN,), probe_every=3,
                 finetune=FinetuneConfig(date_freqs=4, date_hidden=16))
    trainer = Trainer(cfg, model=tiny_model())
    stats = []                                                  # the conditioner's record of every UNet forward pass
    trainer.model.unet.register_forward_hook(
        lambda module, args, output: stats.append(dict(trainer.model.date.last_stats)))
    trainer.train()
    log = read_log(cfg.run_dir)
    steps, probes = [r for r in log if "probe" not in r], [r for r in log if "probe" in r]
    training = stats[6:9]                                       # after the 2 x 3 passes of the probe at step 0
    assert [r["date_rms"] for r in steps] == pytest.approx([s["date_rms"] for s in training])
    assert [r["date_temb_ratio"] for r in steps] == pytest.approx([s["date_rms"] / s["temb_rms"] for s in training])
    assert steps[0]["date_rms"] == steps[0]["date_temb_ratio"] == 0.0           # the first pass: still zero
    assert steps[2]["date_rms"] > 0 and steps[2]["date_temb_ratio"] > 0         # then the embedding has moved
    assert probes[0]["date_temb_ratio"] == 0.0 and probes[1]["date_temb_ratio"] > 0
    printed = capsys.readouterr().out
    assert f"date embedding {steps[2]['date_rms']:.3g} ({steps[2]['date_temb_ratio']:.3f} of temb)" in printed
    assert f"date embedding / timestep embedding {probes[1]['date_temb_ratio']:.3f}" in printed

    lora_only = config(folder, tmp_path / "lora", steps=1, pinned_rows=(AFTER_FROM_MODERN,), probe_every=1,
                       finetune=FinetuneConfig(date=False, lora_rank=2))
    Trainer(lora_only, model=tiny_model()).train()
    assert all(r["date_rms"] is None and r["date_temb_ratio"] is None for r in read_log(lora_only.run_dir))
    printed = capsys.readouterr().out
    assert "no date embedding" in printed and "of temb" not in printed


def test_training_steps_use_the_runs_period_edges_for_the_cross_date_rule(folder, tmp_path):
    """Targets of 2020 with references of 2012: within a gap of 20 years, but on the other side of the fire."""
    loss = LossConfig(precision="fp32", cross_date_pixels="drop_all", cross_date_gap=20.0, prior_dropout=0.0,
                      prior_view_dropout=0.0)
    targets = [False] * 3 + [True] * 3
    for edges, dropped in ((EDGES, targets), ((1970.0,), [False] * 6)):
        cfg = config(folder, tmp_path / f"run{len(edges)}", steps=1, grad_accum=1, pinned_rows=(AFTER_FROM_MODERN,),
                     loss=loss, sampling=ViewSamplingConfig(period_edges=edges))
        trainer = Trainer(cfg, model=tiny_model())
        trainer.train()
        coordinates, pixels = trainer.model.unet.seen[0]["coords"]
        assert is_zero(coordinates).tolist() == is_zero(pixels).tolist() == dropped


def test_training_samples_take_their_number_of_references_from_the_sampler(folder, tmp_path, monkeypatch):
    sampling = ViewSamplingConfig(n_ref=2, n_targets=3, n_ref_choices=(1, 2, 3), min_targets_seen=2, top_k=4)
    seen, real = [], trainer_module.training_loss
    monkeypatch.setattr(trainer_module, "training_loss", lambda model, sample, *args: (
        seen.append((sample["n_ref"], len(sample["rows"]))), real(model, sample, *args))[1])
    trainer = Trainer(config(folder, tmp_path / "run", steps=8, sampling=sampling, probe_every=8), model=tiny_model())
    trainer.train()
    assert {n_ref for n_ref, _ in seen} == {1, 2, 3} and {n_views for _, n_views in seen} == {5}
    assert all(len(rows) == 5 for rows in trainer.loss_probes["train"].rows)    # the probe's sets: 2 references each
    assert trainer.loss_probes["train"].n_ref == 2


def test_view_sets_by_file_key(tmp_path):
    table = ["a.jpg", "Cathédrale_façade.jpg", "b.jpg", "Île.jpg", "twice.jpg", "twice.jpg"]
    assert rows_of_file_keys(table, ["b.jpg", "a.jpg"]) == [2, 0]
    decomposed = unicodedata.normalize("NFD", "Cathédrale_façade.jpg")
    assert decomposed != table[1] and rows_of_file_keys(table, [decomposed, "Île.jpg"]) == [1, 3]
    assert rows_of_file_keys([unicodedata.normalize("NFD", key) for key in table[:4]], [table[1]]) == [1]
    with pytest.raises(KeyError, match="missing.jpg"):
        rows_of_file_keys(table, ["a.jpg", "missing.jpg"])
    with pytest.raises(KeyError, match="several rows"):
        rows_of_file_keys(table, ["twice.jpg"])

    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps([{"refs": ["b.jpg"], "targets": ["Île.jpg", "a.jpg"], "note": "fire"},
                                     {"refs": ["a.jpg"], "targets": ["b.jpg"]}]))
    assert read_keys_file(keys_file, table, n_ref=1) == ((2, 3, 0), (0, 2))
    with pytest.raises(ValueError, match="sample 0 has 1 references, the run uses 3"):
        read_keys_file(keys_file, table, n_ref=3)
    keys_file.write_text(json.dumps([{"refs": ["a.jpg"]}]))
    with pytest.raises(ValueError, match="refs"):
        read_keys_file(keys_file, table, n_ref=1)
    keys_file.write_text(json.dumps({"refs": ["a.jpg"], "targets": ["b.jpg"]}))
    with pytest.raises(ValueError, match="JSON list"):
        read_keys_file(keys_file, table, n_ref=1)

    rows_file = tmp_path / "rows.json"
    rows_file.write_text("[[3, 1, 2, 0], [5, 4, 3]]")
    assert read_rows_file(rows_file) == ((3, 1, 2, 0), (5, 4, 3))
    rows_file.write_text("[3, 1, 2]")
    with pytest.raises(ValueError, match="row lists"):
        read_rows_file(rows_file)


def test_check_view_sets_and_describe_views(scene):
    check_view_sets([(0, 1, 2, 3)], n_ref=3, n_views=N)
    for bad, n_views in (((0, 1, 2), None), ((0, 1, 1, 2), None), ((0, 1, 2, N), N), ((-1, 1, 2, 3), None)):
        with pytest.raises(ValueError, match="view set 0"):
            check_view_sets([bad], n_ref=3, n_views=n_views)
    check_view_sets([(0, 1, 2, N)], n_ref=3)                    # without a table the rows cannot be checked

    scene.usable[9] = False
    scene.depth_ok[[8, 16]] = False                             # only a reference's depth matters
    described = describe_views(scene, (8, 9, 16), n_ref=2)
    assert described == {"rows": [8, 9, 16], "n_ref": 2, "file_keys": ["view_08.png", "view_09.png", "view_16.png"],
                         "years": [2012.0, 2012.1, 2020.0], "periods": [1, 1, 2],
                         "not_usable": {"view_09.png": "not usable"},
                         "references_without_reliable_depth": ["view_08.png"]}
    scene.views["excluded"] = ["rolled" if row == 16 else "" for row in range(N)]      # the v2 table says why
    assert describe_views(scene, (8, 9, 16), n_ref=2)["not_usable"] == {"view_16.png": "rolled"}


# ----------------------------------------------------------------------------- starting and resuming
@pytest.mark.parametrize("pinned", [True, False])
def test_resumed_run_continues_exactly(folder, tmp_path, monkeypatch, pinned):
    sets = (MODERN_FROM_MODERN, AFTER_FROM_MODERN, (0, 1, 2, 3, 4)) if pinned else ()
    settings = dict(steps=4, grad_accum=1 if pinned else 2, warmup_steps=2, lr_schedule="cosine", ema_decay=0.9,
                    latest_every=2, pinned_rows=sets)

    with monkeypatch.context() as patch:
        samples_whole = watch_training(patch)
        whole = Trainer(config(folder, tmp_path / "whole", **settings), model=tiny_model())
        whole.train()

    cfg = config(folder, tmp_path / "cut", **settings)
    stop = 3 * cfg.grad_accum - 1                               # the job dies in its third step
    with monkeypatch.context() as patch:
        samples_cut = watch_training(patch, stop_after=stop)
        with pytest.raises(RuntimeError, match="job killed"):
            Trainer(cfg, model=tiny_model()).train()
    assert [r["step"] for r in read_log(cfg.run_dir)] == [1, 2]

    with monkeypatch.context() as patch:
        samples_resumed = watch_training(patch)
        resumed = Trainer(cfg, model=tiny_model(), resume=checkpoint_to_resume(cfg.run_dir, "auto"))
        assert resumed.step == 2 and len(resumed.grad_norm_history) == 2
        resumed.train()

    assert samples_cut[:2 * cfg.grad_accum] + samples_resumed == samples_whole          # the stream goes on
    if pinned:
        assert samples_whole == [list(sets[i % 3]) for i in range(4)]
    else:
        assert len({tuple(rows) for rows in samples_whole}) == 8                        # 8 different draws
    assert torch.equal(flat_weights(resumed.model), flat_weights(whole.model))          # bit for bit
    assert all(torch.equal(a, b) for a, b in zip(resumed.ema.shadow, whole.ema.shadow))
    assert resumed.ema.updates == whole.ema.updates == 4
    log_whole, log_cut = read_log(tmp_path / "whole"), read_log(cfg.run_dir)
    assert [r["step"] for r in log_cut] == [1, 2, 3, 4]
    assert [r["loss"] for r in log_cut] == [r["loss"] for r in log_whole]
    assert [r["lr_date"] for r in log_cut] == [r["lr_date"] for r in log_whole]
    assert log_whole[0]["lr_date"] < log_whole[1]["lr_date"] and log_whole[3]["lr_date"] < log_whole[2]["lr_date"]


def run_files(run_dir: Path) -> dict[str, bytes]:
    """Every file of a run folder, by its path in the folder, with its content."""
    return {str(path.relative_to(run_dir)): path.read_bytes() for path in sorted(Path(run_dir).rglob("*"))
            if path.is_file()}


def test_a_run_that_died_before_its_first_step_resumes_from_the_latest_of_step_0(folder, tmp_path, monkeypatch):
    """The reviewer's case: the probes of step 0 start the log long before the first periodic ``latest.pt``, and a
    folder with a log and no checkpoint cannot be launched again."""
    settings = dict(steps=3, warmup_steps=2, lr_schedule="cosine", ema_decay=0.9, latest_every=500, probe_every=2,
                    render_every=3, render_rows=(AFTER_FROM_MODERN,), render_dates=("true",), render_seeds=(0,))
    with monkeypatch.context() as patch:
        samples_whole = watch_training(patch)
        whole = Trainer(config(folder, tmp_path / "whole", **settings), model=tiny_model())
        whole.train()

    cfg = config(folder, tmp_path / "cut", **settings)
    run_dir = Path(cfg.run_dir)
    with monkeypatch.context() as patch:
        watch_training(patch, stop_after=0)                     # the job dies before its first training sample
        with pytest.raises(RuntimeError, match="job killed"):
            Trainer(cfg, model=tiny_model()).train()
    evaluated = [(0, "date_loss:train"), (0, "render:set0")]
    assert [(r["step"], r["probe"]) for r in read_log(run_dir)] == evaluated
    latest = checkpoint_to_resume(run_dir, "auto")              # the folder has a log, and something to resume from
    assert latest == run_dir / "checkpoints" / "latest.pt"
    ckpt = torch.load(latest, weights_only=False)
    assert (ckpt["step"], ckpt["grad_norm_history"], ckpt["ema_updates"]) == (0, [], 0)
    with pytest.raises(ValueError, match="already holds a run"):                # as before: not a second fresh start
        Trainer(cfg, model=tiny_model())
    before = run_files(run_dir)

    with monkeypatch.context() as patch:
        samples_resumed = watch_training(patch)
        resumed = Trainer(cfg, model=tiny_model(), resume=latest)
        assert resumed.step == 0 and resumed.started
        resumed.train()

    assert samples_resumed == samples_whole and len(samples_whole) == 6         # the same samples, from the first on
    assert torch.equal(flat_weights(resumed.model), flat_weights(whole.model))  # the same noise: bit for bit
    assert all(torch.equal(a, b) for a, b in zip(resumed.ema.shadow, whole.ema.shadow))

    def measured(run: Path) -> list[dict]:
        """A run's log without what depends on the machine: the speed and the memory."""
        return [{name: value for name, value in record.items() if name not in ("sec_per_step", "max_mem_gb")}
                for record in read_log(run)]

    # step 0 was evaluated once, and every later probe is read against that first evaluation
    assert measured(run_dir) == measured(tmp_path / "whole")
    assert [(r["step"], r["probe"]) for r in read_log(run_dir) if r["step"] == 0] == evaluated
    after = run_files(run_dir)
    for kept in ("probes/train_baseline.json", "renders/step_000000/set0.json", "renders/step_000000/set0.jpg"):
        assert after[kept] == before[kept] == (tmp_path / "whole" / kept).read_bytes()
    raw = [json.loads(line) for line in (run_dir / "probes" / "train_losses.jsonl").read_text().splitlines()]
    assert [r["step"] for r in raw] == [0, 2, 3]

    # without probes the first log line comes with the first step; the step-0 latest.pt is there before it
    bare = config(folder, tmp_path / "bare", latest_every=0)
    with monkeypatch.context() as patch:
        watch_training(patch, stop_after=0)
        with pytest.raises(RuntimeError, match="job killed"):
            Trainer(bare, model=tiny_model()).train()
    assert not (tmp_path / "bare" / "log.jsonl").exists()
    assert torch.load(tmp_path / "bare" / "checkpoints" / "latest.pt", weights_only=False)["step"] == 0


def test_resume_refuses_a_dataset_that_changed_under_the_run(tmp_path):
    """The reviewer's cases: the same settings at both launches, but another ``scene.json`` (so another scale
    factor under ``scale="scene"``) or another view table (so other samples) in the dataset folder."""
    folder = write_folder(tmp_path / "wide", camera_extent=10.0)
    settings = dict(grad_accum=1, latest_every=1, probe_every=2)
    cfg = config(folder, tmp_path / "run", steps=2, **settings)
    trainer = Trainer(cfg, model=tiny_model())
    trainer.train()
    run_dir = Path(cfg.run_dir)
    latest = run_dir / "checkpoints" / "latest.pt"
    for name in ("latest.pt", "step_000002.pt"):                # every checkpoint says which dataset it was trained on
        ckpt = torch.load(run_dir / "checkpoints" / name, weights_only=False)
        assert ckpt["scale_factor"] == 0.5 and ckpt["dataset_fingerprint"] == trainer.scene.fingerprint()
    before = run_files(run_dir)
    longer = config(folder, run_dir, steps=3, **settings)

    # the scene's cameras now span 20 units instead of 10: under the same scale="scene" the factor would halve
    scene_file = (folder / SCENE_FILE).read_text()
    (folder / SCENE_FILE).write_text(json.dumps({**json.loads(scene_file), "camera_extent": 20.0}))
    with pytest.raises(ValueError, match="scene-scale factor was 0.5 and is now 0.25.*scene.json") as refused:
        Trainer(longer, model=tiny_model(), resume=latest)
    assert "fingerprint" not in str(refused.value)              # the views are the same: only the factor is named
    (folder / SCENE_FILE).write_text(scene_file)

    # the view table was rebuilt and one view is not usable any more: other samples would be drawn
    views = pd.read_parquet(folder / "views.parquet")
    rebuilt = views.copy()
    rebuilt.loc[9, "usable"] = False
    rebuilt.to_parquet(folder / "views.parquet", index=False)
    with pytest.raises(ValueError, match="dataset fingerprint differs.*usable flags") as refused:
        Trainer(longer, model=tiny_model(), resume=latest)
    assert "scale factor" not in str(refused.value)
    (folder / SCENE_FILE).write_text(json.dumps({**json.loads(scene_file), "camera_extent": 20.0}))
    with pytest.raises(ValueError, match="scene-scale factor was 0.5.*dataset fingerprint differs"):       # both
        Trainer(longer, model=tiny_model(), resume=latest)
    (folder / SCENE_FILE).write_text(scene_file)
    views.to_parquet(folder / "views.parquet", index=False)

    # a checkpoint that does not say which dataset it was trained on cannot be checked: refused too
    unmarked = tmp_path / "unmarked.pt"
    torch.save({name: value for name, value in torch.load(latest, weights_only=False).items()
                if name != "dataset_fingerprint"}, unmarked)
    with pytest.raises(ValueError, match="does not record the dataset.*new run folder"):
        Trainer(longer, model=tiny_model(), resume=unmarked)
    assert run_files(run_dir) == before                         # every refused launch left the run folder as it was

    resumed = Trainer(longer, model=tiny_model(), resume=latest)                # the dataset is back: the run goes on
    resumed.train()
    assert resumed.step == 3 and [r["step"] for r in read_log(run_dir) if "probe" not in r] == [1, 2, 3]


def test_resume_differences_sorts_what_differs_into_conflicts_and_allowed_changes():
    base = TrainConfig(dataset="d", run_dir="r")
    saved = asdict(base)
    assert resume_differences(saved, saved) == ([], [])
    assert resume_differences(json.loads(json.dumps(saved)), saved) == ([], [])     # a tuple equals its saved list
    assert RESUME_MAY_CHANGE < set(saved)                       # every field that may change is one of the config

    relaunch = TrainConfig(dataset="d", run_dir="r", steps=800, lr_date=1e-5, render_dates=("true",), num_workers=0,
                           loss=LossConfig(class_label=1), sampling=ViewSamplingConfig(n_ref=2), ema_decay=0.0)
    conflicts, changes = resume_differences(saved, asdict(relaunch))
    assert conflicts == ["ema_decay: 0.999 -> 0.0", "loss.class_label: 0 -> 1", "lr_date: 0.0001 -> 1e-05",
                         "sampling.n_ref: 3 -> 2"]
    assert changes == ["num_workers: 4 -> 0", "render_dates: ['true', 'swapped'] -> ['true']", "steps: 10000 -> 800"]
    for name in RESUME_MAY_CHANGE - {"render_rows", "render_dates", "render_seeds", "render_uncond_dates"}:
        number = getattr(base, name) + 1
        assert resume_differences(saved, {**saved, name: number}) == ([], [f"{name}: {number - 1!r} -> {number!r}"])

    # a field that only one of the two has: the checkpoint is of another version of the code
    older = {name: value for name, value in saved.items() if name != "scale"}
    assert resume_differences(older, saved) == (["scale: '(not recorded)' -> 'scene'"], [])
    assert resume_differences(saved, older) == (["scale: 'scene' -> '(not recorded)'"], [])

    # other renders are fine, unless the renders' targets are held out of training: then they change the data
    rendered = dict(sampling=ViewSamplingConfig(n_ref=2), dataset="d", run_dir="r")
    one, other = (TrainConfig(render_rows=(rows,), **rendered) for rows in ((0, 1, 2, 3), (0, 1, 2, 4)))
    assert resume_differences(asdict(one), asdict(other)) == ([], ["render_rows: [[0, 1, 2, 3]] -> [[0, 1, 2, 4]]"])
    held = HoldoutConfig(probe_targets=True)
    one, other = (TrainConfig(render_rows=(rows,), holdout=held, **rendered) for rows in ((0, 1, 2, 3), (0, 1, 2, 4)))
    assert resume_differences(asdict(one), asdict(other)) == (["render_rows: [[0, 1, 2, 3]] -> [[0, 1, 2, 4]]"], [])

    nested = {"a": 1, "b": {"c": (1, (2, 3)), "d": {"e": None}}}
    assert flat_config(nested) == {"a": 1, "b.c": [1, [2, 3]], "b.d.e": None}


def test_resume_refuses_another_configuration_and_names_the_fields(folder, tmp_path, monkeypatch, capsys):
    """The reviewer's case: a gate relaunched with another learning rate would go on at the old one, in silence."""
    settings = dict(steps=2, grad_accum=1, lr_date=1e-3, latest_every=1, pinned_rows=(MODERN_FROM_MODERN,),
                    probe_every=2)
    cfg = config(folder, tmp_path / "run", **settings)
    Trainer(cfg, model=tiny_model()).train()
    run_dir = Path(cfg.run_dir)
    latest = run_dir / "checkpoints" / "latest.pt"
    written = {path.name: path.read_text() for path in run_dir.glob("*.json*")}
    assert set(written) == {"config.json", "log.jsonl", "pinned.json", "probes.json"}

    def relaunch(run_dir=tmp_path / "run", **changes):
        return config(folder, run_dir, **{**settings, **changes})

    monkeypatch.setattr(trainer_module, "load_mvgen", lambda model_dir: pytest.fail("the model was loaded"))
    with pytest.raises(ValueError, match="another configuration") as refused:
        Trainer(relaunch(lr_date=1e-5, steps=8, loss=LossConfig(precision="fp32", class_label=1),
                         pinned_rows=(AFTER_FROM_MODERN,)), resume=latest)      # refused before the model is loaded
    differing = str(refused.value).split("Differing: ")[1].split(". Relaunch")[0].split("; ")
    assert differing == ["loss.class_label: 0 -> 1", "lr_date: 0.001 -> 1e-05",
                         f"pinned_rows: [{list(MODERN_FROM_MODERN)}] -> [{list(AFTER_FROM_MODERN)}]"]
    assert "steps" in str(refused.value).split("Relaunch")[1]                   # ... which names what may change
    for changes, named in ((dict(loss=LossConfig(precision="bf16")), "loss.precision: 'fp32' -> 'bf16'"),
                           (dict(ema_decay=0.5), "ema_decay: 0.0 -> 0.5"), (dict(grad_accum=2), "grad_accum: 1 -> 2"),
                           (dict(scale=0.5), "scale: 'scene' -> 0.5"), (dict(seed=1), "seed: 0 -> 1"),
                           (dict(run_dir=tmp_path / "elsewhere"), "run_dir: "),
                           (dict(finetune=FinetuneConfig(date_freqs=4, date_hidden=16)), "finetune.lora_rank: 2 -> 0")):
        with pytest.raises(ValueError, match=named):
            Trainer(relaunch(**changes), resume=latest)
    with pytest.raises(ValueError, match="weights only"):
        Trainer(cfg, resume=run_dir / "checkpoints" / "step_000002.pt")
    # a refused launch leaves the run folder as it was: config.json still describes the run that is saved there
    assert {path.name: path.read_text() for path in run_dir.glob("*.json*")} == written
    assert not (tmp_path / "elsewhere").exists()
    capsys.readouterr()

    # the number of steps, the intervals and the render settings may change; the learning rate goes on unchanged
    longer = relaunch(steps=4, log_every=2, save_every=3, render_dates=("true",), render_every=4, render_steps=1)
    resumed = Trainer(longer, model=tiny_model(), resume=latest)
    printed = capsys.readouterr().out
    assert "resuming with changed settings: log_every: 1 -> 2; render_dates: ['true', 'swapped'] -> ['true']" in printed
    assert "steps: 2 -> 4" in printed and resumed.step == 2 and "cosine" not in printed
    resumed.train()
    assert json.loads((run_dir / "config.json").read_text())["steps"] == 4
    log = [r for r in read_log(run_dir) if "probe" not in r]
    assert [r["step"] for r in log] == [1, 2, 4] and {r["lr_date"] for r in log} == {1e-3}
    assert (run_dir / "renders" / "step_000004" / "set0.jpg").exists()

    # more steps under a cosine schedule: allowed, with a word on what it does to the learning rate
    cosine = config(folder, tmp_path / "cosine", steps=2, grad_accum=1, lr_schedule="cosine", latest_every=1)
    Trainer(cosine, model=tiny_model()).train()
    capsys.readouterr()
    longer = config(folder, tmp_path / "cosine", steps=6, grad_accum=1, lr_schedule="cosine", latest_every=1)
    Trainer(longer, model=tiny_model(), resume=Path(cosine.run_dir) / "checkpoints" / "latest.pt").train()
    assert "the cosine schedule now runs to the new number of steps" in capsys.readouterr().out
    rates = [r["lr_date"] for r in read_log(cosine.run_dir)]
    assert rates[2] == pytest.approx(0.05 * 1e-2)               # one step at the rate saved with the short schedule
    assert rates[3] > 5 * rates[2] and rates[3:] == sorted(rates[3:], reverse=True)     # then down the longer curve


def test_a_fresh_start_in_a_run_folder_that_holds_a_run_is_refused(folder, tmp_path, monkeypatch):
    cfg = config(folder, tmp_path / "run", steps=1)
    Trainer(cfg, model=tiny_model())                            # set up but never trained: the folder holds no run yet
    Trainer(cfg, model=tiny_model()).train()
    run_dir = Path(cfg.run_dir)
    config_before = (run_dir / "config.json").read_text()
    monkeypatch.setattr(trainer_module, "load_mvgen", lambda model_dir: pytest.fail("the model was loaded"))
    other = config(folder, tmp_path / "run", steps=1, loss=LossConfig(precision="bf16"))
    with pytest.raises(ValueError, match="already holds a run"):
        Trainer(other)
    assert (run_dir / "config.json").read_text() == config_before

    (run_dir / "log.jsonl").unlink()                            # checkpoints alone
    with pytest.raises(ValueError, match="already holds a run"):
        Trainer(other)
    for checkpoint in (run_dir / "checkpoints").iterdir():
        checkpoint.unlink()
    (run_dir / "log.jsonl").write_text("")                      # a log alone: a job that died in its first evaluation
    with pytest.raises(ValueError, match="already holds a run.*latest.pt"):
        Trainer(other)
    (run_dir / "log.jsonl").unlink()
    Trainer(other, model=tiny_model())                          # neither: the folder can be used


def test_checkpoint_to_resume(tmp_path):
    assert checkpoint_to_resume(tmp_path, None) is None
    assert checkpoint_to_resume(tmp_path, "auto") is None       # nothing saved yet: a fresh start
    latest = tmp_path / "checkpoints" / "latest.pt"
    latest.parent.mkdir()
    latest.touch()
    assert checkpoint_to_resume(tmp_path, "auto") == latest
    assert checkpoint_to_resume(tmp_path, "elsewhere/latest.pt") == Path("elsewhere/latest.pt")


# ----------------------------------------------------------------------------- date swapping
def test_swapped_years_cross_the_nearest_period_edge():
    dates = np.array([2021.0, 2015.0, 1890.0, 1975.0, 2019.2877, 2019.28, 1969.99, 1995.0])
    swapped = swapped_years(dates, 0, EDGES, SWAP_YEARS)
    #                           2021 -> modern; 2015 -> after the fire; 1890 -> modern; 1975 -> old (1970 is nearer)
    assert swapped.tolist() == [2015.0, 2021.0, 2015.0, 1890.0, 2015.0, 2021.0, 2015.0, 2021.0]
    assert swapped.dtype == np.float64
    with_refs = swapped_years(dates, 3, EDGES, SWAP_YEARS)
    assert with_refs[:3].tolist() == [2021.0, 2015.0, 1890.0]           # references keep their dates
    assert with_refs[3:].tolist() == swapped[3:].tolist()
    assert dates[0] == 2021.0                                           # the input is not edited

    assert swapped_years(np.array([1960.0, 1980.0]), 0, (1970.0,), (1900.0, 2000.0)).tolist() == [2000.0, 1900.0]
    with pytest.raises(ValueError, match="no period edge|there is none"):
        swapped_years(dates, 0, (), (2000.0,))
    with pytest.raises(ValueError, match="representative years"):
        swapped_years(dates, 0, EDGES, (1890.0, 2015.0))
    nothing_old = (float("nan"), 2015.0, 2021.0)
    assert swapped_years(np.array([2021.0, 2015.0]), 0, EDGES, nothing_old).tolist() == [2015.0, 2021.0]
    with pytest.raises(ValueError, match="period\\(s\\) \\[0\\].*swap_years"):
        swapped_years(np.array([2021.0, 1975.0]), 0, EDGES, nothing_old)
    assert swapped_years(np.array([1975.0, 2021.0]), 1, EDGES, nothing_old).tolist() == [1975.0, 2015.0]


def test_representative_years_are_medians_per_period():
    dates = np.array([1890.0, 1900.0, 1960.0, 2005.0, 2010.0, 2019.0, 2019.2877, 2020.0, 2023.0])
    assert representative_years(dates, EDGES) == (1900.0, 2010.0, 2020.0)
    assert representative_years(dates, (1970.0,)) == (1900.0, 2019.14385)       # the median of an even count
    assert representative_years(dates, ()) == (2010.0,)
    # a pinned run's targets take precedence in the periods they cover
    assert representative_years(dates, EDGES, preferred=np.array([2019.3, 2019.4, 2019.5])) == (1900.0, 2010.0, 2019.4)
    years = representative_years(np.array([2010.0, 2012.0]), EDGES)
    assert np.isnan(years[0]) and years[1] == 2011.0 and np.isnan(years[2])


def test_render_dates_are_parsed():
    assert [parse_render_date(text) for text in ("true", "swapped", "unknown", "2015", "2019.5", 1890)] == \
        ["true", "swapped", "unknown", 2015.0, 2019.5, 1890.0]
    for bad in ("tomorrow", "", None):
        with pytest.raises(ValueError, match="render date"):
            parse_render_date(bad)


# ----------------------------------------------------------------------------- the date-loss probe
def test_date_loss_summary():
    true = np.array([[1.0, 2.0], [4.0, 8.0]])
    losses = {"true": true, "swapped": true * np.array([[1.1, 1.3], [1.0, 1.0]]), "unknown": true * 1.5}
    summary = date_loss_summary(losses, baseline=true * 2, timesteps=(200, 800))
    assert summary["samples"] == 2 and summary["timesteps"] == [200, 800]
    assert summary["loss_true"] == 3.75 and summary["loss_unknown"] == pytest.approx(5.625)
    assert summary["rel_gap_swapped"] == pytest.approx(0.1) and summary["rel_gap_unknown"] == pytest.approx(0.5)
    assert summary["change"] == pytest.approx(-0.5)
    assert summary["sample_rel_gap_swapped"] == pytest.approx([0.2, 0.0])
    assert summary["rel_gap_swapped_by_timestep"] == pytest.approx([0.05, 0.15])
    assert summary["loss_true_by_timestep"] == [2.5, 5.0] and summary["sample_change"] == pytest.approx([-0.5, -0.5])
    assert json.loads(json.dumps(summary)) == summary


def test_date_loss_probe_is_flat_at_zero_init_and_separates_dates_after_training(scene, tmp_path):
    model = adapted_model(trained=False, lora_rank=2)
    probe = DateLossProbe(scene, [AFTER_FROM_MODERN, MODERN_FROM_MODERN], 3, LossConfig(precision="fp32"), SWAP_YEARS,
                          tmp_path / "probes" / "p_baseline.json", timesteps=(200, 800))
    first = probe(model)
    assert first["loss_true"] == first["loss_swapped"] == first["loss_unknown"] > 0     # step 0: the released model
    assert first["rel_gap_swapped"] == first["rel_gap_unknown"] == first["change"] == 0.0
    assert first["sample_change"] == [0.0, 0.0] and first["change_by_timestep"] == [0.0, 0.0]
    assert first["date_rms"] == 0.0 and first["date_temb_ratio"] == 0.0                 # the embedding is still zero
    baseline = json.loads((tmp_path / "probes" / "p_baseline.json").read_text())
    assert baseline["rows"] == [list(AFTER_FROM_MODERN), list(MODERN_FROM_MODERN)]
    assert baseline["timesteps"] == [200, 800]
    assert baseline["loss_true"] == probe.last_losses["true"] and np.shape(baseline["loss_true"]) == (2, 2)
    assert {name: baseline[name] for name in probe.identity()} == probe.identity() == {
        "rows": [list(AFTER_FROM_MODERN), list(MODERN_FROM_MODERN)], "timesteps": [200, 800], "seed": 1234, "n_ref": 3,
        "precision": "fp32", "class_label": 0, "date_views": "all", "cross_date_pixels": "keep", "cross_date_gap": 1.0,
        "period_edges": list(EDGES), "scale_factors": [1.0, 1.0]}

    torch.manual_seed(1)
    nn.init.normal_(model.date.embedding.mlp[-1].weight, std=0.5)       # as after training
    model.unet.seen.clear()
    trained = probe(model)
    assert len({trained["loss_true"], trained["loss_swapped"], trained["loss_unknown"]}) == 3
    assert trained["rel_gap_swapped"] != 0 and trained["rel_gap_unknown"] != 0
    assert trained["change"] == pytest.approx(trained["loss_true"] / first["loss_true"] - 1, abs=0.2)
    assert trained["change"] != 0 and len(trained["sample_change"]) == 2 and len(trained["loss_true_by_timestep"]) == 2
    # the size of the date embedding next to the timestep embedding it is added to, in the true-date passes
    with torch.no_grad():
        sizes = [model.date.embedding(sample["date_mid"], torch.ones(6, dtype=torch.bool)).pow(2).mean().sqrt()
                 for sample in probe.samples]
    assert trained["date_rms"] == pytest.approx(float(torch.stack(sizes).mean()), rel=1e-5)
    assert 0 < trained["date_temb_ratio"] != trained["date_rms"]
    assert probe(model) == trained                                      # same noise, timesteps and views every time
    # gradient-free, and the caller's mode is kept
    assert all(not call["grad"] and not call["training"] for call in model.unet.seen)
    assert len(model.unet.seen) == 2 * (2 * 2 * 3)                      # 2 runs x 2 sets x 2 timesteps x 3 dates
    assert all(p.grad is None for p in model.date.parameters())
    # a later probe on the same baseline (a resumed run) compares with the first evaluation
    again = DateLossProbe(scene, [AFTER_FROM_MODERN, MODERN_FROM_MODERN], 3, LossConfig(precision="fp32"), SWAP_YEARS,
                          tmp_path / "probes" / "p_baseline.json", timesteps=(200, 800))
    assert again(model) == trained
    other = DateLossProbe(scene, [AFTER_FROM_MODERN], 3, LossConfig(precision="fp32"), SWAP_YEARS,
                          tmp_path / "probes" / "p_baseline.json", timesteps=(200, 800))
    with pytest.raises(ValueError, match="other settings \\(rows, scale_factors\\)"):
        other(model)


@pytest.mark.parametrize("changed, named", [
    (dict(loss=LossConfig(precision="bf16")), "precision"),
    (dict(loss=LossConfig(precision="fp32", class_label=1)), "class_label"),
    (dict(loss=LossConfig(precision="fp32", date_views="targets")), "date_views"),
    (dict(loss=LossConfig(precision="fp32", cross_date_pixels="drop_all")), "cross_date_pixels"),
    (dict(loss=LossConfig(precision="fp32", cross_date_gap=2.0)), "cross_date_gap"),
    (dict(n_ref=2), "n_ref"), (dict(scale=0.5), "scale_factors"), (dict(scale="sample"), None),
    (dict(timesteps=(200,)), "timesteps"), (dict(seed=7), "seed"),
    (dict(edges=(1970.0,), swap_years=(1890.0, 2015.0)), "period_edges"),
])
def test_date_loss_baseline_is_tied_to_everything_that_decides_the_loss(folder, tmp_path, changed, named):
    """The reviewer's case: a run restarted in its folder with another precision, domain label, pixel rule or
    scale must not be compared with the first run's step 0."""
    def probe(loss=LossConfig(precision="fp32"), n_ref=3, scale="scene", timesteps=(200, 800), seed=1234, edges=EDGES,
              swap_years=SWAP_YEARS):
        scene = MVGenScene(folder, ViewSamplingConfig(period_edges=edges), resolution=SIZE, scale=scale)
        return DateLossProbe(scene, [AFTER_FROM_MODERN], n_ref, loss, swap_years, tmp_path / "baseline.json",
                             timesteps, seed)

    model = adapted_model()
    first = probe()(model)
    if named is None:           # another rule that gives these cameras the same factor measures the same thing
        assert probe(**changed)(model) == first
    else:
        with pytest.raises(ValueError, match=f"other settings \\({named}\\).*new run folder"):
            probe(**changed)(model)
    assert probe()(model) == first


def test_date_loss_probe_gives_the_model_true_swapped_and_unknown_dates(scene, tmp_path):
    model = adapted_model()
    dates = record_dates(model)
    probe = DateLossProbe(scene, [AFTER_FROM_MODERN], 3, LossConfig(precision="fp32"), SWAP_YEARS,
                          tmp_path / "baseline.json", timesteps=(500,))
    probe(model)
    true = [2012.0, 2012.1, 2012.2, 2020.0, 2020.1, 2020.2]
    (year_true, valid_true), (year_swapped, valid_swapped), (year_unknown, valid_unknown) = dates
    assert year_true.tolist() == pytest.approx(true) and valid_true.all()
    assert year_swapped.tolist() == pytest.approx(true[:3] + [2015.0] * 3) and valid_swapped.all()
    assert year_unknown.tolist() == pytest.approx(true) and not valid_unknown.any()
    assert probe.describe() == [{"rows": list(AFTER_FROM_MODERN), "swapped_years": [2015.0] * 3,
                                 "file_keys": [f"view_{row:02d}.png" for row in AFTER_FROM_MODERN],
                                 "years": pytest.approx(true, abs=1e-3), "scale_factor": 1.0}]
    noisy = [call["sample"] for call in model.unet.seen]
    assert torch.equal(noisy[0], noisy[1]) and torch.equal(noisy[0], noisy[2])          # the same noise for all three

    dates.clear()
    hidden = DateLossProbe(scene, [AFTER_FROM_MODERN], 3, LossConfig(precision="fp32", date_views="targets"),
                           SWAP_YEARS, tmp_path / "hidden.json", timesteps=(500,))
    hidden(model)
    is_target = [False] * 3 + [True] * 3
    assert dates[0][1].tolist() == is_target and dates[1][1].tolist() == is_target and not dates[2][1].any()


def test_date_loss_probe_keeps_the_training_rule_for_cross_date_targets(scene, tmp_path):
    model = adapted_model()
    drop = LossConfig(precision="fp32", cross_date_pixels="drop")
    DateLossProbe(scene, [AFTER_FROM_MODERN, MODERN_FROM_MODERN], 3, drop, SWAP_YEARS, tmp_path / "b.json",
                  timesteps=(500,))(model)
    after_fire, modern = model.unet.seen[:3], model.unet.seen[3:]
    for call in after_fire:             # targets from 2020, references from 2012: no warped pixels in any variant
        assert is_zero(call["coords"][1]).tolist() == [False] * 3 + [True] * 3
        assert not is_zero(call["coords"][0]).any()
    assert all(not is_zero(call["coords"][1]).any() for call in modern)

    # "drop_all": both priors go, as under the authors' per-view prior dropout
    model.unet.seen.clear()
    drop_all = LossConfig(precision="fp32", cross_date_pixels="drop_all")
    DateLossProbe(scene, [AFTER_FROM_MODERN, MODERN_FROM_MODERN], 3, drop_all, SWAP_YEARS, tmp_path / "all.json",
                  timesteps=(500,))(model)
    for call in model.unet.seen[:3]:
        assert is_zero(call["coords"][0]).tolist() == is_zero(call["coords"][1]).tolist() == [False] * 3 + [True] * 3
    assert all(not is_zero(prior).any() for call in model.unet.seen[3:] for prior in call["coords"])

    # the scene's periods count, not the gap alone: within 20 years, but the fire lies between them
    model.unet.seen.clear()
    wide = LossConfig(precision="fp32", cross_date_pixels="drop", cross_date_gap=20.0)
    DateLossProbe(scene, [AFTER_FROM_MODERN], 3, wide, SWAP_YEARS, tmp_path / "wide.json", timesteps=(500,))(model)
    assert is_zero(model.unet.seen[0]["coords"][1]).tolist() == [False] * 3 + [True] * 3
    model.unet.seen.clear()
    one_period = MVGenScene(scene.dataset_dir, ViewSamplingConfig(period_edges=(1970.0,)), resolution=SIZE)
    DateLossProbe(one_period, [AFTER_FROM_MODERN], 3, wide, (1890.0, 2015.0), tmp_path / "one.json",
                  timesteps=(500,))(model)
    assert not is_zero(model.unet.seen[0]["coords"][1]).any()

    with pytest.raises(ValueError, match="outside the scheduler"):
        DateLossProbe(scene, [AFTER_FROM_MODERN], 3, drop, SWAP_YEARS, tmp_path / "c.json", timesteps=(1000,))(model)
    with pytest.raises(ValueError, match="at least one view set"):
        DateLossProbe(scene, [], 3, drop, SWAP_YEARS, tmp_path / "d.json")
    with pytest.raises(ValueError, match="swap_years"):                 # no year stands for "after the fire"
        DateLossProbe(scene, [MODERN_FROM_MODERN], 3, drop, (1890.0, 2015.0, float("nan")), tmp_path / "e.json")


# ----------------------------------------------------------------------------- the render probe
def test_render_probe_renders_every_date_and_seed(scene, tmp_path):
    model = adapted_model()
    dates = record_dates(model)
    probe = RenderProbe(scene, AFTER_FROM_MODERN, 3, "fire", dates=("true", "swapped", "1890", "unknown", "true"),
                        seeds=(0, 1), loss_cfg=LossConfig(precision="fp32", class_label=1), swap_years=SWAP_YEARS,
                        steps=2, guidance=2.0, uncond_dates="unknown")
    depth_before = probe.sample["depth"].clone()
    result = probe(model, tmp_path / "renders", thumb=16)

    assert probe.dates == ("true", "swapped", 1890.0, "unknown")        # the repeated date is rendered once
    records = result["renders"]
    assert [(r["date"], r["seed"]) for r in records] == [(d, s) for d in ("true", "swapped", "1890", "unknown")
                                                         for s in (0, 1)]
    assert all(np.isfinite(r["psnr"]) and len(r["psnr_per_target"]) == 3 for r in records)
    assert all(r["psnr"] == pytest.approx(np.mean(r["psnr_per_target"])) for r in records)
    assert [r["diff_vs_first"] == 0 for r in records] == [True, True] + [False] * 6     # the date changes the picture
    assert result["psnr_by_date"]["swapped"] == pytest.approx(np.mean([r["psnr"] for r in records[2:4]]))
    assert list(result["psnr_by_date"]) == ["true", "swapped", "1890", "unknown"]
    assert result["rows"] == list(AFTER_FROM_MODERN) and result["swapped_years"] == [2015.0] * 3
    assert (result["scale"], result["scale_factor"]) == ("scene", 1.0)
    assert json.loads((tmp_path / "renders" / "fire.json").read_text()) == result
    sheet = Image.open(tmp_path / "renders" / "fire.jpg")
    assert sheet.size == (230 + 6 * 16, 18 + 14 + (1 + 8) * 16)         # label column, header; real row + 8 renders

    pipeline = model.pipelines[-1]
    assert len(model.pipelines) == 1 and len(pipeline.calls) == 8 and pipeline.progress_bar == {"disable": True}
    for call in pipeline.calls:
        assert torch.equal(call["images"][:3], probe.sample["images"][:3])          # the references are given
        assert call["images"][3:].abs().max() == 0                                  # the real targets are not
        assert (call["nframe"], call["cond_num"], call["steps"], call["guidance"]) == (6, 3, 2, 2.0)
        assert (call["class_label"], call["config_nframe"], call["output_type"]) == (1, 6, "np")
        assert call["tag"] == ["custom"] * 6 and call["vae"] is model.vae
        assert torch.equal(call["depth"], depth_before.double()) and torch.equal(call["K"], probe.sample["K"])
        assert call["depth"].dtype == torch.float64         # so that every prior contracts far depth in its own copy
    assert torch.equal(probe.sample["depth"], depth_before)             # the pipeline edited a copy
    assert "nframe" not in model.config
    assert all(not call["grad"] for call in model.unet.seen) and len(model.unet.seen) == 8 * 2

    # the dates the model saw, per render and sampler step: [unconditional half, conditional half]
    true = [2012.0, 2012.1, 2012.2, 2020.0, 2020.1, 2020.2]
    expected = {"true": true, "swapped": true[:3] + [2015.0] * 3, "1890": true[:3] + [1890.0] * 3, "unknown": true}
    assert len(dates) == 8 * 2
    for record, (year, valid) in zip(records, dates[::2]):
        assert year.tolist() == pytest.approx(expected[record["date"]] * 2)
        known = record["date"] != "unknown"
        assert valid.tolist() == [False] * 6 + [known] * 6              # uncond_dates="unknown"


def test_render_probe_options(scene, tmp_path):
    model = adapted_model(trained=False)                                # zero-initialised: the date cannot matter
    probe = RenderProbe(scene, AFTER_FROM_MODERN, 3, "plain", dates=("true", 1890.0), seeds=(0, 1),
                        loss_cfg=LossConfig(precision="fp32"), steps=2)
    result = probe(model, tmp_path)
    assert [r["diff_vs_first"] for r in result["renders"]] == [0.0] * 4
    assert result["renders"][0]["psnr"] != result["renders"][1]["psnr"]             # another seed, another picture
    assert result["renders"][0]["psnr"] == result["renders"][2]["psnr"]
    assert "swapped_years" not in result and probe.swapped is None
    with pytest.raises(ValueError, match="swap_years"):
        RenderProbe(scene, AFTER_FROM_MODERN, 3, "x", dates=("swapped",))
    with pytest.raises(ValueError, match="render date"):
        RenderProbe(scene, AFTER_FROM_MODERN, 3, "x", dates=("soon",))
    with pytest.raises(ValueError, match="at least one date and one seed"):
        RenderProbe(scene, AFTER_FROM_MODERN, 3, "x", seeds=())
    with pytest.raises(ValueError, match="view set 0"):
        RenderProbe(scene, AFTER_FROM_MODERN[:3], 3, "x")

    model = adapted_model()
    dates = record_dates(model)
    hidden = RenderProbe(scene, AFTER_FROM_MODERN, 3, "hidden", dates=("true", "unknown"), seeds=(0,), steps=1,
                         loss_cfg=LossConfig(precision="fp32", date_views="targets"))
    hidden(model, tmp_path)
    assert [valid.tolist() for _, valid in dates] == [([False] * 3 + [True] * 3) * 2, [False] * 12]   # "same" halves
    year, valid = hidden.view_dates(2015.0)
    assert year.tolist() == pytest.approx([2012.0, 2012.1, 2012.2, 2015.0, 2015.0, 2015.0])
    assert hidden.sample["date_mid"][3:].tolist() == pytest.approx([2020.0, 2020.1, 2020.2])     # not edited


def test_render_probe_drops_warped_pixels_by_the_date_it_renders_at(scene, tmp_path):
    model = adapted_model()
    drop = LossConfig(precision="fp32", cross_date_pixels="drop")
    probe = RenderProbe(scene, AFTER_FROM_MODERN, 3, "b", dates=("true", 2012.5, "swapped", "unknown"), seeds=(0,),
                        loss_cfg=drop, swap_years=(1890.0, 2012.9, 2021.0), steps=1)
    probe(model, tmp_path)
    true, near, swapped, unknown = [call["coords"] for call in model.unet.seen]
    dropped = [False] * 6 + [False] * 3 + [True] * 3        # [unconditional half (zeros already), conditional half]
    kept = [False] * 12
    for coordinates, pixels in (true, near, swapped, unknown):
        assert is_zero(pixels)[:6].all() and torch.all(coordinates[6:] == 3.0)       # coordinates are never dropped
    assert (~is_zero(true[1]))[6:].tolist() == [True] * 3 + [False] * 3              # 2020 vs references of 2012
    assert (is_zero(true[1]) & torch.tensor([False] * 6 + [True] * 6)).tolist() == dropped
    assert (is_zero(near[1]) & torch.tensor([False] * 6 + [True] * 6)).tolist() == kept      # rendered at 2012.5
    assert (is_zero(swapped[1]) & torch.tensor([False] * 6 + [True] * 6)).tolist() == kept   # 2012.9: within a year
    assert (is_zero(unknown[1]) & torch.tensor([False] * 6 + [True] * 6)).tolist() == dropped   # by the true dates
    assert len(model.unet._forward_pre_hooks) == 1          # only the date conditioner's hook is left

    keep = RenderProbe(scene, AFTER_FROM_MODERN, 3, "a", dates=("true",), seeds=(0,), steps=1,
                       loss_cfg=LossConfig(precision="fp32"))
    model.unet.seen.clear()
    keep(model, tmp_path)
    assert not is_zero(model.unet.seen[0]["coords"][1])[6:].any()


def test_render_probe_applies_drop_all_and_the_scenes_periods_as_training_does(scene, tmp_path):
    """What the sampler's UNet receives for a cross-date target is what ``build_inputs`` gives a training step."""
    model = adapted_model()
    targets = [False] * 3 + [True] * 3
    for setting, gap in (("drop_all", 1.0), ("drop_all", 20.0), ("drop", 20.0)):
        cfg = LossConfig(precision="fp32", cross_date_pixels=setting, cross_date_gap=gap)
        # dates: the targets' own (after the fire); 2015 (the references' period); 2019.5 (after the fire again,
        # within 20 years of the references of 2012 but not in their period)
        probe = RenderProbe(scene, AFTER_FROM_MODERN, 3, f"{setting}{gap:g}", dates=("true", 2015.0, 2019.5),
                            seeds=(0,), loss_cfg=cfg, steps=1)
        model.unet.seen.clear()
        probe(model, tmp_path)
        expected = {1.0: [targets, targets, targets], 20.0: [targets, [False] * 6, targets]}[gap]   # 2015 - 2012 > 1
        latents = encode_images(model, probe.sample["images"], sample_posterior=False)
        for date, call, cross_date in zip(probe.dates, model.unet.seen, expected, strict=True):
            coordinates, pixels = (prior[6:] for prior in call["coords"])                # the conditional half
            assert is_zero(pixels).tolist() == cross_date
            assert is_zero(coordinates).tolist() == (cross_date if setting == "drop_all" else [False] * 6)
            prior_views, pixel_views = step.cross_date_drops(probe.view_dates(date)[0], 3, cfg, EDGES)
            trained_on = build_inputs(model, probe.sample, latents, dropped_prior_views=prior_views,
                                      dropped_pixel_views=pixel_views)                   # a training step's inputs
            assert torch.equal(coordinates, trained_on.coords[0]) and torch.equal(pixels, trained_on.coords[1])
    assert len(model.unet._forward_pre_hooks) == 1


def test_view_priors_dropped_zeroes_what_build_inputs_would():
    unet = TinyUNet()
    views = torch.tensor([True, True, False])               # view 0 is a reference: never dropped
    sample, add_inputs = torch.zeros(1, 3, 4, LATENT, LATENT), torch.zeros(1, 3, 7, SIZE, SIZE)
    coordinates, pixels = torch.ones(3, 193, LATENT, LATENT), torch.ones(3, 4, SIZE, SIZE)

    def seen_zero(coords, batch=1):
        unet(sample.repeat(batch, 1, 1, 1, 1), torch.tensor([5]), add_inputs=add_inputs.repeat(batch, 1, 1, 1, 1),
             coords=coords)
        return [is_zero(prior).tolist() for prior in unet.seen[-1]["coords"]]

    with view_priors_dropped(unet, 1, pixel_views=views):
        assert seen_zero([coordinates, pixels]) == [[False] * 3, [False, True, False]]
        assert torch.all(pixels == 1) and torch.all(coordinates == 1)                       # the caller's are intact
        with pytest.raises(ValueError, match="cannot fill"):
            unet(sample, torch.tensor([5]), add_inputs=add_inputs, coords=[coordinates, pixels[:2]])
        with pytest.raises(ValueError, match="no warped-pixel prior"):
            unet(sample, torch.tensor([5]), add_inputs=add_inputs, coords=coordinates)
    with view_priors_dropped(unet, 1, prior_views=views):
        assert seen_zero([coordinates, pixels]) == [[False, True, False]] * 2
        doubled = [torch.cat([prior, prior]) for prior in (coordinates, pixels)]            # a guidance batch
        assert seen_zero(doubled, batch=2) == [[False, True, False] * 2] * 2
        with pytest.raises(ValueError, match="not given any 3D priors"):
            unet(sample, torch.tensor([5]), add_inputs=add_inputs, coords=None)
    with view_priors_dropped(unet, 2, prior_views=views, pixel_views=~views):               # two references
        assert seen_zero([coordinates, pixels]) == [[False] * 3, [False, False, True]]
    with view_priors_dropped(unet, 1):                      # nothing to drop: no hook at all
        assert len(unet._forward_pre_hooks) == 0
    assert seen_zero([coordinates, pixels]) == [[False] * 3] * 2 and len(unet._forward_pre_hooks) == 0


def test_unet_precision_puts_only_the_unets_forward_pass_under_autocast():
    unet = TinyUNet()
    inside = []
    unet.conv_in.register_forward_hook(
        lambda module, args, output: inside.append((torch.is_autocast_enabled("cpu"), output.dtype)))
    sample, add_inputs = torch.zeros(1, 2, 4, LATENT, LATENT), torch.zeros(1, 2, 7, SIZE, SIZE)
    coords = [torch.ones(2, 193, LATENT, LATENT), torch.ones(2, 4, SIZE, SIZE)]

    def run():
        return unet(sample, torch.tensor([5]), add_inputs=add_inputs, coords=coords, return_dict=False)

    with torch.no_grad(), unet_precision(unet, "bf16"):
        assert not torch.is_autocast_enabled("cpu")                     # not around the caller's own code
        out = run()
        assert not torch.is_autocast_enabled("cpu")
        assert inside == [(True, torch.bfloat16)] and out[0].dtype == torch.float32
        with pytest.raises(TypeError):                                  # a failed pass closes its autocast context
            unet(sample, torch.tensor([5]), add_inputs=add_inputs, coords=None)
        assert not torch.is_autocast_enabled("cpu")
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16), unet_precision(unet, "fp32"):
        run()                                                           # fp32 also switches an outer autocast off
        assert torch.is_autocast_enabled("cpu")
    assert inside[-1] == (False, torch.float32)
    with torch.no_grad():
        run()
    assert inside[-1] == (False, torch.float32) and len(unet._forward_pre_hooks) == len(unet._forward_hooks) == 0
    assert len(inside) == 3                                             # the failed pass stopped before conv_in


def test_psnr():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    assert psnr(a, a) == pytest.approx(100.0)
    assert psnr(a, a + 255) == pytest.approx(0.0)
    assert psnr(a, a + 26) == pytest.approx(-20 * np.log10(26 / 255))


# ----------------------------------------------------------------------------- probes inside a run
def test_probes_run_at_their_intervals_and_do_not_disturb_training(folder, tmp_path):
    sets = (AFTER_FROM_MODERN, MODERN_FROM_MODERN)
    settings = dict(steps=4, grad_accum=1, ema_decay=0.9, pinned_rows=sets)
    plain = Trainer(config(folder, tmp_path / "plain", **settings), model=tiny_model())
    plain.train()
    cfg = config(folder, tmp_path / "probed", probe_every=2, render_every=3,
                 render_dates=("true", "swapped", "unknown"), **settings)
    probed = Trainer(cfg, model=tiny_model())
    probed.train()

    assert torch.equal(flat_weights(probed.model), flat_weights(plain.model))           # training is unchanged
    assert all(torch.equal(a, b) for a, b in zip(probed.ema.shadow, plain.ema.shadow))
    log = read_log(cfg.run_dir)
    training = [r for r in log if "probe" not in r]
    assert [r["loss"] for r in training] == [r["loss"] for r in read_log(tmp_path / "plain")]
    assert [r["step"] for r in log if r.get("probe") == "date_loss:pinned"] == [0, 2, 4]
    assert [(r["step"], r["probe"]) for r in log if r.get("probe", "").startswith("render")] == \
        [(s, f"render:set{i}") for s in (0, 3, 4) for i in (0, 1)]
    first, last = [r for r in log if r.get("probe") == "date_loss:pinned"][::2]
    assert first["change"] == 0.0 and first["rel_gap_swapped"] == 0.0 and first["loss_true"] == first["loss_unknown"]
    assert last["change"] != 0.0 and last["samples"] == 2 and last["timesteps"] == [200, 800]

    run_dir = Path(cfg.run_dir)
    assert sorted(p.name for p in (run_dir / "renders").iterdir()) == ["step_000000", "step_000003", "step_000004"]
    assert sorted(p.name for p in (run_dir / "renders" / "step_000003").iterdir()) == \
        ["set0.jpg", "set0.json", "set1.jpg", "set1.json"]
    raw = [json.loads(line) for line in (run_dir / "probes" / "pinned_losses.jsonl").read_text().splitlines()]
    assert [r["step"] for r in raw] == [0, 2, 4] and np.shape(raw[0]["true"]) == (2, 2)
    assert json.loads((run_dir / "probes" / "pinned_baseline.json").read_text())["loss_true"] == raw[0]["true"]
    step0 = json.loads((run_dir / "renders" / "step_000000" / "set0.json").read_text())
    assert all(r["diff_vs_first"] == 0 for r in step0["renders"])                       # step 0: no date effect yet
    # every probe ran in eval() mode and without gradients; every training pass in train() mode with them
    seen = probed.model.unet.seen
    assert {(call["training"], call["grad"]) for call in seen} == {(True, True), (False, False)}
    assert sum(call["grad"] for call in seen) == 4


def test_evaluation_restores_mode_weights_and_random_state(folder, tmp_path):
    trainer = Trainer(config(folder, tmp_path / "run", ema_decay=0.5), model=tiny_model())
    unet, weight = trainer.model.unet, trainer.params[0]
    with torch.no_grad():
        weight.add_(1.0)
    raw = weight.detach().clone()
    unet.train()
    torch.manual_seed(5)
    expected = torch.rand(4)
    torch.manual_seed(5)
    with trainer.evaluation():
        assert not unet.training and not torch.is_grad_enabled()
        assert torch.equal(weight.detach(), trainer.ema.shadow[0]) and not torch.equal(weight.detach(), raw)
        torch.rand(100)                                         # a probe drawing from the global random state
    assert unet.training and torch.is_grad_enabled() and torch.equal(weight.detach(), raw)
    assert torch.equal(torch.rand(4), expected)
    with pytest.raises(RuntimeError, match="render failed"), trainer.evaluation():
        raise RuntimeError("render failed")
    assert unet.training and torch.equal(weight.detach(), raw)
    unet.eval()
    with trainer.evaluation():
        pass
    assert not unet.training


# ----------------------------------------------------------------------------- sampling the scene, hold-out
def test_scene_and_heldout_view_sets(scene):
    drawn = scene_view_sets(scene, 4)
    assert len(drawn) == 4 and all(len(rows) == 8 for rows in drawn)
    other_seed = MVGenScene(scene.dataset_dir, resolution=SIZE, seed=7)     # the training seed does not matter
    assert all(np.array_equal(a, b) for a, b in zip(drawn, scene_view_sets(other_seed, 4)))
    assert not any(np.array_equal(drawn[0], scene.indices(i)) for i in range(50))       # not a training draw
    assert heldout_view_sets(scene, 4) == []                    # nothing is held out

    scene.hold_out([0, 1, 8, 9])
    assert heldout_view_sets(scene, 4) == []                    # two held-out views of a period: no group of 5 targets
    small = MVGenScene(scene.dataset_dir, ViewSamplingConfig(n_ref=2, n_targets=3, min_targets_seen=2, top_k=4),
                       resolution=SIZE)
    small.hold_out(np.arange(0, N, 2))                          # every other view: 4 per period
    held = heldout_view_sets(small, 5)
    assert len(held) == 5 and all(np.array_equal(a, b) for a, b in zip(held, heldout_view_sets(small, 5)))
    for rows in held:
        assert small.heldout[rows[2:]].all() and not small.heldout[rows[:2]].any()
        assert len(set(small.period[rows[2:]])) == 1


def test_heldout_views_never_train_and_get_their_own_probe(folder, tmp_path, monkeypatch):
    sampling = ViewSamplingConfig(n_ref=2, n_targets=3, min_targets_seen=2, top_k=4)
    cfg = config(folder, tmp_path / "run", steps=4, sampling=sampling, probe_every=4, render_every=4,
                 render_rows=((8, 9, 16, 17),), render_dates=("true",), render_seeds=(0,),
                 holdout=HoldoutConfig(fraction=0.5, probe_targets=True, seed=1, era_edges=EDGES))
    seen = watch_training(monkeypatch)
    trainer = Trainer(cfg, model=tiny_model())
    heldout = np.flatnonzero(trainer.scene.heldout)
    assert len(heldout) in (12, 13, 14) and {16, 17} <= set(heldout.tolist())           # 4 per period + render targets
    assert [np.sum(trainer.scene.period[heldout] == p) >= 4 for p in range(3)] == [True] * 3
    record = json.loads((tmp_path / "run" / "heldout.json").read_text())
    assert [r["row"] for r in record] == heldout.tolist() and record[0]["file_key"] == f"view_{heldout[0]:02d}.png"

    assert list(trainer.loss_probes) == ["train", "heldout"]
    assert not np.isin(np.array(trainer.loss_probes["train"].rows), heldout).any()
    for rows in trainer.loss_probes["heldout"].rows:
        assert np.isin(rows[2:], heldout).all() and not np.isin(rows[:2], heldout).any()
    assert len(trainer.loss_probes["train"].rows) == len(trainer.loss_probes["heldout"].rows) == 2
    # swapped dates: the medians of the views the run trains on
    training_dates = trainer.scene.dates[trainer.scene.target_ok]
    assert trainer.swap_years == pytest.approx(representative_years(training_dates, EDGES))

    trainer.train()
    assert len(seen) == 8 and not np.isin(np.array(seen), heldout).any()
    probes = {r["probe"] for r in read_log(cfg.run_dir) if "probe" in r}
    assert probes == {"date_loss:train", "date_loss:heldout", "render:set0"}
    assert (tmp_path / "run" / "probes" / "heldout_baseline.json").exists()


def test_no_heldout_probe_when_heldout_views_cannot_form_a_view_set(folder, tmp_path, capsys):
    cfg = config(folder, tmp_path / "run", probe_every=1, holdout=HoldoutConfig(fraction=0.25, era_edges=EDGES))
    trainer = Trainer(cfg, model=tiny_model())
    assert trainer.scene.heldout.sum() == 6 and list(trainer.loss_probes) == ["train"]
    assert "no date-loss probe on held-out views" in capsys.readouterr().out


def test_explicit_swap_years_are_used(folder, tmp_path):
    trainer = Trainer(config(folder, tmp_path / "run", swap_years=SWAP_YEARS, probe_every=1,
                             pinned_rows=(AFTER_FROM_MODERN,)), model=tiny_model())
    assert trainer.swap_years == SWAP_YEARS
    assert trainer.loss_probes["pinned"].swapped[0][3:].tolist() == [2015.0] * 3
    assert torch.load(trainer.save(), weights_only=False)["swap_years"] == list(SWAP_YEARS)


# ----------------------------------------------------------------------------- command lines
def test_train_script_defaults_are_the_configs_defaults(tmp_path):
    script = load_script("train")
    cfg = script.build_config(script.parse_args(["--run-name", "nd"]))
    expected = TrainConfig(dataset=str(REPO / "data/datasets/25512_model0_v2"),
                           run_dir=str(REPO / "data/experiments/mvgen/train/nd"),
                           holdout=HoldoutConfig(era_edges=EDGES))
    assert cfg == expected
    assert Path(cfg.dataset).is_absolute() and Path(cfg.run_dir).is_absolute()     # the job runs in the checkout


def test_train_script_builds_the_config_from_its_options(folder, tmp_path):
    script = load_script("train")
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps([{"refs": ["view_08.png", "view_09.png"], "targets": ["view_16.png", "view_17.png"]},
                                {"refs": ["view_08.png", "view_09.png"], "targets": ["view_10.png", "view_11.png"]}]))
    rows = tmp_path / "rows.json"
    rows.write_text("[[0, 1, 2, 3]]")
    cfg = script.build_config(script.parse_args([
        "--run-name", "gate", "--runs-root", str(tmp_path), "--dataset", str(folder), "--pinned-keys-file", str(keys),
        "--render-rows-file", str(rows), "--n-ref", "2", "--n-targets", "2", "--min-targets-seen", "1",
        "--steps", "400", "--grad-accum", "1", "--lora-rank", "32", "--lora-alpha", "16", "--ema-decay", "0",
        "--grad-clip-factor", "0", "--precision", "fp32", "--no-grad-checkpointing", "--no-date-jitter",
        "--date-views", "targets", "--class-label", "1", "--cross-date-pixels", "drop_all", "--cross-date-gap", "2.5",
        "--scale", "0.5",
        "--date-dropout", "0", "--period-edges", "1970", "--swap-years", "1900", "2000", "--render-dates", "true",
        "swapped", "2015.5", "--render-seeds", "3", "--render-uncond-dates", "unknown", "--probe-timesteps", "10", "20",
        "--probe-every", "50", "--render-every", "200", "--lr-schedule", "constant", "--warmup", "0",
        "--anchor-period-weights", "1", "3", "--lora-targets", "attn1.to_q"]))
    assert cfg.pinned_rows == ((8, 9, 16, 17), (8, 9, 10, 11)) and cfg.render_rows == ((0, 1, 2, 3),)
    assert cfg.run_dir == str(tmp_path / "gate") and cfg.dataset == str(folder)
    assert (cfg.steps, cfg.grad_accum, cfg.ema_decay, cfg.grad_clip_factor) == (400, 1, 0.0, None)
    assert cfg.finetune == FinetuneConfig(lora_rank=32, lora_alpha=16.0, lora_targets=("attn1.to_q",),
                                          grad_checkpointing=False)
    assert cfg.loss == LossConfig(date_dropout=0.0, date_jitter=False, date_views="targets", class_label=1,
                                  cross_date_pixels="drop_all", cross_date_gap=2.5, precision="fp32")
    assert cfg.scale == 0.5 and isinstance(cfg.scale, float)
    for text, scale in (("scene", "scene"), ("sample", "sample"), ("0.0868", 0.0868)):
        assert script.build_config(script.parse_args(["--run-name", "s", "--scale", text])).scale == scale
    with pytest.raises(SystemExit):
        script.parse_args(["--run-name", "s", "--scale", "per-scene"])
    assert cfg.sampling == ViewSamplingConfig(n_ref=2, n_targets=2, min_targets_seen=1, period_edges=(1970.0,),
                                              anchor_period_weights=(1.0, 3.0))
    assert cfg.swap_years == (1900.0, 2000.0) and cfg.render_dates == ("true", "swapped", 2015.5)
    assert (cfg.render_seeds, cfg.render_uncond_dates, cfg.probe_timesteps) == ((3,), "unknown", (10, 20))
    assert (cfg.probe_every, cfg.render_every, cfg.lr_schedule, cfg.warmup_steps) == (50, 200, "constant", 0)
    assert cfg.holdout.era_edges == (1970.0,) and not cfg.holdout.active
    date_only = script.build_config(script.parse_args(["--run-name", "d", "--no-date", "--lora-rank", "4",
                                                       "--holdout-fraction", "0.1", "--holdout-probe-targets"]))
    assert not date_only.finetune.date and date_only.holdout == HoldoutConfig(fraction=0.1, probe_targets=True,
                                                                              era_edges=EDGES)
    with pytest.raises(SystemExit):                             # rows and keys together
        script.parse_args(["--run-name", "x", "--pinned-rows-file", str(rows), "--pinned-keys-file", str(keys)])


def test_train_and_render_scripts_run_end_to_end(folder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(trainer_module, "load_mvgen", lambda model_dir: tiny_model())
    train, render = load_script("train"), load_script("render")
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps([{"refs": ["view_08.png", "view_09.png", "view_10.png"],
                                 "targets": ["view_16.png", "view_17.png", "view_18.png"]},
                                {"refs": ["view_08.png", "view_09.png", "view_10.png"],
                                 "targets": ["view_11.png", "view_12.png", "view_13.png"]}]))
    common = ["--run-name", "gate", "--runs-root", str(tmp_path), "--dataset", str(folder), "--resolution", str(SIZE),
              "--pinned-keys-file", str(keys), "--grad-accum", "2", "--lora-rank", "2", "--date-freqs", "4",
              "--date-hidden", "16", "--precision", "fp32", "--ema-decay", "0", "--num-workers", "0",
              "--log-every", "1", "--probe-every", "2", "--probe-timesteps", "200", "800", "--render-every", "2",
              "--render-steps", "2", "--latest-every", "1", "--class-label", "1", "--date-views", "targets",
              "--resume", "auto"]
    train.main([*common, "--steps", "2"])
    run_dir = tmp_path / "gate"
    assert [r["step"] for r in read_log(run_dir) if "probe" not in r] == [1, 2]
    assert json.loads((run_dir / "pinned.json").read_text())[0]["rows"] == list(AFTER_FROM_MODERN)
    train.main([*common, "--steps", "3"])                       # the same command again, one step further
    log = read_log(run_dir)
    assert [r["step"] for r in log if "probe" not in r] == [1, 2, 3]
    assert [r["step"] for r in log if r.get("probe") == "date_loss:pinned"] == [0, 2, 3]
    assert "resumed from" in capsys.readouterr().out
    assert (run_dir / "renders" / "step_000003" / "set1.jpg").exists()
    # the same command with another learning rate, or without --resume: refused, and config.json keeps the run's
    with pytest.raises(ValueError, match="lr_date: 0.0001 -> 1e-05"):
        train.main([*common, "--steps", "4", "--lr-date", "1e-5"])
    with pytest.raises(ValueError, match="already holds a run"):
        train.main([*common[:-2], "--steps", "4"])
    saved_config = json.loads((run_dir / "config.json").read_text())
    assert (saved_config["lr_date"], saved_config["steps"], saved_config["scale"]) == (1e-4, 3, "scene")

    # render.py: a checkpoint brings its run's settings; the released model is the dateless baseline
    monkeypatch.setattr(mvgen_model, "load_mvgen", lambda model_dir, device: tiny_model())
    models = []
    real_load = render.load_finetuned
    monkeypatch.setattr(render, "load_finetuned",
                        lambda *args: models.append(real_load(*args, device="cpu")) or models[-1])
    monkeypatch.setattr(render, "load_mvgen", lambda model_dir: models.append((tiny_model(), {})) or models[-1][0])
    out = tmp_path / "rendered"
    render.main(["--checkpoint", str(run_dir / "checkpoints" / "step_000003.pt"), "--keys-file", str(keys),
                 "--out", str(out), "--dates", "true", "swapped", "2015", "--seeds", "4", "--steps", "2",
                 "--thumb", "16"])
    results = json.loads((out / "results.json").read_text())
    assert [r["name"] for r in results] == ["set0", "set1"] and results[0]["rows"] == list(AFTER_FROM_MODERN)
    assert [r["date"] for r in results[0]["renders"]] == ["true", "swapped", "2015"]
    assert results[0]["swapped_years"] == pytest.approx([2012.4] * 3)    # the run's swap years, from the checkpoint
    assert (results[0]["scale"], results[0]["scale_factor"]) == ("scene", 1.0)      # and its scene-scale rule
    assert (out / "set0.jpg").exists() and (out / "set1.json").exists()
    model, ckpt = models[-1]
    assert ckpt["step"] == 3 and not model.unet.training
    call = model.pipelines[0].calls[0]
    assert (call["class_label"], call["steps"], call["cond_num"]) == (1, 2, 3)          # the run's domain label
    saved = ckpt["trainable"]
    torch.testing.assert_close(flat_weights(model), torch.cat(
        [saved[part][key].flatten() for part in ("date", "lora") for key in sorted(saved[part])]))

    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([list(AFTER_FROM_MODERN)]))
    baseline = tmp_path / "baseline"
    render.main(["--rows-file", str(rows), "--dataset", str(folder), "--resolution", str(SIZE), "--out", str(baseline),
                 "--dates", "true", "swapped", "--seeds", "0", "--steps", "2", "--precision", "fp32", "--thumb", "16"])
    results = json.loads((baseline / "results.json").read_text())
    assert [r["diff_vs_first"] for r in results[0]["renders"]] == [0.0, 0.0]            # no date input: no effect
    assert results[0]["swapped_years"] == pytest.approx([2012.35] * 3)   # the median modern view of the folder
    assert models[-1][0].date is None and models[-1][0].pipelines[0].calls[0]["class_label"] == 0
    assert (results[0]["scale"], results[0]["scale_factor"]) == ("scene", 1.0)      # the released model's default
    whole_scale = models[-1][0].pipelines[0].calls[0]

    # --scale: another rule than the checkpoint's is used, recorded and pointed out
    capsys.readouterr()
    render.main(["--checkpoint", str(run_dir / "checkpoints" / "step_000003.pt"), "--keys-file", str(keys),
                 "--out", str(tmp_path / "halved"), "--seeds", "0", "--steps", "1", "--thumb", "16", "--scale", "0.5"])
    results = json.loads((tmp_path / "halved" / "results.json").read_text())
    assert (results[0]["scale"], results[0]["scale_factor"]) == (0.5, 0.5)
    halved = models[-1][0].pipelines[0].calls[0]
    torch.testing.assert_close(halved["w2c"][:, :3, 3], whole_scale["w2c"][:, :3, 3] * 0.5)
    torch.testing.assert_close(halved["depth"], whole_scale["depth"] * 0.5)
    printed = capsys.readouterr().out
    assert "scene scale 0.5 (factor 0.5)" in printed and "trained with scene scale 'scene' (factor 1.0)" in printed


def test_check_step_records_or_compares(scene, monkeypatch, capsys):
    script = load_script("check_step")
    sample = scene.make_sample(np.array(MODERN_FROM_MODERN), 3)
    monkeypatch.setattr(script, "load_mvgen", lambda model_dir, device: tiny_model())
    monkeypatch.setattr(script, "exported_sample", lambda folder, longest_side, device: sample)
    script.main(["--record"])                                   # prints the table to paste, compares nothing
    printed = capsys.readouterr().out
    recorded = ast.literal_eval(printed.split("EXPECTED = ")[1].splitlines()[0])
    assert list(recorded) == list(script.EXPECTED) and all(0 < loss < 100 for loss in recorded.values())
    assert "RECORDED, not compared" in printed and "MISMATCH" not in printed and "PASSED" not in printed
    with pytest.raises(SystemExit, match="FAILED: the loss differs by more than 1% at timesteps"):
        script.main([])                                         # the stand-in is not the released model
    assert capsys.readouterr().out.count("MISMATCH") == len(script.EXPECTED)
    monkeypatch.setattr(script, "EXPECTED", recorded)
    script.main([])
    assert "PASSED: all 5 losses" in capsys.readouterr().out


def test_scripts_refuse_tf32_matrix_products_before_doing_anything(tmp_path):
    """MVGenMaster's prior warp is plain fp32 matmul; TF32 would round it as coarsely as fp16."""
    commands = {"train": ["--run-name", "x", "--runs-root", str(tmp_path / "runs")],
                "render": ["--out", str(tmp_path / "render"), "--rows-file", str(tmp_path / "rows.json")],
                "baseline_choices": ["--out", str(tmp_path / "choices")], "check_step": []}
    scripts = {name: load_script(name) for name in commands}
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for name, argv in commands.items():
            with pytest.raises(AssertionError, match="TF32"):
                scripts[name].main(argv)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------------- baseline_choices.py
def choices_key_files(tmp_path: Path) -> list[Path]:
    """Key files for the folder: a named cross-period set and a same-period set; and a file that repeats the first."""
    def keys(rows):
        names = [f"view_{row:02d}.png" for row in rows]
        return {"refs": names[:3], "targets": names[3:]}

    first, again = tmp_path / "gates.json", tmp_path / "again.json"
    first.write_text(json.dumps([{"name": "fire", **keys(AFTER_FROM_MODERN)}, keys(MODERN_FROM_MODERN)]))
    again.write_text(json.dumps([keys(AFTER_FROM_MODERN)]))
    return [first, again]


def test_baseline_choices_collects_and_classes_its_view_sets(folder, tmp_path, monkeypatch, capsys):
    script = load_script("baseline_choices")
    assert script.set_kind([1, 1, 1, 1, 1], 3) == "same" and script.set_kind([1, 1, 1, 2, 2], 3) == "cross"
    assert script.set_kind([0, 1, 2, 2], 2) == "cross"          # no reference in the targets' period
    assert script.set_kind([1, 2, 1, 2, 2], 3) == "mixed" and script.set_kind([1, 1, 1, 2], 2) == "mixed"

    view_sets = script.collect_view_sets(folder, choices_key_files(tmp_path), 3, ViewSamplingConfig(), SIZE)
    assert [(v.name, v.source, v.kind) for v in view_sets[:2]] == [("fire", "gates.json", "cross"),
                                                                   ("gates/1", "gates.json", "same")]
    assert view_sets[0].rows == AFTER_FROM_MODERN and view_sets[1].rows == MODERN_FROM_MODERN
    # targets that no reference shows as they looked: all of 2020 with references of 2012, none of 2012.3 to 2012.5
    assert [v.cross_date_targets for v in view_sets[:2]] == [3, 0]
    assert all(v.warnings() == [] and v.not_usable == {} and v.targets_no_reference_sees == () for v in view_sets)
    assert "again/0 (again.json) is the view set fire: measured once" in capsys.readouterr().out
    # the sampled sets are the ones a training run's date-loss probe uses
    scene = MVGenScene(folder, resolution=SIZE)
    assert [v.name for v in view_sets[2:]] == ["sampled/0", "sampled/1", "sampled/2"]
    assert [v.rows for v in view_sets[2:]] == [tuple(rows.tolist()) for rows in scene_view_sets(scene, 3)]
    for view_set in view_sets[2:]:
        assert view_set.source == "sampler" and len(view_set.rows) == 8
        assert view_set.kind == script.set_kind(scene.period[list(view_set.rows)].tolist(), 3)

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps([{"refs": ["view_08.png", "view_09.png", "nowhere.png"],
                                    "targets": ["view_11.png"]}]))
    with pytest.raises(KeyError, match="nowhere.png"):
        script.collect_view_sets(folder, [unknown], 0, ViewSamplingConfig(), SIZE)

    # --list-sets: the view sets are printed, nothing is loaded or written
    monkeypatch.setattr(script, "load_mvgen", lambda model_dir: pytest.fail("the model was loaded"))
    capsys.readouterr()
    script.main(["--out", str(tmp_path / "out"), "--dataset", str(folder), "--resolution", str(SIZE), "--sampled", "1",
                 *[arg for path in choices_key_files(tmp_path) for arg in ("--keys-file", str(path))], "--list-sets"])
    printed = capsys.readouterr().out
    assert "3 view sets (1 same, " in printed
    assert "periods: ['before 1970', '1970 to 2019.2877', 'from 2019.2877']" in printed
    assert "fire" in printed and "cross references [2012.0, 2012.1, 2012.2] targets [2020.0, 2020.1, 2020.2]" in printed
    assert "targets [2020.0, 2020.1, 2020.2]; 3 of 3 targets cross-date" in printed and "warning" not in printed
    assert not (tmp_path / "out").exists()
    defaults = script.parse_args(["--out", "x"])
    assert defaults.cross_date_pixels == ["keep"] and defaults.cross_date_gap == LossConfig().cross_date_gap == 1.0
    assert script.PRIOR_RULES == (*step.CROSS_DATE_PIXELS, "no_priors")
    assert script.parse_args(["--out", "x", "--cross-date-pixels", "drop", "keep", "drop"]).cross_date_pixels == [
        "drop", "keep"]                                         # a value given twice is measured once
    with pytest.raises(SystemExit):
        script.parse_args(["--out", "x", "--cross-date-pixels", "some"])
    with pytest.raises(SystemExit):                             # the renderer cannot leave out every prior
        script.parse_args(["--out", "x", "--cross-date-pixels", "no_priors", "keep", "--render"])
    assert defaults.scales == ["scene", 1.0, 0.5, 0.2] and defaults.class_labels == [0, 1]
    assert defaults.timesteps == [100, 300, 500, 700, 900] and defaults.noise_seeds == [0, 1] and defaults.sampled == 6
    assert [path.name for path in script.DEFAULT_KEYS_FILES] == [
        "mvgen_gate1a_post.json", "mvgen_gate1b_pre.json", "mvgen_gate2.json", "mvgen_nvs_test_sets.json"]
    assert all(path.exists() for path in script.DEFAULT_KEYS_FILES)


def test_baseline_choices_flags_excluded_views_and_targets_that_no_reference_sees(folder, tmp_path, monkeypatch,
                                                                                  capsys):
    """Two sets of ``configs/mvgen_nvs_test_sets.json`` break the sampler's rules like this: ``easy0`` has a
    sideways target, and ``easy1`` a target that shares fewer than 30 points with each of its references."""
    script = load_script("baseline_choices")
    views = pd.read_parquet(folder / "views.parquet")
    views["excluded"] = ["rolled" if row == 17 else "" for row in range(N)]
    views.loc[17, "usable"] = False                             # a target of the set "fire"
    views.to_parquet(folder / "views.parquet", index=False)
    covis = np.load(folder / "covisibility.npy")
    covis[13, :], covis[:, 13] = 29, 29                         # a target of the other set: just under min_shared
    covis[13, 13] = 400
    np.save(folder / "covisibility.npy", covis)

    assert script.unseen_targets(covis, MODERN_FROM_MODERN, 3, 30) == [13]
    assert script.unseen_targets(covis, MODERN_FROM_MODERN, 3, 29) == []        # 29 shared points are enough for 29
    covis[13, 9] = covis[9, 13] = 30                            # one reference that sees it is enough
    assert script.unseen_targets(covis, MODERN_FROM_MODERN, 3, 30) == []
    assert script.unseen_targets(covis, (13, 8, 9, 10), 1, 30) == [8, 10]       # view 13 as the only reference

    keys_file = choices_key_files(tmp_path)[0]
    fire, modern = script.collect_view_sets(folder, [keys_file], 0, ViewSamplingConfig(), SIZE)
    assert fire.not_usable == {"view_17.png": "rolled"} and fire.targets_no_reference_sees == ()
    assert modern.not_usable == {} and modern.targets_no_reference_sees == ("view_13.png",)
    assert len(fire.warnings()) == len(modern.warnings()) == 1
    options = ["--dataset", str(folder), "--resolution", str(SIZE), "--sampled", "0", "--keys-file", str(keys_file)]
    capsys.readouterr()
    script.main(["--out", str(tmp_path / "unused"), *options, "--list-sets"])   # the printed list says so ...
    lines = capsys.readouterr().out.splitlines()
    flagged = {line.split()[0]: lines[i + 1].strip() for i, line in enumerate(lines[:-1])
               if lines[i + 1].lstrip().startswith("warning:")}
    assert flagged["fire"] == "warning: excluded by the dataset (not usable): {'view_17.png': 'rolled'}"
    assert flagged["gates/1"].startswith("warning: no reference sees target(s) ['view_13.png']: ")
    assert len(flagged) == 2 and not (tmp_path / "unused").exists()

    monkeypatch.setattr(script, "load_mvgen", lambda model_dir: tiny_model())  # ... and so does sets.json
    script.main(["--out", str(tmp_path / "out"), *options, "--scales", "1", "--class-labels", "0", "--timesteps",
                 "500", "--noise-seeds", "0"])
    sets = json.loads((tmp_path / "out" / "sets.json").read_text())
    assert [(s["not_usable"], s["targets_no_reference_sees"]) for s in sets] == [
        ({"view_17.png": "rolled"}, []), ({}, ["view_13.png"])]
    assert len((tmp_path / "out" / "losses.jsonl").read_text().splitlines()) == 2      # they are measured all the same


def test_baseline_choices_measures_every_scale_and_label_on_the_same_noisy_latents(folder, tmp_path, monkeypatch,
                                                                                   capsys):
    script = load_script("baseline_choices")
    models = []
    monkeypatch.setattr(script, "load_mvgen", lambda model_dir: models.append(tiny_model()) or models[-1])
    out = tmp_path / "choices"
    script.main(["--out", str(out), "--dataset", str(folder), "--resolution", str(SIZE), "--sampled", "2",
                 *[arg for path in choices_key_files(tmp_path) for arg in ("--keys-file", str(path))],
                 "--scales", "scene", "0.5", "sample", "--class-labels", "0", "1", "--timesteps", "200", "800",
                 "--noise-seeds", "0", "1", "--render", "--render-steps", "1"])
    printed = capsys.readouterr().out
    model, = models                                             # loaded once
    assert model.date is None and not (out / "dataset").exists()        # every depth map was in the dataset folder

    sets = json.loads((out / "sets.json").read_text())
    assert [(v["name"], v["kind"], v["source"]) for v in sets[:2]] == [("fire", "cross", "gates.json"),
                                                                       ("gates/1", "same", "gates.json")]
    assert sets[0]["rows"] == list(AFTER_FROM_MODERN) and sets[0]["periods"] == [1] * 3 + [2] * 3
    assert sets[0]["file_keys"][0] == "view_08.png" and sets[0]["references_without_reliable_depth"] == []
    assert [(v["cross_date_targets"], v["not_usable"], v["targets_no_reference_sees"]) for v in sets[:2]] == [
        (3, {}, []), (0, {}, [])]
    names = [v["name"] for v in sets]
    assert names == ["fire", "gates/1", "sampled/0", "sampled/1"]

    records = [json.loads(line) for line in (out / "losses.jsonl").read_text().splitlines()]
    assert len(records) == 4 * 3 * 2 * 2 * 2                    # view sets x scales x labels x timesteps x noise seeds
    # without --cross-date-pixels every prior is kept, whatever the targets' dates
    assert {r["cross_date_pixels"] for r in records} == {"keep"}
    assert {r["cross_date_targets"] for r in records if r["set"] == "fire"} == {3}
    assert [(r["set"], r["scale"], r["class_label"], r["timestep"], r["noise_seed"]) for r in records] == [
        (name, scale, label, timestep, seed) for name in names for scale in ("scene", "0.5", "sample")
        for label in (0, 1) for timestep in (200, 800) for seed in (0, 1)]
    assert {(r["scale"], r["scale_factor"]) for r in records} == {("scene", 1.0), ("0.5", 0.5), ("sample", 1.0)}
    assert all(r["loss"] > 0 and r["loss"] == pytest.approx(np.mean(r["loss_per_target"])) for r in records)
    assert {r["kind"] for r in records if r["set"] == "fire"} == {"cross"}
    # what the model was given: no dates, fp32, the label; and per view set, timestep and noise seed the very same
    # noisy latents at every scale and label
    calls = model.unet.seen[:len(records)]
    assert all(not call["grad"] and call["sample"].dtype == torch.float32 for call in calls)
    assert all(not is_zero(prior).any() for call in calls for prior in call["coords"])
    assert [int(call["class_labels"]) for call in calls] == [r["class_label"] for r in records]
    by_noise = {}
    for record, call in zip(records, calls):
        by_noise.setdefault((record["set"], record["timestep"], record["noise_seed"]), []).append(call["sample"])
    assert len(by_noise) == 4 * 2 * 2 and all(len(samples) == 6 for samples in by_noise.values())
    assert all(torch.equal(samples[0], other) for samples in by_noise.values() for other in samples[1:])
    noisy = [samples[0] for samples in by_noise.values()]
    assert all(not torch.equal(a, b) for i, a in enumerate(noisy) for b in noisy[i + 1:])

    summary = json.loads((out / "summary.json").read_text())
    assert summary["cross_date_gap"] == 1.0 and {row["cross_date_pixels"] for row in summary["loss"]} == {"keep"}
    loss_rows = {(row["scale"], row["class_label"], row["kind"]): row for row in summary["loss"]}
    kinds = [kind for kind in ("same", "cross", "mixed") if kind in {v["kind"] for v in sets}] + ["all"]
    assert list(loss_rows) == [(scale, label, kind) for scale in ("scene", "0.5", "sample") for label in (0, 1)
                               for kind in kinds]
    per_set = [np.mean([r["loss"] for r in records if (r["set"], r["scale"], r["class_label"]) == (name, "scene", 0)])
               for name in names]
    assert loss_rows["scene", 0, "all"]["mean"] == pytest.approx(np.mean(per_set))
    assert loss_rows["scene", 0, "all"]["standard_error"] == pytest.approx(np.std(per_set, ddof=1) / 2)
    assert loss_rows["scene", 0, "all"]["sets"] == 4 and loss_rows["scene", 0, "all"]["vs_first"] == 0.0
    assert "target loss of the released model" in printed and "change from scale scene, label 0" in printed
    assert f"{'sample':>8} {1:>5} {'keep':>9} | " in printed and "all (4 sets)" in printed

    renders = [json.loads(line) for line in (out / "renders.jsonl").read_text().splitlines()]
    assert [(r["scale"], r["class_label"], r["set"]) for r in renders] == [
        (scale, label, name) for scale in ("scene", "0.5", "sample") for label in (0, 1) for name in names]
    assert all(np.isfinite(r["psnr"]) for r in renders) and len(summary["psnr"]) == len(summary["loss"])
    assert {r["cross_date_pixels"] for r in renders} == {"keep"} and "rendering with" not in printed
    assert sorted(path.name for path in (out / "renders" / "scale_0.5_label_1").iterdir()) == [
        "fire.jpg", "fire.json", "gates_1.jpg", "gates_1.json", "sampled_0.jpg", "sampled_0.json", "sampled_1.jpg",
        "sampled_1.json"]
    rendered = json.loads((out / "renders" / "scale_0.5_label_1" / "fire.json").read_text())
    assert (rendered["scale"], rendered["scale_factor"]) == (0.5, 0.5)
    assert (rendered["dates"], rendered["seeds"]) == (["true"], [0])
    whole, halved = model.pipelines[0].calls[0], model.pipelines[2 * 4].calls[0]        # fire at scene scale and at 0.5
    torch.testing.assert_close(halved["w2c"][:, :3, 3], whole["w2c"][:, :3, 3] * 0.5)
    assert (halved["class_label"], model.pipelines[4].calls[0]["class_label"], halved["steps"]) == (0, 1, 1)
    assert "PSNR of one render against the real targets" in printed


def test_baseline_choices_compares_cross_date_values_on_the_same_noisy_latents(folder, tmp_path, monkeypatch, capsys):
    """``--cross-date-pixels``: what ``build_inputs`` is told to zero follows the rule of a training step, so it
    changes for the set whose targets lie after the fire and not for the set of one date."""
    script = load_script("baseline_choices")
    models, given, real_build_inputs = [], [], step.build_inputs
    monkeypatch.setattr(script, "load_mvgen", lambda model_dir: models.append(tiny_model()) or models[-1])

    def watched_build_inputs(model, sample, latents, **dropped):
        given.append((sample["rows"].tolist(), {name: views.tolist() if torch.is_tensor(views) else views
                                                for name, views in dropped.items()}))
        return real_build_inputs(model, sample, latents, **dropped)

    monkeypatch.setattr(script, "build_inputs", watched_build_inputs)

    def run(name: str, *options: str) -> tuple[Path, list[dict]]:
        """Run the script into ``tmp_path / name`` on the two keyed view sets; returns the folder and its losses."""
        models.clear()
        given.clear()
        out = tmp_path / name
        script.main(["--out", str(out), "--dataset", str(folder), "--resolution", str(SIZE), "--sampled", "0",
                     "--keys-file", str(choices_key_files(tmp_path)[0]), "--scales", "scene", "--class-labels", "0",
                     "1", "--timesteps", "200", "800", "--noise-seeds", "0", *options])
        return out, [json.loads(line) for line in (out / "losses.jsonl").read_text().splitlines()]

    rules = ["keep", "drop", "drop_all", "no_priors"]
    out, records = run("rules", "--cross-date-pixels", *rules)
    assert [(r["set"], r["class_label"], r["cross_date_pixels"], r["timestep"]) for r in records] == [
        (name, label, rule, timestep) for name in ("fire", "gates/1") for label in (0, 1) for rule in rules
        for timestep in (200, 800)]

    # what build_inputs was given, once per view set and value. "fire": targets of 2020, references of 2012.
    # "gates/1": all of 2012, so the rule marks no view and "drop" and "drop_all" take nothing.
    targets, nobody = [False] * 3 + [True] * 3, [False] * 6
    assert given == [
        (list(AFTER_FROM_MODERN), {"dropped_prior_views": None, "dropped_pixel_views": None}),
        (list(AFTER_FROM_MODERN), {"dropped_prior_views": None, "dropped_pixel_views": targets}),
        (list(AFTER_FROM_MODERN), {"dropped_prior_views": targets, "dropped_pixel_views": None}),
        (list(AFTER_FROM_MODERN), {"drop_priors": True}),
        (list(MODERN_FROM_MODERN), {"dropped_prior_views": None, "dropped_pixel_views": None}),
        (list(MODERN_FROM_MODERN), {"dropped_prior_views": None, "dropped_pixel_views": nobody}),
        (list(MODERN_FROM_MODERN), {"dropped_prior_views": nobody, "dropped_pixel_views": None}),
        (list(MODERN_FROM_MODERN), {"drop_priors": True})]
    # ... and what reached the UNet: (views without coordinate prior, views without warped pixels)
    model, = models
    for record, call in zip(records, model.unet.seen, strict=True):
        cross_date = targets if record["set"] == "fire" else nobody
        expected = {"keep": (nobody, nobody), "drop": (nobody, cross_date), "drop_all": (cross_date, cross_date),
                    "no_priors": ([True] * 6, [True] * 6)}[record["cross_date_pixels"]]         # the references' too
        assert tuple(is_zero(prior).tolist() for prior in call["coords"]) == expected
        assert int(call["class_labels"]) == record["class_label"]
    by_noise = {}
    for record, call in zip(records, model.unet.seen):
        by_noise.setdefault((record["set"], record["timestep"]), []).append(call["sample"])
    assert all(len(noisy) == 2 * 4 and all(torch.equal(noisy[0], other) for other in noisy[1:])
               for noisy in by_noise.values())                  # the same noisy latents for every label and value

    def losses(name: str, rule: str) -> list[float]:
        return [r["loss"] for r in records if (r["set"], r["cross_date_pixels"]) == (name, rule)]

    assert losses("gates/1", "keep") == losses("gates/1", "drop") == losses("gates/1", "drop_all")
    assert losses("gates/1", "no_priors") != losses("gates/1", "keep")
    assert len({tuple(losses("fire", rule)) for rule in ("keep", "drop", "drop_all")}) == 3

    sets = json.loads((out / "sets.json").read_text())
    assert [(v["name"], v["kind"], v["cross_date_targets"]) for v in sets] == [("fire", "cross", 3),
                                                                              ("gates/1", "same", 0)]
    assert {(r["set"], r["cross_date_targets"]) for r in records} == {("fire", 3), ("gates/1", 0)}
    # the table has the axis; every setting is compared with the first one on the same view sets
    summary = json.loads((out / "summary.json").read_text())
    rows = {(row["scale"], row["class_label"], row["cross_date_pixels"], row["kind"]): row for row in summary["loss"]}
    assert list(rows) == [("scene", label, rule, kind) for label in (0, 1) for rule in rules
                          for kind in ("same", "cross", "all")]
    assert rows["scene", 0, "keep", "all"]["vs_first"] == 0.0 and rows["scene", 0, "drop", "same"]["vs_first"] == 0.0
    assert rows["scene", 0, "drop_all", "cross"]["vs_first"] == pytest.approx(
        np.mean(losses("fire", "drop_all")[:2]) / np.mean(losses("fire", "keep")[:2]) - 1)
    assert rows["scene", 0, "drop_all", "cross"]["vs_first"] != 0 != rows["scene", 0, "no_priors", "same"]["vs_first"]
    printed = capsys.readouterr().out
    assert "change from scale scene, label 0, priors keep on the same view sets" in printed
    assert f"{'scene':>8} {1:>5} {'no_priors':>9} | " in printed and f"{'priors':>9} | same (1 sets)" in printed

    # the default is "keep" alone: what was measured before the option existed
    _, default = run("default")
    assert given == [(list(rows_), {"dropped_prior_views": None, "dropped_pixel_views": None})
                     for rows_ in (AFTER_FROM_MODERN, MODERN_FROM_MODERN)]
    assert default == [r for r in records if r["cross_date_pixels"] == "keep"] == run("keep", "--cross-date-pixels",
                                                                                     "keep")[1]
    assert (tmp_path / "default" / "summary.json").read_text() == (tmp_path / "keep" / "summary.json").read_text()
    capsys.readouterr()

    # the first value given is the one the others are compared with, and the one that --render renders
    out, records = run("first", "--cross-date-pixels", "drop", "keep", "--render", "--render-steps", "1")
    printed = capsys.readouterr().out
    summary = json.loads((out / "summary.json").read_text())
    rows = {(row["class_label"], row["cross_date_pixels"], row["kind"]): row for row in summary["loss"]}
    assert rows[0, "drop", "all"]["vs_first"] == 0.0 and rows[0, "keep", "cross"]["vs_first"] != 0
    assert "priors drop on the same view sets" in printed and "rendering with --cross-date-pixels drop only" in printed
    renders = [json.loads(line) for line in (out / "renders.jsonl").read_text().splitlines()]
    assert [(r["class_label"], r["set"], r["cross_date_pixels"]) for r in renders] == [
        (label, name, "drop") for label in (0, 1) for name in ("fire", "gates/1")]
    assert {row["cross_date_pixels"] for row in summary["psnr"]} == {"drop"}
    sampled = models[0].unet.seen[len(records):]                # the sampler's calls: [unconditional, conditional half]
    assert [is_zero(call["coords"][1])[6:].tolist() for call in sampled] == [targets, nobody] * 2
    assert all(not is_zero(call["coords"][0])[6:].any() for call in sampled)

    # another gap: 2012.4 and 2012.5 are more than 0.15 years from every reference (2012.0 to 2012.2), 2012.3 is not.
    # The set is of one period ("same") and has cross-date targets all the same.
    out, records = run("gap", "--cross-date-pixels", "drop", "--cross-date-gap", "0.15")
    assert given[1] == (list(MODERN_FROM_MODERN), {"dropped_prior_views": None,
                                                   "dropped_pixel_views": [False] * 4 + [True] * 2})
    assert [v["cross_date_targets"] for v in json.loads((out / "sets.json").read_text())] == [3, 2]
    assert json.loads((out / "summary.json").read_text())["cross_date_gap"] == 0.15

    # a run in a folder that holds another run's files replaces them: nothing is appended
    out, records = run("gap", "--cross-date-pixels", "keep")
    assert len(records) == 2 * 2 * 2 and {r["cross_date_pixels"] for r in records} == {"keep"}
    assert json.loads((out / "summary.json").read_text())["cross_date_gap"] == 1.0


def test_baseline_choices_summary_counts_a_view_set_once_and_compares_on_the_same_sets():
    script = load_script("baseline_choices")
    records = []
    for name, kind, base in (("a", "same", 1.0), ("b", "same", 3.0), ("c", "cross", 10.0)):
        for scale, factor in (("scene", 1.0), ("0.5", 1.1)):
            for label, added in ((0, 0.0), (1, 0.5)):
                records += [{"set": name, "kind": kind, "scale": scale, "class_label": label,
                             "cross_date_pixels": "keep", "loss": base * factor + added + noise}
                            for noise in (-0.1, 0.1)]
    rows = {(row["scale"], row["class_label"], row["kind"]): row for row in script.summarise(records, "loss", True)}
    assert list(rows) == [(scale, label, kind) for scale in ("scene", "0.5") for label in (0, 1)
                          for kind in ("same", "cross", "all")]
    same = rows["scene", 0, "same"]                              # the sets' means are 1 and 3
    assert (same["sets"], same["mean"], same["standard_error"]) == (2, pytest.approx(2.0), pytest.approx(1.0))
    assert rows["scene", 0, "cross"] == {"scale": "scene", "class_label": 0, "cross_date_pixels": "keep",
                                         "kind": "cross", "sets": 1, "mean": pytest.approx(10.0),
                                         "standard_error": None, "vs_first": 0.0, "vs_first_standard_error": None}
    assert rows["scene", 0, "all"]["mean"] == pytest.approx(14 / 3) and rows["scene", 0, "all"]["sets"] == 3
    # every set is 10% higher at the other scale: the paired comparison says so exactly, whatever the sets' spread
    assert rows["0.5", 0, "all"]["vs_first"] == pytest.approx(0.1)
    assert rows["0.5", 0, "all"]["vs_first_standard_error"] == pytest.approx(0.0, abs=1e-9)
    assert rows["0.5", 0, "all"]["standard_error"] > 2
    assert rows["scene", 1, "same"]["vs_first"] == pytest.approx((0.5 / 1 + 0.5 / 3) / 2)
    differences = {key: row["vs_first"] for key, row in zip(rows, script.summarise(records, "loss", False))}
    assert differences["scene", 1, "all"] == pytest.approx(0.5) and differences["0.5", 0, "cross"] == pytest.approx(1.0)

    lines = script.format_table(list(rows.values()), "the title", 4, True).splitlines()
    assert lines[0] == "the title" and "change from scale scene, label 0" in lines[1] and len(lines) == 3 + 4
    assert "same (2 sets)" in lines[2] and "cross (1 sets)" in lines[2] and "all (3 sets)" in lines[2]
    assert lines[3].split(" | ")[1].strip() == "2.0000 +- 1.0000 [+0.0% +- 0.0%]"
    assert lines[3].split(" | ")[2].strip() == "10.0000 +- n/a [+0.0% +- n/a]"
    assert lines[5].startswith("     0.5     0      keep | ") and "[+10.0% +- 0.0%]" in lines[5]


def test_baseline_choices_computes_missing_depth_as_prepare_depth_does(folder, tmp_path, monkeypatch, capsys):
    script = load_script("baseline_choices")
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "mvgen"))
    prepare_depth = importlib.import_module("prepare_depth")    # the module the script imports when depth is missing
    points = np.column_stack([np.linspace(-1.0, 6.0, 400), np.zeros(400), np.full(400, 2.0)])
    np.savez(folder / "sparse_points.npz", p3d_ids=np.arange(400), p3d_xyz=points,
             view_starts=np.arange(N + 1) * 400, view_pid=np.tile(np.arange(400), N))    # every view sees all points
    depth_path(folder, 9).unlink()                              # a reference without a depth map
    before = {path.name: path.stat().st_mtime_ns for path in sorted(folder.rglob("*"))}
    scaled = []

    def depth_pro_stand_in(depth_model, transform, image, K, sparse):
        scaled.append(len(sparse))
        return np.full(image.shape[-2:], 3.0, dtype=np.float32), {"points": len(sparse), "scale": 1.5,
                                                                   "median_rel_error": 0.02}

    monkeypatch.setattr(prepare_depth, "create_model_and_transforms", lambda device: (nn.Identity(), None))
    monkeypatch.setattr(prepare_depth, "reference_depth", depth_pro_stand_in)
    work = script.folder_with_depth(folder, tmp_path / "work", [8, 9, 10, 9], SIZE)
    assert work == tmp_path / "work" and len(scaled) == 1 and scaled[0] >= 50          # only the missing map is made
    assert {path.name: path.stat().st_mtime_ns for path in sorted(folder.rglob("*"))} == before     # dataset: untouched
    assert depth_path(work, 8).is_symlink() and depth_path(work, 8).resolve() == depth_path(folder, 8).resolve()
    computed = np.load(depth_path(work, 9))
    assert not depth_path(work, 9).is_symlink() and computed.dtype == np.float16 and np.all(computed == 3.0)
    assert (work / "views.parquet").is_symlink() and (work / SCENE_FILE).is_symlink()
    stats = pd.read_parquet(work / "depth_stats.parquet")
    assert stats["row"].tolist() == [8, 9, 10] and stats["ok"].all() and stats["scale"].tolist() == [1.0, 1.5, 1.0]
    scene = MVGenScene(work, resolution=SIZE, require_depth=False)
    sample = scene.make_sample(np.array(AFTER_FROM_MODERN), 3)
    assert torch.all(sample["depth"][1] == 3.0) and torch.all(sample["depth"][0] == 2.0)        # row 8: 1 + 8 / 8
    assert scene.depth_ok[[8, 9, 10]].all() and "1 of the 3 references have no depth map" in capsys.readouterr().out
    script.folder_with_depth(folder, tmp_path / "work", [8, 9, 10], SIZE)       # again, into the same folder
    assert len(scaled) == 2 and depth_path(work, 10).is_symlink()

    # with every map in the dataset folder it is used as it is, and Depth-Pro is not loaded
    monkeypatch.setattr(prepare_depth, "create_model_and_transforms", lambda device: pytest.fail("Depth-Pro loaded"))
    assert script.folder_with_depth(folder, tmp_path / "unused", [8, 10, 11], SIZE) == folder
    assert not (tmp_path / "unused").exists()

    # the whole script on such a folder: the samples come from <out>/dataset
    monkeypatch.setattr(prepare_depth, "create_model_and_transforms", lambda device: (nn.Identity(), None))
    monkeypatch.setattr(script, "load_mvgen", lambda model_dir: tiny_model())
    out = tmp_path / "choices"
    script.main(["--out", str(out), "--dataset", str(folder), "--resolution", str(SIZE), "--sampled", "0",
                 "--keys-file", str(choices_key_files(tmp_path)[0]), "--scales", "1", "--class-labels", "0",
                 "--timesteps", "500", "--noise-seeds", "0"])
    assert np.all(np.load(depth_path(out / "dataset", 9)) == 3.0) and not depth_path(folder, 9).exists()
    assert len((out / "losses.jsonl").read_text().splitlines()) == 2
    assert json.loads((out / "sets.json").read_text())[0]["references_without_reliable_depth"] == []

    points[:, 2] = -2.0                                         # behind every camera: no depth can be scaled
    np.savez(folder / "sparse_points.npz", p3d_ids=np.arange(400), p3d_xyz=points,
             view_starts=np.arange(N + 1) * 400, view_pid=np.tile(np.arange(400), N))
    with pytest.raises(ValueError, match="view 9 \\(view_09.png\\) has no COLMAP point"):
        script.folder_with_depth(folder, tmp_path / "work2", [9], SIZE)


# ----------------------------------------------------------------------------- MVGenMaster's own UNet and pipeline
def small_mvgenmaster(tmp_path: Path) -> MVGen:
    """A small model of MVGenMaster's own classes, set up as its loader sets up the released one.

    The UNet takes the mask, camera rays and warped pixels through ``add_conv_in`` together with the
    coordinate prior, has a domain embedding and no cross-attention; the VAE reduces by 8, as the
    real one. ``sampling_pipeline()`` is MVGenMaster's pipeline, with the scheduler files it reads.

    MVGenMaster builds the last layer of ``add_conv_in`` and the domain embedding as zeros; the
    released weights are not zero there. They get random values here, so that the output depends on
    the mask, the camera rays, both priors and the domain label, as the released model's does
    (``test_small_mvgenmaster_reads_all_its_conditioning``): a test that fed this UNet wrong
    conditioning would otherwise pass.
    """
    scheduler_dir = tmp_path / "sd" / "scheduler"
    scheduler_dir.mkdir(parents=True)
    (scheduler_dir / "scheduler_config.json").write_text(json.dumps({
        "_class_name": "DDIMScheduler", "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
        "clip_sample": False, "num_train_timesteps": 1000, "prediction_type": "v_prediction",
        "set_alpha_to_one": False, "steps_offset": 1}))
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8, block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=None, attention_head_dim=4, use_linear_projection=True,
        class_embed_type="zero_init", num_class_embeds=3)
    unet.use_rope = False                               # what from_pretrained(model_cfg=...) sets on the released UNet
    unet.additional_in_channels, unet.coord_encoder, unet.prior_type = 11, "conv_in", "3dpe+pixel"
    unet.add_conv_in = ExConvEncoder2([11, 193], 32)
    for built_as_zeros in (unet.add_conv_in.conv_out[-1], unet.class_embedding):
        for parameter in built_as_zeros.parameters():
            nn.init.normal_(parameter, std=0.05)
    vae = AutoencoderKL(block_out_channels=(8, 8, 8, 8), down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, norm_num_groups=4, latent_channels=4)
    model_cfg = EasyDict(prior_type="3dpe+pixel", depth_mid_times=4, enable_depth=True, priors3d=True, coord_dim=192,
                         domain_dict={"others": 0, "megascenes": 1, "objaverse": 2})
    config = EasyDict(CONFIG, pretrained_model_name_or_path=str(tmp_path / "sd"), rescale_betas_zero_snr=True,
                      beta_schedule="snr_rescale", snr_rescale=4, nframe=8, model_cfg=model_cfg)
    return MVGen(unet=unet.requires_grad_(False).eval(), vae=vae.requires_grad_(False).eval(), config=config,
                 scheduler=DDPMScheduler.from_config(SCHEDULER))


def far_band(folder: Path, rows, size: int) -> None:
    """Give the depth maps of ``rows`` a band at 40 times their depth (sky), which the prior code contracts."""
    for row in rows:
        depth = np.load(depth_path(folder, row))
        depth[: size // 4] *= 40
        np.save(depth_path(folder, row), depth)


def test_small_mvgenmaster_reads_all_its_conditioning(tmp_path, monkeypatch):
    """Zeroing the priors, the camera rays or changing the domain label each changes the UNet's output."""
    monkeypatch.setattr(step, "get_camera_embedding", get_camera_embedding)        # the real ones, not the fakes
    monkeypatch.setattr(step, "get_3d_priors", get_3d_priors)
    model = small_mvgenmaster(tmp_path)
    sample = MVGenScene(write_folder(tmp_path / "scene64", 64), resolution=64).make_sample(
        np.array(MODERN_FROM_MODERN), 3)
    latents = encode_images(model, sample["images"], sample_posterior=False)
    targets = torch.tensor([False] * 3 + [True] * 3)

    def output(class_label=0, **dropped):
        with torch.no_grad():
            inputs = build_inputs(model, sample, latents, **dropped)
            return predict(model, latents, torch.tensor([500]), inputs, class_label, None, None, "fp32")

    plain = output()
    assert torch.equal(plain, output())
    changed = {"all priors": output(drop_priors=True), "the targets' priors": output(dropped_prior_views=targets),
               "the targets' warped pixels": output(dropped_pixel_views=targets),
               "the camera rays": output(drop_camera=True), "the domain label": output(class_label=1)}
    for what, out in changed.items():
        assert not torch.allclose(out, plain, atol=1e-4), what
    outputs = list(changed.values())
    assert all(not torch.equal(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1:])


def test_no_autocast_is_open_around_the_conditioning_in_a_bf16_run(tmp_path, monkeypatch):
    """The fidelity audit's rule: autocast surrounds the UNet and nothing else, in training, probing and rendering.

    MVGenMaster warps the references with matrix products, which autocast would run in half precision;
    under bf16 the warped pixels then cover a quarter of the targets instead of 70%.
    """
    autocast_on = {"camera rays": [], "priors": [], "unet": []}

    def watched(real, name):
        def function(*args, **kwargs):
            autocast_on[name].append(torch.is_autocast_enabled("cpu"))
            return real(*args, **kwargs)
        return function

    for module in (step, pipeline_module):                      # a training step's and the sampling pipeline's
        monkeypatch.setattr(module, "get_camera_embedding", watched(get_camera_embedding, "camera rays"))
        monkeypatch.setattr(module, "get_3d_priors", watched(get_3d_priors, "priors"))
    folder = write_folder(tmp_path / "scene64", 64)
    cfg = config(folder, tmp_path / "run", resolution=64, steps=1, grad_accum=1, pinned_rows=(AFTER_FROM_MODERN,),
                 probe_every=1, probe_timesteps=(500,), render_every=1, render_dates=("true", "swapped"),
                 render_seeds=(0,), render_steps=1, loss=LossConfig(precision="bf16"))
    trainer = Trainer(cfg, model=small_mvgenmaster(tmp_path))
    trainer.model.unet.conv_in.register_forward_pre_hook(
        lambda module, args: autocast_on["unet"].append(torch.is_autocast_enabled("cpu")))
    trainer.train()
    # one training step; at step 0 and at step 1 the loss probe (one view set) and two renders
    assert autocast_on["camera rays"] == autocast_on["priors"] == [False] * (1 + 2 * (1 + 2))
    assert autocast_on["unet"] == [True] * (1 + 2 * (3 + 2))    # ... the probe: three date variants
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.params)
    # with every prior kept, the date cannot matter at step 0, through MVGenMaster's own forward pass and sampler
    step0 = json.loads((tmp_path / "run" / "renders" / "step_000000" / "set0.json").read_text())
    assert [r["diff_vs_first"] for r in step0["renders"]] == [0.0, 0.0]


@pytest.mark.parametrize("setting", ["drop", "drop_all"])
def test_trainer_and_probes_on_mvgenmasters_own_unet_and_pipeline(tmp_path, monkeypatch, setting):
    monkeypatch.setattr(step, "get_camera_embedding", get_camera_embedding)        # the real ones, not the fakes
    monkeypatch.setattr(step, "get_3d_priors", get_3d_priors)
    size, n_views = 64, 6
    folder = write_folder(tmp_path / "scene64", size)
    cfg = config(folder, tmp_path / "run", resolution=size, steps=2,
                 pinned_rows=(AFTER_FROM_MODERN, MODERN_FROM_MODERN), probe_every=2, render_every=2,
                 render_dates=("true", "swapped"), render_seeds=(0,),
                 loss=LossConfig(precision="fp32", cross_date_pixels=setting))
    trainer = Trainer(cfg, model=small_mvgenmaster(tmp_path))
    unet, blocks = trainer.model.unet, dated_blocks(trainer.model.unet)
    # per ResBlock call: were gradients on? Per UNet call, from the prior encoder's two inputs: (were gradients on?,
    # which views had no warped pixels, which had no coordinate prior)
    block_calls, priors = [], []
    for block in blocks:
        block.register_forward_pre_hook(lambda module, args: block_calls.append(torch.is_grad_enabled()))
    unet.add_conv_in.register_forward_pre_hook(lambda module, args: priors.append(
        (torch.is_grad_enabled(), is_zero(args[0][:, 7:]).tolist(), is_zero(args[1]).tolist())))
    trainer.train()
    pixel_priors = [(grad, pixels) for grad, pixels, _ in priors]

    # step 0 is the base model exactly, through MVGenMaster's own forward pass and sampler; then the adapters learn
    log = read_log(cfg.run_dir)
    first, last = [r for r in log if r.get("probe") == "date_loss:pinned"]
    assert (first["step"], last["step"]) == (0, 2)
    assert first["loss_true"] == first["loss_swapped"] == first["loss_unknown"] and first["change"] == 0.0
    assert last["change"] < 0 and last["rel_gap_swapped"] != 0 and last["rel_gap_unknown"] != 0
    renders = Path(cfg.run_dir) / "renders"
    for name in ("set0", "set1"):
        before = json.loads((renders / "step_000000" / f"{name}.json").read_text())
        after = json.loads((renders / "step_000002" / f"{name}.json").read_text())
        # at step 0 the two dates already give two pictures, through the cross-date rule alone: at one of them
        # the targets are rendered without warped pixels, and the UNet reads those
        assert before["renders"][0]["diff_vs_first"] == 0.0 and before["renders"][1]["diff_vs_first"] > 0
        assert all(np.isfinite(r["psnr"]) for r in after["renders"])
        assert [r["psnr"] for r in after["renders"]] != [r["psnr"] for r in before["renders"]]      # and it learned
    assert Image.open(renders / "step_000002" / "set0.jpg").size == (230 + n_views * 256, 18 + 14 + 3 * 256)

    # MVGenMaster's blocks checkpoint in train() mode: 4 training samples, and backward() re-ran their ResBlocks
    training_calls = sum(block_calls)
    assert 4 * len(blocks) < training_calls <= 2 * 4 * len(blocks)
    assert len([grad for grad, _ in pixel_priors if grad]) == 4

    # warped pixels: dropped for targets no reference is dated near, by the true dates in the loss probe ...
    targets_only = [False] * 3 + [True] * 3
    probed = [dropped for grad, dropped in pixel_priors if not grad and len(dropped) == n_views]
    assert probed == ([targets_only] * 6 + [[False] * n_views] * 6) * 2         # 2 timesteps x 3 dates per view set
    # ... and by the date rendered at in the sampler ([unconditional half: no priors at all, conditional half])
    rendered = [dropped for grad, dropped in pixel_priors if len(dropped) == 2 * n_views]
    without, with_pixels = [True] * n_views + targets_only, [True] * n_views + [False] * n_views
    one_evaluation = [without] * 2 + [with_pixels] * 2 + [with_pixels] * 2 + [without] * 2    # 2 sampler steps each
    assert rendered == one_evaluation * 2
    # the coordinate prior of those targets: kept by "drop", zeroed with the pixels by "drop_all"; in the loss
    # probe and in the sampler alike (whose unconditional half has no priors at all)
    evaluated = [(pixels, coordinates) for grad, pixels, coordinates in priors if not grad]
    assert len(evaluated) == 2 * (2 * 6 + 8)
    for pixels, coordinates in evaluated:
        expected = pixels[-n_views:] if setting == "drop_all" else [False] * n_views
        assert coordinates[-n_views:] == expected and all(coordinates[:-n_views])


def test_sampling_sees_the_priors_of_training_whatever_the_precision(tmp_path, monkeypatch):
    """Under bf16 the UNet runs in bf16 but MVGenMaster's warping must not: it would move the warped pixels."""
    monkeypatch.setattr(step, "get_camera_embedding", get_camera_embedding)
    monkeypatch.setattr(step, "get_3d_priors", get_3d_priors)
    model = small_mvgenmaster(tmp_path)
    prepare_for_finetuning(model, FinetuneConfig(date_freqs=4, date_hidden=16))
    folder = write_folder(tmp_path / "scene64", 64)
    far_band(folder, AFTER_FROM_MODERN[:3], 64)
    sample = MVGenScene(folder, resolution=64).make_sample(np.array(AFTER_FROM_MODERN), 3)
    latents = encode_images(model, sample["images"], sample_posterior=False)
    trained_on = build_inputs(model, sample, latents)                   # fp32, as a training step builds them
    # the far band is in play: handed the sample's float32 depth itself, the prior code would contract it twice
    # for the warped pixels (and edit the sample)
    twice = get_3d_priors(model.config, sample["depth"].clone(), sample["K"], sample["w2c"], 3, nframe=6, device="cpu",
                          colors=sample["images"], latents=latents, vae=model.vae, prior_type="3dpe+pixel")
    assert torch.equal(twice[0], trained_on.coords[0]) and not torch.equal(twice[1], trained_on.coords[1])

    seen, autocast_on = [], {"unet": [], "vae": []}
    model.unet.register_forward_pre_hook(lambda module, args, kwargs: seen.append(kwargs), with_kwargs=True)
    model.unet.conv_in.register_forward_pre_hook(
        lambda module, args: autocast_on["unet"].append(torch.is_autocast_enabled("cpu")))
    model.vae.encoder.register_forward_pre_hook(
        lambda module, args: autocast_on["vae"].append(torch.is_autocast_enabled("cpu")))
    frames = {precision: render_views(model, model.sampling_pipeline(), sample, sample["date_mid"],
                                      torch.ones(6, dtype=torch.bool), seed=0, steps=1, precision=precision)
              for precision in ("fp32", "bf16")}
    assert autocast_on == {"unet": [False, True], "vae": [False, False]}
    for call in seen:                                                   # [unconditional half, conditional half]
        coordinates, pixels = call["coords"]
        assert torch.equal(coordinates[6:], trained_on.coords[0]) and torch.equal(pixels[6:], trained_on.coords[1])
        assert torch.equal(call["add_inputs"][1], trained_on.add_inputs[0])        # the mask and the camera rays
        assert coordinates.dtype == pixels.dtype == torch.float32
    assert frames["bf16"].shape == (6, 64, 64, 3) and frames["bf16"].dtype == np.uint8
    assert np.abs(frames["bf16"].astype(int) - frames["fp32"].astype(int)).mean() < 2      # the same pictures
