"""One training step of date-conditioned MVGenMaster: the conditioning tensors and the v-prediction loss.

A sample (built by :mod:`timemachine.mvgen.data`) is ``F`` views of one scene, the first ``n_ref`` of
them references. The step mirrors the authors' ``train.py`` (``external/MVGenMaster``, lines 700-866):
for the same latents, noise, timestep and dropout decisions the UNet receives the same tensors and
the loss is the same. ``tests/test_mvgen_step.py`` checks this on the CPU against a line-by-line
transcription of that code with the authors' own camera, prior and scheduler functions, and
``scripts/mvgen/check_step.py`` checks the loss of the released model against recorded values on a GPU.

What the UNet is given (:func:`encode_images`, :func:`build_inputs`):

* latents ``[1, F, 4, h, w]``: the VAE's latents times its scaling factor. References enter clean,
  targets noised at one timestep per sample;
* ``add_inputs`` ``[1, F, 7, H, W]``: a mask (0 = reference, 1 = target) and 6 camera-ray channels;
* ``coords``, the 3D priors: the references warped into every view by their depth. For the released
  model (``prior_type = "3dpe+pixel"``) a list of two tensors, the Fourier-encoded 3D coordinates
  ``[F, 193, h, w]`` and the warped pixels with a validity mask ``[F, 4, H, W]``;
* the domain label (``class_label``) and, through the date conditioner, each view's date.

The loss (:func:`v_prediction_loss`) is the mean squared error between the UNet's output and the
velocity target on the target views only. :func:`training_loss` draws what a training step needs:

* the timestep, uniform over the scheduler's training steps, and the noise;
* dropout, by default at the authors' rates, so that classifier-free guidance keeps working: all
  camera rays zeroed (10% of samples), all priors zeroed (10%), otherwise each target's priors
  zeroed (15%);
* the dates: a year inside each view's interval, all marked unknown for 10% of samples (so that
  guidance on the date is possible), and the references' dates always unknown with
  ``date_views="targets"``;
* optionally (``cross_date_pixels``) fewer priors for *cross-date* targets: those that no reference
  shows as they looked, because every reference is dated more than ``cross_date_gap`` years away or
  lies in another period of the scene (:func:`cross_date_views`). The warped pixels then show the
  scene at the references' dates, which is wrong for such a target.

  - ``"drop"`` zeroes their warped pixels and keeps the warped 3D coordinates, which carry no
    appearance (option B in section 3.2 of ``docs/mvgenmaster_time_plan.md``). The released model
    never saw this combination: the authors' training zeroed both priors of a view together.
  - ``"drop_all"`` zeroes both priors of those targets. For a single view this is the state the
    authors' per-view prior dropout produced, so the frozen model has been trained on it. For a
    whole view set it is not: in the authors' training all five targets of a set lost their priors
    while the references kept theirs in about 0.002% of the sets, and with our default sampler this
    rule leaves about 70% of the samples in that state. The only state without any target prior
    that the authors trained on often is the whole-sample prior dropout above, and that one zeroes
    the references' priors as well.

  The probes and the renderer apply the same rule through the same function
  (:func:`cross_date_drops`), so that a run is evaluated on the inputs it trains on.

Look (experiment look1; ``timemachine.mvgen.look``). For a model whose date conditioner carries a look
(``MVGen.has_look``), each view is also given its standardised look code. The sample brings it:
``sample["look"]`` ``[F, 24]`` and ``sample["look_drop"]``, made by ``look.TrainingLooks`` in the data
loader from generators of its own (class fill, unspecified entries, arm D's medium conversion), so a
training step draws nothing more from its generator than it did before the look existed. The step only
ties the look's dropout to the date's: when the sample's dates are all dropped and ``look_drop`` is set
(``LossConfig.look_dropout``, 0.5), every look is unknown too; the look is never dropped while the date
is shown. A model without a look ignores the look entries of a sample.

Region weight (``LossConfig.region_weight``, 1 by default: off). What a date decides at Notre-Dame, the
spire and the roofs, is a few percent of a frame, so in the plain loss a wrong date costs almost
nothing (0.5% of the held-out loss after 4,000 steps, 2.8% inside the changed region for targets
from before the fire). With a weight ``lambda`` other than 1, every latent cell of a target view
counts ``w = 1 + (lambda - 1) * m`` times, where ``m`` in [0, 1] is the share of the cell that lies
where the scene changed (``sample["region"]``, from the dataset folder's regions file:
``timemachine.mvgen.data``), and the loss is ``sum(w * e) / sum(w)`` over the cells of all targets,
with ``e`` the squared error averaged over the latent channels (:func:`region_weighted_loss`).
Nothing else changes: the same random draws, the same forward pass. Such a step also reports the
plain loss and the plain mean error inside and outside the region, for the log. With ``lambda = 1``
the step is the one described above, computed by the same code as before the option existed.

Precision. MVGenMaster's gradient is sound under bf16 autocast and in fp32, but not under fp16
without a gradient scaler (``data/experiments/mvgen/grad_precision.json``), so :class:`LossConfig`
accepts only ``"bf16"`` and ``"fp32"``. Only the UNet call runs under autocast; the latents, the
target and the loss are fp32.

Gradient checkpointing. The UNet's blocks are run again during ``backward()``, and the date
conditioner's hooks run with them. Each forward pass keeps its own date embedding, bound to the
``temb`` tensor of that pass, and the re-run adds exactly that one (``timemachine.mvgen.date``, "One
embedding per UNet forward pass, bound to that pass"). So ``backward()`` may be called after
:func:`predict` has left ``model.dates(...)``, and other UNet forward passes (a probe, a second loss
term) may run between a loss from this module and its ``backward()`` without changing its gradient.

Depth. MVGenMaster's prior code contracts depth beyond ``depth_mid_times`` (4) times a reference's
median depth, in place, in the tensor that ``depth.to(device).float()`` returns. It does so once for
the coordinate prior and once for the warped pixels. The authors' ``train.py`` handed it depth on
the CPU while the model was on a GPU, so each of the two got its own copy and both priors used depth
contracted ONCE; that is what the released weights were trained on. Handed a float32 tensor already
on the device (as the authors' demo script does), the two share one tensor and the warped pixels
are computed from depth contracted twice, which moves sky and far background by a few pixels and
misaligns them with the coordinate prior. :func:`depth_for_priors` therefore hands over a private
float64 copy: every internal ``.float()`` then makes its own copy on any device, far depth is
contracted once for both priors, and a sample is never changed. The renderer does the same
(``timemachine.mvgen.probes.render_views``), so training and rendering see the same priors.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import einops
import torch

from src.modules.camera import get_camera_embedding
from src.modules.position_encoding import get_3d_priors
from timemachine.training.loss import DATE_VIEWS, draw_dates, shown_dates

if TYPE_CHECKING:
    from timemachine.mvgen.model import MVGen

PRECISIONS = ("bf16", "fp32")
# What a cross-date target (cross_date_views) is given: all its priors, no warped pixels, or no priors at all.
CROSS_DATE_PIXELS = ("keep", "drop", "drop_all")


@dataclass(frozen=True)
class LossConfig:
    """What a training step draws and how its loss is computed (see the module docstring)."""

    date_dropout: float = 0.1          # share of samples whose dates are all marked unknown
    camera_dropout: float = 0.1        # share of samples whose camera rays are all zeros
    prior_dropout: float = 0.1         # share of samples whose 3D priors are all zeros
    prior_view_dropout: float = 0.15   # in the other samples: chance per target view that its priors are zeros
    date_jitter: bool = True           # draw years inside each view's interval (else: the midpoint)
    date_views: str = "all"            # one of DATE_VIEWS; "targets": the references' dates are always unknown
    class_label: int = 0               # MVGenMaster's domain label: 0 regular datasets, 1 MegaScenes
    cross_date_pixels: str = "keep"    # one of CROSS_DATE_PIXELS: what cross-date targets lose
    cross_date_gap: float = 1.0        # years between date midpoints beyond which a reference is not "near"
    precision: str = "bf16"            # one of PRECISIONS
    region_weight: float = 1.0         # loss weight of latent cells inside the changed region (1: the plain loss)
    # The look (models with FinetuneConfig.look only; the draws are look.TrainingLooks', in the data loader):
    look_dropout: float = 0.5          # share of the date-dropped samples whose looks are unknown too
    look_class_unspecified: float = 0.3  # chance per view and class that its entry is the population mean
    medium_grey: float = 0.0           # arm D: chance per colour view that it becomes a grey print
    medium_toned: float = 0.0          # arm D: chance per colour view that it becomes a toned print

    def __post_init__(self):
        if not self.region_weight > 0:
            raise ValueError("region_weight must be positive (1: every latent cell counts the same)")
        if self.medium_grey + self.medium_toned > 1:
            raise ValueError("medium_grey + medium_toned is the chance that a colour view is converted: at most 1")
        for name in ("date_dropout", "camera_dropout", "prior_dropout", "prior_view_dropout", "look_dropout",
                     "look_class_unspecified", "medium_grey", "medium_toned"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be a probability in [0, 1]")
        if self.date_views not in DATE_VIEWS:
            raise ValueError(f"date_views must be one of {DATE_VIEWS}")
        if self.class_label < 0:
            raise ValueError("class_label must not be negative")
        if self.cross_date_pixels not in CROSS_DATE_PIXELS:
            raise ValueError(f"cross_date_pixels must be one of {CROSS_DATE_PIXELS}")
        if self.cross_date_gap < 0:
            raise ValueError("cross_date_gap must not be negative")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS} (fp16 without a gradient scaler is unsound)")


def autocast(precision: str, device_type: str = "cuda") -> torch.autocast:
    """The autocast context of a training precision: bf16 autocast, or autocast switched off for fp32."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if precision == "bf16":
        return torch.autocast(device_type, dtype=torch.bfloat16)
    return torch.autocast(device_type, enabled=False)


@torch.no_grad()
def encode_images(model: MVGen, images: torch.Tensor, sample_posterior: bool = True,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """VAE latents of the views, times the VAE's scaling factor.

    Args:
        images: [F, 3, H, W] in [-1, 1].
        sample_posterior: draw from the VAE's posterior as in training (with ``generator``, if given);
            False takes its mode, for reproducible checks.

    Returns:
        [1, F, 4, h, w] float32, on the VAE's device.
    """
    weight = next(model.vae.parameters())
    posterior = model.vae.encode(images.to(device=weight.device, dtype=weight.dtype)).latent_dist
    latents = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
    return (latents.float() * model.vae.config.scaling_factor)[None]


@dataclass
class Inputs:
    """The conditioning of one sample, as MVGenMaster's UNet takes it (see the module docstring)."""

    add_inputs: torch.Tensor                     # [1, F, 7, H, W]: the target mask, then 6 camera-ray channels
    coords: torch.Tensor | list[torch.Tensor]    # the 3D priors; "3dpe+pixel": [coordinates, warped pixels]
    latent_masks: torch.Tensor                   # [1, F, 1, h, w]: 0 for references, 1 for targets


def cross_date_views(date_mid: torch.Tensor, n_ref: int, gap: float, period_edges: Sequence[float] = ()
                     ) -> torch.Tensor:
    """Bool [F]: the targets that no reference shows as they looked on their date. References are never marked.

    A reference shows a target's state if its date midpoint is within ``gap`` years of the target's AND
    lies in the same period. Periods are separated by ``period_edges`` (decimal years; a date equal to
    an edge belongs to the later period, as in ``timemachine.data.holdout.era_index``). Without the
    second condition a target from May 2019 would count a reference from late 2018 as near, although
    the fire of April 2019 lies between them.

    ``period_edges`` are the run's ``ViewSamplingConfig.period_edges`` (``TrainConfig.sampling``; the
    scene carries them as ``MVGenScene.config.period_edges``). The edges are compared in the dates'
    precision (a sample's dates are float32), so a view dated on or after an edge never falls before it.
    """
    edges = torch.as_tensor(tuple(period_edges), dtype=date_mid.dtype, device=date_mid.device)
    period = torch.bucketize(date_mid, edges, right=True)
    far = (date_mid[n_ref:, None] - date_mid[None, :n_ref]).abs() > gap               # [targets, references]
    other_period = period[n_ref:, None] != period[None, :n_ref]
    marked = torch.zeros(len(date_mid), dtype=torch.bool, device=date_mid.device)
    marked[n_ref:] = (far | other_period).all(dim=1)
    return marked


def cross_date_drops(date_mid: torch.Tensor, n_ref: int, cfg: LossConfig, period_edges: Sequence[float] = ()
                     ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """What ``cfg.cross_date_pixels`` takes from a view set with these dates: ``(prior views, pixel views)``.

    Both are bool [F] masks of :func:`cross_date_views`, or None: the views that lose both priors
    (``"drop_all"``) and those that lose only their warped pixels (``"drop"``). They are the
    ``dropped_prior_views`` and ``dropped_pixel_views`` of :func:`build_inputs` and of the renderer
    (``timemachine.mvgen.probes.render_views``); the training step, the loss probe and the renderer
    all get them here.
    """
    if cfg.cross_date_pixels == "keep":
        return None, None
    views = cross_date_views(date_mid, n_ref, cfg.cross_date_gap, period_edges)
    return (views, None) if cfg.cross_date_pixels == "drop_all" else (None, views)


def depth_for_priors(depth: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """A private float64 copy of a sample's depth on ``device``, for MVGenMaster's prior code.

    ``get_3d_priors`` and the sampling pipeline call ``depth.to(device).float()`` once per prior and
    contract far depth in the result, in place. From a float64 tensor each call gets a fresh float32
    copy, so every prior is computed from depth contracted once, as in the authors' training, and
    the caller's tensor stays as it was (see "Depth" in the module docstring).
    """
    return depth.to(device=device, dtype=torch.float64, copy=True)


def look_embedding_size(model: MVGen) -> dict[str, float | None]:
    """How large the look's part was in the model's most recent UNet forward pass (models with a look).

    ``look_rms`` is the root mean square of what the look adds (``injection="temb"``: its embedding;
    ``"film"``: its scales and shifts together), ``look_temb_ratio`` its ratio to the ``temb`` it is added
    to (None for a FiLM look, which is not added to ``temb``), and ``sum_temb_ratio`` the ratio of the SUM
    of date and look embeddings to ``temb``: what every ResBlock's ``time_emb_proj`` reads on top of the
    timestep. All None if that pass had no dates.
    """
    stats = model.date_stats()
    if stats is None or "look_rms" not in stats:
        return {"look_rms": None, "look_temb_ratio": None, "sum_temb_ratio": None}
    temb = max(float(stats["temb_rms"]), 1e-12)
    film = model.date.look.injection == "film"
    return {"look_rms": float(stats["look_rms"]), "look_temb_ratio": None if film else float(stats["look_rms"]) / temb,
            "sum_temb_ratio": float(stats["sum_rms"]) / temb}


def date_embedding_size(model: MVGen) -> dict[str, float | None]:
    """How large the date embedding was in the model's most recent UNet forward pass.

    ``date_rms`` is the root mean square of the date embedding and ``date_temb_ratio`` its ratio to
    that of the ``temb`` (timestep plus domain embedding) it was added to
    (``DateConditioner.last_stats``). A ratio near or above 1 means the date is outgrowing the
    timestep signal that every ResBlock reads through the same vector. Both are None if that pass had
    no dates or the model has no date conditioner.
    """
    stats = model.date_stats()
    if stats is None:
        return {"date_rms": None, "date_temb_ratio": None}
    date_rms, temb_rms = float(stats["date_rms"]), float(stats["temb_rms"])
    return {"date_rms": date_rms, "date_temb_ratio": date_rms / max(temb_rms, 1e-12)}


def _view_counts(sample: dict) -> tuple[int, int]:
    """(F, n_ref) of a sample, after checking that it is a single unbatched view set."""
    images, n_ref = sample["images"], int(sample["n_ref"])
    if images.dim() != 4:
        raise ValueError(f"a sample holds one view set, images [F, 3, H, W]; got {tuple(images.shape)}")
    if not 0 < n_ref < images.shape[0]:
        raise ValueError(f"a sample needs references and targets; got n_ref = {n_ref} of {images.shape[0]} views")
    return images.shape[0], n_ref


def _zero_views(prior: torch.Tensor, views: torch.Tensor, n_ref: int) -> torch.Tensor:
    """``prior`` [F, C, h, w] with the views marked in ``views`` (bool [F]) set to zero; never a reference."""
    if views.dtype != torch.bool or views.shape != prior.shape[:1]:
        raise ValueError(f"a view mask must be bool [{prior.shape[0]}]; got {views.dtype} {tuple(views.shape)}")
    keep = ~views.to(prior.device)
    keep[:n_ref] = True
    return prior * keep.to(prior.dtype).view(-1, 1, 1, 1)


@torch.no_grad()
def build_inputs(model: MVGen, sample: dict, latents: torch.Tensor, *, drop_camera: bool = False,
                 drop_priors: bool = False, dropped_prior_views: torch.Tensor | None = None,
                 dropped_pixel_views: torch.Tensor | None = None) -> Inputs:
    """The masks, camera rays and 3D priors of a sample, with the requested parts zeroed.

    Args:
        sample: ``images`` [F, 3, H, W], ``K`` [F, 3, 3], ``w2c`` [F, 4, 4], ``depth`` [F, 1, H, W] and ``n_ref``
            (section 2.1 of the implementation spec; produced by ``timemachine.mvgen.data``). Not changed.
        latents: [1, F, 4, h, w] from :func:`encode_images`; the tensors are built on its device.
        drop_camera: all camera rays are zeros (the mask channel stays).
        drop_priors: all priors are zeros, the references' too (as in ``train.py``); the two masks below
            then change nothing.
        dropped_prior_views: bool [F], views whose priors are zeros (both tensors of "3dpe+pixel").
        dropped_pixel_views: bool [F], views whose warped-pixel prior is zeros; the coordinate prior is
            kept. Needs a prior type with warped pixels.

    In both masks the references' entries are ignored: a reference is only zeroed by ``drop_priors``.
    Far depth is contracted once for every prior (:func:`depth_for_priors`). No autocast context may
    be open around this function: the warp is a chain of matrix products that half precision moves.
    """
    f, n_ref = _view_counts(sample)
    device, dtype = latents.device, latents.dtype
    images = sample["images"].to(device)
    K, w2c = sample["K"].to(device), sample["w2c"].to(device)
    depth = depth_for_priors(sample["depth"], device)
    H, W = images.shape[-2:]

    masks = torch.ones((1, f, 1, H, W), device=device, dtype=dtype)
    latent_masks = torch.ones((1, f, 1, *latents.shape[-2:]), device=device, dtype=dtype)
    masks[:, :n_ref], latent_masks[:, :n_ref] = 0, 0
    camera = get_camera_embedding(K, w2c, 1, f, H, W, config=model.config).to(device=device, dtype=dtype)
    if drop_camera:
        camera = torch.zeros_like(camera)
    add_inputs = torch.cat([masks, camera], dim=2)

    prior_type = model.config.model_cfg.get("prior_type", "3dpe")
    if dropped_pixel_views is not None and not prior_type.endswith("pixel"):
        raise ValueError(f"prior type {prior_type!r} has no warped-pixel prior to drop")
    coords = get_3d_priors(model.config, depth, K, w2c, n_ref, nframe=f, device=device, colors=images,
                           latents=latents, vae=model.vae, prior_type=prior_type)
    priors = list(coords) if isinstance(coords, list) else [coords]                # the warped pixels come last
    if drop_priors:
        priors = [torch.zeros_like(prior) for prior in priors]
    else:
        if dropped_prior_views is not None:
            priors = [_zero_views(prior, dropped_prior_views, n_ref) for prior in priors]
        if dropped_pixel_views is not None:
            priors[-1] = _zero_views(priors[-1], dropped_pixel_views, n_ref)
    return Inputs(add_inputs=add_inputs, coords=priors if isinstance(coords, list) else priors[0],
                  latent_masks=latent_masks)


def predict(model: MVGen, noisy: torch.Tensor, timesteps: torch.Tensor, inputs: Inputs, class_label: int,
            year: torch.Tensor | None, valid: torch.Tensor | None, precision: str, *,
            look: torch.Tensor | None = None, look_known: torch.Tensor | None = None) -> torch.Tensor:
    """The UNet's output for one sample, run with the views' dates and under ``precision``'s autocast.

    Args:
        noisy: [1, F, 4, h, w] latents, the references clean.  timesteps: [1] long.
        year, valid: [F] the views' dates (decimal years) and which of them are known. ``year=None`` sets
            no dates at all, which gives the model without date conditioning.
        look, look_known: [F, 24] standardised look codes and [F] which of them are known, for a model with a
            look (``MVGen.dates``; ignored without one). None: every look unknown.

    Returns:
        [1, F, 4, h, w] float32, whatever the precision of the forward pass.
    """
    batch, f = noisy.shape[:2]
    device = noisy.device
    class_labels = torch.full((batch,), class_label, dtype=torch.long, device=device)
    if year is None:
        dates = nullcontext()
    elif look is None and look_known is None:
        dates = model.dates(year.to(device), None if valid is None else valid.to(device))
    else:
        dates = model.dates(year.to(device), None if valid is None else valid.to(device), look=look,
                            look_known=look_known)
    with dates, autocast(precision, device.type):
        pred = model.unet(noisy, timesteps, None, add_inputs=inputs.add_inputs, class_labels=class_labels,
                          coords=inputs.coords, return_dict=False)[0]
    return einops.rearrange(pred, "(b f) c h w -> b f c h w", f=f).float()


def v_prediction_loss(model: MVGen, latents: torch.Tensor, inputs: Inputs, timestep: int, noise: torch.Tensor,
                      year: torch.Tensor | None, valid: torch.Tensor | None, cfg: LossConfig, n_ref: int, *,
                      look: torch.Tensor | None = None, look_known: torch.Tensor | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """The v-prediction loss of one sample at a given timestep and noise.

    The targets are noised and the references put back clean, as in ``train.py``. ``year`` and ``valid``
    (and ``look``, ``look_known``) are passed to :func:`predict`; of ``cfg`` only ``class_label`` and
    ``precision`` are used. The scheduler must be a v-prediction one (``ValueError`` otherwise).

    Returns:
        ``(loss, per_view)``: the mean squared error over the target views (equal to
        ``F.mse_loss(pred[:, n_ref:], target[:, n_ref:])``), and that of each target view, [F - n_ref].
    """
    scheduler = model.scheduler
    if scheduler.config.prediction_type != "v_prediction":
        raise ValueError(f"the scheduler predicts {scheduler.config.prediction_type!r}, not the velocity")
    timesteps = torch.tensor([timestep], dtype=torch.long, device=latents.device)
    noise = noise.to(latents)
    noisy = scheduler.add_noise(latents, noise, timesteps)
    noisy = latents * (1 - inputs.latent_masks) + noisy * inputs.latent_masks
    target = scheduler.get_velocity(latents, noise, timesteps)
    pred = predict(model, noisy, timesteps, inputs, cfg.class_label, year, valid, cfg.precision, look=look,
                   look_known=look_known)
    per_view = (pred[0, n_ref:] - target[0, n_ref:].float()).pow(2).flatten(1).mean(1)
    return per_view.mean(), per_view


def v_prediction_error_map(model: MVGen, latents: torch.Tensor, inputs: Inputs, timestep: int, noise: torch.Tensor,
                           year: torch.Tensor | None, valid: torch.Tensor | None, cfg: LossConfig, n_ref: int, *,
                           look: torch.Tensor | None = None, look_known: torch.Tensor | None = None
                           ) -> torch.Tensor:
    """Where in the target views the error of :func:`v_prediction_loss` lies: ``[F - n_ref, h, w]``.

    One value per latent cell of each target view: the squared error between the UNet's output and
    the velocity target, averaged over the latent channels only. The arguments, the noising and the
    forward pass are those of :func:`v_prediction_loss`, so the mean of a view's map is that view's
    entry of ``per_view`` and the mean of the whole map is the loss (up to float rounding: the values
    are summed in another order; ``tests/test_mvgen_eval.py`` compares the two). It is for scores
    restricted to a part of the frame (``timemachine.mvgen.evaluation``); training does not call it.
    """
    scheduler = model.scheduler
    if scheduler.config.prediction_type != "v_prediction":
        raise ValueError(f"the scheduler predicts {scheduler.config.prediction_type!r}, not the velocity")
    timesteps = torch.tensor([timestep], dtype=torch.long, device=latents.device)
    noise = noise.to(latents)
    noisy = scheduler.add_noise(latents, noise, timesteps)
    noisy = latents * (1 - inputs.latent_masks) + noisy * inputs.latent_masks
    target = scheduler.get_velocity(latents, noise, timesteps)
    pred = predict(model, noisy, timesteps, inputs, cfg.class_label, year, valid, cfg.precision, look=look,
                   look_known=look_known)
    return (pred[0, n_ref:] - target[0, n_ref:].float()).pow(2).mean(dim=1)


def v_prediction_loss_and_map(model: MVGen, latents: torch.Tensor, inputs: Inputs, timestep: int, noise: torch.Tensor,
                              year: torch.Tensor | None, valid: torch.Tensor | None, cfg: LossConfig, n_ref: int, *,
                              look: torch.Tensor | None = None, look_known: torch.Tensor | None = None
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """The loss of :func:`v_prediction_loss` and the map of :func:`v_prediction_error_map` from ONE forward pass.

    For the callers that need the loss and where it lies: a training step with a region weight, and
    the date-loss probe on a scene that has changed-region weights. The arguments, the noising and
    the forward pass are those of the two functions. The loss is computed from the UNet's output
    with the operations of :func:`v_prediction_loss`, in their order, so it is that function's loss
    bit for bit and not the mean of the map, which differs from it by rounding
    (``tests/test_mvgen_signal.py`` compares them).

    Returns:
        ``(loss, error)``: the mean squared error over the target views, and the squared error of
        every latent cell of every target, averaged over the latent channels: ``[F - n_ref, h, w]``.
        Both carry the gradient.
    """
    scheduler = model.scheduler
    if scheduler.config.prediction_type != "v_prediction":
        raise ValueError(f"the scheduler predicts {scheduler.config.prediction_type!r}, not the velocity")
    timesteps = torch.tensor([timestep], dtype=torch.long, device=latents.device)
    noise = noise.to(latents)
    noisy = scheduler.add_noise(latents, noise, timesteps)
    noisy = latents * (1 - inputs.latent_masks) + noisy * inputs.latent_masks
    target = scheduler.get_velocity(latents, noise, timesteps)
    pred = predict(model, noisy, timesteps, inputs, cfg.class_label, year, valid, cfg.precision, look=look,
                   look_known=look_known)
    squared = (pred[0, n_ref:] - target[0, n_ref:].float()).pow(2)
    return squared.flatten(1).mean(1).mean(), squared.mean(dim=1)


def target_region(sample: dict, n_ref: int, error: torch.Tensor) -> torch.Tensor:
    """The changed-region weights of a sample's targets, for the error map ``error``: ``[F - n_ref, h, w]``.

    ``sample["region"]`` (``[F, 1, h, w]``, from ``MVGenScene.make_sample`` when the dataset folder has
    a regions file) says how much of each latent cell lies where the scene changed, in [0, 1]. The
    targets' part is returned on the device and in the dtype of ``error``. ``ValueError`` if the
    sample carries no such weights, or weights of another size than the map.
    """
    if "region" not in sample:
        raise ValueError("the sample carries no changed-region weights ('region'): the dataset folder has no "
                         "regions file (scripts/mvgen/prepare_regions.py writes it)")
    region = sample["region"][n_ref:, 0].to(device=error.device, dtype=error.dtype)
    if region.shape != error.shape:
        raise ValueError(f"the changed-region weights of the targets are {tuple(region.shape)}, the error map is "
                         f"{tuple(error.shape)}: the regions file was made for crops of another size")
    return region


def region_means(error: torch.Tensor, region: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """The plain mean of an error map inside the changed region and outside it: ``(inside, outside)``.

    ``error`` and ``region`` are ``[T, h, w]`` (:func:`v_prediction_loss_and_map`, :func:`target_region`).
    A cell counts inside with the share of it that lies in the region and outside with the rest,
    over the cells of all targets together (as ``timemachine.mvgen.evaluation.area_means`` does).
    An area that no cell has a part in is None: ``inside`` for targets that do not show the region.
    """
    def mean(weight: torch.Tensor) -> torch.Tensor | None:
        total = weight.sum()
        return (error * weight).sum() / total if total > 0 else None

    return mean(region), mean(1 - region)


def region_weighted_loss(error: torch.Tensor, region: torch.Tensor, weight: float) -> torch.Tensor:
    """The loss with the changed region counted ``weight`` times: ``sum(w * error) / sum(w)``.

    Every latent cell of every target has the weight ``w = 1 + (weight - 1) * region``: 1 outside the
    region, ``weight`` inside it, in between for a cell on its border. ``error`` and ``region`` are
    ``[T, h, w]``. Dividing by ``sum(w)`` keeps the loss a mean, so its size does not depend on how
    much of the targets the region covers; for targets without any region it is the plain mean of
    the map. The gradient of a cell's error is ``w / sum(w)`` instead of one over the number of cells.
    """
    weights = 1 + (weight - 1) * region
    return (weights * error).sum() / weights.sum()


def training_loss(model: MVGen, sample: dict, cfg: LossConfig, generator: torch.Generator | None = None,
                  period_edges: Sequence[float] = ()) -> tuple[torch.Tensor, dict]:
    """One stochastic loss sample: the VAE posterior draw, timestep, noise, dropout and dates are random.

    ``backward()`` may be called on the loss at any later time, also after other UNet forward passes:
    the date conditioner binds each pass's embedding to that pass (``timemachine.mvgen.date``).

    Args:
        sample: one view set (section 2.1 of the implementation spec): the keys of :func:`build_inputs` plus
            ``date`` [F, 2], the views' date intervals, and ``date_mid`` [F]; with a region weight other
            than 1 also ``region`` [F, 1, h, w], the views' changed-region weights (:func:`target_region`);
            for a model with a look also ``look`` [F, 24] and ``look_drop`` (``look.TrainingLooks``).
        generator: a CPU generator; everything random is drawn from it, so a seed fixes the step.
        period_edges: the years that separate the scene's periods, for the cross-date rule
            (:func:`cross_date_views`): the run's ``ViewSamplingConfig.period_edges``.

    Returns:
        ``(loss, info)``. ``info`` records what was drawn: ``timestep``, ``loss``, ``camera_dropped``,
        ``priors_dropped`` (all of them), ``prior_views_dropped`` and ``pixel_views_dropped`` (how many targets
        lost both priors, at random or as cross-date targets under ``"drop_all"``, or only their warped
        pixels; 0 when all priors were dropped), ``dates_known``, and the size of the date embedding in
        this forward pass, ``date_rms`` and ``date_temb_ratio`` (:func:`date_embedding_size`).

        With ``cfg.region_weight`` other than 1 the loss is the region-weighted one
        (:func:`region_weighted_loss`; "Region weight" in the module docstring) and ``info`` has three
        more entries, which are not trained on: ``loss_unweighted``, the plain loss of the same forward
        pass (what ``loss`` would be with a weight of 1), and ``loss_inside`` and ``loss_outside``, the
        plain mean error inside and outside the changed region (:func:`region_means`; ``loss_inside``
        is None when no target of the sample shows the region).

        For a model with a look ``info`` also has ``looks_known`` and the look's size in this forward pass,
        ``look_rms``, ``look_temb_ratio`` and ``sum_temb_ratio`` (:func:`look_embedding_size`).
    """
    f, n_ref = _view_counts(sample)
    is_ref = torch.arange(f) < n_ref
    latents = encode_images(model, sample["images"], generator=generator)
    timestep = int(torch.randint(model.scheduler.config.num_train_timesteps, (), generator=generator))
    noise = torch.randn(latents.shape, generator=generator)
    drop_camera = bool(torch.rand((), generator=generator) < cfg.camera_dropout)
    drop_priors = bool(torch.rand((), generator=generator) < cfg.prior_dropout)
    prior_views = (torch.rand(f, generator=generator) < cfg.prior_view_dropout) & ~is_ref
    year, valid = draw_dates(sample["date"].cpu(), cfg, generator)      # reads date_jitter and date_dropout
    dates_dropped = not bool(valid.any())
    valid = shown_dates(valid, is_ref, cfg.date_views)
    cross_prior_views, pixel_views = cross_date_drops(sample["date_mid"].cpu(), n_ref, cfg, period_edges)
    if cross_prior_views is not None:
        prior_views = prior_views | cross_prior_views

    looks = {}
    has_look = bool(getattr(model, "has_look", False))      # a stand-in without the attribute has no look
    if has_look:                        # tied to the date's dropout; nothing more is drawn (module docstring)
        if "look" not in sample:
            raise ValueError("the model has a look, but the sample carries no look input ('look'): the trainer makes "
                             "it with timemachine.mvgen.look.TrainingLooks")
        known = not (dates_dropped and bool(sample["look_drop"]))
        looks = {"look": sample["look"], "look_known": torch.full((f,), known, dtype=torch.bool)}

    inputs = build_inputs(model, sample, latents, drop_camera=drop_camera, drop_priors=drop_priors,
                          dropped_prior_views=prior_views, dropped_pixel_views=pixel_views)
    region_info = {}
    if cfg.region_weight == 1:
        loss, _ = v_prediction_loss(model, latents, inputs, timestep, noise, year, valid, cfg, n_ref, **looks)
    else:
        plain, error = v_prediction_loss_and_map(model, latents, inputs, timestep, noise, year, valid, cfg, n_ref,
                                                 **looks)
        region = target_region(sample, n_ref, error)
        loss = region_weighted_loss(error, region, cfg.region_weight)
        inside, outside = region_means(error.detach(), region)
        region_info = {"loss_unweighted": float(plain.detach()),
                       "loss_inside": None if inside is None else float(inside),
                       "loss_outside": None if outside is None else float(outside)}
    info = {"timestep": timestep, "loss": float(loss.detach()), "camera_dropped": drop_camera,
            "priors_dropped": drop_priors,
            "prior_views_dropped": 0 if drop_priors else int(prior_views.sum()),
            "pixel_views_dropped": 0 if drop_priors or pixel_views is None else int(pixel_views.sum()),
            "dates_known": bool(valid.any()), **date_embedding_size(model), **region_info}
    if has_look:
        info.update(looks_known=bool(looks["look_known"].any()), **look_embedding_size(model))
    return loss, info
