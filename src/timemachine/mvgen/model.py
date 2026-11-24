"""MVGenMaster as a date-conditioned model that can be fine-tuned: loading, adapters, checkpoints.

* :func:`load_mvgen` loads the released model as MVGenMaster's ``run_mvgen.py`` does, except that
  the weights stay in fp32. They are released in fp16; training needs fp32 weights (the gradient is
  sound in fp32 and under bf16 autocast over fp32 weights, not in fp16: docs/mvgenmaster_time_plan.md,
  section 2.2). It refuses to load while TF32 matrix products are enabled
  (:func:`assert_exact_matmul`): they would round the warped priors that the model is conditioned on.
* :func:`prepare_for_finetuning` freezes the model and attaches what trains:
  the per-view date embedding (``date.py``; about 0.7M parameters) and, optionally, LoRA adapters on
  the attention and feed-forward layers of every transformer block (8.8M parameters at rank 32) and,
  if listed, on the ResBlock layers that read the date (:data:`TIME_PROJ_TARGET`; 1.5M more).
  Both start as exact no-ops, so at step 0 the model's output equals the released model's: bit for
  bit in fp32 and for a date-only run; a bf16 run with LoRA can differ by rounding (see
  ``LoRALinear`` in ``timemachine/model/lora.py``).
* A checkpoint (:func:`save_checkpoint`) holds only the trainable parameters and the
  :class:`FinetuneConfig` needed to rebuild the adapters; :func:`load_finetuned` puts them back on
  a freshly loaded model.

Like all of ``timemachine.mvgen``, this runs in MVGenMaster's environment with its checkout as the
working directory: the model's config names its files by relative paths.
"""
from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from easydict import EasyDict
from my_diffusers.models import UNet2DConditionModel
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
from my_diffusers.schedulers import DDPMScheduler
from omegaconf import OmegaConf
from src.modules.schedulers import get_diffusion_scheduler
from torch import nn

from timemachine.model.lora import add_lora, load_lora_state_dict, lora_parameters, lora_state_dict
from timemachine.mvgen.date import DateConditioner, DateEmbedding, temb_width

DEFAULT_MODEL_DIR = "check_points/pretrained_model"

# Every nn.Linear inside MVGenMaster's 16 transformer blocks (listed from the UNet built with the
# released config): the self-attention projections and the feed-forward layers, 96 layers in all.
# The released model has no cross-attention (``no_text_cross_attn``), so there is no ``attn2`` and
# no projection that cannot receive a gradient. Not covered, on purpose: the ``proj_in`` /
# ``proj_out`` linears around each block, the timestep MLP and the ResBlocks' ``time_emb_proj``
# (which can be added: ``TIME_PROJ_TARGET``).
DEFAULT_LORA_TARGETS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "ff.net.0.proj", "ff.net.2")

# The linear layer through which each ResBlock reads ``temb``: the only reader of the timestep, the
# domain and the date (22 layers in the released model; 1.5M LoRA parameters at rank 32). Not a default
# target. Listing it in ``FinetuneConfig.lora_targets`` lets the read-out of the date adapt, as it could
# when the authors trained their own extra condition (the domain label) with the whole UNet unfrozen.
# Its input is fp32 even under bf16 autocast, so its adapter always takes the fp32 sum described in
# ``LoRALinear``: with this target a bf16 run differs from the released model by rounding at step 0, on
# a CPU as well as on a GPU.
TIME_PROJ_TARGET = "time_emb_proj"

# Keys that ``save_checkpoint`` writes itself.
_CHECKPOINT_KEYS = ("finetune_config", "step", "trainable")


@dataclass(frozen=True)
class FinetuneConfig:
    """What trains on top of the frozen released model."""

    date: bool = True                   # attach the per-view date embedding
    date_freqs: int = 8                 # Fourier octaves of the date: periods of 200 / 2^k years, k < date_freqs
    date_hidden: int = 512              # width of the date MLP's hidden layer
    lora_rank: int = 0                  # 0: no LoRA
    lora_alpha: float | None = None     # None: alpha = rank (scale 1)
    # Name suffixes of the nn.Linear layers to adapt. TIME_PROJ_TARGET may be added to the default ones.
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    grad_checkpointing: bool = True

    def __post_init__(self):
        if isinstance(self.lora_targets, str):
            raise ValueError("lora_targets must be a sequence of names, not one string")
        object.__setattr__(self, "lora_targets", tuple(self.lora_targets))    # a list after a JSON round trip
        if self.date_freqs < 1 or self.date_hidden < 1:
            raise ValueError("date_freqs and date_hidden must be at least 1")
        if self.lora_rank < 0:
            raise ValueError("lora_rank must be 0 (no LoRA) or positive")
        if self.lora_alpha is not None and self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive (or None for alpha = rank)")
        if self.lora_rank > 0 and not self.lora_targets:
            raise ValueError("lora_rank > 0 needs lora_targets")


def adapter_settings(cfg: FinetuneConfig) -> dict:
    """What of a :class:`FinetuneConfig` decides which trainable weights a model has and what they compute.

    The date embedding's shape (if attached) and LoRA's rank, scale and targets (if attached); not
    gradient checkpointing, which changes how the gradient is computed and nothing else. Two runs
    with equal settings can exchange their trainable weights: this is what a run that starts from
    another run's checkpoint is checked by (``TrainConfig.init_from``).
    """
    settings: dict = {"date": cfg.date, "lora": cfg.lora_rank > 0}
    if cfg.date:
        settings.update(date_freqs=cfg.date_freqs, date_hidden=cfg.date_hidden)
    if cfg.lora_rank > 0:
        alpha = cfg.lora_rank if cfg.lora_alpha is None else cfg.lora_alpha         # None: alpha = rank
        settings.update(lora_rank=cfg.lora_rank, lora_alpha=float(alpha), lora_targets=sorted(cfg.lora_targets))
    return settings


@dataclass
class MVGen:
    """MVGenMaster's parts as training and sampling use them, plus the date conditioner once attached."""

    unet: UNet2DConditionModel
    vae: AutoencoderKL
    config: EasyDict                        # the released config.yaml
    scheduler: DDPMScheduler                # training noise schedule (v-prediction)
    date: DateConditioner | None = None     # attached by prepare_for_finetuning

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    def dates(self, year: torch.Tensor, valid: torch.Tensor | None = None, uncond: str = "same"
              ) -> AbstractContextManager[None]:
        """Context manager: condition the UNet forward passes inside it on these per-view dates.

        See :meth:`DateConditioner.dates`. Does nothing if the model has no date conditioner.
        """
        if self.date is None:
            return nullcontext()
        return self.date.dates(year, valid, uncond)

    def date_stats(self) -> dict[str, float] | None:
        """Sizes in the most recent UNet forward pass: ``{"date_rms": ..., "temb_rms": ...}``.

        See :attr:`DateConditioner.last_stats`. ``None`` if that pass had no dates, or if the model
        has no date conditioner.
        """
        return None if self.date is None else self.date.last_stats

    def sampling_pipeline(self) -> StableDiffusionMultiViewPipeline:
        """MVGenMaster's sampling pipeline around this very ``unet`` and ``vae`` (fp32), with a DDIM scheduler.

        Adapters and hooks therefore apply while sampling. To render at given dates, call the pipeline
        inside ``self.dates(...)``; with guidance it doubles the batch, unconditional half first.
        """
        return StableDiffusionMultiViewPipeline(
            vae=self.vae, unet=self.unet, scheduler=get_diffusion_scheduler(self.config, name="DDIM"),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False)


def assert_exact_matmul() -> None:
    """Raise ``AssertionError`` if fp32 matrix products on the GPU may be computed in TF32.

    MVGenMaster warps the references' points into the target views with plain fp32 matrix products
    (``src/modules/position_encoding.py``). TF32 rounds the inputs of such products to a 10-bit
    mantissa, which is fp16's, so the warped priors would not be the ones the model was trained on.
    TF32 itself was not measured; under fp16 autocast, which rounds the same products as coarsely,
    a quarter of the warped pixels changed (``data/mvgen_fidelity/audit_results.json``, the verdict
    on priors computed under autocast).

    PyTorch leaves TF32 off for matrix products by default and the authors' config keeps it off.
    ``torch.backends.cuda.matmul.allow_tf32 = True`` and ``torch.set_float32_matmul_precision`` with
    ``"high"`` or ``"medium"`` both turn it on, and both show in the flag tested here.

    :func:`load_mvgen` calls this; scripts call it at start-up as well, to fail before any work is done.
    """
    if torch.backends.cuda.matmul.allow_tf32:
        raise AssertionError(
            "TF32 matrix products are enabled (torch.backends.cuda.matmul.allow_tf32 is True, or "
            "torch.set_float32_matmul_precision was set below 'highest'). MVGenMaster's prior warp is plain fp32 "
            "matmul and TF32 would round it as coarsely as fp16, so the priors would not be the ones the model "
            "was trained on. Leave TF32 off for MVGenMaster.")


def load_mvgen(model_dir: str = DEFAULT_MODEL_DIR, device: str = "cuda") -> MVGen:
    """The released MVGenMaster with fp32 weights, frozen, in eval mode (no date conditioner yet).

    Raises ``AssertionError`` before loading anything if TF32 matrix products are enabled
    (:func:`assert_exact_matmul`).
    """
    assert_exact_matmul()
    config = EasyDict(OmegaConf.load(f"{model_dir}/config.yaml"))
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet", rank=0,
                                                model_cfg=config.model_cfg, low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True, local_files_only=True)
    # fp16 tensors, copied into the UNet's fp32 parameters
    unet.load_state_dict(torch.load(f"{model_dir}/ema_unet.pt", map_location="cpu", weights_only=True))
    vae.requires_grad_(False).eval().to(device)
    unet.requires_grad_(False).eval().to(device)
    return MVGen(unet=unet, vae=vae, config=config, scheduler=get_diffusion_scheduler(config, name="DDPM"))


def prepare_for_finetuning(model: MVGen, cfg: FinetuneConfig) -> dict[str, list[nn.Parameter]]:
    """Freeze the model, attach the date conditioner and/or LoRA, enable gradient checkpointing.

    Returns the trainable parameters in the groups ``"date"`` and ``"lora"`` (empty when not attached).

    MVGenMaster's blocks only checkpoint in training mode, so call ``model.unet.train()`` before
    training (its dropout rates are zero: the mode changes nothing else). Checkpointing runs the
    blocks a second time during ``backward()``; every forward pass keeps its own date embedding for
    that re-run (see ``date.py``), so ``backward()`` may run after the ``dates()`` block has been
    left, and other forward passes (a probe, a second loss term) may run in between.

    The conditioner is attached before LoRA. Nothing depends on that order, also when
    :data:`TIME_PROJ_TARGET` is among ``cfg.lora_targets``.
    """
    if model.date is not None:
        raise ValueError("the model already has a date conditioner: prepare a freshly loaded model")
    model.unet.requires_grad_(False)
    model.vae.requires_grad_(False)
    groups: dict[str, list[nn.Parameter]] = {"date": [], "lora": []}
    if cfg.date:
        embedding = DateEmbedding(temb_width(model.unet), cfg.date_freqs, cfg.date_hidden).to(model.device)
        model.date = DateConditioner(model.unet, embedding)
        groups["date"] = list(model.date.parameters())
    if cfg.lora_rank > 0:
        add_lora(model.unet, cfg.lora_targets, cfg.lora_rank, cfg.lora_alpha)
        groups["lora"] = lora_parameters(model.unet)
    if cfg.grad_checkpointing:
        model.unet.enable_gradient_checkpointing()
    return groups


def trainable_state_dict(model: MVGen) -> dict[str, dict[str, torch.Tensor]]:
    """CPU copies of everything that trains: ``{"date": ..., "lora": ...}`` (empty dicts when not attached)."""
    date = {} if model.date is None else {k: v.detach().cpu().clone() for k, v in model.date.state_dict().items()}
    return {"date": date, "lora": lora_state_dict(model.unet)}


def load_trainable_state_dict(model: MVGen, state: dict[str, dict[str, torch.Tensor]]) -> None:
    """Inverse of :func:`trainable_state_dict`; the model must already have the same adapters."""
    if model.date is not None:
        model.date.load_state_dict(state["date"])
    elif state["date"]:
        raise ValueError("the state holds a date embedding but the model has no date conditioner")
    load_lora_state_dict(model.unet, state["lora"])


def save_checkpoint(path: str | Path, model: MVGen, cfg: FinetuneConfig, step: int, extra: dict | None = None) -> None:
    """Write the trainable parameters, ``cfg`` and ``step`` (plus ``extra`` entries) to ``path``."""
    extra = extra or {}
    clash = sorted(set(extra) & set(_CHECKPOINT_KEYS))
    if clash:
        raise ValueError(f"extra may not redefine {clash}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"finetune_config": asdict(cfg), "step": step, "trainable": trainable_state_dict(model), **extra}
    tmp = path.with_name(path.name + ".tmp")        # write, then rename: a job killed mid-save keeps the old file
    torch.save(payload, tmp)
    tmp.replace(path)


def load_finetuned(checkpoint: str | Path, model_dir: str = DEFAULT_MODEL_DIR, device: str = "cuda"
                   ) -> tuple[MVGen, dict]:
    """The released model with a checkpoint's adapters attached and loaded (eval mode). Also returns the checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = load_mvgen(model_dir, device)
    prepare_for_finetuning(model, FinetuneConfig(**ckpt["finetune_config"]))
    load_trainable_state_dict(model, ckpt["trainable"])
    return model, ckpt
