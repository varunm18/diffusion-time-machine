"""Turn date-conditioned SEVA into a fine-tunable model, and save/restore what it learned.

Fine-tuning recipe (first version, for the single-scene overfit):

* SEVA's 1.26B weights are frozen (bf16).
* Trainable: how the date gets in (``date_injection``), plus optional LoRA
  adapters (zero-init) on the attention and feed-forward layers of every
  multi-view transformer block. At step 0 the model is exactly SEVA.
  - ``"film"`` (default): a per-block date scale-and-shift (``date_film.py``).
  - ``"timestep"``: a date vector added to the timestep embedding
    (``date_embedding.py``); runs before 2026-11-03 used this.
* Gradient checkpointing on all UNet blocks.

A checkpoint stores only the trainable parameters plus the :class:`FinetuneConfig`
needed to rebuild the adapters (tens of MB instead of SEVA's 5 GB).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from timemachine.model.date_film import date_film_state_dict, load_date_film_state_dict
from timemachine.model.dated_seva import DatedSGMWrapper
from timemachine.model.grad_checkpoint import enable_gradient_checkpointing
from timemachine.model.loading import SevaComponents, load_components
from timemachine.model.lora import (
    ATTENTION_TARGETS, FEEDFORWARD_TARGETS, add_lora, load_lora_state_dict, lora_parameters, lora_state_dict,
)


@dataclass(frozen=True)
class FinetuneConfig:
    lora_rank: int = 32                 # 0 disables LoRA (date module only)
    lora_alpha: float | None = None     # None: alpha = rank (scale 1)
    lora_targets: tuple[str, ...] = ATTENTION_TARGETS + FEEDFORWARD_TARGETS
    grad_checkpointing: bool = True
    date_max_abs: float | None = None   # soft bound per date-embedding dimension (see DateEmbedding)
    date_injection: str = "film"        # one of DATE_INJECTIONS (see module docstring)
    film_dim: int = 256                 # width of the shared date features read by every FiLM layer

    def __post_init__(self):
        if self.date_injection not in DATE_INJECTIONS:
            raise ValueError(f"date_injection must be one of {DATE_INJECTIONS}")


DATE_INJECTIONS = ("film", "timestep")


def prepare_for_finetuning(model: DatedSGMWrapper, cfg: FinetuneConfig) -> dict[str, list[nn.Parameter]]:
    """Freeze SEVA, add the date path and adapters, enable checkpointing. Returns the trainable parameter groups.

    Groups: ``"date"`` (timestep-embedding date vector), ``"film"`` (date FiLM), ``"lora"``;
    only the chosen date path trains, the other stays at zero.
    """
    model.module.requires_grad_(False)
    model.date_module.date_embedding.max_abs = cfg.date_max_abs
    groups = {"date": [], "film": [], "lora": []}
    if cfg.date_injection == "timestep":
        groups["date"] = model.date_parameters()
    else:
        model.add_date_film(cfg.film_dim)
        groups["film"] = model.film_parameters()
    for p in groups["date"] + groups["film"]:
        p.requires_grad_(True)
    if cfg.lora_rank > 0:
        add_lora(model.module, tuple(cfg.lora_targets), cfg.lora_rank, cfg.lora_alpha)
        groups["lora"] = lora_parameters(model.module)
    if cfg.grad_checkpointing:
        enable_gradient_checkpointing(model.module)
    return groups


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def trainable_state_dict(model: DatedSGMWrapper) -> dict[str, dict]:
    state = {"date": _cpu_state(model.date_module.date_embedding), "lora": lora_state_dict(model.module)}
    if model.date_module.film_encoder is not None:
        state["film"] = {"encoder": _cpu_state(model.date_module.film_encoder),
                         "layers": date_film_state_dict(model.module)}
    return state


def load_trainable_state_dict(model: DatedSGMWrapper, state: dict) -> None:
    model.date_module.date_embedding.load_state_dict(state["date"])
    if state["lora"]:
        load_lora_state_dict(model.module, state["lora"])
    if state.get("film"):
        model.date_module.film_encoder.load_state_dict(state["film"]["encoder"])
        load_date_film_state_dict(model.module, state["film"]["layers"])


def save_checkpoint(path: Path, model: DatedSGMWrapper, cfg: FinetuneConfig, step: int,
                    extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"finetune_config": asdict(cfg), "step": step, "trainable": trainable_state_dict(model)}
    torch.save({**payload, **(extra or {})}, path)


def load_finetuned(checkpoint: Path, hub: Path | None = None, device: str = "cuda") -> tuple[SevaComponents, dict]:
    """SEVA components with a trained date module + adapters (eval mode). Also returns the raw checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = FinetuneConfig(**{"date_injection": "timestep",          # checkpoints from before date FiLM
                            **ckpt["finetune_config"],
                            "lora_targets": tuple(ckpt["finetune_config"]["lora_targets"])})
    components = load_components(hub, device=device, dated=True)
    prepare_for_finetuning(components.model, cfg)
    load_trainable_state_dict(components.model, ckpt["trainable"])
    components.model.to(device).eval()
    return components, ckpt
