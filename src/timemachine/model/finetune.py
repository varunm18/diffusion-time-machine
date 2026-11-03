"""Turn date-conditioned SEVA into a fine-tunable model, and save/restore what it learned.

Fine-tuning recipe (first version, for the single-scene overfit):

* SEVA's 1.26B weights are frozen (bf16).
* Trainable: the date module (``DateEmbedding``, zero-init) plus LoRA adapters
  (zero-init) on the attention and feed-forward layers of every multi-view
  transformer block. At step 0 the model is exactly SEVA.
* Gradient checkpointing on all UNet blocks.

A checkpoint stores only the trainable parameters plus the :class:`FinetuneConfig`
needed to rebuild the adapters (tens of MB instead of SEVA's 5 GB).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

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


def prepare_for_finetuning(model: DatedSGMWrapper, cfg: FinetuneConfig) -> dict[str, list[nn.Parameter]]:
    """Freeze SEVA, add adapters, enable checkpointing. Returns the trainable parameter groups."""
    model.module.requires_grad_(False)
    date_params = model.date_parameters()
    for p in date_params:
        p.requires_grad_(True)
    groups = {"date": date_params, "lora": []}
    if cfg.lora_rank > 0:
        add_lora(model.module, tuple(cfg.lora_targets), cfg.lora_rank, cfg.lora_alpha)
        groups["lora"] = lora_parameters(model.module)
    if cfg.grad_checkpointing:
        enable_gradient_checkpointing(model.module)
    return groups


def trainable_state_dict(model: DatedSGMWrapper) -> dict[str, dict[str, torch.Tensor]]:
    date = {k: v.detach().cpu().clone() for k, v in model.date_module.date_embedding.state_dict().items()}
    return {"date": date, "lora": lora_state_dict(model.module)}


def load_trainable_state_dict(model: DatedSGMWrapper, state: dict) -> None:
    model.date_module.date_embedding.load_state_dict(state["date"])
    if state["lora"]:
        load_lora_state_dict(model.module, state["lora"])


def save_checkpoint(path: Path, model: DatedSGMWrapper, cfg: FinetuneConfig, step: int,
                    extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"finetune_config": asdict(cfg), "step": step, "trainable": trainable_state_dict(model)}
    torch.save({**payload, **(extra or {})}, path)


def load_finetuned(checkpoint: Path, hub: Path | None = None, device: str = "cuda") -> tuple[SevaComponents, dict]:
    """SEVA components with a trained date module + adapters (eval mode). Also returns the raw checkpoint."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = FinetuneConfig(**{**ckpt["finetune_config"],
                            "lora_targets": tuple(ckpt["finetune_config"]["lora_targets"])})
    components = load_components(hub, device=device, dated=True)
    prepare_for_finetuning(components.model, cfg)
    load_trainable_state_dict(components.model, ckpt["trainable"])
    components.model.to(device).eval()
    return components, ckpt
