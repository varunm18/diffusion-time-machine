"""How big is what fine-tuning added, compared with what SEVA already had?

Two numbers matter for stability:

* **Date embedding vs timestep embedding.** The date embedding is *added* to
  SEVA's timestep embedding, which is small: its per-dimension RMS is ~0.1-0.4,
  and noise levels 100 steps apart differ by only ~0.05-0.1 RMS. A date
  embedding much larger than that drowns the noise-level signal, and the
  model stops denoising properly at every level.
* **LoRA update vs base weight.** ``||scale * up @ down|| / ||W||`` per adapted
  layer, grouped by layer kind; values approaching 1 mean the adapter has
  replaced, not adjusted, the layer.
* **Date FiLM scale and shift** (``date_film.py``): RMS over channels of the
  per-block scale and shift at a few years, max over blocks. A scale of 1 doubles
  (or, at -1, erases) a channel. "spread" is how much they differ between years.

The functions work on plain tensors, so they serve both a live model
(:func:`finetune_stats`) and a saved checkpoint (``scripts/inspect_checkpoint.py``).
"""
from __future__ import annotations

import re

import numpy as np
import torch
from torch import nn

from timemachine.model.date_embedding import DateEmbedding

PROBE_YEARS = (1850.0, 1900.0, 1960.0, 2015.0, 2021.0)
PROBE_TIMESTEPS = (0, 250, 500, 750, 999)


def rms(x: torch.Tensor) -> float:
    return float(x.float().pow(2).mean().sqrt())


@torch.no_grad()
def timestep_embedding_rms(time_embed: nn.Module, model_channels: int = 320) -> float:
    """Mean per-dimension RMS of SEVA's timestep embedding over a few noise levels."""
    from seva.modules.layers import timestep_embedding

    t = torch.tensor(PROBE_TIMESTEPS, device=next(time_embed.parameters()).device)
    feats = timestep_embedding(t, model_channels).to(next(time_embed.parameters()).dtype)
    return float(np.mean([rms(e) for e in time_embed(feats)]))


@torch.no_grad()
def date_embedding_rms(date_embedding: DateEmbedding) -> dict[str, float]:
    """Per-dimension RMS of the date embedding at a few years, and for "unknown"."""
    device = next(date_embedding.parameters()).device
    years = torch.tensor(PROBE_YEARS, device=device)
    known = date_embedding(years, torch.ones_like(years, dtype=torch.bool))
    unknown = date_embedding(years[:1], torch.zeros(1, dtype=torch.bool, device=device))
    out = {f"date_rms_{int(y)}": rms(e) for y, e in zip(PROBE_YEARS, known)}
    out["date_rms_unknown"] = rms(unknown)
    out["date_rms_spread"] = rms(known - known.mean(0, keepdim=True))   # how much years differ
    return out


def layer_kind(name: str) -> str:
    """Group adapted layers: spatial vs cross-frame (time-mix) block, and attention/MLP type."""
    block = "timemix" if "time_mix_blocks" in name else "spatial"
    match = re.search(r"\.(attn1|attn2|ff_in|ff)\.", name + ".")
    return f"{block}.{match.group(1) if match else 'other'}"


@torch.no_grad()
def lora_relative_update(down: torch.Tensor, up: torch.Tensor, scale: float, base: torch.Tensor) -> float:
    delta = scale * up.float() @ down.float()
    return float(delta.norm() / base.float().norm().clamp_min(1e-12))


def summarize_lora(updates: dict[str, float]) -> dict[str, float]:
    """Mean and max relative update per layer kind."""
    kinds: dict[str, list[float]] = {}
    for name, value in updates.items():
        kinds.setdefault(layer_kind(name), []).append(value)
    out = {}
    for kind, values in sorted(kinds.items()):
        out[f"lora_rel_{kind}_mean"] = float(np.mean(values))
        out[f"lora_rel_{kind}_max"] = float(np.max(values))
    if updates:
        out["lora_rel_max"] = float(max(updates.values()))
    return out


@torch.no_grad()
def date_film_stats(encoder: nn.Module, layers: dict) -> dict[str, float]:
    """Size of the date-FiLM scale and shift (max over blocks) and how much they differ between years."""
    if not layers:
        return {}
    device = next(encoder.parameters()).device
    years = torch.tensor(PROBE_YEARS, device=device)
    features = encoder(torch.cat([years, years[:1]]),
                       torch.tensor([True] * len(years) + [False], device=device))
    out = {"film_scale_rms_max": 0.0, "film_shift_rms_max": 0.0, "film_spread_max": 0.0}
    for layer in layers.values():
        scale, shift = layer.scale_shift(features)
        known = torch.cat([scale, shift], dim=1)[:-1]
        out["film_scale_rms_max"] = max(out["film_scale_rms_max"], rms(scale))
        out["film_shift_rms_max"] = max(out["film_shift_rms_max"], rms(shift))
        out["film_spread_max"] = max(out["film_spread_max"], rms(known - known.mean(0, keepdim=True)))
    return out


def finetune_stats(model) -> dict[str, float]:
    """Diagnostics for a live :class:`~timemachine.model.dated_seva.DatedSGMWrapper`."""
    from timemachine.model.date_film import date_film_layers
    from timemachine.model.lora import lora_layers

    hook = model.date_module
    stats = {"timestep_rms": timestep_embedding_rms(hook.time_embed, model.module.model_channels),
             **date_embedding_rms(hook.date_embedding)}
    if hook.film_encoder is not None:
        stats.update(date_film_stats(hook.film_encoder, date_film_layers(model.module)))
    updates = {name: lora_relative_update(layer.down.weight, layer.up.weight, layer.scale, layer.base.weight)
               for name, layer in lora_layers(model.module).items()}
    return {**stats, **summarize_lora(updates)}
