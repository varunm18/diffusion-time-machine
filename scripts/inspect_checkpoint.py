#!/usr/bin/env python
"""Size of a checkpoint's date embedding, date FiLM and LoRA updates relative to SEVA (CPU, light).

Reads only the tensors it needs from SEVA's safetensors file (memory-mapped),
so it runs anywhere in seconds. See ``timemachine.model.diagnostics``.

    python scripts/inspect_checkpoint.py data/experiments/train/<run>/checkpoints/step_002500.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from timemachine.model.date_embedding import DateEmbedding
from timemachine.model.date_film import DateEncoder
from timemachine.model.diagnostics import (
    date_embedding_rms, date_film_stats, lora_relative_update, summarize_lora, timestep_embedding_rms,
)
from timemachine.model.loading import SEVA_REPO, default_hub, snapshot_dir


class _FiLMLayer:
    """CPU stand-in for a DateFiLMConv: just its per-block linear layer."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        self.film = nn.Linear(weight.shape[1], weight.shape[0])
        self.film.load_state_dict({"weight": weight.float(), "bias": bias.float()})

    def scale_shift(self, features):
        return self.film(features).chunk(2, dim=1)


def film_stats(state: dict | None, dim: int) -> dict:
    """Date-FiLM scale/shift sizes of a checkpoint (empty for checkpoints without FiLM)."""
    if not state:
        return {}
    encoder = DateEncoder(dim)
    encoder.load_state_dict(state["encoder"])
    names = sorted({k.rsplit(".", 1)[0] for k in state["layers"]})
    layers = {n: _FiLMLayer(state["layers"][f"{n}.weight"], state["layers"][f"{n}.bias"]) for n in names}
    return date_film_stats(encoder, layers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--hub", type=Path, default=None, help="HF hub cache (default: TIMEMACHINE_HF_HUB)")
    args = parser.parse_args()

    weights = snapshot_dir(args.hub or default_hub(), SEVA_REPO) / "modelv1.1.safetensors"
    with safe_open(str(weights), "pt") as f:
        time_embed = nn.Sequential(nn.Linear(320, 1280), nn.SiLU(), nn.Linear(1280, 1280))
        time_embed.load_state_dict({k.removeprefix("time_embed."): f.get_tensor(k).float()
                                    for k in f.keys() if k.startswith("time_embed.")})
        t_rms = timestep_embedding_rms(time_embed)
        for path in args.checkpoints:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ckpt["finetune_config"]
            date = DateEmbedding(max_abs=cfg.get("date_max_abs"))
            date.load_state_dict(ckpt["trainable"]["date"])
            stats = {"checkpoint": str(path), "step": ckpt["step"], "timestep_rms": t_rms, **date_embedding_rms(date)}
            stats.update(film_stats(ckpt["trainable"].get("film"), cfg.get("film_dim", 256)))
            lora = ckpt["trainable"]["lora"]
            scale = (cfg["lora_alpha"] or cfg["lora_rank"]) / max(cfg["lora_rank"], 1)
            layers = sorted({k.rsplit(".", 1)[0] for k in lora})
            updates = {name: lora_relative_update(lora[f"{name}.down"], lora[f"{name}.up"], scale,
                                                  f.get_tensor(f"{name}.weight"))
                       for name in layers}
            stats.update(summarize_lora(updates))
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}, indent=1))


if __name__ == "__main__":
    main()
