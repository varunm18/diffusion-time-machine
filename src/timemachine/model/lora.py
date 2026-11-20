"""Minimal LoRA (low-rank adapters) for SEVA's linear layers.

``add_lora`` replaces selected ``nn.Linear`` layers by :class:`LoRALinear`, which
computes ``base(x) + scale * up(down(x))``. ``up`` starts at zero, so a freshly
adapted model is exactly the original one. Only the adapters are trainable; the
base weights stay frozen (fp32, like the adapters).

Targets are matched by the *suffix* of the module's qualified name, e.g.
``"attn1.to_q"`` matches every multi-view self-attention query projection.
:data:`ATTENTION_TARGETS` covers the attention projections of SEVA's spatial
(``TransformerBlock``) and cross-frame (``TransformerBlockTimeMix``) blocks,
:data:`FEEDFORWARD_TARGETS` their MLPs. The cross-attention query and key
projections (``attn2.to_q`` / ``attn2.to_k``) are left out: SEVA cross-attends to a
single CLIP token, and a softmax over one key is 1 whatever the query and key are,
so those adapters have exactly zero gradient (64 layers, 11% of the LoRA; audit of
2026-11-19). Checkpoints trained with them still load (targets are stored per run).

Adapter weights are saved/loaded on their own (:func:`lora_state_dict`,
:func:`load_lora_state_dict`), keyed by the adapted layer's original name, so a
checkpoint holds only the new parameters.

MVGenMaster's fine-tuning (``timemachine.mvgen.model``) reuses this module with its
own targets. What a wrapped layer returns under bf16 autocast, and what that means
for comparing a LoRA run with the unadapted model, is in :class:`LoRALinear`.
"""
from __future__ import annotations

import math

import torch
from torch import nn

ATTENTION_TARGETS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_v", "attn2.to_out.0")
FEEDFORWARD_TARGETS = ("ff.net.0.proj", "ff.net.2", "ff_in.net.0.proj", "ff_in.net.2")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``rank`` update (zero at init).

    **Output dtype.** The update is cast to the *input's* dtype and added to ``base(x)``. Outside
    autocast that is the dtype of ``base(x)`` too, and a new layer returns ``base(x)`` bit for bit.

    Under bf16 autocast a bare ``nn.Linear`` returns bf16 whatever its input, and so do ``base`` and
    the adapter here. With a bf16 input the sum is bf16 as well. With an fp32 input the update is
    cast back to fp32 and the sum is taken in fp32, so the wrapped layer returns fp32 where the bare
    layer returned bf16. This is on purpose: bf16 keeps 8 significant bits, and in a bf16 sum an
    update below about 0.2% of ``base(x)`` would be rounded away.

    The price: wherever a wrapped layer has fp32 input, the layers after it compute from an fp32
    tensor instead of a bf16 one, so a bf16 LoRA run is not bit-identical to the unadapted model at
    step 0; the two differ by rounding. On a GPU this concerns the layers that follow a layer norm
    (autocast runs the norm in fp32 there; on a CPU it does not, so CPU tests of those layers stay
    exact: ``data/mvgen_review/impl2_results.json``, "learning-hazards" finding 4, derived from
    PyTorch's autocast rules on a CPU stand-in, not measured on a GPU). On any device it concerns
    MVGenMaster's ``time_emb_proj``, whose input is always fp32 (``tests/test_mvgen_model.py``).
    So compare a bf16 LoRA run with its own step-0 probe, not with the released model's output.
    In fp32 the wrapped model is exact at step 0.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapters are fp32; outside autocast, cast half-precision activations (if any) to match.
        lora_in = x if torch.is_autocast_enabled(x.device.type) else x.to(self.down.weight.dtype)
        # The update takes the input's dtype, not base(x)'s: under bf16 autocast an fp32 input keeps the sum in
        # fp32 (see the class docstring).
        return self.base(x) + (self.scale * self.up(self.down(lora_in))).to(x.dtype)


def _matches(name: str, targets: tuple[str, ...]) -> bool:
    return any(name == t or name.endswith("." + t) for t in targets)


def add_lora(model: nn.Module, targets: tuple[str, ...] = ATTENTION_TARGETS, rank: int = 32,
             alpha: float | None = None) -> list[str]:
    """Wrap every ``nn.Linear`` whose name ends with one of ``targets``. Returns the wrapped names."""
    wrapped = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear) and _matches(name, targets):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, LoRALinear(module, rank, alpha))
            wrapped.append(name)
    if not wrapped:
        raise ValueError(f"No nn.Linear matches {targets}")
    return wrapped


def lora_layers(model: nn.Module) -> dict[str, LoRALinear]:
    """Adapted layers by their original (pre-wrapping) qualified name."""
    return {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for layer in lora_layers(model).values() for p in (layer.down.weight, layer.up.weight)]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """``{"<layer>.down": ..., "<layer>.up": ...}`` for every adapted layer (CPU copies)."""
    out = {}
    for name, layer in lora_layers(model).items():
        out[f"{name}.down"] = layer.down.weight.detach().cpu().clone()
        out[f"{name}.up"] = layer.up.weight.detach().cpu().clone()
    return out


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Inverse of :func:`lora_state_dict`; the model must already have the same adapters."""
    layers = lora_layers(model)
    expected = {f"{n}.{part}" for n in layers for part in ("down", "up")}
    if set(state) != expected:
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        raise KeyError(f"LoRA state mismatch: missing {missing[:3]}..., unexpected {extra[:3]}...")
    with torch.no_grad():
        for name, layer in layers.items():
            layer.down.weight.copy_(state[f"{name}.down"])
            layer.up.weight.copy_(state[f"{name}.up"])
