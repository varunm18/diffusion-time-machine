"""Appearance FiLM: a per-view scale and shift from *how the photograph looks*, in every ResBlock.

Why this exists. We measured that our capture dates are largely readable from
the photographic medium alone: across four scenes, a linear probe on colour,
tone and grain separates pre-1945 from modern photos at AUC 0.89, and
sepia-toning a modern photo makes that probe call it old (``docs/findings.md``,
:mod:`timemachine.analysis.shortcut`). So "1890" and "brown, flat, grainy" are
nearly the same signal in our training data. A model asked to condition on the
date can satisfy the loss by learning the film stock, which is what our runs
keep showing: the LoRA learned one sepia cast and applied it at every date, and
swapping a target's date costs ~0.

This module gives the model somewhere else to put that. Each view carries an
explicit **appearance** vector (the same statistics the probe used), conditioned
on exactly like the date is, so lighting, tone and medium have their own
pathway. The intended effect is that the date pathway is left with the residual
- what actually changed about the place - rather than the look of the print. The
idea is WildCAT3D's (a learned per-view appearance code for in-the-wild photos);
what is new here is deriving it from the cues we measured to be the confound, so
the date cannot hide inside it.

Mechanically this mirrors :mod:`timemachine.model.date_film` exactly::

    stats -> AppearanceEncoder (shared) -> per-block Linear -> (scale, shift)
    h = h * (1 + scale) + shift,  then whatever came next in the block

* Per-block layers start at zero, so a fresh model is exactly the model it wrapped.
* It **nests**: if date FiLM is already installed, this wraps it rather than
  replacing it, so a block applies appearance, then date, then its conv. Nothing
  in ``date_film.py`` or in SEVA changes.
* Features are computed once per forward pass and stored in a shared
  :class:`AppearanceState`, so the blocks' recomputation under gradient
  checkpointing sees the same tensor (same contract as ``DateState``).

Appearance is given for inputs *and* targets, as WildCAT3D does: at training
time the target's own look is known, and at sampling time it becomes a control
("render 1890, as a modern colour photo" versus "as an albumen print").
"""
from __future__ import annotations

import torch
from torch import nn

from timemachine.data.appearance import N_APPEARANCE_FEATURES  # noqa: F401  (re-exported)


class AppearanceState:
    """Plain holder (not a module) for the current forward pass's per-view appearance features."""

    def __init__(self):
        self.features: torch.Tensor | None = None


class AppearanceEncoder(nn.Module):
    """(N, F) raw statistics -> (N, dim) features shared by all FiLM layers.

    The raw statistics are on wildly different scales (a saturation in [0, 1]
    next to a Lab chroma spread in the tens), so they are standardized by
    buffers that travel with the checkpoint. Pass the training set's statistics
    via :meth:`fit_normalization`; without them the module falls back to unit
    scaling, which trains but more slowly.

    Views whose appearance is dropped during training get a learned "appearance
    unknown" vector instead, so the model also learns to run without it - needed
    for classifier-free guidance and for sampling when no appearance has been
    chosen.
    """

    def __init__(self, n_features: int = N_APPEARANCE_FEATURES, dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.n_features = n_features
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("scale", torch.ones(n_features))
        self.unknown = nn.Parameter(torch.randn(n_features) * 0.5)
        self.mlp = nn.Sequential(nn.Linear(n_features, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, dim), nn.SiLU())

    @torch.no_grad()
    def fit_normalization(self, stats: torch.Tensor) -> "AppearanceEncoder":
        """Set the standardization buffers from a sample of training statistics."""
        stats = torch.as_tensor(stats, dtype=torch.float32)
        if stats.ndim != 2 or stats.shape[1] != self.n_features:
            raise ValueError(f"Expected (N, {self.n_features}) statistics, got {tuple(stats.shape)}")
        scale = stats.std(dim=0)
        scale[scale < 1e-6] = 1.0          # a statistic that never varies carries nothing
        self.mean.copy_(stats.mean(dim=0))
        self.scale.copy_(scale)
        return self

    def forward(self, stats: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        z = (stats.float() - self.mean) / self.scale
        if valid is not None:
            z = torch.where(valid[:, None].bool(), z, self.unknown.to(z.dtype).expand_as(z))
        return self.mlp(z)


def _in_channels(module: nn.Module) -> int:
    """Input channel count of a conv, or of whatever conv-like module wraps one."""
    if isinstance(module, nn.Conv2d):
        return module.in_channels
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child.in_channels
    raise ValueError(f"No Conv2d found inside {type(module).__name__}")


class AppearanceFiLMConv(nn.Module):
    """Scale-and-shift a block's input channels by its view's appearance, then run what was there.

    ``inner`` is whatever already occupied the slot: SEVA's raw conv, or a
    :class:`~timemachine.model.date_film.DateFiLMConv` if date FiLM is installed.
    Nesting keeps both conditionings live and leaves the other module untouched.
    """

    def __init__(self, inner: nn.Module, state: AppearanceState, feature_dim: int):
        super().__init__()
        self.inner = inner
        channels = _in_channels(inner)
        device = next(inner.parameters()).device
        self.film = nn.Linear(feature_dim, 2 * channels, device=device)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self._state = state                     # plain attribute: shared, not a submodule

    def scale_shift(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, feature_dim) -> per-channel ``(scale, shift)``, each (N, C)."""
        return self.film(features).chunk(2, dim=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        features = self._state.features
        if features is not None:
            if features.shape[0] != h.shape[0]:
                raise ValueError(f"{h.shape[0]} views vs {features.shape[0]} appearance features")
            scale, shift = self.scale_shift(features)
            h = h * (1 + scale[:, :, None, None].to(h.dtype)) + shift[:, :, None, None].to(h.dtype)
        return self.inner(h)


def add_appearance_film(seva: nn.Module, state: AppearanceState, feature_dim: int) -> list[str]:
    """Wrap the first conv (or existing date FiLM) of every SEVA ResBlock. Returns their names."""
    from seva.modules.layers import ResBlock

    wrapped = []
    for name, block in seva.named_modules():
        if isinstance(block, ResBlock) and not isinstance(block.in_layers[-1], AppearanceFiLMConv):
            block.in_layers[-1] = AppearanceFiLMConv(block.in_layers[-1], state, feature_dim)
            wrapped.append(name)
    if not wrapped:
        raise ValueError("No SEVA ResBlock found")
    return wrapped


def appearance_film_layers(seva: nn.Module) -> dict[str, AppearanceFiLMConv]:
    return {name: m for name, m in seva.named_modules() if isinstance(m, AppearanceFiLMConv)}


def appearance_film_state_dict(seva: nn.Module) -> dict[str, torch.Tensor]:
    """``{"<layer>.weight": ..., "<layer>.bias": ...}`` of every per-block FiLM layer (CPU copies)."""
    return {f"{name}.{k}": v.detach().cpu().clone()
            for name, layer in appearance_film_layers(seva).items()
            for k, v in layer.film.state_dict().items()}


def load_appearance_film_state_dict(seva: nn.Module, state: dict[str, torch.Tensor]) -> None:
    layers = appearance_film_layers(seva)
    expected = {f"{n}.{k}" for n in layers for k in ("weight", "bias")}
    if set(state) != expected:
        raise KeyError(f"Appearance FiLM state mismatch: missing {sorted(expected - set(state))[:3]}, "
                       f"unexpected {sorted(set(state) - expected)[:3]}")
    for name, layer in layers.items():
        layer.film.load_state_dict({k: state[f"{name}.{k}"] for k in ("weight", "bias")})
