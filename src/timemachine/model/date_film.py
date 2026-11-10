"""Date FiLM: a per-view, per-channel scale and shift from the capture date, in every ResBlock.

SEVA already conditions every ResBlock on the cameras this way: after the first
GroupNorm + SiLU it applies ``h * (1 + scale) + shift`` with per-pixel values from
the Plücker rays (``seva.modules.layers.ResBlock``). Date FiLM adds a second
modulation right after it, with one value per channel and view::

    year -> Fourier features -> DateEncoder (shared, 256-d) -> per-block Linear -> (scale, shift)
    h = h * (1 + scale) + shift,  then the block's first conv as before

* The per-block layers start at zero, so the model starts out exactly as SEVA.
* Unlike adding a date vector to the timestep embedding (``date_embedding.py``),
  this leaves the noise-level signal untouched, and each block learns its own
  response to the date (scaling channels can drain colour; shifting can tint).

Dates are computed once per UNet forward pass, outside the gradient-checkpointed
blocks: ``TimeEmbedWithDate`` (``dated_seva.py``) encodes them and stores the
features in a shared :class:`DateState`, which every :class:`DateFiLMConv` reads.
The features stay stored until the next forward pass, so the blocks' recomputation
during backward (``grad_checkpoint.py``) sees exactly the same tensor.
"""
from __future__ import annotations

import torch
from torch import nn

from timemachine.model.date_embedding import fourier_date_features, octave_freqs


class DateState:
    """Plain holder (not a module) for the current forward pass's per-view date features."""

    def __init__(self):
        self.features: torch.Tensor | None = None


class DateEncoder(nn.Module):
    """(N,) years + validity -> (N, dim) features shared by all FiLM layers (standard init, not zero)."""

    def __init__(self, dim: int = 256, n_freqs: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.register_buffer("freqs", octave_freqs(n_freqs), persistent=False)
        feature_dim = 2 * n_freqs + 1
        self.unknown = nn.Parameter(torch.randn(feature_dim) * 0.5)
        self.mlp = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, dim), nn.SiLU())

    def forward(self, year: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.mlp(fourier_date_features(year, valid, self.freqs, self.unknown))


class DateFiLMConv(nn.Module):
    """A ResBlock's first conv, preceded by a date scale-and-shift of its input channels."""

    def __init__(self, conv: nn.Conv2d, state: DateState, feature_dim: int):
        super().__init__()
        self.conv = conv
        self.film = nn.Linear(feature_dim, 2 * conv.in_channels, device=conv.weight.device)
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
                raise ValueError(f"{h.shape[0]} views vs {features.shape[0]} date features")
            scale, shift = self.scale_shift(features)
            h = h * (1 + scale[:, :, None, None].to(h.dtype)) + shift[:, :, None, None].to(h.dtype)
        return self.conv(h)


def add_date_film(seva: nn.Module, state: DateState, feature_dim: int) -> list[str]:
    """Put a :class:`DateFiLMConv` in front of the first conv of every SEVA ResBlock. Returns their names."""
    from seva.modules.layers import ResBlock

    wrapped = []
    for name, block in seva.named_modules():
        if isinstance(block, ResBlock) and not isinstance(block.in_layers[-1], DateFiLMConv):
            block.in_layers[-1] = DateFiLMConv(block.in_layers[-1], state, feature_dim)
            wrapped.append(name)
    if not wrapped:
        raise ValueError("No SEVA ResBlock found")
    return wrapped


def date_film_layers(seva: nn.Module) -> dict[str, DateFiLMConv]:
    return {name: m for name, m in seva.named_modules() if isinstance(m, DateFiLMConv)}


def date_film_state_dict(seva: nn.Module) -> dict[str, torch.Tensor]:
    """``{"<layer>.weight": ..., "<layer>.bias": ...}`` of every per-block FiLM layer (CPU copies)."""
    return {f"{name}.{k}": v.detach().cpu().clone()
            for name, layer in date_film_layers(seva).items() for k, v in layer.film.state_dict().items()}


def load_date_film_state_dict(seva: nn.Module, state: dict[str, torch.Tensor]) -> None:
    layers = date_film_layers(seva)
    expected = {f"{n}.{k}" for n in layers for k in ("weight", "bias")}
    if set(state) != expected:
        raise KeyError(f"Date FiLM state mismatch: missing {sorted(expected - set(state))[:3]}, "
                       f"unexpected {sorted(set(state) - expected)[:3]}")
    for name, layer in layers.items():
        layer.film.load_state_dict({k: state[f"{name}.{k}"] for k in ("weight", "bias")})
