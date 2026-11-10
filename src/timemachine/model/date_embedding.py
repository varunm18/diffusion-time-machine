"""Per-view date embedding for date-conditioned SEVA.

A view's capture date (decimal year) becomes a vector of the size of SEVA's
timestep embedding (1280) and is *added* to it, so it reaches every ResBlock of
the UNet once per view (see ``dated_seva.py``).

Encoding: normalize ``t = (year - 1900) / 100`` (so 1500..2100 -> -4..2), take
Fourier features ``sin/cos(2^k * pi * t)`` for k = 0..n_freqs-1 plus ``t``
itself, then a small MLP. Views with no usable date use a learned "unknown date"
feature vector instead. The MLP's last layer is zero-initialized, so a fresh
module outputs exactly zero and the dated model starts out identical to SEVA.

Size matters: SEVA's timestep embedding is small (per-dimension RMS ~0.3;
levels 100 steps apart differ by ~0.05-0.1), so a date embedding that grows as
large drowns the noise-level signal. Trained alone with the plain epsilon loss,
it reached RMS 0.55 by step 2,500 and broke the renders at modern dates; the
same embedding at half size rendered cleanly (docs/findings.md).
``timemachine.model.diagnostics`` tracks the size during training. ``max_abs``
(off by default; ``--date-max-abs``) bounds every output dimension softly,
``max_abs * tanh(x / max_abs)``, which is exact identity at zero init.
"""
from __future__ import annotations

import math

import torch
from torch import nn

REFERENCE_YEAR = 1900.0
YEAR_SCALE = 100.0


def fourier_date_features(year: torch.Tensor, valid: torch.Tensor, freqs: torch.Tensor,
                          unknown: torch.Tensor) -> torch.Tensor:
    """(N,) years + (N,) validity -> (N, 2 * len(freqs) + 1) features; invalid rows get ``unknown``."""
    t = ((year.float() - REFERENCE_YEAR) / YEAR_SCALE)[:, None]
    feats = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs), t], dim=-1)
    return torch.where(valid[:, None].bool(), feats, unknown.to(feats.dtype).expand_as(feats))


def octave_freqs(n_freqs: int) -> torch.Tensor:
    """``2^k * pi`` for k = 0..n_freqs-1: periods of 200, 100, 50, ... years."""
    return (2.0 ** torch.arange(n_freqs)) * math.pi


class DateEmbedding(nn.Module):
    def __init__(self, out_dim: int = 1280, n_freqs: int = 8, hidden_dim: int = 512,
                 max_abs: float | None = None):
        super().__init__()
        self.max_abs = max_abs
        self.register_buffer("freqs", octave_freqs(n_freqs), persistent=False)
        feature_dim = 2 * n_freqs + 1
        self.unknown = nn.Parameter(torch.randn(feature_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def features(self, year: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """(N,) years + (N,) validity -> (N, 2 * n_freqs + 1) features."""
        return fourier_date_features(year, valid, self.freqs, self.unknown)

    def forward(self, year: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if valid is None:
            valid = torch.ones_like(year, dtype=torch.bool)
        out = self.mlp(self.features(year, valid))
        if self.max_abs is not None:
            out = self.max_abs * torch.tanh(out / self.max_abs)
        return out
