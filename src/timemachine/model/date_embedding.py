"""Per-view date embedding for date-conditioned SEVA.

A view's capture date (decimal year) becomes a vector of the size of SEVA's
timestep embedding (1280) and is *added* to it, so it reaches every ResBlock of
the UNet once per view (see ``dated_seva.py``).

Encoding: normalize ``t = (year - 1900) / 100`` (so 1500..2100 -> -4..2), take
Fourier features ``sin/cos(2^k * pi * t)`` for k = 0..n_freqs-1 plus ``t``
itself, then a small MLP. Views with no usable date use a learned "unknown date"
feature vector instead. The MLP's last layer is zero-initialized, so a fresh
module outputs exactly zero and the dated model starts out identical to SEVA.
"""
from __future__ import annotations

import math

import torch
from torch import nn

REFERENCE_YEAR = 1900.0
YEAR_SCALE = 100.0


class DateEmbedding(nn.Module):
    def __init__(self, out_dim: int = 1280, n_freqs: int = 8, hidden_dim: int = 512):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freqs)) * math.pi, persistent=False)
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
        t = ((year.float() - REFERENCE_YEAR) / YEAR_SCALE)[:, None]
        feats = torch.cat([torch.sin(t * self.freqs), torch.cos(t * self.freqs), t], dim=-1)
        return torch.where(valid[:, None].bool(), feats, self.unknown.to(feats.dtype).expand_as(feats))

    def forward(self, year: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if valid is None:
            valid = torch.ones_like(year, dtype=torch.bool)
        return self.mlp(self.features(year, valid))
