"""Date-conditioned SEVA, built around the unmodified SEVA code.

SEVA computes one timestep embedding per view (``Seva.time_embed``) and adds it
inside every ResBlock. We replace that submodule by :class:`TimeEmbedWithDate`,
which returns ``time_embed(t) + date_embedding(date)``; nothing else in SEVA
changes. Because the date embedding starts at zero, the dated model initially
reproduces SEVA exactly.

Dates reach the model in one of two ways:
* explicitly, ``model(x, t, c, date=..., date_valid=...)`` (training);
* through :meth:`DatedSGMWrapper.dates`, a context manager, so SEVA's own sampling
  code (``seva.eval.do_sample``) can be used unchanged at inference. When
  classifier-free guidance doubles the batch, the dates are repeated, so the
  conditional and unconditional branches see the same dates.

Shapes follow SEVA: frames are flattened into the batch, so ``date`` is (B*T,).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from seva.model import Seva, SGMWrapper
from torch import nn

from timemachine.model.date_embedding import DateEmbedding


class TimeEmbedWithDate(nn.Module):
    """SEVA's timestep MLP plus a per-view date embedding (when dates are set)."""

    def __init__(self, time_embed: nn.Module, date_embedding: DateEmbedding):
        super().__init__()
        self.time_embed = time_embed
        self.date_embedding = date_embedding
        self.year: torch.Tensor | None = None
        self.valid: torch.Tensor | None = None

    def forward(self, t_features: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(t_features)
        if self.year is None:
            return emb
        n = emb.shape[0]
        if n % self.year.shape[0]:
            raise ValueError(f"{n} timestep embeddings vs {self.year.shape[0]} dates")
        reps = n // self.year.shape[0]           # >1 when CFG doubles the batch
        year, valid = self.year.repeat(reps), self.valid.repeat(reps)
        return emb + self.date_embedding(year.to(emb.device), valid.to(emb.device)).to(emb.dtype)


class DatedSGMWrapper(SGMWrapper):
    """``seva.model.SGMWrapper`` whose Seva also receives per-view dates."""

    def __init__(self, module: Seva, date_embedding: DateEmbedding | None = None):
        super().__init__(module)
        time_embed_dim = module.model_channels * 4
        module.time_embed = TimeEmbedWithDate(module.time_embed,
                                              date_embedding or DateEmbedding(time_embed_dim))

    @property
    def date_module(self) -> TimeEmbedWithDate:
        return self.module.time_embed

    @contextmanager
    def dates(self, year: torch.Tensor, valid: torch.Tensor | None = None):
        """Condition every forward pass inside the block on these per-view dates."""
        hook = self.date_module
        hook.year = year
        hook.valid = torch.ones_like(year, dtype=torch.bool) if valid is None else valid
        try:
            yield self
        finally:
            hook.year = hook.valid = None

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict,
                date: torch.Tensor | None = None, date_valid: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        if date is None:
            return super().forward(x, t, c, **kwargs)
        with self.dates(date, date_valid):
            return super().forward(x, t, c, **kwargs)

    def date_parameters(self) -> list[nn.Parameter]:
        """The new (date) parameters, e.g. to train them while SEVA is frozen."""
        return list(self.date_module.date_embedding.parameters())
