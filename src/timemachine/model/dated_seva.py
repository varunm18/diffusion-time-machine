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

Optionally (:meth:`DatedSGMWrapper.add_date_film`), the dates also drive a
per-block scale-and-shift of the features (``date_film.py``). The hook then
encodes the dates once per forward pass and leaves the features in a shared
:class:`~timemachine.model.date_film.DateState` for the blocks to read.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from seva.model import Seva, SGMWrapper
from torch import nn

from timemachine.model.date_embedding import DateEmbedding
from timemachine.model.date_film import DateEncoder, DateState, add_date_film, date_film_layers


class TimeEmbedWithDate(nn.Module):
    """SEVA's timestep MLP plus a per-view date embedding (when dates are set).

    With a ``film_encoder`` it also encodes the dates for the date-FiLM layers into
    ``state`` (``None`` when no dates are set, so the FiLM layers do nothing).
    """

    def __init__(self, time_embed: nn.Module, date_embedding: DateEmbedding):
        super().__init__()
        self.time_embed = time_embed
        self.date_embedding = date_embedding
        self.film_encoder: DateEncoder | None = None
        self.state = DateState()
        self.year: torch.Tensor | None = None
        self.valid: torch.Tensor | None = None

    def forward(self, t_features: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(t_features)
        if self.year is None:
            self.state.features = None
            return emb
        n = emb.shape[0]
        if n % self.year.shape[0]:
            raise ValueError(f"{n} timestep embeddings vs {self.year.shape[0]} dates")
        reps = n // self.year.shape[0]           # >1 when CFG doubles the batch
        year, valid = self.year.repeat(reps).to(emb.device), self.valid.repeat(reps).to(emb.device)
        if self.film_encoder is not None:
            self.state.features = self.film_encoder(year, valid)
        return emb + self.date_embedding(year, valid).to(emb.dtype)


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
        """The date embedding added to the timestep embedding, e.g. to train it while SEVA is frozen."""
        return list(self.date_module.date_embedding.parameters())

    def add_date_film(self, feature_dim: int = 256) -> list[str]:
        """Add date FiLM to every ResBlock (zero-init, so the model is unchanged). Returns the block names."""
        hook = self.date_module
        if hook.film_encoder is None:
            device = next(hook.date_embedding.parameters()).device
            hook.film_encoder = DateEncoder(feature_dim).to(device)
        return add_date_film(self.module, hook.state, hook.film_encoder.dim)

    def film_parameters(self) -> list[nn.Parameter]:
        """Date-FiLM parameters: the shared encoder and every per-block layer (empty without FiLM)."""
        encoder = self.date_module.film_encoder
        if encoder is None:
            return []
        return [*encoder.parameters(), *(p for layer in date_film_layers(self.module).values()
                                         for p in layer.film.parameters())]
