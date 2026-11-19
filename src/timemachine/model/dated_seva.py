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
  classifier-free guidance doubles the batch (unconditional half first, as in
  ``seva.sampling.VanillaCFG``), the dates are repeated, so both branches see the
  same dates; or, with ``uncond="unknown"``, the unconditional half sees every date
  as unknown. SEVA's conditional prediction alone gives fragmented renders and
  needs the guidance; dating the unconditional branch weakens it and broke dated
  renders (docs/findings.md, 2026-11-19).

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

# What the unconditional half of a CFG batch sees: the same dates, or all dates unknown.
UNCOND_DATES = ("same", "unknown")

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
        self.uncond_unknown = False        # CFG batch: mark the unconditional (first) half unknown

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
        if reps == 2 and self.uncond_unknown:
            valid = valid.clone()
            valid[: self.year.shape[0]] = False
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
    def dates(self, year: torch.Tensor, valid: torch.Tensor | None = None, uncond: str = "same"):
        """Condition every forward pass inside the block on these per-view dates.

        ``uncond`` (one of ``UNCOND_DATES``): what the unconditional half of a CFG-doubled
        batch sees, the same dates or all dates unknown.
        """
        if uncond not in UNCOND_DATES:
            raise ValueError(f"uncond must be one of {UNCOND_DATES}")
        hook = self.date_module
        hook.year = year
        hook.valid = torch.ones_like(year, dtype=torch.bool) if valid is None else valid
        hook.uncond_unknown = uncond == "unknown"
        try:
            yield self
        finally:
            hook.year = hook.valid = None
            hook.uncond_unknown = False

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

    def add_date_film(self, feature_dim: int = 256, n_freqs: int = 8) -> list[str]:
        """Add date FiLM to every ResBlock (zero-init, so the model is unchanged). Returns the block names.

        ``n_freqs`` octaves give the shortest Fourier period 200 / 2^(n_freqs-1) years (8: 1.6 years).
        """
        hook = self.date_module
        if hook.film_encoder is None:
            device = next(hook.date_embedding.parameters()).device
            hook.film_encoder = DateEncoder(feature_dim, n_freqs=n_freqs).to(device)
        return add_date_film(self.module, hook.state, hook.film_encoder.dim)

    def film_parameters(self) -> list[nn.Parameter]:
        """Date-FiLM parameters: the shared encoder and every per-block layer (empty without FiLM)."""
        encoder = self.date_module.film_encoder
        if encoder is None:
            return []
        return [*encoder.parameters(), *(p for layer in date_film_layers(self.module).values()
                                         for p in layer.film.parameters())]
