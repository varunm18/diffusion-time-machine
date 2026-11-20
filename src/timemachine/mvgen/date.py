"""Per-view capture dates for MVGenMaster: the date embedding and how it reaches every ResBlock.

MVGenMaster's UNet computes one timestep embedding per batch element, adds the domain ("class")
embedding, repeats the sum for every view and hands it to every ``ResnetBlock2D`` as ``temb``
(``my_diffusers/models/unets/unet_2d_condition.py``, step "1. time"). The date goes in at the same
place, with one value per view instead of one per sample: :class:`DateEmbedding` turns each view's
year into a vector of ``temb``'s width, and :class:`DateConditioner` adds it to the ``temb``
argument of every ResBlock through forward pre-hooks. That equals adding it to the timestep
embedding after the per-view repeat, and no vendored code is edited.

The ResBlocks read ``temb`` through their ``time_emb_proj`` linear layers, which are therefore the
only readers of the date. They are frozen by default; LoRA may adapt them
(``timemachine.mvgen.model.TIME_PROJ_TARGET``), and nothing here depends on whether they are
wrapped or on which of the two, conditioner or LoRA, is attached first.

Three rules from the 2026-11-19 audit (docs/audit_2026-11-19.md) shape this module:

* **No-op at the start.** The embedding's last layer is zero-initialised, so a new conditioner
  adds exact zeros: the model's output equals the released model's. With no dates set the hooks
  do nothing at all.
* **One embedding per UNet forward pass, bound to that pass.** Gradient checkpointing runs every
  ResBlock a second time during ``backward()``, and the hooks run again with it. By then the dates
  may have changed or been cleared, and other forward passes may have run, so a hook must never
  compute the embedding from the conditioner's current state. Instead the embedding is computed
  once, when the UNet's forward pass starts, and the first ResBlock hook attaches it to the pass's
  ``temb`` tensor as a Python attribute. Every hook, in the forward pass and in the re-run alike,
  adds the embedding it finds on the ``temb`` it is given. The re-run receives the very same
  ``temb`` object (non-reentrant checkpointing, which all of MVGenMaster's blocks use, keeps its
  inputs), so it always sees its own pass's embedding. A second forward pass before ``backward()``
  (a reference pass of a ranking loss, a probe) therefore cannot corrupt the gradient. On a toy
  model, the same situation without this binding gave a gradient 34% off, silently.

  If a ResBlock ever runs outside a UNet forward pass with a ``temb`` that carries no embedding
  (reentrant checkpointing hands the re-run a copy), the hook raises instead of guessing.

  The embedding is computed in the UNet's own pre-hook and not by the first ResBlock that needs
  it, because MVGenMaster checkpoints each ResBlock call separately: the checkpoint records which
  tensors the call saved for ``backward()`` and expects the re-run to save the same ones, which
  fails (``CheckpointError``) if the first run computed the embedding and the re-run reused it.
* **Precision.** The embedding is computed in fp32 whatever the autocast context, then cast to
  ``temb``'s dtype for the addition. In the released model ``temb`` is fp32 even under bf16
  autocast (the fp32 domain embedding promotes it), so nothing is rounded in the addition; like
  the timestep and domain embeddings, the sum is rounded to bf16 afterwards, inside each block's
  ``time_emb_proj``. The gradient at this place was measured sound in bf16 and in fp32
  (``data/experiments/mvgen/grad_precision.json``, group ``domain_embedding``).

**Size.** Nothing bounds the embedding, and it shares ``temb`` with the timestep and the domain. In
the released model that sum has a root mean square of 0.11 to 0.26, depending on the timestep
(``data/mvgen_review/learning-hazards/x8_date_embedding_growth.py``); a date embedding that
outgrows it shifts the point at which every ResBlock reads the timestep (an embedding of RMS 0.55
broke SEVA's renders, ``timemachine/model/date_embedding.py``). So every dated forward pass records
the RMS of its embedding and of the ``temb`` it was added to in :attr:`DateConditioner.last_stats`,
for the trainer to log. The two numbers are computed once per pass, when the embedding is bound,
from detached tensors: no autograd work, nothing for a checkpoint to save, and the re-run during
``backward()`` leaves them alone.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from my_diffusers.models.resnet import ResnetBlock2D
from torch import nn

from timemachine.model.date_embedding import fourier_date_features, octave_freqs
from timemachine.model.lora import LoRALinear

# What the unconditional half of a guidance-doubled batch sees: the same dates, or every date unknown.
UNCOND_DATES = ("same", "unknown")

# Attribute that binds a forward pass's date embedding (or None: undated pass) to its temb tensor.
_BOUND = "_timemachine_date_embedding"


class DateEmbedding(nn.Module):
    """Year -> vector of the timestep embedding's width; exactly zero at initialisation.

    ``t = (year - 1900) / 100`` -> ``sin`` and ``cos`` of ``2^k * pi * t`` for ``k < n_freqs``, plus
    ``t`` itself -> one hidden layer (SiLU) -> ``out_dim``. A view without a date uses a learned
    "unknown" feature vector in place of the Fourier features. The last layer's weight and bias
    start at zero.

    The default 8 octaves have periods from 200 years down to 1.6 years. The change this project
    first has to learn, the Notre-Dame fire of April 2019, separates photos a few months apart; with
    4 octaves (shortest period 25 years) early and late 2019 are almost the same input.
    """

    def __init__(self, out_dim: int = 1280, n_freqs: int = 8, hidden_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer("freqs", octave_freqs(n_freqs), persistent=False)
        feature_dim = 2 * n_freqs + 1
        self.unknown = nn.Parameter(torch.randn(feature_dim) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, year: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``year`` (N,) decimal years, ``valid`` (N,) bool (False: date unknown) -> (N, out_dim)."""
        return self.mlp(fourier_date_features(year, valid, self.freqs, self.unknown))


def dated_blocks(unet: nn.Module) -> list[ResnetBlock2D]:
    """The ResBlocks of ``unet`` that read the timestep embedding (all 22 in the released model).

    A block counts whether its ``time_emb_proj`` is the original linear layer or a LoRA wrapper around it.
    """
    return [m for m in unet.modules() if isinstance(m, ResnetBlock2D) and m.time_emb_proj is not None]


def temb_width(unet: nn.Module) -> int:
    """Width of the ``temb`` that ``unet``'s ResBlocks read: what a :class:`DateEmbedding` must output."""
    widths = {_time_proj(block).in_features for block in dated_blocks(unet)}
    if len(widths) != 1:
        raise ValueError(f"expected ResnetBlock2D blocks with one timestep-embedding width, found {sorted(widths)}")
    return widths.pop()


def _time_proj(block: ResnetBlock2D) -> nn.Linear:
    """The linear layer through which ``block`` reads ``temb``, also when a LoRA adapter wraps it."""
    proj = block.time_emb_proj
    return proj.base if isinstance(proj, LoRALinear) else proj


def _rms(x: torch.Tensor) -> float:
    """Root mean square of ``x``, taken in fp32 and outside autograd."""
    with torch.no_grad():
        return x.detach().float().square().mean().sqrt().item()


class DateConditioner(nn.Module):
    """Adds a per-view date embedding to the ``temb`` of every ResBlock of a UNet (see the module docstring).

    The UNet is expected to be called as MVGenMaster's is: its first argument ``sample`` is
    ``[B, F, C, h, w]`` (or ``[B, C, h, w]``), and its ResBlocks receive a ``temb`` with one row per
    view, ``[(B*F), width]``, batch elements one after the other.

    The UNet is not registered as a submodule: this module's parameters and state dict are the
    embedding's alone.

    Attributes:
        embedding: the :class:`DateEmbedding`, the only trainable part.
        last_stats: sizes in the most recent UNet forward pass, ``{"date_rms": ..., "temb_rms": ...}``:
            the root mean square of the date embedding (over every view of the batch, those whose
            date is unknown included) and of the ``temb`` it was added to (timestep plus domain,
            before the addition). ``None`` if that pass had no dates or failed before its first
            ResBlock, before the first pass, and after :meth:`remove`.
    """

    def __init__(self, unet: nn.Module, embedding: DateEmbedding):
        super().__init__()
        width = temb_width(unet)
        if embedding.out_dim != width:
            raise ValueError(f"the date embedding has width {embedding.out_dim}, the UNet's temb has {width}")
        self.embedding = embedding
        self.last_stats: dict[str, float] | None = None
        self._dates: tuple[torch.Tensor, torch.Tensor, str] | None = None    # (year, valid, uncond) inside dates()
        self._in_forward = False                    # True while the UNet's forward pass is running
        self._pending: torch.Tensor | None = None   # embedding of the pass that is running; None: undated
        self._handles = [unet.register_forward_pre_hook(self._start_forward, with_kwargs=True),
                         unet.register_forward_hook(self._end_forward, always_call=True)]
        self._handles += [block.register_forward_pre_hook(self._add_to_temb, with_kwargs=True)
                          for block in dated_blocks(unet)]

    @contextmanager
    def dates(self, year: torch.Tensor, valid: torch.Tensor | None = None, uncond: str = "same") -> Iterator[None]:
        """Condition the UNet forward passes inside the block on these per-view dates.

        Args:
            year: (N,) decimal years. N must divide the number of views in the UNet's batch; the dates
                are repeated to fill it, so N = F serves a single sample and a guidance-doubled batch.
            valid: (N,) bool, False where the date is unknown. None: all known.
            uncond: one of ``UNCOND_DATES``. With ``"unknown"``, when the batch holds the dates exactly
                twice (guidance: unconditional half first), the first half sees every date as unknown.

        Leaving the block clears the dates. ``backward()`` may run later, outside the block: each
        forward pass keeps its own embedding (see the module docstring).
        """
        if uncond not in UNCOND_DATES:
            raise ValueError(f"uncond must be one of {UNCOND_DATES}")
        year = torch.as_tensor(year, dtype=torch.float32)
        valid = torch.ones_like(year, dtype=torch.bool) if valid is None else torch.as_tensor(valid, dtype=torch.bool)
        if year.ndim != 1 or valid.shape != year.shape:
            raise ValueError(f"year and valid must be 1-D and of equal length, got {tuple(year.shape)} and "
                             f"{tuple(valid.shape)}")
        previous, self._dates = self._dates, (year, valid, uncond)
        try:
            yield
        finally:
            self._dates = previous

    def remove(self) -> None:
        """Detach the hooks: the UNet is the base model again."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._in_forward, self._pending, self.last_stats = False, None, None

    def _embed(self, rows: int) -> torch.Tensor:
        """The (rows, width) fp32 embedding of the current dates, repeated to fill ``rows`` views."""
        year, valid, uncond = self._dates
        n = year.shape[0]
        if rows % n:
            raise ValueError(f"{n} dates cannot fill the {rows} views of the UNet's batch")
        reps = rows // n
        year, valid = year.repeat(reps), valid.repeat(reps)
        if reps == 2 and uncond == "unknown":
            valid[:n] = False
        device = self.embedding.unknown.device
        with torch.autocast(device.type, enabled=False):
            return self.embedding(year.to(device), valid.to(device))

    def _start_forward(self, unet: nn.Module, args: tuple, kwargs: dict) -> None:
        """UNet pre-hook: compute this forward pass's embedding (None when no dates are set)."""
        self._in_forward, self._pending, self.last_stats = True, None, None
        if self._dates is not None:
            sample = args[0] if args else kwargs["sample"]
            self._pending = self._embed(rows=sample.shape[:-3].numel())

    def _end_forward(self, unet: nn.Module, args: tuple, output: object) -> None:
        """UNet hook (runs after a failed pass too): no forward pass is running any more."""
        self._in_forward, self._pending = False, None

    def _add_to_temb(self, block: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
        """ResBlock pre-hook: ``temb`` -> ``temb`` + the embedding bound to it (no change for an undated pass)."""
        temb = args[1] if len(args) > 1 else kwargs["temb"]
        embedding = self._bound_embedding(temb)
        if embedding is None:
            return None
        if temb.shape[0] != embedding.shape[0]:
            raise ValueError(f"temb has {temb.shape[0]} rows, the date embedding {embedding.shape[0]}")
        shifted = temb + embedding.to(temb.dtype)
        if len(args) > 1:
            return (args[0], shifted, *args[2:]), kwargs
        return args, {**kwargs, "temb": shifted}

    def _bound_embedding(self, temb: torch.Tensor) -> torch.Tensor | None:
        """The embedding of the forward pass that created ``temb``, binding it on first sight.

        The first binding of a dated pass also records :attr:`last_stats`. A re-run during
        ``backward()`` finds its ``temb`` bound already and records nothing.
        """
        if not hasattr(temb, _BOUND):
            if not self._in_forward:
                raise RuntimeError(
                    "a ResBlock ran outside a UNet forward pass with a temb that carries no date embedding: a "
                    "gradient-checkpoint re-run did not receive the original tensor (reentrant checkpointing?), "
                    "so its dates cannot be recovered")
            setattr(temb, _BOUND, self._pending)
            if self._pending is not None and self.last_stats is None:
                self.last_stats = {"date_rms": _rms(self._pending), "temb_rms": _rms(temb)}
        return getattr(temb, _BOUND)
