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

    **Era residualization** (``--appearance-remove-era``). The statistics predict
    era at AUC ~0.87 on their own, so handing them to the model as-is gives the
    date pathway a way to read the year off the film stock. ``era_directions``
    holds k orthonormal directions in *standardized* feature space (one per row)
    that :func:`fit_era_directions` found to carry era; :meth:`forward` projects
    them out before the MLP, so the encoder never sees that part of appearance.
    k = 0 (the default) is exactly the unresidualized encoder. The buffer travels
    with the checkpoint, and loading resizes it to the checkpoint's k.
    """

    def __init__(self, n_features: int = N_APPEARANCE_FEATURES, dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.n_features = n_features
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("scale", torch.ones(n_features))
        self.register_buffer("era_directions", torch.zeros(0, n_features))
        self.unknown = nn.Parameter(torch.randn(n_features) * 0.5)
        self.mlp = nn.Sequential(nn.Linear(n_features, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, dim), nn.SiLU())

    @property
    def removes_era(self) -> bool:
        return self.era_directions.shape[0] > 0

    @torch.no_grad()
    def set_era_directions(self, dirs) -> "AppearanceEncoder":
        """Install (k, F) directions in standardized feature space; k = 0 turns removal off.

        They are orthonormalized here, so :meth:`forward` can project with a single
        matrix product. Directions that are linearly dependent are refused rather than
        silently collapsed: the caller asked for k independent removals.
        """
        dirs = torch.as_tensor(dirs, dtype=torch.float32)
        if dirs.numel() == 0:
            dirs = dirs.reshape(0, self.n_features)
        if dirs.ndim != 2 or dirs.shape[1] != self.n_features:
            raise ValueError(f"Expected (k, {self.n_features}) era directions, got {tuple(dirs.shape)}")
        if dirs.shape[0] > self.n_features:
            raise ValueError(f"{dirs.shape[0]} directions cannot be independent in {self.n_features} dimensions")
        if dirs.shape[0]:
            q, r = torch.linalg.qr(dirs.double().T)          # columns of q: orthonormal basis of the span
            if (r.diagonal().abs() < 1e-6).any():
                raise ValueError("Era directions are linearly dependent")
            dirs = q.T.float()
        self.era_directions = dirs.to(self.mean.device)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        # The buffer's first dimension is k, which a fresh module does not know: it is
        # built with k = 0 and only a checkpoint says otherwise. Without resizing first,
        # every resume and every evaluation of an era-residualized run fails with a size
        # mismatch. Checkpoints from before this buffer existed load as k = 0.
        key = prefix + "era_directions"
        incoming = state_dict.get(key)
        if incoming is not None and incoming.ndim == 2 and incoming.shape[1] == self.n_features:
            self.era_directions = torch.empty(incoming.shape, dtype=self.era_directions.dtype,
                                              device=self.era_directions.device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)
        if incoming is None:
            self.era_directions = torch.zeros(0, self.n_features, device=self.mean.device)
            if key in missing_keys:
                missing_keys.remove(key)

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
        if self.removes_era:
            # After standardization: the directions were fitted in this basis, and in the
            # raw one a unit step means wildly different things per statistic.
            d = self.era_directions
            z = z - (z @ d.T) @ d
        if valid is not None:
            z = torch.where(valid[:, None].bool(), z, self.unknown.to(z.dtype).expand_as(z))
        return self.mlp(z)


def fit_era_directions(encoder: AppearanceEncoder, stats, years, old_before: float = 1945.0,
                       target_auc: float = 0.60, max_dims: int = 8, test_fraction: float = 0.3,
                       seed: int = 0) -> dict:
    """Find the era subspace of this run's appearance statistics and install it on ``encoder``.

    Uses :func:`timemachine.analysis.entanglement.era_subspace`, the measurement that
    found era in 6 of the 14 standardized dimensions: remove the best era direction,
    refit a fresh classifier on what is left, and repeat until it falls below
    ``target_auc`` (held-out rows) or ``max_dims`` are gone. So the directions removed
    are the ones the shortcut probe actually used, not a guess.

    ``stats`` must be the sample :meth:`AppearanceEncoder.fit_normalization` was fitted
    on. The fit runs on the encoder's own standardized features, and on that sample
    they are exactly zero-mean with equal column scales, so the directions found are
    in the basis :meth:`AppearanceEncoder.forward` projects in (``era_subspace``
    re-standardizes, which on such features is a uniform rescale and leaves unit
    directions unchanged). Rows without a finite year or with non-finite statistics
    are left out of the fit, which makes that only approximately true.

    Returns the measurement (``dims_removed``, ``era_removed``, the AUC trace, ...), as
    plain values that can be written next to the run.
    """
    import numpy as np

    from timemachine.analysis.entanglement import era_subspace

    stats = torch.as_tensor(stats, dtype=torch.float32)
    years = np.asarray(years, dtype=np.float64).reshape(-1)
    if stats.ndim != 2 or stats.shape[1] != encoder.n_features or len(stats) != len(years):
        raise ValueError(f"Expected (N, {encoder.n_features}) statistics and N years, "
                         f"got {tuple(stats.shape)} and {len(years)}")
    with torch.no_grad():
        z = ((stats.to(encoder.mean.device) - encoder.mean) / encoder.scale).double().cpu().numpy()
    keep = np.isfinite(years) & np.isfinite(z).all(axis=1)
    z, years = z[keep], years[keep]
    old = years < old_before
    if old.sum() < 2 or (~old).sum() < 2:
        raise ValueError(f"Need old and modern views to find the era direction: {int(old.sum())} before "
                         f"{old_before:g}, {int((~old).sum())} after, of {len(years)} usable rows")

    rng = np.random.default_rng(seed)
    train = np.ones(len(years), dtype=bool)
    # Split each era separately, so both sides of the held-out AUC always have rows.
    for side in (old, ~old):
        rows = np.flatnonzero(side)
        train[rng.choice(rows, size=max(1, int(round(test_fraction * len(rows)))), replace=False)] = False

    sub = era_subspace(z, years, train, old_before=old_before, target_auc=target_auc, max_dims=max_dims)
    encoder.set_era_directions(sub["directions"])
    trace = sub["trace"]
    return {"dims_removed": int(len(sub["directions"])), "n_features": encoder.n_features,
            "era_removed": bool(sub["era_removed"]), "target_auc": target_auc,
            "era_auc_full": float(trace["era_auc"].iloc[0]),
            "era_auc_residual": float(trace["era_auc"].iloc[-1]),
            "variance_kept": float(trace["variance_kept"].iloc[-1]),
            "n_rows": int(len(years)), "n_old": int(old.sum()), "n_test": int((~train).sum()),
            "trace": trace.to_dict(orient="records")}


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

    wrapped, already = [], 0
    for name, block in seva.named_modules():
        if not isinstance(block, ResBlock):
            continue
        if isinstance(block.in_layers[-1], AppearanceFiLMConv):
            already += 1
            continue
        block.in_layers[-1] = AppearanceFiLMConv(block.in_layers[-1], state, feature_dim)
        wrapped.append(name)
    if not wrapped:
        # Distinguish the two ways this can come up. Installing twice would otherwise
        # report "no ResBlock found", which sends you looking in the wrong place.
        raise ValueError(f"Appearance FiLM is already installed on all {already} ResBlocks"
                         if already else "No SEVA ResBlock found")
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
