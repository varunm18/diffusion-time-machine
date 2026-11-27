"""The denoising loss for (dated) SEVA, consistent with SEVA's sampler.

SEVA's denoiser (``seva.sampling.DiscreteDenoiser``) works on 1000 discrete noise
levels ``sigma_t`` (a DDPM schedule with a log-SNR shift of 2.4 towards high
noise). For a clean latent ``z`` and noise ``eps``, the noisy latent is
``x = z + sigma_t * eps``; input views are swapped back to their clean latent
(the ``replace`` condition); the network sees ``x / sqrt(sigma_t^2 + 1)`` and the
integer level ``t``, and the denoised estimate is ``x - sigma_t * network(...)``.
So the network predicts ``eps``, and the loss is the mean squared error between
its output and ``eps`` on the views that were *not* replaced (the targets; all
views in the unconditional branch, where nothing is replaced).

Training choices (see :class:`LossConfig`):
* one noise level per sample, uniform over the 1000 levels (the shift is already
  in SEVA's schedule);
* a per-level loss weight (:func:`level_weight`). Plain epsilon-MSE ("eps", what
  SD 2.1-style models are trained with) makes gradients ~1e5x larger at the few
  near-clean levels than at the ~89% of levels with sigma > 1, where the global
  look of the image (colour era, a missing spire) is decided. "edm" weights the
  MSE by ``sigma^2 + 1`` (EDM's weighting for sigma_data = 1, SEVA's ``c_in``), so
  every level contributes a loss of similar size; "min_snr" (gamma = 5) only
  tames the near-clean levels.
* the unconditional branch (no input images) is trained some of the time, so
  classifier-free guidance keeps working; optionally with every date unknown there
  (``uncond_dates="unknown"``), to match guidance whose unconditional half is undated;
* per-view years are drawn uniformly inside each view's date interval ("circa
  1890" is not exactly 1890.5), and some samples are marked "date unknown";
* optionally a date-discrimination (ranking) term (``rank_weight``). The plain
  denoising loss can be minimised well while ignoring the date, because a scene's
  static geometry dominates the pixels (docs/findings.md, 2026-11-19). So on some
  conditional steps the same noised targets are also scored with a wrong-era date,
  and a hinge on the log ratio of the two losses rewards the true date winning by
  ``rank_margin``: relu(margin - log(loss_wrong / loss_true)), per target view.
  The reference branch gets no gradient (``rank_detach`` must be on), so only the
  true-date prediction can improve: in effect, extra denoising weight where the date
  does not help yet. The reference can also be the unknown date
  (``rank_reference="unknown"``): the true date must then add information over having
  no date.

  Order matters: the date-FiLM layers read the dates of the *last* forward pass during
  gradient-checkpointing recomputation (``date_film.DateState``), so the gradient-free
  reference pass runs first and the true-date pass last. Before 2026-11-19 the order
  was reversed and the reference kept its gradient; the true-date gradients were then
  recomputed with the wrong-era dates (docs/findings.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

NUM_LEVELS = 1000
# Training precision. fp32 (autocast off; ``None`` here) is the default since the 2026-11-19 audit.
# SEVA predicts the noise, and at high noise levels its error is only ~1% of its output, about the
# size of bf16's rounding (0.4%): the gradient of that error is then mostly rounding noise, and the
# adapters' small corrections are partly rounded away when added to half-precision activations.
# Measured: same-seed bf16 runs move their weights in unrelated directions, and a fixed sample's
# loss stalls after a few steps in bf16 but falls steadily in fp32 (docs/findings.md). fp16 without
# a GradScaler is no alternative (many gradient entries underflow to zero). fp32 costs ~15% more
# memory and ~30% more time.
TRAIN_AUTOCAST_DTYPE = None
# By name, for configs and the command line ("bf16": runs of 2026-11-10..19; "fp16": earlier ones).
AUTOCAST_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def autocast(dtype: torch.dtype | None):
    """The autocast context of a training precision: half-precision ``dtype``, or ``None`` for fp32."""
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else torch.autocast("cuda", enabled=False)


WEIGHTINGS = ("eps", "edm", "min_snr")

# Wrong-era dates for the ranking loss and the probes: targets before SWAP_BOUNDARY move to
# MODERN_SWAP, later ones to OLD_SWAP.
SWAP_BOUNDARY = 1980.0
MODERN_SWAP = 2015.0
OLD_SWAP = 1890.0
# Which views see their date. "targets": inputs always get the unknown date, so a view's
# own date cannot double as a hint about the *targets'* era (docs/findings.md, 2026-11-18).
DATE_VIEWS = ("all", "targets")


@dataclass(frozen=True)
class LossConfig:
    uncond_prob: float = 0.1     # share of steps on the unconditional (CFG) branch
    date_dropout: float = 0.1    # share of samples whose dates are all marked unknown
    appearance_dropout: float = 0.1   # share of samples whose appearance is marked unknown
    date_jitter: bool = True     # draw years inside each view's interval (else: midpoint)
    weighting: str = "eps"       # per-level loss weight, one of WEIGHTINGS
    min_snr_gamma: float = 5.0
    date_views: str = "all"     # one of DATE_VIEWS
    uncond_dates: str = "same"  # unconditional-branch steps: dates as drawn, or all "unknown"
    rank_weight: float = 0.0    # weight of the date-discrimination term (0: off)
    rank_margin: float = 0.05   # required log(loss_wrong / loss_true) per target view
    rank_prob: float = 1.0      # share of eligible steps that compute it (one extra forward pass)
    rank_min_sigma: float = 1.0  # only at noise levels where the date can matter (global look)
    rank_reference: str = "wrong"  # compare the true date with a "wrong"-era date or the "unknown" date
    rank_detach: bool = False   # no gradient through the reference branch; required when rank_weight > 0

    def __post_init__(self):
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")
        if self.date_views not in DATE_VIEWS:
            raise ValueError(f"date_views must be one of {DATE_VIEWS}")
        if self.uncond_dates not in ("same", "unknown"):
            raise ValueError("uncond_dates must be 'same' or 'unknown'")
        if self.rank_reference not in ("wrong", "unknown"):
            raise ValueError("rank_reference must be 'wrong' or 'unknown'")
        if self.rank_weight > 0 and not self.rank_detach:
            raise ValueError("rank_weight > 0 needs rank_detach: a second gradient pass would be recomputed with "
                             "the other pass's dates under gradient checkpointing (and can be won by degrading it)")


def level_weight(sigma: float, cfg: LossConfig) -> float:
    """Weight of the epsilon-MSE at noise level ``sigma`` (see module docstring)."""
    if cfg.weighting == "edm":
        return sigma ** 2 + 1.0
    if cfg.weighting == "min_snr":     # min(SNR, gamma) / SNR with SNR = 1 / sigma^2
        return min(1.0, cfg.min_snr_gamma * sigma ** 2)
    return 1.0


def predict_noise(model: nn.Module, noisy: torch.Tensor, sigma: torch.Tensor, t: torch.Tensor, cond: dict,
                  year: torch.Tensor | None = None, date_valid: torch.Tensor | None = None,
                  appearance: torch.Tensor | None = None,
                  appearance_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The network's noise estimate, prepared exactly as in ``DiscreteDenoiser.__call__``.

    Args:
        noisy: (T, C, h, w) ``z + sigma * eps``.  sigma, t: (T,) noise level, as value and index.
        cond: SEVA conditioning dict incl. ``replace`` (see ``training.conditioning``).
        year, date_valid: (T,) per-view dates for the dated model (omit for plain SEVA).
        appearance, appearance_valid: (T, F) per-view appearance statistics and their
            validity, for a model with an appearance pathway (omit otherwise).

    Returns:
        ``(eps_hat, replaced)``: (T, C, h, w) estimate and the (T,) bool mask of replaced (input) views.
    """
    cond = dict(cond)
    clean, mask = cond.pop("replace").split((noisy.shape[1], 1), dim=1)
    x = noisy * (1 - mask) + clean * mask
    c_in = (1.0 / (sigma ** 2 + 1.0) ** 0.5).view(-1, 1, 1, 1)
    extra = {} if year is None else {"date": year, "date_valid": date_valid}
    if appearance is not None:
        extra |= {"appearance": appearance, "appearance_valid": appearance_valid}
    eps_hat = model(x * c_in, t, cond, num_frames=noisy.shape[0], **extra)
    return eps_hat, mask[:, 0, 0, 0] > 0.5


def per_view_noise_error(model: nn.Module, sigmas: torch.Tensor, latents: torch.Tensor, cond: dict,
                         level: int, noise: torch.Tensor, year: torch.Tensor | None = None,
                         date_valid: torch.Tensor | None = None,
                         appearance: torch.Tensor | None = None,
                         appearance_valid: torch.Tensor | None = None) -> torch.Tensor:
    """(K,) mean squared noise error of each non-replaced view, at noise index ``level``."""
    T = latents.shape[0]
    t = torch.full((T,), level, device=latents.device, dtype=torch.long)
    sigma = sigmas[t].to(latents.device)
    noisy = latents.float() + sigma.view(-1, 1, 1, 1) * noise
    eps_hat, replaced = predict_noise(model, noisy, sigma, t, cond, year, date_valid,
                                      appearance, appearance_valid)
    keep = ~replaced
    return (eps_hat.float()[keep] - noise[keep]).pow(2).flatten(1).mean(1)


def noise_prediction_loss(model: nn.Module, sigmas: torch.Tensor, latents: torch.Tensor, cond: dict,
                          level: int, noise: torch.Tensor, year: torch.Tensor | None = None,
                          date_valid: torch.Tensor | None = None,
                          appearance: torch.Tensor | None = None,
                          appearance_valid: torch.Tensor | None = None) -> torch.Tensor:
    """MSE between predicted and true noise on the non-replaced views, at noise index ``level``."""
    return per_view_noise_error(model, sigmas, latents, cond, level, noise, year, date_valid,
                                appearance, appearance_valid).mean()


def draw_appearance_validity(n_views: int, cfg: LossConfig, device: torch.device,
                             generator: torch.Generator | None = None) -> torch.Tensor:
    """(T,) validity for the appearance pathway; whole samples are dropped together.

    Dropping teaches the model to run without an appearance being chosen, which it
    must do when sampling asks only for a date. Unlike the dates, the appearance is
    *kept* in the unconditional branch of classifier-free guidance: WildCAT3D found
    that dropping an appearance code there oversaturated its renders.
    """
    drop = torch.rand((), generator=generator) < cfg.appearance_dropout
    return torch.full((n_views,), not bool(drop), dtype=torch.bool, device=device)


def swapped_years(year: torch.Tensor, is_input: torch.Tensor) -> torch.Tensor:
    """Targets' years moved to the other era (old -> 2015, modern -> 1890); inputs unchanged."""
    swapped = torch.where(year < SWAP_BOUNDARY, torch.full_like(year, MODERN_SWAP), torch.full_like(year, OLD_SWAP))
    return torch.where(is_input, year, swapped)


def shown_dates(valid: torch.Tensor, is_input: torch.Tensor, date_views: str) -> torch.Tensor:
    """Validity after hiding the inputs' dates when only targets see theirs."""
    return valid & ~is_input if date_views == "targets" else valid


def draw_dates(interval: torch.Tensor, cfg: LossConfig, generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """(T,) training years and validity from (T, 2) ``[earliest, latest)`` intervals."""
    lo, hi = interval[:, 0].float(), interval[:, 1].float()
    if cfg.date_jitter:
        u = torch.rand(lo.shape, generator=generator, device=lo.device)
        year = lo + u * (hi - lo)
    else:
        year = (lo + hi) / 2
    drop = torch.rand((), generator=generator, device=lo.device) < cfg.date_dropout
    valid = torch.full_like(year, not bool(drop), dtype=torch.bool)
    return year, valid


def training_loss(model: nn.Module, sigmas: torch.Tensor, latents: torch.Tensor, cond: dict, uncond: dict,
                  date_interval: torch.Tensor | None, cfg: LossConfig,
                  generator: torch.Generator | None = None,
                  is_input: torch.Tensor | None = None,
                  appearance: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """One stochastic (weighted) loss sample: random level, noise, branch and dates. Also returns what was drawn."""
    device = latents.device
    level = int(torch.randint(NUM_LEVELS, (), generator=generator))
    noise = torch.randn(latents.shape, generator=generator).to(device)
    use_uncond = bool(torch.rand((), generator=generator) < cfg.uncond_prob)
    year = valid = None
    if date_interval is not None:
        year, valid = draw_dates(date_interval.cpu(), cfg, generator)
        year, valid = year.to(device), valid.to(device)
        if cfg.date_views != "all":
            if is_input is None:
                raise ValueError(f"date_views={cfg.date_views!r} needs is_input")
            valid = shown_dates(valid, is_input.to(device), cfg.date_views)
        if use_uncond and cfg.uncond_dates == "unknown":
            valid = torch.zeros_like(valid)
    # Drawn once for the whole step: the rank loss compares two passes that must
    # differ only in their dates, so both see the same appearance and the same
    # validity. Drawing inside the branches would make the comparison measure the
    # appearance too.
    appearance_valid = None
    if appearance is not None:
        appearance = appearance.to(device)
        appearance_valid = draw_appearance_validity(latents.shape[0], cfg, device, generator)
    rank = (cfg.rank_weight > 0 and not use_uncond and valid is not None and bool(valid.any())
            and float(sigmas[level]) >= cfg.rank_min_sigma
            and bool(torch.rand((), generator=generator) < cfg.rank_prob))
    if rank:                                    # reference first and gradient-free (see module docstring)
        if is_input is None:
            raise ValueError("rank_weight needs is_input")
        if cfg.rank_reference == "wrong":
            ref_year, ref_valid = swapped_years(year, is_input.to(device)), valid
        else:
            ref_year, ref_valid = year, torch.zeros_like(valid)
        with torch.no_grad():
            errors_ref = per_view_noise_error(model, sigmas, latents, cond, level, noise, ref_year, ref_valid,
                                              appearance, appearance_valid)
    errors = per_view_noise_error(model, sigmas, latents, uncond if use_uncond else cond, level, noise, year, valid,
                                  appearance, appearance_valid)
    eps_mse = errors.mean()
    info = {"level": level, "uncond": use_uncond, "eps_mse": float(eps_mse.detach()),
            "dates_known": bool(valid.any()) if valid is not None else None,
            "appearance_known": bool(appearance_valid.any()) if appearance_valid is not None else None}
    loss = eps_mse * level_weight(float(sigmas[level]), cfg)
    if rank:
        log_ratio = torch.log(errors_ref.clamp_min(1e-12)) - torch.log(errors.clamp_min(1e-12))
        rank_loss = torch.relu(cfg.rank_margin - log_ratio).mean()
        loss = loss + cfg.rank_weight * rank_loss
        info.update(rank_loss=float(rank_loss.detach()), rank_log_ratio=float(log_ratio.detach().mean()))
    return loss, info
