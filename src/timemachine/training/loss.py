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
  classifier-free guidance keeps working;
* per-view years are drawn uniformly inside each view's date interval ("circa
  1890" is not exactly 1890.5), and some samples are marked "date unknown".
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

NUM_LEVELS = 1000


WEIGHTINGS = ("eps", "edm", "min_snr")


@dataclass(frozen=True)
class LossConfig:
    uncond_prob: float = 0.1     # share of steps on the unconditional (CFG) branch
    date_dropout: float = 0.1    # share of samples whose dates are all marked unknown
    date_jitter: bool = True     # draw years inside each view's interval (else: midpoint)
    weighting: str = "eps"       # per-level loss weight, one of WEIGHTINGS
    min_snr_gamma: float = 5.0

    def __post_init__(self):
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")


def level_weight(sigma: float, cfg: LossConfig) -> float:
    """Weight of the epsilon-MSE at noise level ``sigma`` (see module docstring)."""
    if cfg.weighting == "edm":
        return sigma ** 2 + 1.0
    if cfg.weighting == "min_snr":     # min(SNR, gamma) / SNR with SNR = 1 / sigma^2
        return min(1.0, cfg.min_snr_gamma * sigma ** 2)
    return 1.0


def predict_noise(model: nn.Module, noisy: torch.Tensor, sigma: torch.Tensor, t: torch.Tensor, cond: dict,
                  year: torch.Tensor | None = None, date_valid: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """The network's noise estimate, prepared exactly as in ``DiscreteDenoiser.__call__``.

    Args:
        noisy: (T, C, h, w) ``z + sigma * eps``.  sigma, t: (T,) noise level, as value and index.
        cond: SEVA conditioning dict incl. ``replace`` (see ``training.conditioning``).
        year, date_valid: (T,) per-view dates for the dated model (omit for plain SEVA).

    Returns:
        ``(eps_hat, replaced)``: (T, C, h, w) estimate and the (T,) bool mask of replaced (input) views.
    """
    cond = dict(cond)
    clean, mask = cond.pop("replace").split((noisy.shape[1], 1), dim=1)
    x = noisy * (1 - mask) + clean * mask
    c_in = (1.0 / (sigma ** 2 + 1.0) ** 0.5).view(-1, 1, 1, 1)
    date_kwargs = {} if year is None else {"date": year, "date_valid": date_valid}
    eps_hat = model(x * c_in, t, cond, num_frames=noisy.shape[0], **date_kwargs)
    return eps_hat, mask[:, 0, 0, 0] > 0.5


def noise_prediction_loss(model: nn.Module, sigmas: torch.Tensor, latents: torch.Tensor, cond: dict,
                          level: int, noise: torch.Tensor, year: torch.Tensor | None = None,
                          date_valid: torch.Tensor | None = None) -> torch.Tensor:
    """MSE between predicted and true noise on the non-replaced views, at noise index ``level``."""
    T = latents.shape[0]
    t = torch.full((T,), level, device=latents.device, dtype=torch.long)
    sigma = sigmas[t].to(latents.device)
    noisy = latents.float() + sigma.view(-1, 1, 1, 1) * noise
    eps_hat, replaced = predict_noise(model, noisy, sigma, t, cond, year, date_valid)
    keep = ~replaced
    return (eps_hat.float()[keep] - noise[keep]).pow(2).mean()


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
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """One stochastic (weighted) loss sample: random level, noise, branch and dates. Also returns what was drawn."""
    device = latents.device
    level = int(torch.randint(NUM_LEVELS, (), generator=generator))
    noise = torch.randn(latents.shape, generator=generator).to(device)
    use_uncond = bool(torch.rand((), generator=generator) < cfg.uncond_prob)
    year = valid = None
    if date_interval is not None:
        year, valid = draw_dates(date_interval.cpu(), cfg, generator)
        year, valid = year.to(device), valid.to(device)
    eps_mse = noise_prediction_loss(model, sigmas, latents, uncond if use_uncond else cond, level, noise,
                                    year, valid)
    info = {"level": level, "uncond": use_uncond, "eps_mse": float(eps_mse),
            "dates_known": bool(valid.all()) if valid is not None else None}
    return eps_mse * level_weight(float(sigmas[level]), cfg), info
