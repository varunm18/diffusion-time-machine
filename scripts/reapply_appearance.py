#!/usr/bin/env python
"""Re-apply appearance conditioning on top of the 2026-11-19 audit, instead of over it.

Commit 74f40d1 ("Add an appearance conditioning pathway ...") was written against a
snapshot of the repo that predated a05144e ("Pipeline audit: fp32 training, 21-frame
renders"). Being a whole-file rewrite rather than a patch, it silently reverted that
audit in three files:

  * loss.py      TRAIN_AUTOCAST_DTYPE back to bf16, the autocast() helper deleted,
                 rank_reference / rank_detach and their guard deleted, and the
                 reference pass moved after the true-date pass with its gradient
                 restored - the exact ordering bug the audit had fixed.
  * trainer.py   autocast default back to bf16, the checkpoint precision guard,
                 grad_norm_history persistence and render_frames=21 deleted.
  * train_dated_seva.py   --rank-reference, --rank-detach, --render-frames deleted.

The three files have been restored from a05144e. This script adds back only what the
appearance pathway genuinely needs, as anchored edits. Every anchor must appear
exactly once or nothing is written, so a half-applied repair is not possible.

Run once, from the repo root:

    python scripts/reapply_appearance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

Edit = tuple[str, str]


def apply(path: Path, edits: list[Edit]) -> str:
    """Apply anchored edits, or raise before writing anything."""
    text = original = path.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(edits, start=1):
        found = text.count(old)
        if found != 1:
            raise SystemExit(
                f"\n{path}: edit {i} of {len(edits)} matched {found} times, expected 1."
                f"\nNothing was written. The anchor was:\n\n{old}\n"
                f"\nIf the replacement text is already in the file, this repair has been"
                f" applied once and does not need running again. Otherwise the file is not"
                f" the restored a05144e version; check with:"
                f"\n  git diff --stat a05144e -- {path}")
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
    return "changed" if text != original else "already applied"


# --------------------------------------------------------------------------- loss.py

LOSS: list[Edit] = [
    # The dropout share, beside the date's.
    ("""    date_dropout: float = 0.1    # share of samples whose dates are all marked unknown
""",
     """    date_dropout: float = 0.1    # share of samples whose dates are all marked unknown
    appearance_dropout: float = 0.1   # share of samples whose appearance is marked unknown
"""),

    # predict_noise: carry the appearance through to the model call.
    ("""def predict_noise(model: nn.Module, noisy: torch.Tensor, sigma: torch.Tensor, t: torch.Tensor, cond: dict,
                  year: torch.Tensor | None = None, date_valid: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:""",
     """def predict_noise(model: nn.Module, noisy: torch.Tensor, sigma: torch.Tensor, t: torch.Tensor, cond: dict,
                  year: torch.Tensor | None = None, date_valid: torch.Tensor | None = None,
                  appearance: torch.Tensor | None = None,
                  appearance_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:"""),

    ("""        year, date_valid: (T,) per-view dates for the dated model (omit for plain SEVA).
""",
     """        year, date_valid: (T,) per-view dates for the dated model (omit for plain SEVA).
        appearance, appearance_valid: (T, F) per-view appearance statistics and their
            validity, for a model with an appearance pathway (omit otherwise).
"""),

    ("""    date_kwargs = {} if year is None else {"date": year, "date_valid": date_valid}
    eps_hat = model(x * c_in, t, cond, num_frames=noisy.shape[0], **date_kwargs)""",
     """    extra = {} if year is None else {"date": year, "date_valid": date_valid}
    if appearance is not None:
        extra |= {"appearance": appearance, "appearance_valid": appearance_valid}
    eps_hat = model(x * c_in, t, cond, num_frames=noisy.shape[0], **extra)"""),

    # per_view_noise_error and noise_prediction_loss: pass it down.
    ("""def per_view_noise_error(model: nn.Module, sigmas: torch.Tensor, latents: torch.Tensor, cond: dict,
                         level: int, noise: torch.Tensor, year: torch.Tensor | None = None,
                         date_valid: torch.Tensor | None = None) -> torch.Tensor:""",
     """def per_view_noise_error(model: nn.Module, sigmas: torch.Tensor, latents: torch.Tensor, cond: dict,
                         level: int, noise: torch.Tensor, year: torch.Tensor | None = None,
                         date_valid: torch.Tensor | None = None,
                         appearance: torch.Tensor | None = None,
                         appearance_valid: torch.Tensor | None = None) -> torch.Tensor:"""),

    ("""    eps_hat, replaced = predict_noise(model, noisy, sigma, t, cond, year, date_valid)""",
     """    eps_hat, replaced = predict_noise(model, noisy, sigma, t, cond, year, date_valid,
                                      appearance, appearance_valid)"""),

    ("""def noise_prediction_loss(model: nn.Module, sigmas: torch.Tensor, latents: torch.Tensor, cond: dict,
                          level: int, noise: torch.Tensor, year: torch.Tensor | None = None,
                          date_valid: torch.Tensor | None = None) -> torch.Tensor:""",
     """def noise_prediction_loss(model: nn.Module, sigmas: torch.Tensor, latents: torch.Tensor, cond: dict,
                          level: int, noise: torch.Tensor, year: torch.Tensor | None = None,
                          date_valid: torch.Tensor | None = None,
                          appearance: torch.Tensor | None = None,
                          appearance_valid: torch.Tensor | None = None) -> torch.Tensor:"""),

    ("""    return per_view_noise_error(model, sigmas, latents, cond, level, noise, year, date_valid).mean()""",
     """    return per_view_noise_error(model, sigmas, latents, cond, level, noise, year, date_valid,
                                appearance, appearance_valid).mean()


def draw_appearance_validity(n_views: int, cfg: LossConfig, device: torch.device,
                             generator: torch.Generator | None = None) -> torch.Tensor:
    \"\"\"(T,) validity for the appearance pathway; whole samples are dropped together.

    Dropping teaches the model to run without an appearance being chosen, which it
    must do when sampling asks only for a date. Unlike the dates, the appearance is
    *kept* in the unconditional branch of classifier-free guidance: WildCAT3D found
    that dropping an appearance code there oversaturated its renders.
    \"\"\"
    drop = torch.rand((), generator=generator) < cfg.appearance_dropout
    return torch.full((n_views,), not bool(drop), dtype=torch.bool, device=device)"""),

    # training_loss: accept the appearance and draw its validity once, before the
    # rank branch, so the reference and true-date passes differ only in the date.
    ("""                  generator: torch.Generator | None = None,
                  is_input: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:""",
     """                  generator: torch.Generator | None = None,
                  is_input: torch.Tensor | None = None,
                  appearance: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:"""),

    ("""    rank = (cfg.rank_weight > 0 and not use_uncond and valid is not None and bool(valid.any())""",
     """    # Drawn once for the whole step: the rank loss compares two passes that must
    # differ only in their dates, so both see the same appearance and the same
    # validity. Drawing inside the branches would make the comparison measure the
    # appearance too.
    appearance_valid = None
    if appearance is not None:
        appearance = appearance.to(device)
        appearance_valid = draw_appearance_validity(latents.shape[0], cfg, device, generator)
    rank = (cfg.rank_weight > 0 and not use_uncond and valid is not None and bool(valid.any())"""),

    # Both passes carry it. The reference pass stays first and gradient-free.
    ("""            errors_ref = per_view_noise_error(model, sigmas, latents, cond, level, noise, ref_year, ref_valid)
    errors = per_view_noise_error(model, sigmas, latents, uncond if use_uncond else cond, level, noise, year, valid)""",
     """            errors_ref = per_view_noise_error(model, sigmas, latents, cond, level, noise, ref_year, ref_valid,
                                              appearance, appearance_valid)
    errors = per_view_noise_error(model, sigmas, latents, uncond if use_uncond else cond, level, noise, year, valid,
                                  appearance, appearance_valid)"""),

    ("""            "dates_known": bool(valid.any()) if valid is not None else None}""",
     """            "dates_known": bool(valid.any()) if valid is not None else None,
            "appearance_known": bool(appearance_valid.any()) if appearance_valid is not None else None}"""),
]

# ------------------------------------------------------------------------ trainer.py

TRAINER: list[Edit] = [
    ("""from timemachine.data.dataset import DatedViewsDataset""",
     """from timemachine.data.appearance import dataset_appearance_stats
from timemachine.data.dataset import DatedViewsDataset"""),

    ("""    lr_film: float = 1e-5                           # date FiLM (per-block layers act like LoRA: keep it low)""",
     """    lr_film: float = 1e-5                           # date FiLM (per-block layers act like LoRA: keep it low)
    lr_appearance: float = 1e-5                     # appearance FiLM (same shape as date FiLM: keep it low)"""),

    ("""    group_weight_power: float = 0.5                 # scene sampling weight = training views ** this""",
     """    group_weight_power: float = 0.5                 # scene sampling weight = training views ** this
    appearance_stats_samples: int = 64              # samples used to standardize the appearance encoder"""),

    ("""        views = load_views(cfg.dataset)
        self.dataset = DatedViewsDataset(views, cfg.sampling, resolution=cfg.resolution, samples_per_epoch=1000,
                                         seed=cfg.seed, camera_scale=cfg.camera_scale,
                                         group_weight_power=cfg.group_weight_power)""",
     """        views = load_views(cfg.dataset)
        # Appearance statistics are computed per view only when a model asks for them.
        self.with_appearance = cfg.finetune.appearance_injection != "none"
        self.dataset = DatedViewsDataset(views, cfg.sampling, resolution=cfg.resolution, samples_per_epoch=1000,
                                         seed=cfg.seed, camera_scale=cfg.camera_scale,
                                         group_weight_power=cfg.group_weight_power,
                                         with_appearance=self.with_appearance)"""),

    ("""        lrs = {"date": cfg.lr_date, "film": cfg.lr_film, "lora": cfg.lr_lora}""",
     """        if self.with_appearance:
            # The raw statistics are on very different scales; standardize the encoder on
            # the data this run will actually see. The buffers travel with the checkpoint.
            self.model.date_module.appearance_encoder.fit_normalization(
                dataset_appearance_stats(self.dataset, n_samples=cfg.appearance_stats_samples, seed=cfg.seed))
        lrs = {"date": cfg.lr_date, "film": cfg.lr_film, "lora": cfg.lr_lora,
               "appearance": cfg.lr_appearance}"""),

    # Keep Laith's autocast() wrapper; only add the appearance argument.
    ("""                    loss, info = training_loss(self.model, self.sigmas, latents, cond, uncond,
                                               sample["date"], cfg.loss, gen, is_input=sample["is_input"])""",
     """                    loss, info = training_loss(self.model, self.sigmas, latents, cond, uncond,
                                               sample["date"], cfg.loss, gen, is_input=sample["is_input"],
                                               appearance=sample.get("appearance"))"""),
]

# --------------------------------------------------------------- train_dated_seva.py

SCRIPT: list[Edit] = [
    ("""    p.add_argument("--film-dim", type=int, default=FinetuneConfig.film_dim)""",
     """    p.add_argument("--appearance-injection", choices=["film", "none"],
                   default=FinetuneConfig.appearance_injection,
                   help="how a view's appearance (colour, tone, grain) reaches the model: a per-block "
                        "scale-and-shift, or not at all. Our dates are largely readable from the medium "
                        "alone (docs/findings.md), so without this the date path can learn film stock.")
    p.add_argument("--appearance-dim", type=int, default=FinetuneConfig.appearance_dim)
    p.add_argument("--appearance-dropout", type=float, default=LossConfig.appearance_dropout,
                   help="share of samples whose appearance is marked unknown")
    p.add_argument("--lr-appearance", type=float, default=TrainConfig.lr_appearance)
    p.add_argument("--film-dim", type=int, default=FinetuneConfig.film_dim)"""),

    ("""        lr_lora=args.lr_lora, lr_date=args.lr_date, lr_film=args.lr_film, warmup_steps=args.warmup, grad_accum=args.grad_accum,""",
     """        lr_lora=args.lr_lora, lr_date=args.lr_date, lr_film=args.lr_film, lr_appearance=args.lr_appearance,
        warmup_steps=args.warmup, grad_accum=args.grad_accum,"""),

    ("""                        rank_min_sigma=args.rank_min_sigma, rank_reference=args.rank_reference,
                        rank_detach=args.rank_detach),""",
     """                        rank_min_sigma=args.rank_min_sigma, rank_reference=args.rank_reference,
                        rank_detach=args.rank_detach, appearance_dropout=args.appearance_dropout),"""),

    ("""                                film_dim=args.film_dim, film_freqs=args.film_freqs),""",
     """                                film_dim=args.film_dim, film_freqs=args.film_freqs,
                                appearance_injection=args.appearance_injection,
                                appearance_dim=args.appearance_dim),"""),
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    targets = [
        (root / "src/timemachine/training/loss.py", LOSS),
        (root / "src/timemachine/training/trainer.py", TRAINER),
        (root / "scripts/train_dated_seva.py", SCRIPT),
    ]
    missing = [p for p, _ in targets if not p.is_file()]
    if missing:
        raise SystemExit(f"not found: {', '.join(str(p) for p in missing)}")

    for path, edits in targets:
        print(f"  {path.relative_to(root)}: {apply(path, edits)} ({len(edits)} edits)")
    print("\nDone. Verify with:\n"
          "  python -m pytest tests/ -q\n"
          "  python scripts/train_dated_seva.py --help | head -5\n"
          "  git diff --stat a05144e -- src/timemachine/training/loss.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
