#!/usr/bin/env python
"""Fine-tune date-conditioned MVGenMaster on one scene's dataset folder (single GPU).

Trains the per-view date embedding and, with ``--lora-rank``, LoRA adapters; the released model stays
frozen (``timemachine.mvgen``; the loop and the run folder are described in
``timemachine/mvgen/trainer.py``). Runs in MVGenMaster's environment from its checkout, so give
absolute paths:

    SCRIPT=$PWD/scripts/mvgen/train.py ARGS="--run-name nd_lora32 --lora-rank 32 --resume auto" \\
        sbatch --time=1-00:00:00 --job-name=dtm_nd_lora32 slurm/mvgen_gpu.sbatch

The whole scene is sampled by co-visibility unless view sets are pinned, which is how the overfitting
tests (gates 1 and 2 of docs/mvgenmaster_time_plan.md) are run:

* ``--pinned-keys-file``: JSON list of samples ``{"refs": [file_key, ...], "targets": [file_key, ...]}``;
* ``--pinned-rows-file``: JSON list of row lists of the view table, references first.

A pinned run cycles through its view sets, measures the date-loss probe on them and renders them
(other view sets: ``--render-keys-file`` / ``--render-rows-file``); each pinned set's rows, dates and
periods are printed at start-up.

``--resume auto`` continues from the run's ``checkpoints/latest.pt`` if there is one, so the same
command can be submitted again; a run writes that file for the first time at step 0, right after
its first evaluation. A resumed launch must have the run's options: only the number of steps, the
intervals, ``--num-workers`` and the render options may change, and anything else is refused with
the names of the differing options. It must also find the dataset folder as the run found it (the
same scene-scale factor, view table, depth statistics and co-visibility). A launch that would start
from step 0 in a run folder that already holds a log or checkpoints is refused too
(``timemachine/mvgen/trainer.py``, "Starting and resuming").

``--scale`` is the scene-scale rule for camera translations and depth: ``scene`` (default: one
factor from the extent of all the scene's cameras, ``scene.json`` of the dataset folder, as in the
authors' training), ``sample`` (per sample, as in MVGenMaster's demo), or a number (that factor).

Three options strengthen the date's signal; all are off by default, and a run without them trains
exactly as before they existed (``timemachine/mvgen/trainer.py`` has the details):

* ``--region-weight W``: latent cells inside the changed region (the spire and the roofs) count W
  times in the loss. Needs ``changed_regions.npz`` in the dataset folder
  (``scripts/mvgen/prepare_regions.py``).
* ``--balance-cameras CAP``: the anchors of the training samples are drawn so that a camera says
  less about its period; CAP is the largest weight an anchor may get.
* ``--init-from CHECKPOINT``: the new run starts at step 0 from another run's trained weights (its
  ``latest.pt``: the raw weights). With ``--resume auto`` the same line can be submitted again:
  once the run has its own ``latest.pt`` it goes on from there and the other checkpoint is not read.

All three are settings of the run: a resumed launch must give them as the run was started.

Experiment look1 (all off by default; also settings of the run):

* ``--look``: every view, reference and target, is also given its look code (24 colour and medium
  statistics, ``timemachine/mvgen/look.py``), added to the timestep embedding with the date
  (``--look-injection temb``) or as a scale and shift after every ResBlock's second GroupNorm
  (``film``, the fallback). Needs ``look_codes.npz`` in the dataset folder (``scripts/mvgen/prepare_look.py``).
  ``--look-dropout``, ``--look-class-unspecified`` and ``--lr-look`` set how it trains.
* ``--medium-grey P --medium-toned Q`` (arm D, with ``--look``): each colour view becomes a grey print
  with probability P and a toned one with Q, at its unchanged date, with the code of its new pixels.
* ``--holdout-by component``: hold out whole capture days and photographers (``timemachine/data/holdout.py``):
  units are added per period until ``--holdout-fraction`` of its views and of its side views are held
  out, skipping units of more than ``--holdout-max-unit`` views; ``heldout.json`` records each view's unit.
* ``--probe-within`` (with ``--within-years``): the date-loss probes also move dates within their period.
* ``--heldout-probe-keys-file``: the held-out probe's view sets, by file key, instead of sampled ones.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from timemachine.data.holdout import HOLDOUT_BY, HoldoutConfig
from timemachine.mvgen.data import SCALE_RULES, ViewSamplingConfig
from timemachine.mvgen.date import LOOK_INJECTIONS, UNCOND_DATES
from timemachine.mvgen.model import DEFAULT_LORA_TARGETS, DEFAULT_MODEL_DIR, FinetuneConfig, assert_exact_matmul
from timemachine.mvgen.probes import parse_render_date, read_keys_file, read_rows_file
from timemachine.mvgen.step import CROSS_DATE_PIXELS, PRECISIONS, LossConfig
from timemachine.mvgen.trainer import LR_SCHEDULES, TrainConfig, Trainer, checkpoint_to_resume, parse_scale
from timemachine.training.loss import DATE_VIEWS

REPO = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line; every default is the default of the config dataclass the option belongs to."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run = p.add_argument_group("run")
    run.add_argument("--run-name", required=True, help="the run folder is <runs-root>/<run-name>")
    run.add_argument("--runs-root", type=Path, default=REPO / "data/experiments/mvgen/train")
    run.add_argument("--dataset", type=Path, default=REPO / "data/datasets/25512_model0_v2",
                     help="dataset folder (views.parquet, covisibility.npy, depth/)")
    run.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="the released model, relative to the checkout")
    run.add_argument("--resume", help="a latest.pt to continue from, or 'auto': the run's own latest.pt if it "
                                      "exists, else a fresh start")
    run.add_argument("--init-from", type=Path, default=TrainConfig.init_from,
                     help="start this NEW run at step 0 with the trained weights of another run's checkpoint "
                          "(latest.pt: its raw weights; step_XXXXXX.pt: the EMA, if that run kept one). The adapter "
                          "options (--lora-rank, --lora-targets, --date-freqs, ...) must be that run's; the "
                          "optimiser, the schedule and the EMA start anew. A resumed run does not read the file "
                          "again, but the option is one of the run's settings: give it as the run was started")
    run.add_argument("--seed", type=int, default=TrainConfig.seed)
    run.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    run.add_argument("--resolution", type=int, default=TrainConfig.resolution)
    run.add_argument("--scale", type=parse_scale, default=TrainConfig.scale,
                     help=f"scene-scale rule for camera translations and depth: one of {SCALE_RULES}, or a factor")

    opt = p.add_argument_group("optimisation")
    opt.add_argument("--steps", type=int, default=TrainConfig.steps, help="optimiser steps")
    opt.add_argument("--grad-accum", type=int, default=TrainConfig.grad_accum, help="samples per optimiser step")
    opt.add_argument("--lr-date", type=float, default=TrainConfig.lr_date)
    opt.add_argument("--lr-lora", type=float, default=TrainConfig.lr_lora)
    opt.add_argument("--lr-look", type=float, default=TrainConfig.lr_look, help="the look embedding (--look)")
    opt.add_argument("--warmup", type=int, default=TrainConfig.warmup_steps)
    opt.add_argument("--lr-schedule", choices=LR_SCHEDULES, default=TrainConfig.lr_schedule)
    opt.add_argument("--lr-min-ratio", type=float, default=TrainConfig.lr_min_ratio)
    opt.add_argument("--grad-clip-factor", type=float, default=TrainConfig.grad_clip_factor,
                     help="clip only gradients above this x the median recent norm (0: never)")
    opt.add_argument("--ema-decay", type=float, default=TrainConfig.ema_decay,
                     help="EMA of the trainable weights, used by probes and checkpoints (0: off)")
    opt.add_argument("--precision", choices=PRECISIONS, default=LossConfig.precision)

    adapters = p.add_argument_group("what trains")
    adapters.add_argument("--no-date", action="store_true", help="no date embedding (a LoRA-only control)")
    adapters.add_argument("--date-freqs", type=int, default=FinetuneConfig.date_freqs)
    adapters.add_argument("--date-hidden", type=int, default=FinetuneConfig.date_hidden)
    adapters.add_argument("--lora-rank", type=int, default=FinetuneConfig.lora_rank, help="0: the date embedding only")
    adapters.add_argument("--lora-alpha", type=float, default=FinetuneConfig.lora_alpha, help="default: the rank")
    adapters.add_argument("--lora-targets", nargs="+", default=list(DEFAULT_LORA_TARGETS))
    adapters.add_argument("--no-grad-checkpointing", action="store_true",
                          help="keep all activations for backward(): faster, much more memory")
    adapters.add_argument("--look", action="store_true",
                          help="give every view its look code (needs look_codes.npz: scripts/mvgen/prepare_look.py); "
                               "the look embedding starts as an exact no-op and is created after LoRA, so the run "
                               "starts from the date and LoRA weights of the same run without it")
    adapters.add_argument("--look-hidden", type=int, default=FinetuneConfig.look_hidden,
                          help="width of the look MLP's hidden layer")
    adapters.add_argument("--look-injection", choices=LOOK_INJECTIONS, default=FinetuneConfig.look_injection,
                          help="temb: the look embedding is added to the timestep embedding with the date; film: a "
                               "scale and shift per ResBlock after its second GroupNorm (the fallback if the look does "
                               "not steer through temb)")

    loss = p.add_argument_group("loss")
    loss.add_argument("--date-dropout", type=float, default=LossConfig.date_dropout)
    loss.add_argument("--camera-dropout", type=float, default=LossConfig.camera_dropout)
    loss.add_argument("--prior-dropout", type=float, default=LossConfig.prior_dropout)
    loss.add_argument("--prior-view-dropout", type=float, default=LossConfig.prior_view_dropout)
    loss.add_argument("--no-date-jitter", action="store_true", help="use the midpoint of each view's date interval")
    loss.add_argument("--date-views", choices=DATE_VIEWS, default=LossConfig.date_views,
                      help="targets: the references' dates are always unknown")
    loss.add_argument("--class-label", type=int, default=LossConfig.class_label,
                      help="MVGenMaster's domain label: 0 regular datasets, 1 MegaScenes")
    loss.add_argument("--cross-date-pixels", choices=CROSS_DATE_PIXELS, default=LossConfig.cross_date_pixels,
                      help="what a target loses that no reference shows as it looked (none within --cross-date-gap "
                           "years in its period): drop = its warped pixels (option B of the plan), drop_all = both "
                           "its priors (a state the released model was trained on for single views, but almost "
                           "never for all targets of a view set at once, which is what this rule gives most of "
                           "our samples: see timemachine/mvgen/step.py)")
    loss.add_argument("--cross-date-gap", type=float, default=LossConfig.cross_date_gap)
    loss.add_argument("--region-weight", type=float, default=LossConfig.region_weight,
                      help="how many times a latent cell inside the changed region counts in the loss (1: the plain "
                           "loss). The spire and the roofs are a few percent of a frame, so the plain loss hardly "
                           "notices a wrong date. Needs changed_regions.npz in the dataset folder "
                           "(scripts/mvgen/prepare_regions.py); the log then also has the plain loss, and the "
                           "plain error inside and outside the region")
    loss.add_argument("--look-dropout", type=float, default=LossConfig.look_dropout,
                      help="with --look: the share of the samples whose dates are dropped (--date-dropout) whose looks "
                           "are unknown too; the look is never dropped while the date is shown")
    loss.add_argument("--look-class-unspecified", type=float, default=LossConfig.look_class_unspecified,
                      help="with --look: the chance per view and class that its class entry is replaced by the "
                           "population mean (an unshown class is first filled from a random training view)")
    loss.add_argument("--medium-grey", type=float, default=LossConfig.medium_grey,
                      help="with --look (arm D): the chance per colour view, reference or target, that it becomes a "
                           "grey print (its luminance) at its unchanged date, with the code of the grey pixels")
    loss.add_argument("--medium-toned", type=float, default=LossConfig.medium_toned,
                      help="with --look (arm D): the chance per colour view that it becomes a toned print (its "
                           "luminance through the tone curve of a real toned print of the dataset)")

    sampling = p.add_argument_group("view sampling")
    sampling.add_argument("--n-ref", type=int, default=ViewSamplingConfig.n_ref)
    sampling.add_argument("--n-targets", type=int, default=ViewSamplingConfig.n_targets)
    sampling.add_argument("--min-shared", type=int, default=ViewSamplingConfig.min_shared)
    sampling.add_argument("--min-targets-seen", type=int, default=ViewSamplingConfig.min_targets_seen)
    sampling.add_argument("--top-k", type=int, default=ViewSamplingConfig.top_k)
    sampling.add_argument("--modern-tolerance", type=float, default=ViewSamplingConfig.modern_tolerance)
    sampling.add_argument("--old-tolerance", type=float, default=ViewSamplingConfig.old_tolerance)
    sampling.add_argument("--tolerance-boundary", type=float, default=ViewSamplingConfig.tolerance_boundary)
    sampling.add_argument("--period-edges", type=float, nargs="*", default=list(ViewSamplingConfig.period_edges),
                          help="years that cut the scene's history into periods (for cross-period references, "
                               "the hold-out, swapped dates and the cross-date rule)")
    sampling.add_argument("--anchor-period-weights", type=float, nargs="*",
                          help="probability of drawing the targets from each period (default: uniform over anchors)")
    sampling.add_argument("--cross-period-prob", type=float, default=ViewSamplingConfig.cross_period_prob)
    sampling.add_argument("--balance-cameras", type=float, default=ViewSamplingConfig.balance_cameras,
                          help="0: off. Otherwise the largest weight an anchor may get when anchors are weighted "
                               "so that a camera says less about its period: a view whose 15 nearest training "
                               "views (by position and viewing direction) are mostly of other periods is drawn "
                               "more often, by one over the share that are of its own. The numbers that help to "
                               "choose it are printed under the period table at start-up; they are about the "
                               "anchors. The other targets of a sample are not weighted, so among all targets a "
                               "camera still says more (timemachine/mvgen/data.py has what was measured: 5 was "
                               "the cap to use at Notre-Dame)")
    pinned = sampling.add_mutually_exclusive_group()
    pinned.add_argument("--pinned-rows-file", type=Path, help="train on exactly these view sets (JSON, by row)")
    pinned.add_argument("--pinned-keys-file", type=Path, help="train on exactly these view sets (JSON, by file key)")
    sampling.add_argument("--holdout-fraction", type=float, default=HoldoutConfig.fraction,
                          help="share of each period's views held out of training")
    sampling.add_argument("--holdout-probe-targets", action="store_true", help="also hold out the renders' targets")
    sampling.add_argument("--holdout-seed", type=int, default=HoldoutConfig.seed)
    sampling.add_argument("--holdout-by", choices=HOLDOUT_BY, default=HoldoutConfig.by,
                          help="view: random views per period; component: whole capture days and photographers "
                               "(units linked by day or Commons author), added per period until --holdout-fraction "
                               "of its views and of its side views are held out")
    sampling.add_argument("--holdout-max-unit", type=int, default=HoldoutConfig.max_unit,
                          help="component: units of more views than this are never held out")
    sampling.add_argument("--holdout-side-share", type=float, default=HoldoutConfig.side_share,
                          help="component: a side view's stored changed-region mask covers this share of the frame")
    sampling.add_argument("--holdout-unit-periods", type=int, nargs="*", default=list(HoldoutConfig.unit_periods),
                          help="component: the periods (0 = the oldest) held out by a share of their units instead")

    probes = p.add_argument_group("logs, probes and checkpoints (intervals in optimiser steps; 0: off)")
    probes.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    probes.add_argument("--probe-every", type=int, default=TrainConfig.probe_every, help="date-loss probes")
    probes.add_argument("--probe-samples", type=int, default=TrainConfig.probe_samples)
    probes.add_argument("--probe-timesteps", type=int, nargs="+", default=list(TrainConfig.probe_timesteps))
    probes.add_argument("--swap-years", type=float, nargs="*",
                        help="one year per period for swapped dates (default: the median date trained on in each)")
    probes.add_argument("--probe-within", action="store_true",
                        help="date-loss probes also move each target's date within its period (the null of the swap)")
    probes.add_argument("--within-years", type=float, nargs="*",
                        help="two years per period, in order, for --probe-within: a target goes to the one farther "
                             "from its date (default: the 25th and 75th percentiles of the training dates)")
    probes.add_argument("--heldout-probe-keys-file", type=Path,
                        help="the held-out date-loss probe's view sets (JSON, by file key) instead of sets sampled "
                             "among the held-out views")
    probes.add_argument("--render-every", type=int, default=TrainConfig.render_every)
    rendered = probes.add_mutually_exclusive_group()
    rendered.add_argument("--render-rows-file", type=Path, help="view sets to render (JSON, by row)")
    rendered.add_argument("--render-keys-file", type=Path, help="view sets to render (JSON, by file key)")
    probes.add_argument("--render-dates", type=parse_render_date, nargs="+", default=list(TrainConfig.render_dates),
                        help="years, or: true (each target's own date), swapped (its date in another period), unknown")
    probes.add_argument("--render-seeds", type=int, nargs="+", default=list(TrainConfig.render_seeds))
    probes.add_argument("--render-steps", type=int, default=TrainConfig.render_steps)
    probes.add_argument("--render-guidance", type=float, default=TrainConfig.render_guidance)
    probes.add_argument("--render-uncond-dates", choices=UNCOND_DATES, default=TrainConfig.render_uncond_dates,
                        help="what the unconditional half of the guidance batch sees")
    probes.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    probes.add_argument("--latest-every", type=int, default=TrainConfig.latest_every)
    return p.parse_args(argv)


def view_sets(dataset: Path, rows_file: Path | None, keys_file: Path | None, n_ref: int) -> tuple[tuple[int, ...], ...]:
    """The view sets of a rows file or a keys file (resolved against the dataset's view table); none without a file."""
    if rows_file is not None:
        return read_rows_file(rows_file)
    if keys_file is not None:
        table_keys = pd.read_parquet(dataset / "views.parquet", columns=["file_key"])["file_key"].tolist()
        return read_keys_file(keys_file, table_keys, n_ref)
    return ()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """The run's configuration from the parsed command line."""
    edges = tuple(args.period_edges)
    if args.within_years and len(args.within_years) % 2:
        raise ValueError("--within-years takes two years per period")
    return TrainConfig(
        dataset=str(args.dataset), run_dir=str(args.runs_root / args.run_name), model_dir=args.model_dir,
        resolution=args.resolution, scale=args.scale, steps=args.steps, lr_date=args.lr_date, lr_lora=args.lr_lora,
        warmup_steps=args.warmup, lr_schedule=args.lr_schedule, lr_min_ratio=args.lr_min_ratio,
        grad_accum=args.grad_accum, grad_clip_factor=args.grad_clip_factor or None, ema_decay=args.ema_decay,
        seed=args.seed, num_workers=args.num_workers, log_every=args.log_every, probe_every=args.probe_every,
        render_every=args.render_every, save_every=args.save_every, latest_every=args.latest_every,
        probe_samples=args.probe_samples, probe_timesteps=tuple(args.probe_timesteps),
        swap_years=tuple(args.swap_years) if args.swap_years else None,
        lr_look=args.lr_look, probe_within=args.probe_within,
        within_years=(tuple(zip(args.within_years[::2], args.within_years[1::2])) if args.within_years else None),
        heldout_probe_keys=None if args.heldout_probe_keys_file is None else str(args.heldout_probe_keys_file),
        render_rows=view_sets(args.dataset, args.render_rows_file, args.render_keys_file, args.n_ref),
        render_dates=tuple(args.render_dates), render_seeds=tuple(args.render_seeds), render_steps=args.render_steps,
        render_guidance=args.render_guidance, render_uncond_dates=args.render_uncond_dates,
        sampling=ViewSamplingConfig(
            n_ref=args.n_ref, n_targets=args.n_targets, min_shared=args.min_shared,
            min_targets_seen=args.min_targets_seen, top_k=args.top_k, modern_tolerance=args.modern_tolerance,
            old_tolerance=args.old_tolerance, tolerance_boundary=args.tolerance_boundary, period_edges=edges,
            anchor_period_weights=tuple(args.anchor_period_weights) if args.anchor_period_weights else None,
            cross_period_prob=args.cross_period_prob, balance_cameras=args.balance_cameras),
        loss=LossConfig(
            date_dropout=args.date_dropout, camera_dropout=args.camera_dropout, prior_dropout=args.prior_dropout,
            prior_view_dropout=args.prior_view_dropout, date_jitter=not args.no_date_jitter,
            date_views=args.date_views, class_label=args.class_label, cross_date_pixels=args.cross_date_pixels,
            cross_date_gap=args.cross_date_gap, precision=args.precision, region_weight=args.region_weight,
            look_dropout=args.look_dropout, look_class_unspecified=args.look_class_unspecified,
            medium_grey=args.medium_grey, medium_toned=args.medium_toned),
        finetune=FinetuneConfig(
            date=not args.no_date, date_freqs=args.date_freqs, date_hidden=args.date_hidden, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_targets=tuple(args.lora_targets),
            grad_checkpointing=not args.no_grad_checkpointing, look=args.look, look_hidden=args.look_hidden,
            look_injection=args.look_injection),
        holdout=HoldoutConfig(fraction=args.holdout_fraction, probe_targets=args.holdout_probe_targets,
                              seed=args.holdout_seed, era_edges=edges,      # held out per period
                              by=args.holdout_by, max_unit=args.holdout_max_unit, side_share=args.holdout_side_share,
                              unit_periods=tuple(args.holdout_unit_periods)),
        pinned_rows=view_sets(args.dataset, args.pinned_rows_file, args.pinned_keys_file, args.n_ref),
        init_from=None if args.init_from is None else str(args.init_from))


def main(argv: list[str] | None = None) -> None:
    assert_exact_matmul()
    args = parse_args(argv)
    cfg = build_config(args)
    Trainer(cfg, resume=checkpoint_to_resume(cfg.run_dir, args.resume)).train()


if __name__ == "__main__":
    main()
