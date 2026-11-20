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
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from timemachine.data.holdout import HoldoutConfig
from timemachine.mvgen.data import SCALE_RULES, ViewSamplingConfig
from timemachine.mvgen.date import UNCOND_DATES
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
    pinned = sampling.add_mutually_exclusive_group()
    pinned.add_argument("--pinned-rows-file", type=Path, help="train on exactly these view sets (JSON, by row)")
    pinned.add_argument("--pinned-keys-file", type=Path, help="train on exactly these view sets (JSON, by file key)")
    sampling.add_argument("--holdout-fraction", type=float, default=HoldoutConfig.fraction,
                          help="share of each period's views held out of training")
    sampling.add_argument("--holdout-probe-targets", action="store_true", help="also hold out the renders' targets")
    sampling.add_argument("--holdout-seed", type=int, default=HoldoutConfig.seed)

    probes = p.add_argument_group("logs, probes and checkpoints (intervals in optimiser steps; 0: off)")
    probes.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    probes.add_argument("--probe-every", type=int, default=TrainConfig.probe_every, help="date-loss probes")
    probes.add_argument("--probe-samples", type=int, default=TrainConfig.probe_samples)
    probes.add_argument("--probe-timesteps", type=int, nargs="+", default=list(TrainConfig.probe_timesteps))
    probes.add_argument("--swap-years", type=float, nargs="*",
                        help="one year per period for swapped dates (default: the median date trained on in each)")
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
    return TrainConfig(
        dataset=str(args.dataset), run_dir=str(args.runs_root / args.run_name), model_dir=args.model_dir,
        resolution=args.resolution, scale=args.scale, steps=args.steps, lr_date=args.lr_date, lr_lora=args.lr_lora,
        warmup_steps=args.warmup, lr_schedule=args.lr_schedule, lr_min_ratio=args.lr_min_ratio,
        grad_accum=args.grad_accum, grad_clip_factor=args.grad_clip_factor or None, ema_decay=args.ema_decay,
        seed=args.seed, num_workers=args.num_workers, log_every=args.log_every, probe_every=args.probe_every,
        render_every=args.render_every, save_every=args.save_every, latest_every=args.latest_every,
        probe_samples=args.probe_samples, probe_timesteps=tuple(args.probe_timesteps),
        swap_years=tuple(args.swap_years) if args.swap_years else None,
        render_rows=view_sets(args.dataset, args.render_rows_file, args.render_keys_file, args.n_ref),
        render_dates=tuple(args.render_dates), render_seeds=tuple(args.render_seeds), render_steps=args.render_steps,
        render_guidance=args.render_guidance, render_uncond_dates=args.render_uncond_dates,
        sampling=ViewSamplingConfig(
            n_ref=args.n_ref, n_targets=args.n_targets, min_shared=args.min_shared,
            min_targets_seen=args.min_targets_seen, top_k=args.top_k, modern_tolerance=args.modern_tolerance,
            old_tolerance=args.old_tolerance, tolerance_boundary=args.tolerance_boundary, period_edges=edges,
            anchor_period_weights=tuple(args.anchor_period_weights) if args.anchor_period_weights else None,
            cross_period_prob=args.cross_period_prob),
        loss=LossConfig(
            date_dropout=args.date_dropout, camera_dropout=args.camera_dropout, prior_dropout=args.prior_dropout,
            prior_view_dropout=args.prior_view_dropout, date_jitter=not args.no_date_jitter,
            date_views=args.date_views, class_label=args.class_label, cross_date_pixels=args.cross_date_pixels,
            cross_date_gap=args.cross_date_gap, precision=args.precision),
        finetune=FinetuneConfig(
            date=not args.no_date, date_freqs=args.date_freqs, date_hidden=args.date_hidden, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_targets=tuple(args.lora_targets),
            grad_checkpointing=not args.no_grad_checkpointing),
        holdout=HoldoutConfig(fraction=args.holdout_fraction, probe_targets=args.holdout_probe_targets,
                              seed=args.holdout_seed, era_edges=edges),      # held out per period
        pinned_rows=view_sets(args.dataset, args.pinned_rows_file, args.pinned_keys_file, args.n_ref))


def main(argv: list[str] | None = None) -> None:
    assert_exact_matmul()
    args = parse_args(argv)
    cfg = build_config(args)
    Trainer(cfg, resume=checkpoint_to_resume(cfg.run_dir, args.resume)).train()


if __name__ == "__main__":
    main()
