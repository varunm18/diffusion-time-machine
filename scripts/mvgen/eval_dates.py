#!/usr/bin/env python
"""Does the date matter where the scene changed? Loss and renders, inside and outside the changed region (single GPU).

Whole-frame scores barely register Notre-Dame's fire, because the spire and the roofs are a few
percent of a picture. This script scores a model separately ``inside`` the region that changed at
the fire (the masks of ``timemachine.mvgen.regions``), ``outside`` it and over the ``whole`` frame
(``timemachine.mvgen.evaluation``, whose docstring has the definitions):

* **loss with fixed noise**: the training loss on the targets with their true dates, with their dates
  swapped to the period across the nearest edge, and with no dates, on the same noisy latents; and
  the relative gaps ``swapped / true - 1`` and ``unknown / true - 1`` (above zero: the true date gives
  the lower loss), per area, with the standard error over view sets;
* **renders**: every view set generated from its references at its true dates and at its swapped
  dates, with the same noise seeds; the PSNR against the real photos per area, and how much the two
  renders differ per area (mean absolute difference, 0-1), which needs no real photo: if the date
  moves the building and nothing else, it is large inside the region and small outside it.
  For a run that takes priors from cross-date targets (``cross_date_pixels`` ``drop`` or
  ``drop_all``) the two renders also differ in their priors, because the run's render probe applies
  that rule by the date it renders at; the script says so in a note that names the view sets.
  ``--priors-by true`` gives both renders the priors of the true dates, as the loss measurements
  have them: use it to see what the date alone does (the whole-frame PSNR at the swapped dates is
  then no longer the render probe's).

**Model.** ``--run`` names a run folder of ``scripts/mvgen/train.py``; the dataset, the scene-scale
rule, the loss settings (precision, domain label, which views show their date, the cross-date rule),
the period edges, the years for swapped dates (the run's ``probes.json``) and the hold-out are then
the run's, and so are, unless given here, the probe timesteps and the sampler settings of its render
probes. ``--checkpoint`` picks the weights: ``latest`` (default), a step number
(``checkpoints/step_XXXXXX.pt``), a file, or ``released``: the released model under the run's
settings and on the same view sets, which is the run's baseline. A checkpoint is loaded as the
trainer's probes see the model: its ``trainable`` entry, the EMA of the weights when the run keeps
one (``results.json`` says which under ``model.weights``). Without ``--run`` the released model is
evaluated with ``--dataset``, ``--resolution``, ``--scale``, ``--class-label``, ``--precision`` and
``--swap-years`` (as ``scripts/mvgen/render.py``). The released model has no date input, so its
loss gaps are zero and its two renders differ only by what a GPU does not reproduce to the bit.

**View sets.** Those of the keys files (``--keys-file``, repeatable; default
``configs/mvgen_gate2.json``; ``none`` for no keys file), then ``--heldout N`` sets whose targets the
run held out of training, drawn as the run's held-out date-loss probe draws its sets (the first N
of the same sequence, so ``heldout/0`` is that probe's first set). ``--cross-only`` takes from that
sequence only the sets that cross the edge: every target dated after the fire, every reference
before it. It does not limit the sets of keys files. Every set is marked ``cross`` or ``other``; the
tables give the two groups apart, and the list printed at the start says for every set where it
comes from, its dates, how many of its targets the run held out, and the size of each target's
region. ``--list-sets`` prints that list and stops: no GPU, nothing written.

**Output** under ``--out``. Give every evaluation its own folder: ``results.json`` and ``losses.jsonl`` are
written anew by every invocation, but the pictures of view sets or seeds that a later invocation does
not render would stay behind.

* ``results.json``: ``model``, ``settings``, ``region``; ``sets``, per view set its views, its numbers
  per area (``loss``, ``renders``) and the raw render scores per seed and target
  (``render_records``); and ``loss`` and ``renders``, the tables: per group of sets and area the mean
  over view sets and its standard error. It is rewritten after every rendered set, so a job that
  runs out of time leaves the sets it finished;
* ``losses.jsonl``: one line per view set, timestep, noise seed and date variant;
* ``renders/<set>/``: ``<variant>_seed<k>_view<j>.jpg`` (the rendered targets; ``j`` is the view's
  position in the set), ``real_view<j>.jpg`` and ``mask_view<j>.png`` (the region as scored);
* ``sheets/<set>.jpg``: one small sheet per set (rows: the real photos, the true-date render, the
  swapped-date render, the region; at most five targets), of the first seed.

A target whose region is smaller than ``--min-mask-pixels`` is not scored inside (its whole frame
counts as outside). The list marks such targets and counts them (``region.targets_not_scored`` in
``results.json``), and the tables' ``sets`` column counts the view sets behind each number. The
default of 256 pixels suits the lines below, whose targets all have regions of 6% of the frame or
more. Frontal views of the west facade have masks of specks, often larger than 256 pixels: for view
sets with such targets (``--heldout`` without ``--cross-only`` draws some) give
``--min-mask-pixels 5243``, 2% of a 512 x 512 frame.

Runs in MVGenMaster's environment from its checkout, so give absolute paths. Lines for the GPU task
queue (``slurm/mvgen_gpu_queue.sbatch``: one task per line, no quotes), a run's checkpoint and then
its baseline on the same view sets:

    /fs/nexus-scratch/ltahboub/diffusion-time-machine/scripts/mvgen/eval_dates.py --run /fs/nexus-scratch/ltahboub/diffusion-time-machine/data/experiments/mvgen/train/gate3_keep --checkpoint 1000 --heldout 8 --cross-only --out /fs/nexus-scratch/ltahboub/diffusion-time-machine/data/experiments/mvgen/eval_dates/gate3_keep_step1000
    /fs/nexus-scratch/ltahboub/diffusion-time-machine/scripts/mvgen/eval_dates.py --run /fs/nexus-scratch/ltahboub/diffusion-time-machine/data/experiments/mvgen/train/gate3_keep --checkpoint released --heldout 8 --cross-only --out /fs/nexus-scratch/ltahboub/diffusion-time-machine/data/experiments/mvgen/eval_dates/gate3_keep_released

With ``--list-sets`` the same options run on a login node, to see the view sets first (``--out`` is
then not written to):

    cd external/MVGenMaster && PYTHONPATH=$PWD .venv/bin/python $REPO/scripts/mvgen/eval_dates.py --list-sets \\
        --run $REPO/data/experiments/mvgen/train/gate3_keep --heldout 8 --cross-only --out $REPO/data/unused

**Weights.** ``--weights raw`` evaluates the raw (not averaged) weights, which only a ``latest.pt`` of a
run with an EMA holds; ``--expect-step N`` refuses a checkpoint of another step (a running job
overwrites its ``latest.pt``). The default, the checkpoint's evaluated weights, is what it always was.

**Experiment look1** (``--eval-sets configs/look1_eval_sets.json``, with ``--run``). The view sets, the
loss and the renders above are replaced by look1's (``timemachine.mvgen.evaluation``, "Experiment
look1", has the definitions): the sets of the file's 2x2 (``--cells`` some of them, ``--strict`` the
sets whose five targets are all held out), checked against the run's hold-out; every set rendered under
the conditions that ``--readouts`` (default: R1 to R6 and the weight-level date profile) need, with
``--primary-seeds`` (4) for the base look at the true, cross-fire and within-period dates and
``--other-seeds`` (2) for the rest; the loss with fixed noise at the base look for R2; the colour-blind
state read-out from ``--readout-probe`` (default ``<dataset>/state_readout.npz``,
``scripts/mvgen/fit_readout.py``), which is refused if it was fitted on any scored target. What a model
cannot have is skipped with a note: a run without a look (arm A) has no R1, R4, R5, neutral or own
code; the released model (``--checkpoint released``) has R6 alone. Printed: each read-out with the
spec's pass line beside it, then with ``--compare ARM=results.json`` (other arms at the same step: ``A``,
``A2``, ``B``, ``released``) the cross-arm lines, paired by target. ``--from-results results.json``
prints them again without a model. ``--list-sets`` prints the sets, the renders and the estimated GPU
time (22.25 s per rendered set, 0.233 s per loss pass) and stops.

Output under ``--out``: ``results.json`` (``records``: per condition, seed and scored target its scores;
``pairs``: the swaps' render differences; ``look1``: the read-outs, with their per-target values;
``date_profile``), ``losses.jsonl`` and ``renders/<set>/<look>_<date>_seed<k>_view<j>.jpg`` of the
scored targets. The queue lines of experiment look1 are in ``data/look1_build/evals.txt``.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from timemachine.mvgen.data import SCALE_RULES
from timemachine.mvgen.date import UNCOND_DATES
from timemachine.mvgen.evaluation import (
    HELDOUT_SOURCE, LOOK1_READOUTS, PRIORS_BY, RELEASED, RENDER_DATES, LookInputs, ViewSet, check_eval_sets,
    check_run_records, checkpoint_path, collect_view_sets, compare_look1, date_profile, describe_set, format_table,
    look1_conditions, look1_lines, measure_look1_losses, measure_losses, open_scene, priors_differ, read_eval_sets,
    render_look1_set, render_set, run_config, run_swap_years, stored_masks, summarise_groups, summarise_look1,
    summarise_set_losses, summarise_set_renders, target_masks,
)
from timemachine.mvgen.model import assert_exact_matmul, load_finetuned, load_mvgen, load_trainable_state_dict
from timemachine.mvgen.probes import RenderProbe, describe_views, representative_years
from timemachine.mvgen.regions import ChangedRegions
from timemachine.mvgen.step import PRECISIONS, LossConfig
from timemachine.mvgen.trainer import TrainConfig, parse_scale

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / "data/datasets/25512_model0_v2"
DEFAULT_KEYS_FILE = REPO / "configs/mvgen_gate2.json"
NO_KEYS_FILE = "none"                               # the --keys-file value that switches the default file off
WEIGHTS = ("ema", "raw")                            # the checkpoint's evaluated weights, or a latest.pt's raw ones
READOUT_FILE = "state_readout.npz"                  # the state read-out in the dataset folder (scripts/mvgen/fit_readout.py)
# Measured GPU times (A6000, data/look1_critique/feasibility/07_costs.py): per rendered view set, per loss pass.
SECONDS_PER_RENDER, SECONDS_PER_PASS = 22.25, 0.233
# The settings that are given on the command line without --run, and are the run's with it.
RELEASED_OPTIONS = ("dataset", "resolution", "scale", "class_label", "precision", "swap_years")
# The tables: (column heading, the number's name in evaluation.summarise_groups, its format).
LOSS_COLUMNS = (("swapped date", "gap_swapped", "+.2%"), ("unknown date", "gap_unknown", "+.2%"),
                ("loss at the true date", "loss_true", ".4f"))
RENDER_COLUMNS = (("PSNR, true date", "psnr_true", ".2f"), ("PSNR, swapped date", "psnr_swapped", ".2f"),
                  ("true - swapped", "psnr_gain", "+.2f"), ("render difference", "difference", ".4f"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="output folder")

    model = p.add_argument_group("model")
    model.add_argument("--run", type=Path,
                       help="a run folder of scripts/mvgen/train.py: its settings, hold-out and swap years are used")
    model.add_argument("--checkpoint",
                       help=f"with --run: latest (default), a step number, a checkpoint file, or {RELEASED} (the "
                            "released model under the run's settings: the run's baseline)")
    model.add_argument("--model-dir", help="the released model, relative to the checkout (default: the run's)")
    model.add_argument("--weights", choices=WEIGHTS, default=WEIGHTS[0],
                       help="ema (default): the checkpoint's evaluated weights (the EMA when the run keeps one, as the "
                            "trainer's probes); raw: the raw weights, which only a latest.pt holds")
    model.add_argument("--expect-step", type=int,
                       help="refuse a checkpoint of another step (for latest.pt, which a running job overwrites)")

    released = p.add_argument_group("settings of the released model (without --run)")
    released.add_argument("--dataset", type=Path, help=f"dataset folder (default: {DEFAULT_DATASET})")
    released.add_argument("--resolution", type=int,
                          help=f"side of the crops; the depth maps' size (default: {TrainConfig.resolution})")
    released.add_argument("--scale", type=parse_scale,
                          help=f"scene-scale rule: one of {SCALE_RULES}, or a factor (default: scene)")
    released.add_argument("--class-label", type=int,
                          help="MVGenMaster's domain label: 0 regular datasets (default), 1 MegaScenes")
    released.add_argument("--precision", choices=PRECISIONS, help=f"default: {LossConfig.precision}")
    released.add_argument("--swap-years", type=float, nargs="+",
                          help="one year per period for swapped dates (default: each period's median date)")

    sets = p.add_argument_group("view sets")
    sets.add_argument("--keys-file", action="append",
                      help='view sets as a JSON list of {"refs": [file_key, ...], "targets": [file_key, ...]}; '
                           f"repeatable; {NO_KEYS_FILE}: no keys file (default: {DEFAULT_KEYS_FILE})")
    sets.add_argument("--heldout", type=int, default=0,
                      help="with --run: also this many view sets with held-out targets, drawn as the run's "
                           "held-out date-loss probe draws them")
    sets.add_argument("--cross-only", action="store_true",
                      help="of the held-out sets, only those with every target after the edge and every reference "
                           "before it")
    sets.add_argument("--list-sets", action="store_true",
                      help="print the view sets and stop (no GPU, nothing written)")

    region = p.add_argument_group("changed region")
    region.add_argument("--region-edge", type=float,
                        help="the date at which the scene changed, in decimal years (default: the last period edge, "
                             "for Notre-Dame the fire)")
    region.add_argument("--min-mask-pixels", type=int, default=256,
                        help="a target whose region has fewer pixels is not scored inside")

    loss = p.add_argument_group("loss with fixed noise")
    loss.add_argument("--no-losses", action="store_true", help="skip it")
    loss.add_argument("--timesteps", type=int, nargs="+",
                      help=f"default: the run's probe timesteps, else {list(TrainConfig.probe_timesteps)}")
    loss.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1], help="default: 0 1")

    renders = p.add_argument_group("renders")
    renders.add_argument("--no-renders", action="store_true", help="skip them")
    renders.add_argument("--seeds", type=int, nargs="+",
                         help=f"noise seeds (default: the run's render seeds, else {list(TrainConfig.render_seeds)})")
    renders.add_argument("--steps", type=int,
                         help=f"sampler steps (default: the run's, else {TrainConfig.render_steps})")
    renders.add_argument("--guidance", type=float, help=f"default: the run's, else {TrainConfig.render_guidance}")
    renders.add_argument("--uncond-dates", choices=UNCOND_DATES,
                         help="what the unconditional half of the guidance batch sees (default: the run's, else "
                              f"{TrainConfig.render_uncond_dates})")
    renders.add_argument("--priors-by", choices=PRIORS_BY, default=PRIORS_BY[0],
                         help="for runs whose cross-date rule takes priors from targets: apply it by the dates "
                              "rendered at (default, as the run's render probe) or by the true dates in both renders "
                              "(then the two renders differ in the date alone)")
    renders.add_argument("--thumb", type=int, default=192, help="side of the sheets' pictures, in pixels")

    look1 = p.add_argument_group("experiment look1 (replaces the view sets, the loss and the renders above)")
    look1.add_argument("--eval-sets", type=Path,
                       help="with --run: the look1 evaluation-sets file (configs/look1_eval_sets.json): read-outs "
                            "R1-R6 and the date profile on its 2x2 of held-out sets")
    look1.add_argument("--readouts", nargs="+", choices=LOOK1_READOUTS, default=list(LOOK1_READOUTS),
                       help="which read-outs (default: all; those a model cannot have are skipped)")
    look1.add_argument("--primary-seeds", type=int, default=4,
                       help="seeds of the primary conditions: the base look at the true, swapped and within-period "
                            "dates (default: 4)")
    look1.add_argument("--other-seeds", type=int, default=2, help="seeds of every other condition (default: 2)")
    look1.add_argument("--cells", nargs="+",
                       help="only these cells, e.g. modern->after (quoted in a shell; default: all four)")
    look1.add_argument("--strict", action="store_true", help="the strict sets (all five targets held out) instead")
    look1.add_argument("--readout-probe", type=Path,
                       help=f"the state read-out (default: <dataset>/{READOUT_FILE}, scripts/mvgen/fit_readout.py)")
    look1.add_argument("--arm", help="this evaluation's arm, for the comparisons (A, A2, B, D or released)")
    look1.add_argument("--compare", action="append", default=[], metavar="ARM=RESULTS",
                       help="results.json of another arm at the same step: the cross-arm pass lines (repeatable)")
    look1.add_argument("--from-results", type=Path,
                       help="no model: print the read-outs of this look1 results.json and the --compare lines")
    args = p.parse_args(argv)

    if args.from_results is not None:
        return args
    if args.eval_sets is not None:
        if args.run is None:
            p.error("--eval-sets needs --run: its sets are checked against the run's hold-out")
        if args.keys_file or args.heldout or args.cross_only:
            p.error("--eval-sets brings its own view sets: no --keys-file, --heldout or --cross-only")
        if args.primary_seeds < 1 or args.other_seeds < 1:
            p.error("--primary-seeds and --other-seeds must be at least 1")
        if args.no_renders:
            p.error("look1's read-outs are made from renders: --no-renders does not apply (--no-losses does)")
        args.keys_file = []
        return args
    if args.strict or args.cells or args.compare or args.arm or args.readout_probe:
        p.error("--strict, --cells, --compare, --arm and --readout-probe go with --eval-sets")

    if args.run is None:
        if args.checkpoint is not None:
            p.error(f"--checkpoint needs --run (for the released model with a run's settings: --checkpoint {RELEASED})")
        if args.heldout:
            p.error("--heldout needs --run: held-out view sets are drawn from a run's hold-out")
    else:
        given = [f"--{name.replace('_', '-')}" for name in RELEASED_OPTIONS if getattr(args, name) is not None]
        if given:
            p.error(f"{', '.join(given)} cannot be given with --run: these settings are the run's")
    if args.cross_only and not args.heldout:
        p.error("--cross-only limits the --heldout sets: give --heldout N")
    if args.no_losses and args.no_renders and not args.list_sets:
        p.error("--no-losses and --no-renders leave nothing to measure")
    files = dict.fromkeys(args.keys_file or [str(DEFAULT_KEYS_FILE)])       # a file given twice is read once
    args.keys_file = [Path(file) for file in files if file != NO_KEYS_FILE]
    if not args.keys_file and not args.heldout:
        p.error("no view sets: give a --keys-file or --heldout N")
    return args


def first_given(*values):
    """The first value that is not None."""
    return next(value for value in values if value is not None)


def settings(args: argparse.Namespace) -> tuple[TrainConfig, tuple[float, ...] | None, Path | None]:
    """What the model is evaluated under: ``(config, swap years, checkpoint file)``.

    The config is the run's ``TrainConfig``, or without ``--run`` a ``TrainConfig`` with the given dataset,
    resolution, scale, domain label and precision and the defaults of everything else. The swap years
    are the run's (its ``probes.json``) or the given ones; None: each period's median date, from the
    data. The checkpoint file is None for the released model.
    """
    if args.run is not None:
        return run_config(args.run), run_swap_years(args.run), checkpoint_path(args.run, args.checkpoint or "latest")
    loss = {name: getattr(args, name) for name in ("class_label", "precision") if getattr(args, name) is not None}
    cfg = TrainConfig(dataset=str(args.dataset or DEFAULT_DATASET), run_dir=str(args.out),
                      resolution=first_given(args.resolution, TrainConfig.resolution),
                      scale=first_given(args.scale, TrainConfig.scale), loss=LossConfig(**loss),
                      swap_years=tuple(args.swap_years) if args.swap_years else None)
    return cfg, cfg.swap_years, None


def print_view_sets(described: Sequence[dict], edge: float, min_pixels: int) -> None:
    """One line per view set (its record of ``evaluation.describe_set``): name, group and source; the dates of
    its references and of its targets and the years the targets are swapped to; how many targets the run
    held out; and each target's region as a share of the frame. Then what a reader should be warned of."""
    n_cross = sum(record["group"] == "cross" for record in described)
    print(f"{len(described)} view sets: {n_cross} cross (every target dated from {edge:.10g} on, every reference "
          f"before it), {len(described) - n_cross} other", flush=True)
    for record in described:
        n_ref, years = record["n_ref"], [f"{year:.2f}" for year in record["years"]]
        swapped = " ".join(dict.fromkeys(f"{year:.2f}" for year in record["swapped_years"]))
        held = ""
        if record["held_out"] is not None:
            n_held = sum(record["held_out"][n_ref:])
            held = f" ({n_held} of {len(years) - n_ref} held out of training)"
        shares = " ".join(f"{share:.1%}" + ("" if scored else "*")
                          for share, scored in zip(record["region_share"], record["region_scored"]))
        print(f"   {record['name']:<16} {record['group']:<5} from {record['source']}: references "
              f"{' '.join(years[:n_ref])} | targets {' '.join(years[n_ref:])}{held}, swapped to {swapped} | "
              f"region {shares} of the frame", flush=True)
        for concern in ("not_usable", "references_without_reliable_depth"):
            if record[concern]:
                print(f"      note, {concern.replace('_', ' ')}: {record[concern]}", flush=True)
    scored = [is_scored for record in described for is_scored in record["region_scored"]]
    if not all(scored):
        print(f"   * a region of fewer than {min_pixels} pixels: the target is not scored inside "
              f"({len(scored) - sum(scored)} of {len(scored)} targets)", flush=True)


def by_timestep(summary: dict, timesteps: Sequence[int]) -> str:
    """One line: the swapped-date gap inside the region at each timestep, for the first group of the summary."""
    group = next(iter(summary))
    gaps = summary[group]["inside"].get("gap_swapped_by_timestep")
    if gaps is None:
        return ""
    return (f"inside the region, swapped date, by timestep ({group} sets): "
            + ", ".join(f"t={timestep} {gap:+.2%}" for timestep, gap in zip(timesteps, gaps)))


def collect_results(header: dict, view_sets: Sequence[ViewSet], described: Sequence[dict],
                    loss_records: dict[str, list[dict]], render_records: dict[str, list[dict]]) -> dict:
    """The content of ``results.json`` from what has been measured so far (records by view-set name)."""
    results = {**header, "sets_evaluated": {"loss": len(loss_records), "renders": len(render_records),
                                            "of": len(view_sets)}, "sets": []}
    summaries = {"loss": {name: summarise_set_losses(records) for name, records in loss_records.items()},
                 "renders": {name: summarise_set_renders(records) for name, records in render_records.items()}}
    for view_set, record in zip(view_sets, described):
        entry = {**record, **{kind: per_set[view_set.name] for kind, per_set in summaries.items()
                              if view_set.name in per_set}}
        if view_set.name in render_records:
            entry["render_records"] = render_records[view_set.name]
        results["sets"].append(entry)
    for kind, per_set in summaries.items():
        if per_set:
            results[kind] = summarise_groups([view_set for view_set in view_sets if view_set.name in per_set], per_set)
    return results


def load_model(checkpoint: Path | None, model_dir: str, weights: str, expect_step: int | None):
    """``(model, checkpoint dict)``: the released model (``{}``), or a checkpoint's adapters with its evaluated weights
    or, with ``weights="raw"``, the raw weights of a ``latest.pt``. ``SystemExit`` for a checkpoint of another step
    than ``expect_step`` or raw weights that the file does not hold."""
    if checkpoint is None:
        if weights == "raw" or expect_step is not None:
            raise SystemExit("--weights raw and --expect-step need a checkpoint, not the released model")
        return load_mvgen(model_dir), {}
    model, ckpt = load_finetuned(checkpoint, model_dir)
    if expect_step is not None and ckpt["step"] != expect_step:
        raise SystemExit(f"{checkpoint} is at step {ckpt['step']}, not {expect_step}")
    if weights == "raw":
        if "raw_trainable" not in ckpt:
            if ckpt.get("weights") == "raw":            # a run without an EMA: its weights are the raw ones
                return model, ckpt
            raise SystemExit(f"{checkpoint} holds no raw weights (only a latest.pt of a run with an EMA does)")
        load_trainable_state_dict(model, ckpt["raw_trainable"])
        ckpt = {**ckpt, "weights": "raw"}
    return model, ckpt


def look1_plan_lines(plan, conditions: dict[str, list], seeds: dict[bool, int], loss_passes: int) -> tuple[list[str], float]:
    """The sets, the renders they need and the estimated GPU time (``SECONDS_PER_RENDER``, ``SECONDS_PER_PASS``)."""
    lines, renders = [], 0
    for view_set in plan.sets:
        n = sum(seeds[c.primary] for c in conditions[view_set.name])
        renders += n
        lines.append(f"   {view_set.name:<14} {view_set.cell:<15} rows {list(view_set.rows)} scored "
                     f"{sum(view_set.scored)} of {len(view_set.scored)}, units {sorted(set(view_set.units))}: {n} renders")
    hours = (renders * SECONDS_PER_RENDER + loss_passes * SECONDS_PER_PASS) / 3600
    lines.append(f"{len(plan.sets)} sets, {renders} renders, {loss_passes} loss passes: about {hours:.2f} GPU-h "
                 f"({SECONDS_PER_RENDER} s per rendered set, {SECONDS_PER_PASS} s per pass)")
    return lines, hours


def print_look1(results: dict, comparisons: dict[str, dict], arm: str | None) -> None:
    """The look1 read-outs of a results.json with their pass lines, then the cross-arm lines."""
    print("\nLOOK1 READ-OUTS (mean +- standard error over hold-out units)", flush=True)
    for line in look1_lines(results["look1"], results["eval_sets"]["yardsticks"]):
        print(line, flush=True)
    profile = results.get("date_profile")
    if profile:
        print("\nWEIGHT-LEVEL DATE PROFILE (RMS distances of the date embedding)", flush=True)
        for step, numbers in profile.items():
            print(f"   step {step}: " + ", ".join(f"{name} {value:.4f}" for name, value in numbers.items()), flush=True)
    if comparisons:
        print("\nAGAINST OTHER ARMS (paired by target, same sets and seeds)", flush=True)
        for line in compare_look1(results, comparisons, arm):
            print(line, flush=True)


def read_comparisons(pairs: Sequence[str]) -> dict[str, dict]:
    """``ARM=results.json`` options as ``{arm: results}``."""
    out = {}
    for pair in pairs:
        arm, _, path = pair.partition("=")
        if not path:
            raise SystemExit(f"--compare takes ARM=RESULTS, got {pair!r}")
        out[arm] = json.loads(Path(path).read_text())
    return out


def run_date_profile(run_dir: Path, ckpt: dict, plan, swap_years: Sequence[float], edge: float) -> dict:
    """:func:`evaluation.date_profile` of every ``step_XXXXXX.pt`` of the run and of the evaluated weights."""
    profile = {}
    for path in sorted((run_dir / "checkpoints").glob("step_*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=False)
        config = state["finetune_config"]
        profile[str(state["step"])] = date_profile(state["trainable"]["date"], config["date_freqs"],
                                                   config["date_hidden"], edge, plan.within_years, swap_years)
    if ckpt:
        config = ckpt["finetune_config"]
        state = ckpt["raw_trainable" if ckpt.get("weights") == "raw" and "raw_trainable" in ckpt else "trainable"]
        profile[f"{ckpt['step']} ({ckpt.get('weights')}, evaluated)"] = date_profile(
            state["date"], config["date_freqs"], config["date_hidden"], edge, plan.within_years, swap_years)
    return profile


def main_look1(args: argparse.Namespace) -> None:
    """Experiment look1: the read-outs R1-R6 and the date profile on the sets of ``--eval-sets`` (module docstring)."""
    if args.from_results is not None:
        print_look1(json.loads(args.from_results.read_text()), read_comparisons(args.compare), args.arm)
        return
    cfg, swap_years, checkpoint = settings(args)
    n_ref, edges = cfg.sampling.n_ref, cfg.sampling.period_edges
    edge = edges[-1] if args.region_edge is None else args.region_edge
    timesteps = list(args.timesteps or cfg.probe_timesteps)
    steps, guidance = first_given(args.steps, cfg.render_steps), first_given(args.guidance, cfg.render_guidance)
    uncond_dates = first_given(args.uncond_dates, cfg.render_uncond_dates)
    scene = open_scene(cfg)
    check_run_records(scene, args.run)
    if swap_years is None:
        swap_years = representative_years(scene.dates[scene.usable], edges)
    plan = read_eval_sets(args.eval_sets, scene, args.strict, args.cells)
    check_eval_sets(plan, scene)
    has_date = checkpoint is not None
    has_look = has_date and bool(getattr(cfg.finetune, "look", False))
    readouts = list(args.readouts)
    skipped = sorted(set(readouts) - {"R6"} if not has_date else set(readouts) & ({"R1", "R4", "R5"} if not has_look
                                                                                   else set()))
    if skipped:
        print(f"note: skipped for this model ({'no date' if not has_date else 'no look'} input): {skipped}", flush=True)
    seeds = {True: args.primary_seeds, False: args.other_seeds}
    conditions = {s.name: look1_conditions(readouts, has_date, has_look, s.target_period) for s in plan.sets}
    with_losses = has_date and "R2" in readouts and not args.no_losses
    loss_passes = len(plan.sets) * 4 * len(timesteps) * len(args.noise_seeds) if with_losses else 0
    lines, hours = look1_plan_lines(plan, conditions, seeds, loss_passes)
    print(f"look1 sets of {args.eval_sets} (hold-out seed {plan.seed}{', strict sets' if args.strict else ''}):",
          flush=True)
    for line in lines:
        print(line, flush=True)
    if args.list_sets:
        return

    model_dir = args.model_dir or cfg.model_dir
    model, ckpt = load_model(checkpoint, model_dir, args.weights, args.expect_step)
    model.unet.eval()
    if model.config.camera_longest_side != scene.camera_longest_side:
        raise ValueError(f"the model's camera_longest_side is {model.config.camera_longest_side}, the scene was "
                         f"opened with {scene.camera_longest_side}: its scale factor is not the model's")
    if ckpt:
        check_run_records(scene, args.run, ckpt)
    inputs = None
    if has_look:
        from timemachine.mvgen.look import read_class_maps, read_look

        keys = scene.views["file_key"].tolist()
        look = read_look(scene.dataset_dir, keys)
        if look is None or getattr(model, "look_stats", None) is None:
            raise SystemExit("a model with a look needs the scene's look codes (scripts/mvgen/prepare_look.py) and the "
                             "checkpoint's look statistics")
        grey = any(c.look == "grey" for of_set in conditions.values() for c in of_set)
        inputs = LookInputs(look=look, stats=model.look_stats, class_maps=read_class_maps(scene.dataset_dir, keys)
                            if grey else None)
    readout = None
    if has_date and set(readouts) & {"R2", "R4"}:
        from timemachine.mvgen.readout import StateReadout

        readout = StateReadout.load(args.readout_probe or Path(cfg.dataset) / READOUT_FILE, device=model.device)
        readout.check_unseen([scene.views.file_key[row] for s in plan.sets
                              for row, scored in zip(s.targets, s.scored) if scored])
    weights = ckpt.get("weights", "not recorded") if ckpt else RELEASED
    probes = {s.name: RenderProbe(scene, s.rows, s.n_ref, s.name, RENDER_DATES, (0,), cfg.loss, swap_years, steps,
                                  guidance, uncond_dates) for s in plan.sets}
    masks = {s.name: stored_masks(scene, s.targets) for s in plan.sets}
    header = {
        "model": {"run": str(args.run), "arm": args.arm, "checkpoint": None if checkpoint is None else str(checkpoint),
                  "step": ckpt.get("step"), "weights": weights, "finetune": ckpt.get("finetune_config"),
                  "has_date": has_date, "has_look": has_look},
        "settings": {"dataset": cfg.dataset, "resolution": cfg.resolution, "scale": cfg.scale,
                     "scale_factor": scene.scale_factor, "n_ref": n_ref, "period_edges": list(edges),
                     "loss": asdict(cfg.loss), "swap_years": list(swap_years), "timesteps": timesteps,
                     "noise_seeds": args.noise_seeds, "primary_seeds": args.primary_seeds,
                     "other_seeds": args.other_seeds, "render_steps": steps, "render_guidance": guidance,
                     "render_uncond_dates": uncond_dates, "render_priors_by": args.priors_by, "readouts": readouts,
                     "readout_probe": None if readout is None else str(args.readout_probe or Path(cfg.dataset) / READOUT_FILE),
                     "estimated_gpu_hours": hours},
        "eval_sets": {"path": str(args.eval_sets), "seed": plan.seed, "strict": args.strict, "cells": args.cells,
                      "within_years": dict(plan.within_years), "yardsticks": plan.yardsticks},
        "sets": [{"name": s.name, "cell": s.cell, **describe_views(scene, s.rows, s.n_ref), "scored": list(s.scored),
                  "units": list(s.units), "donors": {k: list(v) for k, v in s.donors.items()},
                  "conditions": [c.name for c in conditions[s.name]]} for s in plan.sets]}
    print(f"model: {checkpoint or 'the released model'} (step {ckpt.get('step', 0)}, {weights} weights); look "
          f"{'yes' if has_look else 'no'}; {cfg.loss.precision}, domain label {cfg.loss.class_label}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    records, pairs, loss_records = [], [], []

    def write_results() -> dict:
        results = {**header, "records": records, "pairs": pairs,
                   "look1": summarise_look1(records, pairs, loss_records, plan, readouts, has_date, has_look)}
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
        return results

    if with_losses:
        with open(args.out / "losses.jsonl", "w") as f:
            for number, s in enumerate(plan.sets, start=1):
                started = time.time()
                measured = measure_look1_losses(model, probes[s.name], s, plan, masks[s.name], timesteps,
                                                args.noise_seeds, swap_years, inputs)
                loss_records += measured
                f.writelines(json.dumps(record) + "\n" for record in measured)
                f.flush()
                print(f"   loss {number} / {len(plan.sets)} {s.name} ({time.time() - started:.0f}s)", flush=True)
    pipeline = model.sampling_pipeline()
    pipeline.set_progress_bar_config(disable=True)
    for number, s in enumerate(plan.sets, start=1):
        started = time.time()
        new_records, new_pairs = render_look1_set(model, pipeline, probes[s.name], s, plan, conditions[s.name], seeds,
                                                  swap_years, masks[s.name], inputs, readout,
                                                  args.out / "renders" / s.folder, args.priors_by)
        records += new_records
        pairs += new_pairs
        write_results()
        print(f"   renders {number} / {len(plan.sets)} {s.name} ({time.time() - started:.0f}s)", flush=True)
    if "profile" in readouts and has_date:
        header["date_profile"] = run_date_profile(args.run, ckpt, plan, swap_years, edge)
    results = write_results()
    print_look1(results, read_comparisons(args.compare), args.arm)
    print(f"\nwrote results.json{', losses.jsonl' if with_losses else ''} and renders/ to {args.out}")


def main(argv: list[str] | None = None) -> None:
    assert_exact_matmul()
    args = parse_args(argv)
    if args.eval_sets is not None or args.from_results is not None:
        main_look1(args)
        return
    cfg, swap_years, checkpoint = settings(args)
    n_ref, edges = cfg.sampling.n_ref, cfg.sampling.period_edges
    if args.region_edge is None and not edges:
        raise SystemExit("the settings have no period edge: give --region-edge")
    edge = edges[-1] if args.region_edge is None else args.region_edge
    timesteps = list(args.timesteps or cfg.probe_timesteps)
    seeds = list(args.seeds or cfg.render_seeds)
    steps, guidance = first_given(args.steps, cfg.render_steps), first_given(args.guidance, cfg.render_guidance)
    uncond_dates = first_given(args.uncond_dates, cfg.render_uncond_dates)

    # Everything that needs no model comes first, so that a wrong option fails before the model is loaded.
    scene = open_scene(cfg)
    if args.run is not None:
        check_run_records(scene, args.run)
    if swap_years is None:
        swap_years = representative_years(scene.dates[scene.usable], edges)
    view_sets = collect_view_sets(scene, args.keys_file, args.heldout, args.cross_only, edge)
    n_heldout = sum(view_set.source == HELDOUT_SOURCE for view_set in view_sets)
    if n_heldout < args.heldout:
        print(f"note: {n_heldout} of the {args.heldout} held-out view sets asked for were found"
              + (" (the run holds out no views)" if not scene.heldout.any() else ""), flush=True)
    if not view_sets:
        raise SystemExit("no view sets to evaluate")
    regions = ChangedRegions(scene, edge)
    print(f"changed points at {edge:.10g}: " + ", ".join(
        f"{counts['kept']} {kind}" for kind, counts in regions.points.counts.items()), flush=True)
    probes = [RenderProbe(scene, view_set.rows, n_ref, view_set.name, RENDER_DATES, seeds, cfg.loss, swap_years,
                          steps, guidance, uncond_dates) for view_set in view_sets]
    masks = [target_masks(regions, view_set.rows[n_ref:], cfg.resolution, args.min_mask_pixels)
             for view_set in view_sets]
    described = [describe_set(scene, *of_set) for of_set in zip(view_sets, probes, masks)]
    print_view_sets(described, edge, args.min_mask_pixels)
    other_priors = [probe.name for probe in probes if priors_differ(probe)]
    if other_priors and args.priors_by == "rendered" and not args.no_renders:
        print(f"note: the run's cross-date rule ({cfg.loss.cross_date_pixels}) gives the targets of "
              f"{len(other_priors)} of the {len(probes)} view sets other priors at their swapped dates than at "
              "their true dates, so the two renders of those sets differ in more than the date, for the released "
              f"model too ({' '.join(other_priors)}). --priors-by true gives both renders the true dates' priors.",
              flush=True)
    if args.list_sets:
        return

    model_dir = args.model_dir or cfg.model_dir
    if args.weights == "ema" and args.expect_step is None:
        model, ckpt = load_finetuned(checkpoint, model_dir) if checkpoint is not None else (load_mvgen(model_dir), {})
    else:
        model, ckpt = load_model(checkpoint, model_dir, args.weights, args.expect_step)
    model.unet.eval()
    if model.config.camera_longest_side != scene.camera_longest_side:
        raise ValueError(f"the model's camera_longest_side is {model.config.camera_longest_side}, the scene was "
                         f"opened with {scene.camera_longest_side}: its scale factor is not the model's")
    if args.run is not None and ckpt:
        check_run_records(scene, args.run, ckpt)
    weights = ckpt.get("weights", "not recorded") if ckpt else RELEASED
    header = {
        "model": {"run": None if args.run is None else str(args.run),
                  "checkpoint": None if checkpoint is None else str(checkpoint), "step": ckpt.get("step"),
                  "weights": weights, "finetune": ckpt.get("finetune_config")},
        "settings": {"dataset": cfg.dataset, "resolution": cfg.resolution, "scale": cfg.scale,
                     "scale_factor": scene.scale_factor, "n_ref": n_ref, "period_edges": list(edges),
                     "loss": asdict(cfg.loss), "swap_years": list(swap_years), "timesteps": timesteps,
                     "noise_seeds": args.noise_seeds, "render_seeds": seeds, "render_steps": steps,
                     "render_guidance": guidance, "render_uncond_dates": uncond_dates,
                     "render_priors_by": args.priors_by, "sets_with_other_priors_when_swapped": other_priors},
        "region": {"edge": edge, "min_mask_pixels": args.min_mask_pixels, "points": regions.points.counts,
                   "targets": sum(len(of_targets.counted) for of_targets in masks),
                   "targets_not_scored": sum(int((~of_targets.counted).sum()) for of_targets in masks)}}
    what = "the released model (no date input)" if checkpoint is None else (
        f"{checkpoint} (step {ckpt['step']}, {'the EMA of the' if weights == 'ema' else weights} weights)")
    print(f"model: {what}; {cfg.loss.precision}, domain label {cfg.loss.class_label}, scene scale {cfg.scale!r} "
          f"(factor {scene.scale_factor}); swapped dates use the years {[round(year, 2) for year in swap_years]}",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    loss_records: dict[str, list[dict]] = {}
    render_records: dict[str, list[dict]] = {}

    def write_results() -> dict:
        results = collect_results(header, view_sets, described, loss_records, render_records)
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
        return results

    if not args.no_losses:
        with open(args.out / "losses.jsonl", "w") as f:
            for number, (probe, of_targets) in enumerate(zip(probes, masks), start=1):
                started = time.time()
                loss_records[probe.name] = measure_losses(model, probe, of_targets, timesteps, args.noise_seeds)
                f.writelines(json.dumps(record) + "\n" for record in loss_records[probe.name])
                f.flush()
                print(f"   loss {number} / {len(probes)} {probe.name} ({time.time() - started:.0f}s)", flush=True)
        ratios = [record["date_temb_ratio"] for records in loss_records.values() for record in records
                  if record.get("date_temb_ratio") is not None]
        header["model"]["date_temb_ratio"] = float(np.mean(ratios)) if ratios else None
        table = write_results()["loss"]
        size = "" if not ratios else f"; date embedding / timestep embedding {header['model']['date_temb_ratio']:.2f}"
        print("\n" + format_table(
            "LOSS with fixed noise: gap to the loss at the true date (above zero: the true date fits better)\n"
            f"mean +- standard error over view sets; timesteps {timesteps}, noise seeds {args.noise_seeds}{size}",
            table, LOSS_COLUMNS), flush=True)
        print(by_timestep(table, timesteps) + "\n", flush=True)

    if not args.no_renders:
        pipeline = model.sampling_pipeline()
        pipeline.set_progress_bar_config(disable=True)
        for number, (view_set, probe, of_targets) in enumerate(zip(view_sets, probes, masks), start=1):
            started = time.time()
            render_records[probe.name] = render_set(model, pipeline, probe, of_targets,
                                                    args.out / "renders" / view_set.folder,
                                                    args.out / "sheets" / f"{view_set.folder}.jpg", args.thumb,
                                                    args.priors_by)
            table = write_results()["renders"]
            print(f"   renders {number} / {len(probes)} {probe.name} ({time.time() - started:.0f}s)", flush=True)
        print("\n" + format_table(
            "RENDERS: PSNR against the real photos (dB), and the difference between the true-date and the "
            "swapped-date render (mean absolute, 0-1)\n"
            f"mean +- standard error over view sets; seeds {seeds}, {steps} sampler steps, guidance {guidance:g}, "
            f"unconditional dates {uncond_dates}"
            + (f", cross-date priors by the {args.priors_by} dates" if other_priors else ""),
            table, RENDER_COLUMNS), flush=True)
    print(f"\nwrote results.json{'' if args.no_losses else ', losses.jsonl'}"
          f"{'' if args.no_renders else ', renders/ and sheets/'} to {args.out}")


if __name__ == "__main__":
    main()
