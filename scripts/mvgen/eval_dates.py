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
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from timemachine.mvgen.data import SCALE_RULES
from timemachine.mvgen.date import UNCOND_DATES
from timemachine.mvgen.evaluation import (
    HELDOUT_SOURCE, PRIORS_BY, RELEASED, RENDER_DATES, ViewSet, check_run_records, checkpoint_path,
    collect_view_sets, describe_set, format_table, measure_losses, open_scene, priors_differ, render_set,
    run_config, run_swap_years, summarise_groups, summarise_set_losses, summarise_set_renders, target_masks,
)
from timemachine.mvgen.model import assert_exact_matmul, load_finetuned, load_mvgen
from timemachine.mvgen.probes import RenderProbe, representative_years
from timemachine.mvgen.regions import ChangedRegions
from timemachine.mvgen.step import PRECISIONS, LossConfig
from timemachine.mvgen.trainer import TrainConfig, parse_scale

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / "data/datasets/25512_model0_v2"
DEFAULT_KEYS_FILE = REPO / "configs/mvgen_gate2.json"
NO_KEYS_FILE = "none"                               # the --keys-file value that switches the default file off
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
    args = p.parse_args(argv)

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


def main(argv: list[str] | None = None) -> None:
    assert_exact_matmul()
    args = parse_args(argv)
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
    model, ckpt = load_finetuned(checkpoint, model_dir) if checkpoint is not None else (load_mvgen(model_dir), {})
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
