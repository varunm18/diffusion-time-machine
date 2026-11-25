#!/usr/bin/env python
"""The evaluation sets of experiment look1: a 2x2 of reference period x target period, on views held out by day and photographer.

Experiment look1 (an untrained per-view look code on Notre-Dame) asks whether the date learns the
place's state rather than the photo's look. Its read-outs (``timemachine.mvgen.evaluation``,
``scripts/mvgen/eval_dates.py --eval-sets``) are measured on held-out views, in four *cells*: the
references from the modern period (1970 to the fire) or from after the fire, the targets from either.
This script fixes those views before any arm trains, and writes them to ``configs/look1_eval_sets.json``.

**Hold-out** (``timemachine.data.holdout``, ``HoldoutConfig(by="component")``: the functions training
uses, so the file's hold-out is the runs'). Photos of one day, or of one photographer, share light,
weather and camera; a held-out photo whose day or photographer trains is not unseen. So the unit that
is held out is a connected component of a period's usable views, linked by the same calendar day or the
same Commons author (``holdout_units``; the author is the ``artist`` column of
``data/tables/image_metadata/<scene id>.parquet``, white space collapsed, 60 characters). Per period the
units are shuffled with the seed, units of more than 12 views skipped, and units added until 10% of the
period's usable views and 10% of its side views (stored mask of at least 5% of the frame) are held out;
the old period gives 10% of its units.

**Seed** (``first_holdout_seed``): the first seed (from 0) that holds out at most 15% of every period and
gives every cell at least 4 disjoint sets. The look1 critique found seed 4 with its set rule, whose
references only had to see 3 targets each (``data/look1_critique/feasibility/01e*``); with the spec's
references, which share 30 points with every scored target, seed 4 leaves 3 sets in one cell and the
first seed that passes is 7. The script recomputes the seed, and checks the critique's numbers for its
seed with the critique's set rule (:func:`critique_cells`): that the units and the hold-out are its.

**A set** (``evaluation_sets``, ``evaluation_cells``): 3 references and 5 targets. The anchor is a
held-out side view of the target period; at least 2 more targets are held out; the others are training
views of the period (rendered, not scored), all within the anchor's tolerance (modern: 5 years; after
the fire: 1 year) and co-visible with it. The references are training views of the reference period
with reliable depth that share 30 points with every scored target and together see every target. Per
cell the sets are packed so that no scored target is in two of them. *Strict* sets, whose 5 targets are
all held out, are packed separately per cell. Every scored target's hold-out unit is recorded: standard
errors are taken over units, not sets.

**Donors** (:func:`choose_donors`), for the look read-outs R1 (steering) and R4 (does the look carry the
state): every target of a set gets two donor views whose look code it is rendered with, one of the
other period and one of its own, both from units that none of the set's views belongs to, held out or
not, and matched to the target on tone, season and sky class (the appearance study's labels,
``data/appearance_study/confounds/features.parquet``: ``tone_checked``, ``season``, ``sky_class``; where
no view matches on all three, the sky class and then the season are let go, and each donor records
what it matches). No view is the donor of two targets of one set.

**Yardsticks from real photos** (:func:`look_gap`, :func:`chroma_yardsticks`), measured on the training
views of this hold-out, outside each view's stored changed region: the after-fire-minus-modern gap of
six look measures of the side views in within-period standard deviations (the direction R3 projects a
date swap on), and the mean CIELAB chroma of grey, toned and colour photos (R1's grey flag and R5).
They need the images: about 600 views are loaded at 512 px, a minute or two on 4 CPU threads.

Runs in MVGenMaster's environment (it uses ``timemachine.mvgen.evaluation``), on a CPU:

    cd external/MVGenMaster && PYTHONPATH=$PWD OMP_NUM_THREADS=4 nice .venv/bin/python $REPO/scripts/mvgen/make_eval_sets.py

It prints, per cell, the sets, the scored targets and the independent units behind them, and refuses to
overwrite an existing file without ``--overwrite``. The file's ``holdout.config`` is the ``HoldoutConfig``
the arms train with.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import torch

from timemachine.data.dataset import load_view
from timemachine.data.holdout import EVAL_TOLERANCES, HoldoutConfig, disjoint_sets, evaluation_cells, first_holdout_seed
from timemachine.mvgen.data import MVGenScene, ViewSamplingConfig
from timemachine.mvgen.evaluation import LOOK_MEASURES, clustered_mean, look_measures, mean_chroma, stored_masks

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / "data/datasets/25512_model0_v2"
DEFAULT_OUT = REPO / "configs/look1_eval_sets.json"
STUDY_LABELS = REPO / "data/appearance_study/confounds/features.parquet"
PERIODS = ("old", "modern", "after")         # by ViewSamplingConfig.period_edges: 1970 and the fire
CELL_PERIODS = (1, 2)                        # the 2x2 has modern and after-fire views only
HOLDOUT_SHARE = 0.10                         # of a period's usable views and of its side views
SEEDS_TRIED = 50
CRITIQUE_SEED = 4                            # the seed the critique's log reports, with its own set rule
# The labels donors are matched on, most important first: where no view matches on all, the last ones are dropped.
DONOR_LABELS = ("tone_checked", "season", "sky_class")
DONOR_SEED = 2026
GAP_MIN_SHARE = 0.02                         # side views of the look gap: stored mask of at least 2% of the frame
# The two years of each period between which the within-period swap moves a target (to the farther one): the same
# state, about as far apart as the swap across the fire. Modern: the spec's 2013.4 <-> 2016.5 (the gate-3 date
# embedding moves 0.165 between them, 0.176 across the fire); after the fire: June 2019 <-> July 2021, a pair the
# appearance study measured (feasibility.md 3.2), both in summer.
WITHIN_YEARS = {"modern": [2013.4, 2016.5], "after": [2019.45, 2021.5]}


# ----------------------------------------------------------------------------
# Hold-out and sets
# ----------------------------------------------------------------------------
def holdout_config(scene: MVGenScene, seed: int) -> HoldoutConfig:
    """The hold-out of look1 under ``seed``: whole components, by the scene's periods."""
    return HoldoutConfig(fraction=HOLDOUT_SHARE, by="component", seed=int(seed), era_edges=scene.config.period_edges)


def heldout_views(scene: MVGenScene, seed: int) -> np.ndarray:
    """bool ``[N]``: the views the hold-out of ``seed`` takes (``MVGenScene.holdout_rows``, as the trainer)."""
    held = np.zeros(len(scene.views), dtype=bool)
    held[scene.holdout_rows(holdout_config(scene, seed))] = True
    return held


def cell_name(reference_period: int, target_period: int) -> str:
    return f"{PERIODS[reference_period]}->{PERIODS[target_period]}"


def cells_of(scene: MVGenScene, held: np.ndarray, side: np.ndarray, seed: int, min_scored: int | None = None
             ) -> dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    """The disjoint sets of the four cells (``holdout.evaluation_cells``), by cell name; ``min_scored`` 5: strict."""
    cfg = scene.config
    options = {} if min_scored is None else {"min_scored": min_scored}
    cells = evaluation_cells(seed, CELL_PERIODS, heldout=held, usable=scene.usable, ref_ok=scene.depth_ok,
                             dates=scene.dates, period=scene.period, covis=scene.covis, side=side, n_ref=cfg.n_ref,
                             n_targets=cfg.n_targets, min_shared=cfg.min_shared, top_k=cfg.top_k, **options)
    return {cell_name(*key): [(tuple(int(r) for r in refs), tuple(int(t) for t in targets)) for refs, targets in sets]
            for key, sets in cells.items()}


def critique_cells(scene: MVGenScene, held: np.ndarray, side: np.ndarray, seed: int) -> dict[str, list]:
    """The critique's disjoint sets (``01b_holdout_alternatives.sets_mixed``: each reference sees at least 3 targets
    and together they see all), only to check that this hold-out is the critique's: its log's counts come back."""
    cfg, dates, covis, usable, period = scene.config, scene.dates, scene.covis, scene.usable, scene.period
    rng = np.random.default_rng(1000 + seed)
    cells = {}
    for target_period in CELL_PERIODS:
        for reference_period in CELL_PERIODS:
            ref_ok = usable & ~held & scene.depth_ok & (period == reference_period)
            drawn = []
            for anchor in np.flatnonzero(usable & held & (period == target_period) & side):
                partners = ((np.abs(dates - dates[anchor]) <= EVAL_TOLERANCES[target_period]) & usable
                            & (period == target_period) & (covis[anchor] >= cfg.min_shared))
                partners[anchor] = False
                partners = np.flatnonzero(partners)
                held_partners = partners[held[partners]]
                if len(partners) < cfg.n_targets - 1 or len(held_partners) < 2:
                    continue
                top = partners[np.argsort(-covis[anchor, partners].astype(np.int64), kind="stable")[:cfg.top_k]]
                held_top = np.intersect1d(top, held_partners)
                for _ in range(40):
                    first = rng.choice(held_top if len(held_top) >= 2 else held_partners, 2, replace=False)
                    rest_pool = np.setdiff1d(top, first)
                    if len(rest_pool) < cfg.n_targets - 3:
                        break
                    targets = np.concatenate([[anchor], first, rng.choice(rest_pool, cfg.n_targets - 3, replace=False)])
                    sees = covis[:, targets] >= cfg.min_shared
                    candidates = ref_ok & (sees.sum(1) >= cfg.min_targets_seen)
                    candidates[targets] = False
                    pool = np.flatnonzero(candidates)
                    if len(pool) < cfg.n_ref or not sees[pool].any(0).all():
                        continue
                    weights = sees[pool].sum(1) / sees[pool].sum()
                    for _ in range(20):
                        refs = rng.choice(pool, cfg.n_ref, replace=False, p=weights)
                        if sees[refs].any(0).all():
                            drawn.append((refs, targets))
                            break
            cells[cell_name(reference_period, target_period)] = [drawn[i] for i in disjoint_sets(drawn, held)]
    return cells


def period_shares(held: np.ndarray, usable: np.ndarray, period: np.ndarray, side: np.ndarray) -> dict:
    """Per period: usable views, held-out views and their share, and held-out side views."""
    shares = {}
    for p, name in enumerate(PERIODS):
        members = usable & (period == p)
        shares[name] = {"usable": int(members.sum()), "held_out": int((held & members).sum()),
                        "share": float((held & members).sum() / max(1, members.sum())),
                        "side_views_held_out": int((held & members & side).sum())}
    return shares


# ----------------------------------------------------------------------------
# Donors
# ----------------------------------------------------------------------------
def choose_donors(targets: Sequence[int], set_views: Sequence[int], target_period: int, units: np.ndarray,
                  usable: np.ndarray, period: np.ndarray, labels: pd.DataFrame, rng: np.random.Generator
                  ) -> dict[str, list]:
    """Two donors per target (see the module docstring): ``other`` and ``same`` rows, and per donor the labels it
    matches the target on (``*_match``)."""
    busy_units = set(units[list(set_views)].tolist())
    free = usable & ~np.isin(units, list(busy_units))
    other_period = 2 if target_period == 1 else 1
    taken: set[int] = set()
    donors: dict[str, list] = {"other": [], "same": [], "other_match": [], "same_match": []}
    for target in targets:
        for kind, donor_period in (("other", other_period), ("same", target_period)):
            pool = free & (period == donor_period)
            pool[list(taken)] = False
            chosen, matched = None, []
            for n_labels in range(len(DONOR_LABELS), -1, -1):
                keys = list(DONOR_LABELS[:n_labels])
                match = pool.copy()
                for key in keys:
                    match &= labels[key].to_numpy() == labels[key].iloc[target]
                if match.any():
                    chosen, matched = int(rng.choice(np.flatnonzero(match))), keys
                    break
            if chosen is None:
                raise SystemExit(f"no {kind} donor for target row {target}")
            taken.add(chosen)
            donors[kind].append(chosen)
            donors[f"{kind}_match"].append(matched)
    return donors


# ----------------------------------------------------------------------------
# Yardsticks from real photos
# ----------------------------------------------------------------------------
def photo(scene: MVGenScene, row: int) -> np.ndarray:
    """uint8 ``[S, S, 3]``: the view's crop as ``MVGenScene.make_sample`` gives it (rounded to 8 bits)."""
    image, _ = load_view(scene.views.iloc[row], scene.resolution)
    return ((image.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)


def look_gap(scene: MVGenScene, training: np.ndarray, share: np.ndarray) -> dict:
    """The after-fire-minus-modern gap of the six look measures (``evaluation.LOOK_MEASURES``) of the training side
    views (stored mask of at least 2% of the frame), outside their regions, in the mean of the two periods'
    standard deviations; and its unit direction, on which R3 projects a date swap."""
    side = training & (share >= GAP_MIN_SHARE)
    rows = {p: np.flatnonzero(side & (scene.period == p)) for p in CELL_PERIODS}
    measures = {p: np.stack([look_measures(photo(scene, row), ~stored_masks(scene, [row])[0]) for row in rows[p]])
                for p in CELL_PERIODS}
    sd = np.mean([np.nanstd(measures[p], axis=0, ddof=1) for p in CELL_PERIODS], axis=0)
    gap = (np.nanmean(measures[2], axis=0) - np.nanmean(measures[1], axis=0)) / sd
    return {"measures": list(LOOK_MEASURES), "sd": sd.tolist(), "gap": gap.tolist(),
            "unit": (gap / np.linalg.norm(gap)).tolist(), "length": float(np.linalg.norm(gap)),
            "views": {PERIODS[p]: int(len(rows[p])) for p in CELL_PERIODS},
            "rule": f"training views of the hold-out, modern and after the fire, stored mask >= {GAP_MIN_SHARE:.0%} "
                    "of the frame; measures outside the stored mask"}


def chroma_yardsticks(scene: MVGenScene, training: np.ndarray, tones: np.ndarray, units: np.ndarray,
                      colour_rows: np.ndarray) -> dict:
    """Mean CIELAB chroma outside the stored region of real photos by tone (``bw`` grey, ``toned``, ``colour``):
    training views, the colour ones among the modern and after-fire training views given; the standard error is
    clustered by hold-out unit."""
    out = {}
    for name, label, rows in (("grey", "bw", None), ("toned", "toned", None), ("colour", "colour", colour_rows)):
        rows = np.flatnonzero(training & (tones == label)) if rows is None else rows[tones[rows] == label]
        values = {str(row): mean_chroma(photo(scene, row), ~stored_masks(scene, [row])[0]) for row in rows}
        stats = clustered_mean(values, {str(row): int(units[row]) for row in rows})
        out[name] = {"mean": stats["mean"], "se": stats["se"], "photos": stats["targets"], "units": stats["units"]}
    return out


# ----------------------------------------------------------------------------
# The file
# ----------------------------------------------------------------------------
def set_entries(scene: MVGenScene, sets: Sequence[tuple[tuple[int, ...], tuple[int, ...]]], held: np.ndarray,
                units: np.ndarray, labels: pd.DataFrame, cell_index: int, cell: str, strict: bool) -> list[dict]:
    """The sets of a cell as the file stores them, with their donors."""
    keys = scene.views["file_key"].tolist()
    target_period = PERIODS.index(cell.split("->")[1])
    short = "-".join(name[:3] for name in cell.split("->"))
    entries = []
    for i, (refs, targets) in enumerate(sets):
        rng = np.random.default_rng([DONOR_SEED, cell_index, i, int(strict)])
        donors = choose_donors(targets, (*refs, *targets), target_period, units, scene.usable, scene.period, labels,
                               rng)
        rows = [*refs, *targets]
        entries.append({"name": f"{short}/{'strict' if strict else ''}{i}", "n_ref": len(refs), "rows": rows,
                        "file_keys": [keys[row] for row in rows], "anchor": int(targets[0]),
                        "scored": [bool(held[t]) for t in targets], "units": [int(units[t]) for t in targets],
                        "strict": bool(held[list(targets)].all()),
                        "donors": {**donors, "other_file_keys": [keys[row] for row in donors["other"]],
                                   "same_file_keys": [keys[row] for row in donors["same"]]}})
    return entries


def cell_counts(entries: Sequence[dict]) -> dict:
    """Sets, scored targets and the independent hold-out units behind them."""
    scored = [(unit, row) for e in entries for row, unit, s in zip(e["rows"][e["n_ref"]:], e["units"], e["scored"]) if s]
    return {"sets": len(entries), "scored_targets": len(scored), "units": len({unit for unit, _ in scored})}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="dataset folder (default: %(default)s)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the evaluation-sets file (default: %(default)s)")
    p.add_argument("--labels", type=Path, default=STUDY_LABELS,
                   help="the appearance study's per-view labels the donors are matched on (default: %(default)s)")
    p.add_argument("--resolution", type=int, default=512, help="side of the crops, as the runs' (default: 512)")
    p.add_argument("--seed", type=int, help="the hold-out seed (default: the first that passes, see the docstring)")
    p.add_argument("--seeds-tried", type=int, default=SEEDS_TRIED, help="seeds tried from 0 (default: %(default)s)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists: give --overwrite to replace it")
    torch.set_num_threads(4)
    scene = MVGenScene(args.dataset, ViewSamplingConfig(), resolution=args.resolution)
    views = scene.views
    if scene.regions is None:
        raise SystemExit(f"{args.dataset} has no changed-region file: run scripts/mvgen/prepare_regions.py")
    side = scene.side_views(holdout_config(scene, 0).side_share)
    units = scene.holdout_units(holdout_config(scene, 0).unit_periods)
    labels = pd.read_parquet(args.labels)
    if labels["file_key"].tolist() != views["file_key"].tolist():
        raise SystemExit(f"{args.labels} is not row for row the view table of {args.dataset}")

    critique = heldout_views(scene, CRITIQUE_SEED)
    shares = period_shares(critique, scene.usable, scene.period, side)
    check = {"seed": CRITIQUE_SEED, "held_out": {name: entry["held_out"] for name, entry in shares.items()},
             "side_after": shares["after"]["side_views_held_out"],
             "cells": {cell: {"sets": len(sets), "units": len({int(units[t]) for _, ts in sets for t in ts if critique[t]})}
                       for cell, sets in critique_cells(scene, critique, side, CRITIQUE_SEED).items()}}
    print(f"the critique's rule, seed {CRITIQUE_SEED}, its sampler reference rule: held out modern "
          f"{check['held_out']['modern']}, after {check['held_out']['after']} (side {check['side_after']}); sets "
          + ", ".join(f"{cell} {c['sets']} ({c['units']} units)" for cell, c in check["cells"].items())
          + "  [its log, 01e: 65, 54 (28); 7 (18), 4 (10), 5 (6), 6 (8)]", flush=True)

    seed = args.seed
    if seed is None:
        seed, records = first_holdout_seed(holdout_config(scene, 0), dates=scene.dates, units=units,
                                           usable=scene.usable, side=side, ref_ok=scene.depth_ok, covis=scene.covis,
                                           seeds=range(args.seeds_tried))
        if seed is None:
            raise SystemExit(f"no seed below {args.seeds_tried} gives every cell 4 disjoint sets: {records[-3:]}")
    held = heldout_views(scene, seed)
    cells = cells_of(scene, held, side, seed)
    strict_cells = cells_of(scene, held, side, seed, min_scored=scene.config.n_targets)
    shares = period_shares(held, scene.usable, scene.period, side)
    print(f"hold-out seed {seed}: " + "; ".join(f"{name} {e['held_out']} of {e['usable']} ({e['share']:.1%}), side "
                                                f"{e['side_views_held_out']}" for name, e in shares.items()), flush=True)

    out_cells = {}
    for index, (cell, sets) in enumerate(cells.items()):
        reference, target = cell.split("->")
        entries = set_entries(scene, sets, held, units, labels, index, cell, strict=False)
        strict = set_entries(scene, strict_cells[cell], held, units, labels, index, cell, strict=True)
        out_cells[cell] = {"reference_period": reference, "target_period": target, "sets": entries,
                           "strict_sets": strict, "counts": cell_counts(entries), "strict_counts": cell_counts(strict)}
        counts = out_cells[cell]["counts"]
        print(f"   {cell:15s} sets {counts['sets']}, scored targets {counts['scored_targets']}, independent units "
              f"{counts['units']}; strict sets {len(strict)} ({out_cells[cell]['strict_counts']['units']} units)",
              flush=True)

    training = scene.usable & ~held
    share = scene.regions.astype(np.float64).mean(axis=(1, 2))
    print("measuring the yardsticks on real training photos (a minute or two) ...", flush=True)
    gap = look_gap(scene, training, share)
    colour_rows = np.flatnonzero(training & np.isin(scene.period, CELL_PERIODS) & (share >= GAP_MIN_SHARE))
    chroma = chroma_yardsticks(scene, training, labels["tone_checked"].to_numpy(), units, colour_rows)
    print(f"   look gap (after minus modern, sd): {dict(zip(LOOK_MEASURES, np.round(gap['gap'], 3)))}, length "
          f"{gap['length']:.3f}", flush=True)
    print("   chroma outside the region: " + ", ".join(
        f"{name} " + ("n/a" if c["mean"] is None else f"{c['mean']:.2f} +- {c['se'] or 0:.2f}") + f" ({c['photos']})"
        for name, c in chroma.items()), flush=True)

    keys = views["file_key"].tolist()
    held_rows = np.flatnonzero(held)
    config = asdict(holdout_config(scene, seed))
    content = {
        "description": "experiment look1: evaluation sets, a 2x2 of reference period x target period on views held out "
                       "by connected components of day and photographer (scripts/mvgen/make_eval_sets.py)",
        "made_on": date.today().isoformat(), "dataset": str(args.dataset), "n_views": len(views),
        "period_edges": list(scene.config.period_edges), "periods": list(PERIODS),
        "holdout": {"rule": "component", "seed": seed, "config": config, "rows": held_rows.tolist(),
                    "file_keys": [keys[row] for row in held_rows], "units": [int(units[row]) for row in held_rows],
                    "unit_source": "timemachine.data.holdout.holdout_units: author from the artist column of "
                                   "data/tables/image_metadata/<scene id>.parquet; day: date_hi - date_lo <= 0.0031",
                    "shares": shares, "critique_check": check},
        "settings": {"sampling": asdict(scene.config), "min_scored": 3, "target_tolerance":
                     {PERIODS[p]: tol for p, tol in EVAL_TOLERANCES.items()},
                     "reference_rule": "training views of the reference period with reliable depth, each sharing "
                                       f"{scene.config.min_shared} points with every scored target, together seeing "
                                       "every target", "donor_labels": list(DONOR_LABELS),
                     "donor_label_source": str(args.labels)},
        "within_years": WITHIN_YEARS,
        "cells": out_cells,
        "yardsticks": {"look_gap": gap, "chroma": chroma},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(content, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
