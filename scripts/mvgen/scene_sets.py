#!/usr/bin/env python
"""Draw view sets of any scene folder for date sweeps: targets of one period, references of another.

For the multi-scene run's demos and checks on scenes other than Notre-Dame (held-out scenes above all,
whose views no run has trained on). Per target period, ``--per-period`` sets are drawn with the training
sampler (``MVGenScene.draw``: an anchor and its most co-visible views of the same date as targets) with
every anchor in that period and ALL references from another period (``cross_period_prob = 1``), so the
date has to say what the targets show. Every usable view may be drawn (the scene's hold-out is not
applied: for a held-out scene all views are unseen; for a training scene, pass ``--heldout-only`` to
draw only among the 10% of views per period that the multi-scene run held out).

Writes a rows file for ``scripts/mvgen/render.py --rows-file`` (references first) and, next to it, a
``.txt`` that lists each set's period and the targets' and references' dates.

    python scripts/mvgen/scene_sets.py --dataset <folder> --out <sets.json> --per-period 3
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from timemachine.data.holdout import HoldoutConfig
from timemachine.mvgen.data import MVGenScene, ViewSamplingConfig
from timemachine.mvgen.multiscene import scene_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-period", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout-only", action="store_true",
                        help="only the views the multi-scene run holds out of this (training) scene")
    args = parser.parse_args()

    edges = scene_edges(args.dataset)
    config = ViewSamplingConfig(period_edges=edges, cross_period_prob=1.0)
    scene = MVGenScene(args.dataset, config, seed=args.seed, require_depth=True, scale="scene")
    if args.heldout_only:          # the multi-scene run's rule for scenes without regions (SceneMixture)
        heldout = np.zeros(len(scene.views), dtype=bool)
        heldout[np.asarray(scene.holdout_rows(HoldoutConfig(fraction=0.1, seed=7)), dtype=int)] = True
        scene.hold_out(np.flatnonzero(~heldout))     # train on nothing: draw among the held-out views
    n_periods = len(edges) + 1
    sets, notes = [], []
    for period in range(n_periods):
        weights = tuple(float(p == period) for p in range(n_periods))
        anchors = scene.period[scene.anchors]
        if not (anchors == period).any():
            notes.append(f"period {period}: no anchor, no set")
            continue
        scene.config = replace(config, anchor_period_weights=weights)
        drawn, attempts = 0, 0
        while drawn < args.per_period and attempts < 50 * args.per_period:
            rows, n_ref = scene.draw(attempts, epoch=0, seed=args.seed)
            attempts += 1
            refs, targets = rows[:n_ref], rows[n_ref:]
            if (scene.period[refs] == period).any() or any(rows.tolist() == s for s in sets):
                continue                    # a same-period fallback: not a cross-period set
            sets.append(rows.tolist())
            drawn += 1
            notes.append(f"set {len(sets) - 1}: targets period {period} dates "
                         f"{np.round(scene.dates[targets], 1).tolist()}; references dates "
                         f"{np.round(scene.dates[refs], 1).tolist()}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sets))
    args.out.with_suffix(".txt").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))
    print(f"{len(sets)} sets written to {args.out}")


if __name__ == "__main__":
    main()
