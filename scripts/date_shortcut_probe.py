#!/usr/bin/env python
"""Measure how much of our date signal is just the look of old film (CPU only).

Every date-conditioned experiment assumes an image's date says something about
what the *scene* looked like. But old photos are also grey, sepia, flat and
grainy, so a model can score well on dates while having learned only "brown
means old". This measures how big that shortcut is in our own data.

    python scripts/date_shortcut_probe.py --dataset data/datasets/25512_model0
    python scripts/date_shortcut_probe.py --dataset data/datasets/*_model0 --max-views 4000

It fits a deliberately weak linear probe on appearance-only statistics (colour,
tone, grain, framing - never scene content) to predict the capture year, under
three conditions: the photos as they are, in grayscale, and tone-matched. It
also fits the probe on one feature group at a time, and runs a causal check:
sepia-tone the *modern* photos and see how much younger the probe thinks they
are, with the scene untouched.

Train/test splits reuse the near-duplicate grouping from timemachine.eval.splits,
so the same photograph (or a burst from one spot) can never sit on both sides.

Reading the output: high skill means the date is trivially readable from the
medium alone, so any model trained on this data can reach a good date loss
without learning anything about how places change. See docs/findings.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from timemachine.analysis import shortcut as sc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, nargs="+", required=True, help="folder(s) with views.parquet")
    p.add_argument("--out", type=Path, help="output folder (default: data/experiments/date_shortcut)")
    p.add_argument("--max-views", type=int, default=4000, help="0 for all; images are read one by one")
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=1.0, help="ridge strength")
    p.add_argument("--old-before", type=float, default=1945.0, help="boundary for old/modern accuracy")
    p.add_argument("--modern-after", type=float, default=1995.0, help="photos counted as modern for the sepia test")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = args.out or Path("data/experiments/date_shortcut")
    out.mkdir(parents=True, exist_ok=True)

    views = sc.load_views(list(args.dataset), args.max_views, args.seed)
    years = views["date_mid"].to_numpy(float)
    train, groups = sc.grouped_split(views, args.test_fraction, args.seed)
    print(f"{len(views)} views from {views['dataset'].nunique()} scene(s); "
          f"{int(train.sum())} train / {int((~train).sum())} test, "
          f"{len(np.unique(groups))} near-duplicate groups")
    print(f"years {years.min():.0f}-{years.max():.0f}, "
          f"{(years < args.old_before).mean():.0%} before {args.old_before:.0f}")

    paths = views["image_path"].tolist()
    scores, cache = {}, {}
    for treatment in sc.TREATMENTS:
        print(f"  features: {treatment} ...", flush=True)
        X = cache[treatment] = sc.features_for(paths, treatment)
        scores[treatment] = sc.score_probe(X[train], years[train], X[~train], years[~train],
                                           args.alpha, args.old_before)

    table = sc.results_table(scores)
    group_table = sc.group_skill_table(cache["original"][train], years[train],
                                       cache["original"][~train], years[~train], args.alpha)

    # Causal check: tone the modern test photos and re-ask the untreated probe.
    shift = None
    modern = (~train) & (years >= args.modern_after)
    if modern.sum() >= 10:
        print(f"  sepia check on {int(modern.sum())} modern photos ...", flush=True)
        modern_paths = [paths[i] for i in np.flatnonzero(modern)]
        sepia_rows = []
        for path in modern_paths:
            with Image.open(path) as im:
                sepia_rows.append(sc.appearance_features(sc.apply_sepia(im.convert("RGB"))))
        shift = sc.sepia_shift(scores["original"]["probe"], cache["original"][modern],
                               sc.feature_matrix(sepia_rows), years[modern])
    else:
        print(f"  skipping the sepia check: only {int(modern.sum())} modern test photos")

    report = sc.render_report(table, shift, views["dataset"].nunique(), group_table)
    (out / "report.md").write_text(report, encoding="utf-8")
    table.to_csv(out / "treatments.csv", index=False)
    group_table.to_csv(out / "feature_groups.csv", index=False)
    pd.DataFrame({"file_key": views.get("file_key", views.index), "dataset": views["dataset"],
                  "year": years, "group": groups, "train": train}).to_csv(out / "split.csv", index=False)
    if shift:
        (out / "sepia_shift.json").write_text(json.dumps(shift, indent=2), encoding="utf-8")
    print("\n" + report)
    print(f"wrote {out}/report.md, treatments.csv, feature_groups.csv, split.csv")


if __name__ == "__main__":
    main()
