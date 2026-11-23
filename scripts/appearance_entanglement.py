#!/usr/bin/env python
"""Can we condition on appearance without handing the model the date? (CPU only)

`scripts/date_shortcut_probe.py` showed that our appearance statistics predict era
at AUC 0.89. `timemachine.model.appearance_film` conditions the model on those same
statistics, which risks giving it the date through a side channel - possibly making
date-blindness worse rather than better.

This checks whether an era-free part of appearance exists, and whether it still
carries something we want the model to see (lighting).

Step 1 - make a labelling sheet (no network needed; thumbnails are embedded):

    python scripts/appearance_entanglement.py --dataset data/datasets/*_model0 \
        --make-labels --n-labels 150

Step 2 - label day vs night in the browser, press "Download lighting_labels.csv".

Step 3 - run the analysis:

    python scripts/appearance_entanglement.py --dataset data/datasets/*_model0 \
        --labels ~/Downloads/lighting_labels.csv

Without --labels it still reports whether era can be removed, but not whether
anything useful survives, which is the half that decides the design.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from timemachine.analysis import entanglement as ent
from timemachine.analysis import shortcut as sc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, nargs="+", required=True, help="folder(s) with views.parquet")
    p.add_argument("--out", type=Path, help="output folder (default: data/experiments/appearance_entanglement)")
    p.add_argument("--max-views", type=int, default=4000, help="0 for all")
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--old-before", type=float, default=1945.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--make-labels", action="store_true", help="write a lighting labelling sheet and stop")
    p.add_argument("--n-labels", type=int, default=150, help="images to put on the sheet")
    p.add_argument("--labels", type=Path, help="filled lighting_labels.csv")
    args = p.parse_args()

    out = args.out or Path("data/experiments/appearance_entanglement")
    out.mkdir(parents=True, exist_ok=True)

    # Check this before the expensive part: computing appearance features for a few
    # thousand images takes minutes, and finding out afterwards that the labels file
    # was missing or mis-spelled wastes all of it.
    if args.labels and not args.labels.exists():
        sheet = out / "lighting_sheet.html"
        raise SystemExit(
            f"{args.labels} does not exist.\n"
            + (f"Label {sheet} in a browser first, then pass the CSV it downloads."
               if sheet.exists() else
               "Run with --make-labels first, label the sheet it writes, then pass the CSV."))

    views = sc.load_views(list(args.dataset), args.max_views, args.seed)
    years = views["date_mid"].to_numpy(float)
    print(f"{len(views)} views from {views['dataset'].nunique()} scene(s); "
          f"{(years < args.old_before).mean():.0%} before {args.old_before:.0f}")

    if args.make_labels:
        # Spread the sheet over eras so lighting and era are not confounded by the draw
        # itself: a sheet of mostly modern photos would make "night" and "modern" agree.
        sample = ent.era_stratified_sample(views, args.n_labels, args.seed)
        ids = sample["file_key"].astype(str).tolist()
        print(f"rendering {len(ids)} thumbnails ...", flush=True)
        uris = [ent.thumbnail_uri(p) for p in sample["image_path"]]
        (out / "lighting_sheet.html").write_text(
            ent.render_lighting_sheet(ids, uris, "Lighting labels"), encoding="utf-8")
        sample[["file_key", "image_path", "date_mid", "dataset"]].to_csv(
            out / "lighting_key.csv", index=False, encoding="utf-8")
        print(f"\nwrote {out}/lighting_sheet.html  (label this; it needs no network)")
        print(f"      {out}/lighting_key.csv      (do not show the labeller: it has the dates)")
        return

    print("computing appearance features ...", flush=True)
    X = sc.features_for(views["image_path"].tolist(), "original")
    train, _ = sc.grouped_split(views, args.test_fraction, args.seed)
    report = ent.residual_report(X, years, train, args.old_before, args.alpha)

    lighting = None
    if args.labels:
        labels = pd.read_csv(args.labels, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if {"id", "lighting"} - set(labels.columns):
            raise SystemExit(f"{args.labels} needs columns id and lighting")
        merged = views[["file_key"]].astype(str).merge(
            labels.rename(columns={"id": "file_key"}), on="file_key", how="left")
        merged["lighting"] = merged["lighting"].fillna("")
        is_night, labelled = ent.parse_lighting_labels(merged)
        print(f"{labelled.sum()} usable lighting labels ({is_night[labelled].sum()} night)")
        if labelled.sum() < 20:
            print("  too few to analyse; skipping the lighting half")
        else:
            lighting = ent.label_separability(X[labelled], is_night[labelled],
                                              train[labelled], report["direction"], args.alpha)

    text = ent.render_report(report, lighting, views["dataset"].nunique())
    (out / "report.md").write_text(text, encoding="utf-8")
    report["loadings"].to_csv(out / "era_loadings.csv", index=False)
    ent.group_alignment(report["direction"]).to_csv(out / "group_alignment.csv", index=False)
    np.save(out / "era_direction.npy", report["direction"])
    print("\n" + text)
    print(f"wrote {out}/report.md, era_loadings.csv, group_alignment.csv, era_direction.npy")


if __name__ == "__main__":
    main()
