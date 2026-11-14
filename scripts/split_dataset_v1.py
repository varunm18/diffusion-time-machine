#!/usr/bin/env python
"""Dataset v1, step 1d: hold out whole scene clusters for testing.

Groups the selected scenes into clusters of scenes that share Commons files
(see ``timemachine.data.splits``), picks the held-out clusters and writes
``<data>/tables/splits_v1.parquet`` (one row per scene: cluster, split).
Then checks the downloaded views for leaks between the splits.

    python scripts/split_dataset_v1.py                       # automatic choice
    python scripts/split_dataset_v1.py --holdout 25512,113197   # hold out these scenes' clusters
"""
from __future__ import annotations

import argparse

import pandas as pd

from timemachine.config import DataPaths
from timemachine.data.splits import choose_holdout, find_leaks, scene_clusters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-shared", type=int, default=1, help="shared files that link two scenes")
    parser.add_argument("--fraction", type=float, default=0.1, help="share of clusters to hold out")
    parser.add_argument("--holdout", default="", help="comma-separated scene ids to hold out instead")
    args = parser.parse_args()

    paths = DataPaths()
    models = pd.read_parquet(paths.tables / "models_v1.parquet")
    models = models[models["role"] != "excluded"]
    scenes = models.groupby(["scene_id", "scene"], as_index=False).agg(
        models=("model", "nunique"), old_views=("old_views", "sum"), old_anchors=("old_anchors", "sum"))
    files = {int(sid): set(pd.read_parquet(paths.image_metadata_table(int(sid)), columns=["file_key"])["file_key"])
             for sid in scenes["scene_id"]}
    for k in sorted({1, 3, 10, args.min_shared}):
        sizes = pd.Series(scene_clusters(files, k)).value_counts()
        print(f"min_shared {k:>2}: {len(sizes)} clusters, {(sizes > 1).sum()} with several scenes, largest {sizes.max()}")

    scenes["cluster"] = scenes["scene_id"].map(scene_clusters(files, args.min_shared))
    clusters = scenes.groupby("cluster", as_index=False).agg(
        scenes=("scene_id", "nunique"), old_views=("old_views", "sum"), names=("scene", lambda s: ", ".join(s)))
    if args.holdout:
        wanted = {int(s) for s in args.holdout.split(",")}
        held = sorted(scenes.loc[scenes["scene_id"].isin(wanted), "cluster"].unique())
    else:
        held = choose_holdout(clusters, args.fraction)
    scenes["split"] = scenes["cluster"].isin(held).map({True: "test", False: "train"})
    scenes.to_parquet(paths.tables / "splits_v1.parquet", index=False)

    pd.set_option("display.width", 200, "display.max_colwidth", 90)
    print(f"\nClusters with several scenes (using min_shared {args.min_shared}):")
    print(clusters[clusters["scenes"] > 1].sort_values("old_views", ascending=False).to_string(index=False))
    print(f"\nHeld out: {len(held)} of {len(clusters)} clusters")
    print(scenes[scenes["split"] == "test"][["scene", "scene_id", "models", "old_views", "old_anchors"]]
          .sort_values("old_views", ascending=False).to_string(index=False))
    summary = scenes.groupby("split").agg(scenes=("scene_id", "nunique"), models=("models", "sum"),
                                          old_views=("old_views", "sum"), old_anchors=("old_anchors", "sum"))
    print(f"\n{summary.to_string()}")

    views = []
    for m in models.itertuples():
        path = paths.dataset_dir(int(m.scene_id), int(m.model)) / "views.parquet"
        if path.exists():
            views.append(pd.read_parquet(path, columns=["scene_id", "file_key", "dhash"]))
    if views:
        views = pd.concat(views).merge(scenes[["scene_id", "split"]], on="scene_id")
        name_leaks, pixel_leaks = find_leaks(views, "file_key"), find_leaks(views, "dhash")
        print(f"\nLeak check on {len(views):,} downloaded views: {len(name_leaks)} file names and "
              f"{len(pixel_leaks)} identical image hashes occur in both splits")
        if name_leaks:
            raise SystemExit(f"file names in both splits (first 5): {name_leaks[:5]}")


if __name__ == "__main__":
    main()
