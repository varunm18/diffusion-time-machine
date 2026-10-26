#!/usr/bin/env python
"""Build a single-scene dataset: posed + dated views of one COLMAP model, with images.

Requires the scene to be processed (scripts/process_scenes.py) and the torch
extra (``uv pip install -e ".[torch]"``). Writes
``<data>/datasets/<scene_id>_model<m>/views.parquet`` plus preview contact sheets.

    python scripts/build_scene_dataset.py --scene Cathédrale_Notre-Dame_de_Paris
    python scripts/build_scene_dataset.py --scene 25512 --model 0 --max-modern 800
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from timemachine.analysis.census import load_scene, model_census
from timemachine.config import DataPaths
from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.images import download_view_images
from timemachine.data.preview import save_sample_previews
from timemachine.data.sampling import SamplingConfig
from timemachine.data.views import build_view_table, confident_registrations
from timemachine.megascenes.index import select_scenes
from timemachine.megascenes.s3 import MegaScenesBucket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="scene name (Commons category) or id")
    parser.add_argument("--model", type=int, help="COLMAP model (default: the one with most pre-1970 photos)")
    parser.add_argument("--max-modern", type=int, help="keep at most this many 2000+ views (random subset)")
    parser.add_argument("--max-side", type=int, default=1024, help="longest side of cached images")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vggt-tag", help="also add confident VGGT-Omega registrations from "
                        "<data>/posing/<id>_model<m>/register_<tag>_views.parquet")
    parser.add_argument("--max-center-spread", type=float, default=0.1,
                        help="confidence filter for VGGT poses (fraction of the scene size)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = DataPaths()
    scene = select_scenes(pd.read_parquet(paths.scene_table), names=[args.scene]).iloc[0]
    sid = int(scene.scene_id)
    model = args.model
    if model is None:
        models = model_census(load_scene(paths, sid), scene.scene)
        model = int(models.sort_values(["pre-1970", "2000+"], ascending=False)["model"].iloc[0])
    print(f"{scene.scene} ({sid}), model {model}")

    views = build_view_table(paths, sid, model)
    if args.max_modern is not None:
        modern = views[views["era"] == "2000+"]
        drop = modern.sample(max(len(modern) - args.max_modern, 0), random_state=args.seed).index
        views = views.drop(index=drop).reset_index(drop=True)
    print(f"{len(views)} views to fetch; eras: {views['era'].value_counts().to_dict()}")

    views = download_view_images(MegaScenesBucket(), paths, views, args.max_side, args.workers)
    if views.attrs.get("failed"):
        print(f"{len(views.attrs['failed'])} images failed to download (skipped)")
    if args.vggt_tag:
        registered = pd.read_parquet(paths.posing_dir(sid, model) / f"register_{args.vggt_tag}_views.parquet")
        confident = confident_registrations(registered, max_center_spread=args.max_center_spread)
        confident = confident[~confident["file_key"].isin(views["file_key"])]
        print(f"adding {len(confident)} of {len(registered)} VGGT-registered views (confidence filter)")
        views = pd.concat([views, confident], ignore_index=True)
    out = paths.dataset_dir(sid, model)
    out.mkdir(parents=True, exist_ok=True)
    views.to_parquet(out / "views.parquet", index=False)

    summary = views.groupby(["era", "pose_source"]).agg(views=("file_key", "size"),
                                                        monochrome=("is_monochrome", "mean"))
    print(f"\nwrote {out / 'views.parquet'}\n{summary.round(2).to_string()}")

    dataset = DatedViewsDataset(views, SamplingConfig(old_target_fraction=0.5), seed=args.seed)
    print(f"\n{len(dataset.anchors)} views can anchor a sample of {dataset.config.n_views} views")
    for path in save_sample_previews(dataset, out / "previews"):
        print("preview:", path)
    ratios = views["image_width"] / views["image_height"] - views["camera_width"] / views["camera_height"]
    print(f"max aspect mismatch between cached image and COLMAP camera: {np.abs(ratios).max():.4f}")


if __name__ == "__main__":
    main()
