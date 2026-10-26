#!/usr/bin/env python
"""Steps 1-3 per scene: harvest Commons metadata, fetch poses, date every image.

Each step is cached per scene under ``<data>/tables/``, so reruns only do new
work (pass --overwrite to redo). Requires step 0 (scripts/fetch_index.py).

    python scripts/process_scenes.py --scene-file configs/scenes_pilot.txt
    python scripts/process_scenes.py --scenes "Brandenburg_Gate" 110925
    python scripts/process_scenes.py --top 200            # largest reconstructions
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import pandas as pd

from timemachine.config import DataPaths
from timemachine.dating.pipeline import date_scene
from timemachine.megascenes.index import load_recon_metadata, select_scenes
from timemachine.megascenes.metadata import harvest_scene
from timemachine.megascenes.poses import fetch_scene_poses
from timemachine.megascenes.s3 import MegaScenesBucket

STEPS = ("meta", "poses", "dates")


def read_scene_file(path: Path) -> list[str]:
    """One scene name or id per line; blank lines and '#' comments ignored."""
    lines = (ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines())
    return [ln for ln in lines if ln]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--scenes", nargs="+", help="scene names (Commons categories) or ids")
    which.add_argument("--scene-file", type=Path, help="text file with one scene per line")
    which.add_argument("--top", type=int, help="the N scenes with the largest reconstructions")
    parser.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {STEPS}")
    parser.add_argument("--min-model-images", type=int, default=50,
                        help="only fetch poses of COLMAP models at least this large (default 50)")
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads per scene")
    parser.add_argument("--overwrite", action="store_true", help="recompute cached tables")
    args = parser.parse_args()
    steps = set(args.steps.split(","))
    assert steps <= set(STEPS), f"unknown steps: {steps - set(STEPS)}"

    paths, bucket = DataPaths(), MegaScenesBucket()
    scene_table = pd.read_parquet(paths.scene_table)
    names = args.scenes or (read_scene_file(args.scene_file) if args.scene_file else None)
    scenes = select_scenes(scene_table, names=names, top=args.top)
    recon = load_recon_metadata(paths.recon_metadata)

    # Subcategory folders of the selected scenes (one raw_metadata.json each).
    index = pd.read_parquet(paths.images_index, columns=["cat", "subcat"],
                            filters=[("cat", "in", scenes["scene"].tolist())])
    subcats = index.groupby("cat")["subcat"].unique()

    print(f"{len(scenes)} scenes, steps: {sorted(steps)}")
    failures = []
    for i, row in enumerate(scenes.itertuples(), 1):
        start = time.time()
        sid = int(row.scene_id)
        try:
            msg = []
            if "meta" in steps:
                meta = harvest_scene(bucket, paths, sid, subcats[row.scene], args.workers, args.overwrite)
                missing = meta.attrs.get("missing_subcats", [])
                msg.append(f"{len(meta):,} files" + (f" ({len(missing)} subcats missing)" if missing else ""))
            if "poses" in steps:
                models = recon[(recon.scene_id == sid) & (recon.n_images >= args.min_model_images)]["model"]
                poses = fetch_scene_poses(bucket, paths, sid, models.tolist(), args.overwrite)
                msg.append(f"{len(models)} models, {poses['file_key'].nunique():,} posed files")
            if "dates" in steps:
                dates = date_scene(paths, sid, args.overwrite)
                msg.append(f"{dates['date_lo'].notna().mean():.0%} dated")
            print(f"[{i}/{len(scenes)}] {row.scene} ({sid}): {', '.join(msg)} [{time.time() - start:.0f}s]",
                  flush=True)
        except Exception as exc:  # keep going; report at the end
            failures.append((row.scene, repr(exc)))
            print(f"[{i}/{len(scenes)}] {row.scene} ({sid}): FAILED {exc!r}", flush=True)
            traceback.print_exc()
    if failures:
        print(f"\n{len(failures)} scenes failed:")
        for name, err in failures:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
