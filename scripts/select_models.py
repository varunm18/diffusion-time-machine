#!/usr/bin/env python
"""Dataset v1, step 1c: pick COLMAP models by the samples they can produce.

For every model of the candidate scenes, builds the view table (posed + dated +
photographic files), counts the views that can anchor a target group per era,
and assigns a role (see ``timemachine.data.selection``). Needs the scenes to be
processed first (scripts/process_scenes.py). Writes ``<data>/tables/models_v1.parquet``.

    python scripts/select_models.py --scene-file configs/scenes_v1.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from timemachine.config import DataPaths
from timemachine.data.selection import WIDE_OLD_TOLERANCE, model_row
from timemachine.data.views import build_view_table
from timemachine.megascenes.index import select_scenes


def read_scene_file(path: Path) -> list[str]:
    """One scene name or id per line; blank lines and '#' comments ignored."""
    lines = (ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines())
    return [ln for ln in lines if ln]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-file", type=Path, required=True, help="text file with one scene per line")
    args = parser.parse_args()

    paths = DataPaths()
    scenes = select_scenes(pd.read_parquet(paths.scene_table), names=read_scene_file(args.scene_file))
    rows = []
    for scene in scenes.itertuples():
        sid = int(scene.scene_id)
        models = sorted(pd.read_parquet(paths.poses_table(sid), columns=["model"])["model"].unique())
        for model in models:
            views = build_view_table(paths, sid, int(model))
            rows.append({"scene_id": sid, "scene": scene.scene, "model": int(model), **model_row(views)})
    table = pd.DataFrame(rows).sort_values(["old_anchors", "old_views"], ascending=False)
    out = paths.tables / "models_v1.parquet"
    table.to_parquet(out, index=False)

    kept = table[table["role"] != "excluded"]
    target = kept[kept["role"] == "old_target"]
    wide = kept[kept["old_anchors_wide"] > 0]
    print(f"{len(table)} models in {table['scene_id'].nunique()} scenes examined; wrote {out}\n")
    print(f"A. keep old_target + old_input: {len(kept)} models, {kept['scene_id'].nunique()} scenes, "
          f"{kept['old_views'].sum():,} old views, {kept['modern_views'].sum():,} modern views")
    print(f"B. old_target only:             {len(target)} models, {target['scene_id'].nunique()} scenes, "
          f"{target['old_views'].sum():,} old views, {target['old_anchors'].sum():,} old anchors")
    print(f"C. old tolerance {WIDE_OLD_TOLERANCE:.0f} years:      {len(wide)} models, "
          f"{wide['scene_id'].nunique()} scenes with an old anchor, {wide['old_anchors_wide'].sum():,} old anchors\n")
    cols = ["scene", "model", "views", "old_views", "modern_views", "old_anchors", "old_anchors_wide", "role"]
    print(kept[cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
