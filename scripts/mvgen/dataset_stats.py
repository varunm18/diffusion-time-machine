#!/usr/bin/env python
"""Statistics of the multi-scene dataset (training and held-out scene folders), as one JSON file.

Reads the scene lists written by ``scripts/mvgen/write_scene_lists.py`` and, per folder, ``views.parquet``
and ``depth_stats.parquet``; names, places and types come from the scene-pool selection table. For the
dataset report (what the multi-scene run trains on). Light CPU only (small parquet files).

    python scripts/mvgen/dataset_stats.py --out data/multiscene_plan/dataset_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "data" / "multiscene_plan"
OLD_BEFORE = 1970.0
DECADES = list(range(1840, 2030, 10))


def folder_id(folder: Path) -> int:
    return int(folder.name.split("_")[0])


def scene_stats(folder: Path) -> dict:
    views = pd.read_parquet(folder / "views.parquet")
    depth = pd.read_parquet(folder / "depth_stats.parquet")
    depth_ok = np.zeros(len(views), dtype=bool)
    depth_ok[depth["row"].to_numpy()[depth["ok"].to_numpy(dtype=bool)]] = True
    usable = views["usable"].to_numpy(dtype=bool)
    dates = views["date_mid"].to_numpy(dtype=float)[usable]
    mono = views["is_monochrome"].fillna(False).to_numpy(dtype=bool)[usable]
    decade = np.clip((np.floor(dates / 10) * 10).astype(int), DECADES[0], DECADES[-1])
    return {
        "folder": str(folder), "views": int(len(views)), "usable": int(usable.sum()),
        "depth_ok": int((usable & depth_ok).sum()),
        "old": int((dates < OLD_BEFORE).sum()), "y1970_1999": int(((dates >= 1970) & (dates < 2000)).sum()),
        "y2000": int((dates >= 2000).sum()),
        "oldest": float(dates.min()) if len(dates) else None, "newest": float(dates.max()) if len(dates) else None,
        "median_year": float(np.median(dates)) if len(dates) else None,
        "monochrome": int(mono.sum()), "monochrome_modern": int((mono & (dates >= OLD_BEFORE)).sum()),
        "decades": {str(d): int((decade == d).sum()) for d in DECADES},
        "date_conf_mean": float(views["date_confidence"].to_numpy(dtype=float)[usable].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=PLAN / "dataset_stats.json")
    args = parser.parse_args()
    meta = pd.read_csv(PLAN / "scene_pool" / "selection.csv").set_index("scene_id")
    events = pd.read_csv(PLAN / "scene_pool" / "events_table.csv")
    scenes = []
    for split, listing in (("train", "scenes_train.txt"), ("heldout", "scenes_heldout.txt")):
        for line in (PLAN / listing).read_text().split():
            folder = Path(line)
            sid = folder_id(folder)
            row = meta.loc[sid] if sid in meta.index else None
            stats = scene_stats(folder)
            stats.update(split=split, scene_id=sid, name=str(row["scene"]).replace("_", " ") if row is not None else str(sid))
            if row is not None:
                for key in ("role", "country", "continent", "city", "type", "known_change", "lat", "lon", "top_source"):
                    value = row[key]
                    stats[key] = None if pd.isna(value) else (value.item() if hasattr(value, "item") else value)
            dated = events[(events["scene_id"] == sid) & events["year"].notna()]
            stats["events"] = [f"{int(e.year)}: {e.event}" for e in dated.itertuples() if e.kind != "construction/extension"][:6]
            scenes.append(stats)
    args.out.write_text(json.dumps({"old_before": OLD_BEFORE, "decades": DECADES, "scenes": scenes}, indent=1))
    train = [s for s in scenes if s["split"] == "train"]
    print(f"{len(train)} training scenes, {len(scenes) - len(train)} held out; "
          f"usable views {sum(s['usable'] for s in scenes)}, old {sum(s['old'] for s in scenes)} -> {args.out}")


if __name__ == "__main__":
    main()
