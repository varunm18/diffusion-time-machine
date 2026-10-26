#!/usr/bin/env python
"""Step 4: summarize all processed scenes (era counts, posed counts, go/no-go gate).

Writes ``<data>/tables/census_scenes.csv`` (one row per scene) and
``<data>/tables/census_models.csv`` (one row per COLMAP model with >= 50 files),
then prints the headline numbers.

    python scripts/census.py
"""
from __future__ import annotations

import argparse

import pandas as pd

from timemachine.analysis.census import ERA_LABELS, load_scene, model_census, passes_gate, scene_summary
from timemachine.config import DataPaths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-old", type=int, default=10, help="gate: minimum pre-1970 photos")
    parser.add_argument("--n-modern", type=int, default=10, help="gate: minimum 2000+ photos")
    args = parser.parse_args()

    paths = DataPaths()
    names = pd.read_parquet(paths.scene_table, columns=["scene_id", "scene"]).dropna()
    names = dict(zip(names["scene_id"].astype(int), names["scene"]))
    scene_ids = sorted(int(p.stem) for p in (paths.tables / "dates").glob("*.parquet"))

    summaries, models, all_files = [], [], []
    for sid in scene_ids:
        files = load_scene(paths, sid)
        summaries.append(scene_summary(files, names.get(sid, "")))
        models.append(model_census(files, names.get(sid, "")))
        all_files.append(files)
    summary = passes_gate(pd.DataFrame(summaries), args.n_old, args.n_modern)
    model_table = pd.concat([m for m in models if len(m)], ignore_index=True)
    files = pd.concat(all_files, ignore_index=True)

    summary.to_csv(paths.tables / "census_scenes.csv", index=False)
    model_table.to_csv(paths.tables / "census_models.csv", index=False)

    unique = files.drop_duplicates("file_key")
    print(f"{len(summary)} scenes, {len(unique):,} unique files "
          f"({unique['photographic'].mean():.0%} photographic or unknown type)\n")
    print("Unique files per era (all types / photographic / photographic and posed):")
    eras = pd.DataFrame({
        "all": unique["era"].value_counts(),
        "photographic": unique[unique["photographic"]]["era"].value_counts(),
        "posed": unique[unique["photographic"] & unique["registered"]]["era"].value_counts(),
    }).reindex(ERA_LABELS).fillna(0).astype(int)
    print(eras.to_string(), "\n")
    print("Image types:", unique["medium"].value_counts().to_dict(), "\n")
    print(f"Gate (>= {args.n_old} pre-1970 and >= {args.n_modern} 2000+ photographic files):")
    print(f"  already posed in one COLMAP model: {summary['gate_posed'].sum()} / {len(summary)} scenes")
    print(f"  dated at all (needs our own posing): {summary['gate_any'].sum()} / {len(summary)} scenes\n")
    cols = ["scene", "files", "photo_pre-1970", "photo_2000+", "posed_pre-1970",
            "best_model", "best_model_pre-1970", "best_model_2000+", "gate_posed", "gate_any"]
    print(summary.sort_values("best_model_pre-1970", ascending=False)[cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
