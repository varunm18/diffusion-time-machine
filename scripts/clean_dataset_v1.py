#!/usr/bin/env python
"""Dataset v1, step 2c: shape check and near-duplicate clusters for downloaded views.

Adds four columns to every ``<data>/datasets/<scene_id>_model<m>/views.parquet``
(scripts/build_dataset_v1.py): ``shape_status``, ``dhash`` (hex), ``dup_cluster``
and ``drop_reason`` (empty for views to keep). No row is deleted. Hashes are
cached, so trying another ``--dup-distance`` takes seconds.

Also saves contact sheets of image pairs at each hash distance to
``<data>/datasets/dup_pairs_v1/``, to choose the distance by eye.

    python scripts/clean_dataset_v1.py
    python scripts/clean_dataset_v1.py --dup-distance 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from timemachine.config import DataPaths
from timemachine.data.cleaning import dhash, duplicate_clusters, hamming_matrix, shape_status

BANDS = ((0, 0), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12))
THRESHOLDS = (0, 2, 4, 6, 8, 10, 12)


def load_hashes(cache: Path, files: pd.DataFrame, workers: int) -> pd.DataFrame:
    """dHash (hex) per (scene_id, file_key), computed for files not yet in ``cache``."""
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["scene_id", "file_key", "dhash"])
    todo = files.merge(done[["scene_id", "file_key"]], on=["scene_id", "file_key"], how="left", indicator=True)
    todo = todo[todo["_merge"] == "left_only"]
    if len(todo):
        print(f"hashing {len(todo):,} images")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            hashes = list(pool.map(lambda p: f"{dhash(Image.open(p)):016x}", todo["image_path"]))
        new = pd.DataFrame({"scene_id": todo["scene_id"].to_numpy(), "file_key": todo["file_key"].to_numpy(),
                            "dhash": hashes})
        done = pd.concat([done, new], ignore_index=True)
        done.to_parquet(cache, index=False)
    return done


def pair_sheet(pairs: list[tuple[str, str, int]], thumb: int = 224) -> Image.Image:
    """One row per pair: the two images side by side, labelled with their hash distance."""
    sheet = Image.new("RGB", (2 * thumb + 60, thumb * len(pairs)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (a, b, dist) in enumerate(pairs):
        for col, path in enumerate((a, b)):
            sheet.paste(Image.open(path).convert("RGB").resize((thumb, thumb)), (col * thumb, row * thumb))
        draw.text((2 * thumb + 8, row * thumb + 8), f"d={dist}", fill=(200, 0, 0))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dup-distance", type=int, default=6, help="max differing hash bits for a duplicate")
    parser.add_argument("--pairs-per-band", type=int, default=8, help="example pairs per contact sheet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = DataPaths()
    models = pd.read_parquet(paths.tables / "models_v1.parquet")
    models = models[models["role"] != "excluded"]
    tables = {(int(m.scene_id), int(m.model)): paths.dataset_dir(int(m.scene_id), int(m.model)) / "views.parquet"
              for m in models.itertuples()}
    views = {key: pd.read_parquet(path) for key, path in tables.items() if path.exists()}
    print(f"{len(views)} of {len(tables)} models have a views.parquet")
    new_columns = ["shape_status", "dhash", "dup_cluster", "drop_reason"]
    everything = pd.concat([v.drop(columns=new_columns, errors="ignore") for v in views.values()], ignore_index=True)
    files = everything.drop_duplicates(["scene_id", "file_key"])[["scene_id", "file_key", "image_path"]]
    datasets = paths.root / "datasets"
    files = files.merge(load_hashes(datasets / "dhash_v1.parquet", files, args.workers), on=["scene_id", "file_key"])

    # Near-duplicates are searched within a scene (its models share one image cache).
    rng = np.random.default_rng(args.seed)
    in_cluster = {t: 0 for t in THRESHOLDS}
    band_pairs: dict[tuple[int, int], list] = {band: [] for band in BANDS}
    cluster_of: dict[tuple[int, str], str] = {}
    for sid, group in files.groupby("scene_id"):
        group = group.reset_index(drop=True)
        dist = hamming_matrix([int(h, 16) for h in group["dhash"]])
        for t in THRESHOLDS:
            labels = duplicate_clusters(dist, t)
            in_cluster[t] += int((np.bincount(labels)[labels] > 1).sum())
        labels = duplicate_clusters(dist, args.dup_distance)
        for key, label in zip(group["file_key"], labels):
            cluster_of[(int(sid), key)] = f"{sid}/{group['file_key'][label]}"
        upper = np.triu(np.ones_like(dist, dtype=bool), k=1)
        for lo, hi in BANDS:
            for a, b in zip(*np.nonzero(upper & (dist >= lo) & (dist <= hi))):
                band_pairs[(lo, hi)].append((group["image_path"][a], group["image_path"][b], int(dist[a, b])))

    print(f"\n{len(files):,} unique files. Files that share a duplicate cluster, by max hash distance:")
    for t in THRESHOLDS:
        print(f"  <= {t:>2} bits: {in_cluster[t]:>6,} files ({in_cluster[t] / len(files):.1%})")
    sheets = datasets / "dup_pairs_v1"
    sheets.mkdir(parents=True, exist_ok=True)
    print("\nExample pairs per distance band:")
    for (lo, hi), pairs in band_pairs.items():
        if not pairs:
            print(f"  {lo}-{hi} bits: no pairs")
            continue
        chosen = [pairs[i] for i in rng.choice(len(pairs), min(args.pairs_per_band, len(pairs)), replace=False)]
        out = sheets / f"dist_{lo:02d}-{hi:02d}.jpg"
        pair_sheet(chosen).save(out, quality=90)
        print(f"  {lo}-{hi} bits: {len(pairs):,} pairs -> {out}")

    status_counts: dict[str, int] = {}
    for (sid, model), table in views.items():
        table = table.drop(columns=new_columns, errors="ignore")
        table["shape_status"] = [shape_status(r.image_width, r.image_height, r.camera_width, r.camera_height)
                                 for r in table.itertuples()]
        table = table.merge(files[["scene_id", "file_key", "dhash"]], on=["scene_id", "file_key"], how="left")
        table["dup_cluster"] = [cluster_of[(sid, key)] for key in table["file_key"]]
        table["drop_reason"] = table["shape_status"].where(table["shape_status"] != "match", "")
        for status, n in table["shape_status"].value_counts().items():
            status_counts[status] = status_counts.get(status, 0) + int(n)
        tmp = tables[(sid, model)].with_name("views.parquet.part")
        table.to_parquet(tmp, index=False)
        tmp.replace(tables[(sid, model)])
    print(f"\nShape check over all views: {status_counts}")
    print(f"Wrote shape_status, dhash, dup_cluster (distance <= {args.dup_distance}) and drop_reason "
          f"to {len(views)} view tables.")


if __name__ == "__main__":
    main()
