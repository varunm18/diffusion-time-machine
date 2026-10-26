#!/usr/bin/env python
"""Pose old photos with VGGT-Omega, anchored to a MegaScenes COLMAP model (GPU).

Two modes, both using the scene's dataset view table (scripts/build_scene_dataset.py):

  eval      Treat the old photos COLMAP *did* register as unknown, re-pose them with
            VGGT-Omega against modern anchors, and compare with their COLMAP poses.
            This measures how well cross-decade registration works.
  register  Pose the scene's old photos COLMAP did *not* register (downloads them).

Needs VGGT_OMEGA_ROOT (VGGT-Omega checkout, used read-only) and the checkpoint in
the HF cache (or --checkpoint). Outputs go to <data>/posing/<scene_id>_model<m>/.

    VGGT_OMEGA_ROOT=/path/to/vggt-omega python scripts/pose_with_vggt.py --scene 25512 --model 0 --mode eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from timemachine.config import DataPaths
from timemachine.data.cameras import c2w_to_colmap, colmap_to_c2w
from timemachine.data.images import download_view_images
from timemachine.data.views import confident_registrations, registered_to_views, unposed_candidates
from timemachine.megascenes.s3 import MegaScenesBucket
from timemachine.posing.register import (
    RegistrationConfig, evaluate_registration, register_images, scene_size, summarize_errors,
)
from timemachine.posing.retrieval import GlobalDescriptor
from timemachine.posing.vggt import VGGTOmegaRunner


def view_c2ws(views: pd.DataFrame) -> np.ndarray:
    return colmap_to_c2w(views[["qw", "qx", "qy", "qz"]].to_numpy(), views[["tx", "ty", "tz"]].to_numpy())


def load_or_compute_embeddings(cache: Path, images: pd.DataFrame) -> dict[str, np.ndarray]:
    """DINOv2 descriptors per file_key, cached in an .npz next to the posing outputs."""
    emb = dict(np.load(cache)) if cache.exists() else {}
    todo = images.drop_duplicates("file_key")
    todo = todo[~todo["file_key"].isin(emb)]
    if len(todo):
        print(f"embedding {len(todo)} images with DINOv2")
        vectors = GlobalDescriptor().embed(todo["image_path"].tolist())
        emb.update(zip(todo["file_key"], vectors))
        np.savez(cache, **emb)
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=int, required=True, help="scene id")
    parser.add_argument("--model", type=int, required=True, help="COLMAP model of the dataset")
    parser.add_argument("--mode", choices=["eval", "register"], required=True)
    parser.add_argument("--before", type=float, default=1970.0, help="old = dated before this year")
    parser.add_argument("--checkpoint", type=Path, help="VGGT-Omega checkpoint (default: HF cache)")
    defaults = RegistrationConfig()
    parser.add_argument("--queries-per-batch", type=int, default=defaults.queries_per_batch)
    parser.add_argument("--anchors-per-batch", type=int, default=defaults.anchors_per_batch)
    parser.add_argument("--repeats", type=int, default=defaults.repeats)
    parser.add_argument("--alignment", choices=["rotation_first", "centers"], default=defaults.alignment)
    parser.add_argument("--center-threshold", type=float, default=defaults.center_threshold)
    parser.add_argument("--no-retrieval", action="store_true",
                        help="spread anchors over the scene instead of picking look-alikes (DINOv2)")
    parser.add_argument("--tag", default="", help="suffix for output files (to compare settings)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = DataPaths()
    views = pd.read_parquet(paths.dataset_dir(args.scene, args.model) / "views.parquet")
    anchors = views[views["date_mid"] >= 2000].reset_index(drop=True)
    if args.mode == "eval":
        queries = views[views["date_mid"] < args.before].reset_index(drop=True)
    else:
        queries = unposed_candidates(paths, args.scene, args.model, args.before)
        queries = download_view_images(MegaScenesBucket(), paths, queries).reset_index(drop=True)
    print(f"{len(queries)} queries, {len(anchors)} candidate anchors")

    out_dir = paths.posing_dir(args.scene, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_emb = anchor_emb = None
    if not args.no_retrieval:
        emb = load_or_compute_embeddings(out_dir / "dinov2_embeddings.npz",
                                         pd.concat([queries, anchors])[["file_key", "image_path"]])
        query_emb = np.stack([emb[k] for k in queries["file_key"]])
        anchor_emb = np.stack([emb[k] for k in anchors["file_key"]])

    runner = VGGTOmegaRunner(args.checkpoint)
    config = RegistrationConfig(args.queries_per_batch, args.anchors_per_batch, args.repeats,
                                alignment=args.alignment, center_threshold=args.center_threshold,
                                seed=args.seed)
    anchor_c2ws = view_c2ws(anchors)
    results, batches = register_images(runner.predict, queries["image_path"].tolist(),
                                       anchors["image_path"].tolist(), anchor_c2ws, config,
                                       query_emb, anchor_emb)
    print(f"batches used: {batches['used'].sum()}/{len(batches)}; "
          f"median inlier anchors {batches['n_inliers'].median():.0f}")

    name = f"{args.mode}{'_' + args.tag if args.tag else ''}"
    batches.to_csv(out_dir / f"{name}_batches.csv", index=False)
    table = pd.concat([queries[["file_key", "date_mid", "medium"]], results.drop(columns=["query"])], axis=1)

    if args.mode == "eval":
        table = evaluate_registration(table, view_c2ws(queries), scene_size(anchor_c2ws))
        summary = summarize_errors(table)
        print(json.dumps(summary, indent=2))
        summary["config"] = {k: v for k, v in vars(args).items() if k not in ("checkpoint",)}
        (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Store poses COLMAP-style so they can join the view table later.
    placed = table["c2w"].notna()
    qt = [c2w_to_colmap(c) if ok else (np.full(4, np.nan), np.full(3, np.nan))
          for c, ok in zip(table["c2w"], placed)]
    table[["qw", "qx", "qy", "qz"]] = np.stack([q for q, _ in qt])
    table[["tx", "ty", "tz"]] = np.stack([t for _, t in qt])
    table = table.drop(columns=["c2w"]).assign(pose_source="vggt_omega")
    table.to_parquet(out_dir / f"{name}_poses.parquet", index=False)
    print(f"wrote {out_dir / f'{name}_poses.parquet'}")
    if args.mode == "register":
        # Full view-table rows, ready to be merged into the scene dataset.
        views_out = registered_to_views(queries, table, args.model)
        views_out.to_parquet(out_dir / f"{name}_views.parquet", index=False)
        print(f"wrote {out_dir / f'{name}_views.parquet'} "
              f"({len(confident_registrations(views_out))} of {len(views_out)} pass the default confidence filter)")


if __name__ == "__main__":
    main()
