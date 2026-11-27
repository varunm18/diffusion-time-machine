#!/usr/bin/env python
"""Backfill k1/k2 into view tables built before the lens-distortion columns existed.

    python scripts/migrate_distortion.py --dataset data/datasets/* --dry-run
    python scripts/migrate_distortion.py --dataset data/datasets/*

``DatedViewsDataset`` requires those columns, so a stale table is rejected outright
and no training run can start. The coefficients are already in each reconstruction's
cached ``cameras.bin``, so this re-derives the poses table locally and joins them
on - no re-download, and the cached images are untouched.

Originals are kept beside each table as ``views.pre_distortion.parquet``, and the
poses tables as ``<scene>.pre_distortion.parquet``, so this is reversible.

Views whose coefficients cannot be used (uninvertible over their own frame, or with
no match) are dropped from the written table. Tables migrated before that rule
hold them as NaN instead; clean those with scripts/prune_unloadable_views.py.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timemachine.config import DataPaths                                    # noqa: E402
from timemachine.data.distortion import (                                   # noqa: E402
    BACKUP_SUFFIX, DISTORTION_COLUMNS, migrate_folder, parse_dataset_folder, render_report,
)
from timemachine.megascenes.poses import load_model                         # noqa: E402
from timemachine.megascenes.s3 import MegaScenesBucket                      # noqa: E402


def scene_poses(bucket, paths: DataPaths, scene_id: int, models: set[int],
                rewrite: bool, backup: bool, log=print) -> pd.DataFrame:
    """Re-derive one scene's poses from its cached reconstructions.

    Reads the reconstruction rather than the cached table, because the table is the
    stale artefact we are replacing. ``load_model`` pulls the radial coefficients
    straight out of ``cameras.bin``; when that file is cached - which it is for any
    scene whose dataset was built - this makes no network call.

    Every model already present in the stored table is re-derived, not only the ones
    with datasets, so the rewritten table stays as complete as the one it replaces.
    """
    table = paths.poses_table(scene_id)
    stored = pd.read_parquet(table) if table.is_file() else None
    if stored is not None and "model" in stored.columns:
        models = models | set(stored["model"].unique().tolist())

    frames, failed = [], []
    for model in sorted(models):
        try:
            frames.append(load_model(bucket, paths, scene_id, model))
        except Exception as exc:                       # noqa: BLE001 - one bad model is survivable
            failed.append(f"model{model} ({type(exc).__name__})")
    if failed:
        log(f"  scene {scene_id}: could not re-derive {', '.join(failed)}")
    if not frames:
        return pd.DataFrame()

    poses = pd.concat(frames, ignore_index=True)
    if rewrite and stored is not None and not poses.empty:
        if backup:
            spare = table.with_name(str(scene_id) + BACKUP_SUFFIX)
            if not spare.exists():
                stored.to_parquet(spare, index=False)
        poses.to_parquet(table, index=False)
    return poses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True, metavar="DIR",
                        help="dataset folders to migrate (shell globs are fine)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing anything")
    parser.add_argument("--no-backup", action="store_true",
                        help="do not keep the original tables beside the new ones")
    parser.add_argument("--no-rewrite-poses", action="store_true",
                        help="join onto the view tables but leave the poses tables stale "
                             "(they are what build_scene_dataset reads, so rewriting them "
                             "keeps future builds correct)")
    parser.add_argument("--keep-uninvertible", action="store_true",
                        help="write coefficients even where the radial map cannot be "
                             "inverted over the camera's own frame. COLMAP fits wild "
                             "values to weakly-observed cameras, and undistorting with "
                             "one warps a good image, so these are dropped by default")
    parser.add_argument("--allow-partial", action="store_true",
                        help="write tables where some rows found no coefficients, dropping "
                             "those views. They are never written as non-finite: load_view "
                             "raises on such a row rather than skipping it")
    args = parser.parse_args()

    paths, bucket = DataPaths(), MegaScenesBucket()

    # Group by scene so each reconstruction is read once even when a scene
    # contributes several COLMAP models.
    by_scene: dict[int, set[int]] = defaultdict(set)
    folders: dict[Path, tuple[int, int]] = {}
    unparsed = []
    for folder in args.dataset:
        parsed = parse_dataset_folder(folder)
        if parsed is None:
            unparsed.append(folder)
            continue
        folders[folder] = parsed
        by_scene[parsed[0]].add(parsed[1])
    for folder in unparsed:
        print(f"  skip  {folder.name:<22} not a <scene>_model<n> folder")
    if not folders:
        print("Nothing to migrate.")
        return 1

    print(f"Re-deriving poses for {len(by_scene)} scene(s) from cached reconstructions ...")
    poses = {scene: scene_poses(bucket, paths, scene, models,
                                rewrite=not args.no_rewrite_poses, backup=not args.no_backup)
             for scene, models in sorted(by_scene.items())}

    results = []
    for folder, (scene, model) in sorted(folders.items()):
        table = poses.get(scene)
        if table is None or table.empty:
            from timemachine.data.distortion import Migration

            results.append(Migration(folder=folder, note="no poses could be re-derived"))
            continue
        for_model = table[table["model"] == model] if "model" in table.columns else table
        results.append(migrate_folder(folder, for_model, dry_run=args.dry_run,
                                      backup=not args.no_backup,
                                      allow_partial=args.allow_partial,
                                      drop_uninvertible=not args.keep_uninvertible))

    print()
    print(render_report(results))
    if args.dry_run:
        print("\nNothing was written (--dry-run). Re-run without it to migrate.")
    return 0 if all(r.ok or r.rows == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
