#!/usr/bin/env python
"""Build the MVGenMaster dataset folder of any MegaScenes scene (the multi-scene version of build_dataset.py).

``scripts/mvgen/build_dataset.py`` builds Notre-Dame with hand-made date fixes. This script builds the
same folder layout for any scene, with generic rules only, so that ``timemachine.mvgen.data.MVGenScene``
(and therefore training, depth and look-code preparation) reads it unchanged:

* ``views.parquet``: the posed, dated, photographic views of one COLMAP model (``build_view_table``),
  of which all views dated before 2000 and at most ``--max-modern`` views from 2000 on (chosen by a
  stable hash of the file name, so a rebuild picks the same ones), with the image columns and the
  flags ``rolled``, ``duplicate_of``, ``excluded`` (``rolled``, ``duplicate``, ``bad_distortion``)
  and ``usable``. Stereo cards and images that cannot be downloaded are left out.
* ``sparse_points.npz`` and ``covisibility.npy``: the COLMAP points each view observes and the number
  of points two views share. They come from the model's ``points3D.bin`` alone (its tracks name the
  image ids), streamed from the bucket and never stored; ``images.bin`` is not needed.
* ``scene.json``: the box of the camera centres of ALL images registered in the model (from the poses
  table), which sets the scene scale, plus ``period_edges``: the years at which this scene's history
  is cut into periods (1970 by default; a scene with a known change adds its date, see
  ``KNOWN_EDGES``).

Images are cached at 1024 px under ``--image-root``/images (default: our folder on the wildnvs
project space, because the repository's disk is nearly full). Depth and look codes are added later,
on a GPU, by ``prepare_depth.py`` and ``prepare_look.py``, keyed by row, so a rebuild refuses to
renumber the rows of an existing table.

Runs in OUR environment (``.venv``):

    .venv/bin/python scripts/mvgen/build_scene.py --scene-id 209 --out-root /fs/nexus-projects/wildnvs/diffusion-time-machine/datasets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from timemachine.config import REPO_ROOT, DataPaths
from timemachine.data.cameras import colmap_to_c2w
from timemachine.data.images import download_view_images
from timemachine.data.views import build_view_table, likely_stereo_cards
from timemachine.megascenes.s3 import MegaScenesBucket
from timemachine.mvgen.data import SCENE_FILE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import (  # noqa: E402  (shared helpers of the Notre-Dame builder)
    DateFixes, camera_box, check_rows_are_kept, flag_views, load_thumbnails, shared_point_counts, write_together,
)

WILD_ROOT = Path("/fs/nexus-projects/wildnvs/diffusion-time-machine")
CENSUS = REPO_ROOT / "docs" / "census" / "census_scenes_top800.csv"
BUCKET_URL = "https://megascenes.s3.us-west-2.amazonaws.com"
MODERN_FROM = 2000.0
DEFAULT_EDGES = (1970.0,)
# Scenes with a known, dated change in their posed views get it as an extra period edge (decimal years).
KNOWN_EDGES = {
    25512: (1970.0, 2019.2877),      # Notre-Dame: the fire (built by build_dataset.py, listed for completeness)
}
# A generic DateFixes that excludes nothing by date (no fire, no lists): only rolled, duplicate, bad lens.
NO_FIXES = DateFixes(fire_start=1e9, after_fire_from=1e9 + 1)
_POINT_HEADER = struct.Struct("<Q3d3BdQ")


def stable_rank(keys: pd.Series) -> np.ndarray:
    """A rank of the file keys that does not change between runs (first 8 bytes of their SHA-1)."""
    return np.array([int.from_bytes(hashlib.sha1(k.encode()).digest()[:8], "little") for k in keys])


def select_views(full: pd.DataFrame, max_modern: int) -> pd.DataFrame:
    """All views dated before ``MODERN_FROM`` plus at most ``max_modern`` later ones (stable choice), in table order."""
    modern = full["date_mid"] >= MODERN_FROM
    keep = ~modern.to_numpy()
    if modern.sum() > max_modern:
        rank = stable_rank(full.loc[modern, "file_key"])
        chosen = full.loc[modern].index[np.argsort(rank)[:max_modern]]
        keep[full.index.get_indexer(chosen)] = True
    else:
        keep |= modern.to_numpy()
    return full[keep].reset_index(drop=True)


def parse_points3d(buf: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``points3D.bin`` -> (point ids sorted, xyz ``[P, 3]`` in that order, point index per observation, image id
    per observation). Point indices refer to the sorted order."""
    (n,) = struct.unpack_from("<Q", buf, 0)
    ids = np.empty(n, dtype=np.int64)
    xyz = np.empty((n, 3), dtype=np.float64)
    starts = np.empty(n, dtype=np.int64)
    lens = np.empty(n, dtype=np.int64)
    off = 8
    for i in range(n):
        pid, x, y, z, _, _, _, _, tl = _POINT_HEADER.unpack_from(buf, off)
        ids[i], xyz[i] = pid, (x, y, z)
        starts[i], lens[i] = off + _POINT_HEADER.size, tl
        off += _POINT_HEADER.size + 8 * tl
    if off != len(buf):
        raise ValueError(f"points3D.bin: {n} points end at byte {off}, not at the end ({len(buf)})")
    # byte offset of each observation's image id (track elements are image id, keypoint index; uint32 each)
    total = int(lens.sum())
    rel = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    byte = np.repeat(starts, lens) + 8 * rel
    img = np.empty(total, dtype=np.int64)
    for k in range(4):              # read uint32 at any byte offset through four shifted views
        sel = (byte % 4) == k
        if sel.any():
            view = np.frombuffer(buf, dtype="<u4", offset=k, count=(len(buf) - k) // 4)
            img[sel] = view[(byte[sel] - k) // 4]
    order = np.argsort(ids)
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)
    return ids[order], xyz[order], position[np.repeat(np.arange(n), lens)], img


def sparse_points(points3d: bytes, views: pd.DataFrame, poses: pd.DataFrame) -> dict[str, np.ndarray]:
    """The arrays of ``sparse_points.npz`` (as build_dataset.py writes them) for the table's views.

    ``poses`` are the model's rows of the poses table; every image id registered for a view's file (a file
    can be registered more than once) counts as that view's observations."""
    ids, xyz, point_index, image_id = parse_points3d(points3d)
    row_of_key = pd.Series(np.arange(len(views)), index=views["file_key"].to_numpy())
    registered = poses[poses["file_key"].isin(row_of_key.index)]
    row_of_image = pd.Series(row_of_key.reindex(registered["file_key"]).to_numpy(), index=registered["image_id"].to_numpy())
    row_of_image = row_of_image[~row_of_image.index.duplicated()]
    row = row_of_image.reindex(image_id).to_numpy()
    seen = ~np.isnan(row)
    row, point_index = row[seen].astype(np.int64), point_index[seen]
    pairs = np.unique(np.stack([row, ids[point_index]], axis=1), axis=0)       # one entry per (view, point)
    used = np.unique(pairs[:, 1])
    counts = np.bincount(pairs[:, 0], minlength=len(views))
    keep = np.isin(ids, used)
    return {"p3d_ids": ids[keep], "p3d_xyz": xyz[keep],
            "view_starts": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            "view_pid": pairs[:, 1].astype(np.int64), "view_names": views["path"].to_numpy().astype(str)}


def best_model(scene_id: int) -> int:
    census = pd.read_csv(CENSUS)
    row = census[census["scene_id"] == scene_id]
    if row.empty or pd.isna(row["best_model"].iloc[0]):
        raise ValueError(f"scene {scene_id} has no best model in {CENSUS}; pass --model")
    return int(row["best_model"].iloc[0])


def build(scene_id: int, model: int, out: Path, image_root: Path, max_modern: int, workers: int) -> dict:
    started = time.time()
    paths = DataPaths()
    full = build_view_table(paths, scene_id, model).reset_index(drop=True)
    views = select_views(full, max_modern)
    print(f"[{scene_id}] model {model}: {len(views)} of {len(full)} posed, dated views selected", flush=True)

    views = download_view_images(MegaScenesBucket(), DataPaths(root=image_root), views, 1024, workers)
    failed = views.attrs.pop("failed", [])
    stereo = likely_stereo_cards(views)
    views = views[~stereo].reset_index(drop=True)
    print(f"[{scene_id}] {len(views)} images on disk ({len(failed)} failed, {int(stereo.sum())} stereo cards "
          f"dropped) ({time.time() - started:.0f} s)", flush=True)

    poses = pd.read_parquet(paths.poses_table(scene_id))
    poses = poses[poses["model"] == model]
    response = requests.get(f"{BUCKET_URL}/reconstruct/{scene_id // 1000:03d}/{scene_id % 1000:03d}/colmap/{model}"
                            f"/points3D.bin", timeout=600, headers={"User-Agent": "research-data-script/0.1"})
    response.raise_for_status()
    points = sparse_points(response.content, views, poses)
    del response
    covis = shared_point_counts(points["view_starts"], points["view_pid"])
    views = flag_views(views, load_thumbnails(views["image_path"]), covis, NO_FIXES)

    centres = colmap_to_c2w(poses[["qw", "qx", "qy", "qz"]].to_numpy(), poses[["tx", "ty", "tz"]].to_numpy())[:, :3, 3]
    scene = camera_box(centres, scene_id, model)
    scene["period_edges"] = list(KNOWN_EDGES.get(scene_id, DEFAULT_EDGES))

    check_rows_are_kept(out / "views.parquet", views["file_key"].tolist())
    out.mkdir(parents=True, exist_ok=True)
    write_together({out / "sparse_points.npz": lambda path: np.savez(path, **points),
                    out / "covisibility.npy": lambda path: np.save(path, covis),
                    out / SCENE_FILE: lambda path: path.write_text(json.dumps(scene, indent=1), encoding="utf-8"),
                    out / "views.parquet": lambda path: views.to_parquet(path, index=False)})
    old = views["date_mid"] < 1970
    linked = (covis[np.ix_(old.to_numpy(), ~old.to_numpy())] >= 30).sum(1) if old.any() else np.zeros(0)
    summary = {"scene_id": scene_id, "model": model, "views": len(views), "usable": int(views["usable"].sum()),
               "old": int(old.sum()), "old_linked": int((linked >= 1).sum()),
               "1970_1999": int(((views["date_mid"] >= 1970) & (views["date_mid"] < 2000)).sum()),
               "modern": int((views["date_mid"] >= 2000).sum()), "failed": len(failed),
               "camera_extent": scene["camera_extent"], "seconds": round(time.time() - started)}
    (out / "build_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[{scene_id}] done: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-id", type=int, nargs="+", required=True)
    parser.add_argument("--model", type=int, help="COLMAP model (default: the census' best model of the scene)")
    parser.add_argument("--out-root", type=Path, default=WILD_ROOT / "datasets")
    parser.add_argument("--image-root", type=Path, default=WILD_ROOT, help="images go to <image-root>/images")
    parser.add_argument("--max-modern", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for scene_id in args.scene_id:
        model = args.model if args.model is not None else best_model(scene_id)
        out = args.out_root / f"{scene_id}_model{model}"
        if (out / "build_summary.json").exists():
            print(f"[{scene_id}] already built: {out}", flush=True)
            continue
        try:
            build(scene_id, model, out, args.image_root, args.max_modern, args.workers)
        except Exception as error:      # one bad scene must not stop the others
            print(f"[{scene_id}] FAILED: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
