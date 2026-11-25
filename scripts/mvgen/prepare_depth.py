#!/usr/bin/env python
"""Depth maps for an MVGenMaster dataset folder: Depth-Pro, scaled to each view's COLMAP points (GPU).

MVGenMaster warps every reference view into the target cameras with a depth map, which must be in
the units of the cameras. As in its training on SfM data, and as ``nvs_test.py`` did for our first
tests (agreement with COLMAP of 1-14% on easy view sets), a view's depth is monocular depth
(Depth-Pro, given the view's focal length) multiplied by one number: the median ratio between the
depth of the view's COLMAP points and the monocular depth at their pixels.

For every view of the folder that can be loaded (an image file and a supported lens model; usable
or not, so that excluded views can serve as references in diagnostic view sets) this writes

* ``depth/<row:05d>.npy``: float16 z-depth of the view's ``--size`` crop, in COLMAP units, finite
  and positive everywhere (the run stops at a map that is not);
* one row of ``depth_stats.parquet``: ``row``, ``file_key``, ``n_points`` (COLMAP points inside the
  crop), ``scale``, ``median_rel_error`` (the median of ``|ratio / scale - 1|``: how well one number
  aligns the two) and ``ok`` (at least 50 points and an error of at most 0.15).

Views that are not ``ok`` can still be targets, but are not drawn as references
(:class:`timemachine.mvgen.data.MVGenScene`). A view with no COLMAP point in its crop gets no depth
map. The run is resumable: a view is skipped if it has a stats row and its depth map is on disk,
and the stats are saved every few views and when the run stops. If the job's time limit ends it,
submit it again.

:func:`view_depth` makes the map of one view, for scripts that need a depth map the folder lacks.

Runs in MVGenMaster's environment from its checkout:

    SCRIPT=$PWD/scripts/mvgen/prepare_depth.py ARGS="--dataset $PWD/data/datasets/25512_model0_v2" \\
        sbatch slurm/mvgen_gpu.sbatch
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from depth_pro.depth_pro import create_model_and_transforms
from timemachine.data.cameras import colmap_to_c2w
from timemachine.data.dataset import load_view
from timemachine.data.holdout import era_index
from timemachine.mvgen.data import (
    DEPTH_STATS_COLUMNS, depth_is_reliable, depth_path, period_names, read_depth_stats, sparse_depth_points,
)

sys.path.insert(0, str(Path(__file__).parent))
from nvs_test import reference_depth  # noqa: E402

SPARSE_ARRAYS = ("p3d_ids", "p3d_xyz", "view_starts", "view_pid")
STATS_DTYPES = {"row": np.int64, "file_key": object, "n_points": np.int64, "scale": np.float64,
                "median_rel_error": np.float64, "ok": bool}


def load_sparse(dataset_dir: Path) -> dict[str, np.ndarray]:
    """The arrays of ``sparse_points.npz``, read once (``np.load`` alone re-reads an array at every access)."""
    with np.load(dataset_dir / "sparse_points.npz") as archive:
        return {name: archive[name] for name in SPARSE_ARRAYS}


def loadable_views(views: pd.DataFrame) -> np.ndarray:
    """``[N]`` bool: the views ``load_view`` can load: the image file exists and the lens model is supported."""
    has_image = views["image_path"].map(lambda path: Path(path).exists()).to_numpy(dtype=bool)
    return has_image & np.isfinite(views[["k1", "k2"]].to_numpy(dtype=np.float64)).all(axis=1)


def finished_views(dataset_dir: Path, views: pd.DataFrame) -> dict[int, dict]:
    """Stats of the views an earlier run finished, by row: those with a depth map on disk, or with no points at all.

    Stats made for another view table are refused (``read_depth_stats``).
    """
    stats = read_depth_stats(dataset_dir, views["file_key"].to_numpy())
    if stats is None:
        return {}
    return {int(record["row"]): record for record in stats.to_dict("records")
            if record["n_points"] == 0 or depth_path(dataset_dir, record["row"]).exists()}


def write_stats(dataset_dir: Path, records: dict[int, dict]) -> pd.DataFrame:
    """Save the stats table sorted by row (to a temporary file, then renamed) and return it."""
    stats = pd.DataFrame([records[row] for row in sorted(records)], columns=list(DEPTH_STATS_COLUMNS))
    stats = stats.astype(STATS_DTYPES)
    path = dataset_dir / "depth_stats.parquet"
    tmp = path.with_name(path.name + ".tmp")
    stats.to_parquet(tmp, index=False)
    tmp.replace(path)
    return stats


def as_stored(depth: np.ndarray, what: str) -> np.ndarray:
    """A depth map as it is stored: float16, values beyond float16's range clipped to its largest.

    Raises if the map is not finite and positive everywhere, before or after the rounding.
    MVGenMaster's warp does not check the depth it is given: a pixel with zero depth would be
    placed at the reference's camera centre, and one that is not finite nowhere at all. ``what``
    names the view in the error.
    """
    stored = np.clip(depth, 0, np.finfo(np.float16).max).astype(np.float16)
    if not (np.isfinite(depth).all() and (stored > 0).all()):
        raise ValueError(f"The depth map of {what} is not finite and positive everywhere: "
                         f"{int((~np.isfinite(depth)).sum())} values are not finite, {int((stored <= 0).sum())} "
                         "are zero or negative.")
    return stored


def save_depth(path: Path, depth: np.ndarray) -> None:
    """Save a depth map as float16, to a temporary file that is then renamed: a killed job leaves no partial map.

    The map must be finite and positive (:func:`as_stored`).
    """
    stored = as_stored(depth, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, stored)
    tmp.replace(path)


def view_depth(views: pd.DataFrame, row: int, sparse: dict[str, np.ndarray], depth_model, transform,
               size: int = 512) -> tuple[np.ndarray | None, dict]:
    """The depth map of one view of a view table, and its row of the stats table.

    Args:
        views: the dataset folder's view table.
        sparse: the folder's COLMAP points (:func:`load_sparse`).
        depth_model, transform: Depth-Pro's (``create_model_and_transforms``).

    The image, its intrinsics and its COLMAP points are those of the ``size`` crop that training
    loads (``load_view``), and the scaling is ``reference_depth`` of ``nvs_test.py``. Returns the
    map exactly as the folder stores it (float16 ``[size, size]``, COLMAP units; None for a view
    with no COLMAP point in its crop, or whose fitted map is not finite and positive everywhere, which
    is then not ``ok``) and the stats record (``DEPTH_STATS_COLUMNS``).
    """
    view = views.iloc[row]
    c2w = colmap_to_c2w(view[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)[None],
                        view[["tx", "ty", "tz"]].to_numpy(dtype=np.float64)[None])[0]
    image, K = load_view(view, size)
    points = sparse_depth_points(sparse, row, c2w, K, size)
    record = {"row": row, "file_key": view.file_key, "n_points": len(points), "scale": np.nan,
              "median_rel_error": np.nan}
    depth = None
    if len(points):
        depth, fit = reference_depth(depth_model, transform, image, K, points)
        record.update(scale=fit["scale"], median_rel_error=fit["median_rel_error"])
        try:
            depth = as_stored(depth, f"view {row} ({view.file_key})")
        except ValueError as err:       # e.g. a fit with a negative scale (panoramas, drawings): no map, not ok
            print(f"   no depth map stored: {err}", flush=True)
            depth = None
    record["ok"] = depth is not None and bool(depth_is_reliable(record["n_points"], record["median_rel_error"]))
    return depth, record


def prepare(dataset_dir: Path, depth_model, transform, size: int = 512, save_every: int = 50) -> pd.DataFrame:
    """Write the depth map and the stats of every loadable view that lacks them; returns the whole stats table.

    ``depth_model`` and ``transform`` are Depth-Pro's (``create_model_and_transforms``); each view is
    handled by :func:`view_depth`.
    """
    views = pd.read_parquet(dataset_dir / "views.parquet")
    sparse = load_sparse(dataset_dir)
    records = finished_views(dataset_dir, views)
    loadable = loadable_views(views)
    todo = [int(row) for row in np.flatnonzero(loadable) if int(row) not in records]
    print(f"{len(todo)} views to do, {len(records)} done before, {int((~loadable).sum())} cannot be loaded (no image "
          "file or no supported lens model)", flush=True)
    try:
        for i, row in enumerate(todo, start=1):
            depth, records[row] = view_depth(views, row, sparse, depth_model, transform, size)
            if depth is not None:
                save_depth(depth_path(dataset_dir, row), depth)
            if i % save_every == 0:
                write_stats(dataset_dir, records)
                print(f"   {i} / {len(todo)}", flush=True)
    finally:
        stats = write_stats(dataset_dir, records)
    return stats


def summarise(stats: pd.DataFrame, dates: np.ndarray, period_edges: tuple[float, ...]) -> str:
    """A table per period: views, how many are ``ok``, and the spread of their median relative error.

    ``dates`` holds the date of every row of the view table; periods are separated by ``period_edges``.
    """
    period = era_index(np.asarray(dates)[stats["row"].to_numpy()], period_edges)
    lines = [f"{'period':>18} {'views':>6} {'ok':>6} {'no points':>10} {'median error':>13} {'90th pct':>9}"]
    for p, name in enumerate(period_names(period_edges)):
        sel = stats[period == p]
        error = sel["median_rel_error"].dropna().to_numpy()
        median, high = (np.median(error), np.percentile(error, 90)) if len(error) else (np.nan, np.nan)
        lines.append(f"{name:>18} {len(sel):6d} {int(sel['ok'].sum()):6d} {int((sel['n_points'] == 0).sum()):10d} "
                     f"{median:13.3f} {high:9.3f}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="dataset folder (views.parquet, sparse_points.npz)")
    p.add_argument("--size", type=int, default=512, help="crop size; must be the training resolution")
    p.add_argument("--period-edges", type=float, nargs="*", default=[1859.0, 1970.0, 2019.2877],
                   help="years that separate the periods of the summary")
    args = p.parse_args()

    depth_model, transform = create_model_and_transforms(device=torch.device("cuda"))
    depth_model.eval()
    stats = prepare(args.dataset, depth_model, transform, args.size)
    views = pd.read_parquet(args.dataset / "views.parquet")
    usable = stats[views["usable"].to_numpy(dtype=bool)[stats["row"].to_numpy()]]
    print(f"\n{int(stats['ok'].sum())} of {len(stats)} views have reliable depth; among the usable views, the ones "
          f"that can be drawn as references, {int(usable['ok'].sum())} of {len(usable)}\n"
          f"median relative error of the usable views' scaled depth against their COLMAP points, by period:\n"
          f"{summarise(usable, views['date_mid'].to_numpy(), tuple(args.period_edges))}", flush=True)


if __name__ == "__main__":
    main()
