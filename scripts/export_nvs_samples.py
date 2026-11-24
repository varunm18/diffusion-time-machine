#!/usr/bin/env python
"""Export fixed view sets of a scene as plain files, to test another model (e.g. MVGenMaster) on them.

Each set is 3 input views + 5 target views, written to ``<out>/<name>/``:

    view_0.png ... view_7.png     undistorted centre crops at --size (inputs first)
    sample.json                   per view: file_key, is_input, year, pixel intrinsics K (3x3, for
                                  the crop), camera-to-world c2w (4x4, OpenCV axes, COLMAP units)
    sparse_0.npy ... sparse_7.npy (N, 3) COLMAP points seen by the view: u, v (crop pixels), depth
                                  (only with --sparse-cache; used to scale monocular depth)

Sets: ``easy:N`` = the first N "easy" sets of the 2026-11-19 audit (8 mutually nearest cameras of
one modern year, same random draw, so results compare with plain SEVA's in
``data/audit_2026-11-19/gpu/plain_seva``); a probe name from --probes (e.g. ``fire``).

    python scripts/export_nvs_samples.py --dataset data/datasets/25512_model0 --sets easy:4 fire \\
        --sparse-cache data/audit_2026-11-19/data_cameras/out/colmap_cache.npz --out data/experiments/mvgen/samples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from timemachine.data.dataset import DatedViewsDataset, load_view
from timemachine.data.sampling import SamplingConfig
from timemachine.training.probes import load_probe_specs, select_probe_rows

N_INPUTS = 3


def easy_rows(ds: DatedViewsDataset, rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """As in the audit's ``audit_plain_seva.py``: 8 mutually near cameras of one modern (pre-fire) year."""
    modern = np.flatnonzero((ds.dates >= 2008) & (ds.dates < 2019.2))
    out = []
    while len(out) < n:
        a = rng.choice(modern)
        same = np.flatnonzero(np.abs(ds.dates - ds.dates[a]) <= 1.0)
        near = same[np.argsort(np.linalg.norm(ds.camera_feats[same] - ds.camera_feats[a], axis=1))[:8]]
        if len(near) == 8:
            out.append(rng.permutation(near))
    return out


def sparse_points(cache, row: int, c2w: np.ndarray, K: np.ndarray, size: int) -> np.ndarray:
    """(N, 3) u, v, depth of the COLMAP points observed by view-table row ``row``, in the crop."""
    s, e = cache["view_starts"][row], cache["view_starts"][row + 1]
    pids = cache["view_pid"][s:e]
    xyz = cache["p3d_xyz"][np.searchsorted(cache["p3d_ids"], pids)]
    w2c = np.linalg.inv(c2w)
    cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    uv = (cam / cam[:, 2:3]) @ K.T
    keep = (cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
    return np.column_stack([uv[keep, :2], cam[keep, 2]]).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--sets", nargs="+", required=True, help="easy:N and/or probe names")
    p.add_argument("--probes", type=Path, default=Path("configs/probes_notre_dame.json"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--sparse-cache", type=Path, help="npz of COLMAP points per view-table row (audit's colmap_cache.npz)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    views = pd.read_parquet(args.dataset / "views.parquet")
    ds = DatedViewsDataset(views, SamplingConfig(n_inputs=N_INPUTS, n_targets=5, max_neighbors=24), resolution=args.size)
    cache = np.load(args.sparse_cache) if args.sparse_cache else None
    specs = {s.name: s for s in load_probe_specs(args.probes)}
    sets = {}
    for name in args.sets:
        if name.startswith("easy:"):
            for i, rows in enumerate(easy_rows(ds, np.random.default_rng(0), 12)[: int(name.split(":")[1])]):
                sets[f"easy{i}"] = rows
        else:
            sets[name] = select_probe_rows(ds, specs[name])

    for name, rows in sets.items():
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        meta = []
        for j, r in enumerate(rows):
            row = views.iloc[int(r)]
            image, K = load_view(row, args.size)
            arr = ((image.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out / f"view_{j}.png")
            c2w = ds.c2ws[int(r)]
            meta.append({"file_key": row.file_key, "is_input": j < N_INPUTS, "year": float(row.date_mid),
                         "K": np.asarray(K, dtype=float).tolist(), "c2w": np.asarray(c2w, dtype=float).tolist()})
            if cache is not None:
                np.save(out / f"sparse_{j}.npy", sparse_points(cache, int(r), np.asarray(c2w), np.asarray(K), args.size))
        (out / "sample.json").write_text(json.dumps({"size": args.size, "views": meta}, indent=1))
        n_sparse = [len(np.load(out / f"sparse_{j}.npy")) for j in range(len(rows))] if cache is not None else []
        print(f"{name}: years {[round(m['year']) for m in meta]}, sparse points per view {n_sparse}", flush=True)


if __name__ == "__main__":
    main()
