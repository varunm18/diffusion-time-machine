#!/usr/bin/env python
"""Assign train / val / test splits to a single-scene dataset, without leaking near-duplicates.

Reads ``views.parquet`` from a dataset folder (as written by build_scene_dataset.py),
groups near-duplicate views (same moment, same camera position), splits each era
separately, and writes ``splits.parquet`` next to the views:

    file_key, group_id, split

It prints how many views of each era ended up in each split, and how big the
near-duplicate groups are, so you can judge whether the tolerances are sensible.

    python scripts/make_splits.py --dataset data/datasets/25512_model0

Save and reuse the resulting splits.parquet once you start reporting numbers: if rows
are added to the view table later, a fresh run can move a few assignments (see the
note in src/timemachine/eval/splits.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from timemachine.eval.splits import (
    DATE_TOL, POSE_TOL, assert_no_leak, assign_splits, near_duplicate_groups, split_summary, view_c2ws,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="folder with views.parquet")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date-tol", type=float, default=DATE_TOL, help="years (default %(default)s)")
    parser.add_argument("--pose-tol", type=float, default=POSE_TOL, help="camera distance (default %(default)s)")
    parser.add_argument("--out", type=Path, help="output file (default: <dataset>/splits.parquet)")
    args = parser.parse_args()

    views = pd.read_parquet(args.dataset / "views.parquet")
    groups = near_duplicate_groups(views["date_mid"].to_numpy(), view_c2ws(views), args.date_tol, args.pose_tol)
    sizes = np.bincount(groups)
    print(f"{len(views)} views -> {len(sizes)} near-duplicate groups "
          f"({int((sizes > 1).sum())} groups of 2+ views, largest {sizes.max()})")

    split = assign_splits(views, groups, args.val_fraction, args.test_fraction, args.seed)
    assert_no_leak(groups, split)            # a group must never straddle two splits

    print("\nviews per era and split:")
    print(split_summary(views, split).to_string())

    out = args.out or args.dataset / "splits.parquet"
    pd.DataFrame({"file_key": views["file_key"], "group_id": groups, "split": split}).to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
