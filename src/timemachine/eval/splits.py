"""Train / validation / test splits that don't leak near-duplicate photos.

WHY THIS FILE EXISTS
--------------------
The repo has no train/validation/test split yet, and a plain random split would
quietly be wrong for this data. The same physical photo shows up as several
different files (three scans of one museum print, a burst of frames from one
tourist, the same shot uploaded twice). If one copy lands in training and another
in the test set, the model is tested on something it has effectively already seen,
and every score looks better than it should.

The existing `file_key` de-duplication only removes the SAME Commons file listed
under several paths. It cannot catch two different files that show the same
moment from the same spot. This module catches those using only what is already
in the view table: when a photo was taken and where the camera stood.

TWO IDEAS
---------
1. Near-duplicate groups. Two views are near-duplicates if their dates are very
   close AND their cameras are almost in the same place and pointing the same way.
   Near-duplicate-ness chains (A~B and B~C puts A, B, C in one group), and a whole
   group always goes to the same split.

2. Era-stratified, deterministic assignment. Old photos are scarce (Notre-Dame has
   only 113 posed pre-1970 images against 3,133 from 2000 on), so a plain random
   split could leave the test set with no old photos at all. We therefore split
   each era separately. Within an era, groups are ordered by a hash of their name
   rather than by a random number generator, so the result is the same every run
   on every machine, with no dependence on row order.

NOTE ON STABILITY
-----------------
Assignments are fixed for a given table and seed. If rows are added later (for
example VGGT-posed old photos), group sizes and era counts change, so some
assignments near the boundaries can move. Once you start reporting numbers, save
the split column (scripts/make_splits.py writes splits.parquet) and reuse it.

This module needs only numpy and pandas (no torch).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from timemachine.eval.retrieval import camera_features, pose_distance

SPLITS = ("train", "val", "test")

# Defaults. Two views closer than BOTH tolerances count as near-duplicates.
#   date: 0.1 years is about five weeks, enough for a burst or a re-scan.
#   pose: position (in scene widths) plus direction (as a fraction of 180 degrees),
#         so 0.05 is roughly "within 5% of the scene's width and a few degrees".
DATE_TOL = 0.1
POSE_TOL = 0.05


# ----------------------------------------------------------------------------
# Cameras straight from the view table (numpy only)
# ----------------------------------------------------------------------------

def view_c2ws(views: pd.DataFrame) -> np.ndarray:
    """Camera-to-world matrices (N, 4, 4) from the view table's COLMAP columns.

    COLMAP stores world-to-camera as a quaternion (qw, qx, qy, qz) and a
    translation t. The camera-to-world rotation is the transpose of the
    world-to-camera one, and the camera's position is -R^T t.

    This duplicates `colmap_to_c2w` in timemachine.data.cameras on purpose: that
    module imports torch at the top, and this one is meant to run without it.
    A test cross-checks the two whenever torch is installed.
    """
    q = views[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)
    t = views[["tx", "ty", "tz"]].to_numpy(dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rot = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)                                            # (N, 3, 3), world -> camera
    c2w = np.tile(np.eye(4), (len(views), 1, 1))
    c2w[:, :3, :3] = rot.transpose(0, 2, 1)               # transpose = camera -> world
    c2w[:, :3, 3] = -np.einsum("nij,nj->ni", rot.transpose(0, 2, 1), t)
    return c2w


# ----------------------------------------------------------------------------
# Near-duplicate groups
# ----------------------------------------------------------------------------

def near_duplicate_groups(dates: np.ndarray, c2ws: np.ndarray,
                          date_tol: float = DATE_TOL, pose_tol: float = POSE_TOL) -> np.ndarray:
    """Group id per view, so that near-duplicate views share an id.

    Returns an int array of length N. Ids are consecutive integers (0, 1, 2, ...)
    numbered in order of first appearance. Views with no date can't be compared,
    so each one is its own group.

    To avoid comparing every pair, views are sorted by date and each view is only
    compared with the ones within `date_tol` after it.
    """
    dates = np.asarray(dates, dtype=np.float64)
    n = len(dates)
    centers, directions, scale = camera_features(c2ws)

    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]     # path compression keeps the trees shallow
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)   # the smaller index becomes the root

    dated = np.flatnonzero(~np.isnan(dates))
    order = dated[np.argsort(dates[dated], kind="stable")]
    sorted_dates = dates[order]

    for pos, i in enumerate(order):
        # Everything later in date order and within the date tolerance.
        end = np.searchsorted(sorted_dates, dates[i] + date_tol, side="right")
        nearby = order[pos + 1:end]
        if nearby.size == 0:
            continue
        close = pose_distance(centers, directions, scale, int(i), nearby) <= pose_tol
        for j in nearby[close]:
            union(int(i), int(j))

    roots = np.array([find(i) for i in range(n)])
    _, first_seen_index, inverse = np.unique(roots, return_index=True, return_inverse=True)
    # np.unique numbers groups by sorted root; renumber them by first appearance.
    rank = np.argsort(np.argsort(first_seen_index))
    return rank[inverse].astype(int)


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------

def _stable_hash(text: str) -> int:
    """A hash that is the same on every machine and run (Python's own hash() is not)."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def assign_splits(views: pd.DataFrame, groups: np.ndarray, val_fraction: float = 0.1,
                  test_fraction: float = 0.1, seed: int = 0, min_groups_to_hold_out: int = 4) -> pd.Series:
    """'train' / 'val' / 'test' for every view; a whole group always shares one split.

    `views` needs a `file_key` column and, ideally, an `era` column (from the
    census). Without `era`, everything is treated as one stratum.

    Each era is split on its own so that scarce old eras still reach val and test.
    The fractions apply to GROUPS, so view counts come out approximately (not
    exactly) at those fractions when groups differ in size.

    An era with fewer than `min_groups_to_hold_out` groups stays entirely in train.
    There is nothing sensible to hold out of three photos, and an eval set of one
    group would be noise.
    """
    if not (0 <= val_fraction and 0 <= test_fraction and val_fraction + test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to less than 1")
    groups = np.asarray(groups)
    if len(groups) != len(views):
        raise ValueError("groups must have one entry per view")

    file_keys = views["file_key"].to_numpy()
    eras = views["era"].to_numpy() if "era" in views else np.full(len(views), "all", dtype=object)

    # One record per group: a stable name (its smallest file_key) and its era.
    records = []
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        era_values, counts = np.unique(eras[members].astype(str), return_counts=True)
        records.append((g, min(file_keys[members]), era_values[np.argmax(counts)]))   # majority era
    table = pd.DataFrame(records, columns=["group", "name", "era"])
    table["hash"] = [_stable_hash(f"{seed}:{name}") for name in table["name"]]

    group_split: dict[int, str] = {}
    for era, stratum in table.groupby("era"):
        ordered = stratum.sort_values("hash")["group"].tolist()
        n = len(ordered)
        if n < min_groups_to_hold_out:
            n_test = n_val = 0
        else:
            n_test = max(1, round(n * test_fraction)) if test_fraction > 0 else 0
            n_val = max(1, round(n * val_fraction)) if val_fraction > 0 else 0
            while n_test + n_val > n - 1:          # always leave at least one group to train on
                if n_val >= n_test and n_val > 0:
                    n_val -= 1
                else:
                    n_test -= 1
        for k, g in enumerate(ordered):
            group_split[g] = "test" if k < n_test else "val" if k < n_test + n_val else "train"

    return pd.Series([group_split[g] for g in groups], index=views.index, name="split")


def assert_no_leak(groups: np.ndarray, split: pd.Series) -> None:
    """Raise if any near-duplicate group is spread over more than one split."""
    per_group = pd.Series(np.asarray(split)).groupby(np.asarray(groups)).nunique()
    bad = per_group[per_group > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} near-duplicate group(s) straddle splits, e.g. group {bad.index[0]}")


def split_summary(views: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """View counts per era (rows) and split (columns), for a quick sanity check."""
    era = views["era"] if "era" in views else pd.Series("all", index=views.index)
    return pd.crosstab(era, split).reindex(columns=list(SPLITS), fill_value=0)
