"""Train/test split of dataset v1 by scene cluster.

Evaluation must use landmarks the model never trained on, so whole scenes are
held out. Two scenes can still show the same place (Dam Square and the Royal
Palace face each other) and even share Commons files, so scenes are first
grouped into **clusters**: scenes that share at least ``min_shared`` files are
linked, and linked scenes form one cluster. A cluster is never divided between
splits, so no photo can sit on both sides.

The held-out clusters are picked deterministically and spread over the range
of old-photo counts (not only tiny scenes, not only rich ones). Clusters that
hold a large share of all old photos stay in training.
"""
from __future__ import annotations

import hashlib

import pandas as pd


def scene_clusters(scene_files: dict[int, set[str]], min_shared: int = 1) -> dict[int, int]:
    """Cluster id of each scene: the smallest scene id among the scenes it is linked to."""
    parent = {sid: sid for sid in scene_files}

    def find(sid: int) -> int:
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    scenes_of_file: dict[str, list[int]] = {}
    for sid, files in scene_files.items():
        for key in files:
            scenes_of_file.setdefault(key, []).append(sid)
    shared: dict[tuple[int, int], int] = {}
    for sids in scenes_of_file.values():
        for i, a in enumerate(sids):
            for b in sids[i + 1:]:
                pair = (min(a, b), max(a, b))
                shared[pair] = shared.get(pair, 0) + 1
    for (a, b), n in shared.items():
        if n >= min_shared:
            ra, rb = find(a), find(b)
            parent[max(ra, rb)] = min(ra, rb)
    return {sid: find(sid) for sid in scene_files}


def choose_holdout(clusters: pd.DataFrame, fraction: float = 0.1, max_share: float = 0.05) -> list[int]:
    """Cluster ids to hold out, given one row per cluster with ``cluster`` and ``old_views``.

    Clusters with more than ``max_share`` of all old views are never held out.
    The rest are sorted by old views and cut into consecutive groups of
    ``round(1 / fraction)``; one cluster per full group is taken, chosen by a
    hash of its id (stable across runs, and not always the largest or smallest).
    """
    total = clusters["old_views"].sum()
    eligible = clusters[clusters["old_views"] <= max_share * total].sort_values(
        ["old_views", "cluster"], ascending=[False, True])
    step = round(1 / fraction)
    ids = eligible["cluster"].tolist()
    groups = [ids[i:i + step] for i in range(0, len(ids) - step + 1, step)]
    return sorted(min(group, key=lambda c: hashlib.sha1(str(c).encode()).hexdigest()) for group in groups)


def find_leaks(views: pd.DataFrame, column: str = "file_key") -> list[str]:
    """Values of ``column`` that occur in more than one split (``views`` has a ``split`` column)."""
    n_splits = views.groupby(column)["split"].nunique()
    return sorted(n_splits[n_splits > 1].index)
