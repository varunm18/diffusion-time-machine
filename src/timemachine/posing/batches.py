"""Grouping images to pose into VGGT-Omega batches, each with diverse anchors.

Every batch holds some *query* images (to be posed) plus *anchor* images with
known COLMAP poses. Anchors are chosen by farthest-point sampling over camera
positions and viewing directions, so each batch covers the scene from many
sides; a query then has a good chance of overlapping some anchor. Each query
appears in several batches with different anchors, so the spread of its
predicted poses can serve as a confidence measure.
"""
from __future__ import annotations

import numpy as np


def farthest_point_sample(features: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` indices spread out over ``features`` (N, D), starting from a random point."""
    n = len(features)
    if k >= n:
        return rng.permutation(n)
    chosen = [int(rng.integers(n))]
    dist = np.linalg.norm(features - features[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(features - features[nxt], axis=1))
    return np.array(chosen)


def anchor_features(c2ws: np.ndarray) -> np.ndarray:
    """Per-camera feature for diversity: normalized position + viewing direction."""
    centers = c2ws[:, :3, 3]
    spread = np.median(np.linalg.norm(centers - np.median(centers, 0), axis=1)) or 1.0
    return np.concatenate([(centers - np.median(centers, 0)) / spread, c2ws[:, :3, 2]], axis=1)


def make_batches(
    n_queries: int, anchor_c2ws: np.ndarray, rng: np.random.Generator,
    queries_per_batch: int = 32, anchors_per_batch: int = 32, repeats: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Batches of ``(query_indices, anchor_indices)``; each query appears ``repeats`` times."""
    features = anchor_features(anchor_c2ws)
    batches = []
    for _ in range(repeats):
        order = rng.permutation(n_queries)
        for start in range(0, n_queries, queries_per_batch):
            anchors = farthest_point_sample(features, anchors_per_batch, rng)
            batches.append((order[start:start + queries_per_batch], anchors))
    return batches
