"""Cleaning rules for downloaded views: shape check and near-duplicate clusters.

**Shape check.** A view's intrinsics come from its COLMAP camera, which was
calibrated on the image as it was when MegaScenes reconstructed the scene. If the
file we downloaded has a different shape (replaced on Commons, or rotated), the
intrinsics no longer describe it. Cached images are downscaled, so shapes are
compared by aspect ratio.

**Near-duplicates.** The same photograph is often on Commons under several file
names (a re-upload, another scan, a different resolution). File names differ, so
only the pixels can tell. Each image gets a 64-bit *difference hash* (dHash):
shrink to 9x8 grey pixels and record, for each of the 8x8 neighbouring pairs,
whether the left pixel is brighter than the right. Resizing, recompression and
toning (sepia vs. grey) barely change those comparisons, so copies of one photo
differ in only a few bits, while different photos differ in about half (32).
Images within ``max_distance`` bits are linked, and linked groups form a
duplicate cluster. A sample must never hold two views of one cluster, and a
train/test split must keep each cluster on one side.

Not caught: copies that are cropped or framed differently (their pixels shift).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

ASPECT_TOLERANCE = 0.015   # relative aspect-ratio difference accepted as "same shape"
HASH_SIZE = 8              # 8x8 comparisons -> 64-bit hash


def shape_status(image_width: int, image_height: int, camera_width: int, camera_height: int,
                 tolerance: float = ASPECT_TOLERANCE) -> str:
    """``match``, ``rotated`` (width and height swapped) or ``mismatch``."""
    image_aspect = image_width / image_height
    camera_aspect = camera_width / camera_height
    if abs(image_aspect / camera_aspect - 1.0) <= tolerance:
        return "match"
    if abs(image_aspect * camera_aspect - 1.0) <= tolerance:
        return "rotated"
    return "mismatch"


def dhash(image: Image.Image, size: int = HASH_SIZE) -> int:
    """Difference hash of an image as a ``size * size``-bit integer."""
    grey = np.asarray(image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (grey[:, :-1] > grey[:, 1:]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_matrix(hashes: list[int], n_bits: int = HASH_SIZE * HASH_SIZE) -> np.ndarray:
    """(N, N) number of differing bits between every pair of hashes."""
    bits = np.array([[(h >> i) & 1 for i in range(n_bits)] for h in hashes], dtype=np.int32).reshape(-1, n_bits)
    same = bits @ bits.T + (1 - bits) @ (1 - bits).T
    return n_bits - same


def duplicate_clusters(distances: np.ndarray, max_distance: int) -> np.ndarray:
    """Cluster label per item: items linked by a chain of pairs within ``max_distance``.

    Labels are the smallest item index of each cluster, so an item with no
    duplicate is its own cluster. Uses union-find over the linked pairs.
    """
    parent = np.arange(len(distances))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(*np.nonzero(np.triu(distances <= max_distance, k=1))):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(len(distances))])
