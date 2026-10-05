"""Registered images and camera poses of a scene, as one flat table.

For each COLMAP model of a scene we download ``cameras.bin`` and
``images.minibin`` (falling back to the full ``images.bin`` if the minibin is
missing), cache them locally, and flatten them into one row per registered
image *entry*. A Commons file registered under two paths gives two rows with
the same ``file_key``: deduplicate downstream when counting photos.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from timemachine.config import DataPaths
from timemachine.megascenes import colmap
from timemachine.megascenes.index import file_key
from timemachine.megascenes.s3 import MegaScenesBucket, reconstruction_key

POSE_COLUMNS = [
    "scene_id", "model", "image_id", "path", "subcat", "file_key",
    "camera_model", "width", "height", "fx", "fy", "cx", "cy",
    "qw", "qx", "qy", "qz", "tx", "ty", "tz",
]


def _fetch(bucket: MegaScenesBucket, paths: DataPaths, scene_id: int, model: int, filename: str) -> bytes:
    dest = paths.reconstruction_dir(scene_id, model) / filename
    return bucket.download(reconstruction_key(scene_id, model, filename), dest).read_bytes()


def load_model(bucket: MegaScenesBucket, paths: DataPaths, scene_id: int, model: int) -> pd.DataFrame:
    """Registered images of one COLMAP model with intrinsics and poses."""
    cameras = {c.camera_id: c for c in colmap.read_cameras_bin(
        _fetch(bucket, paths, scene_id, model, "cameras.bin"))}
    try:
        images = colmap.read_images_minibin(_fetch(bucket, paths, scene_id, model, "images.minibin"))
    except FileNotFoundError:
        images = colmap.read_images_bin(_fetch(bucket, paths, scene_id, model, "images.bin"))

    rows = []
    for im in images:
        cam = cameras[im.camera_id]
        fx, fy, cx, cy = cam.focal_and_center()
        rows.append((
            scene_id, model, im.image_id, im.name, im.name.split("/")[1], file_key(im.name),
            cam.model, cam.width, cam.height, fx, fy, cx, cy, *im.qvec, *im.tvec,
        ))
    return pd.DataFrame(rows, columns=POSE_COLUMNS)


def fetch_scene_poses(
    bucket: MegaScenesBucket, paths: DataPaths, scene_id: int, models: Iterable[int],
    overwrite: bool = False,
) -> pd.DataFrame:
    """All models of a scene in one table, cached at ``paths.poses_table(scene_id)``.

    Models listed in ``recon_metadata.json`` but missing from the bucket are
    skipped and reported in ``attrs["missing_models"]``.
    """
    out = paths.poses_table(scene_id)
    if out.exists() and not overwrite:
        return pd.read_parquet(out)
    frames, missing = [], []
    for m in models:
        try:
            frames.append(load_model(bucket, paths, scene_id, m))
        except FileNotFoundError:
            missing.append(m)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=POSE_COLUMNS)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    table.attrs["missing_models"] = missing
    return table
