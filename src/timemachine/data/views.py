"""The view table: every usable (posed, dated, photographic) image of one scene.

One row per unique Commons file registered in a chosen COLMAP model, with its
intrinsics, pose, date interval and type. This is the bridge between the data
pipeline tables (metadata / poses / dates) and the PyTorch dataset; poses from
other sources (e.g. VGGT-Omega for old photos) can be appended in the same
format later, as long as they are in the same coordinate frame.
"""
from __future__ import annotations

import pandas as pd

from timemachine.analysis.census import assign_era
from timemachine.config import DataPaths
from timemachine.dating.medium import PHOTOGRAPHIC

VIEW_COLUMNS = [
    "scene_id", "model", "file_key", "path", "title",
    "camera_width", "camera_height", "fx", "fy", "cx", "cy",
    "qw", "qx", "qy", "qz", "tx", "ty", "tz",
    "date_lo", "date_hi", "date_mid", "date_confidence", "era", "medium", "color_hint",
    "pose_source",
]


def build_view_table(
    paths: DataPaths, scene_id: int, model: int,
    max_date_width: float = 30.0, min_date_confidence: float = 0.3,
    include_unknown_medium: bool = True,
) -> pd.DataFrame:
    """Posed, dated, photographic views of one COLMAP model of a scene.

    Filters: the date interval must be at most ``max_date_width`` years wide and
    have confidence >= ``min_date_confidence``; the image must be a photo or
    postcard (or of unknown type if ``include_unknown_medium``). Files registered
    several times (duplicate paths) keep their first entry.
    """
    poses = pd.read_parquet(paths.poses_table(scene_id))
    poses = poses[poses["model"] == model].drop_duplicates("file_key")
    dates = pd.read_parquet(paths.dates_table(scene_id))
    meta = pd.read_parquet(paths.image_metadata_table(scene_id), columns=["file_key", "title"])

    views = poses.merge(dates.drop(columns=["scene_id"], errors="ignore"), on="file_key", how="inner")
    views = views.merge(meta, on="file_key", how="left")
    allowed = set(PHOTOGRAPHIC) | ({"unknown"} if include_unknown_medium else set())
    keep = (
        views["date_lo"].notna()
        & (views["date_width"] <= max_date_width)
        & (views["date_confidence"] >= min_date_confidence)
        & views["medium"].isin(allowed)
    )
    views = views[keep].copy()
    views["era"] = assign_era(views["date_mid"], views["date_width"])
    views = views.rename(columns={"width": "camera_width", "height": "camera_height"})
    views["pose_source"] = "megascenes_colmap"
    return views[VIEW_COLUMNS].reset_index(drop=True)
