"""The view table: every usable (posed, dated, photographic) image of one scene.

One row per unique Commons file registered in a chosen COLMAP model, with its
intrinsics, pose, date interval and type. This is the bridge between the data
pipeline tables (metadata / poses / dates) and the PyTorch dataset; poses from
other sources (e.g. VGGT-Omega for old photos) can be appended in the same
format later, as long as they are in the same coordinate frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from timemachine.analysis.census import assign_era, load_scene
from timemachine.config import DataPaths
from timemachine.data.distortion import usable_distortion
from timemachine.dating.medium import PHOTOGRAPHIC
from timemachine.megascenes.index import load_image_index

VIEW_COLUMNS = [
    "scene_id", "model", "file_key", "path", "title",
    "camera_width", "camera_height", "fx", "fy", "cx", "cy", "k1", "k2",
    "qw", "qx", "qy", "qz", "tx", "ty", "tz",
    "date_lo", "date_hi", "date_mid", "date_confidence", "era", "medium", "color_hint",
    "pose_source",
]


def build_view_table(
    paths: DataPaths, scene_id: int, model: int,
    max_date_width: float = 30.0, min_date_confidence: float = 0.3,
    include_unknown_medium: bool = True, drop_bad_distortion: bool = True,
) -> pd.DataFrame:
    """Posed, dated, photographic views of one COLMAP model of a scene.

    Filters: the date interval must be at most ``max_date_width`` years wide and
    have confidence >= ``min_date_confidence``; the image must be a photo or
    postcard (or of unknown type if ``include_unknown_medium``). Files registered
    several times (duplicate paths) keep their first entry.

    ``drop_bad_distortion``: also drop views whose lens coefficients are unusable -
    not finite (a camera model ``load_view`` cannot undo, on which it raises), or not
    invertible over the camera's own frame (COLMAP's fit for a weakly-observed
    camera; undistorting with it warps the photo). The same rule
    (:func:`~timemachine.data.distortion.usable_distortion`) the distortion migration
    applies, so a rebuilt table matches a migrated one. Their ``file_key`` values are
    in ``views.attrs["bad_distortion"]``.
    """
    poses = pd.read_parquet(paths.poses_table(scene_id))
    if "k1" not in poses:
        raise ValueError(f"{paths.poses_table(scene_id)} predates lens-distortion columns; "
                         "re-fetch it (fetch_scene_poses(..., overwrite=True))")
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
    bad = ~usable_distortion(views) if drop_bad_distortion else np.zeros(len(views), dtype=bool)
    rejected = views.loc[bad, "file_key"].tolist()
    views = views[VIEW_COLUMNS][~bad].reset_index(drop=True)
    views.attrs["bad_distortion"] = rejected
    return views


def unposed_candidates(paths: DataPaths, scene_id: int, model: int, before: float,
                       max_date_width: float = 30.0, min_date_confidence: float = 0.3) -> pd.DataFrame:
    """Dated photographic files of a scene older than ``before`` that ``model`` did not register.

    These are the images to place with a feed-forward model (e.g. VGGT-Omega).
    Includes the image ``path`` from the MegaScenes index so they can be downloaded.
    The date filters match :func:`build_view_table`.
    """
    files = load_scene(paths, scene_id)
    keep = ((files["date_mid"] < before) & (files["date_width"] <= max_date_width)
            & (files["date_confidence"] >= min_date_confidence)
            & files["photographic"] & ~files["models"].apply(lambda m: model in m))
    scene_name = pd.read_parquet(paths.scene_table).set_index("scene_id").loc[scene_id, "scene"]
    index = load_image_index(paths.images_index, columns=["cat", "image"])
    index = index[index["cat"] == scene_name].drop_duplicates("file_key")
    found = files[keep].merge(index[["file_key", "image"]].rename(columns={"image": "path"}), on="file_key")
    return found.assign(scene_id=scene_id).reset_index(drop=True)


def registered_to_views(queries: pd.DataFrame, poses: pd.DataFrame, model: int) -> pd.DataFrame:
    """View-table rows for images placed by a feed-forward model (e.g. VGGT-Omega).

    ``queries`` are the registered images (with dates and cached-image columns, as
    produced by :func:`unposed_candidates` + ``download_view_images``); ``poses``
    holds, row-aligned, COLMAP-style ``qw..tz``, ``focal`` (pixels of the cached
    image) and the confidence columns ``rot_spread_deg`` / ``center_spread``.
    Intrinsics assume a centred principal point (all VGGT-Omega predicts).
    """
    views = queries.reset_index(drop=True).copy()
    poses = poses.reset_index(drop=True)
    for col in ("qw", "qx", "qy", "qz", "tx", "ty", "tz", "rot_spread_deg", "center_spread", "n_batches"):
        views[col] = poses[col].to_numpy()
    views["model"] = model
    views["camera_width"], views["camera_height"] = views["image_width"], views["image_height"]
    views["fx"] = views["fy"] = poses["focal"].to_numpy()
    views["cx"], views["cy"] = views["image_width"] / 2, views["image_height"] / 2
    views["k1"] = views["k2"] = 0.0                        # VGGT-Omega predicts pinhole cameras
    views["era"] = assign_era(views["date_mid"], views["date_width"])
    views["pose_source"] = "vggt_omega"
    extra = [c for c in ("image_path", "image_width", "image_height", "saturation", "chroma_minor",
                         "is_monochrome", "rot_spread_deg", "center_spread", "n_batches") if c in views]
    return views[VIEW_COLUMNS + extra]


def confident_registrations(views: pd.DataFrame, max_rot_spread: float = 10.0,
                            max_center_spread: float = 0.1, min_batches: int = 2) -> pd.DataFrame:
    """Keep registered views whose pose was consistent across batches."""
    keep = ((views["n_batches"] >= min_batches) & (views["rot_spread_deg"] <= max_rot_spread)
            & (views["center_spread"] <= max_center_spread) & views["qw"].notna())
    return views[keep].reset_index(drop=True)


def likely_stereo_cards(views: pd.DataFrame, before: float = 1930.0,
                        aspect: tuple[float, float] = (1.9, 2.1)) -> pd.Series:
    """Bool mask of views that look like stereo cards: two side-by-side photos in one image.

    COLMAP registers such a card as a single pinhole camera, which describes neither
    half, and a centre crop lands on the seam. Rule: an old view (``date_mid`` before
    ``before``) with an aspect ratio within ``aspect``, or a title mentioning
    stereo/stéréo. On Notre-Dame it flags exactly the 9 stereo cards among the 14 old
    views wider than 1.6:1 (the others are panoramas at 1.75-1.85 and a 2.23 postcard).
    """
    ratio = views["image_width"] / views["image_height"]
    shaped = (views["date_mid"] < before) & ratio.between(*aspect)
    titled = views["title"].fillna("").str.contains(r"st[eé]r[eé]o", case=False, regex=True)
    return shaped | titled
