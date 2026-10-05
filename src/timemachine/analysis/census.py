"""Scene census: how many dated images, from which eras, and how many are posed.

Joins the per-scene tables produced by the pipeline (metadata, dates, poses)
into one row per unique Commons file, then counts files per era for each scene
and each COLMAP model. Counting is always by unique file (``file_key``), never
by path, because MegaScenes stores many files under several paths.

Era assignment uses the middle of the date interval, and only when the interval
is reasonably tight (``MAX_ERA_WIDTH``): "1900-1930" counts as 1900-1944, while
"before 1945" (1826-1945) counts as undated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from timemachine.config import DataPaths
from timemachine.dating.medium import PHOTOGRAPHIC

# (label, first year inclusive, last year exclusive)
ERAS: tuple[tuple[str, float, float], ...] = (
    ("pre-1900", -np.inf, 1900),
    ("1900-1944", 1900, 1945),
    ("1945-1969", 1945, 1970),
    ("1970-1999", 1970, 2000),
    ("2000+", 2000, np.inf),
)
ERA_LABELS = [label for label, _, _ in ERAS] + ["undated"]
OLD_ERAS = ("pre-1900", "1900-1944", "1945-1969")   # = pre-1970
MODERN_ERAS = ("2000+",)
MAX_ERA_WIDTH = 30.0  # years; wider intervals are too vague to place in an era
MIN_MODEL_FILES = 50  # smaller COLMAP models are fragments


def assign_era(date_mid: pd.Series, date_width: pd.Series) -> pd.Series:
    """Era label per image (``undated`` when missing or too vague)."""
    era = pd.Series("undated", index=date_mid.index, dtype=object)
    placeable = date_mid.notna() & (date_width <= MAX_ERA_WIDTH)
    for label, lo, hi in ERAS:
        era[placeable & (date_mid >= lo) & (date_mid < hi)] = label
    return era


def load_scene(paths: DataPaths, scene_id: int) -> pd.DataFrame:
    """One row per unique file of a scene: metadata essentials + date + registration.

    Registration columns: ``models`` (sorted list of COLMAP models the file is
    registered in) and ``registered`` (bool).
    """
    meta = pd.read_parquet(paths.image_metadata_table(scene_id),
                           columns=["scene_id", "file_key", "title", "width", "height", "license"])
    dates = pd.read_parquet(paths.dates_table(scene_id)).drop(columns=["scene_id"], errors="ignore")
    files = meta.merge(dates, on="file_key", how="left")

    poses_path = paths.poses_table(scene_id)
    if poses_path.exists():
        poses = pd.read_parquet(poses_path, columns=["file_key", "model"])
        models = poses.groupby("file_key")["model"].agg(lambda m: sorted(set(m)))
        files["models"] = files["file_key"].map(models)
    else:
        files["models"] = None
    files["models"] = files["models"].apply(lambda m: m if isinstance(m, list) else [])
    files["registered"] = files["models"].str.len() > 0
    files["era"] = assign_era(files["date_mid"], files["date_width"])
    files["photographic"] = files["medium"].isin(PHOTOGRAPHIC) | (files["medium"] == "unknown")
    return files


def era_counts(files: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Counts of unique files per era, grouped by ``by`` (columns = ERA_LABELS)."""
    counts = files.groupby([*by, "era"])["file_key"].nunique().unstack("era", fill_value=0)
    counts = counts.reindex(columns=ERA_LABELS, fill_value=0)
    counts["pre-1970"] = counts[list(OLD_ERAS)].sum(axis=1)
    return counts


def model_census(files: pd.DataFrame, scene_name: str = "") -> pd.DataFrame:
    """Per-model era counts of *photographic* registered files (models >= MIN_MODEL_FILES)."""
    reg = files[files["photographic"]].explode("models").dropna(subset=["models"])
    if reg.empty:
        return pd.DataFrame()
    reg["model"] = reg["models"].astype(int)
    sizes = files.explode("models").dropna(subset=["models"]).groupby("models")["file_key"].nunique()
    counts = era_counts(reg, ["scene_id", "model"])
    counts["model_files"] = [sizes.get(m, 0) for _, m in counts.index]
    counts = counts[counts["model_files"] >= MIN_MODEL_FILES]
    counts.insert(0, "scene", scene_name)
    return counts.reset_index()


def scene_summary(files: pd.DataFrame, scene_name: str = "") -> dict:
    """Headline numbers for one scene (all files, photographic files, posed files)."""
    photo = files[files["photographic"]]
    eras_all = files["era"].value_counts()
    eras_photo = photo["era"].value_counts()
    eras_posed = photo[photo["registered"]]["era"].value_counts()
    models = model_census(files, scene_name)
    best = models.sort_values(["pre-1970", "2000+"], ascending=False).head(1) if len(models) else models
    summary = {
        "scene_id": int(files["scene_id"].iloc[0]),
        "scene": scene_name,
        "files": len(files),
        "photographic": len(photo),
        "dated": int((files["era"] != "undated").sum()),
        "registered": int(files["registered"].sum()),
    }
    for label in ERA_LABELS:
        summary[f"all_{label}"] = int(eras_all.get(label, 0))
        summary[f"photo_{label}"] = int(eras_photo.get(label, 0))
        summary[f"posed_{label}"] = int(eras_posed.get(label, 0))
    summary["photo_pre-1970"] = sum(summary[f"photo_{e}"] for e in OLD_ERAS)
    summary["posed_pre-1970"] = sum(summary[f"posed_{e}"] for e in OLD_ERAS)
    summary["best_model"] = int(best["model"].iloc[0]) if len(best) else None
    summary["best_model_pre-1970"] = int(best["pre-1970"].iloc[0]) if len(best) else 0
    summary["best_model_2000+"] = int(best["2000+"].iloc[0]) if len(best) else 0
    return summary


def passes_gate(summary: pd.DataFrame, n_old: int = 10, n_modern: int = 10) -> pd.DataFrame:
    """The proposal's go/no-go test, counted two ways.

    ``gate_posed``: one COLMAP model already holds >= n_old pre-1970 and >= n_modern
    2000+ photographic files (usable today, same coordinate frame).
    ``gate_any``: the scene has that many dated photographic files at all (usable
    once we pose old images ourselves, e.g. with VGGT-Omega).
    """
    out = summary.copy()
    out["best_model"] = out["best_model"].astype("Int64")
    out["gate_posed"] = (out["best_model_pre-1970"] >= n_old) & (out["best_model_2000+"] >= n_modern)
    out["gate_any"] = (out["photo_pre-1970"] >= n_old) & (out["photo_2000+"] >= n_modern)
    return out
