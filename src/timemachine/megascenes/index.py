"""Global MegaScenes index files and the per-scene summary table built from them.

Three small files describe the whole dataset without downloading any images:

``images_index.parquet``  one row per image *path* (8.7M rows): scene, subcategory,
                          path, size, license. No dates.
``categories.json``       scene name (a Commons category) -> integer scene id.
``recon_metadata.json``   from the MegaScenes web viewer: per scene, the number of
                          registered images and 3D points of each COLMAP model.

Duplicates: MegaScenes stores a Commons file once per subcategory it belongs to,
so one file can appear under several paths (7.6% of rows overall, 30-50% in some
big scenes). :func:`file_key` gives the canonical name used to deduplicate and to
join paths, COLMAP names and Commons titles.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pandas as pd


def file_key(name: str) -> str:
    """Canonical key for a Commons file, usable across all MegaScenes sources.

    Accepts a bare file name, a MegaScenes/COLMAP path, or a Commons title, and
    normalizes the differences between them:

    >>> file_key("commons/Some_subcat/0/pictures/A_B.jpg")
    'A_B.jpg'
    >>> file_key("File:A B.jpg")
    'A_B.jpg'

    Unicode is NFC-normalized because the same accented letter can be stored in
    two byte-different forms.
    """
    name = name.rsplit("/", 1)[-1]
    if name.startswith("File:"):
        name = name[len("File:"):]
    return unicodedata.normalize("NFC", name).replace(" ", "_")


def load_image_index(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load ``images_index.parquet`` and add a ``file_key`` column.

    Columns in the file: cat (scene name), subcat, image (path), image_name,
    width, height, license_id, license_url, license_short_name, usage_terms.
    """
    wanted = None if columns is None else sorted(set(columns) | {"image_name"})
    df = pd.read_parquet(path, columns=wanted)
    df["file_key"] = df["image_name"].map(file_key)
    return df


def load_categories(path: Path) -> dict[str, int]:
    """Scene name -> MegaScenes scene id."""
    with open(path, encoding="utf-8") as f:
        return {name: int(scene_id) for name, scene_id in json.load(f).items()}


def load_recon_metadata(path: Path) -> pd.DataFrame:
    """Per-model reconstruction sizes, one row per (scene, model).

    The web viewer stores each scene as a flat list
    ``[name, n_models, n_images_0, n_points_0, n_images_1, n_points_1, ...]``,
    with models numbered in order (model 0 is not necessarily the largest).
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for scene_id, entry in raw.items():
        name, n_models, sizes = entry[0], int(entry[1]), entry[2:]
        if len(sizes) != 2 * n_models:
            raise ValueError(f"Malformed recon_metadata entry for scene {scene_id}: {entry[:3]}...")
        for model in range(n_models):
            rows.append((int(scene_id), name, model, int(sizes[2 * model]), int(sizes[2 * model + 1])))
    return pd.DataFrame(rows, columns=["scene_id", "scene", "model", "n_images", "n_points"])


def select_scenes(
    scenes: pd.DataFrame,
    names: list[str] | None = None,
    top: int | None = None,
    min_model_images: int = 0,
) -> pd.DataFrame:
    """Pick scenes from the scene table.

    ``names`` selects scenes by name (Commons category) or numeric id; otherwise the
    ``top`` scenes with the largest reconstructions are taken, keeping only those
    whose largest model has at least ``min_model_images`` registered entries.
    """
    if names:
        ids = {int(n) for n in names if str(n).isdigit()}
        picked = scenes[scenes["scene"].isin(names) | scenes["scene_id"].isin(ids)]
        missing = set(names) - set(picked["scene"]) - {str(i) for i in picked["scene_id"]}
        if missing:
            raise KeyError(f"Unknown scenes: {sorted(missing)}")
        return picked
    picked = scenes[scenes["largest_model_images"] >= min_model_images]
    picked = picked.sort_values("largest_model_images", ascending=False)
    return picked.head(top) if top else picked


def build_scene_table(
    index: pd.DataFrame, categories: dict[str, int], recon: pd.DataFrame
) -> pd.DataFrame:
    """One row per scene with image counts (raw and deduplicated) and model sizes.

    Note: ``recon`` image counts include duplicate copies of a file, so
    ``largest_model_images`` overstates the number of distinct posed photos.
    """
    per_scene = index.groupby("cat").agg(
        n_paths=("file_key", "size"),
        n_files=("file_key", "nunique"),
        n_subcats=("subcat", "nunique"),
    )
    per_scene.index.name = "scene"
    per_scene = per_scene.reset_index()
    per_scene["scene_id"] = per_scene["scene"].map(categories)

    models = recon.groupby("scene_id").agg(
        n_models=("model", "size"),
        largest_model=("n_images", "idxmax"),
        largest_model_images=("n_images", "max"),
        registered_paths=("n_images", "sum"),
    )
    # idxmax returns a row label of `recon`; translate it into a model number.
    models["largest_model"] = recon.loc[models["largest_model"], "model"].to_numpy()

    table = per_scene.merge(models, how="left", left_on="scene_id", right_index=True)
    for col in ("n_models", "largest_model_images", "registered_paths"):
        table[col] = table[col].fillna(0).astype(int)
    table["largest_model"] = table["largest_model"].astype("Int64")  # NA when unreconstructed
    table["scene_id"] = table["scene_id"].astype("Int64")
    cols = ["scene_id", "scene", "n_paths", "n_files", "n_subcats",
            "n_models", "largest_model", "largest_model_images", "registered_paths"]
    return table[cols].sort_values("n_files", ascending=False).reset_index(drop=True)
