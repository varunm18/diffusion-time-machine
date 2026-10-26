"""Download view images from the MegaScenes bucket into a local, resized cache.

Images are stored as RGB JPEGs no larger than ``max_side`` pixels under
``<data>/images/<scene folder>/``. While each image is open we also compute two
cheap color statistics as a *weak* monochrome (black-and-white or sepia) hint;
old photos are mostly monochrome, see the "photo medium" confound in
docs/findings.md. On Notre-Dame, ``chroma_minor < 1.0`` flags ~90% of 1900-1944
photos but only ~33% of (brown-toned, card-mounted) pre-1900 prints, and ~7% of
modern photos (grey stone, overcast skies). Proper labels need a classifier/VLM.
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from timemachine.config import DataPaths, safe_filename, scene_prefix
from timemachine.megascenes.s3 import MegaScenesBucket, image_key

MONOCHROME_CHROMA = 1.0  # chroma_minor below this -> probably monochrome (see module docstring)


def local_image_path(paths: DataPaths, scene_id: int, file_key: str) -> Path:
    return paths.root / "images" / scene_prefix(scene_id) / f"{safe_filename(file_key)}.jpg"


def color_stats(image: Image.Image) -> dict[str, float]:
    """Weak monochrome cues, computed on a 96x96 thumbnail.

    ``saturation``: mean HSV saturation in [0, 1].
    ``chroma_minor``: spread of the Lab (a, b) chroma along its *minor* axis. A
    toned print (grey, sepia, cyanotype) varies along one color axis only, so its
    minor-axis spread is near zero even when it is strongly tinted.
    """
    thumb = image.convert("RGB").resize((96, 96))
    hsv = np.asarray(thumb.convert("HSV"), dtype=np.float32)
    lab = np.asarray(thumb.convert("LAB"))
    # Pillow stores Lab a/b as signed bytes.
    ab = np.stack([lab[..., 1].view(np.int8), lab[..., 2].view(np.int8)], -1).reshape(-1, 2)
    minor_var = np.linalg.eigvalsh(np.cov(ab.astype(np.float64).T))[0]
    return {"saturation": float(hsv[..., 1].mean() / 255.0), "chroma_minor": float(np.sqrt(max(minor_var, 0.0)))}


def fetch_image(bucket: MegaScenesBucket, paths: DataPaths, scene_id: int, file_key: str,
                path: str, max_side: int = 1024) -> dict:
    """Download, convert to RGB, downscale and cache one image. Returns its stats."""
    dest = local_image_path(paths, scene_id, file_key)
    if dest.exists():
        image = Image.open(dest)
    else:
        image = Image.open(io.BytesIO(bucket.get(image_key(scene_id, path))))
        image = image.convert("RGB")  # handles L, P, CMYK, 16-bit, alpha
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        image.save(tmp, format="JPEG", quality=95)
        tmp.replace(dest)
    return {"file_key": file_key, "image_path": str(dest), "image_width": image.width,
            "image_height": image.height, **color_stats(image)}


def download_view_images(bucket: MegaScenesBucket, paths: DataPaths, views: pd.DataFrame,
                         max_side: int = 1024, workers: int = 8) -> pd.DataFrame:
    """Fetch all images of a view table; returns the table with image columns added.

    Adds ``image_path``, ``image_width``, ``image_height``, ``saturation``,
    ``chroma_minor`` and ``is_monochrome``. Images that fail to download or decode are dropped (and
    listed in ``attrs["failed"]``).
    """
    def task(row) -> dict | None:
        try:
            return fetch_image(bucket, paths, int(row.scene_id), row.file_key, row.path, max_side)
        except Exception as exc:  # broken/missing files are rare; skip them
            return {"file_key": row.file_key, "error": repr(exc)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(task, views.itertuples()))
    failed = [r for r in results if "error" in r]
    stats = pd.DataFrame([r for r in results if "error" not in r])
    out = views.merge(stats, on="file_key", how="inner")
    out["is_monochrome"] = out["chroma_minor"] < MONOCHROME_CHROMA
    out.attrs["failed"] = failed
    return out
