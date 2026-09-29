"""Shared helpers for the MegaScenes date census (steps 00-03)."""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import quote

import pandas as pd

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CENSUS_DATA", ROOT / "data"))
MEGASCENES = DATA / "MegaScenes"   # local mirror of s3://megascenes (metadata only)
REGISTERED = DATA / "registered"   # step 01: one CSV per scene
DATES = DATA / "dates.csv"         # step 02: Wikimedia date cache
CENSUS_TABLE = DATA / "census_table.csv"  # step 03

BUCKET = "s3://megascenes"
MIN_MODEL_IMAGES = 50  # smaller COLMAP models are fragments; ignore them


# ------------------------------------------------------------------ scenes
def load_categories() -> dict[str, int]:
    """Map scene name (Wikimedia category) -> MegaScenes scene ID."""
    path = MEGASCENES / "metadata" / "categories.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. See README: 'Download the metadata'.")
    with open(path, encoding="utf-8") as f:
        return {name: int(sid) for name, sid in json.load(f).items()}


def read_scene_list(path: str | Path) -> list[str]:
    """One scene name per line; blank lines and '#' comments ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


# ------------------------------------------------------------------ S3 via s5cmd
class S5Error(RuntimeError):
    def __init__(self, args, output):
        super().__init__(f"s5cmd {' '.join(args)} failed:\n{output.strip()}")
        self.output = output


def s5(*args: str) -> str:
    result = subprocess.run(["s5cmd", "--no-sign-request", *args],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise S5Error(args, result.stdout + result.stderr)
    return result.stdout


def scene_prefix(scene_id: int) -> str:
    """Scene 25512 lives under reconstruct/025/512."""
    return f"reconstruct/{scene_id // 1000:03d}/{scene_id % 1000:03d}"


def models_from_listing(listing: str) -> list[str]:
    """Model numbers from an `s5cmd ls` of a scene (lines end in colmap/<m>/images.bin)."""
    models = set()
    for line in listing.splitlines():
        parts = line.split()
        if parts and parts[-1].endswith("images.bin"):
            models.add(parts[-1].split("/")[-2])
    return sorted(models, key=int)


def list_models(scene_id: int) -> list[str]:
    """COLMAP models for a scene; [] if the scene was never reconstructed.

    Only ~100K of MegaScenes' ~430K scenes have reconstructions (the Eiffel
    Tower is one without). Any other s5cmd failure is raised, so a network
    error can't be mistaken for "no reconstruction".
    """
    try:
        listing = s5("ls", f"{BUCKET}/{scene_prefix(scene_id)}/*")
    except S5Error as e:
        if "no object found" in e.output.lower():
            return []
        raise
    return models_from_listing(listing)


# ------------------------------------------------------------------ COLMAP
# images.bin record: image_id (uint32), qvec w,x,y,z (4 x float64),
# tvec (3 x float64), camera_id (uint32), name (null-terminated),
# num_points2D (uint64), then 24 bytes per 2D point.
_IMAGE_HEADER = struct.Struct("<I4d3dI")


def read_colmap_images(path: str | Path) -> list[dict]:
    """Registered images in a COLMAP images.bin, with their poses.

    Poses are COLMAP's world-to-camera convention: x_cam = R(q) @ x_world + t,
    so the camera center in world coordinates is -R(q).T @ t. Poses are only
    comparable within one model (each model has its own frame and scale).
    """
    rows = []
    with open(path, "rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = \
                _IMAGE_HEADER.unpack(f.read(_IMAGE_HEADER.size))
            name = bytearray()
            while (c := f.read(1)) != b"\x00":
                if not c:
                    raise ValueError(f"{path} ended mid-record (truncated download?)")
                name += c
            (num_points,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * num_points, 1)  # skip 2D keypoints without loading them
            rows.append({"image": name.decode("utf-8"), "image_id": image_id,
                         "camera_id": camera_id, "qw": qw, "qx": qx, "qy": qy,
                         "qz": qz, "tx": tx, "ty": ty, "tz": tz})
    return rows


# ------------------------------------------------------------------ registered images
def image_name(path: str) -> str:
    """Wikimedia file title for a MegaScenes image path, NFC-normalized.

    'commons/Views_from_Tower_Bridge/0/pictures/X.jpg' -> 'X.jpg'. NFC matters
    because accented letters can be stored in two byte-different forms.
    """
    return unicodedata.normalize("NFC", path.rsplit("/", 1)[-1])


def load_registered(min_images: int = MIN_MODEL_IMAGES, dedupe: bool = True) -> pd.DataFrame:
    """All step-01 CSVs as one table, one row per (scene, model, image).

    MegaScenes downloads a file once per Wikimedia category it belongs to, and
    COLMAP registers every copy, so one photo can appear several times in a
    model under different category paths. dedupe=True keeps one row per file.
    Models with fewer than `min_images` unique images are dropped.
    """
    paths = sorted(REGISTERED.glob("*.csv"))
    if not paths:
        raise SystemExit(f"No CSVs in {REGISTERED}. Run 01_fetch_registered.py first.")
    reg = pd.concat([pd.read_csv(p, dtype={"image": str}) for p in paths],
                    ignore_index=True)
    reg["image_name"] = reg["image"].map(image_name)
    if dedupe:
        reg = reg.drop_duplicates(["scene_id", "model", "image_name"])
    size = reg.groupby(["scene_id", "model"])["image_name"].transform("nunique")
    return reg[size >= min_images].reset_index(drop=True)


# ------------------------------------------------------------------ dates
# Wikimedia often appends a hidden Wikidata div, e.g.
# 'circa 1910<div style="display: none;">date QS:P571,+1910-00-00T...</div>'
_HIDDEN = re.compile(r"<div[^>]*display:\s*none[^>]*>.*?</div>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_YEAR = re.compile(r"(?<!\d)(1[89]\d\d|20[0-2]\d)(?!\d)")


def clean_date_text(raw: str) -> str:
    return " ".join(_TAG.sub(" ", _HIDDEN.sub(" ", raw)).split())


def parse_date(raw) -> tuple[int | None, int | None, str]:
    """(earliest year, latest year, flag) from a Wikimedia date string.

    Flags mark dates to treat with care:
      upload   'original upload date': an upload date, not a capture date
      bound    'before 19 June 2013' / 'after ...': only a limit
      range    several years mentioned ('between 1855 and 1900')
      circa    'circa 1910', 'ca. 1930', 'c. 1905'
      decade   '1850s'
      missing  no date at all;  unparsed  text with no recognizable year
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, None, "missing"
    text = clean_date_text(raw)
    low = text.lower()
    years = [int(y) for y in _YEAR.findall(text)]
    if not years:
        return None, None, "unparsed"
    if "upload" in low:
        flag = "upload"
    elif re.search(r"\b(before|after)\b", low):
        flag = "bound"
    elif len(set(years)) > 1:
        flag = "range"
    elif re.search(r"\bcirca\b|\bca?\.\s*\d", low):
        flag = "circa"
    elif re.search(r"\d0s\b", low):
        flag = "decade"
    else:
        flag = ""
    return min(years), max(years), flag


def commons_url(name: str) -> str:
    return ("https://commons.wikimedia.org/wiki/File:"
            + quote(name.replace(" ", "_"), safe="()_,.'-!&"))
