"""Project-wide settings: where data lives and which remote services we use.

Everything machine-specific (paths) or service-specific (URLs, User-Agent) is
defined here so the rest of the code never hard-codes it.

Environment variables
---------------------
TIMEMACHINE_DATA
    Root directory for downloaded and derived data. Defaults to ``<repo>/data``
    (git-ignored).
TIMEMACHINE_CONTACT
    Contact string put in the HTTP User-Agent, as Wikimedia asks automated
    clients to identify themselves. Defaults to the project repository URL.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TIMEMACHINE_DATA", REPO_ROOT / "data"))

MEGASCENES_S3_URL = "https://megascenes.s3.us-west-2.amazonaws.com"
# Per-reconstruction registered-image counts, shipped with the MegaScenes web viewer.
RECON_METADATA_URL = (
    "https://raw.githubusercontent.com/MegaScenes/web-viewer/main/public/data/recon_metadata.json"
)

_CONTACT = os.environ.get(
    "TIMEMACHINE_CONTACT", "https://github.com/varunm18/diffusion-time-machine"
)
USER_AGENT = f"DiffusionTimeMachine-research/0.1 ({_CONTACT})"


def scene_prefix(scene_id: int) -> str:
    """MegaScenes folder for a scene id: 110925 -> ``"110/925"``."""
    return f"{scene_id // 1000:03d}/{scene_id % 1000:03d}"


_MAX_FILENAME_BYTES = 180


def safe_filename(name: str) -> str:
    """A filesystem-safe, length-limited file name for an arbitrary string.

    Commons category names can contain characters some filesystems reject
    (``:``, ``"``, ``/``, ``?``...) and can be very long. Unsafe characters are
    percent-encoded; overly long results are truncated and suffixed with a hash
    so they stay unique.
    """
    safe = quote(name, safe=" ()_,.'-&!+=@~[]")
    if len(safe.encode()) > _MAX_FILENAME_BYTES:
        digest = hashlib.sha1(name.encode()).hexdigest()[:12]
        safe = safe.encode()[: _MAX_FILENAME_BYTES - 13].decode(errors="ignore") + "_" + digest
    return safe


@dataclass(frozen=True)
class DataPaths:
    """Layout of everything we store under the data root.

    ``megascenes/`` mirrors (a small part of) the S3 bucket; ``tables/`` holds
    our own derived per-scene tables as Parquet files.
    """

    root: Path = DATA_ROOT

    # --- MegaScenes mirror -------------------------------------------------
    @property
    def megascenes_metadata(self) -> Path:
        return self.root / "megascenes" / "metadata"

    @property
    def images_index(self) -> Path:
        return self.megascenes_metadata / "images_index.parquet"

    @property
    def categories(self) -> Path:
        return self.megascenes_metadata / "categories.json"

    @property
    def recon_metadata(self) -> Path:
        return self.megascenes_metadata / "recon_metadata.json"

    def raw_metadata_dir(self, scene_id: int) -> Path:
        """Cached (gzipped) ``raw_metadata.json`` files, one per subcategory."""
        return self.root / "megascenes" / "raw_metadata" / scene_prefix(scene_id)

    def reconstruction_dir(self, scene_id: int, model: int) -> Path:
        """Cached COLMAP files (cameras.bin, images.minibin) for one model."""
        return self.root / "megascenes" / "reconstruct" / scene_prefix(scene_id) / str(model)

    # --- Derived tables ----------------------------------------------------
    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def scene_table(self) -> Path:
        return self.tables / "scenes.parquet"

    def image_metadata_table(self, scene_id: int) -> Path:
        return self.tables / "image_metadata" / f"{scene_id}.parquet"

    def poses_table(self, scene_id: int) -> Path:
        return self.tables / "poses" / f"{scene_id}.parquet"

    def dates_table(self, scene_id: int) -> Path:
        return self.tables / "dates" / f"{scene_id}.parquet"

    def posing_dir(self, scene_id: int, model: int) -> Path:
        """Outputs of registering extra images (e.g. VGGT-Omega) into one COLMAP model."""
        return self.root / "posing" / f"{scene_id}_model{model}"

    # --- Training datasets -------------------------------------------------
    def dataset_dir(self, scene_id: int, model: int) -> Path:
        """View table + previews of a single-scene dataset (one COLMAP model)."""
        return self.root / "datasets" / f"{scene_id}_model{model}"
