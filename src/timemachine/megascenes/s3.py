"""Read-only HTTPS access to the public MegaScenes S3 bucket.

The bucket allows anonymous reads, so plain HTTP GETs are enough: no AWS
credentials or CLI needed. This module knows how keys are laid out in the
bucket and how to fetch them robustly (retries, atomic writes, listing with
pagination). It knows nothing about what the files contain.

Bucket layout (scene 110925 -> folder ``110/925``)::

    images/110/925/commons/<subcat>/raw_metadata.json        Commons metadata
    images/110/925/commons/<subcat>/0/pictures/<file>.jpg    the images
    reconstruct/110/925/colmap/<model>/{cameras,images,points3D}.bin
    reconstruct_aux/110/925/colmap/<model>/images.minibin    poses only
"""
from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from timemachine.config import MEGASCENES_S3_URL, USER_AGENT, scene_prefix

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


# ----------------------------------------------------------------------------
# Key builders
# ----------------------------------------------------------------------------
def raw_metadata_key(scene_id: int, subcat: str) -> str:
    """Key of the Commons metadata file for one subcategory of a scene."""
    return f"images/{scene_prefix(scene_id)}/commons/{subcat}/raw_metadata.json"


def image_key(scene_id: int, image_path: str) -> str:
    """Key of an image, given its path as stored in ``images_index.parquet``.

    The index stores paths with spaces (``commons/<subcat>/0/pictures/A B.jpg``)
    but the bucket keys use underscores, like COLMAP image names.
    """
    return f"images/{scene_prefix(scene_id)}/{image_path.replace(' ', '_')}"


def reconstruction_key(scene_id: int, model: int, filename: str) -> str:
    """Key of a COLMAP file. ``.minibin`` files live in ``reconstruct_aux/``."""
    root = "reconstruct_aux" if filename.endswith(".minibin") else "reconstruct"
    return f"{root}/{scene_prefix(scene_id)}/colmap/{model}/{filename}"


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------
@dataclass
class ListResult:
    """Objects (key, size in bytes) and sub-prefixes found under a prefix."""

    objects: list[tuple[str, int]] = field(default_factory=list)
    prefixes: list[str] = field(default_factory=list)


class MegaScenesBucket:
    """Thread-safe reader for the MegaScenes bucket.

    Each thread gets its own ``requests.Session`` (sessions are not guaranteed
    to be thread-safe), with automatic retries on throttling and server errors.
    """

    def __init__(self, base_url: str = MEGASCENES_S3_URL, retries: int = 5, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.retries = retries
        self.timeout = timeout
        self._local = threading.local()

    # -- HTTP plumbing ---------------------------------------------------------
    @property
    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            retry = Retry(
                total=self.retries,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET", "HEAD"),
            )
            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
            session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
            self._local.session = session
        return session

    def url(self, key: str) -> str:
        return f"{self.base_url}/{quote(key, safe='/')}"

    def get(self, key: str) -> bytes:
        """Fetch an object's bytes. Raises ``FileNotFoundError`` on 404."""
        response = self._session.get(self.url(key), timeout=self.timeout)
        if response.status_code in (403, 404):  # S3 answers 403 for missing keys sometimes
            raise FileNotFoundError(key)
        response.raise_for_status()
        return response.content

    def download(self, key: str, dest: Path, overwrite: bool = False) -> Path:
        """Download an object to ``dest`` atomically (no half-written files on crash)."""
        dest = Path(dest)
        if dest.exists() and not overwrite:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(self.get(key))
        tmp.replace(dest)
        return dest

    def list(self, prefix: str, delimiter: str | None = "/") -> ListResult:
        """List objects and sub-prefixes under ``prefix`` (follows pagination)."""
        result = ListResult()
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        while True:
            response = self._session.get(self.base_url + "/", params=params, timeout=self.timeout)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall("s3:Contents", _S3_NS):
                key = item.findtext("s3:Key", namespaces=_S3_NS)
                size = int(item.findtext("s3:Size", default="0", namespaces=_S3_NS))
                result.objects.append((key, size))
            for item in root.findall("s3:CommonPrefixes", _S3_NS):
                result.prefixes.append(item.findtext("s3:Prefix", namespaces=_S3_NS))
            token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
            if not token:
                return result
            params["continuation-token"] = token
