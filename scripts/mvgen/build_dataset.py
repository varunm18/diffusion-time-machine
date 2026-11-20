#!/usr/bin/env python
"""Build the Notre-Dame dataset for teaching MVGenMaster dates ("dataset v2").

The first table (``data/datasets/25512_model0``, 925 views) kept only 800 of the modern views
and nothing that says which views see the same part of the building. The 2026-11-19 audit found
the consequences: 24-74% of training targets were not visible from their inputs, 7% of the views
are sideways, and 18 pairs of views are the same photo twice. This script builds the table the
MVGenMaster work samples from (``timemachine.mvgen.data``), with what is needed to avoid all three.

It writes, to ``data/datasets/25512_model0_v2/``:

* ``views.parquet``: the columns of the first table (so ``timemachine.data.dataset.load_view``
  works) plus

  - ``rolled`` (bool): the camera is rotated more than 45 degrees about its viewing axis;
  - ``duplicate_of`` (int): the row of the first copy of a duplicated photo, or -1;
  - ``excluded`` (str): why the view must not be used, or ``""``. One of ``EXCLUSIONS``, the first
    that applies: ``"misdated"`` (listed in the date-fixes file, or a copy of a photo listed
    there), ``"montage"`` (one image that shows two states of the building), ``"fire_day"`` (dated
    on the day of the fire, so it shows the fire itself), ``"uncertain_date"`` (dated after the
    fire, but with a date too uncertain to trust there), ``"before_spire"`` (dated before the
    earliest state we model), ``"rolled"``, ``"duplicate"`` (a later copy),
    ``"bad_distortion"`` (no finite ``k1``/``k2``);
  - ``usable`` (bool): ``excluded == ""``.

  Views: everything dated before 1970, everything dated from the start of the fire on, and the
  views of the first table. Views whose image cannot be downloaded and stereo cards are left out.
* ``sparse_points.npz``: COLMAP's 3D points and the ids of the points each view observes
  (``p3d_ids`` sorted, ``p3d_xyz``, ``view_starts``, ``view_pid``; plus ``view_names``, the COLMAP
  image name of each row, which ties the file to the table).
* ``covisibility.npy``: uint16 ``[N, N]``, the number of 3D points two views share.
* ``scene.json``: the box of the camera centres of ALL images registered in the COLMAP model (not
  only the table's views) and its longest side, ``camera_extent``. MVGenMaster's training scaled a
  scene once, by this extent, and ``timemachine.mvgen.data.MVGenScene`` does the same.

Depth maps are added later, on a GPU, by ``scripts/mvgen/prepare_depth.py``. They are stored by
row of the table, so a rebuild must not renumber the rows: it refuses to replace a table whose
rows would get another view, unless ``--renumber`` is given (delete the depth maps then). Changing
flags or the date-fixes file never renumbers; a view whose image could not be downloaded before
and can be now may.

What is known about the scene's dates and photos comes from ``configs/notre_dame_date_fixes.json``
(:class:`DateFixes`): the dates of the fire, photos that are misdated or montages, the rule for
uncertain after-fire dates, the earliest usable date, and same-photograph pairs found by hand.

Runs in OUR environment (``.venv``, not MVGenMaster's), because it needs the MegaScenes download
code. A rerun is cheap: images already on disk are not downloaded again, and the 867 MB
``images.bin`` is not read again while ``sparse_points.npz`` holds every view of the table and
``scene.json`` exists (delete a file to force a new parse). The four files are written together
at the end, each under a temporary name that is then renamed, so an interrupted run leaves the
folder as it was and the folder is never without a view table.

    .venv/bin/python scripts/mvgen/build_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

from timemachine.config import REPO_ROOT, DataPaths
from timemachine.data.cameras import colmap_to_c2w
from timemachine.data.holdout import era_index
from timemachine.data.images import download_view_images
from timemachine.data.views import build_view_table, likely_stereo_cards
from timemachine.megascenes.s3 import MegaScenesBucket
from timemachine.mvgen.data import SCENE_FILE, period_names

SCENE_ID, MODEL = 25512, 0                 # Cathédrale Notre-Dame de Paris, its largest COLMAP model
OLD_BEFORE = 1970.0                        # every view dated before this is kept
SPIRE_BUILT = 1859.0                       # the summary counts the few views from before the spire separately
DATE_FIXES = REPO_ROOT / "configs" / "notre_dame_date_fixes.json"
EXCLUSIONS = ("misdated", "montage", "fire_day", "uncertain_date", "before_spire", "rolled", "duplicate",
              "bad_distortion")            # in order of precedence

MAX_ROLL_DEG = 45.0                        # beyond this a view counts as sideways (or upside down)
# Signs that two views show the same picture (two of the three make a duplicate, see duplicate_pairs):
THUMB_SIZE = 32                            # side of the grey thumbnails that are compared
MIN_THUMB_CORR = 0.9                       # 1. the thumbnails correlate above this
MAX_CENTRE_DIST = 0.05                     # 2. the cameras are this close, in units of the scene's spread,
MAX_ROTATION_DEG = 5.0                     # ... and their orientations differ by less than this
MIN_POINT_OVERLAP = 0.6                    # 3. of the 3D points either view observes, both observe this share
MIN_SHARED = 30                            # two views are co-visible when they share this many 3D points

_UINT64 = struct.Struct("<Q")
_IMAGE_HEADER = struct.Struct("<I4d3dI")   # image id, qw qx qy qz, tx ty tz, camera id
_POINT2D = np.dtype([("x", "<f8"), ("y", "<f8"), ("point3d_id", "<i8")])
_POINT3D_HEADER = struct.Struct("<Q3d3BdQ")  # id, x y z, r g b, error, track length
_TRACK_ELEMENT_BYTES = 8                   # image id, keypoint index (uint32 each)


# ----------------------------------------------------------------------------
# Which views
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DateFixes:
    """What ``configs/notre_dame_date_fixes.json`` says about the scene's dates (decimal years) and photos.

    Photos are named by their ``file_key``.

    * Views dated in ``[fire_start, after_fire_from)`` show the fire itself; views dated from
      ``after_fire_from`` on show the burnt building.
    * ``misdated``: photos whose recorded date contradicts what they show.
    * ``montages``: images that show two states of the building at once (before/after montages).
    * ``after_fire_min_confidence``: a view dated from ``after_fire_from`` on is only trusted with a
      ``date_confidence`` of at least this, or if it is in ``after_fire_verified`` (checked by eye).
      Upload and scan dates often pass for capture dates, and after the fire that puts the intact
      building on the wrong side. None: no such rule.
    * ``exclude_before``: views dated before this are not used (None: no limit). For Notre-Dame it is
      the year the spire was built: the few earlier views are too few, and too doubtful, to be a state.
    * ``duplicates``: pairs of photos that are the same photograph but that :func:`duplicate_pairs`
      does not recognise (found by eye).
    """

    fire_start: float
    after_fire_from: float
    misdated: frozenset[str] = frozenset()
    montages: frozenset[str] = frozenset()
    after_fire_min_confidence: float | None = None
    after_fire_verified: frozenset[str] = frozenset()
    exclude_before: float | None = None
    duplicates: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if not self.fire_start < self.after_fire_from:
            raise ValueError("fire_start must be before after_fire_from.")
        if self.after_fire_min_confidence is not None and not 0.0 < self.after_fire_min_confidence <= 1.0:
            raise ValueError("after_fire_min_confidence must be in (0, 1].")
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in self.duplicates):
            raise ValueError("duplicates must be pairs of two different file keys.")

    @property
    def listed_keys(self) -> frozenset[str]:
        """Every file key the fixes name, in any list."""
        return (self.misdated | self.montages | self.after_fire_verified
                | frozenset(key for pair in self.duplicates for key in pair))


def load_date_fixes(path: Path, scene_id: int = SCENE_ID) -> DateFixes:
    """Read a date-fixes file, refusing one written for another scene.

    Only the scene and the two dates of the fire are required. ``misdated`` and ``montages`` map file
    keys to a note for the reader; ``after_fire_verified`` is a list of file keys and ``duplicates`` a
    list of pairs of them.
    """
    fixes = json.loads(Path(path).read_text(encoding="utf-8"))
    if fixes["scene_id"] != scene_id:
        raise ValueError(f"{path} is for scene {fixes['scene_id']}, not {scene_id}")
    min_confidence, exclude_before = fixes.get("after_fire_min_confidence"), fixes.get("exclude_before")
    return DateFixes(
        fire_start=float(fixes["fire_start"]), after_fire_from=float(fixes["after_fire_from"]),
        misdated=frozenset(fixes.get("misdated", ())), montages=frozenset(fixes.get("montages", ())),
        after_fire_min_confidence=None if min_confidence is None else float(min_confidence),
        after_fire_verified=frozenset(fixes.get("after_fire_verified", ())),
        exclude_before=None if exclude_before is None else float(exclude_before),
        duplicates=tuple(tuple(pair) for pair in fixes.get("duplicates", ())))


def select_views(full: pd.DataFrame, base_keys: Iterable[str], old_before: float, event_start: float) -> pd.DataFrame:
    """Rows of the full view table to keep, in the table's order.

    Kept: every view dated (``date_mid``) before ``old_before``, every view dated on or after
    ``event_start``, and every view whose ``file_key`` is in ``base_keys`` (the first table's views).
    """
    dates = full["date_mid"]
    keep = (dates < old_before) | (dates >= event_start) | full["file_key"].isin(set(base_keys))
    return full[keep].reset_index(drop=True)


def exclusion_reasons(views: pd.DataFrame, fixes: DateFixes) -> np.ndarray:
    """Why each view must not be used: one of ``EXCLUSIONS`` (the first that applies), or ``""``.

    Needs the columns ``file_key``, ``date_mid``, ``rolled``, ``duplicate_of``, ``k1`` and ``k2``, and
    ``date_confidence`` when the fixes set ``after_fire_min_confidence``.

    * ``misdated``: the photo is listed, or any copy of it is (a member of its duplicate group): a
      photograph shows what it shows under every file name, and a copy's date is no better.
    * ``montage``: the photo is listed.
    * ``fire_day``: dated in ``[fire_start, after_fire_from)``.
    * ``uncertain_date``: dated from ``after_fire_from`` on, with a ``date_confidence`` below
      ``after_fire_min_confidence`` (or none), and not in ``after_fire_verified``.
    * ``before_spire``: dated before ``exclude_before``.
    * ``rolled``, ``duplicate`` (a later copy), ``bad_distortion`` (``k1`` or ``k2`` not finite).
    """
    n, keys, dates = len(views), views["file_key"], views["date_mid"].to_numpy()
    duplicate_of = views["duplicate_of"].to_numpy()
    first_copy = np.where(duplicate_of >= 0, duplicate_of, np.arange(n))        # names each view's duplicate group
    uncertain, too_early = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    if fixes.after_fire_min_confidence is not None:
        confident = views["date_confidence"].to_numpy() >= fixes.after_fire_min_confidence    # False for NaN
        uncertain = ((dates >= fixes.after_fire_from) & ~confident
                     & ~keys.isin(fixes.after_fire_verified).to_numpy())
    if fixes.exclude_before is not None:
        too_early = dates < fixes.exclude_before
    applies = {
        "misdated": np.isin(first_copy, first_copy[keys.isin(fixes.misdated).to_numpy()]),
        "montage": keys.isin(fixes.montages).to_numpy(),
        "fire_day": (dates >= fixes.fire_start) & (dates < fixes.after_fire_from),
        "uncertain_date": uncertain,
        "before_spire": too_early,
        "rolled": views["rolled"].to_numpy(dtype=bool),
        "duplicate": duplicate_of >= 0,
        "bad_distortion": ~(np.isfinite(views["k1"].to_numpy()) & np.isfinite(views["k2"].to_numpy())),
    }
    return np.select([applies[reason] for reason in EXCLUSIONS], EXCLUSIONS, default="")


# ----------------------------------------------------------------------------
# Sideways views
# ----------------------------------------------------------------------------
def roll_angles(c2w: np.ndarray) -> np.ndarray:
    """Roll of each camera about its viewing axis relative to the scene's up direction, in degrees.

    The scene's up is the component-wise median of the cameras' up vectors (``-y`` in OpenCV axes),
    as in the 2026-11-19 audit: most photos are held upright, and the median ignores the ones that
    are not. The roll is the angle, in the image plane, between the image's up and the scene's up:
    0 for an upright photo, about +-90 for a sideways one (positive when the scene's up points to
    the image's right), 180 for upside down. It is undefined for a camera that looks straight along
    the up direction.

    The estimate of up, and so the roll, is only good to a few degrees: a camera that looks up leans
    its up vector backwards, and the median depends on which views are in the table. That is enough
    to tell sideways photos from upright ones, the only use made of it.

    Args:
        c2w: ``[N, 4, 4]`` camera-to-world matrices, OpenCV axes.
    """
    up = np.median(-c2w[:, :3, 1], axis=0)
    up = up / np.linalg.norm(up)
    up_in_camera = np.einsum("nji,j->ni", c2w[:, :3, :3], up)
    return np.degrees(np.arctan2(up_in_camera[:, 0], -up_in_camera[:, 1]))


# ----------------------------------------------------------------------------
# Duplicate photos
# ----------------------------------------------------------------------------
def grey_thumbnail(image: Image.Image, size: int = THUMB_SIZE) -> np.ndarray:
    """The whole image as a ``size`` x ``size`` grey thumbnail, mean-removed and of unit norm, flattened.

    The dot product of two such vectors is the correlation of the two thumbnails. A constant image
    gives the zero vector (it correlates with nothing).
    """
    grey = np.asarray(image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float64).ravel()
    grey = grey - grey.mean()
    norm = np.linalg.norm(grey)
    return grey / norm if norm > 0 else grey


def load_thumbnails(image_paths: Iterable[str]) -> np.ndarray:
    """``[N, THUMB_SIZE ** 2]``: the :func:`grey_thumbnail` of every image file, in order."""
    thumbs = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            thumbs.append(grey_thumbnail(image))
    return np.stack(thumbs)


def pose_differences(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise differences between cameras: centre distance and rotation angle.

    Returns:
        ``[N, N]`` distances between camera centres in units of the scene's spread (the median
        distance of a camera from the median centre), and ``[N, N]`` angles in degrees of the
        rotation that takes one camera's orientation to the other's (it includes roll).
    """
    centres = c2w[:, :3, 3]
    spread = np.median(np.linalg.norm(centres - np.median(centres, axis=0), axis=1))
    distance = np.linalg.norm(centres[:, None] - centres[None], axis=-1) / (spread if spread > 0 else 1.0)
    # trace(Ri^T Rj) = 1 + 2 cos(angle), and it is the sum of the elementwise products of Ri and Rj.
    trace = np.einsum("iab,jab->ij", c2w[:, :3, :3], c2w[:, :3, :3])
    angle = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    return distance, angle


def duplicate_pairs(thumbs: np.ndarray, c2w: np.ndarray, shared: np.ndarray, date_lo: np.ndarray,
                    date_hi: np.ndarray, *, min_corr: float = MIN_THUMB_CORR,
                    max_centre_dist: float = MAX_CENTRE_DIST, max_rotation_deg: float = MAX_ROTATION_DEG,
                    min_point_overlap: float = MIN_POINT_OVERLAP) -> np.ndarray:
    """Pairs of views that show the same picture: one photo uploaded, scanned or cropped twice, or two
    near-identical shots.

    Three signs are compared:

    1. thumbnails: the grey thumbnails correlate above ``min_corr``;
    2. cameras: the centres are within ``max_centre_dist`` of the scene's spread and the orientations
       within ``max_rotation_deg``;
    3. points: of the 3D points that either view observes, both observe more than
       ``min_point_overlap`` (the Jaccard index of the two sets of points).

    A pair is a duplicate when at least two signs agree and the date intervals ``[date_lo, date_hi)``
    overlap. No sign is enough alone, and each pair of signs catches copies the third misses:

    * thumbnails + cameras: the plain case. Different photos of the facade can have similar
      thumbnails, but not from the same camera.
    * cameras + points: copies that differ in tone, border or crop, whose thumbnails disagree.
    * thumbnails + points: copies that COLMAP placed apart. Two scans of one print can get
      different focal lengths, and then different camera positions.

    The points are compared with the union, not with the view that has fewer: a zoomed-in shot from
    the same spot shares nearly all of its own points with the wider shot, and is not a copy of it.
    The dates must overlap because one photograph has one date, and two photos from the same spot at
    different dates are exactly what a date-conditioned model learns from.

    Args:
        thumbs: ``[N, D]`` rows from :func:`grey_thumbnail`.
        c2w: ``[N, 4, 4]`` camera-to-world matrices.
        shared: ``[N, N]`` shared 3D point counts with each view's own count on the diagonal
            (:func:`shared_point_counts`).
        date_lo, date_hi: ``[N]`` date intervals in decimal years.

    Returns:
        ``[M, 2]`` int array of row pairs ``(i, j)`` with ``i < j``, sorted.
    """
    distance, angle = pose_differences(c2w)
    same_camera = (distance < max_centre_dist) & (angle < max_rotation_deg)
    same_thumbnail = thumbs @ thumbs.T > min_corr
    own = np.diag(shared).astype(np.float64)
    either = own[:, None] + own[None] - shared
    same_points = np.divide(shared, either, out=np.zeros_like(either), where=either > 0) > min_point_overlap
    signs = same_thumbnail.astype(int) + same_camera + same_points
    same_date = (date_lo[:, None] < date_hi[None]) & (date_lo[None] < date_hi[:, None])
    return np.argwhere(np.triu(same_date & (signs >= 2), k=1))


def duplicate_groups(n: int, pairs: np.ndarray) -> np.ndarray:
    """``duplicate_of`` for ``n`` views from duplicate pairs.

    Pairs are chained (if a = b and b = c then all three are one photo). The first row of each
    group keeps -1; every later copy gets the row of that first copy. Views in no pair get -1.
    """
    first = np.arange(n)

    def find(i: int) -> int:
        while first[i] != i:
            first[i] = first[first[i]]
            i = first[i]
        return i

    for a, b in np.asarray(pairs, dtype=int).reshape(-1, 2):
        ra, rb = find(a), find(b)
        first[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)], dtype=int)
    return np.where(roots == np.arange(n), -1, roots)


def listed_pairs(file_keys: Sequence[str], key_pairs: Iterable[tuple[str, str]]) -> np.ndarray:
    """Row pairs ``[M, 2]`` of the pairs of file keys that are both in the table (the others are left out)."""
    row_of = {key: row for row, key in enumerate(file_keys)}
    rows = [(row_of[a], row_of[b]) for a, b in key_pairs if a in row_of and b in row_of]
    return np.array(rows, dtype=int).reshape(-1, 2)


def flag_views(views: pd.DataFrame, thumbs: np.ndarray, shared: np.ndarray, fixes: DateFixes) -> pd.DataFrame:
    """The view table with ``rolled``, ``duplicate_of``, ``excluded`` and ``usable`` added (see the module docstring).

    Duplicates are the pairs :func:`duplicate_pairs` finds and the pairs the fixes list by hand,
    chained into groups together.

    Args:
        views: the table (pose, date interval, ``file_key``, ``k1``/``k2``, and ``date_confidence`` if the
            fixes use it), rows numbered from 0.
        thumbs: ``[N, D]`` thumbnails of the rows' images (:func:`load_thumbnails`).
        shared: ``[N, N]`` shared 3D point counts (:func:`shared_point_counts`).
    """
    c2w = colmap_to_c2w(views[["qw", "qx", "qy", "qz"]].to_numpy(), views[["tx", "ty", "tz"]].to_numpy())
    pairs = np.concatenate([
        duplicate_pairs(thumbs, c2w, shared, views["date_lo"].to_numpy(), views["date_hi"].to_numpy()),
        listed_pairs(views["file_key"].tolist(), fixes.duplicates)])
    flagged = views.copy()
    flagged["rolled"] = np.abs(roll_angles(c2w)) > MAX_ROLL_DEG
    flagged["duplicate_of"] = duplicate_groups(len(views), pairs)
    flagged["excluded"] = exclusion_reasons(flagged, fixes)
    flagged["usable"] = flagged["excluded"] == ""
    return flagged


# ----------------------------------------------------------------------------
# COLMAP: which 3D points each view observes, and where all its cameras are
# ----------------------------------------------------------------------------
def _read_exact(stream: BinaryIO, n_bytes: int) -> bytes:
    """The next ``n_bytes`` of a stream; an error if the file ends before."""
    data = stream.read(n_bytes)
    if len(data) != n_bytes:
        raise ValueError(f"unexpected end of file ({len(data)} of {n_bytes} bytes)")
    return data


def _read_name(stream: BinaryIO) -> str:
    """A null-terminated UTF-8 string."""
    raw = bytearray()
    while (byte := _read_exact(stream, 1)) != b"\x00":
        raw += byte
    return raw.decode("utf-8")


def _image_records(stream: BinaryIO, images_bin: Path) -> Iterator[tuple[tuple, str, int]]:
    """Walk the images of a COLMAP ``images.bin`` opened as ``stream``.

    The file is read as a stream (it is 867 MB for Notre-Dame): each record is a 64-byte header
    (image id, rotation quaternion ``qw qx qy qz``, translation ``tx ty tz``, camera id), a
    null-terminated name, a count and that many keypoints (x, y, 3D point id). Yields the unpacked
    header, the name and the number of keypoints of each image, with the stream at its keypoints:
    the caller may read them, and need not, because the walk seeks to the next image itself.
    Raises if the images do not end at the end of the file.
    """
    (n_images,) = _UINT64.unpack(_read_exact(stream, _UINT64.size))
    for _ in range(n_images):
        header = _IMAGE_HEADER.unpack(_read_exact(stream, _IMAGE_HEADER.size))
        name = _read_name(stream)
        (n_keypoints,) = _UINT64.unpack(_read_exact(stream, _UINT64.size))
        next_image = stream.tell() + n_keypoints * _POINT2D.itemsize
        yield header, name, n_keypoints
        stream.seek(next_image)
    if stream.tell() != os.path.getsize(images_bin):
        raise ValueError(f"{images_bin}: {n_images} images end at byte {stream.tell()}, not at the end of the file")


def read_observed_points(images_bin: Path, names: Iterable[str]) -> dict[str, np.ndarray]:
    """Ids of the 3D points observed by the named images of a COLMAP ``images.bin``.

    Keypoints of images that were not asked for are skipped. Keypoints without a 3D point (id -1)
    are dropped; the others keep the file's order.

    Returns:
        ``{name: int64 point ids}`` for the names found (a name missing from the file is absent).
    """
    wanted = set(names)
    observed: dict[str, np.ndarray] = {}
    with open(images_bin, "rb") as stream:
        for _, name, n_keypoints in _image_records(stream, images_bin):
            if name in wanted:
                keypoints = np.frombuffer(_read_exact(stream, n_keypoints * _POINT2D.itemsize), dtype=_POINT2D)
                ids = keypoints["point3d_id"]
                observed[name] = ids[ids != -1].astype(np.int64)
    return observed


def read_camera_centres(images_bin: Path) -> np.ndarray:
    """Camera centres ``[M, 3]`` (float64, COLMAP units) of ALL images registered in a COLMAP ``images.bin``."""
    with open(images_bin, "rb") as stream:
        poses = np.array([header[1:8] for header, _, _ in _image_records(stream, images_bin)], dtype=np.float64)
    poses = poses.reshape(-1, 7)                            # qw qx qy qz, tx ty tz
    return colmap_to_c2w(poses[:, :4], poses[:, 4:])[:, :3, 3]


def camera_box(centres: np.ndarray, scene_id: int, model: int) -> dict:
    """The contents of ``scene.json``: the box of a COLMAP model's camera centres ``[M, 3]``.

    ``camera_extent`` is the box's longest side, the number MVGenMaster's scale rule compares with
    its ``camera_longest_side`` (see ``timemachine.mvgen.data``).
    """
    low, high = centres.min(axis=0), centres.max(axis=0)
    return {"scene_id": int(scene_id), "model": int(model), "n_registered_cameras": len(centres),
            "camera_min": low.tolist(), "camera_max": high.tolist(), "camera_extent": float((high - low).max())}


def load_camera_box(colmap_dir: Path, cache: Path, scene_id: int, model: int) -> dict:
    """:func:`camera_box` of a COLMAP model: from ``cache`` (the ``scene.json`` an earlier run wrote for the
    same scene and model) if it exists, otherwise from ``images.bin`` in ``colmap_dir``."""
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if (cached.get("scene_id"), cached.get("model")) == (scene_id, model):
            return cached
    return camera_box(read_camera_centres(colmap_dir / "images.bin"), scene_id, model)


def read_points3d(points3d_bin: Path) -> tuple[np.ndarray, np.ndarray]:
    """Ids (int64, sorted) and positions (float64 ``[P, 3]``) of the points of a COLMAP ``points3D.bin``."""
    data = Path(points3d_bin).read_bytes()
    (n_points,) = _UINT64.unpack_from(data, 0)
    ids, xyz = np.empty(n_points, dtype=np.int64), np.empty((n_points, 3), dtype=np.float64)
    offset = _UINT64.size
    try:
        for i in range(n_points):
            ids[i], x, y, z, _, _, _, _, track_length = _POINT3D_HEADER.unpack_from(data, offset)
            xyz[i] = x, y, z
            offset += _POINT3D_HEADER.size + _TRACK_ELEMENT_BYTES * track_length
    except struct.error as error:
        raise ValueError(f"{points3d_bin}: the file ends inside point {i} of {n_points}") from error
    if offset != len(data):
        raise ValueError(f"{points3d_bin}: {n_points} points end at byte {offset}, not at the end of the file")
    order = np.argsort(ids)
    return ids[order], xyz[order]


def pack_observations(observed: dict[str, np.ndarray], names: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the per-view point ids in row order.

    Returns:
        ``view_starts`` (int64 ``[N + 1]``) and ``view_pid`` (int64): row ``i`` observes
        ``view_pid[view_starts[i]:view_starts[i + 1]]``. A name without observations gets none.
    """
    per_view = [observed.get(name, np.empty(0, dtype=np.int64)) for name in names]
    starts = np.concatenate([[0], np.cumsum([len(ids) for ids in per_view])]).astype(np.int64)
    return starts, np.concatenate(per_view) if per_view else np.empty(0, dtype=np.int64)


def read_cached_sparse_points(cache: Path, names: Sequence[str]
                              ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray] | None:
    """What an earlier run parsed, from its ``sparse_points.npz``, if that covers the views ``names``.

    Returns:
        ``({name: point ids}, p3d_ids, p3d_xyz)``, or None if the file does not exist or lacks one
        of the names (the COLMAP model must then be parsed again).
    """
    if not cache.exists():
        return None
    with np.load(cache) as archive:
        cached_names = archive["view_names"].tolist()
        if not set(names) <= set(cached_names):
            return None
        starts, point_ids = archive["view_starts"], archive["view_pid"]
        observed = {name: point_ids[starts[i]:starts[i + 1]] for i, name in enumerate(cached_names)}
        return observed, archive["p3d_ids"], archive["p3d_xyz"]


def load_sparse_points(colmap_dir: Path, cache: Path, names: Sequence[str]) -> dict[str, np.ndarray]:
    """The arrays of ``sparse_points.npz`` for the views with the COLMAP image names ``names``, in that order.

    Taken from ``cache`` (the file an earlier run wrote) when it covers the names, otherwise parsed
    from ``images.bin`` and ``points3D.bin`` in ``colmap_dir``. Raises if a view is not in the COLMAP
    model, or observes a point that is not in ``points3D.bin``.
    """
    cached = read_cached_sparse_points(cache, names)
    if cached is None:
        observed = read_observed_points(colmap_dir / "images.bin", names)
        missing = [name for name in names if name not in observed]
        if missing:
            raise ValueError(f"{len(missing)} views are not in {colmap_dir / 'images.bin'}, e.g. {missing[0]}")
        p3d_ids, p3d_xyz = read_points3d(colmap_dir / "points3D.bin")
    else:
        observed, p3d_ids, p3d_xyz = cached
    view_starts, view_pid = pack_observations(observed, names)
    if not np.isin(view_pid, p3d_ids).all():
        raise ValueError(f"views observe 3D points that are not in {colmap_dir / 'points3D.bin'}")
    return {"p3d_ids": p3d_ids, "p3d_xyz": p3d_xyz, "view_starts": view_starts, "view_pid": view_pid,
            "view_names": np.array(names)}


def shared_point_counts(view_starts: np.ndarray, view_pid: np.ndarray) -> np.ndarray:
    """Number of 3D points each pair of views shares.

    Args:
        view_starts, view_pid: as returned by :func:`pack_observations`.

    Returns:
        uint16 ``[N, N]``, symmetric; the diagonal is the number of distinct points the view
        observes. A point seen through two keypoints of one view counts once. Counts above
        65,535 saturate.
    """
    n = len(view_starts) - 1
    view = np.repeat(np.arange(n), np.diff(view_starts))
    points, point = np.unique(view_pid, return_inverse=True)
    sees = sparse.csr_matrix((np.ones(len(view), dtype=np.int32), (view, point)), shape=(n, len(points)))
    sees.data[:] = 1                                        # the constructor summed repeated (view, point) entries
    counts = (sees @ sees.T).toarray()
    return np.minimum(counts, np.iinfo(np.uint16).max).astype(np.uint16)


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def covisible_medians(covis: np.ndarray, period: np.ndarray, usable: np.ndarray, n_periods: int,
                      min_shared: int = MIN_SHARED) -> pd.DataFrame:
    """Per period: how many usable views a usable view is co-visible with, in its period and outside it.

    Two views are co-visible when they share at least ``min_shared`` 3D points. Returns one row per
    period with the two medians, ``same_period`` and ``other_periods`` (NaN for a period without
    usable views).
    """
    linked = (covis >= min_shared) & usable[None, :]
    np.fill_diagonal(linked, False)
    rows = []
    for p in range(n_periods):
        members = usable & (period == p)
        same = linked[np.ix_(members, period == p)].sum(axis=1)
        other = linked[np.ix_(members, period != p)].sum(axis=1)
        rows.append({"same_period": float(np.median(same)) if len(same) else float("nan"),
                     "other_periods": float(np.median(other)) if len(other) else float("nan")})
    return pd.DataFrame(rows)


def summary_table(views: pd.DataFrame, covis: np.ndarray, edges: Sequence[float]) -> pd.DataFrame:
    """Per period, and for all views: the number of views, of views excluded for each reason, and of usable
    views, and the two medians of :func:`covisible_medians`.

    A view counts under one reason only (its ``excluded`` value), so each row adds up:
    ``views`` = the reasons + ``usable``.
    """
    names = period_names(edges)
    period = era_index(views["date_mid"].to_numpy(), tuple(edges))
    excluded = views["excluded"].to_numpy()
    counts = {"views": np.bincount(period, minlength=len(names))}
    for reason in (*EXCLUSIONS, ""):
        counts[reason or "usable"] = np.bincount(period[excluded == reason], minlength=len(names))
    table = pd.DataFrame(counts, index=names)
    table.loc["all"] = table.sum()
    links = covisible_medians(covis, period, views["usable"].to_numpy(dtype=bool), len(names))
    links.index = names
    return table.join(links)


def print_summary(views: pd.DataFrame, covis: np.ndarray, edges: Sequence[float], n_stereo: int, n_failed: int
                  ) -> None:
    """Print :func:`summary_table` with the counts it cannot show (flags that overlap, views left out)."""
    print(f"\n{len(views)} views; {int(views['usable'].sum())} usable; {n_failed} downloads failed and "
          f"{n_stereo} stereo cards dropped (not in the table)")
    print(f"flags: {int(views['rolled'].sum())} rolled, {int((views['duplicate_of'] >= 0).sum())} later copies of "
          "duplicates; below, a view with several reasons counts under the first")
    print(f"same_period / other_periods: median number of usable views sharing >= {MIN_SHARED} points with a "
          "usable view of the period")
    print(summary_table(views, covis, edges).to_string(na_rep="-", float_format=lambda value: f"{value:g}"))


# ----------------------------------------------------------------------------
# Building
# ----------------------------------------------------------------------------
def renumbered_rows(old_keys: Sequence[str], new_keys: Sequence[str]) -> np.ndarray:
    """Rows of an existing view table that a new table would give to another view, or drop.

    ``old_keys`` and ``new_keys`` are the ``file_key`` columns of the two tables. Depth maps are
    stored by row, so they survive a rebuild only if this is empty; rows added at the end are fine.
    """
    old, new = np.asarray(old_keys, dtype=object), np.asarray(new_keys, dtype=object)
    kept = np.zeros(len(old), dtype=bool)
    n = min(len(old), len(new))
    kept[:n] = old[:n] == new[:n]
    return np.flatnonzero(~kept)


def check_rows_are_kept(existing_table: Path, new_keys: Sequence[str]) -> None:
    """Raise if replacing the view table at ``existing_table`` would renumber its rows (nothing to check if
    there is no table yet). Prints the outcome otherwise."""
    if not existing_table.exists():
        return
    old_keys = pd.read_parquet(existing_table, columns=["file_key"])["file_key"].tolist()
    moved = renumbered_rows(old_keys, new_keys)
    if len(moved):
        row = int(moved[0])
        becomes = repr(new_keys[row]) if row < len(new_keys) else "no view"
        raise ValueError(
            f"{existing_table}: {len(moved)} of its {len(old_keys)} rows would get another view or disappear (the "
            f"first is row {row}, {old_keys[row]!r}, which would become {becomes}). Depth maps are stored by row: "
            "delete the folder's depth maps and depth_stats.parquet, then run again with --renumber.")
    print(f"rows kept: all {len(old_keys)} rows of the existing table keep their view "
          f"({len(new_keys) - len(old_keys)} rows added at the end)", flush=True)


def write_together(writers: dict[Path, Callable[[Path], None]]) -> None:
    """Write several files under temporary names, then rename them all.

    The files of the dataset folder describe the same rows, so they must change together: a run
    that is interrupted while writing leaves the folder as it was (plus ``partial_`` files, which
    the next run overwrites). The temporary names keep the suffix, because numpy appends
    ``.npy``/``.npz`` to names without it.
    """
    temporary = {path: path.with_name(f"partial_{path.name}") for path in writers}
    for path, write in writers.items():
        write(temporary[path])
    for path in writers:
        temporary[path].replace(path)


def main() -> None:
    paths = DataPaths()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-table", type=Path, default=paths.dataset_dir(SCENE_ID, MODEL) / "views.parquet",
                        help="the first view table; all its views are kept")
    parser.add_argument("--date-fixes", type=Path, default=DATE_FIXES,
                        help="JSON with the dates of the fire and the photos to exclude (see DateFixes)")
    parser.add_argument("--colmap-dir", type=Path,
                        default=REPO_ROOT / "data/audit_2026-11-19/data_cameras/colmap_25512_0",
                        help="folder with the COLMAP model's images.bin and points3D.bin")
    parser.add_argument("--out", type=Path, default=paths.root / "datasets" / f"{SCENE_ID}_model{MODEL}_v2")
    parser.add_argument("--max-side", type=int, default=1024, help="longest side of cached images")
    parser.add_argument("--workers", type=int, default=4, help="download threads")
    parser.add_argument("--renumber", action="store_true",
                        help="replace an existing table even if its rows get other views (its depth maps, stored "
                             "by row, must be deleted first)")
    args = parser.parse_args()
    started = time.time()

    fixes = load_date_fixes(args.date_fixes)
    full = build_view_table(paths, SCENE_ID, MODEL)
    unknown = sorted(fixes.listed_keys - set(full["file_key"]))
    if unknown:
        raise ValueError(f"{args.date_fixes} names files that are not posed and dated views of the scene: {unknown}")
    base = pd.read_parquet(args.base_table)
    views = select_views(full, base["file_key"], OLD_BEFORE, fixes.fire_start)
    print(f"{len(views)} of {len(full)} posed and dated views selected "
          f"({int(views['file_key'].isin(base['file_key']).sum())} of the first table's {len(base)})", flush=True)

    views = download_view_images(MegaScenesBucket(), paths, views, args.max_side, args.workers)
    failed = views.attrs.pop("failed", [])
    for failure in failed:
        print(f"download failed: {failure['file_key']}: {failure['error']}")
    stereo = likely_stereo_cards(views)
    print(f"{len(views)} images on disk ({len(failed)} downloads failed); dropping {int(stereo.sum())} likely "
          f"stereo cards: {views.loc[stereo, 'file_key'].tolist()}", flush=True)
    views = views[~stereo].reset_index(drop=True)

    points = load_sparse_points(args.colmap_dir, args.out / "sparse_points.npz", views["path"].tolist())
    covis = shared_point_counts(points["view_starts"], points["view_pid"])
    print(f"COLMAP: {len(points['p3d_ids'])} 3D points, {len(points['view_pid'])} observations in the "
          f"{len(views)} views ({time.time() - started:.0f} s)", flush=True)
    views = flag_views(views, load_thumbnails(views["image_path"]), covis, fixes)
    scene = load_camera_box(args.colmap_dir, args.out / SCENE_FILE, SCENE_ID, MODEL)
    print(f"scene: {scene['n_registered_cameras']} registered cameras, their centres span "
          f"{scene['camera_extent']:.4f} COLMAP units along the longest axis", flush=True)

    if not args.renumber:
        check_rows_are_kept(args.out / "views.parquet", views["file_key"].tolist())
    args.out.mkdir(parents=True, exist_ok=True)
    write_together({args.out / "sparse_points.npz": lambda path: np.savez(path, **points),
                    args.out / "covisibility.npy": lambda path: np.save(path, covis),
                    args.out / SCENE_FILE: lambda path: path.write_text(json.dumps(scene, indent=1), encoding="utf-8"),
                    args.out / "views.parquet": lambda path: views.to_parquet(path, index=False)})
    print(f"wrote {args.out}: views.parquet, sparse_points.npz, covisibility.npy, {SCENE_FILE} "
          f"({time.time() - started:.0f} s)")
    print_summary(views, covis, (SPIRE_BUILT, OLD_BEFORE, fixes.after_fire_from), int(stereo.sum()), len(failed))


if __name__ == "__main__":
    main()
