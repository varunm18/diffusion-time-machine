"""Dated view sets of one scene for MVGenMaster: co-visibility sampling, cameras and depth.

A sample is some reference views followed by target views of one COLMAP model (by default 3 and
5). MVGenMaster keeps the references clean and denoises the targets. It is told where every view
is (camera rays) and what the references show from each target's camera (the references warped
by their depth maps: its "3D priors"). So besides images, cameras and dates a sample needs a
depth map for every reference, in the cameras' units, and it is only well-posed if the
references show what the targets show.

**Sampling** (:func:`sample_rows`). The 2026-11-19 audit found that 24-74% of the targets drawn
for SEVA shared essentially nothing with their inputs, because views were picked by camera
distance. Here they are picked by *co-visibility*, the number of COLMAP 3D points two views share:

1. an anchor target is drawn among the views with enough co-visible views of the same date;
2. the other targets are drawn among the ``top_k`` such views that share most points with the
   anchor. All targets lie within the anchor's date tolerance and in its period, so they are
   one consistent snapshot of the scene;
3. references must share points with at least ``min_targets_seen`` of the targets and are drawn
   in proportion to the number of targets they see. Together they must see *every* target: a
   draw that leaves a target without a co-visible reference is redrawn (before this rule about
   5% of the targets, in a fifth of the samples, had to be invented). References may have any
   date. With probability ``cross_period_prob`` they all come from another *period* than the
   targets, when such views can form a reference set. Periods are separated by ``period_edges``
   (for Notre-Dame: 1970, and the fire of April 2019). In those samples the references show the
   scene as it was at another time, so what the targets look like has to come from their dates.

The number of references is ``n_ref``, or, with ``n_ref_choices``, drawn per sample (MVGenMaster's
last training stage used 1 to 3); a sample always has ``n_ref + n_targets`` views.

**Dataset folder** (:class:`MVGenScene` reads it; ``scripts/mvgen/build_dataset.py`` and
``scripts/mvgen/prepare_depth.py`` write it):

* ``views.parquet``: one row per view, with the columns :func:`timemachine.data.dataset.load_view`
  needs, the COLMAP pose, the date interval, and ``usable`` (bool: the build found no reason to
  exclude the view, such as a wrong date, a sideways camera or being a copy of another view);
* ``covisibility.npy``: uint16 ``[N, N]``, the number of 3D points two views share;
* ``scene.json``: the box of the camera centres of ALL images registered in the COLMAP model, not
  only the table's (``camera_min``, ``camera_max``, and ``camera_extent``, its longest side);
* ``sparse_points.npz``: the COLMAP points and which of them each view observes (``p3d_ids``
  sorted, ``p3d_xyz``, ``view_starts`` ``[N + 1]``, ``view_pid``); see :func:`sparse_depth_points`;
* ``depth/<row:05d>.npy``: float16 ``[512, 512]`` z-depth of the view's crop in COLMAP units
  (monocular depth scaled to the view's COLMAP points);
* ``depth_stats.parquet``: per view ``row``, ``file_key``, ``n_points``, ``scale``,
  ``median_rel_error`` and ``ok`` (:func:`depth_is_reliable`). Only views with ``ok`` are drawn as
  references. ``file_key`` ties the depth maps to the view table they were made for: rebuilding
  the table can renumber its rows, and a depth map would then silently belong to another photo.

**Sample** (F views, references first):

=============  ====================  ==========================================================
``images``     float32 [F, 3, S, S]  undistorted centre crops in [-1, 1]
``K``          float32 [F, 3, 3]     pixel intrinsics of the crops
``w2c``        float32 [F, 4, 4]     world-to-camera, OpenCV axes, after the scene-scale rule
``depth``      float32 [F, 1, S, S]  z-depth in the same (rescaled) units; zeros for targets
``n_ref``      int                   number of reference views
``scale``      float                 the factor that translations and depth were multiplied by
``date``       float32 [F, 2]        date interval ``[lo, hi)`` in decimal years
``date_mid``   float32 [F]           interval midpoints
``rows``       int64 [F]             rows of the view table
``file_keys``  list of str           Commons file of each view
=============  ====================  ==========================================================

**Scene scale** (:func:`shrink_factor`; the ``scale`` argument of :class:`MVGenScene`). COLMAP's
unit is arbitrary. MVGenMaster shrinks cameras whose centres span more than
``camera_longest_side`` (5) along an axis so that they span exactly that, and leaves smaller sets
alone; translations and depth get the same factor. Its *training* applied the rule once per
scene, to all of the scene's cameras (``src/datasets/dl3dv_dataset.py`` in its checkout), so every
sample of a scene had the same factor: this is ``scale="scene"``, the default, with the extent
from ``scene.json`` (Notre-Dame: 57.57, a factor of 0.087). Its demo script applies the rule to
the cameras of each view set instead (``scale="sample"``); the factor then depends on which views
are drawn together. A number is used as the constant factor itself.

Nothing here imports MVGenMaster's code, so the module also loads in our own environment.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from timemachine.data.cameras import colmap_to_c2w
from timemachine.data.dataset import load_view
from timemachine.data.holdout import era_index

# A scaled depth map is trusted for warping if it was scaled with at least DEPTH_MIN_POINTS COLMAP
# points and, after scaling, differs from them by at most DEPTH_MAX_REL_ERROR (median relative error).
DEPTH_MIN_POINTS = 50
DEPTH_MAX_REL_ERROR = 0.15
DEPTH_STATS_COLUMNS = ("row", "file_key", "n_points", "scale", "median_rel_error", "ok")
SCENE_FILE = "scene.json"
SCALE_RULES = ("scene", "sample")           # the named values of MVGenScene's ``scale`` (or a number)

_REQUIRED = ("file_key", "image_path", "image_width", "image_height", "camera_width", "camera_height",
             "fx", "fy", "cx", "cy", "k1", "k2", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
             "date_lo", "date_hi", "date_mid", "usable")
# The numeric view-table columns a sample is made from: its camera, its pose and its dates (MVGenScene.fingerprint).
_SAMPLE_COLUMNS = tuple(column for column in _REQUIRED if column not in ("file_key", "image_path", "usable"))
_MAX_ATTEMPTS = 1000        # anchors tried per sample before giving up (a drawn anchor can lack references)
_REF_ATTEMPTS = 20          # reference sets drawn for one target group before giving up on it


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ViewSamplingConfig:
    n_ref: int = 3
    n_targets: int = 5                      # 8 views in total, as MVGenMaster was trained
    # If given, each sample draws its number of references uniformly from these entries (a repeated
    # entry is drawn more often) and has n_ref + n_targets - drawn targets. None: always n_ref.
    n_ref_choices: tuple[int, ...] | None = None
    min_shared: int = 30                    # two views are co-visible if they share this many 3D points
    min_targets_seen: int = 3               # a reference must be co-visible with this many of the targets
    top_k: int = 24                         # targets come from the k views most co-visible with the anchor
    modern_tolerance: float = 1.0           # years around the anchor's date, for anchors from tolerance_boundary on
    old_tolerance: float = 5.0              # years, for older anchors (their dates are often "circa")
    tolerance_boundary: float = 1990.0
    period_edges: tuple[float, ...] = (1970.0, 2019.2877)   # Notre-Dame: old | modern | after the fire
    anchor_period_weights: tuple[float, ...] | None = None  # P(anchor's period); None: uniform over anchors
    cross_period_prob: float = 0.5          # P(all references are drawn from another period than the targets')

    def __post_init__(self):
        if self.n_ref < 1 or self.n_targets < 1:
            raise ValueError("n_ref and n_targets must be at least 1.")
        choices = self.n_ref_choices
        if choices is not None and (len(choices) == 0 or any(
                isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k < self.n_views for k in choices)):
            raise ValueError("n_ref_choices must hold whole numbers of references, each at least 1 and leaving "
                             "at least one of the n_ref + n_targets views as a target.")
        if self.min_shared < 1:
            raise ValueError("min_shared must be at least 1: views sharing no point are not co-visible.")
        if not 1 <= self.min_targets_seen <= min(self.target_counts):
            raise ValueError("min_targets_seen must be between 1 and the number of targets (the smallest that "
                             "n_ref_choices allows).")
        if self.top_k < max(self.target_counts) - 1:
            raise ValueError("top_k must leave room for the targets drawn next to the anchor (one fewer than the "
                             "number of targets; the largest that n_ref_choices allows).")
        if self.modern_tolerance < 0 or self.old_tolerance < 0:
            raise ValueError("date tolerances must not be negative.")
        if any(b <= a for a, b in zip(self.period_edges, self.period_edges[1:])):
            raise ValueError("period_edges must be strictly increasing.")
        weights = self.anchor_period_weights
        if weights is not None and (len(weights) != len(self.period_edges) + 1 or min(weights) < 0
                                    or sum(weights) <= 0):
            raise ValueError("anchor_period_weights needs one non-negative weight per period "
                             "(len(period_edges) + 1), not all zero.")
        if not 0.0 <= self.cross_period_prob <= 1.0:
            raise ValueError("cross_period_prob must be in [0, 1].")

    @property
    def n_views(self) -> int:
        return self.n_ref + self.n_targets

    @property
    def ref_counts(self) -> tuple[int, ...]:
        """The equally likely numbers of references of a sample: ``n_ref_choices``, or ``n_ref`` alone."""
        return (self.n_ref,) if self.n_ref_choices is None else tuple(int(k) for k in self.n_ref_choices)

    @property
    def target_counts(self) -> tuple[int, ...]:
        """The numbers of targets a sample can have, sorted: one per distinct number of references."""
        return tuple(sorted({self.n_views - k for k in self.ref_counts}))


def draw_n_ref(cfg: ViewSamplingConfig, rng: np.random.Generator) -> int:
    """The number of references of the next sample: ``n_ref``, or a uniform draw from ``n_ref_choices``.

    Nothing is drawn from ``rng`` when the number is fixed.
    """
    return cfg.n_ref if cfg.n_ref_choices is None else int(rng.choice(cfg.n_ref_choices))


def period_names(edges: Sequence[float]) -> list[str]:
    """Names of the ``len(edges) + 1`` periods that ``edges`` separate, e.g. ``"1859 to 1970"``."""
    if len(edges) == 0:
        return ["all"]
    return ([f"before {edges[0]:.10g}"] + [f"{a:.10g} to {b:.10g}" for a, b in zip(edges, edges[1:])]
            + [f"from {edges[-1]:.10g}"])


def date_tolerance(cfg: ViewSamplingConfig, anchor_date: np.ndarray | float) -> np.ndarray:
    """Date tolerance (years) of a target group anchored at ``anchor_date``."""
    return np.where(np.asarray(anchor_date) >= cfg.tolerance_boundary, cfg.modern_tolerance, cfg.old_tolerance)


def target_partners(dates: np.ndarray, covis: np.ndarray, cfg: ViewSamplingConfig, target_ok: np.ndarray,
                    anchors: np.ndarray) -> np.ndarray:
    """``[len(anchors), N]`` bool: the views each anchor could share a target group with.

    A partner is another ``target_ok`` view within the anchor's date tolerance, in the anchor's
    period, that shares at least ``min_shared`` points with the anchor.
    """
    dates, anchors = np.asarray(dates, dtype=np.float64), np.asarray(anchors, dtype=np.int64)
    period = era_index(dates, cfg.period_edges)
    same_date = np.abs(dates[anchors, None] - dates[None, :]) <= date_tolerance(cfg, dates[anchors])[:, None]
    partners = (same_date & (period[anchors, None] == period[None, :])
                & (covis[anchors] >= cfg.min_shared) & np.asarray(target_ok, dtype=bool)[None, :])
    partners[np.arange(len(anchors)), anchors] = False
    return partners


def eligible_anchors(dates: np.ndarray, covis: np.ndarray, cfg: ViewSamplingConfig, target_ok: np.ndarray,
                     n_targets: int | None = None) -> np.ndarray:
    """Rows that can anchor a sample of ``n_targets`` targets (default ``cfg.n_targets``): ``target_ok`` views
    with at least ``n_targets - 1`` partners."""
    rows = np.flatnonzero(target_ok)
    n_partners = target_partners(dates, covis, cfg, target_ok, rows).sum(axis=1)
    return rows[n_partners >= (cfg.n_targets if n_targets is None else n_targets) - 1]


def anchor_probabilities(anchors: np.ndarray, period: np.ndarray, cfg: ViewSamplingConfig) -> np.ndarray | None:
    """Probability of drawing each of ``anchors``, or None if no anchor can be drawn.

    Uniform by default. With ``anchor_period_weights``, a period is drawn with those probabilities
    and then an anchor uniformly inside it; a period without anchors is skipped (the weights of
    the others are renormalised; :class:`MVGenScene` refuses to train like that).
    """
    if len(anchors) == 0:
        return None
    if cfg.anchor_period_weights is None:
        return np.full(len(anchors), 1.0 / len(anchors))
    anchor_period = period[anchors]
    per_period = np.bincount(anchor_period, minlength=len(cfg.anchor_period_weights))
    weights = np.asarray(cfg.anchor_period_weights, dtype=np.float64)[anchor_period] / per_period[anchor_period]
    return weights / weights.sum() if weights.sum() > 0 else None


def draw_references(pool: np.ndarray, sees: np.ndarray, n_ref: int, rng: np.random.Generator) -> np.ndarray | None:
    """``n_ref`` views of ``pool`` that together see every target, or None if no such set is found.

    Args:
        pool: rows of the views that may be references; each sees at least one target.
        sees: ``[N, T]`` bool, whether each view of the table is co-visible with each target.

    The views are drawn without replacement, in proportion to the number of targets they see. A
    draw that leaves a target unseen is redrawn, up to ``_REF_ATTEMPTS`` times: the result is a
    draw from the same distribution, limited to the sets that see every target. None when the
    pool is too small, cannot see every target at all, or the attempts run out.
    """
    if len(pool) < n_ref or not sees[pool].any(axis=0).all():
        return None
    weights = sees[pool].sum(axis=1).astype(np.float64)
    weights /= weights.sum()
    for _ in range(_REF_ATTEMPTS):
        refs = rng.choice(pool, n_ref, replace=False, p=weights)
        if sees[refs].any(axis=0).all():
            return refs
    return None


def sample_rows(dates: np.ndarray, covis: np.ndarray, cfg: ViewSamplingConfig, rng: np.random.Generator,
                target_ok: np.ndarray, ref_ok: np.ndarray, anchors: np.ndarray | None = None,
                n_ref: int | None = None) -> np.ndarray | None:
    """Draw one sample: the rows ``[references..., targets...]``, or None if the drawn anchor cannot be completed.

    Args:
        dates: ``[N]`` one representative date per view (decimal years).
        covis: ``[N, N]`` number of 3D points shared by two views.
        target_ok, ref_ok: ``[N]`` bool, the views that may be targets and references.
        anchors: the result of :func:`eligible_anchors` for the same arguments and the sample's number of
            targets, to avoid recomputing it.
        n_ref: the number of references (default ``cfg.n_ref``); the other ``cfg.n_views - n_ref`` views
            are targets.

    The steps are described in the module docstring. The first target is the anchor. None is
    returned when no view can anchor a sample, or when no ``n_ref`` candidate references that see
    every drawn target are found; the caller retries with the same generator.
    """
    dates = np.asarray(dates, dtype=np.float64)
    period = era_index(dates, cfg.period_edges)
    n_ref = cfg.n_ref if n_ref is None else n_ref
    n_targets = cfg.n_views - n_ref
    if anchors is None:
        anchors = eligible_anchors(dates, covis, cfg, target_ok, n_targets)
    probabilities = anchor_probabilities(anchors, period, cfg)
    if probabilities is None:
        return None
    anchor = int(rng.choice(anchors, p=probabilities))

    partners = np.flatnonzero(target_partners(dates, covis, cfg, target_ok, np.array([anchor]))[0])
    most_shared = partners[np.argsort(-covis[anchor, partners].astype(np.int64), kind="stable")[:cfg.top_k]]
    targets = np.concatenate([[anchor], rng.choice(most_shared, n_targets - 1, replace=False)])

    sees = covis[:, targets] >= cfg.min_shared              # [N, T]: view i is co-visible with target j
    candidate = np.asarray(ref_ok, dtype=bool) & (sees.sum(axis=1) >= cfg.min_targets_seen)
    candidate[targets] = False
    pool = np.flatnonzero(candidate)
    refs = None
    if rng.random() < cfg.cross_period_prob:
        refs = draw_references(pool[period[pool] != period[anchor]], sees, n_ref, rng)
    if refs is None:
        refs = draw_references(pool, sees, n_ref, rng)
    if refs is None:
        return None
    return np.concatenate([refs, targets]).astype(np.int64)


# ----------------------------------------------------------------------------
# Scene scale and depth
# ----------------------------------------------------------------------------
def shrink_factor(extent: float, longest_side: float) -> float:
    """MVGenMaster's scale rule: the factor for the translations and depth of a set of cameras.

    ``extent`` is the largest span of the camera centres along an axis. Above ``longest_side`` the
    factor shrinks it to ``longest_side``; smaller sets are not enlarged (factor 1).
    """
    return longest_side / extent if extent > longest_side else 1.0


def scene_rescale(centres: np.ndarray, longest_side: float) -> float:
    """:func:`shrink_factor` of the cameras with the centres ``centres`` (``[F, 3]``).

    Applied to the cameras of one sample this is the rule of MVGenMaster's demo (``load_dataset``
    in its ``run_mvgen.py``, ``normalise_scale`` in ``scripts/mvgen/nvs_test.py``).
    """
    return shrink_factor(float((centres.max(axis=0) - centres.min(axis=0)).max()), longest_side)


def checked_scale(scale: str | float) -> str | float:
    """A valid ``scale`` setting of :class:`MVGenScene`: one of ``SCALE_RULES``, or a positive float."""
    if isinstance(scale, str):
        if scale not in SCALE_RULES:
            raise ValueError(f"scale must be one of {SCALE_RULES} or a positive number, got {scale!r}.")
        return scale
    if isinstance(scale, bool) or not isinstance(scale, Real) or not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be one of {SCALE_RULES} or a positive number, got {scale!r}.")
    return float(scale)


def read_camera_extent(dataset_dir: Path) -> float:
    """``camera_extent`` of a dataset folder's ``scene.json``: the longest side of the box of ALL the COLMAP
    model's camera centres, in COLMAP units."""
    path = Path(dataset_dir) / SCENE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{dataset_dir} has no {SCENE_FILE}, which scale='scene' needs: run "
                                "scripts/mvgen/build_dataset.py to write it, or pass scale='sample' or a number.")
    extent = json.loads(path.read_text(encoding="utf-8")).get("camera_extent")
    if isinstance(extent, bool) or not isinstance(extent, Real) or not (math.isfinite(extent) and extent > 0):
        raise ValueError(f"{path}: camera_extent must be a positive number, got {extent!r}.")
    return float(extent)


def depth_path(dataset_dir: Path, row: int) -> Path:
    """Where the depth map of view-table row ``row`` is stored."""
    return Path(dataset_dir) / "depth" / f"{int(row):05d}.npy"


def depth_is_reliable(n_points: np.ndarray | int, median_rel_error: np.ndarray | float) -> np.ndarray:
    """The ``ok`` column of ``depth_stats.parquet``: enough COLMAP points, and agreement with them.

    A view without points has an undefined (NaN) error and is not reliable.
    """
    return (np.asarray(n_points) >= DEPTH_MIN_POINTS) & (np.asarray(median_rel_error) <= DEPTH_MAX_REL_ERROR)


def read_depth_stats(dataset_dir: Path, file_keys: np.ndarray) -> pd.DataFrame | None:
    """``depth_stats.parquet`` of a dataset folder, or None if it has not been written yet.

    ``file_keys`` are those of the folder's view table. Stats made for another view table are
    refused (wherever their ``file_key`` column shows it): the depth maps are stored by row, and
    a rebuilt table can number its rows differently.
    """
    path = Path(dataset_dir) / "depth_stats.parquet"
    if not path.exists():
        return None
    stats = pd.read_parquet(path)
    rows = stats["row"].to_numpy()
    if (rows >= len(file_keys)).any() or ("file_key" in stats and not np.array_equal(stats["file_key"].to_numpy(),
                                                                                     np.asarray(file_keys)[rows])):
        raise ValueError(f"{path} was made for another view table: delete it and the depth folder, then run "
                         "scripts/mvgen/prepare_depth.py again.")
    return stats


def sparse_depth_points(sparse: Mapping[str, np.ndarray], row: int, c2w: np.ndarray, K: np.ndarray, size: int
                        ) -> np.ndarray:
    """``[P, 3]`` float32 ``(u, v, z)`` of the COLMAP points view ``row`` observes, inside its crop.

    Args:
        sparse: the arrays of ``sparse_points.npz`` (``p3d_ids`` sorted, ``p3d_xyz``, ``view_starts``,
            ``view_pid``); pass a dict, not the lazy ``np.load`` object, when calling this many times.
        c2w: ``[4, 4]`` camera-to-world of the view in COLMAP units, before any rescaling.
        K: ``[3, 3]`` pixel intrinsics of the undistorted ``size`` x ``size`` crop.

    ``u, v`` are crop pixels (the top-left pixel's centre is at 0.5, 0.5) and ``z`` the depth along
    the viewing axis. Points behind the camera or outside the crop are left out. A port of
    ``sparse_points()`` in ``scripts/export_nvs_samples.py``, which only imports in SEVA's environment.
    """
    point_ids = sparse["view_pid"][sparse["view_starts"][row]:sparse["view_starts"][row + 1]]
    index = np.searchsorted(sparse["p3d_ids"], point_ids).clip(max=len(sparse["p3d_ids"]) - 1)
    if not np.array_equal(sparse["p3d_ids"][index], point_ids):
        raise ValueError(f"sparse points: view {row} observes point ids that are not in p3d_ids")
    w2c = np.linalg.inv(c2w)
    cam = sparse["p3d_xyz"][index] @ w2c[:3, :3].T + w2c[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):       # points in the camera plane; dropped below
        uv = (cam / cam[:, 2:3]) @ np.asarray(K, dtype=np.float64).T
    keep = (cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
    return np.column_stack([uv[keep, :2], cam[keep, 2]]).astype(np.float32)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class MVGenScene(Dataset):
    """Random dated samples from one scene's dataset folder (see the module docstring).

    Sampling is deterministic per (seed, epoch, index); call :meth:`set_epoch` each epoch for
    fresh samples. :meth:`hold_out` removes views from the samples; held-out views can still be
    loaded with :meth:`make_sample`, for evaluation on photos the model never trained on.
    :meth:`fingerprint` is a hash of the folder's content that decides the samples, by which a
    resumed run notices that the dataset was rebuilt under it.

    When the scene is built, and again whenever the hold-out changes, it prints how many views and
    anchors each period has (:meth:`period_table`), and it refuses a period that
    ``anchor_period_weights`` gives a weight but that has no anchor.

    Attributes (N = number of views): ``views`` (the table), ``dates`` (``[N]`` interval midpoints),
    ``covis`` (``[N, N]``), ``period`` (``[N]`` period index by ``config.period_edges``), ``c2ws``
    (``[N, 4, 4]``, COLMAP units), ``usable`` and ``depth_ok`` (``[N]`` bool, from the folder),
    ``scale`` (the setting) and ``scale_factor`` (the constant factor of every sample, or None for
    ``scale="sample"``), and, set by :meth:`hold_out`: ``heldout``, ``target_ok``, ``ref_ok`` (``[N]``
    bool) and ``anchors`` (the rows that can anchor a sample of ``config.n_targets`` targets).

    Args:
        resolution: side of the square crops; the depth maps must have been prepared at this size.
        camera_longest_side: MVGenMaster's ``config.camera_longest_side`` (see :func:`shrink_factor`).
        require_depth: draw references only among views whose depth map is ``ok`` in
            ``depth_stats.parquet``. Without it every usable view can be drawn as a reference, which is
            for studying the sampler before depth exists: :meth:`make_sample` needs the references'
            depth files either way.
        scale: how camera translations and depth are scaled (the module docstring's "Scene scale"):
            ``"scene"``, one factor for every sample from the extent of all the scene's cameras in
            ``scene.json`` (what MVGenMaster was trained with); ``"sample"``, the rule applied to
            each sample's own cameras (its demo); or a positive number, that constant factor.
    """

    def __init__(self, dataset_dir: str | Path, config: ViewSamplingConfig = ViewSamplingConfig(),
                 resolution: int = 512, samples_per_epoch: int = 1000, seed: int = 0,
                 camera_longest_side: float = 5.0, require_depth: bool = True, scale: str | float = "scene"):
        self.dataset_dir = Path(dataset_dir)
        self.config = config
        self.resolution = resolution
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.camera_longest_side = camera_longest_side
        self.require_depth = require_depth
        self.scale = checked_scale(scale)
        self.scale_factor = self._constant_scale_factor()
        self.epoch = 0

        views = pd.read_parquet(self.dataset_dir / "views.parquet")
        missing = [c for c in _REQUIRED if c not in views]
        if missing:
            raise ValueError(f"View table is missing columns: {missing}")
        self.views = views.reset_index(drop=True)
        n = len(self.views)
        self.covis = np.load(self.dataset_dir / "covisibility.npy")
        if self.covis.shape != (n, n):
            raise ValueError(f"covisibility.npy has shape {self.covis.shape}, but the view table has {n} rows.")
        self.dates = self.views["date_mid"].to_numpy(dtype=np.float64)
        self.period = era_index(self.dates, config.period_edges)
        self.c2ws = colmap_to_c2w(self.views[["qw", "qx", "qy", "qz"]].to_numpy(),
                                  self.views[["tx", "ty", "tz"]].to_numpy())
        self.usable = self.views["usable"].to_numpy(dtype=bool)
        self.depth_ok = self._read_depth_ok()
        self.hold_out(np.array([], dtype=int))

    def _constant_scale_factor(self) -> float | None:
        """The factor every sample is scaled by, or None when each sample has its own (``scale="sample"``)."""
        if self.scale == "sample":
            return None
        if self.scale == "scene":
            return shrink_factor(read_camera_extent(self.dataset_dir), self.camera_longest_side)
        return self.scale

    def _read_depth_ok(self) -> np.ndarray:
        """``[N]`` bool: views whose depth map is ``ok`` in ``depth_stats.parquet`` (none if there is no such file)."""
        depth_ok = np.zeros(len(self.views), dtype=bool)
        stats = read_depth_stats(self.dataset_dir, self.views["file_key"].to_numpy())
        if stats is not None:
            depth_ok[stats["row"].to_numpy()[stats["ok"].to_numpy(dtype=bool)]] = True
        elif self.require_depth:
            raise FileNotFoundError(f"{self.dataset_dir} has no depth_stats.parquet: run scripts/mvgen/prepare_depth.py"
                                    " on it, or pass require_depth=False to study the sampler without depth.")
        return depth_ok

    def fingerprint(self) -> str:
        """A hash (SHA-256, in hex) of what in the dataset folder decides which samples are drawn and what
        cameras and dates they carry.

        It covers, for every row of the view table in order: the file key, the ``usable`` flag, the
        camera (size, intrinsics, distortion, pose), the dates and, with ``require_depth``, whether
        the depth map is reliable, which is whether the view may be drawn as a reference; and the
        co-visibility matrix. Two scenes opened with the same settings whose fingerprints agree
        draw the same view sets with the same cameras and dates.

        It leaves out what is too large to read (the pixels and the depth maps), where the image
        files are stored, the hold-out (which follows from the dates and the run's settings), and the
        scene-scale factor: ``scale_factor`` is one number, which callers compare on its own.

        The trainer stores both in every checkpoint and refuses to resume a run on a folder that
        gives another (``timemachine.mvgen.trainer``, "Starting and resuming"): a rebuilt view table
        or a changed ``scene.json`` would otherwise continue the run on other samples or at another
        scale, in silence.
        """
        digest = hashlib.sha256()
        digest.update("\0".join(self.views["file_key"].astype(str)).encode("utf-8"))
        parts = [self.views[list(_SAMPLE_COLUMNS)].to_numpy(dtype=np.float64), self.usable, self.covis]
        if self.require_depth:
            parts.append(self.depth_ok)
        for part in parts:
            digest.update(np.ascontiguousarray(part).tobytes())
        return digest.hexdigest()

    def hold_out(self, rows: np.ndarray) -> None:
        """Draw samples only from the other views (replaces any earlier hold-out).

        Sets ``heldout``, ``target_ok`` (usable and not held out), ``ref_ok`` (``target_ok`` with
        reliable depth, if ``require_depth``) and ``anchors``. Raises if no sample can be drawn any
        more, or none in a period that ``anchor_period_weights`` gives a weight; otherwise prints
        :meth:`period_table`.
        """
        self.heldout = np.zeros(len(self.views), dtype=bool)
        self.heldout[np.asarray(rows, dtype=int)] = True
        self.target_ok = self.usable & ~self.heldout
        self.ref_ok = self.target_ok & self.depth_ok if self.require_depth else self.target_ok.copy()
        self._anchors: dict[int, np.ndarray] = {}           # by number of targets, filled by _anchors_for
        self.anchors = self._anchors_for(self.config.n_targets)
        self._check_anchors()
        largest = max(self.config.target_counts)
        print(f"[scene] {self.dataset_dir}: {len(self.views)} views, {int(self.heldout.sum())} held out; scale "
              f"{self.scale!r}" + ("" if self.scale_factor is None else f" (factor {self.scale_factor:.6g})")
              + f"; anchors are counted for {largest} targets\n"
              + self.period_table().to_string(float_format=lambda value: f"{value:.3f}"), flush=True)

    def _anchors_for(self, n_targets: int) -> np.ndarray:
        """Rows that can anchor a sample of ``n_targets`` targets, under the current hold-out."""
        if n_targets not in self._anchors:
            self._anchors[n_targets] = eligible_anchors(self.dates, self.covis, self.config, self.target_ok,
                                                        n_targets)
        return self._anchors[n_targets]

    def _check_anchors(self) -> None:
        """Raise unless samples can be drawn as configured: with every number of targets, and in every period
        that has an anchor weight.

        The anchors of the largest target group are checked: they can anchor the smaller ones too.
        """
        cfg = self.config
        anchors = self._anchors_for(max(cfg.target_counts))
        if len(anchors) == 0:
            raise ValueError("No sample can be drawn: no training view has enough co-visible views of its date to "
                             "anchor a target group.")
        if cfg.anchor_period_weights is None:
            return
        names = period_names(cfg.period_edges)
        per_period = np.bincount(self.period[anchors], minlength=len(names))
        empty = [name for name, weight, count in zip(names, cfg.anchor_period_weights, per_period)
                 if weight > 0 and count == 0]
        if empty:
            raise ValueError(f"No sample can be drawn in {empty}: anchor_period_weights gives these periods a "
                             "weight, but no training view there has enough co-visible views of its date to anchor "
                             f"a target group (anchors per period: {dict(zip(names, per_period.tolist()))}). Set "
                             "their weights to 0 or move period_edges.")

    def period_table(self) -> pd.DataFrame:
        """Per period: usable views, training views, possible references, anchors, and the share of samples.

        ``training`` are the usable views that are not held out; ``references`` those of them that
        can be drawn as references (with reliable depth, if ``require_depth``); ``anchors`` those
        that can anchor a target group of the largest size the config draws (the fewest anchors);
        ``anchor_share`` is the probability that an anchor is drawn in the period (the share of
        samples anchored there differs a little, because an anchor that cannot be completed is
        drawn again). A period with a handful of anchors and a large share is trained on those few
        views over and over.
        """
        cfg, names = self.config, period_names(self.config.period_edges)

        def per_period(views: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
            return np.bincount(self.period[views], weights=weights, minlength=len(names))

        share = np.zeros(len(names))
        for n_ref in cfg.ref_counts:
            anchors = self._anchors_for(cfg.n_views - n_ref)
            share += per_period(anchors, anchor_probabilities(anchors, self.period, cfg)) / len(cfg.ref_counts)
        return pd.DataFrame({"usable": per_period(self.usable), "training": per_period(self.target_ok),
                             "references": per_period(self.ref_ok),
                             "anchors": per_period(self._anchors_for(max(cfg.target_counts))),
                             "anchor_share": share}, index=names)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def draw(self, index: int, epoch: int | None = None, seed: int | None = None, n_ref: int | None = None
             ) -> tuple[np.ndarray, int]:
        """View-table rows of sample ``index`` (references first, then targets) and its number of references.

        ``epoch`` / ``seed`` default to the dataset's own; probes pass fixed values so their
        samples do not depend on the training run's seed. ``n_ref`` fixes the number of references;
        by default it is the config's: ``n_ref``, or a draw from ``n_ref_choices``. It is drawn
        once per sample, and anchors that cannot be completed are redrawn with the same generator,
        so the retries do not favour the numbers of references that are easier to complete.
        """
        cfg = self.config
        rng = np.random.default_rng((self.seed if seed is None else seed,
                                     self.epoch if epoch is None else epoch, index))
        n_ref = draw_n_ref(cfg, rng) if n_ref is None else int(n_ref)
        n_targets = cfg.n_views - n_ref
        if n_ref < 1 or not cfg.min_targets_seen <= n_targets <= cfg.top_k + 1:
            raise ValueError(f"n_ref = {n_ref} leaves {n_targets} of the {cfg.n_views} views as targets, which the "
                             f"config does not allow (min_targets_seen {cfg.min_targets_seen}, top_k {cfg.top_k}).")
        anchors = self._anchors_for(n_targets)
        for _ in range(_MAX_ATTEMPTS):
            rows = sample_rows(self.dates, self.covis, cfg, rng, self.target_ok, self.ref_ok, anchors, n_ref)
            if rows is not None:
                return rows, n_ref
        raise RuntimeError(f"No sample of {n_ref} references and {n_targets} targets could be completed in "
                           f"{_MAX_ATTEMPTS} attempts: too few of the {int(self.ref_ok.sum())} views that can be "
                           "references see the targets.")

    def indices(self, index: int, epoch: int | None = None, seed: int | None = None, n_ref: int | None = None
                ) -> np.ndarray:
        """The rows of :meth:`draw` alone, for callers that know the number of references.

        With ``n_ref_choices`` that number changes from sample to sample and the rows do not say
        where the references end, so ``n_ref`` must then be given (view sets of a fixed shape, as
        probes use); training samples come from :meth:`draw` or ``scene[index]``.
        """
        if n_ref is None and self.config.n_ref_choices is not None:
            raise ValueError("With n_ref_choices the number of references varies per sample: call draw(), which "
                             "returns it with the rows, or pass n_ref for view sets of a fixed shape.")
        return self.draw(index, epoch, seed, n_ref)[0]

    def __getitem__(self, index: int) -> dict:
        return self.make_sample(*self.draw(index))

    def _load_depth(self, row: int) -> np.ndarray:
        """The stored depth map of one view, ``[S, S]`` float32 in COLMAP units."""
        path = depth_path(self.dataset_dir, row)
        if not path.exists():
            raise FileNotFoundError(f"View {row} ({self.views.file_key[row]}) is a reference but has no depth map "
                                    f"({path}): run scripts/mvgen/prepare_depth.py on the folder.")
        depth = np.load(path).astype(np.float32)
        if depth.shape != (self.resolution, self.resolution):
            raise ValueError(f"{path} has shape {depth.shape}, but samples are made at resolution "
                             f"{self.resolution}: prepare the depth at that size.")
        return depth

    def make_sample(self, rows: np.ndarray, n_ref: int) -> dict:
        """A sample from explicit view-table rows; the first ``n_ref`` are the references.

        Works for any rows of the table (held-out views, views not ``usable``), as long as every
        reference has a depth file. The fields are listed in the module docstring; camera
        translations and depth are multiplied by ``sample["scale"]``, the scene's ``scale_factor``
        or, with ``scale="sample"``, the factor of these views' cameras.
        """
        rows = np.array(rows, dtype=np.int64)
        if not 1 <= n_ref <= len(rows):
            raise ValueError(f"n_ref must be between 1 and the number of views ({len(rows)}), got {n_ref}.")
        sub = self.views.iloc[rows]
        images, Ks = zip(*(load_view(row, self.resolution) for row in sub.itertuples()))
        c2w = self.c2ws[rows]                           # a copy (indexed by an array)
        factor = self.scale_factor
        if factor is None:
            factor = scene_rescale(c2w[:, :3, 3], self.camera_longest_side)
        c2w[:, :3, 3] *= factor
        depth = torch.zeros(len(rows), 1, self.resolution, self.resolution)
        for j in range(n_ref):
            depth[j, 0] = torch.from_numpy(self._load_depth(int(rows[j]))) * factor
        return {
            "images": torch.stack(images),
            "K": torch.from_numpy(np.stack(Ks)).float(),
            "w2c": torch.from_numpy(np.linalg.inv(c2w)).float(),
            "depth": depth,
            "n_ref": int(n_ref),
            "scale": float(factor),
            "date": torch.from_numpy(sub[["date_lo", "date_hi"]].to_numpy(dtype=np.float32)),
            "date_mid": torch.from_numpy(sub["date_mid"].to_numpy(dtype=np.float32)),
            "rows": torch.from_numpy(rows),
            "file_keys": sub["file_key"].tolist(),
        }
