"""How training samples are drawn from a scene's view table.

A sample is ``n_inputs`` input views plus ``n_targets`` target views from one
COLMAP model (same coordinate frame). All targets share one date (a single,
3D-consistent snapshot of the scene); inputs may come from any date, or only
from dates at least ``min_input_gap`` years away from the targets. "Same
date" is a tolerance around an anchor date: tight for modern photos (whose dates
are exact and whose scenes can change within a year, e.g. Notre-Dame in 2019)
and looser for old photos (whose dates are often "circa").

Because inputs may come from any date, most samples mix eras (e.g. 2015 inputs
for 1880 targets): the model can only lower its loss by using the target dates
instead of copying the inputs' look.

``max_neighbors`` keeps a sample spatially coherent: targets and inputs are
drawn from the views whose cameras are closest to the anchor's (see
:func:`timemachine.data.cameras.camera_features`). Without it, views can come
from opposite sides of the building and share no content, so the targets
cannot be inferred from the inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    n_inputs: int = 3
    n_targets: int = 5                  # 8 views total: SEVA's training setting on 48 GB GPUs
    modern_tolerance: float = 1.0       # years, for anchors dated >= tolerance_boundary
    old_tolerance: float = 5.0          # years, for older anchors
    tolerance_boundary: float = 1990.0
    min_input_gap: float = 0.0          # inputs must differ from the target date by more than this
    old_target_fraction: float | None = None  # e.g. 0.5: half the samples get pre-`old_before` targets
    old_before: float = 1970.0
    max_neighbors: int | None = None    # draw targets/inputs among the k cameras nearest the anchor

    def __post_init__(self):
        if self.max_neighbors is not None and self.max_neighbors < max(self.n_targets - 1, self.n_inputs):
            raise ValueError("max_neighbors must leave room for all targets and inputs.")

    @property
    def n_views(self) -> int:
        return self.n_inputs + self.n_targets


def tolerance(cfg: SamplingConfig, anchor_date: np.ndarray | float) -> np.ndarray | float:
    """Date tolerance (years) for target groups anchored at ``anchor_date``."""
    return np.where(np.asarray(anchor_date) >= cfg.tolerance_boundary, cfg.modern_tolerance, cfg.old_tolerance)


def eligible_anchors(dates: np.ndarray, cfg: SamplingConfig) -> np.ndarray:
    """Indices of views whose same-date group can supply all targets and leave enough inputs."""
    diff = np.abs(dates[:, None] - dates[None, :])                    # (N, N)
    same = diff <= tolerance(cfg, dates)[:, None]
    if cfg.min_input_gap > 0:   # conservative: inputs from outside the target group only
        n_inputs_available = ((diff > cfg.min_input_gap) & ~same).sum(axis=1)
    else:
        n_inputs_available = np.full(len(dates), len(dates) - cfg.n_targets)
    ok = (same.sum(axis=1) >= cfg.n_targets) & (n_inputs_available >= cfg.n_inputs)
    return np.flatnonzero(ok)


def _nearest(pool: np.ndarray, distance: np.ndarray, k: int | None) -> np.ndarray:
    """The (at most) ``k`` members of ``pool`` with the smallest ``distance``; all of them if k is None."""
    if k is None or len(pool) <= k:
        return pool
    return pool[np.argsort(distance[pool], kind="stable")[:k]]


def sample_views(dates: np.ndarray, cfg: SamplingConfig, rng: np.random.Generator,
                 anchors: np.ndarray | None = None,
                 camera_feats: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw one sample: ``(input_indices, target_indices)`` into ``dates``.

    ``dates`` holds one representative date (decimal year) per view. Pass
    precomputed ``anchors`` (from :func:`eligible_anchors`) to avoid recomputing.
    ``camera_feats`` (N, D) is required when ``cfg.max_neighbors`` is set.
    """
    dates = np.asarray(dates, dtype=np.float64)
    if anchors is None:
        anchors = eligible_anchors(dates, cfg)
    if len(anchors) == 0:
        raise ValueError("No view can anchor a sample: too few views share a date.")
    if cfg.max_neighbors is not None and camera_feats is None:
        raise ValueError("max_neighbors needs camera features.")

    pool = anchors
    if cfg.old_target_fraction is not None and rng.random() < cfg.old_target_fraction:
        old = anchors[dates[anchors] < cfg.old_before]
        pool = old if len(old) else anchors
    anchor = rng.choice(pool)
    distance = (np.zeros(len(dates)) if camera_feats is None
                else np.linalg.norm(camera_feats - camera_feats[anchor], axis=1))

    gap = np.abs(dates - dates[anchor])
    same = np.flatnonzero(gap <= tolerance(cfg, dates[anchor]))
    others = _nearest(np.setdiff1d(same, [anchor]), distance, cfg.max_neighbors)
    targets = np.concatenate([[anchor], rng.choice(others, cfg.n_targets - 1, replace=False)])

    candidates = np.setdiff1d(np.arange(len(dates)), targets)
    if cfg.min_input_gap > 0:
        candidates = candidates[gap[candidates] > cfg.min_input_gap]
    candidates = _nearest(candidates, distance, cfg.max_neighbors)
    inputs = rng.choice(candidates, cfg.n_inputs, replace=False)
    return inputs, targets
