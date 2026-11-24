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

*Change events* (``events``, ``event_prob``): for scenes with a known physical
change (Notre-Dame's 2019 fire, the Sphinx excavated around 1930, the Sydney bridge
completed in 1932), some samples take the targets from one side of the change and
the inputs from the other. There the inputs cannot tell what the targets look like;
only the date can, so the date matters for the denoising loss.
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
    input_gap_fraction: float = 1.0     # share of samples that enforce min_input_gap (others: any date)
    gap_old_targets_only: bool = False  # enforce the gap only for old targets (never modern-from-old samples)
    old_target_fraction: float | None = None  # e.g. 0.5: half the samples get pre-`old_before` targets
    old_before: float = 1970.0
    max_neighbors: int | None = None    # draw targets/inputs among the k cameras nearest the anchor
    events: tuple[tuple[int, float], ...] = ()  # (scene_id, year of a physical change)
    event_prob: float = 0.0             # share of samples drawn across an event (scenes with one)
    event_window: float | None = 15.0   # event inputs within this many years of the change, when enough exist

    def __post_init__(self):
        if not 0.0 <= self.input_gap_fraction <= 1.0:
            raise ValueError("input_gap_fraction must be in [0, 1].")
        if self.max_neighbors is not None and self.max_neighbors < max(self.n_targets - 1, self.n_inputs):
            raise ValueError("max_neighbors must leave room for all targets and inputs.")
        if not 0.0 <= self.event_prob <= 1.0:
            raise ValueError("event_prob must be in [0, 1].")

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
    if cfg.min_input_gap > 0:   # conservative: far enough from *any* date the targets can have
        reach = cfg.min_input_gap + tolerance(cfg, dates)[:, None]
        n_inputs_available = ((diff > reach) & ~same).sum(axis=1)
    else:
        n_inputs_available = np.full(len(dates), len(dates) - cfg.n_targets)
    ok = (same.sum(axis=1) >= cfg.n_targets) & (n_inputs_available >= cfg.n_inputs)
    return np.flatnonzero(ok)


def event_anchors(dates: np.ndarray, cfg: SamplingConfig, split_year: float) -> np.ndarray:
    """Views that can anchor a sample across a change at ``split_year``: enough same-date views
    on their side for the targets, and enough views on the other side for the inputs."""
    after = dates >= split_year
    same = np.abs(dates[:, None] - dates[None, :]) <= tolerance(cfg, dates)[:, None]
    same_side = same & (after[:, None] == after[None, :])
    other_side = np.where(after, (~after).sum(), after.sum())
    return np.flatnonzero((same_side.sum(axis=1) >= cfg.n_targets) & (other_side >= cfg.n_inputs))


def _nearest(pool: np.ndarray, distance: np.ndarray, k: int | None) -> np.ndarray:
    """The (at most) ``k`` members of ``pool`` with the smallest ``distance``; all of them if k is None."""
    if k is None or len(pool) <= k:
        return pool
    return pool[np.argsort(distance[pool], kind="stable")[:k]]


def sample_views(dates: np.ndarray, cfg: SamplingConfig, rng: np.random.Generator,
                 anchors: np.ndarray | None = None,
                 camera_feats: np.ndarray | None = None,
                 split_year: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw one sample: ``(input_indices, target_indices)`` into ``dates``.

    ``dates`` holds one representative date (decimal year) per view. Pass
    precomputed ``anchors`` (from :func:`eligible_anchors`, or :func:`event_anchors`
    with ``split_year``) to avoid recomputing. ``camera_feats`` (N, D) is required
    when ``cfg.max_neighbors`` is set. With ``split_year``, the targets come from the
    anchor's side of that year and the inputs from the other side (the input gap is
    not applied).
    """
    dates = np.asarray(dates, dtype=np.float64)
    if anchors is None:
        anchors = eligible_anchors(dates, cfg) if split_year is None else event_anchors(dates, cfg, split_year)
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
    if split_year is not None:
        after = dates >= split_year
        same = same[after[same] == after[anchor]]
    others = _nearest(np.setdiff1d(same, [anchor]), distance, cfg.max_neighbors)
    targets = np.concatenate([[anchor], rng.choice(others, cfg.n_targets - 1, replace=False)])

    candidates = np.setdiff1d(np.arange(len(dates)), targets)
    if split_year is not None:                  # inputs from the other side of the change
        candidates = candidates[(dates[candidates] >= split_year) != (dates[anchor] >= split_year)]
        if cfg.event_window is not None:      # prefer inputs close to the change: it, not the era, differs
            close = candidates[np.abs(dates[candidates] - split_year) <= cfg.event_window]
            candidates = close if len(close) >= cfg.n_inputs else candidates
        inputs = rng.choice(_nearest(candidates, distance, cfg.max_neighbors), cfg.n_inputs, replace=False)
        return inputs, targets
    # Draw only when mixing, so input_gap_fraction = 1 keeps the exact sample stream of older runs.
    enforce_gap = cfg.min_input_gap > 0 and (cfg.input_gap_fraction >= 1.0
                                             or rng.random() < cfg.input_gap_fraction)
    if cfg.gap_old_targets_only and dates[anchor] >= cfg.old_before:
        enforce_gap = False     # modern targets: inputs of any date, so the target date says nothing about them
    if enforce_gap:             # more than the gap from every target's date
        to_targets = np.abs(dates[candidates, None] - dates[None, targets]).min(axis=1)
        candidates = candidates[to_targets > cfg.min_input_gap]
    candidates = _nearest(candidates, distance, cfg.max_neighbors)
    inputs = rng.choice(candidates, cfg.n_inputs, replace=False)
    return inputs, targets
