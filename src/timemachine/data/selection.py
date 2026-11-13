"""Choose which COLMAP models go into the multi-scene dataset, and in what role.

Having old photos is not enough: a sample's targets must share one date (see
:mod:`timemachine.data.sampling`), so a model can only produce *old targets* if
several of its old photos are close in date. A view that has enough same-date
companions to start a target group is an **anchor**. Each model gets a role:

``old_target``  at least one pre-1970 anchor: can produce samples whose targets are old.
``old_input``   enough old and modern views, but no old anchor: its old photos can
                only be inputs for modern targets (kept: those samples teach the
                model not to copy an old input's look into a modern view).
``excluded``    too few old or modern views.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd

from timemachine.analysis.census import ERAS, MODERN_ERAS, OLD_ERAS
from timemachine.data.sampling import SamplingConfig, eligible_anchors

MIN_ROLE_VIEWS = 10          # per side (pre-1970 and 2000+), same as the census gate
WIDE_OLD_TOLERANCE = 10.0    # years; reported for comparison only, not used for roles


def era_of(dates: np.ndarray) -> np.ndarray:
    """Era label of each date (decimal year)."""
    dates = np.asarray(dates, dtype=np.float64)
    labels = np.empty(len(dates), dtype=object)
    for label, lo, hi in ERAS:
        labels[(dates >= lo) & (dates < hi)] = label
    return labels


def anchor_counts(dates: np.ndarray, cfg: SamplingConfig = SamplingConfig()) -> dict[str, int]:
    """Number of views per era that can anchor a target group under ``cfg``."""
    dates = np.asarray(dates, dtype=np.float64)
    eras = era_of(dates[eligible_anchors(dates, cfg)]) if len(dates) else np.array([], dtype=object)
    return {label: int((eras == label).sum()) for label, _, _ in ERAS}


def assign_role(n_old: int, n_modern: int, old_anchors: int, min_views: int = MIN_ROLE_VIEWS) -> str:
    """Role of a model from its view counts and its number of pre-1970 anchors."""
    if n_old < min_views or n_modern < min_views:
        return "excluded"
    return "old_target" if old_anchors > 0 else "old_input"


def model_row(views: pd.DataFrame, cfg: SamplingConfig = SamplingConfig()) -> dict:
    """Selection statistics of one model's view table (see ``data.views.build_view_table``)."""
    dates = views["date_mid"].to_numpy(dtype=np.float64)
    eras = era_of(dates)
    anchors = anchor_counts(dates, cfg)
    wide = anchor_counts(dates, replace(cfg, old_tolerance=WIDE_OLD_TOLERANCE))
    n_old = int(np.isin(eras, OLD_ERAS).sum())
    n_modern = int(np.isin(eras, MODERN_ERAS).sum())
    old_anchors = sum(anchors[e] for e in OLD_ERAS)
    row = {"views": len(views), "old_views": n_old, "modern_views": n_modern}
    row.update({f"anchors_{label}": n for label, n in anchors.items()})
    row["old_anchors"] = old_anchors
    row["old_anchors_wide"] = sum(wide[e] for e in OLD_ERAS)
    row["role"] = assign_role(n_old, n_modern, old_anchors)
    return row


def cap_modern_views(views: pd.DataFrame, max_modern: int) -> pd.DataFrame:
    """Keep every pre-2000 view and at most ``max_modern`` 2000+ views of a view table.

    Modern views are ranked by a hash of their file name, not drawn at random, so
    the choice is the same on every run and a larger cap keeps a superset of a
    smaller one (raising the cap later only adds downloads).
    """
    modern = views["era"].isin(MODERN_ERAS)
    rank = views.loc[modern, "file_key"].map(lambda k: hashlib.sha1(k.encode()).hexdigest())
    keep = rank.sort_values().index[:max_modern]
    return views[~modern | views.index.isin(keep)].reset_index(drop=True)
