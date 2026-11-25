"""Views a training run never sees, to tell learning a scene apart from memorising its photos.

On a single scene, every probe so far rendered views the model had trained on, so a
model that memorised each photo's look by date was indistinguishable from one that
learned the scene's eras (docs/findings.md, 2026-11-18). A hold-out removes views
from training samples (:meth:`DatedViewsDataset.hold_out`); the held-out probe
(:class:`timemachine.training.probes.HeldoutProbe`) then compares the loss on them
with the loss on trained views over the course of training.

Held-out views are drawn at random within each era of each scene, so the few old
photos are represented; optionally the render probes' target views are held out as
well, so their contact sheets show cameras and photos the model never trained on.
Whole scenes can also be held out, to measure generalisation to unseen places.

**Whole capture days and photographers** (``HoldoutConfig.by = "component"``; experiment look1). A
random view's capture day and photographer usually train: at Notre-Dame 43% of the held-out modern
views and 88% of the after-fire ones had a training view of the same day or of the same photographer
on another occasion, so a held-out photo's look was mostly seen in training. Here the unit of the
hold-out is a connected component of a period's usable views (:func:`holdout_units`), linked by the
same calendar day (views dated to a day: ``date_hi - date_lo <= DAY``) or the same Commons author
(:func:`view_authors`); views not dated to a day are linked by ``(author, date_lo, date_hi)``. Per
period, with one generator seeded by the hold-out seed and used for the periods in their order, the
units are shuffled and, skipping units larger than ``max_unit`` views, added until at least
``fraction`` of the period's usable views AND of its side views (where the stored changed-region mask
covers at least ``side_share`` of the frame) are held out. In the periods of ``unit_periods``
(Notre-Dame's old photos, with 7 side views in all) the first ``fraction`` of the shuffled units is
held out instead, and there, as in the look1 critique, a unit is a day or an ``(author, interval)``
group, not a component. The order in which units are shuffled is that of their labels
(:func:`holdout_units`); it is the critique's and ``scripts/mvgen/make_eval_sets.py``'s, so a seed holds
out the same views in training and in the evaluation. The default (``by = "view"``) is the hold-out
above, unchanged.

**Evaluation sets and the hold-out seed** (:func:`evaluation_sets`, :func:`evaluation_cells`,
:func:`first_holdout_seed`). Look1 scores renders of held-out targets in a 2 x 2 of reference period x
target period. A set is a held-out side-view anchor and at least two more held-out targets (scored),
the remaining targets views of the anchor's period (rendered; scored only if held out), all within a
tolerance of the anchor's date (modern: 5 years, after the fire: 1) and co-visible with it; its
references are training views of the chosen period with reliable depth that share at least
``min_shared`` COLMAP points with every scored target and together see every target. A cell's
capacity is the number of such sets of which no two share a scored target. The seed is the first
whose every cell has at least ``EVAL_MIN_SETS`` sets and that holds out at most ``EVAL_MAX_SHARE`` of
any period. The rule is ``scripts/mvgen/make_eval_sets.py``'s, which writes the evaluation sets.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
import pandas as pd

HOLDOUT_BY = ("view", "component")
DAY = 0.0031                    # years: a view whose date interval is at most this long is dated to a day
AUTHOR_CHARS = 60               # an author's name is compared on its first 60 characters (as the study did)
EVAL_MIN_SETS = 4               # the seed rule: disjoint evaluation sets per 2 x 2 cell, at least
EVAL_MAX_SHARE = 0.15           # ... and at most this share of any period held out
EVAL_TOLERANCES = {1: 5.0, 2: 1.0}      # years around an evaluation anchor's date, by period (modern, after)
EVAL_MIN_SCORED = 3             # held-out targets of an evaluation set: the anchor and at least 2 more
EVAL_SET_TRIES, EVAL_REFERENCE_TRIES = 40, 20       # target groups tried per anchor, reference draws per group
EVAL_SEED_OFFSET = 1000         # the sets of hold-out seed s are drawn with np.random.default_rng(1000 + s)


@dataclass(frozen=True)
class HoldoutConfig:
    fraction: float = 0.0                       # share of each era's views held out at random (0: none)
    probe_targets: bool = False                 # also hold out the render probes' target views
    scenes: tuple[int, ...] = ()                # scene ids held out entirely (evaluation only)
    seed: int = 0                               # fixed separately from the training seed
    era_edges: tuple[float, ...] = (1900.0, 1970.0, 2000.0)
    # "view": random views per era (the default). "component": whole days and photographers (module docstring).
    by: str = "view"
    max_unit: int = 12                          # component: units of more views than this are never held out
    side_share: float = 0.05                    # component: a side view's stored mask covers this share of the frame
    unit_periods: tuple[int, ...] = (0,)        # component: periods held out by a share of their units

    def __post_init__(self):
        if not 0.0 <= self.fraction < 1.0:
            raise ValueError("holdout fraction must be in [0, 1).")
        if self.by not in HOLDOUT_BY:
            raise ValueError(f"holdout by must be one of {HOLDOUT_BY}")
        if self.max_unit < 1 or not 0.0 <= self.side_share <= 1.0:
            raise ValueError("holdout max_unit must be at least 1 and side_share in [0, 1]")
        object.__setattr__(self, "unit_periods", tuple(int(p) for p in self.unit_periods))

    @property
    def active(self) -> bool:
        return self.fraction > 0 or self.probe_targets or bool(self.scenes)


def era_index(dates: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Era bin of each date: 0 before ``edges[0]``, ..., ``len(edges)`` from ``edges[-1]`` on."""
    return np.digitize(np.asarray(dates, dtype=np.float64), edges)


def heldout_rows(dates: np.ndarray, cfg: HoldoutConfig, extra_rows=(), groups: np.ndarray | None = None, *,
                 units: np.ndarray | None = None, usable: np.ndarray | None = None, side: np.ndarray | None = None
                 ) -> np.ndarray:
    """Sorted view rows to hold out: ``round(fraction * n)`` random views per era (per group, if
    ``groups`` gives each view's scene/model group), plus ``extra_rows``.

    With ``cfg.by == "component"`` whole units instead (the module docstring), which needs ``units``
    (:func:`holdout_units`), ``usable`` and ``side`` (bool ``[N]``: the side views); ``groups`` does not
    apply. ``extra_rows`` are then held out as views: their units may still train.
    """
    if cfg.by == "component":
        if units is None or usable is None or side is None:
            raise ValueError("a hold-out by component needs the views' units, usable flags and side views")
        if groups is not None:
            raise ValueError("a hold-out by component is made per scene: groups do not apply")
        held = _component_holdout(dates, cfg, np.asarray(units), np.asarray(usable, dtype=bool),
                                  np.asarray(side, dtype=bool))
        return np.unique(np.concatenate([np.flatnonzero(held), np.asarray(extra_rows, dtype=int)]))
    rng = np.random.default_rng(cfg.seed)
    era = era_index(dates, cfg.era_edges)
    if groups is not None:                      # one stratum per (group, era); group 0 alone = era only
        era = np.asarray(groups) * (len(cfg.era_edges) + 1) + era
    chosen = [np.asarray(extra_rows, dtype=int)]
    for e in np.unique(era):
        members = np.flatnonzero(era == e)
        chosen.append(rng.choice(members, int(round(cfg.fraction * len(members))), replace=False))
    return np.unique(np.concatenate(chosen))


def _component_holdout(dates: np.ndarray, cfg: HoldoutConfig, units: np.ndarray, usable: np.ndarray,
                       side: np.ndarray) -> np.ndarray:
    """``[N]`` bool: the views a hold-out by component takes (the module docstring)."""
    rng = np.random.default_rng(cfg.seed)
    period = era_index(dates, cfg.era_edges)
    held = np.zeros(len(units), dtype=bool)
    for p in range(len(cfg.era_edges) + 1):
        members = usable & (period == p) & (units >= 0)
        order = rng.permutation(np.unique(units[members]))
        if p in cfg.unit_periods:
            held |= members & np.isin(units, order[:int(round(cfg.fraction * len(order)))])
            continue
        n_side = int((members & side).sum())
        for unit in order:
            unit_views = members & (units == unit)
            if unit_views.sum() > cfg.max_unit:
                continue
            if ((held & members).sum() >= cfg.fraction * members.sum()
                    and (held & members & side).sum() >= cfg.fraction * n_side):
                break
            held |= unit_views
    return held


# ----------------------------------------------------------------------------
# Units: days and photographers
# ----------------------------------------------------------------------------
def calendar_day(year: float) -> str:
    """The calendar day (``YYYY-MM-DD``) of a decimal year, as the appearance study wrote it."""
    whole = int(np.floor(year))
    start, end = datetime(whole, 1, 1), datetime(whole + 1, 1, 1)
    return (start + (end - start) * (year - whole)).strftime("%Y-%m-%d")


def normalise_author(names: pd.Series) -> pd.Series:
    """Authors as they are compared: white space collapsed, stripped, the first ``AUTHOR_CHARS`` characters ("" when
    not known)."""
    return names.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, AUTHOR_CHARS)


def view_authors(views: pd.DataFrame) -> np.ndarray:
    """``[N]`` str: each view's Commons author ("" when not known).

    From the view table's ``author`` column if it has one; otherwise from the Commons metadata table
    of its scene (``data/tables/image_metadata/<scene_id>.parquet``, column ``artist``), by file key.
    """
    if "author" in views:
        return normalise_author(views["author"]).to_numpy()
    if "scene_id" not in views:
        raise ValueError("the view table has neither an author nor a scene_id column to find its authors by")
    from timemachine.config import DataPaths
    authors = pd.Series("", index=views.index, dtype=object)
    for scene_id in views["scene_id"].unique():
        path = DataPaths().image_metadata_table(int(scene_id))
        if not path.exists():
            raise FileNotFoundError(f"no Commons metadata table {path} to read the authors of scene {scene_id} from")
        meta = pd.read_parquet(path, columns=["file_key", "artist"]).drop_duplicates("file_key")
        artist = dict(zip(meta["file_key"], normalise_author(meta["artist"])))
        here = views["scene_id"] == scene_id
        authors[here] = views.loc[here, "file_key"].map(artist).fillna("")
    return authors.to_numpy()


def unit_labels(views: pd.DataFrame, authors: np.ndarray) -> np.ndarray:
    """``[N]`` str: ``"D:<yyyy-mm-dd>"`` for a view dated to a day (the calendar day of ``date_mid``), else
    ``"I:<author>|<date_lo>|<date_hi>"`` (an unknown author is the empty string)."""
    lo, hi = views["date_lo"].to_numpy(np.float64), views["date_hi"].to_numpy(np.float64)
    mid = views["date_mid"].to_numpy(np.float64)
    return np.array([f"D:{calendar_day(m)}" if h - l <= DAY else f"I:{a}|{l:.4f}|{h:.4f}"
                     for l, h, m, a in zip(lo, hi, mid, authors)], dtype=object)


def holdout_units(views: pd.DataFrame, edges: Sequence[float] = (), authors: np.ndarray | None = None,
                  day_only_periods: Sequence[int] = (0,)) -> np.ndarray:
    """``[N]`` int64: each view's hold-out unit.

    In every period (``edges``) but those of ``day_only_periods``, a unit is a connected component of the
    period's usable views (the ``usable`` column; all views without one), linked when they have the same
    :func:`unit_labels` label (the same calendar day, or for views not dated to a day the same author and
    date interval) or the same known author. In ``day_only_periods`` (Notre-Dame's old photos, as the look1
    critique held them out) and for unusable views a unit is the label itself. ``authors`` defaults to
    :func:`view_authors`.

    Units are numbered in the order of their labels, ``"C<period>:<root>"`` for a component (the root of a
    union-find over the rows in order, days linked first, then authors) and the label elsewhere: this
    order decides how a seed shuffles them (``heldout_rows``), and it is the look1 critique's
    (``data/look1_critique/feasibility/01e_holdout_by_component.py``) and ``scripts/mvgen/make_eval_sets.py``'s.
    """
    n = len(views)
    usable = views["usable"].to_numpy(bool) if "usable" in views else np.ones(n, dtype=bool)
    authors = view_authors(views) if authors is None else np.asarray(authors, dtype=object)
    period = era_index(views["date_mid"].to_numpy(np.float64), tuple(edges))
    labels = unit_labels(views, authors)
    days = labels.copy()
    for p in sorted(set(range(len(edges) + 1)) - {int(q) for q in day_only_periods}):
        rows = np.flatnonzero(usable & (period == p))
        parent = {row: row for row in rows}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for keys in (days, authors):
            first: dict[str, int] = {}
            for row in rows:
                if keys is authors and not keys[row]:
                    continue                                # an unknown author links nothing by itself
                if keys[row] in first:
                    a, b = find(row), find(first[keys[row]])
                    if a != b:
                        parent[a] = b
                else:
                    first[keys[row]] = row
        labels[rows] = [f"C{p}:{find(row)}" for row in rows]
    return np.unique(labels.astype(str), return_inverse=True)[1].astype(np.int64)


# ----------------------------------------------------------------------------
# Evaluation sets (the 2 x 2 of reference period x target period) and the seed rule
# ----------------------------------------------------------------------------
def evaluation_sets(target_period: int, reference_period: int, *, heldout: np.ndarray, usable: np.ndarray,
                    ref_ok: np.ndarray, dates: np.ndarray, period: np.ndarray, covis: np.ndarray, side: np.ndarray,
                    rng: np.random.Generator, tolerance: float | None = None, n_ref: int = 3, n_targets: int = 5,
                    min_scored: int = EVAL_MIN_SCORED, min_shared: int = 30, top_k: int = 24,
                    tries: int = EVAL_SET_TRIES, reference_tries: int = EVAL_REFERENCE_TRIES
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Candidate evaluation sets of one 2 x 2 cell, in the order they are drawn: ``(references, targets)`` rows,
    the anchor the first target.

    Args:
        heldout, usable, ref_ok, side: bool ``[N]``: the held-out views, the usable ones, those that may be
            references (reliable depth) and the side views.
        dates, period: ``[N]`` date midpoints and periods.
        covis: ``[N, N]`` COLMAP points shared by two views.
        rng: drawn from in a fixed order; the same generator state gives the same sets.
        tolerance: years around the anchor's date within which its targets lie (default ``EVAL_TOLERANCES``).
        min_scored: held-out targets per set, the anchor included (``n_targets``: strict sets).

    For each held-out side view of ``target_period`` as anchor: its partners are the usable views of
    the period within ``tolerance`` that share ``min_shared`` points with it; ``tries`` times,
    ``min_scored - 1`` held-out partners (among its ``top_k`` most co-visible partners when enough of them
    are held out) and the remaining targets from those ``top_k``; the targets that are held out are
    scored. The references are ``n_ref`` training views of ``reference_period`` (with ``ref_ok``) that
    share ``min_shared`` points with every scored target, drawn in proportion to how many of the targets
    they see until, within ``reference_tries`` draws, together they see every target.
    """
    heldout, usable, side = (np.asarray(x, dtype=bool) for x in (heldout, usable, side))
    tolerance = EVAL_TOLERANCES.get(target_period, 1.0) if tolerance is None else tolerance
    references = np.asarray(ref_ok, dtype=bool) & usable & ~heldout & (period == reference_period)
    in_period = usable & (period == target_period)
    pool_targets = in_period & (heldout | (min_scored < n_targets))
    sets = []
    for anchor in np.flatnonzero(in_period & heldout & side):
        partners = (np.abs(dates - dates[anchor]) <= tolerance) & pool_targets & (covis[anchor] >= min_shared)
        partners[anchor] = False
        partners = np.flatnonzero(partners)
        held = partners[heldout[partners]]
        if len(partners) < n_targets - 1 or len(held) < min_scored - 1:
            continue
        top = partners[np.argsort(-covis[anchor, partners].astype(np.int64), kind="stable")[:top_k]]
        held_top = np.intersect1d(top, held)
        held_pool = held_top if len(held_top) >= min_scored - 1 else held
        for _ in range(tries):
            first = rng.choice(held_pool, min_scored - 1, replace=False)
            rest_pool = np.setdiff1d(top, first)
            if len(rest_pool) < n_targets - min_scored:
                break
            rest = rng.choice(rest_pool, n_targets - min_scored, replace=False)
            targets = np.concatenate([[anchor], first, rest]).astype(np.int64)
            sees = covis[:, targets] >= min_shared
            candidates = references & sees[:, heldout[targets]].all(axis=1)
            candidates[targets] = False
            pool = np.flatnonzero(candidates)
            if len(pool) < n_ref or not sees[pool].any(axis=0).all():
                continue
            weights = sees[pool].sum(axis=1).astype(np.float64)
            weights /= weights.sum()
            for _ in range(reference_tries):
                refs = rng.choice(pool, n_ref, replace=False, p=weights)
                if sees[refs].any(axis=0).all():
                    sets.append((refs.astype(np.int64), targets))
                    break
    return sets


def disjoint_sets(sets: Sequence[tuple[np.ndarray, np.ndarray]], heldout: np.ndarray) -> list[int]:
    """Indices of a greedy packing of ``sets`` in their order: a set is kept if none of its scored (held-out)
    targets is a scored target of a set kept before it."""
    heldout = np.asarray(heldout, dtype=bool)
    used: set[int] = set()
    kept = []
    for index, (_, targets) in enumerate(sets):
        scored = {int(t) for t in targets if heldout[t]}
        if used.isdisjoint(scored):
            kept.append(index)
            used |= scored
    return kept


def evaluation_cells(seed: int, periods: Sequence[int] = (1, 2), **arrays) -> dict[tuple[int, int], list]:
    """The disjoint evaluation sets of every cell under hold-out seed ``seed``: ``{(reference period, target
    period): [(references, targets), ...]}``.

    The cells are drawn in the order target period, then reference period, from ONE generator,
    ``np.random.default_rng(EVAL_SEED_OFFSET + seed)`` (the look1 critique's order); each cell's candidates
    (:func:`evaluation_sets`, with ``arrays`` its keyword arguments but ``rng``) are packed by
    :func:`disjoint_sets`.
    """
    rng = np.random.default_rng(EVAL_SEED_OFFSET + int(seed))
    cells = {}
    for target in periods:
        for reference in periods:
            sets = evaluation_sets(target, reference, rng=rng, **arrays)
            cells[reference, target] = [sets[i] for i in disjoint_sets(sets, arrays["heldout"])]
    return cells


def first_holdout_seed(cfg: HoldoutConfig, *, dates: np.ndarray, units: np.ndarray, usable: np.ndarray,
                       side: np.ndarray, ref_ok: np.ndarray, covis: np.ndarray, periods: Sequence[int] = (1, 2),
                       seeds: Sequence[int] = range(50), min_sets: int = EVAL_MIN_SETS,
                       max_share: float = EVAL_MAX_SHARE) -> tuple[int | None, list[dict]]:
    """The first seed of ``seeds`` under which ``cfg`` (by component) meets the seed rule, and a record per seed tried.

    A record holds the seed, per period the held-out views and their share of the period's usable
    views and, for a seed within ``max_share``, per cell (``"<reference period>-><target period>"``, over
    ``periods``) its disjoint evaluation sets (:func:`evaluation_cells`). ``(None, records)`` when no seed
    meets the rule.
    """
    dates = np.asarray(dates, dtype=np.float64)
    period = era_index(dates, cfg.era_edges)
    usable = np.asarray(usable, dtype=bool)
    records = []
    for seed in seeds:
        held = np.zeros(len(dates), dtype=bool)
        held[heldout_rows(dates, replace(cfg, seed=int(seed)), units=units, usable=usable, side=side)] = True
        record: dict = {"seed": int(seed)}
        shares = []
        for p in range(len(cfg.era_edges) + 1):
            n_period = int((usable & (period == p)).sum())
            n_held = int((held & usable & (period == p)).sum())
            shares.append(n_held / max(n_period, 1))
            record[f"held_{p}"], record[f"share_{p}"] = n_held, shares[-1]
        records.append(record)
        if max(shares) > max_share:
            continue
        cells = evaluation_cells(seed, periods, heldout=held, usable=usable, ref_ok=ref_ok, dates=dates,
                                 period=period, covis=covis, side=side)
        record.update({f"{reference}->{target}": len(sets) for (reference, target), sets in cells.items()})
        if min(len(sets) for sets in cells.values()) >= min_sets:
            return int(seed), records
    return None, records
