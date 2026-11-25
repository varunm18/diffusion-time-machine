"""Which scenes should we build next, what will they cost, and what will we actually get?

Everything we have measured is limited by scene count. The shortcut probe and the
appearance-entanglement analysis both ran on four scenes and 58 old photos in the
test split; the training runs use thirteen. Old, posed photographs are the scarce
resource and the only thing that makes any of these numbers tighter.

`docs/census/census_scenes_top800.csv` already says, for 802 scenes, how many
posed photographs each has per era. This module turns that into a build plan: rank
the candidates by what they add, stop at a storage budget, and say up front what
the resulting dataset would look like - including the parts that are awkward.

**The awkward part is why this exists.** Old photographs are not spread evenly
over the world. Across the 140 scenes that pass the posed gate, the top 20 hold
51% of all old posed photos, and eight Amsterdam scenes alone hold 33%. So "old"
and "Amsterdam" are correlated in anything built from this census. That matters
for any per-scene conditioning signal - a location input especially, which on a
small scene set is close to a scene identifier, and which could learn "Amsterdam
means old" rather than anything geographic. :func:`concentration` measures it so a
plan can be judged before it is downloaded rather than after.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# Per-era counts of photographs that MegaScenes' own COLMAP models placed. "posed"
# is the number that matters: an undated or unposed photo cannot become a view.
ERA_COLUMNS = ("posed_pre-1900", "posed_1900-1944", "posed_1945-1969",
               "posed_1970-1999", "posed_2000+")
OLD_COLUMN = "posed_pre-1970"
GATES = {"posed": "gate_posed", "any": "gate_any", "none": None}

# Cached images are JPEG, longest side 1024 (see data/images.py). Measured at ~190 kB
# on the four scenes built so far; calibrate_bytes_per_image() replaces this with the
# real figure whenever a built dataset is available.
DEFAULT_BYTES_PER_IMAGE = 190_000


def load_census(path: str | Path) -> pd.DataFrame:
    """Read a census CSV and check it has what the planner needs."""
    census = pd.read_csv(path)
    missing = [c for c in (*ERA_COLUMNS, OLD_COLUMN, "scene", "scene_id") if c not in census]
    if missing:
        raise ValueError(f"{path} is not a scene census: missing {missing}")
    return census


def calibrate_bytes_per_image(dataset_dirs: Iterable[str | Path]) -> tuple[int, int]:
    """``(bytes_per_image, images_counted)`` measured from already-built datasets.

    Guessing storage is how teams end up with a full quota halfway through a
    download, so measure it from what is already on disk whenever possible.
    """
    total_bytes = total_images = 0
    for folder in dataset_dirs:
        for image in Path(folder).rglob("*.jpg"):
            total_bytes += image.stat().st_size
            total_images += 1
    if not total_images:
        return DEFAULT_BYTES_PER_IMAGE, 0
    return int(round(total_bytes / total_images)), total_images


def rank_candidates(census: pd.DataFrame, gate: str = "posed", exclude: Sequence = (),
                    by: str = OLD_COLUMN) -> pd.DataFrame:
    """Gated scenes, best first, by what they contribute (old posed photos by default).

    ``exclude`` takes scene names or ids already built, so a plan only ever proposes
    new work.
    """
    if gate not in GATES:
        raise ValueError(f"gate must be one of {sorted(GATES)}")
    out = census.copy()
    column = GATES[gate]
    if column is not None:
        if column not in out:
            raise ValueError(f"census has no {column} column; use gate='none'")
        out = out[out[column].astype(bool)]
    drop = {str(e) for e in exclude}
    out = out[~out["scene"].astype(str).isin(drop) & ~out["scene_id"].astype(str).isin(drop)]
    return out.sort_values(by, ascending=False).reset_index(drop=True)


def plan_scenes(census: pd.DataFrame, n_scenes: int | None = None, budget_gb: float | None = None,
                max_modern: int | None = 300, bytes_per_image: int = DEFAULT_BYTES_PER_IMAGE,
                gate: str = "posed", exclude: Sequence = ()) -> pd.DataFrame:
    """A build order: which scenes, in what order, and what each one costs cumulatively.

    ``max_modern`` caps the modern photos taken from each scene, as
    ``build_scene_dataset.py --max-modern`` does. Modern photos are abundant and
    dominate storage; old ones are the point, so they are never capped.

    Stops at ``n_scenes``, at ``budget_gb``, or at the end of the candidates.
    """
    if n_scenes is None and budget_gb is None:
        raise ValueError("Give n_scenes, budget_gb, or both.")
    ranked = rank_candidates(census, gate, exclude)
    if ranked.empty:
        return ranked.assign(images=[], gigabytes=[], cum_gigabytes=[], cum_old=[])

    old = ranked[OLD_COLUMN].to_numpy()
    modern = ranked["best_model_2000+"].to_numpy() if "best_model_2000+" in ranked \
        else ranked["posed_2000+"].to_numpy()
    if max_modern is not None:
        modern = np.minimum(modern, max_modern)
    # Mid-century photos come along with the scene and are neither capped nor the target.
    middle = ranked["posed_1970-1999"].to_numpy() if "posed_1970-1999" in ranked else 0

    images = old + modern + middle
    gigabytes = images * bytes_per_image / 1e9
    plan = ranked.assign(images=images, gigabytes=gigabytes,
                         cum_gigabytes=np.cumsum(gigabytes), cum_old=np.cumsum(old))
    if budget_gb is not None:
        plan = plan[plan["cum_gigabytes"] <= budget_gb]
    if n_scenes is not None:
        plan = plan.head(n_scenes)
    return plan.reset_index(drop=True)


def concentration(values: Sequence[float], tops: Sequence[int] = (1, 5, 10, 20)) -> dict:
    """How unevenly a quantity is spread over scenes: top-k shares and a Gini coefficient.

    A plan whose old photographs come mostly from a handful of scenes will teach a
    model about those scenes, not about age. Gini runs 0 (even) to 1 (one scene has
    everything).
    """
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    total = v.sum()
    if total <= 0:
        return {"total": 0.0, "gini": float("nan"), "top_shares": {}}
    ascending = v[::-1]
    n = len(v)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ ascending / (n * total))
    return {"total": float(total), "gini": gini,
            "top_shares": {k: float(v[:k].sum() / total) for k in tops if k <= n}}


def era_totals(plan: pd.DataFrame, max_modern: int | None = 300) -> pd.Series:
    """Posed photographs per era that a plan would yield, after the modern cap."""
    totals = {}
    for column in ERA_COLUMNS:
        if column not in plan:
            continue
        values = plan[column].to_numpy()
        if column == "posed_2000+" and max_modern is not None:
            values = np.minimum(values, max_modern)
        totals[column.replace("posed_", "")] = int(values.sum())
    return pd.Series(totals)


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.0f}%"


def render_plan(plan: pd.DataFrame, bytes_per_image: int, measured_from: int = 0,
                have: Sequence = (), max_modern: int | None = 300, show: int = 20) -> str:
    """Markdown build plan, ready to paste into a message or docs/."""
    if plan.empty:
        return "No candidate scenes left: everything passing the gate is already built.\n"
    old_conc = concentration(plan[OLD_COLUMN])
    eras = era_totals(plan, max_modern)
    size_note = (f"measured at {bytes_per_image / 1e3:.0f} kB/image over {measured_from:,} "
                 "cached images" if measured_from else
                 f"estimated at {bytes_per_image / 1e3:.0f} kB/image (no built dataset to measure)")
    lines = [
        "## Scene build plan",
        f"*{len(plan)} scenes, **{plan['cum_gigabytes'].iloc[-1]:.1f} GB**, "
        f"**{int(plan['cum_old'].iloc[-1]):,} old posed photos** "
        f"(pre-1970). Modern photos capped at {max_modern} per scene; {size_note}.*",
    ]
    if len(have):
        lines.append(f"*Already built and excluded: {', '.join(str(h) for h in have)}.*")
    lines += [
        "", "| # | scene | old (pre-1970) | images | GB | cum GB | cum old |",
        "|---|---|---|---|---|---|---|",
    ]
    # OLD_COLUMN contains a hyphen, so itertuples would rename it; index by label instead.
    for i, (_, row) in enumerate(plan.head(show).iterrows(), 1):
        lines.append(f"| {i} | {row['scene']} | {int(row[OLD_COLUMN]):,} | {int(row['images']):,} "
                     f"| {row['gigabytes']:.2f} | {row['cum_gigabytes']:.1f} | {int(row['cum_old']):,} |")
    if len(plan) > show:
        lines.append(f"| ... | *{len(plan) - show} more* | | | | "
                     f"{plan['cum_gigabytes'].iloc[-1]:.1f} | {int(plan['cum_old'].iloc[-1]):,} |")
    lines += [
        "", "### What the dataset would contain", "",
        "| era | posed photos |", "|---|---|",
        *[f"| {era} | {count:,} |" for era, count in eras.items()],
        "",
        "### How concentrated the old photos are", "",
        f"Gini **{old_conc['gini']:.2f}** (0 = spread evenly over scenes, 1 = one scene has all of them).",
        "",
        *[f"- the top {k} scene(s) hold **{_pct(share)}** of the old photos"
          for k, share in old_conc["top_shares"].items()],
        "",
        "Old photographs are not spread evenly over the world, so any per-scene signal - a "
        "location input most of all - risks learning *which scene* rather than what we meant. "
        "Worth measuring on the built dataset before conditioning on it.",
    ]
    return "\n".join(lines) + "\n"
