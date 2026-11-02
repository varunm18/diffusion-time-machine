"""Retrieval baselines: "don't generate anything, just copy a real photo".

WHY THIS FILE EXISTS
--------------------
Our proposal promises a nearest-dated-real-image baseline. Most scenes barely
change over the years, so simply handing back a real photo from about the right
date can already look surprisingly close to the truth. If our date-conditioned
model cannot beat that, it hasn't really learned how the scene changes.

It also acts as a sanity check on the metrics. Laith's plain-SEVA baseline scores
about 8 to 9 dB PSNR, which says little by itself. Seeing what a model-free
"copy a real photo" scores puts that number in context.

THE THREE BASELINES
-------------------
All three answer the same question: "for this target view, which existing real
photo would I hand back as my prediction?"

    input_date      From the sample's 3 INPUT views, take the one whose date is
                    nearest the target's date (ties broken by camera). This is
                    exactly what a date-aware model sees, so it is the fairest bar.
    input_pose      From the same inputs, take the one with the nearest CAMERA and
                    ignore dates. A date-blind control: if a model beats
                    input_date but not this, its "date awareness" is just
                    geometry in disguise.
    pool_date_pose  From the WHOLE scene, shortlist the photos nearest in date,
                    then take the one with the nearest camera. It sees more
                    photos than any model would at inference, so treat it as a
                    strong reference, not a fair competitor. See the caveat below.

IMPORTANT CAVEAT: LEAKAGE
-------------------------
The repo has no train/validation split yet. Until it does, `pool_date_pose` can
pick a near-identical neighbor (another frame from the same photographer's burst)
that a real held-out test would never allow, so its scores are optimistic. Pass
`pool=` with the training split's rows once splits exist. Retrieval never returns
another TARGET of the same sample (see predict_sample), which stops the most
obvious leak.

This module deliberately does not import torch, so the selection logic can be
tested anywhere. The script scripts/retrieval_baseline.py does the image loading.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

# strategy name -> how candidates are ranked; source -> where candidates come from
BASELINES: dict[str, tuple[str, str]] = {
    "input_date": ("date", "inputs"),
    "input_pose": ("pose", "inputs"),
    "pool_date_pose": ("date_then_pose", "pool"),
}

# Buckets for "how far apart in time is the retrieved photo from the target?"
GAP_BUCKETS = ((0.0, 2.0, "<2y"), (2.0, 10.0, "2-10y"), (10.0, 50.0, "10-50y"), (50.0, np.inf, ">50y"))


# ----------------------------------------------------------------------------
# Camera geometry: how "close" are two views?
# ----------------------------------------------------------------------------

def camera_features(c2ws: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Camera centers, viewing directions and a scene scale from camera-to-world matrices.

    `c2ws` is (N, 4, 4) in the repo's convention: OpenCV axes, so a camera looks
    along its +z axis, which is the third column of the rotation part.

    The scale is the median distance of the cameras from their median position.
    COLMAP's units are arbitrary per scene, so we measure distances in "scene
    widths" instead. (Same idea as `neighbor_rows` in scripts/seva_baseline.py.)
    """
    c2ws = np.asarray(c2ws, dtype=np.float64)
    centers = c2ws[:, :3, 3]
    directions = c2ws[:, :3, 2]
    directions = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    spread = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
    scale = float(np.median(spread)) or 1.0   # `or 1.0` guards a degenerate single-position scene
    return centers, directions, scale


def pose_distance(centers: np.ndarray, directions: np.ndarray, scale: float,
                  target: int, candidates: np.ndarray) -> np.ndarray:
    """How far each candidate camera is from the target camera (smaller = more similar).

    Two dimensionless parts added together:
      * position: distance between camera centers, in scene widths;
      * direction: angle between viewing directions, as a fraction of 180 degrees.
    Both are roughly 0 to 1 for typical views, so neither swamps the other.
    """
    position = np.linalg.norm(centers[candidates] - centers[target], axis=1) / scale
    cosine = np.clip(directions[candidates] @ directions[target], -1.0, 1.0)
    angle = np.arccos(cosine) / np.pi
    return position + angle


# ----------------------------------------------------------------------------
# Choosing a photo
# ----------------------------------------------------------------------------

def retrieve(strategy: str, target: int, candidates: Sequence[int], dates: np.ndarray,
             centers: np.ndarray, directions: np.ndarray, scale: float, k_date: int = 20) -> int:
    """Pick the one candidate row to hand back for `target`.

    strategy
      "date"            nearest date wins; a tie goes to the nearer camera
      "pose"            nearest camera wins; a tie goes to the nearer date
      "date_then_pose"  keep the `k_date` candidates nearest in date, then take the
                        nearest camera among them
    """
    candidates = np.asarray(candidates, dtype=int)
    if candidates.size == 0:
        raise ValueError("No candidates to retrieve from.")

    gap = np.abs(dates[candidates] - dates[target])
    pose = pose_distance(centers, directions, scale, target, candidates)

    # np.lexsort sorts by its LAST key first, so (secondary, primary).
    if strategy == "date":
        best = np.lexsort((pose, gap))[0]
    elif strategy == "pose":
        best = np.lexsort((gap, pose))[0]
    elif strategy == "date_then_pose":
        shortlist = np.argsort(gap, kind="stable")[:k_date]
        best = shortlist[np.argmin(pose[shortlist])]
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")
    return int(candidates[best])


def predict_sample(baseline: str, inputs: Sequence[int], targets: Sequence[int], dates: np.ndarray,
                   centers: np.ndarray, directions: np.ndarray, scale: float,
                   pool: Sequence[int] | None = None, k_date: int = 20) -> list[int]:
    """For one sample, the row that `baseline` returns for each target (in target order).

    "inputs" baselines choose among the sample's input views. "pool" baselines
    choose among `pool` (default: every view in the table). In both cases ALL the
    sample's targets are removed first: the targets share one date and often one
    part of the scene, so a sibling target would be a near-perfect copy of the
    answer and would make the baseline look unrealistically good.
    """
    strategy, source = BASELINES[baseline]
    if source == "inputs":
        candidates = np.asarray(inputs, dtype=int)
    else:
        everything = np.arange(len(dates)) if pool is None else np.asarray(pool, dtype=int)
        candidates = np.setdiff1d(everything, np.asarray(targets, dtype=int))
    return [retrieve(strategy, int(t), candidates, dates, centers, directions, scale, k_date)
            for t in targets]


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two images with values in [-1, 1].

    Same formula as `psnr` in scripts/seva_baseline.py, so the two baselines are
    scored identically. (Dividing the difference by 2 maps the [-1, 1] range to
    [0, 1], where the peak value is 1.)
    """
    mse = float(np.mean(((np.asarray(a) - np.asarray(b)) / 2.0) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-10)))


def gap_bucket(gap_years: float) -> str:
    """Label for how many years separate the retrieved photo from the target."""
    for low, high, label in GAP_BUCKETS:
        if low <= gap_years < high:
            return label
    return GAP_BUCKETS[-1][2]


def run_baselines(dataset, n_samples: int, load_image: Callable[[pd.Series], np.ndarray],
                  baselines: Iterable[str] = tuple(BASELINES), k_date: int = 20,
                  pool: Sequence[int] | None = None) -> pd.DataFrame:
    """Score the retrieval baselines on the same samples the dataset would draw.

    `dataset` is a DatedViewsDataset (or anything with `.views`, `.dates`, `.c2ws`,
    `.config.n_inputs` and `.indices(i)`). Because sampling is deterministic per
    (seed, index), using the same dataset settings as scripts/seva_baseline.py
    gives the SAME samples, so the two CSVs line up row for row.

    `load_image(row)` returns one view as a (C, H, W) array in [-1, 1]. It is passed
    in, so tests need no real images and this module needs no torch.

    One output row per (target view, baseline).
    """
    centers, directions, scale = camera_features(dataset.c2ws)
    dates = np.asarray(dataset.dates, dtype=np.float64)
    views = dataset.views
    n_inputs = dataset.config.n_inputs
    out: list[dict] = []

    for i in range(n_samples):
        rows = np.asarray(dataset.indices(i))
        inputs, targets = rows[:n_inputs], rows[n_inputs:]
        old_target_sample = bool(dates[targets].max() < 1970)   # same rule as seva_baseline.py
        input_years = [round(float(dates[j]), 1) for j in inputs]

        cache: dict[int, np.ndarray] = {}   # each photo is loaded once per sample

        def image(row_index: int) -> np.ndarray:
            if row_index not in cache:
                cache[row_index] = load_image(views.iloc[row_index])
            return cache[row_index]

        for name in baselines:
            chosen = predict_sample(name, inputs, targets, dates, centers, directions, scale, pool, k_date)
            for k, (t, r) in enumerate(zip(targets, chosen)):
                out.append({
                    "sample": i,
                    "view": n_inputs + k,                    # position in the sample, as in seva_baseline.py
                    "file_key": views.iloc[t]["file_key"],
                    "target_year": float(dates[t]),
                    "old_target_sample": old_target_sample,
                    "input_years": input_years,
                    "baseline": name,
                    "retrieved_file_key": views.iloc[r]["file_key"],
                    "retrieved_year": float(dates[r]),
                    "date_gap": float(abs(dates[r] - dates[t])),
                    "pose_dist": float(pose_distance(centers, directions, scale, int(t), np.array([r]))[0]),
                    "psnr": psnr_np(image(int(r)), image(int(t))),
                })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def summarize(table: pd.DataFrame) -> dict:
    """Mean PSNR per baseline: overall, by old-vs-modern targets, and by retrieved date gap."""
    table = table.assign(gap_bucket=table["date_gap"].map(gap_bucket))
    return {
        "psnr_mean": table.groupby("baseline")["psnr"].mean().round(3).to_dict(),
        "psnr_by_target_group": {
            name: grp.groupby("old_target_sample")["psnr"].mean()
                     .rename({True: "old_targets", False: "modern_targets"}).round(3).to_dict()
            for name, grp in table.groupby("baseline")
        },
        "psnr_by_date_gap": {
            name: grp.groupby("gap_bucket")["psnr"].mean().round(3).to_dict()
            for name, grp in table.groupby("baseline")
        },
        "n_targets": int(table.groupby("baseline").size().iloc[0]) if len(table) else 0,
    }


def compare_with_seva(table: pd.DataFrame, seva: pd.DataFrame) -> dict:
    """Retrieval vs plain SEVA on the targets both scored (matched on sample and view).

    Only meaningful if both runs used the same dataset, seed and sampling config.
    `seva` is the psnr.csv written by scripts/seva_baseline.py. We also match on
    the target's file, so a run with a different seed or config matches nothing
    instead of silently comparing different targets.
    """
    seva = seva[["sample", "view", "file_key", "psnr"]].rename(columns={"psnr": "seva_psnr"})
    merged = table.merge(seva, on=["sample", "view", "file_key"], how="inner")
    result = {}
    for name, grp in merged.groupby("baseline"):
        result[name] = {
            "matched_targets": int(len(grp)),
            "retrieval_psnr": round(float(grp["psnr"].mean()), 3),
            "seva_psnr": round(float(grp["seva_psnr"].mean()), 3),
            "retrieval_beats_seva": round(float((grp["psnr"] > grp["seva_psnr"]).mean()), 3),
        }
    return result
