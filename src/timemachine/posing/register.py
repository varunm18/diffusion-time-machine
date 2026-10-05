"""Register new images (e.g. old photos) into an existing COLMAP model.

Procedure (see :mod:`.align` and :mod:`.batches`):

1. Split the queries into batches; add diverse anchors (images with known
   COLMAP poses) to every batch; each query appears in ``repeats`` batches.
2. Predict all cameras of a batch with a feed-forward model.
3. Fit a robust similarity transform from predicted to COLMAP anchor centres;
   skip the batch if too few anchors agree.
4. Map the queries' predicted cameras into the COLMAP frame.
5. Average each query's poses over its batches; the spread across batches is a
   confidence signal (a query the model cannot place lands in different spots).

The predictor is injected (any ``paths -> {"c2w", "focal"}`` function), so the
procedure is testable without a GPU.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from timemachine.posing.align import mean_rotation, ransac_umeyama, rotation_error_deg
from timemachine.posing.batches import make_batches

PredictFn = Callable[[list[str]], dict[str, np.ndarray]]


@dataclass(frozen=True)
class RegistrationConfig:
    queries_per_batch: int = 32
    anchors_per_batch: int = 32
    repeats: int = 3
    ransac_threshold: float = 0.05   # fraction of the scene size
    min_inliers: int = 8             # anchors that must agree for a batch to count
    seed: int = 0


def scene_size(c2ws: np.ndarray) -> float:
    """Median distance of camera centres to their median: the unit for position errors."""
    centers = c2ws[:, :3, 3]
    return float(np.median(np.linalg.norm(centers - np.median(centers, 0), axis=1))) or 1.0


def register_images(
    predict: PredictFn, query_paths: list[str], anchor_paths: list[str], anchor_c2ws: np.ndarray,
    config: RegistrationConfig = RegistrationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pose ``query_paths`` in the frame of ``anchor_c2ws``.

    Returns ``(queries, batches)``:
    * ``queries``: one row per query with ``c2w`` (4x4, or None if never placed),
      ``focal`` (pixels), ``n_batches`` used, ``rot_spread_deg`` and
      ``center_spread`` (relative to the scene size) across batches.
    * ``batches``: one row per batch with alignment diagnostics.
    """
    rng = np.random.default_rng(config.seed)
    size = scene_size(anchor_c2ws)
    poses: dict[int, list[np.ndarray]] = defaultdict(list)
    focals: dict[int, list[float]] = defaultdict(list)
    log = []

    for b, (qi, ai) in enumerate(make_batches(len(query_paths), anchor_c2ws, rng, config.queries_per_batch,
                                              config.anchors_per_batch, config.repeats)):
        pred = predict([query_paths[i] for i in qi] + [anchor_paths[j] for j in ai])
        nq = len(qi)
        sim, inliers = ransac_umeyama(pred["c2w"][nq:, :3, 3], anchor_c2ws[ai, :3, 3], rng,
                                      threshold=config.ransac_threshold)
        aligned = sim.apply_c2w(pred["c2w"])
        anchor_rot = rotation_error_deg(aligned[nq:, :3, :3], anchor_c2ws[ai, :3, :3])
        used = int(inliers.sum()) >= config.min_inliers
        log.append({"batch": b, "n_queries": nq, "n_anchors": len(ai), "n_inliers": int(inliers.sum()),
                    "anchor_rot_err_median": float(np.median(anchor_rot[inliers])) if inliers.any() else np.nan,
                    "used": used})
        if not used:
            continue
        for k, q in enumerate(qi):
            poses[q].append(aligned[k])
            focals[q].append(float(pred["focal"][k]))

    rows = []
    for q in range(len(query_paths)):
        if not poses[q]:
            rows.append({"query": q, "c2w": None, "focal": np.nan, "n_batches": 0,
                         "rot_spread_deg": np.nan, "center_spread": np.nan})
            continue
        stack = np.stack(poses[q])
        R = mean_rotation(stack[:, :3, :3])
        center = np.median(stack[:, :3, 3], axis=0)
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, center
        rows.append({
            "query": q, "c2w": c2w, "focal": float(np.median(focals[q])), "n_batches": len(stack),
            "rot_spread_deg": float(rotation_error_deg(stack[:, :3, :3], R).max()),
            "center_spread": float(np.linalg.norm(stack[:, :3, 3] - center, axis=1).max() / size),
        })
    return pd.DataFrame(rows), pd.DataFrame(log)


def evaluate_registration(queries: pd.DataFrame, gt_c2ws: np.ndarray, size: float) -> pd.DataFrame:
    """Per-query errors against ground-truth poses (same order as the queries).

    ``rot_err_deg``: angle between predicted and true rotation.
    ``center_err``: camera-centre distance divided by the scene size.
    """
    placed = queries["c2w"].notna().to_numpy()
    out = queries.copy()
    out["rot_err_deg"] = np.nan
    out["center_err"] = np.nan
    if placed.any():
        pred = np.stack(queries.loc[placed, "c2w"].to_list())
        gt = gt_c2ws[placed]
        out.loc[placed, "rot_err_deg"] = rotation_error_deg(pred[:, :3, :3], gt[:, :3, :3])
        out.loc[placed, "center_err"] = np.linalg.norm(pred[:, :3, 3] - gt[:, :3, 3], axis=1) / size
    return out


def summarize_errors(evaluated: pd.DataFrame) -> dict[str, float]:
    """Headline accuracy numbers (fractions are over *all* queries, unplaced count as failures)."""
    rot, ctr = evaluated["rot_err_deg"], evaluated["center_err"]
    n = len(evaluated)
    return {
        "n_queries": n,
        "placed": float(evaluated["c2w"].notna().mean()),
        "rot_err_median_deg": float(rot.median()),
        "within_5deg": float((rot < 5).sum() / n),
        "within_10deg": float((rot < 10).sum() / n),
        "within_20deg": float((rot < 20).sum() / n),
        "center_err_median": float(ctr.median()),
    }
