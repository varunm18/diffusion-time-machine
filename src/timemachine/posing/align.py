"""Aligning predicted cameras to an existing reconstruction, and measuring pose error.

A feed-forward model (VGGT-Omega) predicts cameras in its own frame (first
camera at the origin, arbitrary scale). To place new images into a MegaScenes
COLMAP model we run them together with *anchor* images whose COLMAP poses are
known, fit a similarity transform (scale, rotation, translation) mapping the
predicted anchor camera centres onto their COLMAP centres, and apply it to the
new images. RANSAC makes the fit robust to anchors the model got wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sim3:
    """x -> scale * R @ x + t."""

    scale: float
    R: np.ndarray
    t: np.ndarray

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        return self.scale * points @ self.R.T + self.t

    def apply_c2w(self, c2w: np.ndarray) -> np.ndarray:
        """Move camera-to-world matrices (..., 4, 4) into the target frame."""
        out = np.array(c2w, dtype=np.float64, copy=True)
        out[..., :3, :3] = self.R @ c2w[..., :3, :3]
        out[..., :3, 3] = self.apply_points(c2w[..., :3, 3])
        return out


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares similarity transform with ``dst ~ s R src + t`` (Umeyama 1991)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = (xs ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(S) @ D) / var_s) if with_scale and var_s > 0 else 1.0
    return Sim3(scale, R, mu_d - scale * R @ mu_s)


def ransac_umeyama(
    src: np.ndarray, dst: np.ndarray, rng: np.random.Generator,
    iterations: int = 500, threshold: float = 0.05,
) -> tuple[Sim3, np.ndarray]:
    """Robust :func:`umeyama`.

    ``threshold`` is relative: an anchor is an inlier if its aligned centre lies
    within ``threshold * scene_size`` of its target, where ``scene_size`` is the
    median distance of the targets to their centroid. Returns the transform
    refit on all inliers, and the inlier mask.
    """
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if len(src) < 3:
        raise ValueError("Need at least 3 anchors to fit a similarity transform.")
    scene_size = np.median(np.linalg.norm(dst - dst.mean(0), axis=1)) or 1.0
    tol = threshold * scene_size
    best = np.zeros(len(src), dtype=bool)
    for _ in range(iterations):
        pick = rng.choice(len(src), 3, replace=False)
        try:
            model = umeyama(src[pick], dst[pick])
        except np.linalg.LinAlgError:
            continue
        inliers = np.linalg.norm(model.apply_points(src) - dst, axis=1) < tol
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 3:
        return umeyama(src, dst), best
    return umeyama(src[best], dst[best]), best


def rotation_first_alignment(
    pred_c2w: np.ndarray, target_c2w: np.ndarray, rng: np.random.Generator,
    rotation_threshold_deg: float = 10.0, center_threshold: float = 0.15, iterations: int = 300,
) -> tuple[Sim3, np.ndarray]:
    """Similarity transform fitted rotation-first, for when predicted *positions* are noisy.

    Feed-forward models predict orientations more reliably than positions for
    wide-baseline images. Each anchor implies a global rotation
    ``R_i = R_target_i @ R_pred_i^T``; we take the largest set of anchors whose
    implied rotations agree within ``rotation_threshold_deg``, average them, and
    only then fit scale and translation to the camera centres (RANSAC over
    anchor pairs, inlier if within ``center_threshold * scene_size``).

    Returns the transform and the inlier mask (agrees on rotation *and* position).
    """
    R_pred, R_tgt = pred_c2w[:, :3, :3], target_c2w[:, :3, :3]
    implied = R_tgt @ np.swapaxes(R_pred, -1, -2)                       # (N, 3, 3)
    agree = rotation_error_deg(implied[:, None], implied[None, :]) < rotation_threshold_deg
    rot_inliers = agree[np.argmax(agree.sum(axis=1))]
    R = mean_rotation(implied[rot_inliers])

    src = pred_c2w[:, :3, 3] @ R.T                                       # rotated predicted centres
    dst = target_c2w[:, :3, 3]
    size = np.median(np.linalg.norm(dst - dst.mean(0), axis=1)) or 1.0
    candidates = np.flatnonzero(rot_inliers)

    def fit(idx: np.ndarray) -> tuple[float, np.ndarray]:
        xs, xd = src[idx] - src[idx].mean(0), dst[idx] - dst[idx].mean(0)
        scale = float((xs * xd).sum() / max((xs ** 2).sum(), 1e-12))
        return scale, dst[idx].mean(0) - scale * src[idx].mean(0)

    best = np.zeros(len(src), dtype=bool)
    if len(candidates) >= 2:
        for _ in range(iterations):
            pair = rng.choice(candidates, 2, replace=False)
            scale, t = fit(pair)
            if scale <= 0:
                continue
            inliers = rot_inliers & (np.linalg.norm(scale * src + t - dst, axis=1) < center_threshold * size)
            if inliers.sum() > best.sum():
                best = inliers
    if best.sum() >= 2:
        scale, t = fit(np.flatnonzero(best))
    else:
        scale, t = fit(candidates) if len(candidates) >= 2 else (1.0, np.zeros(3))
    return Sim3(max(scale, 1e-12), R, t), best


# ----------------------------------------------------------------------------
# Error metrics
# ----------------------------------------------------------------------------
def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Angle (degrees) of the relative rotation between (batches of) rotations."""
    rel = np.swapaxes(R_a, -1, -2) @ R_b
    cos = (np.trace(rel, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def mean_rotation(Rs: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of rotations (average the matrices, project back onto SO(3))."""
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt
