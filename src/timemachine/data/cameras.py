"""Camera conventions shared by the dataset and (later) the model.

Conventions (matching Stable Virtual Camera / SEVA):

* ``c2w``: 4x4 camera-to-world matrix, OpenCV axes (x right, y down, z forward).
  COLMAP stores world-to-camera ``(q, t)``; :func:`colmap_to_c2w` converts.
* Intrinsics ``K`` are 3x3. SEVA wants them *normalized*: the first row divided
  by the image width and the second by the height, so the principal point is
  in ``[0, 1]`` regardless of resolution (:func:`normalize_intrinsics`).
* Scene scale: COLMAP scale is arbitrary, so SEVA re-centres cameras on the mean
  camera position (ignoring far outliers) and rescales so the reference camera
  sits at distance ``camera_scale`` (2.0 by default) from the centre
  (:func:`normalize_cameras`, a port of ``seva/eval.py``).
* Plücker rays are computed relative to the reference camera (frame 0), as in
  ``seva/geometry.py::get_plucker_coordinates`` (:func:`plucker_rays`).
"""
from __future__ import annotations

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Conversions
# ----------------------------------------------------------------------------
def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) quaternion(s) -> rotation matrix/matrices, shape (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def colmap_to_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP world-to-camera (q, t) -> camera-to-world 4x4 (batched over leading dims)."""
    R = quat_to_rotmat(qvec)
    t = np.asarray(tvec, dtype=np.float64)
    c2w = np.zeros((*R.shape[:-2], 4, 4))
    c2w[..., :3, :3] = np.swapaxes(R, -1, -2)
    c2w[..., :3, 3] = -np.einsum("...ji,...j->...i", R, t)
    c2w[..., 3, 3] = 1.0
    return c2w


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------------
# Image-space edits (keep K consistent with resized/cropped images)
# ----------------------------------------------------------------------------
def resize_and_center_crop_params(width: int, height: int, size: int) -> tuple[float, int, int]:
    """Scale factor and crop offsets for "resize short side to ``size``, centre crop".

    Returns ``(scale, left, top)`` in resized-image pixels.
    """
    scale = size / min(width, height)
    new_w, new_h = round(width * scale), round(height * scale)
    return scale, (new_w - size) // 2, (new_h - size) // 2


def adjust_intrinsics(K: np.ndarray, scale_x: float, scale_y: float,
                      left: float = 0.0, top: float = 0.0) -> np.ndarray:
    """Intrinsics after scaling an image by (scale_x, scale_y) and cropping at (left, top)."""
    K = K.copy()
    K[0, :] *= scale_x
    K[1, :] *= scale_y
    K[0, 2] -= left
    K[1, 2] -= top
    return K


def normalize_intrinsics(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resolution-independent intrinsics (SEVA's expected format)."""
    K = K.copy()
    K[0, :] /= width
    K[1, :] /= height
    return K


# ----------------------------------------------------------------------------
# Scene normalization (port of seva/eval.py)
# ----------------------------------------------------------------------------
def normalize_cameras(
    c2ws: np.ndarray, reference_c2ws: np.ndarray | None = None, camera_scale: float = 2.0,
) -> np.ndarray:
    """Re-centre and rescale a set of cameras the way SEVA does at inference.

    ``c2ws[0]`` is the reference (source) camera. The centre is the mean
    position of ``reference_c2ws`` (defaults to ``c2ws``) after dropping cameras
    farther from the median than 10x the 97th-percentile distance. Translations
    are then scaled so camera 0 lies at distance ``camera_scale`` from the centre.
    """
    c2ws = np.array(c2ws, dtype=np.float64, copy=True)
    ref = c2ws if reference_c2ws is None else np.asarray(reference_c2ws, dtype=np.float64)
    centers = ref[:, :3, 3]
    dist = np.linalg.norm(centers - np.median(centers, axis=0, keepdims=True), axis=-1)
    inliers = dist <= min(np.quantile(dist, 0.97) * 10, 1e6)
    c2ws[:, :3, 3] -= centers[inliers].mean(axis=0)
    ref_dist = np.linalg.norm(c2ws[0, :3, 3])
    if ref_dist > 1e-5:
        c2ws[:, :3, 3] *= camera_scale / ref_dist
    else:
        c2ws[:, :3, 3] *= camera_scale
    return c2ws


# ----------------------------------------------------------------------------
# Plücker rays (port of seva/geometry.py::get_plucker_coordinates)
# ----------------------------------------------------------------------------
def plucker_rays(c2ws: torch.Tensor, K_norm: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Per-pixel Plücker coordinates ``(d, o x d)`` relative to camera 0.

    Args:
        c2ws: (V, 4, 4) camera-to-world matrices.
        K_norm: (V, 3, 3) normalized intrinsics (see :func:`normalize_intrinsics`).
        height, width: output grid size (SEVA uses the latent size, e.g. 72x72).

    Returns:
        (V, 6, height, width) tensor: unit ray directions, then moments.
    """
    # Express all cameras in the frame of camera 0.
    rel = torch.linalg.inv(c2ws[0])[None] @ c2ws                         # (V, 4, 4)
    K = K_norm.clone()
    K[:, 0] *= width
    K[:, 1] *= height
    # Pixel centres in homogeneous image coordinates.
    ys, xs = torch.meshgrid(torch.arange(height, dtype=c2ws.dtype) + 0.5,
                            torch.arange(width, dtype=c2ws.dtype) + 0.5, indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).reshape(-1, 3)   # (HW, 3)
    cam_dirs = torch.einsum("vij,nj->vni", torch.linalg.inv(K), pix)     # (V, HW, 3)
    dirs = torch.einsum("vij,vnj->vni", rel[:, :3, :3], cam_dirs)
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    origins = rel[:, None, :3, 3].expand_as(dirs)
    plucker = torch.cat([dirs, torch.cross(origins, dirs, dim=-1)], dim=-1)  # (V, HW, 6)
    return plucker.permute(0, 2, 1).reshape(c2ws.shape[0], 6, height, width)
