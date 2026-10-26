"""Minimal readers for the COLMAP binary files MegaScenes ships.

We only need cameras (intrinsics) and image poses, so this avoids a pycolmap
dependency. Formats follow COLMAP's ``read_write_model.py``.

``images.minibin`` (MegaScenes web viewer, in ``reconstruct_aux/``) is
``images.bin`` without the per-image 2D keypoint lists: ~1000x smaller for big
models (210 KB vs 211 MB for one Frauenkirche model) and enough for poses.

Pose convention (COLMAP): world-to-camera, ``x_cam = R(q) @ x_world + t``,
quaternion ``q = (w, x, y, z)``. Poses are only comparable within one model.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# COLMAP camera model id -> (name, number of parameters)
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

_CAMERA_HEADER = struct.Struct("<iiQQ")    # camera_id, model_id, width, height
_IMAGE_HEADER = struct.Struct("<I4d3dI")   # image_id, qw qx qy qz, tx ty tz, camera_id
_UINT64 = struct.Struct("<Q")
_POINT2D_BYTES = 24                        # x, y (float64) + point3D_id (int64)


@dataclass(frozen=True)
class Camera:
    """Intrinsics of one COLMAP camera. ``params`` follow the model's convention
    (e.g. SIMPLE_RADIAL: f, cx, cy, k)."""

    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def focal_and_center(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) for the pinhole part of the model (distortion ignored)."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                          "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
            return p[0], p[0], p[1], p[2]
        return p[0], p[1], p[2], p[3]


@dataclass(frozen=True)
class ImagePose:
    """A registered image: its world-to-camera pose and camera id."""

    image_id: int
    qvec: tuple[float, float, float, float]  # w, x, y, z
    tvec: tuple[float, float, float]
    camera_id: int
    name: str

    def rotation(self) -> np.ndarray:
        """3x3 world-to-camera rotation matrix."""
        w, x, y, z = self.qvec
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    def camera_center(self) -> np.ndarray:
        """Camera position in world coordinates, ``-R^T t``."""
        return -self.rotation().T @ np.asarray(self.tvec)


def read_cameras_bin(data: bytes) -> list[Camera]:
    """Parse the bytes of a COLMAP ``cameras.bin``."""
    (count,) = _UINT64.unpack_from(data, 0)
    offset = _UINT64.size
    cameras = []
    for _ in range(count):
        camera_id, model_id, width, height = _CAMERA_HEADER.unpack_from(data, offset)
        offset += _CAMERA_HEADER.size
        model, n_params = CAMERA_MODELS[model_id]
        params = struct.unpack_from(f"<{n_params}d", data, offset)
        offset += 8 * n_params
        cameras.append(Camera(camera_id, model, width, height, tuple(params)))
    return cameras


def _read_images(data: bytes, has_points2d: bool) -> list[ImagePose]:
    (count,) = _UINT64.unpack_from(data, 0)
    offset = _UINT64.size
    images = []
    for _ in range(count):
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _IMAGE_HEADER.unpack_from(data, offset)
        offset += _IMAGE_HEADER.size
        end = data.index(b"\x00", offset)  # image name is null-terminated
        name = data[offset:end].decode("utf-8")
        offset = end + 1
        if has_points2d:
            (n_points,) = _UINT64.unpack_from(data, offset)
            offset += _UINT64.size + _POINT2D_BYTES * n_points
        images.append(ImagePose(image_id, (qw, qx, qy, qz), (tx, ty, tz), camera_id, name))
    if offset != len(data):
        raise ValueError(f"Trailing bytes after {count} images ({len(data) - offset} left): wrong format?")
    return images


def read_images_bin(data: bytes) -> list[ImagePose]:
    """Parse the bytes of a full COLMAP ``images.bin`` (keypoints are skipped)."""
    return _read_images(data, has_points2d=True)


def read_images_minibin(data: bytes) -> list[ImagePose]:
    """Parse the bytes of a MegaScenes ``images.minibin`` (poses only)."""
    return _read_images(data, has_points2d=False)
