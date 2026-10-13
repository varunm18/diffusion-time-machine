"""Read camera poses out of MegaScenes' COLMAP files, with no extra libraries.

WHY THIS FILE EXISTS
--------------------
Each MegaScenes scene comes with a COLMAP "sparse reconstruction": for every
photo that COLMAP managed to place in 3D, it stores *where the camera was* and
*which way it pointed* (the pose), plus the camera's lens settings (the
intrinsics). Our model needs exactly that for every training image.

The reconstruction lives in binary files:
    cameras.bin    one entry per physical camera setup (lens + image size)
    images.bin     one entry per registered photo (pose + which camera it used)
    points3D.bin   the 3D point cloud (we don't need it for poses)

We read the binary format ourselves (a few `struct.unpack` calls) so there's
nothing extra to install. The format is documented in COLMAP's "Output Format"
page (binary section).

TWO THINGS WORTH KNOWING
------------------------
* images.bin is HUGE (about 540 MB for the Taj Mahal) because almost all of it
  is 2D keypoints, which we don't need. We skip over those bytes instead of
  loading them, so reading is fast and light on memory.
* COLMAP poses are WORLD-TO-CAMERA:   x_camera = R @ x_world + t
  and quaternions are ordered (w, x, y, z). Other tools use different
  conventions, so converting to whatever SEVA expects is a separate, explicit
  step (see docs/interface_decisions.md, Section 6).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# COLMAP identifies each camera "model" (the lens math) by a number. Each model
# stores a different count of float parameters, and we need that count to know
# how many bytes to read. MegaScenes' Taj Mahal uses only SIMPLE_RADIAL (id 2).
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),         # f, cx, cy
    1: ("PINHOLE", 4),                # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),          # f, cx, cy, k
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class Camera:
    """One camera setup. Several photos can share the same Camera."""

    camera_id: int
    model: str                  # e.g. "SIMPLE_RADIAL"
    width: int                  # the image size the lens was calibrated at
    height: int
    params: Tuple[float, ...]   # lens parameters; meaning depends on `model`


@dataclass(frozen=True)
class ImagePose:
    """One registered photo: its pose, which Camera it used, and its filename."""

    image_id: int
    qvec: Tuple[float, float, float, float]  # rotation as a quaternion (w, x, y, z)
    tvec: Tuple[float, float, float]         # translation
    camera_id: int                           # key into the cameras dict
    name: str                                # e.g. "commons/<scene>/0/pictures/<file>"
    num_points2d: int                        # how many keypoints it had (we skip them)

    def rotation(self) -> np.ndarray:
        """3x3 rotation matrix R (world -> camera)."""
        return qvec_to_rotmat(self.qvec)

    def world_to_camera(self) -> np.ndarray:
        """The pose as a single 4x4 matrix [[R, t], [0, 1]]."""
        m = np.eye(4)
        m[:3, :3] = self.rotation()
        m[:3, 3] = self.tvec
        return m

    def camera_center(self) -> np.ndarray:
        """Where the camera physically sits, in world coordinates.

        COLMAP stores the world->camera transform, so the camera's own position
        isn't stored directly. Solving R @ C + t = 0 for the point C that maps to
        the camera origin gives C = -R^T @ t.
        """
        return -self.rotation().T @ np.asarray(self.tvec)


def qvec_to_rotmat(q) -> np.ndarray:
    """Convert a unit quaternion (w, x, y, z) into a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _read(f, fmt: str):
    """Read and unpack one little-endian record from the file.

    `fmt` uses struct codes: i = int32, Q = uint64, d = float64, q = int64.
    If the file ends early we raise a clear error instead of silently
    returning garbage from a truncated download.
    """
    size = struct.calcsize("<" + fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError("Unexpected end of COLMAP file (truncated download?)")
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path) -> Dict[int, Camera]:
    """Read cameras.bin into {camera_id: Camera}.

    Layout: a uint64 count, then for each camera:
        int32 camera_id, int32 model_id, uint64 width, uint64 height,
        then `n_params` float64 values (n_params depends on the model).
    """
    cameras: Dict[int, Camera] = {}
    with open(path, "rb") as f:
        (num_cameras,) = _read(f, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read(f, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unknown COLMAP camera model id {model_id}")
            name, n_params = CAMERA_MODELS[model_id]
            params = _read(f, "d" * n_params)
            cameras[camera_id] = Camera(camera_id, name, width, height, tuple(params))
    return cameras


def read_images_binary(path) -> Dict[int, ImagePose]:
    """Read images.bin into {image_id: ImagePose}, skipping the 2D keypoints.

    Layout: a uint64 count, then for each registered image:
        int32 image_id, 4 x float64 quaternion, 3 x float64 translation,
        int32 camera_id, a null-terminated filename, uint64 num_points2d,
        then num_points2d keypoints of 24 bytes each (x, y, point3D_id).
    The keypoints are most of the file and we don't need them, so we seek past.
    """
    images: Dict[int, ImagePose] = {}
    with open(path, "rb") as f:
        (num_images,) = _read(f, "Q")
        for _ in range(num_images):
            # id, quaternion (w x y z), translation (x y z), camera id
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read(f, "idddddddi")

            # The filename is stored as raw bytes ending in a zero byte, so we
            # read one byte at a time until we hit that terminator.
            chars = bytearray()
            while True:
                c = f.read(1)
                if not c:
                    raise EOFError("Unexpected end of COLMAP file in an image name")
                if c == b"\x00":
                    break
                chars += c
            name = chars.decode("utf-8")

            (num_points2d,) = _read(f, "Q")
            # Each keypoint = x (8 bytes) + y (8 bytes) + point3D id (8 bytes).
            # Jump over all of them without reading. If this arithmetic were off
            # by even one byte, every later record would be garbage.
            f.seek(num_points2d * 24, 1)

            images[image_id] = ImagePose(
                image_id, (qw, qx, qy, qz), (tx, ty, tz), camera_id, name, num_points2d
            )
    return images


def read_model(model_dir) -> Tuple[Dict[int, Camera], Dict[int, ImagePose]]:
    """Convenience: read cameras.bin and images.bin from one model folder."""
    model_dir = Path(model_dir)
    cameras = read_cameras_binary(model_dir / "cameras.bin")
    images = read_images_binary(model_dir / "images.bin")
    return cameras, images


def intrinsics_matrix(camera: Camera) -> np.ndarray:
    """Build the 3x3 pinhole intrinsics matrix K for a camera.

        K = [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]

    This IGNORES lens distortion (the radial term in SIMPLE_RADIAL), which is
    fine for a first pass but worth revisiting if edges look warped.

    The parameter order differs by model:
      SIMPLE_PINHOLE / SIMPLE_RADIAL / RADIAL: (f, cx, cy, ...)   one focal length
      PINHOLE / OPENCV / FULL_OPENCV:          (fx, fy, cx, cy, ...)
    """
    p = camera.params
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                        "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        raise NotImplementedError(f"Intrinsics for model {camera.model} not implemented")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
