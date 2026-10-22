"""Tests for data_pipeline/colmap_reader.py.

HOW THESE TESTS WORK
--------------------
We can't ship the real (hundreds of MB) COLMAP files, so each test WRITES a tiny
cameras.bin / images.bin following COLMAP's documented binary format, then
checks our reader reads it back correctly. That proves the reader matches the
format spec. The proof against REAL MegaScenes data is running
data_pipeline/inspect_reconstruction.py on a downloaded model (we did: 5,334
images and 3,560 cameras read without error).

Run with:   pytest tests/data_pipeline/test_colmap_reader.py
"""

import struct

import numpy as np
import pytest

from data_pipeline.colmap_reader import (
    intrinsics_matrix,
    qvec_to_rotmat,
    read_cameras_binary,
    read_images_binary,
    read_model,
)


def write_cameras(path, cams):
    """Write a cameras.bin. `cams` is a list of (id, model_id, width, height, params)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))                    # how many cameras
        for cid, mid, w, h, params in cams:
            f.write(struct.pack("<iiQQ", cid, mid, w, h))        # header for this camera
            f.write(struct.pack("<" + "d" * len(params), *params))  # its lens parameters


def write_images(path, imgs):
    """Write an images.bin. `imgs` is a list of (id, qvec, tvec, camera_id, name, n_keypoints)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))                    # how many images
        for iid, q, t, cid, name, n in imgs:
            f.write(struct.pack("<idddddddi", iid, *q, *t, cid))  # id, rotation, translation, camera
            f.write(name.encode("utf-8") + b"\x00")              # null-terminated filename
            f.write(struct.pack("<Q", n))                        # keypoint count
            for k in range(n):                                   # fake keypoints (x, y, point id)
                f.write(struct.pack("<ddq", float(k), float(k) * 2, k))


@pytest.fixture
def model_dir(tmp_path):
    """A small fake model: two cameras and three images, written to a temp folder."""
    write_cameras(
        tmp_path / "cameras.bin",
        [
            (1, 2, 1024, 768, (900.0, 512.0, 384.0, 0.01)),   # SIMPLE_RADIAL: f, cx, cy, k
            (2, 1, 640, 480, (500.0, 510.0, 320.0, 240.0)),   # PINHOLE: fx, fy, cx, cy
        ],
    )
    write_images(
        tmp_path / "images.bin",
        [
            (10, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 5.0), 1, "commons/Taj_Mahal/0/pictures/a b.jpg", 7),
            (11, (0.0, 0.0, 0.0, 1.0), (1.0, 2.0, 3.0), 2, "ünïcode name.jpg", 0),
            (12, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "last.jpg", 3),
        ],
    )
    return tmp_path


def test_read_cameras(model_dir):
    cams = read_cameras_binary(model_dir / "cameras.bin")
    assert set(cams) == {1, 2}
    assert cams[1].model == "SIMPLE_RADIAL"
    assert (cams[1].width, cams[1].height) == (1024, 768)
    assert cams[2].params == (500.0, 510.0, 320.0, 240.0)


def test_read_images_skips_keypoints_correctly(model_dir):
    imgs = read_images_binary(model_dir / "images.bin")
    # The keypoints are skipped by byte arithmetic. If it were off by even one
    # byte, the LATER images (11 and 12) would come out as garbage, so checking
    # them proves the skipping is exactly right.
    assert set(imgs) == {10, 11, 12}
    assert imgs[10].name == "commons/Taj_Mahal/0/pictures/a b.jpg"
    assert imgs[10].num_points2d == 7
    assert imgs[11].name == "ünïcode name.jpg"   # non-ASCII names must survive
    assert imgs[12].name == "last.jpg"
    assert imgs[12].camera_id == 1


def test_pose_math(model_dir):
    imgs = read_images_binary(model_dir / "images.bin")
    # Identity rotation with translation (0, 0, 5) puts the camera at (0, 0, -5).
    assert np.allclose(imgs[10].camera_center(), [0, 0, -5])
    w2c = imgs[10].world_to_camera()
    assert w2c.shape == (4, 4)
    assert np.allclose(w2c[:3, 3], [0, 0, 5])
    assert np.allclose(w2c[3], [0, 0, 0, 1])   # the bottom row of a pose matrix


def test_rotation_is_orthonormal():
    # A valid rotation matrix satisfies R @ R.T = I and has determinant +1.
    R = qvec_to_rotmat(np.array([0.5, 0.5, 0.5, 0.5]))
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_intrinsics(model_dir):
    cams, _ = read_model(model_dir)
    # SIMPLE_RADIAL has one focal length, used for both axes.
    assert np.allclose(intrinsics_matrix(cams[1]), [[900, 0, 512], [0, 900, 384], [0, 0, 1]])
    # PINHOLE has separate fx and fy.
    assert np.allclose(intrinsics_matrix(cams[2]), [[500, 0, 320], [0, 510, 240], [0, 0, 1]])


def test_truncated_file_raises(tmp_path):
    # A file that claims 3 cameras but contains none (a cut-off download) must
    # fail loudly instead of returning nonsense.
    p = tmp_path / "cameras.bin"
    p.write_bytes(struct.pack("<Q", 3))
    with pytest.raises(EOFError):
        read_cameras_binary(p)
