"""Tests for the training-data package (cameras, sampling, dataset)."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from timemachine.data import cameras
from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.sampling import SamplingConfig, eligible_anchors, sample_views, tolerance
from timemachine.megascenes.colmap import ImagePose


# --- cameras -------------------------------------------------------------------
def test_colmap_to_c2w_matches_colmap_pose():
    q = np.array([0.9, 0.1, -0.3, 0.2]); q /= np.linalg.norm(q)
    t = np.array([0.5, -1.0, 2.0])
    pose = ImagePose(1, tuple(q), tuple(t), 1, "x")
    c2w = cameras.colmap_to_c2w(q, t)
    np.testing.assert_allclose(c2w[:3, :3], pose.rotation().T, atol=1e-12)
    np.testing.assert_allclose(c2w[:3, 3], pose.camera_center(), atol=1e-12)


def test_intrinsics_follow_resize_and_crop():
    K = cameras.intrinsics_matrix(500, 500, 400, 300)          # 800x600 image
    point = np.array([0.3, -0.2, 2.0])
    u, v, _ = K @ point / point[2]
    scale, left, top = cameras.resize_and_center_crop_params(800, 600, 576)
    K2 = cameras.adjust_intrinsics(K, scale, scale, left, top)
    u2, v2, _ = K2 @ point / point[2]
    assert u2 == pytest.approx(u * scale - left) and v2 == pytest.approx(v * scale - top)
    Kn = cameras.normalize_intrinsics(K2, 576, 576)
    assert 0 <= Kn[0, 2] <= 1 and 0 <= Kn[1, 2] <= 1


def test_normalize_cameras_like_seva():
    # 40 cameras on a unit circle around (5, 0, 0), plus one far outlier.
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    c2ws = np.tile(np.eye(4), (41, 1, 1))
    c2ws[:40, 0, 3] = 5 + np.cos(angles)
    c2ws[:40, 1, 3] = np.sin(angles)
    c2ws[40, :3, 3] = [1e5, 0, 0]
    out = cameras.normalize_cameras(c2ws)
    # centred on the inliers' mean (5, 0, 0), scaled so camera 0 is at distance 2
    assert np.linalg.norm(out[0, :3, 3]) == pytest.approx(2.0)
    np.testing.assert_allclose(out[:40, :3, 3].mean(0), 0.0, atol=1e-9)


def test_plucker_rays():
    c2ws = torch.eye(4).repeat(2, 1, 1)
    c2ws[1, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
    K = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]).repeat(2, 1, 1)
    rays = cameras.plucker_rays(c2ws, K, 4, 4)
    assert rays.shape == (2, 6, 4, 4)
    d, m = rays[:, :3], rays[:, 3:]
    torch.testing.assert_close(d.norm(dim=1), torch.ones(2, 4, 4))
    assert m[0].abs().max() < 1e-6                             # reference camera at the origin
    torch.testing.assert_close((d * m).sum(dim=1), torch.zeros(2, 4, 4), atol=1e-6, rtol=0)


# --- sampling -------------------------------------------------------------------
DATES = np.array([1890.0, 1891.0, 1893.0, 1894.0, 1895.0, 1960.0,
                  2010.1, 2010.3, 2010.5, 2010.7, 2010.9, 2015.0, 2018.0, 2019.5])


def test_targets_share_a_date_and_inputs_are_disjoint():
    cfg = SamplingConfig(n_inputs=3, n_targets=5)
    rng = np.random.default_rng(0)
    for _ in range(50):
        inputs, targets = sample_views(DATES, cfg, rng)
        assert len(set(inputs) | set(targets)) == 8
        anchor = DATES[targets[0]]
        assert np.all(np.abs(DATES[targets] - anchor) <= tolerance(cfg, anchor))


def test_old_targets_and_input_gap():
    cfg = SamplingConfig(n_inputs=2, n_targets=5, old_target_fraction=1.0, min_input_gap=50)
    inputs, targets = sample_views(DATES, cfg, np.random.default_rng(1))
    assert DATES[targets].max() < 1970
    assert np.all(np.abs(DATES[inputs] - DATES[targets[0]]) > 50)


def test_no_eligible_anchor():
    assert len(eligible_anchors(np.array([1900.0, 1950.0, 2000.0]), SamplingConfig())) == 0


# --- dataset --------------------------------------------------------------------
@pytest.fixture
def views(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i, date in enumerate(DATES):
        path = tmp_path / f"{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(path)
        angle = i * 0.3
        q = [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]
        rows.append(dict(
            model=0, file_key=f"{i}.jpg", image_path=str(path), image_width=80, image_height=60,
            camera_width=800, camera_height=600, fx=700.0, fy=700.0, cx=400.0, cy=300.0,
            qw=q[0], qx=q[1], qy=q[2], qz=q[3], tx=0.1 * i, ty=0.0, tz=3.0,
            date_lo=np.floor(date), date_hi=np.floor(date) + 1, date_mid=date,
            medium="photo", is_monochrome=date < 1970,
        ))
    return pd.DataFrame(rows)


def test_dataset_item(views):
    ds = DatedViewsDataset(views, SamplingConfig(n_inputs=3, n_targets=5), resolution=32, samples_per_epoch=3)
    item = ds[0]
    assert item["images"].shape == (8, 3, 32, 32)
    assert item["images"].min() >= -1 and item["images"].max() <= 1
    assert item["K"].shape == (8, 3, 3) and torch.all((item["K"][:, :2, 2] >= 0) & (item["K"][:, :2, 2] <= 1))
    assert item["c2w"][0, :3, 3].norm().item() == pytest.approx(2.0, rel=1e-5)
    assert item["is_input"].tolist() == [True] * 3 + [False] * 5
    assert len(item["file_keys"]) == 8


def test_dataset_is_deterministic_per_epoch(views):
    ds = DatedViewsDataset(views, SamplingConfig(n_inputs=3, n_targets=5), resolution=16)
    first = ds.indices(0)
    assert np.array_equal(first, ds.indices(0))
    ds.set_epoch(1)
    assert not all(np.array_equal(first, ds.indices(i)) for i in range(5))


def test_dataset_rejects_mixed_models(views):
    views.loc[0, "model"] = 1
    with pytest.raises(ValueError):
        DatedViewsDataset(views)


def test_color_stats_flags_toned_prints_not_color_photos():
    from timemachine.data.images import MONOCHROME_CHROMA, color_stats
    ramp = np.linspace(0, 1, 96)[None, :, None] * np.ones((96, 96, 1))
    gray = Image.fromarray((ramp * np.array([255, 255, 255])).astype(np.uint8))
    sepia = Image.fromarray((ramp * np.array([255, 200, 140])).astype(np.uint8))   # one hue, varying intensity
    rng = np.random.default_rng(0)
    color = Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8))
    assert color_stats(gray)["chroma_minor"] < MONOCHROME_CHROMA
    assert color_stats(sepia)["chroma_minor"] < MONOCHROME_CHROMA < color_stats(color)["chroma_minor"]
