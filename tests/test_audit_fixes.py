"""Tests for the fixes from the 2026-11-10 code audit (see docs/findings.md)."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from timemachine.data.cameras import adjust_intrinsics, intrinsics_matrix, undistorted_crop
from timemachine.data.dataset import load_view
from timemachine.data.sampling import SamplingConfig, eligible_anchors, sample_views
from timemachine.data.views import likely_stereo_cards
from timemachine.dating.sources.text import parse_date_text
from timemachine.megascenes.colmap import Camera


# --- lens distortion ----------------------------------------------------------
def test_undistorted_crop_without_distortion_is_the_plain_crop():
    image = torch.rand(3, 40, 60)
    K = intrinsics_matrix(50.0, 50.0, 30.0, 20.0)
    out, K_out = undistorted_crop(image, K, 0.0, 0.0, left=10, top=5, size=32)
    torch.testing.assert_close(out, image[:, 5:37, 10:42], atol=1e-5, rtol=0)
    np.testing.assert_allclose(K_out, adjust_intrinsics(K, 1, 1, 10, 5))


def test_undistorted_crop_moves_a_point_to_its_pinhole_projection():
    """A dot drawn where a radial lens images a 3D point must land at that point's pinhole pixel."""
    k1, k2, size = -0.6, 0.05, 64
    K = intrinsics_matrix(60.0, 60.0, 50.0, 40.0)                     # 100 x 80 source image
    X = np.array([0.4, -0.3, 1.0])                                     # point, camera frame (in the crop)
    xn = X[:2] / X[2]
    r2 = xn @ xn
    xd = xn * (1 + k1 * r2 + k2 * r2 ** 2)
    u, v = K[0, 0] * xd[0] + K[0, 2], K[1, 1] * xd[1] + K[1, 2]       # distorted pixel (corner convention)
    image = torch.zeros(3, 80, 100)
    image[:, int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2] = 1.0       # 3x3 dot around (u, v)
    out, K_out = undistorted_crop(image, K, k1, k2, left=18, top=8, size=size)
    expected = K_out @ np.array([*xn, 1.0])
    ys, xs = torch.nonzero(out[0] > 0.5, as_tuple=True)
    assert len(xs) > 0
    found = np.array([xs.float().mean() + 0.5, ys.float().mean() + 0.5])
    assert np.abs(found - expected[:2]).max() < 1.0
    # ... and not where the distorted position would put it (the shift is several pixels).
    assert np.abs(np.array([u - 18, v - 8]) - expected[:2]).max() > 3


def _row(tmp_path, k1, k2):
    path = tmp_path / "v.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(path)
    return SimpleNamespace(file_key="v.jpg", image_path=str(path), image_width=80, image_height=60,
                           camera_width=800, camera_height=600, fx=700.0, fy=700.0, cx=400.0, cy=300.0,
                           k1=k1, k2=k2)


def test_load_view_undistorts_and_rejects_unknown_lens_models(tmp_path):
    plain, K0 = load_view(_row(tmp_path, 0.0, 0.0), 32)
    distorted, K1 = load_view(_row(tmp_path, -0.3, 0.0), 32)
    np.testing.assert_allclose(K0, K1)                    # same pinhole intrinsics either way
    assert plain.shape == distorted.shape == (3, 32, 32) and not torch.allclose(plain, distorted)
    with pytest.raises(ValueError):
        load_view(_row(tmp_path, float("nan"), float("nan")), 32)


def test_colmap_radial_terms_and_fov_focal():
    assert Camera(1, "SIMPLE_RADIAL", 10, 10, (5.0, 5.0, 5.0, -0.1)).radial_distortion() == (-0.1, 0.0)
    assert Camera(1, "RADIAL", 10, 10, (5.0, 5.0, 5.0, -0.1, 0.02)).radial_distortion() == (-0.1, 0.02)
    assert Camera(1, "PINHOLE", 10, 10, (5.0, 6.0, 5.0, 5.0)).radial_distortion() == (0.0, 0.0)
    assert all(math.isnan(k) for k in Camera(1, "OPENCV_FISHEYE", 10, 10, (1.0,) * 8).radial_distortion())
    # FOV is (fx, fy, cx, cy, omega), not single-focal.
    assert Camera(1, "FOV", 10, 10, (5.0, 6.0, 4.0, 3.0, 0.9)).focal_and_center() == (5.0, 6.0, 4.0, 3.0)


# --- stereo cards ---------------------------------------------------------------
def test_likely_stereo_cards():
    views = pd.DataFrame({
        "image_width": [1024, 1024, 1024, 1024, 1024],
        "image_height": [512, 554, 512, 459, 700],
        "date_mid": [1860.0, 1859.0, 2015.0, 1900.0, 1880.0],
        "title": ["card", "pont Saint-Michel", "panorama", "Parvis postcard", "Vues pour Stéréoscope"],
    })
    assert likely_stereo_cards(views).tolist() == [True, False, False, False, True]


# --- sampling: the input gap is measured from every target ---------------------------
def test_min_input_gap_holds_for_every_target():
    rng = np.random.default_rng(0)
    dates = np.concatenate([1880 + rng.uniform(-5, 5, 30), 1905 + rng.uniform(-5, 5, 30),
                            2015 + rng.uniform(-1, 1, 30)])
    cfg = SamplingConfig(n_inputs=3, n_targets=5, min_input_gap=20.0)
    anchors = eligible_anchors(dates, cfg)
    for _ in range(300):
        inputs, targets = sample_views(dates, cfg, rng, anchors)
        assert np.abs(dates[inputs][:, None] - dates[targets][None]).min() > 20.0


# --- dating ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, lo, hi", [
    ("not before 1890", 1890.0, None), ("before 1890", None, 1891.0),
    ("not after 1900", None, 1901.0), ("after 1900", 1900.0, None),
])
def test_negated_bounds(text, lo, hi):
    c = parse_date_text(text)
    assert (lo is None or c.interval.lo == lo) and (hi is None or c.interval.hi == hi)
    assert (lo is None) == (c.interval.lo < 1500) and (hi is None) == (c.interval.hi > 2025)


def test_capture_on_the_upload_day_is_not_clipped():
    """Day-precision dates on (or, by time zone, the day after) the upload day stay untouched."""
    import datetime as dt

    from timemachine.dating.interval import DateCandidate, DateInterval, Precision, Source, decimal_year
    from timemachine.dating.resolve import resolve
    for day in (dt.date(2015, 1, 3), dt.date(2015, 12, 31), dt.date(2016, 6, 30), dt.date(2015, 7, 4)):
        upload = dt.datetime.combine(day, dt.time(23, 0)).isoformat()
        for d in (day, day + dt.timedelta(days=1)):
            iv = DateInterval(decimal_year(d), decimal_year(d + dt.timedelta(days=1)))
            result = resolve([DateCandidate(iv, Source.EXIF, Precision.DAY, 0.9, "exif")], upload_time=upload)
            assert result.interval == iv, (day, d, result.flags)
            assert not any(f.startswith("impossible") for f in result.flags)
            assert not any("clipped" in c.flags for c in result.candidates)


# --- training: spike-guard clipping ---------------------------------------------------
def test_clip_limit_is_off_until_history_then_relative():
    from collections import deque

    from timemachine.training.trainer import Trainer, TrainConfig
    t = object.__new__(Trainer)
    t.cfg = TrainConfig(dataset="x", run_dir="x")
    t.grad_norm_history = deque(maxlen=200)
    assert t.clip_limit() == math.inf
    t.grad_norm_history.extend([2.0] * 60)
    assert t.clip_limit() == pytest.approx(20.0)
    t.cfg = TrainConfig(dataset="x", run_dir="x", grad_clip=5.0)
    assert t.clip_limit() == pytest.approx(5.0)
