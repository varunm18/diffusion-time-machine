"""Tests for registering images into an existing reconstruction (CPU, fake predictor)."""
import numpy as np
import pytest

from timemachine.data.cameras import c2w_to_colmap, colmap_to_c2w, quat_to_rotmat, rotmat_to_quat
from timemachine.posing.align import (
    Sim3, mean_rotation, ransac_umeyama, rotation_error_deg, rotation_first_alignment, umeyama,
)
from timemachine.posing.batches import farthest_point_sample, make_batches
from timemachine.posing.register import (
    RegistrationConfig, evaluate_registration, register_images, scene_size, summarize_errors,
)


def random_rotation(rng):
    q = rng.normal(size=4)
    return quat_to_rotmat(q / np.linalg.norm(q))


def random_cameras(rng, n):
    c2w = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        c2w[i, :3, :3] = random_rotation(rng)
    c2w[:, :3, 3] = rng.normal(size=(n, 3)) * 5
    return c2w


def test_quaternion_and_colmap_roundtrips():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = random_rotation(rng)
        np.testing.assert_allclose(quat_to_rotmat(rotmat_to_quat(R)), R, atol=1e-10)
    c2w = random_cameras(rng, 1)[0]
    q, t = c2w_to_colmap(c2w)
    np.testing.assert_allclose(colmap_to_c2w(q, t), c2w, atol=1e-10)


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(1)
    truth = Sim3(2.5, random_rotation(rng), rng.normal(size=3))
    src = rng.normal(size=(20, 3))
    est = umeyama(src, truth.apply_points(src))
    assert est.scale == pytest.approx(2.5)
    np.testing.assert_allclose(est.R, truth.R, atol=1e-9)
    np.testing.assert_allclose(est.t, truth.t, atol=1e-9)


def test_ransac_ignores_outliers():
    rng = np.random.default_rng(2)
    truth = Sim3(0.7, random_rotation(rng), rng.normal(size=3))
    src = rng.normal(size=(30, 3))
    dst = truth.apply_points(src)
    dst[:6] += rng.normal(size=(6, 3)) * 10        # 20% gross outliers
    est, inliers = ransac_umeyama(src, dst, rng)
    assert not inliers[:6].any() and inliers[6:].all()
    assert est.scale == pytest.approx(0.7, rel=1e-6)


def test_rotation_metrics():
    rng = np.random.default_rng(3)
    R = random_rotation(rng)
    small = quat_to_rotmat(np.array([np.cos(np.radians(5)), np.sin(np.radians(5)), 0, 0]))  # 10 deg
    assert rotation_error_deg(R, R @ small) == pytest.approx(10.0)
    np.testing.assert_allclose(mean_rotation(np.stack([R, R, R])), R, atol=1e-10)


def test_batches_cover_every_query_repeatedly():
    rng = np.random.default_rng(4)
    anchors = random_cameras(rng, 50)
    batches = make_batches(70, anchors, rng, queries_per_batch=16, anchors_per_batch=10, repeats=3)
    counts = np.bincount(np.concatenate([q for q, _ in batches]), minlength=70)
    assert np.all(counts == 3)
    assert all(len(set(a)) == 10 for _, a in batches)
    assert len(set(farthest_point_sample(rng.normal(size=(30, 4)), 12, rng))) == 12


def test_register_images_with_fake_predictor():
    rng = np.random.default_rng(5)
    n_queries, n_anchors = 20, 40
    gt = random_cameras(rng, n_queries + n_anchors)
    paths = [f"img{i}" for i in range(len(gt))]
    index = {p: i for i, p in enumerate(paths)}
    bad_anchors = {paths[n_queries + k] for k in range(4)}   # the model gets these badly wrong

    def fake_predict(batch_paths):
        # Ground truth seen through a random similarity (VGGT's own frame), plus small noise.
        frame = Sim3(rng.uniform(0.2, 3.0), random_rotation(rng), rng.normal(size=3))
        inv_R = frame.R.T
        c2w = gt[[index[p] for p in batch_paths]].copy()
        c2w[:, :3, :3] = inv_R @ c2w[:, :3, :3]
        c2w[:, :3, 3] = ((c2w[:, :3, 3] - frame.t) @ inv_R.T) / frame.scale
        c2w[:, :3, 3] += rng.normal(scale=0.002, size=(len(batch_paths), 3))
        for k, p in enumerate(batch_paths):
            if p in bad_anchors:
                c2w[k, :3, 3] += 5.0
        return {"c2w": c2w, "focal": np.full(len(batch_paths), 500.0)}

    for alignment in ("rotation_first", "centers"):
        cfg = RegistrationConfig(queries_per_batch=8, anchors_per_batch=16, repeats=2, min_inliers=6,
                                 alignment=alignment, center_threshold=0.05)
        queries, batches = register_images(fake_predict, paths[:n_queries], paths[n_queries:], gt[n_queries:], cfg)
        assert batches["used"].all(), alignment
        evaluated = evaluate_registration(queries, gt[:n_queries], scene_size(gt[n_queries:]))
        summary = summarize_errors(evaluated)
        assert summary["placed"] == 1.0
        assert summary["rot_err_median_deg"] < 0.5 and summary["center_err_median"] < 0.01
        assert (queries["n_batches"] == 2).all()


def test_rotation_first_alignment_tolerates_noisy_positions():
    rng = np.random.default_rng(6)
    gt = random_cameras(rng, 30)
    frame = Sim3(1.7, random_rotation(rng), rng.normal(size=3))
    pred = gt.copy()                                  # gt expressed in a scrambled frame
    pred[:, :3, :3] = frame.R.T @ gt[:, :3, :3]
    pred[:, :3, 3] = ((gt[:, :3, 3] - frame.t) @ frame.R) / frame.scale
    pred[:, :3, 3] += rng.normal(scale=0.3, size=(30, 3))   # noisy positions, exact rotations
    pred[:5, :3, :3] = pred[:5, :3, :3] @ random_rotation(rng)  # a few anchors with wrong rotations
    sim, inliers = rotation_first_alignment(pred, gt, rng)
    assert not inliers[:5].any() and inliers.sum() >= 15
    assert rotation_error_deg(sim.R, frame.R) < 1e-6


def test_retrieval_batches_group_similar_queries():
    from timemachine.posing.batches import make_retrieval_batches
    rng = np.random.default_rng(7)
    centers = np.eye(4)[:2] * 10                       # two visual clusters
    q = np.concatenate([centers[0] + rng.normal(size=(8, 4)), centers[1] + rng.normal(size=(8, 4))])
    a = np.concatenate([centers[0] + rng.normal(size=(20, 4)), centers[1] + rng.normal(size=(20, 4))])
    q /= np.linalg.norm(q, axis=1, keepdims=True); a /= np.linalg.norm(a, axis=1, keepdims=True)
    batches = make_retrieval_batches(q, a, rng, queries_per_batch=8, anchors_per_batch=10, repeats=2)
    for group, anchors in batches:
        cluster = group[0] >= 8
        assert np.all((group >= 8) == cluster) and np.all((anchors >= 20) == cluster)
