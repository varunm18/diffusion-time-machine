"""Tests for timemachine.eval.splits (near-duplicate groups and leak-free splits).

HOW THESE WORK
--------------
Everything is checked on small synthetic scenes where the right answer is known by
construction: planted copies of a photo (same date, same camera) must end up in one
group, while photos that differ in date OR camera must not. Split tests check the
properties that matter for evaluation: no group is ever spread over two splits,
scarce eras still reach val and test, and results don't depend on row order.
"""
import numpy as np
import pandas as pd
import pytest

from timemachine.eval.splits import (
    assert_no_leak, assign_splits, near_duplicate_groups, split_summary, view_c2ws,
)


def c2w(position, yaw=0.0):
    """Camera-to-world matrix at `position`, turned by `yaw` about the up axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    m[:3, 3] = position
    return m


def ring(n, radius=5.0, seed=0):
    """n cameras on a ring, all looking toward the middle, in random order around it."""
    angles = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    return np.stack([c2w([radius * np.cos(a), 0, radius * np.sin(a)], a + np.pi) for a in angles])


# ----------------------------------------------------------------------------
# Cameras from the table
# ----------------------------------------------------------------------------

def test_view_c2ws_identity_pose():
    # Identity rotation and translation (0, 0, 5): the camera sits at (0, 0, -5).
    views = pd.DataFrame([dict(qw=1, qx=0, qy=0, qz=0, tx=0, ty=0, tz=5)])
    c = view_c2ws(views)[0]
    assert np.allclose(c[:3, :3], np.eye(3))
    assert np.allclose(c[:3, 3], [0, 0, -5])


def test_view_c2ws_quarter_turn_about_z():
    # A 90 degree rotation about z has quaternion (cos 45, 0, 0, sin 45). Going world -> camera
    # it sends the world x axis to the camera y axis, and the camera center must map to the origin.
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    views = pd.DataFrame([dict(qw=q[0], qx=q[1], qy=q[2], qz=q[3], tx=1, ty=2, tz=3)])
    c = view_c2ws(views)[0]
    w2c_rot = c[:3, :3].T
    assert np.allclose(w2c_rot @ [1, 0, 0], [0, 1, 0], atol=1e-9)    # +x in the world is +y in the camera
    assert np.allclose(w2c_rot @ c[:3, 3] + [1, 2, 3], 0, atol=1e-9)  # the center maps to the camera origin


def test_view_c2ws_matches_repo_cameras_when_torch_is_available():
    # Cross-check against timemachine.data.cameras (which needs torch).
    pytest.importorskip("torch")
    from timemachine.data.cameras import colmap_to_c2w
    rng = np.random.default_rng(1)
    q = rng.normal(size=(6, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    t = rng.normal(size=(6, 3))
    views = pd.DataFrame(np.hstack([q, t]), columns=["qw", "qx", "qy", "qz", "tx", "ty", "tz"])
    assert np.allclose(view_c2ws(views), colmap_to_c2w(q, t), atol=1e-9)


# ----------------------------------------------------------------------------
# Near-duplicate groups
# ----------------------------------------------------------------------------

def test_planted_copies_form_one_group():
    # Rows 0-2 are the same moment from the same spot (a print scanned three times).
    cams = ring(6)
    cams[1] = cams[0]
    cams[2] = cams[0]
    dates = np.array([1863.0, 1863.0, 1863.0, 1920.0, 1955.0, 2015.0])
    g = near_duplicate_groups(dates, cams)
    assert g[0] == g[1] == g[2]
    assert len(set(g[3:])) == 3 and not set(g[3:]) & {g[0]}


def test_same_camera_but_different_decade_is_not_a_duplicate():
    cams = ring(2)
    cams[1] = cams[0]
    g = near_duplicate_groups(np.array([1863.0, 1913.0]), cams)
    assert g[0] != g[1]


def test_same_date_but_different_camera_is_not_a_duplicate():
    cams = ring(2, seed=3)
    cams[1] = c2w([5.0, 0, 5.0], 0.0)           # a clearly different spot and direction
    g = near_duplicate_groups(np.array([2015.0, 2015.0]), cams)
    assert g[0] != g[1]


def test_duplicates_chain_transitively():
    # Three cameras 0.4 apart along a line: A~B and B~C are close enough, but A and C
    # (0.8 apart) are not. They must still end up in one group through B.
    # Ten far-away cameras on a ring of radius 10 make the scene scale about 10,
    # so 0.4 is a distance of 0.04 scene widths (under the 0.05 tolerance).
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    far = [c2w([10 * np.cos(a), 0, 10 * np.sin(a)]) for a in angles]
    trio = [c2w([0.0, 0, 0]), c2w([0.4, 0, 0]), c2w([0.8, 0, 0])]
    cams = np.stack(trio + far)
    dates = np.full(len(cams), 2015.0)

    g = near_duplicate_groups(dates, cams, date_tol=0.1, pose_tol=0.05)
    assert g[0] == g[1] == g[2]
    assert len(set(g[3:])) == 10                      # the ring cameras stay separate

    # Without the middle camera, A and C alone are too far apart to be linked.
    without_b = np.stack([trio[0], trio[2]] + far)
    g2 = near_duplicate_groups(np.full(len(without_b), 2015.0), without_b, date_tol=0.1, pose_tol=0.05)
    assert g2[0] != g2[1]


def test_group_partition_does_not_depend_on_row_order():
    cams = ring(12, seed=5)
    cams[4] = cams[1]
    cams[9] = cams[1]
    dates = np.array([2015.0, 2015.0, 1900.0, 1950.0, 2015.0, 1900.0, 1960.0, 2001.0, 1990.0, 2015.0, 1880.0, 2010.0])
    base = near_duplicate_groups(dates, cams)

    perm = np.random.default_rng(0).permutation(12)
    shuffled = near_duplicate_groups(dates[perm], cams[perm])
    # Same partition: two original rows share a group in one run iff they do in the other.
    for a in range(12):
        for b in range(12):
            same_before = base[perm[a]] == base[perm[b]]
            same_after = shuffled[a] == shuffled[b]
            assert same_before == same_after


def test_undated_views_are_their_own_groups():
    cams = ring(3)
    cams[1] = cams[0]
    g = near_duplicate_groups(np.array([np.nan, np.nan, 2015.0]), cams)
    assert len(set(g)) == 3


def test_group_ids_are_consecutive_in_order_of_first_appearance():
    cams = ring(4)
    cams[3] = cams[0]
    g = near_duplicate_groups(np.array([2015.0, 1900.0, 1950.0, 2015.0]), cams)
    assert list(g) == [0, 1, 2, 0]


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------

def make_views(n_modern=300, n_old=30, seed=0):
    """A table with an 'old' era (scarce) and a 'modern' era, every view its own group."""
    views = pd.DataFrame({
        "file_key": [f"file_{i:04d}" for i in range(n_modern + n_old)],
        "era": ["modern"] * n_modern + ["old"] * n_old,
    })
    return views, np.arange(len(views))


def test_split_values_and_alignment():
    views, groups = make_views()
    split = assign_splits(views, groups)
    assert set(split) <= {"train", "val", "test"}
    assert split.index.equals(views.index)


def test_fractions_are_roughly_respected_per_era():
    views, groups = make_views()
    summary = split_summary(views, assign_splits(views, groups, 0.1, 0.1))
    assert summary.loc["modern", "val"] == 30 and summary.loc["modern", "test"] == 30
    assert summary.loc["old", "val"] == 3 and summary.loc["old", "test"] == 3


def test_scarce_era_still_reaches_val_and_test():
    # Only 6 old groups: a plain random 10%/10% split could easily give 0. We guarantee at least one.
    views, groups = make_views(n_modern=100, n_old=6)
    summary = split_summary(views, assign_splits(views, groups, 0.1, 0.1))
    assert summary.loc["old", "val"] >= 1 and summary.loc["old", "test"] >= 1
    assert summary.loc["old", "train"] >= 1


def test_tiny_era_stays_in_train():
    views, groups = make_views(n_modern=100, n_old=3)
    summary = split_summary(views, assign_splits(views, groups))
    assert summary.loc["old", "train"] == 3 and summary.loc["old", "val"] == 0 and summary.loc["old", "test"] == 0


def test_groups_never_straddle_splits():
    views, _ = make_views(n_modern=200, n_old=40)
    rng = np.random.default_rng(2)
    groups = rng.integers(0, 60, len(views))      # many views share a group
    views["era"] = pd.Series(groups % 3).map({0: "a", 1: "b", 2: "c"}).to_numpy()   # an era is constant per group
    split = assign_splits(views, groups)
    assert_no_leak(groups, split)


def test_assert_no_leak_catches_a_split_group():
    groups = np.array([0, 0, 1])
    with pytest.raises(AssertionError):
        assert_no_leak(groups, pd.Series(["train", "test", "train"]))


def test_deterministic_and_seed_sensitive():
    views, groups = make_views()
    a = assign_splits(views, groups, seed=0)
    assert a.equals(assign_splits(views, groups, seed=0))
    assert not a.equals(assign_splits(views, groups, seed=1))


def test_assignment_does_not_depend_on_row_order():
    views, groups = make_views()
    shuffled = views.sample(frac=1.0, random_state=0)
    a = assign_splits(views, groups).set_axis(views["file_key"])
    b = assign_splits(shuffled, groups[shuffled.index.to_numpy()]).set_axis(shuffled["file_key"])
    assert (a.loc[b.index] == b).all()


def test_works_without_an_era_column():
    views, groups = make_views()
    split = assign_splits(views.drop(columns="era"), groups)
    assert (split == "test").sum() == 33 and (split == "val").sum() == 33


def test_invalid_fractions_rejected():
    views, groups = make_views(20, 0)
    with pytest.raises(ValueError):
        assign_splits(views, groups, val_fraction=0.6, test_fraction=0.5)
    with pytest.raises(ValueError):
        assign_splits(views, groups[:-1])
