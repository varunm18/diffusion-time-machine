"""Tests for timemachine.eval.retrieval (the model-free retrieval baselines).

HOW THESE WORK
--------------
No real photos and no torch are needed. We build a tiny synthetic scene (cameras
on a ring, a few dates) where the right answer to "which photo should a baseline
hand back?" is obvious by construction, then check each rule against it. The
end-to-end test uses a fake dataset whose "images" are flat gray squares whose
brightness encodes the photo's date, so the expected PSNR can be worked out by hand.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from timemachine.eval.retrieval import (
    BASELINES, camera_features, compare_with_seva, gap_bucket, pose_distance,
    predict_sample, psnr_np, retrieve, run_baselines, summarize,
)


def c2w(position, yaw=0.0):
    """A camera-to-world matrix at `position`, turned by `yaw` radians about the up axis.

    With zero yaw the camera looks along +z (the repo's OpenCV convention).
    """
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    m[:3, 3] = position
    return m


def features(positions, yaws=None, dates=None):
    """(centers, directions, scale, dates) for a list of cameras."""
    yaws = yaws if yaws is not None else [0.0] * len(positions)
    centers, dirs, scale = camera_features(np.stack([c2w(p, y) for p, y in zip(positions, yaws)]))
    return centers, dirs, scale, np.asarray(dates, dtype=float)


# ----------------------------------------------------------------------------
# Camera geometry
# ----------------------------------------------------------------------------

def test_scale_is_the_ring_radius():
    # Cameras on a ring of radius 5 around the origin: the scene is "5 wide".
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = [[5 * np.cos(a), 0, 5 * np.sin(a)] for a in angles]
    _, _, scale, _ = features(ring, dates=[2000] * 40)
    assert scale == pytest.approx(5.0, rel=0.05)


def test_single_position_scene_does_not_divide_by_zero():
    _, _, scale, _ = features([[1, 1, 1]] * 3, dates=[2000] * 3)
    assert scale == 1.0


def test_pose_distance_grows_with_position_and_angle():
    pos = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]]
    yaw = [0.0, 0.0, 0.0, np.pi / 2]
    centers, dirs, _, _ = features(pos, yaw, dates=[2000] * 4)
    d = pose_distance(centers, dirs, 1.0, 0, np.array([1, 2, 3]))
    assert d[0] == pytest.approx(0.0)            # identical camera
    assert d[1] == pytest.approx(1.0)            # one scene-width away, same direction
    assert d[2] == pytest.approx(0.5)            # same place, turned 90 degrees = half of 180


# ----------------------------------------------------------------------------
# Choosing a photo
# ----------------------------------------------------------------------------

def test_date_strategy_picks_nearest_date():
    #   row:    0     1     2     3
    # date:   1960  1900  1950  2010      target is row 0
    centers, dirs, scale, dates = features([[0, 0, 0]] * 4, dates=[1960, 1900, 1950, 2010])
    assert retrieve("date", 0, [1, 2, 3], dates, centers, dirs, scale) == 2


def test_date_tie_goes_to_nearer_camera():
    # Rows 1 and 2 have the same date; row 2's camera is much closer to the target's.
    centers, dirs, scale, dates = features([[0, 0, 0], [9, 0, 0], [0.1, 0, 0]], dates=[1960, 1950, 1950])
    assert retrieve("date", 0, [1, 2], dates, centers, dirs, 1.0) == 2


def test_pose_strategy_ignores_date():
    # Row 1 has the nearest camera but is 100 years away; row 2 has the nearest date but a far camera.
    centers, dirs, _, dates = features([[0, 0, 0], [0.1, 0, 0], [5, 0, 0]], dates=[2010, 1910, 2009])
    assert retrieve("pose", 0, [1, 2], dates, centers, dirs, 1.0) == 1
    assert retrieve("date", 0, [1, 2], dates, centers, dirs, 1.0) == 2


def test_date_then_pose_respects_the_shortlist():
    # Rows 1-3 are the three nearest in date but have far cameras; row 4 has a perfect
    # camera but is far in date. With a shortlist of 3 row 4 is out; with 4 it is in.
    positions = [[0, 0, 0], [3, 0, 0], [2, 0, 0], [4, 0, 0], [0, 0, 0]]
    centers, dirs, _, dates = features(positions, dates=[2000, 2001, 2002, 2003, 1900])
    cands = [1, 2, 3, 4]
    assert retrieve("date_then_pose", 0, cands, dates, centers, dirs, 1.0, k_date=3) == 2
    assert retrieve("date_then_pose", 0, cands, dates, centers, dirs, 1.0, k_date=4) == 4


def test_retrieve_errors():
    centers, dirs, scale, dates = features([[0, 0, 0], [1, 0, 0]], dates=[2000, 2001])
    with pytest.raises(ValueError):
        retrieve("date", 0, [], dates, centers, dirs, scale)
    with pytest.raises(ValueError):
        retrieve("nonsense", 0, [1], dates, centers, dirs, scale)


# ----------------------------------------------------------------------------
# One sample: where candidates come from
# ----------------------------------------------------------------------------

def scene(n=30, seed=0):
    """30 views: a modern bunch, a 1900 bunch and a 1960 bunch, on a ring."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    positions = [[5 * np.cos(a), 0, 5 * np.sin(a)] for a in angles]
    dates = np.concatenate([rng.uniform(2010, 2019, 14), rng.uniform(1895, 1905, 8), rng.uniform(1955, 1965, 8)])
    return features(positions, yaws=list(angles + np.pi), dates=dates)


def test_input_baselines_only_return_inputs():
    centers, dirs, scale, dates = scene()
    inputs, targets = [0, 1, 2], [14, 15, 16, 17, 18]
    for name in ("input_date", "input_pose"):
        chosen = predict_sample(name, inputs, targets, dates, centers, dirs, scale)
        assert len(chosen) == len(targets)
        assert set(chosen) <= set(inputs)


def test_pool_baseline_never_returns_a_sibling_target():
    # Targets share a date, so a sibling target would be an easy, unrealistic copy.
    centers, dirs, scale, dates = scene()
    inputs, targets = [0, 1, 2], [14, 15, 16, 17, 18]
    chosen = predict_sample("pool_date_pose", inputs, targets, dates, centers, dirs, scale)
    assert not set(chosen) & set(targets)


def test_pool_argument_restricts_candidates():
    centers, dirs, scale, dates = scene()
    allowed = [3, 4, 5, 6]     # e.g. the training split's rows
    chosen = predict_sample("pool_date_pose", [0, 1, 2], [14, 15], dates, centers, dirs, scale, pool=allowed)
    assert set(chosen) <= set(allowed)


# ----------------------------------------------------------------------------
# Scoring helpers
# ----------------------------------------------------------------------------

def test_psnr_identical_images_hits_the_cap():
    a = np.random.default_rng(0).uniform(-1, 1, (3, 4, 4))
    assert psnr_np(a, a) == pytest.approx(100.0)      # mse floor of 1e-10 -> 100 dB


def test_psnr_known_value():
    # A gap of 0.2 on the [-1, 1] scale is 0.1 on [0, 1]: mse = 0.01, so 20 dB.
    assert psnr_np(np.zeros((3, 4, 4)), np.full((3, 4, 4), 0.2)) == pytest.approx(20.0)


@pytest.mark.parametrize("gap,label", [(0.0, "<2y"), (1.9, "<2y"), (2.0, "2-10y"), (9.9, "2-10y"),
                                       (10.0, "10-50y"), (49.9, "10-50y"), (50.0, ">50y"), (400.0, ">50y")])
def test_gap_bucket(gap, label):
    assert gap_bucket(gap) == label


# ----------------------------------------------------------------------------
# End to end with a fake dataset
# ----------------------------------------------------------------------------

def fake_dataset(n_inputs=3, n_targets=5):
    centers, dirs, scale, dates = scene()
    c2ws = np.stack([c2w(p, np.arctan2(d[0], d[2])) for p, d in zip(centers, dirs)])
    views = pd.DataFrame({"file_key": [f"f{i}" for i in range(len(dates))], "date_mid": dates})

    def indices(i):
        # Deterministic per index, like DatedViewsDataset.indices.
        rng = np.random.default_rng((0, 0, i))
        return rng.choice(len(dates), n_inputs + n_targets, replace=False)

    return SimpleNamespace(views=views, dates=dates, c2ws=c2ws,
                           config=SimpleNamespace(n_inputs=n_inputs), indices=indices)


def fake_load(row):
    """A flat gray 'photo' whose brightness is a simple function of its date."""
    return np.full((3, 4, 4), (row["date_mid"] - 1800.0) / 250.0 - 1.0)


def test_run_baselines_output_shape_and_columns():
    ds = fake_dataset()
    table = run_baselines(ds, 6, fake_load)
    assert len(table) == 6 * 5 * len(BASELINES)                 # samples x targets x baselines
    assert set(table["baseline"]) == set(BASELINES)
    for col in ("sample", "view", "file_key", "target_year", "old_target_sample", "input_years",
                "retrieved_file_key", "retrieved_year", "date_gap", "pose_dist", "psnr"):
        assert col in table.columns
    assert set(table["view"]) == {3, 4, 5, 6, 7}                # target positions within a sample of 8


def test_run_baselines_values_are_consistent():
    ds = fake_dataset()
    table = run_baselines(ds, 4, fake_load)
    # date_gap really is |retrieved - target|
    assert np.allclose(table["date_gap"], (table["retrieved_year"] - table["target_year"]).abs())
    # PSNR matches a hand computation from the brightness-encodes-date images
    row = table.iloc[0]
    a = (row["retrieved_year"] - 1800.0) / 250.0 - 1.0
    b = (row["target_year"] - 1800.0) / 250.0 - 1.0
    assert row["psnr"] == pytest.approx(psnr_np(np.full(1, a), np.full(1, b)))


def test_run_baselines_pool_never_picks_a_target_of_its_own_sample():
    ds = fake_dataset()
    table = run_baselines(ds, 8, fake_load, baselines=("pool_date_pose",))
    for i, grp in table.groupby("sample"):
        assert not set(grp["retrieved_file_key"]) & set(grp["file_key"])


def test_input_date_beats_input_pose_on_date_gap():
    # By construction input_date minimizes the date gap among the inputs, so on
    # average its gap can never exceed the date-blind control's.
    table = run_baselines(fake_dataset(), 20, fake_load, baselines=("input_date", "input_pose"))
    mean_gap = table.groupby("baseline")["date_gap"].mean()
    assert mean_gap["input_date"] <= mean_gap["input_pose"]


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def test_summarize_structure():
    table = run_baselines(fake_dataset(), 10, fake_load)
    report = summarize(table)
    assert set(report["psnr_mean"]) == set(BASELINES)
    assert set(report["psnr_by_target_group"]["input_date"]) <= {"old_targets", "modern_targets"}
    assert report["n_targets"] == 10 * 5


def test_compare_with_seva_matches_on_sample_view_and_file():
    table = run_baselines(fake_dataset(), 3, fake_load, baselines=("input_date",))
    seva = table[["sample", "view", "file_key"]].copy()
    seva["psnr"] = table["psnr"] - 1.0                    # pretend SEVA is exactly 1 dB worse
    result = compare_with_seva(table, seva)["input_date"]
    assert result["matched_targets"] == len(table)
    assert result["retrieval_beats_seva"] == 1.0
    assert result["retrieval_psnr"] - result["seva_psnr"] == pytest.approx(1.0, abs=1e-3)


def test_compare_with_seva_mismatched_run_matches_nothing():
    # A different seed means different targets: nothing matches, rather than a wrong comparison.
    table = run_baselines(fake_dataset(), 3, fake_load, baselines=("input_date",))
    seva = table[["sample", "view", "file_key"]].copy()
    seva["file_key"] = "some_other_file"
    seva["psnr"] = 10.0
    assert compare_with_seva(table, seva) == {}
