"""Tests for the math in data_pipeline/plot_poses.py (no plotting needed).

Run with:   pytest tests/test_plot_poses.py
"""

import numpy as np

from data_pipeline.plot_poses import old_vs_modern_stats, plot_axes


def make_scene(seed=0):
    """400 modern cameras in a blob, plus 10 'normal' old ones and 5 absurdly far ones."""
    rng = np.random.default_rng(seed)
    modern = rng.normal(0, 1, size=(400, 3))
    old_near = rng.normal(0, 1, size=(10, 3))
    old_far = rng.normal(0, 1, size=(5, 3)) + 100.0   # 100 units away: clearly wrong
    centers = np.vstack([modern, old_near, old_far])
    years = np.concatenate([np.full(400, 2015.0), np.full(15, 1860.0)])
    return centers, years


def test_stats_flag_far_away_old_cameras():
    centers, years = make_scene()
    s = old_vs_modern_stats(centers, years)
    assert s["n_old"] == 15 and s["n_modern"] == 400
    assert s["too_far"] >= 5                      # the 5 distant ones are caught
    assert s["within_modern_range"] < 0.8         # so the "plausible" fraction drops


def test_stats_flag_cameras_placed_too_close_to_center():
    # The case our first (one-sided) check missed: old cameras sitting almost on
    # top of the scene center, e.g. inside the building. Modern cameras here form
    # a ring of radius ~5, so a camera at radius ~0.1 is implausible.
    rng = np.random.default_rng(3)
    angles = rng.uniform(0, 2 * np.pi, 400)
    ring = np.column_stack([5 * np.cos(angles), 5 * np.sin(angles), rng.normal(0, 0.1, 400)])
    near_center = rng.normal(0, 0.05, size=(6, 3))
    centers = np.vstack([ring, near_center])
    years = np.concatenate([np.full(400, 2015.0), np.full(6, 1860.0)])
    s = old_vs_modern_stats(centers, years)
    assert s["too_close"] == 6
    assert s["within_modern_range"] == 0.0


def test_stats_all_plausible():
    # If old and modern cameras come from the same distribution, most old ones
    # should land inside the modern band (about 90% in expectation).
    rng = np.random.default_rng(1)
    centers = rng.normal(0, 1, size=(300, 3))
    years = np.where(np.arange(300) < 10, 1900.0, 2015.0)
    s = old_vs_modern_stats(centers, years)
    assert s["within_modern_range"] > 0.6


def test_plot_axes_picks_largest_spread():
    # Cameras spread widely along x and z but barely along y: the 2D map should
    # use x and z (large spread) and ignore y.
    rng = np.random.default_rng(2)
    pts = np.column_stack([rng.normal(0, 10, 500), rng.normal(0, 0.1, 500), rng.normal(0, 5, 500)])
    xy = plot_axes(pts)
    assert xy.shape == (500, 2)
    assert xy[:, 0].std() > 8 and 4 < xy[:, 1].std() < 6
