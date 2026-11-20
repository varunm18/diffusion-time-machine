"""Tests for the MVGenMaster data module: the co-visibility sampler, the scene dataset and depth preparation."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

my_diffusers = pytest.importorskip("my_diffusers", reason="runs in MVGenMaster's environment, from its checkout")

# Under pytest the repo root is on the import path (its conftest.py), so ``src`` is the repo's own ``src``
# package and hides MVGenMaster's ``src`` folder, which ``scripts/mvgen/nvs_test.py`` (used by the depth
# script) imports from. Let the package search that folder too.
import src  # noqa: E402

MVGEN_SRC = str(Path(my_diffusers.__file__).resolve().parents[1] / "src")
if MVGEN_SRC not in list(src.__path__):
    src.__path__.append(MVGEN_SRC)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from timemachine.mvgen.data import (  # noqa: E402
    DEPTH_STATS_COLUMNS, MVGenScene, ViewSamplingConfig, anchor_probabilities, date_tolerance, depth_is_reliable,
    depth_path, draw_n_ref, draw_references, eligible_anchors, period_names, sample_rows, scene_rescale, shrink_factor,
    sparse_depth_points, target_partners,
)

REPO = Path(__file__).resolve().parents[1]


# --- sampler ---------------------------------------------------------------------
def two_sided_scene(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dates and co-visibility of 60 views of a building with two sides that share no points.

    Each side has 8 old, 12 modern and 10 post-fire views; views of one side share 0-199 points.
    """
    rng = np.random.default_rng(seed)
    side = np.repeat([0, 1], 30)
    dates = np.tile(np.concatenate([np.linspace(1890, 1894, 8), np.linspace(2012.0, 2012.9, 12),
                                    np.linspace(2020.0, 2020.9, 10)]), 2)
    shared = rng.integers(0, 200, (60, 60))
    covis = np.where(side[:, None] == side[None, :], np.minimum(shared, shared.T), 0).astype(np.uint16)
    np.fill_diagonal(covis, 500)
    return dates, covis


ALL = np.ones(60, dtype=bool)
NONE = np.zeros(60, dtype=bool)


def completed_samples(n: int, dates, covis, cfg, rng, target_ok, ref_ok, **kwargs) -> np.ndarray:
    """``[n, F]`` rows of ``n`` samples, drawing again whenever an anchor cannot be completed (as the scene does)."""
    samples = []
    while len(samples) < n:
        rows = sample_rows(dates, covis, cfg, rng, target_ok, ref_ok, **kwargs)
        if rows is not None:
            samples.append(rows)
    return np.stack(samples)


def seen_by_a_reference(covis: np.ndarray, rows: np.ndarray, n_ref: int, min_shared: int = 30) -> np.ndarray:
    """``[T]`` bool: whether each target of a sample shares ``min_shared`` points with at least one reference."""
    return (covis[np.ix_(rows[:n_ref], rows[n_ref:])] >= min_shared).any(axis=0)


def test_targets_share_a_date_group_and_references_see_them():
    dates, covis = two_sided_scene()
    cfg = ViewSamplingConfig()
    period = np.digitize(dates, cfg.period_edges)
    rng = np.random.default_rng(0)
    for _ in range(300):
        rows = sample_rows(dates, covis, cfg, rng, ALL, ALL)
        refs, targets = rows[:cfg.n_ref], rows[cfg.n_ref:]
        anchor = targets[0]
        assert rows.dtype == np.int64 and len(set(rows)) == cfg.n_views == 8
        assert np.all(np.abs(dates[targets] - dates[anchor]) <= date_tolerance(cfg, dates[anchor]))
        assert np.all(period[targets] == period[anchor])
        assert np.all(covis[anchor, targets[1:]] >= cfg.min_shared)
        assert np.all((covis[np.ix_(refs, targets)] >= cfg.min_shared).sum(axis=1) >= cfg.min_targets_seen)
        assert seen_by_a_reference(covis, rows, cfg.n_ref).all()


def test_targets_do_not_straddle_a_period_edge():
    # six views just before the fire and six just after, all within a year and co-visible, plus references
    dates = np.concatenate([np.linspace(2018.9, 2019.25, 6), np.linspace(2019.3, 2019.8, 6), np.full(6, 2010.0)])
    covis = np.full((18, 18), 100, dtype=np.uint16)
    cfg, ok = ViewSamplingConfig(), np.ones(18, dtype=bool)
    rng = np.random.default_rng(0)
    for _ in range(100):
        targets = sample_rows(dates, covis, cfg, rng, ok, ok)[cfg.n_ref:]
        assert len(set(dates[targets] >= 2019.2877)) == 1


def test_eligible_anchors_match_a_count_by_hand():
    dates, covis = two_sided_scene()
    cfg = ViewSamplingConfig()
    target_ok = np.random.default_rng(1).random(60) < 0.7
    period = np.digitize(dates, cfg.period_edges)
    expected = []
    for a in np.flatnonzero(target_ok):
        tolerance = 1.0 if dates[a] >= 1990 else 5.0
        partners = [b for b in np.flatnonzero(target_ok) if b != a and abs(dates[b] - dates[a]) <= tolerance
                    and period[b] == period[a] and covis[a, b] >= 30]
        assert np.flatnonzero(target_partners(dates, covis, cfg, target_ok, [a])[0]).tolist() == partners
        if len(partners) >= 4:
            expected.append(a)
    assert 0 < len(expected) < target_ok.sum()          # the case tells eligible views from the others
    assert eligible_anchors(dates, covis, cfg, target_ok).tolist() == expected


def test_targets_come_from_the_views_most_covisible_with_the_anchor():
    dates, covis = two_sided_scene()
    cfg = ViewSamplingConfig(top_k=4)                   # exactly the n_targets - 1 best partners
    rng = np.random.default_rng(0)
    for _ in range(100):
        rows = sample_rows(dates, covis, cfg, rng, ALL, ALL)
        anchor, others = rows[cfg.n_ref], rows[cfg.n_ref + 1:]
        partners = np.flatnonzero(target_partners(dates, covis, cfg, ALL, [anchor])[0])
        assert sorted(covis[anchor, others]) == sorted(covis[anchor, partners])[-4:]


def test_references_come_from_another_period_at_the_configured_rate():
    dates, covis = two_sided_scene()
    period = np.digitize(dates, (1970.0, 2019.2877))
    rate = {}
    for prob in (0.0, 0.5, 1.0):
        cfg = ViewSamplingConfig(cross_period_prob=prob)
        rng = np.random.default_rng(1)
        samples = [sample_rows(dates, covis, cfg, rng, ALL, ALL) for _ in range(600)]
        rate[prob] = np.mean([np.all(period[rows[:cfg.n_ref]] != period[rows[cfg.n_ref]]) for rows in samples])
    assert rate[1.0] == 1.0                             # here views of other periods can always show the targets
    assert rate[0.0] < 0.8                              # by chance only
    assert rate[0.5] == pytest.approx(0.5 + 0.5 * rate[0.0], abs=0.07)


def test_references_fall_back_to_all_candidates_without_enough_views_of_other_periods():
    dates, covis = two_sided_scene()
    period = np.digitize(dates, (1970.0, 2019.2877))
    cfg = ViewSamplingConfig(cross_period_prob=1.0, anchor_period_weights=(0, 1, 0))
    ref_ok = period == 1
    ref_ok[[0, 1]] = True                               # only two views of another period: fewer than n_ref
    rng = np.random.default_rng(0)
    for rows in completed_samples(100, dates, covis, cfg, rng, ALL, ref_ok):
        refs = rows[:cfg.n_ref]
        assert ref_ok[refs].all() and (period[refs] == 1).any()
    one_period = ViewSamplingConfig(cross_period_prob=1.0, period_edges=())
    assert sample_rows(dates, covis, one_period, rng, ALL, ALL) is not None


def test_references_fall_back_when_the_views_of_other_periods_cannot_show_every_target():
    # Rows 0-4 are the modern targets. Three old views (5-7) see targets 0-3 only, so no set of them shows
    # target 4; three modern views (8-10) see all five.
    dates = np.array([2012.0] * 5 + [1900.0] * 3 + [2012.5] * 3)
    covis = np.full((11, 11), 100, dtype=np.uint16)
    covis[5:8, 4] = covis[4, 5:8] = 0
    target_ok = np.arange(11) < 5
    cfg = ViewSamplingConfig(cross_period_prob=1.0)
    rng = np.random.default_rng(0)
    samples = completed_samples(200, dates, covis, cfg, rng, target_ok, ~target_ok)
    assert all(seen_by_a_reference(covis, rows, cfg.n_ref).all() for rows in samples)
    assert np.isin(samples[:, :cfg.n_ref], [8, 9, 10]).any(axis=1).all()        # never the three old views alone
    assert np.isin(samples[:, :cfg.n_ref], [5, 6, 7]).any()                     # they still serve next to a modern one
    covis[5:8, 4] = covis[4, 5:8] = 100                                         # now they do show every target
    samples = completed_samples(50, dates, covis, cfg, rng, target_ok, ~target_ok)
    assert np.isin(samples[:, :cfg.n_ref], [5, 6, 7]).all()


def test_only_allowed_views_are_drawn():
    dates, covis = two_sided_scene()
    allowed = np.random.default_rng(5)
    target_ok = allowed.random(60) < 0.8                # the others stand for held-out or unusable views
    ref_ok = target_ok & (allowed.random(60) < 0.7)     # the others stand for views without reliable depth
    cfg = ViewSamplingConfig()
    rng = np.random.default_rng(0)
    samples = [sample_rows(dates, covis, cfg, rng, target_ok, ref_ok) for _ in range(300)]
    samples = np.stack([rows for rows in samples if rows is not None])
    assert len(samples) > 200
    assert ref_ok[samples[:, :cfg.n_ref]].all() and target_ok[samples[:, cfg.n_ref:]].all()
    assert all(seen_by_a_reference(covis, rows, cfg.n_ref).all() for rows in samples)
    assert not ref_ok[samples[:, cfg.n_ref:]].all()     # a view without reliable depth can still be a target


def test_none_when_no_sample_exists():
    dates, covis = two_sided_scene()
    cfg = ViewSamplingConfig()
    rng = np.random.default_rng(0)
    assert sample_rows(dates, covis, cfg, rng, NONE, ALL) is None                     # nothing may be a target
    assert sample_rows(dates, np.zeros_like(covis), cfg, rng, ALL, ALL) is None       # nothing is co-visible
    assert sample_rows(np.arange(60) * 20.0, covis, cfg, rng, ALL, ALL) is None       # no two views share a date
    assert sample_rows(dates, covis, cfg, rng, ALL, NONE) is None                     # nothing may be a reference
    two_refs = NONE.copy()
    two_refs[[0, 1]] = True
    assert sample_rows(dates, covis, cfg, rng, ALL, two_refs) is None                 # fewer references than n_ref


def test_anchor_period_weights():
    dates, covis = two_sided_scene()
    period = np.digitize(dates, (1970.0, 2019.2877))

    def anchor_periods(weights, target_ok=ALL):
        cfg = ViewSamplingConfig(anchor_period_weights=weights)
        rng = np.random.default_rng(2)
        return np.array([period[sample_rows(dates, covis, cfg, rng, target_ok, ALL)[cfg.n_ref]] for _ in range(400)])

    assert np.mean(anchor_periods(None) == 0) < 0.4                 # uniform over anchors: 16 of 60 views are old
    assert np.mean(anchor_periods((0.5, 0.5, 0)) == 0) == pytest.approx(0.5, abs=0.09)
    assert set(anchor_periods((0, 0, 1))) == {2}
    not_old = period != 0                                           # a period without anchors is skipped
    assert set(anchor_periods((0.5, 0.5, 0), not_old)) == {1}
    only_old = ViewSamplingConfig(anchor_period_weights=(1, 0, 0))
    assert sample_rows(dates, covis, only_old, np.random.default_rng(0), not_old, ALL) is None
    anchors = eligible_anchors(dates, covis, only_old, ALL)
    assert anchor_probabilities(anchors, period, only_old).sum() == pytest.approx(1.0)
    assert anchor_probabilities(anchors, period, only_old)[period[anchors] != 0].max() == 0


def test_references_are_drawn_in_proportion_to_the_targets_they_see():
    # Rows 0-4 are the only possible targets. Row 5 sees all five of them, row 6 targets 2-4 and row 7 targets
    # 0-2, so any two of the three show every target and no draw is rejected.
    dates = np.full(8, 2012.0)
    covis = np.full((8, 8), 100, dtype=np.uint16)
    covis[6, :2] = covis[:2, 6] = 0
    covis[7, 3:5] = covis[3:5, 7] = 0
    target_ok = np.arange(8) < 5
    rng = np.random.default_rng(0)
    cfg = ViewSamplingConfig(n_ref=2, n_targets=5)
    refs = np.stack([sample_rows(dates, covis, cfg, rng, target_ok, ~target_ok)[:2] for _ in range(1500)])
    # Weights 5 : 3 : 3 without replacement: row 5 is left out with probability 6/11 x 3/8.
    assert np.mean((refs == 5).any(axis=1)) == pytest.approx(1 - 6 / 11 * 3 / 8, abs=0.04)
    strict = ViewSamplingConfig(n_ref=1, n_targets=5, min_targets_seen=4)
    assert {sample_rows(dates, covis, strict, rng, target_ok, ~target_ok)[0] for _ in range(50)} == {5}


def test_every_target_is_seen_by_a_reference():
    # Rows 0-4 are the targets. Each candidate reference sees three of them: rows 5 and 6 targets 0-2, row 7
    # targets 2-4, row 8 targets 0, 1 and 3. Of the four possible sets of three, {5, 6, 8} leaves target 4 unseen.
    dates = np.full(9, 2012.0)
    covis = np.full((9, 9), 100, dtype=np.uint16)
    for ref, unseen in ((5, [3, 4]), (6, [3, 4]), (7, [0, 1]), (8, [2, 4])):
        covis[ref, unseen] = covis[unseen, ref] = 0
    target_ok = np.arange(9) < 5
    cfg, rng = ViewSamplingConfig(), np.random.default_rng(0)
    samples = completed_samples(600, dates, covis, cfg, rng, target_ok, ~target_ok)
    assert all(seen_by_a_reference(covis, rows, cfg.n_ref).all() for rows in samples)
    sets = {tuple(sorted(rows[:cfg.n_ref])) for rows in samples}
    assert sets == {(5, 6, 7), (5, 7, 8), (6, 7, 8)}                # every set that shows all targets, and no other
    counts = [np.mean([tuple(sorted(rows[:cfg.n_ref])) == s for rows in samples]) for s in sorted(sets)]
    assert counts == pytest.approx([1 / 3] * 3, abs=0.06)           # equal weights: the sets left are equally likely
    covis[7, 4] = covis[4, 7] = 0                                   # now nobody sees target 4
    assert sample_rows(dates, covis, cfg, rng, target_ok, ~target_ok) is None


def test_draw_references():
    sees = np.zeros((40, 5), dtype=bool)
    sees[:, :4] = True                                              # everyone sees targets 0-3 ...
    sees[7, 4] = True                                               # ... and only view 7 sees target 4
    pool, rng = np.arange(40), np.random.default_rng(0)
    assert draw_references(pool[:2], sees, 3, rng) is None          # fewer views than references
    assert draw_references(pool[8:], sees, 3, rng) is None          # nobody in the pool sees target 4
    assert sorted(draw_references(pool[5:8], sees, 3, rng)) == [5, 6, 7]
    # One reference must be view 7, which a draw picks with probability 5/161: the redraws are limited, so
    # some calls give up although a set exists (the caller then starts again from another anchor).
    single = [draw_references(pool, sees, 1, rng) for _ in range(200)]
    found = [refs for refs in single if refs is not None]
    assert 40 < len(found) < 160 and all(refs.tolist() == [7] for refs in found)


def test_sampling_is_deterministic_and_takes_precomputed_anchors():
    dates, covis = two_sided_scene()
    cfg = ViewSamplingConfig()
    anchors = eligible_anchors(dates, covis, cfg, ALL)
    first, second, third = (np.random.default_rng(3) for _ in range(3))
    for _ in range(20):
        rows = sample_rows(dates, covis, cfg, first, ALL, ALL)
        assert np.array_equal(rows, sample_rows(dates, covis, cfg, second, ALL, ALL))
        assert np.array_equal(rows, sample_rows(dates, covis, cfg, third, ALL, ALL, anchors=anchors))


def test_sample_rows_takes_the_number_of_references():
    dates, covis = two_sided_scene()
    cfg, rng = ViewSamplingConfig(), np.random.default_rng(0)
    period = np.digitize(dates, cfg.period_edges)
    for n_ref in (1, 2, 3, 4):
        for rows in completed_samples(40, dates, covis, cfg, rng, ALL, ALL, n_ref=n_ref):
            assert len(set(rows)) == cfg.n_views == 8                   # always eight views: fewer references,
            targets = rows[n_ref:]                                      # more targets
            assert np.all(period[targets] == period[targets[0]])
            assert np.all(covis[targets[0], targets[1:]] >= cfg.min_shared)
            assert seen_by_a_reference(covis, rows, n_ref).all()
    # Seven targets need an anchor with six partners: fewer views have that many.
    for_seven, for_five = (eligible_anchors(dates, covis, cfg, ALL, n_targets=n) for n in (7, None))
    assert 0 < len(for_seven) < len(for_five) and np.isin(for_seven, for_five).all()
    a, b = np.random.default_rng(4), np.random.default_rng(4)          # the default is the config's n_ref
    assert np.array_equal(sample_rows(dates, covis, cfg, a, ALL, ALL),
                          sample_rows(dates, covis, cfg, b, ALL, ALL, n_ref=3))


def test_draw_n_ref():
    fixed, rng = ViewSamplingConfig(), np.random.default_rng(0)
    state = rng.bit_generator.state
    assert draw_n_ref(fixed, rng) == 3 and rng.bit_generator.state == state    # nothing is drawn
    assert (fixed.ref_counts, fixed.target_counts) == ((3,), (5,))
    varied = ViewSamplingConfig(n_ref_choices=(1, 2, 3))
    assert (varied.n_views, varied.ref_counts, varied.target_counts) == (8, (1, 2, 3), (5, 6, 7))
    drawn = np.array([draw_n_ref(varied, rng) for _ in range(1200)])
    assert [np.mean(drawn == k) for k in (1, 2, 3)] == pytest.approx([1 / 3] * 3, abs=0.05)
    assert isinstance(draw_n_ref(varied, rng), int)
    mostly_three = ViewSamplingConfig(n_ref_choices=(1, 3, 3, 3))              # a repeated entry is drawn more often
    assert np.mean([draw_n_ref(mostly_three, rng) == 3 for _ in range(1200)]) == pytest.approx(0.75, abs=0.05)
    assert mostly_three.target_counts == (5, 7)


@pytest.mark.parametrize("kwargs", [
    dict(n_ref=0), dict(n_targets=0), dict(min_shared=0), dict(min_targets_seen=0), dict(min_targets_seen=6),
    dict(top_k=3), dict(old_tolerance=-1.0), dict(period_edges=(2019.2877, 1970.0)), dict(cross_period_prob=1.5),
    dict(anchor_period_weights=(1.0, 1.0)), dict(anchor_period_weights=(0.0, 0.0, 0.0)),
    dict(anchor_period_weights=(1.0, -1.0, 1.0)),
    dict(n_ref_choices=()), dict(n_ref_choices=(0, 1)), dict(n_ref_choices=(1, 8)), dict(n_ref_choices=(1.5,)),
    dict(n_ref_choices=(True,)),
    dict(n_ref_choices=(6,)),                   # two targets left: fewer than min_targets_seen
    dict(n_ref_choices=(1,), top_k=5),          # seven targets: six are drawn next to the anchor, more than top_k
])
def test_config_is_validated(kwargs):
    with pytest.raises(ValueError):
        ViewSamplingConfig(**kwargs)


# --- a tiny dataset folder with known geometry ---------------------------------------
N, S = 24, 32               # views; crop size
CROP_K = [[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]]
PLANE_Z = 4.0
FAR_VIEW, AWAY_VIEW = 22, 23
MONO = 2.0                  # what the fake Depth-Pro predicts
SCENE_EXTENT = 20.0         # longest side of the box of ALL the model's cameras (scene.json): wider than the table's
SCENE_FACTOR = 5.0 / SCENE_EXTENT


def stored_depth(row: int) -> float:
    """The (constant) depth map the fixture stores for a view; exact in float16."""
    return 1.0 + 0.125 * row


def write_dataset(root: Path) -> SimpleNamespace:
    """Write a dataset folder of 24 views and return what the tests need to know about it.

    * Cameras: 96 x 64 pixels with focal length 80 and a centred principal point; the stored images
      are 48 x 32, so the 32-pixel centre crop has the intrinsics ``CROP_K``. Camera i sits at
      (0.4 i, 0.1 (i mod 3), 0) and looks along +z, turned by 0.01 i about its y axis. View 22 sits
      at the far edge of the points; view 23 looks away from them. Image i is plain, with red = 10 i.
    * Dates: rows 0-7 are old (1900-1903.5), 8-15 modern (2012), 16-23 after the fire (2020).
    * Rows 5 and 12 are not usable. All views share 100 points, except 22 and 23, which share none.
    * COLMAP points: a grid on the plane z = 4; a view "observes" those within 3 units of it in x
      (more than its crop shows; for view 23 they lie behind the camera).
    * Depth: a constant map per usable view (none for view 23); not ``ok`` for views 9, 22 and 23.
    * ``scene.json``: the COLMAP model's 40 cameras span ``SCENE_EXTENT`` along x, more than these 24.
    """
    (root / "images").mkdir(parents=True)
    (root / "depth").mkdir()
    index = np.arange(N)
    centres = np.column_stack([0.4 * index, 0.1 * (index % 3), np.zeros(N)])
    yaws = 0.01 * index
    centres[FAR_VIEW], yaws[FAR_VIEW] = [13.3, 0.0, 0.0], 0.0
    centres[AWAY_VIEW], yaws[AWAY_VIEW] = [4.0, 0.0, 0.0], np.pi
    c2w = np.tile(np.eye(4), (N, 1, 1))
    c2w[:, 0, 0] = c2w[:, 2, 2] = np.cos(yaws)          # camera axes in the world: x right, y down, z forward
    c2w[:, 0, 2], c2w[:, 2, 0] = np.sin(yaws), -np.sin(yaws)
    c2w[:, :3, 3] = centres
    date_mid = np.concatenate([1900 + 0.5 * np.arange(8), 2012 + 0.1 * np.arange(8), 2020 + 0.1 * np.arange(8)])
    usable = ~np.isin(index, [5, 12])
    rows = []
    for i in index:
        path = root / "images" / f"view_{i:02d}.png"
        Image.fromarray(np.full((32, 48, 3), (10 * i, 100, 200), dtype=np.uint8)).save(path)
        t = -c2w[i, :3, :3].T @ centres[i]              # COLMAP stores world-to-camera: x_cam = R x_world + t
        rows.append(dict(
            scene_id=1, model=0, file_key=path.name, image_path=str(path), image_width=48, image_height=32,
            camera_width=96, camera_height=64, fx=80.0, fy=80.0, cx=48.0, cy=32.0, k1=0.0, k2=0.0,
            qw=np.cos(yaws[i] / 2), qx=0.0, qy=-np.sin(yaws[i] / 2), qz=0.0, tx=t[0], ty=t[1], tz=t[2],
            date_lo=date_mid[i] - 0.25, date_hi=date_mid[i] + 0.25, date_mid=date_mid[i],
            rolled=False, duplicate_of=-1, usable=bool(usable[i])))
    views = pd.DataFrame(rows)
    views.to_parquet(root / "views.parquet", index=False)

    covis = np.full((N, N), 100, dtype=np.uint16)
    covis[FAR_VIEW:], covis[:, FAR_VIEW:] = 0, 0
    np.fill_diagonal(covis, 400)
    np.save(root / "covisibility.npy", covis)

    xs, ys = np.meshgrid(np.linspace(-2, 12, 60), np.linspace(-1.5, 1.5, 12))
    xyz = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, PLANE_Z)])
    ids = 10 + 3 * np.arange(len(xyz))                  # sorted, with gaps, as COLMAP's ids
    observed = [np.flatnonzero(np.abs(xyz[:, 0] - centres[i, 0]) < 3) for i in index]
    np.savez(root / "sparse_points.npz", p3d_ids=ids, p3d_xyz=xyz, view_pid=np.concatenate([ids[o] for o in observed]),
             view_starts=np.concatenate([[0], np.cumsum([len(o) for o in observed])]))

    with_depth = usable & (index != AWAY_VIEW)
    for i in np.flatnonzero(with_depth):
        np.save(depth_path(root, i), np.full((S, S), stored_depth(i), dtype=np.float16))
    depth_ok = with_depth & ~np.isin(index, [9, FAR_VIEW])
    stats = pd.DataFrame({"row": index[usable], "file_key": views.file_key[usable].to_numpy(), "n_points": 100,
                          "scale": 1.0, "median_rel_error": 0.05, "ok": depth_ok[usable]})
    stats.to_parquet(root / "depth_stats.parquet", index=False)
    write_scene_file(root, SCENE_EXTENT)
    return SimpleNamespace(dir=root, c2w=c2w, date_mid=date_mid, usable=usable, xyz=xyz, observed=observed)


def write_scene_file(root: Path, extent) -> None:
    """``scene.json`` as the build script writes it, for a model whose cameras span ``extent`` along x."""
    (root / "scene.json").write_text(json.dumps({
        "scene_id": 1, "model": 0, "n_registered_cameras": 40, "camera_min": [0.0, 0.0, 0.0],
        "camera_max": [extent, 0.2, 0.0], "camera_extent": extent}))


def crop_points(folder: SimpleNamespace, row: int) -> np.ndarray:
    """``(u, v, z)`` of the points view ``row`` observes inside its crop, worked out point by point."""
    centre, axes = folder.c2w[row, :3, 3], folder.c2w[row, :3, :3]
    points = []
    for xyz in folder.xyz[folder.observed[row]]:
        x, y, z = (xyz - centre) @ axes                 # camera coordinates
        if z > 0 and 0 <= 40 * x / z + 16 < S and 0 <= 40 * y / z + 16 < S:
            points.append((40 * x / z + 16, 40 * y / z + 16, z))
    return np.array(points, dtype=np.float32).reshape(-1, 3)


def drawn_rows(scene: MVGenScene, n: int = 200) -> np.ndarray:
    """``[n, 8]`` the rows of the scene's first ``n`` samples."""
    return np.stack([scene.indices(i) for i in range(n)])


@pytest.fixture
def folder(tmp_path) -> SimpleNamespace:
    return write_dataset(tmp_path / "scene")


# --- scene dataset -----------------------------------------------------------------
def test_sample_fields(folder):
    scene = MVGenScene(folder.dir, resolution=S)
    rows, sample = scene.indices(0), scene[0]
    assert set(sample) == {"images", "K", "w2c", "depth", "n_ref", "scale", "date", "date_mid", "rows", "file_keys"}
    shapes = {"images": (8, 3, S, S), "K": (8, 3, 3), "w2c": (8, 4, 4), "depth": (8, 1, S, S), "date": (8, 2),
              "date_mid": (8,)}
    for key, shape in shapes.items():
        assert sample[key].shape == shape and sample[key].dtype == torch.float32, key
    assert sample["n_ref"] == 3 and isinstance(sample["n_ref"], int)
    assert sample["scale"] == SCENE_FACTOR and isinstance(sample["scale"], float)
    assert sample["rows"].dtype == torch.int64 and sample["rows"].tolist() == rows.tolist()
    assert sample["file_keys"] == [f"view_{row:02d}.png" for row in rows]
    assert sample["images"].min() >= -1 and sample["images"].max() <= 1
    red = torch.tensor(10 * rows / 127.5 - 1, dtype=torch.float32)           # each view's own image
    torch.testing.assert_close(sample["images"][:, 0].mean(dim=(1, 2)), red, atol=0.01, rtol=0)
    torch.testing.assert_close(sample["K"], torch.tensor(CROP_K).expand(8, 3, 3))
    dates = torch.tensor(folder.date_mid[rows], dtype=torch.float32)
    torch.testing.assert_close(sample["date_mid"], dates)
    torch.testing.assert_close(sample["date"], torch.stack([dates - 0.25, dates + 0.25], dim=1))


def test_scene_attributes(folder):
    scene = MVGenScene(folder.dir, resolution=S)
    assert len(scene) == 1000 and len(scene.views) == N and scene.covis.shape == (N, N)
    np.testing.assert_array_equal(scene.dates, folder.date_mid)
    assert scene.period.tolist() == [0] * 8 + [1] * 8 + [2] * 8
    assert not scene.heldout.any()
    np.testing.assert_array_equal(scene.target_ok, folder.usable)
    np.testing.assert_array_equal(scene.ref_ok, folder.usable & ~np.isin(np.arange(N), [9, FAR_VIEW, AWAY_VIEW]))
    np.testing.assert_allclose(scene.c2ws, folder.c2w, atol=1e-12)
    assert np.isin(scene.anchors, np.flatnonzero(folder.usable)[:-2]).all() and len(scene.anchors) == 20


def test_cameras_are_world_to_camera_in_opencv_axes(folder):
    scene = MVGenScene(folder.dir, resolution=S, scale=1.0)            # COLMAP units, to compare with the points
    rows = np.array([0, 1, 2, 3])
    sample = scene.make_sample(rows, n_ref=2)
    np.testing.assert_allclose(sample["w2c"].numpy() @ folder.c2w[rows], np.tile(np.eye(4), (4, 1, 1)), atol=1e-6)
    for j, row in enumerate(rows):                      # the COLMAP points land where the crop shows them
        points = crop_points(folder, row)
        world = np.column_stack([folder.xyz[folder.observed[row]], np.ones(len(folder.observed[row]))])
        cam = world @ sample["w2c"][j].numpy().T
        uvz = cam[:, :3] @ sample["K"][j].numpy().T
        inside = (uvz[:, :2] / uvz[:, 2:] >= 0).all(axis=1) & (uvz[:, :2] / uvz[:, 2:] < S).all(axis=1)
        np.testing.assert_allclose(uvz[inside, :2] / uvz[inside, 2:], points[:, :2], atol=1e-3)
        np.testing.assert_allclose(cam[inside, 2], points[:, 2], atol=1e-4)


NEAR, FAR = np.array([0, 1, 2, 3]), np.array([0, 20, 1, 21])           # camera centres span 1.2 and 8.4 along x


@pytest.mark.parametrize("scale, setting, constant, near_factor, far_factor", [
    ("scene", "scene", SCENE_FACTOR, SCENE_FACTOR, SCENE_FACTOR),      # one factor, from all the model's cameras
    ("sample", "sample", None, 1.0, 5.0 / 8.4),                        # each sample's own cameras decide
    (0.5, 0.5, 0.5, 0.5, 0.5),                                         # a given constant
    (2, 2.0, 2.0, 2.0, 2.0),
])
def test_scene_scale_rule(folder, scale, setting, constant, near_factor, far_factor):
    scene = MVGenScene(folder.dir, resolution=S, scale=scale)
    assert scene.scale == setting and scene.scale_factor == constant
    for rows, factor in ((NEAR, near_factor), (FAR, far_factor)):
        sample = scene.make_sample(rows, n_ref=2)
        assert sample["scale"] == pytest.approx(factor, rel=1e-12) and isinstance(sample["scale"], float)
        c2w = np.linalg.inv(sample["w2c"].numpy().astype(np.float64))
        np.testing.assert_allclose(c2w[:, :3, 3], folder.c2w[rows, :3, 3] * factor, atol=1e-5)
        np.testing.assert_allclose(c2w[:, :3, :3], folder.c2w[rows, :3, :3], atol=1e-6)
        for j in range(2):                              # references: the stored depth, in the rescaled units
            torch.testing.assert_close(sample["depth"][j], torch.full((1, S, S), stored_depth(rows[j]) * factor))
        assert sample["depth"][2:].abs().max() == 0     # targets: no depth
    if scale == "sample":
        assert np.ptp(c2w[:, :3, 3], axis=0).max() == pytest.approx(5.0, abs=1e-5)


def test_scale_follows_camera_longest_side(folder):
    tight = MVGenScene(folder.dir, resolution=S, camera_longest_side=1.0, scale="sample").make_sample(NEAR, n_ref=2)
    torch.testing.assert_close(tight["depth"][1], torch.full((1, S, S), stored_depth(1) / 1.2))
    assert MVGenScene(folder.dir, resolution=S, camera_longest_side=1.0).scale_factor == 1.0 / SCENE_EXTENT
    assert MVGenScene(folder.dir, resolution=S, camera_longest_side=1.0, scale=0.5).scale_factor == 0.5
    write_scene_file(folder.dir, 4.0)                   # a scene smaller than the limit is not enlarged
    small = MVGenScene(folder.dir, resolution=S)
    assert small.scale_factor == 1.0 and small.make_sample(FAR, n_ref=2)["scale"] == 1.0


def test_the_scene_scale_needs_the_scene_file(folder):
    (folder.dir / "scene.json").unlink()
    with pytest.raises(FileNotFoundError, match="build_dataset"):
        MVGenScene(folder.dir, resolution=S)
    assert MVGenScene(folder.dir, resolution=S, scale="sample").scale_factor is None      # the other rules do not
    assert MVGenScene(folder.dir, resolution=S, scale=0.2).scale_factor == 0.2
    for extent in (0.0, -1.0, None, "wide", float("nan"), True):
        write_scene_file(folder.dir, extent)
        with pytest.raises(ValueError, match="camera_extent"):
            MVGenScene(folder.dir, resolution=S)


@pytest.mark.parametrize("scale", ["scenes", "0.5", "", 0, 0.0, -1.0, float("nan"), float("inf"), True, None])
def test_scale_setting_is_validated(folder, scale):
    with pytest.raises(ValueError, match="scale must be"):
        MVGenScene(folder.dir, resolution=S, scale=scale)


def test_shrink_factor_and_scene_rescale():
    assert shrink_factor(57.5721, 5.0) == pytest.approx(0.086848, rel=1e-5)    # Notre-Dame, all 5,115 cameras
    assert shrink_factor(5.0, 5.0) == 1.0 and shrink_factor(0.6, 5.0) == 1.0   # small sets are not enlarged
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 8.0, 2.0], [-1.0, 3.0, 0.5]])
    assert scene_rescale(centres, 5.0) == pytest.approx(5.0 / 8.0)      # the longest axis decides
    assert scene_rescale(centres, 8.0) == 1.0
    assert scene_rescale(centres * 0.01, 5.0) == 1.0


def test_samples_are_deterministic_per_seed_epoch_and_index(folder):
    scene, twin = MVGenScene(folder.dir, resolution=S, seed=3), MVGenScene(folder.dir, resolution=S, seed=3)
    first = drawn_rows(scene, 20)
    np.testing.assert_array_equal(first, drawn_rows(twin, 20))
    a, b = scene[4], twin[4]
    assert all(torch.equal(a[key], b[key]) for key in ("images", "K", "w2c", "depth", "date", "date_mid", "rows"))
    assert a["file_keys"] == b["file_keys"] and (a["n_ref"], a["scale"]) == (b["n_ref"], b["scale"])
    scene.set_epoch(1)
    assert not np.array_equal(first, drawn_rows(scene, 20))
    np.testing.assert_array_equal(first, np.stack([scene.indices(i, epoch=0) for i in range(20)]))
    other = MVGenScene(folder.dir, resolution=S, seed=4)
    assert not np.array_equal(first, drawn_rows(other, 20))
    np.testing.assert_array_equal(first, np.stack([other.indices(i, seed=3) for i in range(20)]))


def test_scene_draws_only_usable_views_and_references_with_reliable_depth(folder):
    drawn = drawn_rows(MVGenScene(folder.dir, resolution=S))
    refs, targets = drawn[:, :3], drawn[:, 3:]
    assert not np.isin(drawn, [5, 12]).any()                        # not usable
    assert not np.isin(drawn, [FAR_VIEW, AWAY_VIEW]).any()          # co-visible with nothing
    assert 9 not in refs and 9 in targets                           # depth not ok: a target only
    scene = MVGenScene(folder.dir, resolution=S)
    assert all(seen_by_a_reference(scene.covis, rows, 3).all() for rows in drawn)
    assert 9 in drawn_rows(MVGenScene(folder.dir, resolution=S, require_depth=False))[:, :3]


def test_hold_out(folder):
    scene = MVGenScene(folder.dir, resolution=S)
    scene.hold_out([0, 17])
    assert np.flatnonzero(scene.heldout).tolist() == [0, 17]
    assert not np.isin(drawn_rows(scene), [0, 17]).any()
    sample = scene.make_sample(np.array([1, 2, 0, 17]), n_ref=2)    # held-out views can still be loaded
    assert sample["rows"].tolist() == [1, 2, 0, 17]
    scene.hold_out([3])                                             # replaces the earlier hold-out
    assert np.flatnonzero(scene.heldout).tolist() == [3]
    drawn = drawn_rows(scene)
    assert 0 in drawn and 17 in drawn and 3 not in drawn
    with pytest.raises(ValueError, match="No sample can be drawn"):
        scene.hold_out(np.arange(N))


def test_fingerprint_follows_what_decides_the_samples(folder):
    """A resumed run compares it with its checkpoint's: it changes with the folder's content and with nothing else."""
    def fingerprint(**settings) -> str:
        return MVGenScene(folder.dir, resolution=S, **settings).fingerprint()

    first = fingerprint()
    assert len(first) == 64 and int(first, 16) >= 0 and fingerprint() == first
    assert fingerprint(seed=5, samples_per_epoch=10, scale=0.5, camera_longest_side=1.0) == first
    scene = MVGenScene(folder.dir, resolution=S)
    scene.hold_out([0, 17])                                         # the hold-out follows from the run's settings
    assert scene.fingerprint() == first
    write_scene_file(folder.dir, 4.0)                               # the scale factor is compared on its own
    assert fingerprint() == first
    without_depth = fingerprint(require_depth=False)                # every usable view can then be a reference
    assert without_depth != first

    views = pd.read_parquet(folder.dir / "views.parquet")

    def rebuilt(column: str, value, row: int = 3) -> str:
        """The fingerprint of the folder with one entry of its view table changed (the table is then put back)."""
        changed = views.copy()
        changed.loc[row, column] = value
        changed.to_parquet(folder.dir / "views.parquet", index=False)
        try:
            return fingerprint()
        finally:
            views.to_parquet(folder.dir / "views.parquet", index=False)

    changed = [rebuilt("usable", False), rebuilt("date_mid", 1950.0), rebuilt("date_hi", 1960.0), rebuilt("tx", 9.0),
               rebuilt("fx", 81.0), rebuilt("file_key", "another.png", row=5)]      # view 5 has no depth statistics
    assert len({first, without_depth, *changed}) == 2 + len(changed)        # each gives a fingerprint of its own
    assert rebuilt("image_path", "/moved/view_03.png") == first    # where the files lie does not matter
    assert fingerprint() == first

    stats = pd.read_parquet(folder.dir / "depth_stats.parquet")
    stats.loc[stats["row"] == 3, "ok"] = False                      # view 3 cannot be a reference any more
    stats.to_parquet(folder.dir / "depth_stats.parquet", index=False)
    assert fingerprint() != first and fingerprint(require_depth=False) == without_depth
    stats.loc[stats["row"] == 3, "ok"] = True
    stats.to_parquet(folder.dir / "depth_stats.parquet", index=False)
    covis = np.load(folder.dir / "covisibility.npy")
    covis[0, 1] = covis[1, 0] = 7
    np.save(folder.dir / "covisibility.npy", covis)
    assert fingerprint() != first


PERIODS = ["before 1970", "1970 to 2019.2877", "from 2019.2877"]


def test_period_table_counts_views_and_anchors(folder, capsys):
    scene = MVGenScene(folder.dir, resolution=S)
    printed = capsys.readouterr().out                               # the table is printed when the scene is built
    assert "[scene]" in printed and "scale 'scene' (factor 0.25)" in printed and "anchor_share" in printed
    assert all(name in printed for name in PERIODS)
    table = scene.period_table()
    assert table.index.tolist() == PERIODS
    assert table["usable"].tolist() == [7, 7, 8]                    # views 5 and 12 are not usable
    assert table["training"].tolist() == [7, 7, 8]
    assert table["references"].tolist() == [7, 6, 6]                # 9, 22 and 23 have no reliable depth
    assert table["anchors"].tolist() == [7, 7, 6]                   # 22 and 23 are co-visible with nothing
    assert table["anchor_share"].tolist() == pytest.approx([7 / 20, 7 / 20, 6 / 20])

    scene.hold_out([0, 1, 16, 17, 18])                              # the table follows the hold-out; printed again
    assert "5 held out" in capsys.readouterr().out
    table = scene.period_table()
    assert table["usable"].tolist() == [7, 7, 8] and table["training"].tolist() == [5, 7, 5]
    assert table["anchors"].tolist() == [5, 7, 0]                   # three after-fire views are left: no target group
    assert table["anchor_share"].tolist() == pytest.approx([5 / 12, 7 / 12, 0])

    weighted = MVGenScene(folder.dir, ViewSamplingConfig(anchor_period_weights=(3, 1, 0)), resolution=S)
    assert weighted.period_table()["anchor_share"].tolist() == pytest.approx([0.75, 0.25, 0])
    assert weighted.period_table()["anchors"].tolist() == [7, 7, 6]


def test_a_weighted_period_without_anchors_is_refused(folder):
    equal = ViewSamplingConfig(anchor_period_weights=(1, 1, 1))
    scene = MVGenScene(folder.dir, equal, resolution=S)
    with pytest.raises(ValueError, match=r"No sample can be drawn in \['from 2019.2877'\]"):
        scene.hold_out([16, 17, 18])                                # the hold-out empties a weighted period
    MVGenScene(folder.dir, ViewSamplingConfig(anchor_period_weights=(1, 1, 0)), resolution=S).hold_out([16, 17, 18])
    MVGenScene(folder.dir, resolution=S).hold_out([16, 17, 18])     # no weights: the period just gives no samples

    # Nothing in this table is dated before 1859: a weight there cannot be honoured.
    edges = (1859.0, 1970.0, 2019.2877)
    with pytest.raises(ValueError, match=r"No sample can be drawn in \['before 1859'\].*anchors per period"):
        MVGenScene(folder.dir, ViewSamplingConfig(period_edges=edges, anchor_period_weights=(1, 1, 1, 1)), resolution=S)
    thin = MVGenScene(folder.dir, ViewSamplingConfig(period_edges=edges, anchor_period_weights=(0, 1, 1, 1)),
                      resolution=S)
    assert thin.period_table()["anchors"].tolist() == [0, 7, 7, 6]
    assert MVGenScene(folder.dir, ViewSamplingConfig(period_edges=edges), resolution=S).period.min() == 1


def test_period_names():
    assert period_names((1859.0, 1970.0, 2019.2877)) == ["before 1859", "1859 to 1970", "1970 to 2019.2877",
                                                        "from 2019.2877"]
    assert period_names((1970.0,)) == ["before 1970", "from 1970"] and period_names(()) == ["all"]


def test_number_of_references_drawn_per_sample(folder):
    cfg = ViewSamplingConfig(n_ref_choices=(1, 2, 3))
    scene, twin = MVGenScene(folder.dir, cfg, resolution=S), MVGenScene(folder.dir, cfg, resolution=S)
    drawn = [scene.draw(i) for i in range(300)]
    counts = np.array([n_ref for _, n_ref in drawn])
    assert [np.mean(counts == k) for k in (1, 2, 3)] == pytest.approx([1 / 3] * 3, abs=0.08)
    for i, (rows, n_ref) in enumerate(drawn[:60]):
        assert len(set(rows)) == 8 and isinstance(n_ref, int)       # always eight views
        assert scene.ref_ok[rows[:n_ref]].all() and scene.target_ok[rows[n_ref:]].all()
        assert len(set(scene.period[rows[n_ref:]])) == 1 and seen_by_a_reference(scene.covis, rows, n_ref).all()
        again, again_n_ref = twin.draw(i)                           # deterministic, the number of references too
        assert np.array_equal(rows, again) and n_ref == again_n_ref
    index = int(np.flatnonzero(counts == 1)[0])
    sample = scene[index]
    assert sample["n_ref"] == 1 and sample["rows"].tolist() == drawn[index][0].tolist()
    assert sample["images"].shape[0] == 8
    assert (sample["depth"].flatten(1).abs().max(dim=1).values > 0).tolist() == [True] + [False] * 7

    # The rows alone do not say where the references end: indices() needs to be told.
    with pytest.raises(ValueError, match="n_ref_choices"):
        scene.indices(0)
    fixed = np.stack([scene.indices(i, n_ref=3) for i in range(40)])
    assert scene.ref_ok[fixed[:, :3]].all() and all(seen_by_a_reference(scene.covis, rows, 3).all() for rows in fixed)
    assert np.array_equal(scene.draw(5, n_ref=2)[0], scene.indices(5, n_ref=2)) and scene.draw(5, n_ref=2)[1] == 2
    with pytest.raises(ValueError, match="n_ref = 6"):              # two targets: fewer than min_targets_seen
        scene.draw(0, n_ref=6)
    with pytest.raises(ValueError, match="n_ref = 0"):
        scene.draw(0, n_ref=0)

    # Seven targets (one reference) need six partners: the six co-visible after-fire views cannot anchor them.
    table = scene.period_table()
    assert table["anchors"].tolist() == [7, 7, 0]
    assert table["anchor_share"].tolist() == pytest.approx(
        [(7 / 14 + 2 * 7 / 20) / 3, (7 / 14 + 2 * 7 / 20) / 3, (2 * 6 / 20) / 3])
    after_fire = [n_ref for rows, n_ref in drawn if scene.period[rows[n_ref]] == 2]
    assert after_fire and 1 not in after_fire
    with pytest.raises(ValueError, match=r"No sample can be drawn in \['from 2019.2877'\]"):
        MVGenScene(folder.dir, ViewSamplingConfig(n_ref_choices=(1, 2, 3), anchor_period_weights=(1, 1, 1)),
                   resolution=S)


def test_a_fixed_number_of_references_is_the_default(folder):
    scene = MVGenScene(folder.dir, resolution=S)
    for i in range(20):
        rows, n_ref = scene.draw(i)
        assert n_ref == 3 and np.array_equal(rows, scene.indices(i))
    assert np.array_equal(scene.indices(3, epoch=2, seed=5), scene.draw(3, epoch=2, seed=5)[0])
    assert scene[3]["n_ref"] == 3


def test_missing_depth_is_reported(folder):
    depth_path(folder.dir, 1).unlink()
    scene = MVGenScene(folder.dir, resolution=S)
    with pytest.raises(FileNotFoundError, match="prepare_depth"):
        scene.make_sample(np.array([1, 2, 3]), n_ref=1)
    assert scene.make_sample(np.array([2, 1, 3]), n_ref=1)["depth"][1].abs().max() == 0    # fine as a target
    with pytest.raises(ValueError, match="resolution"):                                     # depth made at 32
        MVGenScene(folder.dir, resolution=16).make_sample(np.array([2, 3]), n_ref=1)
    with pytest.raises(ValueError, match="n_ref"):
        scene.make_sample(np.array([2, 3]), n_ref=3)

    (folder.dir / "depth_stats.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="prepare_depth"):
        MVGenScene(folder.dir, resolution=S)
    sampler_only = MVGenScene(folder.dir, resolution=S, require_depth=False)
    assert not sampler_only.depth_ok.any() and sampler_only.indices(0).shape == (8,)


def test_inconsistent_folders_are_refused(folder):
    stats = pd.read_parquet(folder.dir / "depth_stats.parquet")
    stats.drop(columns="file_key").to_parquet(folder.dir / "depth_stats.parquet", index=False)
    assert MVGenScene(folder.dir, resolution=S).depth_ok.sum() == stats["ok"].sum()    # the key column is optional
    stats.assign(file_key=stats["file_key"].iloc[::-1].to_numpy()).to_parquet(folder.dir / "depth_stats.parquet")
    with pytest.raises(ValueError, match="another view table"):
        MVGenScene(folder.dir, resolution=S)
    stats.assign(row=stats["row"] + 5).to_parquet(folder.dir / "depth_stats.parquet")  # rows the table lacks
    with pytest.raises(ValueError, match="another view table"):
        MVGenScene(folder.dir, resolution=S)

    np.save(folder.dir / "covisibility.npy", np.zeros((N - 1, N - 1), dtype=np.uint16))
    with pytest.raises(ValueError, match="covisibility"):
        MVGenScene(folder.dir, resolution=S, require_depth=False)
    views = pd.read_parquet(folder.dir / "views.parquet")
    views.drop(columns="usable").to_parquet(folder.dir / "views.parquet", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        MVGenScene(folder.dir, resolution=S, require_depth=False)


# --- depth ---------------------------------------------------------------------------
def test_sparse_depth_points(folder):
    with np.load(folder.dir / "sparse_points.npz") as archive:
        sparse = dict(archive)
    counts = {}
    for row in (0, 7, 21, FAR_VIEW, AWAY_VIEW):
        points = sparse_depth_points(sparse, row, folder.c2w[row], np.array(CROP_K), S)
        assert points.dtype == np.float32
        np.testing.assert_allclose(points, crop_points(folder, row), atol=1e-4)
        counts[row] = len(points)
    assert 50 < counts[0] < len(folder.observed[0])                 # the crop shows part of what the view observes
    assert counts[FAR_VIEW] == 24 and counts[AWAY_VIEW] == 0        # few points; all behind the camera
    sparse["view_pid"] = sparse["view_pid"] + 1
    with pytest.raises(ValueError, match="not in p3d_ids"):
        sparse_depth_points(sparse, 0, folder.c2w[0], np.array(CROP_K), S)


EXPORTED = REPO / "data/experiments/mvgen/samples/easy0"
COLMAP_CACHE = REPO / "data/audit_2026-11-19/data_cameras/out/colmap_cache.npz"
OLD_VIEWS = REPO / "data/datasets/25512_model0/views.parquet"


@pytest.mark.skipif(not (EXPORTED.exists() and COLMAP_CACHE.exists() and OLD_VIEWS.exists()),
                    reason="needs the exported view set and the audit's COLMAP cache (under data/, not in git)")
def test_sparse_depth_points_reproduce_the_exported_view_set():
    """Same points as ``scripts/export_nvs_samples.py`` wrote for the view set the first MVGenMaster tests used."""
    meta = json.loads((EXPORTED / "sample.json").read_text())
    file_keys = pd.read_parquet(OLD_VIEWS, columns=["file_key"])["file_key"].to_numpy()
    with np.load(COLMAP_CACHE) as archive:
        sparse = {name: archive[name] for name in ("p3d_ids", "p3d_xyz", "view_starts", "view_pid")}
    for j in (0, 5):
        view = meta["views"][j]
        row = int(np.flatnonzero(file_keys == view["file_key"])[0])
        points = sparse_depth_points(sparse, row, np.array(view["c2w"]), np.array(view["K"]), meta["size"])
        np.testing.assert_allclose(points, np.load(EXPORTED / f"sparse_{j}.npy"), atol=1e-3)


def test_depth_is_reliable():
    ok = depth_is_reliable(np.array([49, 50, 5000, 5000, 0]), np.array([0.01, 0.15, 0.16, 0.01, np.nan]))
    assert ok.tolist() == [False, True, False, True, False]


class FakeDepthPro:
    """Stands in for Depth-Pro: depth ``MONO`` everywhere, or three bands of rows for the views in ``bad``
    (no single scale aligns those with the plane). It recognises the view by its colour (red = 10 x row)."""

    def __init__(self, bad: tuple[int, ...] = (), fail_at_call: int | None = None, holes: dict | None = None):
        self.bad, self.fail_at_call, self.holes = bad, fail_at_call, holes or {}
        self.calls, self.focal = [], []

    def eval(self) -> None:
        """Depth-Pro is a module; the script puts it in evaluation mode."""

    def infer(self, x: torch.Tensor, f_px: torch.Tensor) -> dict:
        if len(self.calls) == self.fail_at_call:
            raise RuntimeError("job killed")
        row = int(round(float(x[0, 0, 0]) / 10))
        self.calls.append(row)
        self.focal.append(float(f_px))
        depth = torch.full((S, S), MONO)
        if row in self.bad:
            depth[0::3], depth[2::3] = MONO / 2, MONO * 2
        if row in self.holes:                           # one pixel with a value no depth map may hold
            depth[3, 4] = self.holes[row]
        return {"depth": depth}


@pytest.fixture(scope="module")
def prepare_depth():
    """The script ``scripts/mvgen/prepare_depth.py`` as a module."""
    spec = importlib.util.spec_from_file_location("prepare_depth", REPO / "scripts/mvgen/prepare_depth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bare_folder(folder) -> SimpleNamespace:
    """The dataset folder as the build script leaves it: no depth yet."""
    shutil.rmtree(folder.dir / "depth")
    (folder.dir / "depth_stats.parquet").unlink()
    return folder


def test_prepare_depth_scales_mono_depth_to_the_colmap_points(bare_folder, prepare_depth):
    folder, model = bare_folder, FakeDepthPro(bad=(10,))
    stats = prepare_depth.prepare(folder.dir, model, torch.from_numpy, size=S)
    all_rows = list(range(N))                           # views 5 and 12 are not usable: they get depth all the same
    assert tuple(stats.columns) == DEPTH_STATS_COLUMNS and stats["row"].tolist() == all_rows
    assert model.calls == all_rows[:-1]                 # view 23 has no point in its crop: nothing to scale
    assert set(model.focal) == {40.0}                   # Depth-Pro is given the crop's focal length
    pd.testing.assert_frame_equal(stats, pd.read_parquet(folder.dir / "depth_stats.parquet"))
    stats = stats.set_index("row")
    for row in (0, 5, 7, 21, FAR_VIEW):
        z = crop_points(folder, row)[:, 2]
        scale = float(np.median(z / MONO))
        assert stats.n_points[row] == len(z) and stats.file_key[row] == f"view_{row:02d}.png"
        assert stats.scale[row] == pytest.approx(scale, rel=1e-5)
        assert stats.median_rel_error[row] == pytest.approx(np.median(np.abs(z / MONO / scale - 1)), abs=1e-5)
        depth = np.load(depth_path(folder.dir, row))
        assert depth.dtype == np.float16 and depth.shape == (S, S)
        np.testing.assert_allclose(depth, MONO * scale, rtol=1e-3)
    assert stats.ok[0] and stats.ok[5] and stats.ok[7] and stats.ok[21]
    assert not stats.ok[FAR_VIEW] and stats.n_points[FAR_VIEW] == 24                    # too few points
    assert not stats.ok[10] and stats.median_rel_error[10] > 0.15                       # no single scale fits
    assert not stats.ok[AWAY_VIEW] and stats.n_points[AWAY_VIEW] == 0 and np.isnan(stats.scale[AWAY_VIEW])
    assert not depth_path(folder.dir, AWAY_VIEW).exists()
    assert sorted(p.name for p in (folder.dir / "depth").iterdir()) == [f"{row:05d}.npy" for row in all_rows[:-1]]


def test_view_depth_returns_the_map_as_it_is_stored(bare_folder, prepare_depth):
    folder = bare_folder
    views = pd.read_parquet(folder.dir / "views.parquet")
    sparse = prepare_depth.load_sparse(folder.dir)
    stats = prepare_depth.prepare(folder.dir, FakeDepthPro(), torch.from_numpy, size=S).set_index("row")
    for row in (3, 12):
        depth, record = prepare_depth.view_depth(views, row, sparse, FakeDepthPro(), torch.from_numpy, size=S)
        assert depth.dtype == np.float16 and np.array_equal(depth, np.load(depth_path(folder.dir, row)))
        assert set(record) == set(DEPTH_STATS_COLUMNS) and record == {"row": row, **stats.loc[row].to_dict()}
    depth, record = prepare_depth.view_depth(views, AWAY_VIEW, sparse, FakeDepthPro(), torch.from_numpy, size=S)
    assert depth is None and record["n_points"] == 0 and not record["ok"]


def test_prepare_depth_skips_views_that_cannot_be_loaded(bare_folder, prepare_depth, capsys):
    folder = bare_folder
    views = pd.read_parquet(folder.dir / "views.parquet")
    Path(views.image_path[2]).unlink()                  # no image file
    views.loc[6, "k1"] = np.nan                         # a lens model load_view does not support
    views.to_parquet(folder.dir / "views.parquet", index=False)
    assert prepare_depth.loadable_views(views).tolist() == [row not in (2, 6) for row in range(N)]
    model = FakeDepthPro()
    stats = prepare_depth.prepare(folder.dir, model, torch.from_numpy, size=S)
    assert stats["row"].tolist() == [row for row in range(N) if row not in (2, 6)]
    assert 2 not in model.calls and 6 not in model.calls
    assert f"{N - 2} views to do, 0 done before, 2 cannot be loaded" in capsys.readouterr().out


@pytest.mark.parametrize("hole", [float("nan"), float("inf"), 0.0, -1.0])
def test_a_depth_map_that_is_not_finite_and_positive_stops_the_run(bare_folder, prepare_depth, hole):
    folder = bare_folder
    with pytest.raises(ValueError, match=r"view 4 \(view_04.png\) is not finite and positive"):
        prepare_depth.prepare(folder.dir, FakeDepthPro(holes={4: hole}), torch.from_numpy, size=S)
    assert not depth_path(folder.dir, 4).exists() and depth_path(folder.dir, 3).exists()
    assert pd.read_parquet(folder.dir / "depth_stats.parquet")["row"].tolist() == [0, 1, 2, 3]   # the finished views


def test_stored_depth_is_float16_finite_and_positive(prepare_depth, tmp_path):
    depth = np.array([[1e-3, 2.5], [7e4, 1e9]], dtype=np.float32)
    stored = prepare_depth.as_stored(depth, "a view")
    assert stored.dtype == np.float16 and stored[1].tolist() == [65504.0, 65504.0]     # clipped to float16's largest
    np.testing.assert_allclose(stored[0], depth[0], rtol=1e-3)
    assert np.array_equal(prepare_depth.as_stored(stored, "a view"), stored)           # storing twice changes nothing
    for bad in (np.nan, np.inf, 0.0, -2.0, 1e-9):       # 1e-9 is positive, but rounds to zero in float16
        with pytest.raises(ValueError, match="a view is not finite and positive"):
            prepare_depth.as_stored(np.array([[1.0, bad]], dtype=np.float32), "a view")
    with pytest.raises(ValueError, match="not finite and positive"):
        prepare_depth.save_depth(tmp_path / "depth" / "00000.npy", np.zeros((2, 2), dtype=np.float32))
    assert not (tmp_path / "depth").exists()            # nothing is written for a refused map


def test_prepare_depth_resumes(bare_folder, prepare_depth):
    folder = bare_folder
    n_scaled = N - 1                                    # every view but 23
    with pytest.raises(RuntimeError, match="killed"):
        prepare_depth.prepare(folder.dir, FakeDepthPro(fail_at_call=5), torch.from_numpy, size=S, save_every=2)
    assert pd.read_parquet(folder.dir / "depth_stats.parquet")["row"].tolist() == [0, 1, 2, 3, 4]
    rest = FakeDepthPro()
    stats = prepare_depth.prepare(folder.dir, rest, torch.from_numpy, size=S, save_every=2)
    assert len(rest.calls) == n_scaled - 5 and len(stats) == n_scaled + 1
    again = FakeDepthPro()
    pd.testing.assert_frame_equal(prepare_depth.prepare(folder.dir, again, torch.from_numpy, size=S), stats)
    assert again.calls == []                            # nothing left to do, view 23 included
    depth_path(folder.dir, 3).unlink()
    redo = FakeDepthPro()
    pd.testing.assert_frame_equal(prepare_depth.prepare(folder.dir, redo, torch.from_numpy, size=S), stats)
    assert redo.calls == [3] and depth_path(folder.dir, 3).exists()                     # a lost map is made again


def test_prepare_depth_refuses_the_depth_of_another_view_table(folder, prepare_depth):
    views = pd.read_parquet(folder.dir / "views.parquet")
    views.loc[1, "file_key"] = "another_photo.png"      # as after rebuilding the table with other rows
    views.to_parquet(folder.dir / "views.parquet", index=False)
    with pytest.raises(ValueError, match="another view table"):
        prepare_depth.prepare(folder.dir, FakeDepthPro(), torch.from_numpy, size=S)


def test_scene_reads_prepared_depth(bare_folder, prepare_depth):
    folder = bare_folder
    stats = prepare_depth.prepare(folder.dir, FakeDepthPro(bad=(10,)), torch.from_numpy, size=S).set_index("row")
    scene = MVGenScene(folder.dir, resolution=S)
    assert np.flatnonzero(scene.depth_ok).tolist() == stats.index[stats.ok].tolist()
    assert not scene.depth_ok[[10, FAR_VIEW, AWAY_VIEW]].any() and scene.depth_ok.sum() == N - 3
    assert not scene.ref_ok[[5, 12]].any()              # depth or not, views that are not usable are not drawn
    sample = MVGenScene(folder.dir, resolution=S, scale=1.0).make_sample(np.array([0, 3, 2]), n_ref=2)
    for j, row in enumerate((0, 3)):
        np.testing.assert_allclose(sample["depth"][j].numpy(), MONO * stats.scale[row], rtol=1e-3)
    assert 10 not in drawn_rows(scene)[:, :3]


def test_summary_by_period(prepare_depth):
    stats = pd.DataFrame({"row": [0, 1, 2, 3], "n_points": [100, 0, 80, 60],
                          "median_rel_error": [0.02, np.nan, 0.3, 0.1], "ok": [True, False, False, True]})
    dates = np.array([1900.0, 1950.0, 2015.0, 2021.0])
    lines = prepare_depth.summarise(stats, dates, (1970.0, 2019.2877)).splitlines()
    assert len(lines) == 4                              # a header and three periods
    assert lines[1].split() == ["before", "1970", "2", "1", "1", "0.020", "0.020"]
    assert lines[2].split() == ["1970", "to", "2019.2877", "1", "0", "0", "0.300", "0.300"]
    assert lines[3].split() == ["from", "2019.2877", "1", "1", "0", "0.100", "0.100"]
    assert prepare_depth.summarise(stats, dates, ()).splitlines()[1].split() == ["all", "4", "2", "1", "0.100", "0.260"]


@pytest.mark.parametrize("arguments", [[], ["--period-edges", "1859", "1970", "2019.2877"]],
                         ids=["default edges", "the queued job's command line"])
def test_prepare_depth_command_line(bare_folder, prepare_depth, monkeypatch, capsys, arguments):
    """The script from its command line, with a fake Depth-Pro: the summary's periods end at the fire's edge."""
    folder, model = bare_folder, FakeDepthPro()
    monkeypatch.setattr(prepare_depth, "create_model_and_transforms", lambda device: (model, torch.from_numpy))
    monkeypatch.setattr(sys, "argv", ["prepare_depth.py", "--dataset", str(folder.dir), "--size", str(S)] + arguments)
    prepare_depth.main()
    printed = capsys.readouterr().out
    n_usable = int(folder.usable.sum())                 # views 22 and 23, both usable, have too few points
    assert f"{N - 2} of {N} views have reliable depth" in printed
    assert f"can be drawn as references, {n_usable - 2} of {n_usable}" in printed
    for name in ("before 1859", "1859 to 1970", "1970 to 2019.2877", "from 2019.2877"):
        assert name in printed
    assert len(list((folder.dir / "depth").iterdir())) == N - 1
