"""Tests for timemachine.analysis.planning.

The planner's job is to say, before anything is downloaded, which scenes to build,
what they cost, and what is uncomfortable about the result. The last part matters
most: old photographs are concentrated in a few places, so a plan can look large
and still teach a model about Amsterdam.
"""
import numpy as np
import pandas as pd
import pytest

from timemachine.analysis import planning as pl


def census(n=12, seed=0):
    """A census with a deliberately skewed old-photo distribution, as the real one has."""
    rng = np.random.default_rng(seed)
    old = np.array([400, 200, 120, 90, 70, 55, 40, 30, 25, 20, 15, 10])[:n]
    return pd.DataFrame({
        "scene_id": range(1, n + 1),
        "scene": [f"Scene_{i}" for i in range(1, n + 1)],
        "posed_pre-1900": old // 2,
        "posed_1900-1944": old - old // 2,
        "posed_1945-1969": rng.integers(0, 20, n),
        "posed_1970-1999": rng.integers(0, 40, n),
        "posed_2000+": rng.integers(100, 3000, n),
        "best_model_2000+": rng.integers(100, 3000, n),
        pl.OLD_COLUMN: old,
        "gate_posed": [True] * (n - 2) + [False] * 2,
        "gate_any": [True] * n,
    })


# ----------------------------------------------------------------- loading
def test_load_census_rejects_a_file_that_is_not_one(tmp_path):
    path = tmp_path / "nope.csv"
    pd.DataFrame({"scene": ["a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not a scene census"):
        pl.load_census(path)

    good = tmp_path / "ok.csv"
    census().to_csv(good, index=False)
    assert len(pl.load_census(good)) == 12


# ----------------------------------------------------------------- ranking
def test_ranking_respects_the_gate_and_excludes_what_is_built():
    c = census()
    assert len(pl.rank_candidates(c, gate="posed")) == 10      # two fail the gate
    assert len(pl.rank_candidates(c, gate="any")) == 12
    assert len(pl.rank_candidates(c, gate="none")) == 12

    ranked = pl.rank_candidates(c, gate="posed")
    assert ranked[pl.OLD_COLUMN].is_monotonic_decreasing       # best first
    # exclusion works by name or by id, since callers have both
    assert "Scene_1" not in set(pl.rank_candidates(c, exclude=["Scene_1"])["scene"])
    assert 2 not in set(pl.rank_candidates(c, exclude=["2"])["scene_id"])
    with pytest.raises(ValueError, match="gate must be"):
        pl.rank_candidates(c, gate="sometimes")


# ------------------------------------------------------------------ budget
def test_budget_and_count_both_stop_the_plan():
    c = census()
    by_count = pl.plan_scenes(c, n_scenes=4)
    assert len(by_count) == 4
    assert by_count["cum_old"].iloc[-1] == by_count[pl.OLD_COLUMN].sum()
    assert by_count["cum_gigabytes"].is_monotonic_increasing

    budget = float(by_count["cum_gigabytes"].iloc[1])          # room for exactly two
    by_budget = pl.plan_scenes(c, budget_gb=budget)
    assert len(by_budget) == 2 and by_budget["cum_gigabytes"].iloc[-1] <= budget + 1e-9

    # the tighter of the two wins when both are given
    assert len(pl.plan_scenes(c, n_scenes=10, budget_gb=budget)) == 2
    with pytest.raises(ValueError, match="n_scenes, budget_gb"):
        pl.plan_scenes(c)


def test_capping_modern_photos_cuts_storage_but_not_old_photos():
    c = census()
    capped = pl.plan_scenes(c, n_scenes=6, max_modern=300)
    uncapped = pl.plan_scenes(c, n_scenes=6, max_modern=None)
    assert capped["cum_gigabytes"].iloc[-1] < uncapped["cum_gigabytes"].iloc[-1]
    assert capped["cum_old"].iloc[-1] == uncapped["cum_old"].iloc[-1]   # the point is kept


def test_a_plan_with_nothing_left_to_build_is_empty():
    c = census()
    built = pl.rank_candidates(c, gate="posed")["scene"].tolist()
    assert pl.plan_scenes(c, n_scenes=5, exclude=built).empty
    assert "already built" in pl.render_plan(pd.DataFrame(), pl.DEFAULT_BYTES_PER_IMAGE)


# ----------------------------------------------------------------- storage
def test_storage_is_measured_from_real_files_when_available(tmp_path):
    folder = tmp_path / "1_model0" / "images"
    folder.mkdir(parents=True)
    for i in range(4):
        (folder / f"v{i}.jpg").write_bytes(b"x" * 1000)
    size, counted = pl.calibrate_bytes_per_image([tmp_path / "1_model0"])
    assert size == 1000 and counted == 4

    # and falls back loudly-but-safely when there is nothing to measure
    size, counted = pl.calibrate_bytes_per_image([tmp_path / "empty"])
    assert size == pl.DEFAULT_BYTES_PER_IMAGE and counted == 0


def test_bytes_per_image_scales_the_whole_plan():
    c = census()
    small = pl.plan_scenes(c, n_scenes=5, bytes_per_image=100_000)
    big = pl.plan_scenes(c, n_scenes=5, bytes_per_image=200_000)
    assert big["cum_gigabytes"].iloc[-1] == pytest.approx(2 * small["cum_gigabytes"].iloc[-1])


# ------------------------------------------------------------ concentration
def test_concentration_distinguishes_even_from_lopsided():
    even = pl.concentration([10] * 20)
    assert even["gini"] == pytest.approx(0.0, abs=1e-9)
    assert even["top_shares"][1] == pytest.approx(0.05)

    lopsided = pl.concentration([1000] + [1] * 19)
    assert lopsided["gini"] > 0.9
    assert lopsided["top_shares"][1] > 0.97
    assert pl.concentration([0, 0])["total"] == 0.0            # no crash on an empty plan


def test_report_surfaces_the_concentration_and_the_warning():
    c = census()
    plan = pl.plan_scenes(c, n_scenes=8)
    text = pl.render_plan(plan, 190_000, measured_from=5000, have=["Scene_99"])
    assert "Scene build plan" in text and "Scene_99" in text
    assert "measured at 190 kB/image over 5,000 cached images" in text
    assert "Gini" in text and "top 1 scene(s) hold" in text
    assert "location input most of all" in text               # the point of the whole table
    assert "| pre-1900 |" in text and "| 2000+ |" in text

    unmeasured = pl.render_plan(plan, 190_000)
    assert "no built dataset to measure" in unmeasured


def test_era_totals_apply_the_modern_cap():
    c = census()
    plan = pl.plan_scenes(c, n_scenes=5, max_modern=300)
    totals = pl.era_totals(plan, max_modern=300)
    assert totals["2000+"] <= 5 * 300
    assert totals["pre-1900"] == int(plan["posed_pre-1900"].sum())


# ------------------------------------------------------ against the real census
def test_the_shipped_census_plans_without_surprises():
    """Guards against a column rename in the census breaking the planner silently."""
    from pathlib import Path

    path = Path("census.csv")
    if not path.exists():
        pytest.skip("real census not available in this checkout")
    real = pl.load_census(path)
    plan = pl.plan_scenes(real, budget_gb=50)
    assert len(plan) > 50
    assert plan["cum_old"].iloc[-1] > 3000
    # The whole gated set is small; if this ever fails, the size model changed.
    assert plan["cum_gigabytes"].iloc[-1] < 100
    text = pl.render_plan(plan, pl.DEFAULT_BYTES_PER_IMAGE)
    assert "old posed photos" in text
