"""Tests for timemachine.analysis.entanglement.

The module answers a two-by-two: after removing the era direction from the
appearance vector, is era gone, and is anything else still there? These tests
build synthetic appearance spaces where we *know* the answer - era and lighting
on orthogonal axes, then on the same axis - and check the report says so.
"""
import numpy as np
import pytest

from timemachine.analysis import entanglement as ent
from timemachine.analysis.shortcut import FEATURE_NAMES

F = len(FEATURE_NAMES)


def make_space(n=400, era_axis=0, light_axis=1, seed=0, era_strength=3.0, light_strength=3.0):
    """Appearance features where era lives on one axis and lighting on another."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, F))
    old = rng.random(n) < 0.3
    night = rng.random(n) < 0.4
    X[:, era_axis] += np.where(old, era_strength, -era_strength)
    X[:, light_axis] += np.where(night, light_strength, -light_strength)
    years = np.where(old, rng.uniform(1870, 1940, n), rng.uniform(1990, 2020, n))
    train = np.arange(n) % 4 != 0
    return X, years, night, train


# ------------------------------------------------------------- basic pieces
def test_standardize_handles_constant_columns():
    X = np.random.default_rng(0).normal(5, 3, (50, F))
    X[:, 2] = 7.0
    Z, mean, scale = ent.standardize(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-9)
    assert scale[2] == 1.0 and np.isfinite(Z).all()
    # reusing fitted statistics must reproduce the same transform
    assert np.allclose(ent.standardize(X, mean, scale)[0], Z)


def test_era_direction_points_at_the_axis_that_carries_era():
    X, years, _, _ = make_space(era_axis=3)
    d = ent.era_direction(X, years)
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert np.argmax(np.abs(d)) == 3            # the planted axis dominates
    assert np.abs(d[3]) > 0.8


def test_era_direction_needs_both_eras():
    X, years, _, _ = make_space()
    with pytest.raises(ValueError, match="both old and modern"):
        ent.era_direction(X, np.full(len(years), 2010.0))


def test_residualize_removes_exactly_that_direction():
    X, years, _, _ = make_space(era_axis=0)
    d = ent.era_direction(X, years)
    residual = ent.residualize(X, d)
    # nothing is left along d, and the orthogonal complement is untouched
    assert np.allclose(residual @ d, 0, atol=1e-9)
    Z, _, _ = ent.standardize(X)
    other = np.zeros(F); other[5] = 1.0
    other = other - (other @ d) * d
    np.testing.assert_allclose(residual @ other, Z @ other, atol=1e-9)
    assert ent.residualize(X, d * 7.0).shape == residual.shape      # scale-invariant


# --------------------------------------------------------- the main report
def test_removing_the_era_direction_actually_removes_era():
    X, years, _, train = make_space(era_axis=0, light_axis=1)
    r = ent.residual_report(X, years, train)
    assert r["era_auc_full"] > 0.95                  # era is trivially readable to start with
    assert r["era_auc_residual"] < 0.65 and r["era_removed"]
    assert r["dims_removed"] >= 1 and r["variance_kept"] < 1.0
    assert list(r["trace"]["dims_removed"]) == list(range(r["dims_removed"] + 1))
    assert r["loadings"].iloc[0]["feature"] == FEATURE_NAMES[0]


def test_report_refits_rather_than_assuming_the_projection_worked():
    """Two era-carrying axes: removing one must NOT be reported as success."""
    rng = np.random.default_rng(1)
    n = 500
    X = rng.normal(0, 1, (n, F))
    old = rng.random(n) < 0.35
    X[:, 0] += np.where(old, 3.0, -3.0)
    X[:, 1] += np.where(old, 2.5, -2.5)              # a second, independent era signal
    years = np.where(old, rng.uniform(1870, 1940, n), rng.uniform(1990, 2020, n))
    train = np.arange(n) % 4 != 0

    r = ent.residual_report(X, years, train)
    # The single best direction mixes both axes, so a refit on the residual still
    # finds era if anything survived. The test is that we MEASURE it, not assume it.
    assert np.isfinite(r["era_auc_residual"])
    assert r["era_auc_residual"] <= r["era_auc_full"]


def test_group_alignment_sums_to_one_and_finds_the_right_group():
    d = np.zeros(F)
    d[FEATURE_NAMES.index("saturation_mean")] = 1.0      # a colour statistic
    table = ent.group_alignment(d).set_index("features")
    assert np.isclose(table["share_of_era_direction"].sum(), 1.0)
    assert table.loc["colour", "share_of_era_direction"] == pytest.approx(1.0)
    assert table.loc["tone", "share_of_era_direction"] == pytest.approx(0.0)


# ------------------------------------------------------- the two-by-two
def test_orthogonal_lighting_survives_the_era_removal():
    """The good case: era and lighting on different axes -> clean factorization."""
    X, years, night, train = make_space(era_axis=0, light_axis=1)
    r = ent.residual_report(X, years, train)
    light = ent.label_separability(X[train], night[train], r["directions"])

    assert light["auc_full"] > 0.95
    assert light["auc_residual"] > 0.9               # lighting is untouched
    assert light["retained"] > 0.9
    assert r["era_auc_residual"] < 0.65              # while era is gone
    assert "**Separable**" in ent.render_report(r, light, n_scenes=4)


def test_lighting_on_the_same_axis_as_era_does_not_survive():
    """The bad case: the two are the same signal, so neither can be kept alone."""
    X, years, night, train = make_space(era_axis=2, light_axis=2)    # same axis
    r = ent.residual_report(X, years, train)
    light = ent.label_separability(X[train], night[train], r["directions"])

    assert light["auc_full"] > 0.6                   # lighting is readable to start with
    assert light["auc_residual"] < light["auc_full"] # and is damaged by the removal
    assert "**Entangled**" in ent.render_report(r, light, n_scenes=4)


def test_label_separability_handles_a_one_sided_label():
    X, years, _, train = make_space()
    r = ent.residual_report(X, years, train)
    out = ent.label_separability(X, np.ones(len(years), dtype=bool), r["directions"])
    assert np.isnan(out["auc_full"]) and np.isnan(out["auc_residual"])
    assert out["sufficient"] is False


# ----------------------------------------------------------------- report
def test_report_without_labels_says_what_it_cannot_conclude():
    X, years, _, train = make_space()
    report = ent.render_report(ent.residual_report(X, years, train), None, n_scenes=1)
    assert "No lighting labels supplied" in report
    assert "**Separable**" not in report and "**Entangled**" not in report
    assert "dimensions removed" in report and "share of the direction" in report
    assert "unanswered until lighting labels" in report


# ------------------------------------------------- collecting lighting labels
def test_thumbnail_uri_is_self_contained_and_small(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    path = tmp_path / "v.jpg"
    Image.fromarray(rng.integers(0, 255, (600, 800, 3), dtype=np.uint8), "RGB").save(path)
    uri = ent.thumbnail_uri(str(path))
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(uri) < 80_000          # a 150-image sheet stays a few MB, openable anywhere


def test_lighting_sheet_shows_no_dates_and_escapes_ids():
    nasty = '<script>x</script> "q" & id'
    page = ent.render_lighting_sheet([nasty, "b"], ["data:image/jpeg;base64,AA", "data:image/jpeg;base64,BB"])
    assert "<script>x</script>" not in page and "&lt;script&gt;" in page
    assert page.count('class="cell"') == 2
    for choice in ent.LIGHTING_CLASSES:
        assert f'data-v="{choice}"' in page
    # The labeller must not see any per-image date or statistic, or the labels are
    # contaminated by the very thing they are meant to be independent of. (The word
    # "dates" does appear once, in the instruction not to look them up.)
    page_with_data = ent.render_lighting_sheet(["img_1", "img_2"],
                                               ["data:image/jpeg;base64,AA"] * 2)
    for value in ("1890", "1935", "2015", "0.4821", "saturation_mean", "chroma_minor"):
        assert value not in page_with_data
    assert page_with_data.count("<img") == 2        # the image is all they get
    with pytest.raises(ValueError, match="same length"):
        ent.render_lighting_sheet(["a"], [])


def test_parse_lighting_labels_reads_only_usable_rows():
    import pandas as pd

    frame = pd.DataFrame({"id": list("abcde"),
                          "lighting": ["night", "Day", "unclear", "", "NIGHT"]})
    is_night, labelled = ent.parse_lighting_labels(frame)
    assert list(labelled) == [True, True, False, False, True]
    assert list(is_night[labelled]) == [True, False, True]


def test_era_stratified_sample_breaks_the_modern_majority():
    """A plain random sheet would be ~75% modern, confounding lighting with era."""
    import pandas as pd

    rng = np.random.default_rng(0)
    years = np.concatenate([rng.uniform(1860, 1899, 20), rng.uniform(1900, 1944, 15),
                            rng.uniform(1945, 1989, 10), rng.uniform(1990, 2023, 400)])
    views = pd.DataFrame({"date_mid": years, "file_key": [f"v{i}" for i in range(len(years))]})

    sample = ent.era_stratified_sample(views, 40, seed=0)
    assert len(sample) == 40 and sample["file_key"].is_unique
    counts = pd.Series(ent.era_bucket(sample["date_mid"].to_numpy())).value_counts()
    assert len(counts) == 4                       # every era represented
    assert counts.max() - counts.min() <= 2       # and evenly, not 75% modern
    assert (sample["date_mid"] < 1945).mean() > 0.4

    # Deterministic, and it stops rather than repeating when a bucket runs dry.
    assert sample["file_key"].tolist() == ent.era_stratified_sample(views, 40, 0)["file_key"].tolist()
    assert len(ent.era_stratified_sample(views, 10_000, 0)) == len(views)
    with pytest.raises(ValueError):
        ent.era_stratified_sample(views, 0)


def test_verdict_covers_the_case_where_era_cannot_be_removed():
    report = {"era_removed": False, "dims_removed": 8, "n_features": 14,
              "era_auc_residual": 0.91, "variance_kept": 0.3, "target_auc": 0.6}
    text = ent.verdict(report, {"auc_full": 0.9, "auc_residual": 0.9, "retained": 1.0,
                                "n_positive": 30, "n_negative": 40, "sufficient": True})
    assert "**Era could not be removed.**" in text and "8 of 14" in text
    # It must say this even when lighting looks fine: era surviving is the headline.
    assert "Separable" not in text


def test_verdict_flags_labels_that_were_never_informative():
    report = {"era_removed": True, "dims_removed": 2, "n_features": 14,
              "era_auc_residual": 0.52, "variance_kept": 0.8, "target_auc": 0.6}
    text = ent.verdict(report, {"auc_full": 0.55, "auc_residual": 0.51, "retained": 0.2,
                                "n_positive": 20, "n_negative": 25, "sufficient": True})
    assert "**Inconclusive.**" in text          # losing a signal that was never there proves nothing
    assert "Entangled" not in text


def test_sheet_javascript_escapes_survive_into_the_page():
    """Regression: the JS block must be a raw string.

    Written as a normal Python string, the CSV builder's escape sequences were
    interpreted by Python instead of passed through, so a real newline landed inside
    a JavaScript string literal and split it across two lines. The whole script then
    failed to parse with "Invalid or unexpected token", and no button did anything -
    the sheet looked normal and silently recorded nothing.
    """
    page = ent.render_lighting_sheet(["a"], ["data:image/jpeg;base64,AA"])
    script = page[page.index("<script>") + 8:page.index("</script>")]

    # Built from chr(92) so this test cannot fall into the same trap it guards against.
    backslash = chr(92)
    assert backslash + "ufeff" in script
    assert backslash + "r" + backslash + "n" in script
    # ... and the interpreted forms must NOT be there.
    assert "\r" not in script and "\ufeff" not in script

    # The specific line that broke: the CSV builder must be intact on one line.
    blob = [line for line in script.splitlines() if "new Blob(" in line]
    assert len(blob) == 1 and blob[0].rstrip().endswith(");"), blob


def test_verdict_refuses_to_conclude_from_too_few_night_photos():
    """The real run came back with 1 night photo in the test split; an AUC over one
    positive is that photo's rank, not a measurement."""
    report = {"era_removed": True, "dims_removed": 7, "n_features": 14,
              "era_auc_residual": 0.53, "variance_kept": 0.59, "target_auc": 0.6}
    text = ent.verdict(report, {"auc_full": 0.947, "auc_residual": 0.947, "retained": 1.0,
                                "n_positive": 1, "n_negative": 19, "sufficient": False})
    assert "not answerable from these labels" in text
    assert "**Separable**" not in text          # the bug this guards against
    assert "1 night and 19 day" in text


def test_cross_validation_uses_every_label_not_a_quarter_of_them():
    """A single hold-out wastes 75% of a hand-labelled sheet; CV scores all of it."""
    rng = np.random.default_rng(5)
    n = 200
    X = rng.normal(0, 1, (n, F))
    night = rng.random(n) < 0.12                 # rare, as night photos really are
    X[:, 7] += np.where(night, 2.5, -2.5)
    years = rng.uniform(1990, 2020, n)
    directions = np.zeros((1, F)); directions[0, 0] = 1.0

    out = ent.label_separability(X, night, directions)
    assert out["n_positive"] == int(night.sum()) and out["n_negative"] == int((~night).sum())
    assert out["folds"] >= 2
    assert out["auc_full"] > 0.9                 # every label contributed a score
    assert out["sufficient"] is (int(night.sum()) >= ent.MIN_LABELS_PER_CLASS)


# ------------------------------------------------- choosing a usable question
def test_sun_question_is_the_default_shape_and_night_still_works():
    import pandas as pd

    assert set(ent.QUESTIONS) == {"sun", "night"}
    for name, spec in ent.QUESTIONS.items():
        assert spec["positive"] in spec["classes"] and "unclear" in spec["classes"]
        page = ent.render_lighting_sheet(["a"], ["data:image/jpeg;base64,AA"], spec["title"], name)
        for c in spec["classes"]:
            assert f'data-v="{c}"' in page
        assert "Do not look them up" in page

    frame = pd.DataFrame({"id": list("abcdef"),
                          "lighting": ["sunlit", "overcast", "unclear", "", "SUNLIT", "night"]})
    is_sun, labelled = ent.parse_lighting_labels(frame, "sun")
    assert list(labelled) == [True, True, False, False, True, False]   # "night" is not a sun answer
    assert list(is_sun[labelled]) == [True, False, True]

    with pytest.raises(KeyError, match="Unknown question"):
        ent.parse_lighting_labels(frame, "weather")
    with pytest.raises(KeyError, match="Unknown question"):
        ent.render_lighting_sheet(["a"], ["data:image/jpeg;base64,AA"], "t", "weather")


def test_each_question_asks_about_light_not_about_age():
    """The instruction that keeps the labels independent of what we are testing."""
    for name, spec in ent.QUESTIONS.items():
        assert "age of the photo" in spec["help"], name
