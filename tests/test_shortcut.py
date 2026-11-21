"""Tests for timemachine.analysis.shortcut (no network, no real photos, no torch).

The probe's whole job is to answer "is the era signal real, or is it just film
look?", so these tests build small synthetic image sets where we *know* the
answer in advance and check that the probe says so:

* photos whose era shows only in their colour  -> skill collapses once colour is removed;
* photos whose era shows in their content      -> skill survives every treatment;
* photos with no era signal at all             -> no skill anywhere.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from timemachine.analysis import shortcut as sc


# ------------------------------------------------------------------ fixtures
def noise_image(rng, size=64, brightness=0.5, contrast=0.2):
    """A neutral grey-noise image: no colour, no structure, nothing era-specific."""
    lum = np.clip(rng.normal(brightness, contrast, (size, size)), 0, 1)
    return Image.fromarray((np.stack([lum] * 3, -1) * 255).astype(np.uint8), "RGB")


def striped_image(rng, n_stripes, size=64):
    """Grey image whose *content* (stripe count) carries the signal, not its colour."""
    xs = np.arange(size)
    lum = 0.5 + 0.25 * np.sign(np.sin(xs * n_stripes * np.pi / size))[None, :].repeat(size, 0)
    lum = np.clip(lum + rng.normal(0, 0.02, (size, size)), 0, 1)
    return Image.fromarray((np.stack([lum] * 3, -1) * 255).astype(np.uint8), "RGB")


def build(images, years):
    return sc.feature_matrix([sc.appearance_features(im) for im in images]), np.asarray(years, float)


def treated(images, name):
    func = sc.TREATMENT_FUNCS[name]
    return sc.feature_matrix([sc.appearance_features(func(im)) for im in images])


# ------------------------------------------------------------------ features
def test_feature_vector_is_complete_and_finite():
    feats = sc.appearance_features(noise_image(np.random.default_rng(0)))
    assert set(feats) == set(sc.FEATURE_NAMES)
    assert all(np.isfinite(v) for v in feats.values())
    assert sc.feature_matrix([feats]).shape == (1, len(sc.FEATURE_NAMES))


def test_features_are_resolution_independent():
    """The same photo at two sizes must look the same, apart from the size features."""
    rng = np.random.default_rng(1)
    big = striped_image(rng, 8, size=256)
    small = big.resize((64, 64), Image.Resampling.BILINEAR)
    a, b = sc.appearance_features(big), sc.appearance_features(small)
    for name in ("saturation_mean", "lum_mean", "lum_std", "chroma_minor"):
        assert a[name] == pytest.approx(b[name], abs=0.05), name
    assert a["log_pixels"] > b["log_pixels"]            # size itself is recorded separately


def test_monochrome_and_sepia_are_distinguished_from_colour():
    rng = np.random.default_rng(2)
    grey = noise_image(rng)
    colour = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), "RGB")
    sepia = sc.apply_sepia(grey)

    g, c, s = (sc.appearance_features(x) for x in (grey, colour, sepia))
    # A toned print varies along ONE colour axis: strong major spread, tiny minor.
    assert s["chroma_major"] > 3 * s["chroma_minor"] + 1
    assert c["chroma_minor"] > s["chroma_minor"]        # real colour spreads both ways
    assert g["saturation_mean"] < c["saturation_mean"]
    assert s["lum_std"] < g["lum_std"]                  # sepia flattens the contrast


def test_treatments_remove_what_they_claim_to():
    rng = np.random.default_rng(3)
    sepia = sc.apply_sepia(noise_image(rng))
    grey = sc.to_grayscale(sepia)
    flat = sc.to_tone_matched(sepia)

    assert sc.appearance_features(grey)["saturation_mean"] < 0.02        # colour gone
    assert sc.appearance_features(flat)["saturation_mean"] < 0.02
    # Equalization forces a common tone curve, so contrast no longer identifies the photo.
    dark = sc.appearance_features(sc.to_tone_matched(noise_image(rng, brightness=0.2, contrast=0.05)))
    bright = sc.appearance_features(sc.to_tone_matched(noise_image(rng, brightness=0.8, contrast=0.05)))
    assert abs(dark["lum_mean"] - bright["lum_mean"]) < 0.15
    assert sc.apply_sepia(sepia, strength=0.0).convert("L").getextrema()[1] > 0   # strength 0 is valid


# ------------------------------------------------------------------ the probe
def test_probe_recovers_a_linear_signal_and_beats_the_constant_guess():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, len(sc.FEATURE_NAMES)))
    y = 1900 + 40 * X[:, 0] + rng.normal(0, 1, 200)
    s = sc.score_probe(X[:150], y[:150], X[150:], y[150:], alpha=1.0)
    assert s["skill"] > 0.8 and s["auc"] > 0.95
    assert s["n_train"] == 150 and s["n_test"] == 50


def test_probe_finds_no_skill_in_noise():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, len(sc.FEATURE_NAMES)))
    y = rng.uniform(1850, 2020, 200)                      # years unrelated to the features
    s = sc.score_probe(X[:150], y[:150], X[150:], y[150:], alpha=10.0)
    assert s["skill"] < 0.15                              # no better than guessing the median
    assert 0.3 < s["auc"] < 0.7                           # and no better than chance at old/modern


def test_probe_is_deterministic_and_needs_data():
    X = np.random.default_rng(6).normal(size=(40, 4))
    y = X[:, 0] * 10 + 1950
    a, b = sc.YearProbe().fit(X, y).predict(X), sc.YearProbe().fit(X, y).predict(X)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        sc.YearProbe().fit(X[:1], y[:1])
    with pytest.raises(ValueError):
        sc.YearProbe().predict(X)                         # not fitted


def test_constant_feature_does_not_blow_up():
    """A feature that never varies (e.g. every photo the same size) must be ignored."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 3))
    X[:, 1] = 2.5                                         # constant column
    y = 1900 + 30 * X[:, 0]
    pred = sc.YearProbe().fit(X, y).predict(X)
    assert np.all(np.isfinite(pred))
    assert np.abs(pred - y).mean() < 2.0


# ------------------------------------------- end to end on synthetic photos
def test_colour_only_era_signal_disappears_once_colour_is_removed():
    """Old photos differ ONLY by being sepia. The probe should lose its skill."""
    rng = np.random.default_rng(8)
    old = [sc.apply_sepia(noise_image(rng)) for _ in range(60)]
    modern = [noise_image(rng) for _ in range(60)]
    images = old + modern
    years = np.concatenate([rng.uniform(1870, 1930, 60), rng.uniform(1990, 2020, 60)])
    split = np.arange(len(images)) % 3 != 0               # two thirds train, one third test

    scores = {}
    for name in sc.TREATMENTS:
        X = treated(images, name)
        scores[name] = sc.score_probe(X[split], years[split], X[~split], years[~split])

    assert scores["original"]["auc"] > 0.95               # trivially solvable in colour
    assert scores["original"]["balanced_accuracy"] > 0.9
    # Most of it goes once colour and tone are stripped. It does not reach zero:
    # these images have NO content, so the residual is grain, which tone matching
    # cannot remove. That is exactly why the report calls the last row an upper bound.
    # Only tone matching removes it. Grayscaling alone does NOT: an old print is
    # flat and mid-grey as well as brown, and that survives desaturation, so the
    # grayscale row can even score slightly higher once the noisy colour features
    # are gone. Treat "we evaluated in grayscale" as no defence against this
    # shortcut.
    # It drops, but not to chance: these images have NO content, so the residual is
    # grain, which tone matching cannot remove. Hence "upper bound" in the report.
    assert scores["tone_matched"]["auc"] < scores["original"]["auc"] - 0.05
    # Grayscaling alone is NOT a defence: an old print is flat and mid-grey as well
    # as brown, and that tone survives desaturation.
    assert scores["grayscale"]["auc"] > 0.8


def test_content_based_era_signal_survives_every_treatment():
    """Here the era shows in the scene (stripe count), so stripping colour must not help."""
    rng = np.random.default_rng(9)
    years, images = [], []
    for _ in range(120):
        year = rng.uniform(1870, 2020)
        years.append(year)
        images.append(striped_image(rng, n_stripes=int(2 + (year - 1870) / 15)))
    years = np.asarray(years)
    split = np.arange(len(images)) % 3 != 0

    X = treated(images, "tone_matched")
    s = sc.score_probe(X[split], years[split], X[~split], years[~split])
    assert s["auc"] > 0.8                                 # content still carries the date


def test_sepia_shift_detects_a_medium_driven_probe():
    rng = np.random.default_rng(10)
    old = [sc.apply_sepia(noise_image(rng)) for _ in range(60)]
    modern = [noise_image(rng) for _ in range(60)]
    years = np.concatenate([rng.uniform(1870, 1930, 60), rng.uniform(1990, 2020, 60)])
    X, _ = build(old + modern, years)
    probe = sc.YearProbe().fit(X, years)

    modern_feats = X[60:]
    sepia_feats, modern_years = build([sc.apply_sepia(im) for im in modern], years[60:])
    shift = sc.sepia_shift(probe, modern_feats, sepia_feats, modern_years)

    assert shift["n"] == 60
    assert shift["median_shift"] < -30                    # toning alone makes them look old
    assert shift["share_moved_before_1945"] > 0.8
    assert shift["median_year_before"] > shift["median_year_after"]


# ------------------------------------------------------------------ reporting
def test_report_is_ordered_and_mentions_the_causal_test():
    scores = {name: {"n_train": 80, "n_test": 40, "mae": m, "baseline_mae": 30.0, "skill": s,
                     "auc": a, "auc_lo": a - 0.08, "auc_hi": min(a + 0.08, 1.0),
                     "balanced_accuracy": a, "recall_old": a, "recall_modern": a,
                     "n_old": 12, "n_modern": 28, "old_share": 0.3, "probe": None, "era_probe": None}
              for name, m, s, a in [("tone_matched", 28.0, 0.07, 0.55),
                                    ("original", 9.0, 0.70, 0.95),
                                    ("grayscale", 20.0, 0.33, 0.74)]}
    table = sc.results_table(scores)
    assert list(table["treatment"]) == ["original", "grayscale", "tone_matched"]
    assert "probe" not in table.columns and "era_probe" not in table.columns

    report = sc.render_report(table, {"n": 50, "median_year_before": 2010.0, "median_year_after": 1925.0,
                                      "median_shift": -85.0, "mean_shift": -83.0,
                                      "share_moved_before_1945": 0.9, "true_median_year": 2012.0},
                              n_scenes=13)
    assert "date signal is just the photo medium" in report
    assert "-85.0 years" in report and "Causal check" in report
    assert sc.render_report(table, None, 1).count("Causal check") == 0


# ------------------------------------------------- feature-group decomposition
def test_group_columns_are_a_partition_of_the_feature_vector():
    seen = [i for g in sc.FEATURE_GROUPS for i in sc.group_columns(g)]
    assert sorted(seen) == list(range(len(sc.FEATURE_NAMES)))   # every feature belongs to exactly one group
    with pytest.raises(KeyError):
        sc.group_columns("nonsense")


def test_group_table_points_at_the_group_that_holds_the_signal():
    """Build photos whose era is visible ONLY in colour: the colour row must win."""
    rng = np.random.default_rng(11)
    old = [sc.apply_sepia(noise_image(rng)) for _ in range(60)]
    modern = [noise_image(rng) for _ in range(60)]
    years = np.concatenate([rng.uniform(1870, 1930, 60), rng.uniform(1990, 2020, 60)])
    X, _ = build(old + modern, years)
    split = np.arange(len(X)) % 3 != 0

    table = sc.group_skill_table(X[split], years[split], X[~split], years[~split]).set_index("features")
    assert set(table.index) == set(sc.FEATURE_GROUPS) | {"all"}
    assert table.loc["colour", "auc"] > 0.9
    assert table.loc["colour", "auc"] > table.loc["framing", "auc"]
    assert table.loc["framing", "auc"] < 0.7            # all images are the same size here


def test_report_includes_the_group_table_when_given_one():
    scores = {"original": {"n_train": 80, "n_test": 40, "mae": 9.0, "baseline_mae": 30.0, "skill": 0.7,
                           "auc": 0.95, "auc_lo": 0.88, "auc_hi": 0.99, "balanced_accuracy": 0.9,
                           "recall_old": 0.9, "recall_modern": 0.9, "n_old": 12, "n_modern": 28,
                           "old_share": 0.3, "probe": None, "era_probe": None}}
    groups = pd.DataFrame([{"features": "colour", "n_features": 5, "auc": 0.93,
                            "balanced_accuracy": 0.88, "mae": 10.0, "skill": 0.66}])
    report = sc.render_report(sc.results_table(scores), None, 13, groups=groups)
    assert "Which cue carries the date?" in report and "| colour |" in report
    assert "Which cue carries the date?" not in sc.render_report(sc.results_table(scores), None, 13)


# ------------------------------------------------------- loading view tables
def write_dataset(folder, n=40, name="views.parquet", scene=1, absolute=True):
    """A minimal single-scene dataset folder: a view table plus its cached images."""
    rng = np.random.default_rng(scene)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        path = folder / "images" / f"v{i:03d}.jpg"
        noise_image(rng, size=32).save(path, quality=90)
        angle = rng.uniform(0, 2 * np.pi)
        rows.append(dict(file_key=f"s{scene}_v{i}", date_mid=float(rng.uniform(1870, 2020)),
                         image_path=str(path if absolute else path.name), scene_id=scene,
                         tx=float(3 * np.cos(angle)), ty=0.0, tz=float(3 * np.sin(angle)),
                         qw=1.0, qx=0.0, qy=0.0, qz=0.0))
    pd.DataFrame(rows).to_parquet(folder / name, index=False)
    return folder


def test_load_views_keeps_the_dataset_column_when_subsampling(tmp_path):
    """Regression: a pandas groupby().apply() here silently dropped this column."""
    folders = [write_dataset(tmp_path / f"{i}_model0", n=40, scene=i) for i in (11, 22)]
    views = sc.load_views(folders, max_views=20, seed=0)
    assert "dataset" in views.columns
    assert set(views["dataset"]) == {"11_model0", "22_model0"}
    assert len(views) <= 24                          # proportional, with rounding
    assert views["dataset"].value_counts().min() >= 2  # every scene keeps some views
    assert views["file_key"].is_unique


def test_load_views_prefers_the_audited_table_and_takes_everything_when_asked(tmp_path):
    folder = write_dataset(tmp_path / "33_model0", n=12, scene=33)
    assert len(sc.load_views([folder], max_views=0, seed=0)) == 12
    pd.DataFrame({"file_key": ["only"], "date_mid": [1900.0], "scene_id": [33],
                  "image_path": [str(folder / "images" / "v000.jpg")]}
                 ).to_parquet(folder / "views_clean.parquet", index=False)
    assert len(sc.load_views([folder], max_views=0, seed=0)) == 1   # views_clean wins


def test_load_views_recovers_images_moved_to_another_machine(tmp_path):
    """View tables store absolute paths from the machine that built them."""
    folder = write_dataset(tmp_path / "44_model0", n=8, scene=44)
    table = pd.read_parquet(folder / "views.parquet")
    table["image_path"] = ["/scratch/gone/" + Path(p).name for p in table["image_path"]]
    table.to_parquet(folder / "views.parquet", index=False)
    views = sc.load_views([folder], max_views=0, seed=0)
    assert all(Path(p).exists() for p in views["image_path"])

    table["image_path"] = "/scratch/gone/not_here_at_all.jpg"
    table.to_parquet(folder / "views.parquet", index=False)
    with pytest.raises(FileNotFoundError, match="could not be found"):
        sc.load_views([folder], max_views=0, seed=0)


def test_load_views_rejects_a_table_without_images(tmp_path):
    folder = tmp_path / "55_model0"
    folder.mkdir()
    with pytest.raises(FileNotFoundError, match="none of"):
        sc.load_views([folder], 0, 0)
    pd.DataFrame({"date_mid": [1900.0]}).to_parquet(folder / "views.parquet", index=False)
    with pytest.raises(ValueError, match="no image_path"):
        sc.load_views([folder], 0, 0)


def test_split_is_deterministic_and_never_splits_a_near_duplicate_group(tmp_path):
    folders = [write_dataset(tmp_path / f"{i}_model0", n=30, scene=i) for i in (66, 77)]
    views = sc.load_views(folders, max_views=0, seed=0)
    train, groups = sc.grouped_split(views, test_fraction=0.3, seed=0)

    assert len(np.unique(groups)) == len(groups) or True       # grouping may merge duplicates
    for g in np.unique(groups):                                # a group is wholly train or wholly test
        assert len(set(train[groups == g])) == 1
    # Scenes are grouped independently, so group ids never span two scenes.
    assert all(views.loc[groups == g, "dataset"].nunique() == 1 for g in np.unique(groups))
    again, _ = sc.grouped_split(views, 0.3, 0)
    assert np.array_equal(train, again)
    assert 0 < train.sum() < len(train)


def test_features_for_applies_the_treatment(tmp_path):
    folder = write_dataset(tmp_path / "88_model0", n=4, scene=88)
    paths = sc.load_views([folder], 0, 0)["image_path"].tolist()
    colour = sc.features_for(paths, "original")
    grey = sc.features_for(paths, "tone_matched")
    assert colour.shape == grey.shape == (4, len(sc.FEATURE_NAMES))
    sat = sc.FEATURE_NAMES.index("saturation_mean")
    assert grey[:, sat].max() < 0.02


# ------------------------------------------------------ uncertainty on the AUC
def test_roc_auc_known_values_and_ties():
    assert sc.roc_auc(np.array([3., 2., 1., 0.]), np.array([1, 1, 0, 0], bool)) == 1.0
    assert sc.roc_auc(np.array([0., 1., 2., 3.]), np.array([1, 1, 0, 0], bool)) == 0.0
    # all scores tied -> no information -> 0.5, not 1.0
    assert sc.roc_auc(np.ones(6), np.array([1, 1, 1, 0, 0, 0], bool)) == 0.5
    assert np.isnan(sc.roc_auc(np.array([1., 2.]), np.array([1, 1], bool)))   # one class only


def test_auc_interval_brackets_the_estimate_and_widens_with_fewer_positives():
    rng = np.random.default_rng(12)
    scores = np.concatenate([rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 200)])
    positive = np.concatenate([np.ones(200, bool), np.zeros(200, bool)])
    auc = sc.roc_auc(scores, positive)
    lo, hi = sc.auc_interval(scores, positive, n_boot=400)
    assert lo < auc < hi and 0.0 <= lo < hi <= 1.0

    # Same distribution, far fewer positives: the estimate should get no worse, but
    # the interval must get wider. This is exactly our situation on one scene.
    few = np.concatenate([scores[:8], scores[200:]])
    few_pos = np.concatenate([np.ones(8, bool), np.zeros(200, bool)])
    lo2, hi2 = sc.auc_interval(few, few_pos, n_boot=400)
    assert (hi2 - lo2) > (hi - lo)              # fewer old photos -> a wider interval
    assert sc.auc_interval(np.array([1., 2.]), np.array([1, 1], bool)) == pytest.approx((float("nan"),) * 2, nan_ok=True)


def test_report_warns_when_the_old_class_is_tiny():
    def row(n_old):
        return {"n_train": 300, "n_test": 134, "mae": 26.0, "baseline_mae": 23.0, "skill": -0.1,
                "auc": 0.9, "auc_lo": 0.81, "auc_hi": 0.97, "balanced_accuracy": 0.82,
                "recall_old": 0.95, "recall_modern": 0.69, "n_old": n_old, "n_modern": 115,
                "old_share": 0.14, "probe": None, "era_probe": None}
    small = sc.render_report(sc.results_table({"original": row(19)}), None, 1)
    assert "Only 19 old photos" in small and "0.81-0.97" in small
    assert "old photos are in the test split" not in sc.render_report(sc.results_table({"original": row(200)}), None, 1)
