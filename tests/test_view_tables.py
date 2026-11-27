"""No view table may hold a row that ``load_view`` cannot load.

``load_view`` raises on non-finite ``k1``/``k2`` - it does not skip them - so one
such row kills a training run at whichever step first draws it. The first
distortion migration wrote 49 of them across the 34 datasets, believing they would
be skipped; the preflight passed all 34 tables anyway.

The rule now holds at every point a row can enter a table:

* the scene build drops them (``build_view_table``), and uninvertible ones too;
* the migration drops them (``attach_distortion``; tested in test_distortion.py);
* existing tables are cleaned by ``prune_table`` / scripts/prune_unloadable_views.py;
* ``DatedViewsDataset`` refuses a table that still has one, at construction;
* the preflight reports one, per dataset.

The first test checks the real tables. They live under ``data/``, which is not in
git, so on a machine without them that one test has nothing to check and says so;
everything after it uses synthetic tables and runs everywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timemachine.config import DataPaths
from timemachine.data import distortion as dz

REPO = Path(__file__).resolve().parents[1]


def _real_view_tables() -> list[Path]:
    roots = {DataPaths().root / "datasets", REPO / "data" / "datasets"}
    return sorted(t for root in roots if root.is_dir()
                  for name in ("views.parquet", "views_clean.parquet")
                  for t in root.glob(f"*/{name}"))


# --------------------------------------------------------------------------- the real tables

def test_no_real_view_table_has_non_finite_lens_coefficients():
    tables = _real_view_tables()
    if not tables:
        pytest.skip("no view tables on this machine (data/ is not in git); "
                    "the synthetic tests below still cover every code path")
    offenders = {}
    for table in tables:
        views = pd.read_parquet(table, columns=["file_key", "k1", "k2"])
        bad = ~dz.finite_distortion(views)
        if bad.any():
            offenders[f"{table.parent.name}/{table.name}"] = views.loc[bad, "file_key"].tolist()[:3]
    assert not offenders, (f"{len(offenders)} table(s) hold views load_view cannot load; run "
                           f"scripts/prune_unloadable_views.py: {offenders}")


# --------------------------------------------------------------------------- the rule itself

def _geometry(n: int, k1, k2=0.0) -> pd.DataFrame:
    return pd.DataFrame({"file_key": [f"v{i}" for i in range(n)], "k1": k1, "k2": k2,
                         "fx": 700.0, "fy": 700.0, "cx": 400.0, "cy": 300.0,
                         "camera_width": 800, "camera_height": 600})


def test_finite_means_both_coefficients_are_finite():
    views = _geometry(4, k1=[0.0, np.nan, -0.02, np.inf], k2=[0.0, 0.0, np.nan, 0.0])
    assert dz.finite_distortion(views).tolist() == [True, False, False, False]


def test_a_table_without_the_columns_has_no_loadable_row():
    assert not dz.finite_distortion(pd.DataFrame({"file_key": ["a", "b"]})).any()


def test_usable_also_requires_an_invertible_map():
    views = _geometry(3, k1=[-0.05, -3.0, np.nan])
    assert dz.finite_distortion(views).tolist() == [True, True, False]
    assert dz.usable_distortion(views).tolist() == [True, False, False]


def test_usable_refuses_to_guess_without_the_geometry():
    with pytest.raises(KeyError):
        dz.usable_distortion(pd.DataFrame({"k1": [0.0], "k2": [0.0]}))


# --------------------------------------------------------------------------- pruning

@pytest.fixture
def dirty(tmp_path: Path) -> Path:
    folder = tmp_path / "1752_model0"
    folder.mkdir()
    views = _geometry(10, k1=-0.02)
    views.loc[[2, 7], "k1"] = np.nan                   # what the first migration wrote
    views.to_parquet(folder / "views.parquet", index=False)
    return folder


def test_prune_drops_exactly_the_non_finite_rows(dirty: Path):
    [result] = dz.prune_folder(dirty)
    assert result.ok and result.written
    assert result.dropped_keys == ["v2", "v7"]
    kept = pd.read_parquet(dirty / "views.parquet")
    assert kept["file_key"].tolist() == [f"v{i}" for i in range(10) if i not in (2, 7)]
    assert dz.finite_distortion(kept).all()
    assert kept.index.tolist() == list(range(8))        # a clean index, not holes


def test_prune_keeps_the_original_and_never_overwrites_that_backup(dirty: Path):
    dz.prune_folder(dirty)
    spare = dirty / ("views" + dz.PRUNE_BACKUP_SUFFIX)
    assert len(pd.read_parquet(spare)) == 10
    # A second run finds nothing to do and must leave the 10-row backup alone.
    [again] = dz.prune_folder(dirty)
    assert again.dropped == 0 and not again.written
    assert len(pd.read_parquet(spare)) == 10


def test_prune_does_not_touch_the_migration_backup(dirty: Path):
    migration_backup = dirty / ("views" + dz.BACKUP_SUFFIX)
    pd.DataFrame({"file_key": ["original"]}).to_parquet(migration_backup)
    dz.prune_folder(dirty)
    assert pd.read_parquet(migration_backup)["file_key"].tolist() == ["original"]


def test_prune_dry_run_writes_nothing(dirty: Path):
    before = (dirty / "views.parquet").read_bytes()
    [result] = dz.prune_folder(dirty, dry_run=True)
    assert result.dropped == 2 and not result.written
    assert (dirty / "views.parquet").read_bytes() == before
    assert not (dirty / ("views" + dz.PRUNE_BACKUP_SUFFIX)).exists()


def test_prune_without_backup(dirty: Path):
    dz.prune_folder(dirty, backup=False)
    assert not (dirty / ("views" + dz.PRUNE_BACKUP_SUFFIX)).exists()


def test_prune_refuses_a_stale_table_rather_than_emptying_it(tmp_path: Path):
    """An all-NaN or column-less table needs migrating; pruning it would leave nothing."""
    for name, frame in [("nan_model0", _geometry(4, k1=np.nan)),
                        ("nocols_model0", _geometry(4, k1=0.0).drop(columns=["k1", "k2"]))]:
        folder = tmp_path / name
        folder.mkdir()
        frame.to_parquet(folder / "views.parquet", index=False)
        [result] = dz.prune_folder(folder)
        assert not result.ok and not result.written, name
        assert len(pd.read_parquet(folder / "views.parquet")) == 4


def test_prune_cleans_both_tables_when_a_folder_has_both(dirty: Path):
    """The trainer reads views.parquet; migration and preflight prefer views_clean."""
    pd.read_parquet(dirty / "views.parquet").to_parquet(dirty / "views_clean.parquet", index=False)
    results = dz.prune_folder(dirty)
    assert sorted(r.table.name for r in results) == ["views.parquet", "views_clean.parquet"]
    for name in ("views.parquet", "views_clean.parquet"):
        assert dz.finite_distortion(pd.read_parquet(dirty / name)).all(), name


def test_prune_report_counts_rows_and_tables(dirty: Path):
    rendered = dz.render_prune_report(dz.prune_folder(dirty))
    assert "dropped 2 of 10 rows" in rendered
    assert "8 rows remain" in rendered


# --------------------------------------------------------------------------- the scene build

def _scene_tables(root: Path, k1: list[float]) -> DataPaths:
    """The three per-scene tables build_view_table reads, for one model of one scene."""
    from timemachine.dating.medium import Medium

    paths = DataPaths(root=root)
    n = len(k1)
    keys = [f"f{i}" for i in range(n)]
    poses = pd.DataFrame({
        "scene_id": 7, "model": 0, "image_id": range(n), "name": [f"x/{k}.jpg" for k in keys],
        "path": [f"x/{k}.jpg" for k in keys], "file_key": keys, "camera_model": "SIMPLE_RADIAL",
        "width": 800, "height": 600, "fx": 700.0, "fy": 700.0, "cx": 400.0, "cy": 300.0,
        "k1": k1, "k2": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
        "tx": 0.0, "ty": 0.0, "tz": 1.0})
    dates = pd.DataFrame({"file_key": keys, "date_lo": 1900.0, "date_hi": 1910.0,
                          "date_mid": 1905.0, "date_width": 10.0, "date_confidence": 0.9,
                          "medium": Medium.PHOTO,
                          "color_hint": "mono"})
    meta = pd.DataFrame({"file_key": keys, "title": keys})
    for table, frame in [(paths.poses_table(7), poses), (paths.dates_table(7), dates),
                         (paths.image_metadata_table(7), meta)]:
        table.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(table, index=False)
    return paths


def test_the_scene_build_drops_unusable_coefficients(tmp_path: Path):
    """The migration rewrote the poses tables with the raw cameras.bin values, so a
    rebuild sees the wild finite k1 the migration rejected. It must reject it too."""
    from timemachine.data.views import build_view_table

    paths = _scene_tables(tmp_path, k1=[-0.05, -3.0, np.nan, -0.02])
    views = build_view_table(paths, 7, 0)
    assert views["file_key"].tolist() == ["f0", "f3"]
    assert views.attrs["bad_distortion"] == ["f1", "f2"]
    assert dz.usable_distortion(views).all()


def test_the_scene_build_filter_can_be_turned_off(tmp_path: Path):
    from timemachine.data.views import build_view_table

    paths = _scene_tables(tmp_path, k1=[-0.05, -3.0])
    views = build_view_table(paths, 7, 0, drop_bad_distortion=False)
    assert views["file_key"].tolist() == ["f0", "f1"]
    assert views.attrs["bad_distortion"] == []


def test_a_rebuilt_table_matches_a_migrated_one(tmp_path: Path):
    """Both paths into a view table apply one rule, so they cannot disagree."""
    from timemachine.data.views import build_view_table

    k1 = [-0.05, -3.0, -0.02, -1.5]
    paths = _scene_tables(tmp_path, k1)
    built = build_view_table(paths, 7, 0)
    # A table from before the lens columns existed: every view, no coefficients.
    stale = build_view_table(paths, 7, 0, drop_bad_distortion=False).drop(columns=["k1", "k2"])
    poses = pd.DataFrame({"file_key": [f"f{i}" for i in range(4)], "model": 0,
                          "camera_model": "SIMPLE_RADIAL", "k1": k1, "k2": 0.0})
    migrated, _ = dz.attach_distortion(stale, poses)
    assert migrated["file_key"].tolist() == built["file_key"].tolist()


# --------------------------------------------------------------------------- the dataset

def test_the_dataset_refuses_a_table_with_an_unloadable_row():
    """At construction, not at the step that first draws the row."""
    from timemachine.data.dataset import _REQUIRED, DatedViewsDataset

    views = pd.DataFrame({c: [0.0] * 3 for c in _REQUIRED})
    views["file_key"] = ["a", "b", "c"]
    views.loc[1, "k1"] = np.nan
    with pytest.raises(ValueError, match="prune_unloadable_views"):
        DatedViewsDataset(views)
