"""Tests for the lens-distortion backfill.

The migration rewrites tables that training depends on, so the tests care most
about the two ways it could do quiet damage: writing a wrong coefficient where the
real one was available, and writing a partial table that trains happily while
silently dropping views.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timemachine.data import distortion as dz


def _views(n: int = 10, *, with_k: bool = False, camera_model: str = "SIMPLE_RADIAL",
           pose_source: str = "megascenes_colmap", keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys if keys is not None else [f"key{i}" for i in range(n)]
    frame = pd.DataFrame({
        "scene_id": 113197, "model": 0, "file_key": keys,
        "path": [f"images/{k}.jpg" for k in keys],
        "camera_width": 800, "camera_height": 600,
        "fx": 700.0, "fy": 700.0, "cx": 400.0, "cy": 300.0,
        "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "tx": 0.0, "ty": 0.0, "tz": 1.0,
        "date_mid": 1970.0, "era": "modern", "medium": "photo",
        "camera_model": camera_model, "pose_source": pose_source,
    })
    if with_k:
        frame["k1"], frame["k2"] = -0.01, 0.0
    return frame


def _poses(keys: list[str], k1: float = -0.042, k2: float = 0.0,
           camera_model: str = "SIMPLE_RADIAL") -> pd.DataFrame:
    return pd.DataFrame({"file_key": keys, "model": 0, "camera_model": camera_model,
                         "k1": k1, "k2": k2})


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    out = tmp_path / "113197_model0"
    out.mkdir()
    _views(10).to_parquet(out / "views.parquet")
    return out


# --------------------------------------------------------------------------- parsing

@pytest.mark.parametrize("name, expected", [
    ("113197_model0", (113197, 0)),
    ("1752_model9", (1752, 9)),
    ("373910_model2", (373910, 2)),
])
def test_dataset_folders_parse(tmp_path: Path, name: str, expected: tuple[int, int]):
    assert dz.parse_dataset_folder(tmp_path / name) == expected


@pytest.mark.parametrize("name", ["images", "113197", "113197_model", "scene_model0", ""])
def test_other_folders_are_rejected(tmp_path: Path, name: str):
    assert dz.parse_dataset_folder(tmp_path / name) is None


# --------------------------------------------------------------------------- detection

def test_a_table_without_the_columns_needs_migrating():
    assert dz.needs_migration(_views(5))


def test_a_table_with_real_coefficients_does_not():
    assert not dz.needs_migration(_views(5, with_k=True))


def test_an_all_unknown_table_needs_migrating():
    """A half-finished migration leaves the columns present but useless."""
    views = _views(5, with_k=True)
    views["k1"] = np.nan
    assert dz.needs_migration(views)


def test_a_partly_filled_table_is_left_alone():
    """Some rows genuinely have no recoverable coefficients; that is not staleness."""
    views = _views(5, with_k=True)
    views.loc[0, "k1"] = np.nan
    assert not dz.needs_migration(views)


# --------------------------------------------------------------------------- the join

def test_real_coefficients_are_joined_on():
    views = _views(10)
    out, result = dz.attach_distortion(views, _poses(list(views["file_key"]), k1=-0.042))
    assert result.ok
    assert result.matched == 10
    assert result.nonzero_k1 == 10
    assert out["k1"].eq(-0.042).all()
    assert result.k1_max == pytest.approx(0.042)


def test_existing_stale_columns_are_replaced_not_merged():
    """A table with the columns present but wrong must end up with the real values."""
    views = _views(5, with_k=True)                     # carries k1 = -0.01
    out, _ = dz.attach_distortion(views, _poses(list(views["file_key"]), k1=-0.042))
    assert out["k1"].eq(-0.042).all()
    assert list(out.columns).count("k1") == 1


def test_coefficients_land_where_the_schema_puts_them():
    out, _ = dz.attach_distortion(_views(5), _poses([f"key{i}" for i in range(5)]))
    columns = list(out.columns)
    assert columns[columns.index("cy") + 1:columns.index("cy") + 3] == ["k1", "k2"]


def test_unmatched_rows_are_dropped_rather_than_zeroed():
    """Zero is a real coefficient value, so it must never stand in for 'unknown'.

    A zeroed row trains on a crop that was never undistorted. A non-finite one is
    no better: load_view raises on it (it does not skip), killing the run at the
    first step that draws it. So the row goes, and the counts keep it visible.
    """
    views = _views(10)
    out, result = dz.attach_distortion(views, _poses([f"key{i}" for i in range(6)]))
    assert result.matched == 6
    assert result.unmatched == 4
    assert result.dropped == 4
    assert not result.ok
    assert len(out) == 6
    assert out["file_key"].tolist() == [f"key{i}" for i in range(6)]
    assert np.isfinite(out[["k1", "k2"]].to_numpy()).all()
    assert (out["k1"] == 0).sum() == 0


def test_pinhole_cameras_are_filled_with_exact_zeros():
    """PINHOLE has no distortion parameters, so zero is its value, not a guess."""
    views = _views(5, camera_model="PINHOLE")
    out, result = dz.attach_distortion(views, _poses([], camera_model="PINHOLE"))
    assert result.ok
    assert out["k1"].eq(0.0).all() and out["k2"].eq(0.0).all()
    assert result.nonzero_k1 == 0


def test_predicted_poses_are_filled_with_exact_zeros():
    """VGGT-Omega predicts pinhole cameras; those rows are not in a COLMAP table."""
    views = _views(5, pose_source="vggt_omega")
    out, result = dz.attach_distortion(views, _poses([]))
    assert result.ok
    assert out["k1"].eq(0.0).all()
    assert result.predicted == 5


def test_radial_cameras_are_never_filled_with_zeros():
    """The whole point: a SIMPLE_RADIAL row with no match is not given k1 = 0.

    It is dropped instead - checked by count, since "every remaining k1 is NaN"
    would also hold, vacuously, of an empty table.
    """
    out, result = dz.attach_distortion(_views(5), _poses([]))
    assert not result.ok
    assert result.matched == 0 and result.dropped == 5
    assert out.empty


def test_mixed_models_are_handled_row_by_row():
    views = pd.concat([_views(3, keys=["a", "b", "c"]),
                       _views(2, camera_model="PINHOLE", keys=["d", "e"])], ignore_index=True)
    out, result = dz.attach_distortion(views, _poses(["a", "b", "c"], k1=-0.03))
    assert result.ok and result.matched == 5
    assert out.loc[out["file_key"].isin(["a", "b", "c"]), "k1"].eq(-0.03).all()
    assert out.loc[out["file_key"].isin(["d", "e"]), "k1"].eq(0.0).all()


def test_duplicate_registrations_do_not_multiply_rows():
    """A file registered twice in one model shares its camera; the join must not fan out."""
    keys = [f"key{i}" for i in range(5)]
    poses = pd.concat([_poses(keys), _poses(keys)], ignore_index=True)
    out, result = dz.attach_distortion(_views(5), poses)
    assert len(out) == 5 and result.ok


def test_poses_without_the_columns_are_refused_with_a_reason():
    """The poses table can be stale too, which is the case that bit these datasets."""
    poses = _poses([f"key{i}" for i in range(5)]).drop(columns=["k1", "k2"])
    _, result = dz.attach_distortion(_views(5), poses)
    assert not result.ok
    assert "re-derive" in result.note


def test_a_table_without_file_key_cannot_be_joined():
    _, result = dz.attach_distortion(_views(5).drop(columns=["file_key"]), _poses([]))
    assert not result.ok
    assert "file_key" in result.note


# --------------------------------------------------------------------------- writing

def test_migration_writes_the_table_and_keeps_the_original(folder: Path):
    result = dz.migrate_folder(folder, _poses([f"key{i}" for i in range(10)], k1=-0.042))
    assert result.written and result.ok

    out = pd.read_parquet(folder / "views.parquet")
    assert out["k1"].eq(-0.042).all()
    spare = folder / f"views{dz.BACKUP_SUFFIX}"
    assert spare.is_file()
    assert "k1" not in pd.read_parquet(spare).columns


def test_a_second_run_does_not_overwrite_the_first_backup(folder: Path):
    poses = _poses([f"key{i}" for i in range(10)], k1=-0.042)
    dz.migrate_folder(folder, poses)
    before = (folder / f"views{dz.BACKUP_SUFFIX}").read_bytes()
    # Force a second migration over the already-migrated table.
    views = pd.read_parquet(folder / "views.parquet")
    views["k1"] = np.nan
    views.to_parquet(folder / "views.parquet")
    dz.migrate_folder(folder, poses)
    assert (folder / f"views{dz.BACKUP_SUFFIX}").read_bytes() == before


def test_a_partial_migration_is_refused_by_default(folder: Path):
    """Training on a half-migrated table looks healthy and silently drops views."""
    result = dz.migrate_folder(folder, _poses([f"key{i}" for i in range(6)]))
    assert not result.written
    assert "--allow-partial" in result.note
    assert "k1" not in pd.read_parquet(folder / "views.parquet").columns


def test_a_partial_migration_can_be_forced_and_drops_the_unmatched_rows(folder: Path):
    result = dz.migrate_folder(folder, _poses([f"key{i}" for i in range(6)]),
                               allow_partial=True)
    assert result.written
    written = pd.read_parquet(folder / "views.parquet")
    assert len(written) == 6
    assert np.isfinite(written[["k1", "k2"]].to_numpy()).all()
    assert "4 unmatched, dropped" in result.line()


def test_a_forced_migration_never_writes_an_empty_table(folder: Path):
    result = dz.migrate_folder(folder, _poses([]), allow_partial=True)
    assert not result.written
    assert "empty" in result.note
    assert "k1" not in pd.read_parquet(folder / "views.parquet").columns


def test_dry_run_changes_nothing(folder: Path):
    result = dz.migrate_folder(folder, _poses([f"key{i}" for i in range(10)]), dry_run=True)
    assert result.ok and not result.written
    assert result.dry_run
    assert "k1" not in pd.read_parquet(folder / "views.parquet").columns
    assert not (folder / f"views{dz.BACKUP_SUFFIX}").exists()


def test_backup_can_be_declined(folder: Path):
    dz.migrate_folder(folder, _poses([f"key{i}" for i in range(10)]), backup=False)
    assert not (folder / f"views{dz.BACKUP_SUFFIX}").exists()


def test_an_already_migrated_folder_is_left_alone(folder: Path):
    _views(10, with_k=True).to_parquet(folder / "views.parquet")
    result = dz.migrate_folder(folder, _poses([f"key{i}" for i in range(10)], k1=-0.9))
    assert not result.written
    assert "already" in result.note
    assert pd.read_parquet(folder / "views.parquet")["k1"].eq(-0.01).all()


def test_a_folder_without_a_table_is_skipped(tmp_path: Path):
    empty = tmp_path / "209_model0"
    empty.mkdir()
    result = dz.migrate_folder(empty, _poses([]))
    assert not result.written
    assert "no views parquet" in result.note


def test_the_clean_table_is_migrated_when_present(folder: Path):
    _views(4).to_parquet(folder / "views_clean.parquet")
    dz.migrate_folder(folder, _poses([f"key{i}" for i in range(10)], k1=-0.042))
    assert "k1" in pd.read_parquet(folder / "views_clean.parquet").columns
    assert "k1" not in pd.read_parquet(folder / "views.parquet").columns


# --------------------------------------------------------------------------- report

def test_report_totals_across_datasets(tmp_path: Path):
    results = []
    for name in ("113197_model0", "1752_model0"):
        out = tmp_path / name
        out.mkdir()
        _views(10).to_parquet(out / "views.parquet")
        results.append(dz.migrate_folder(out, _poses([f"key{i}" for i in range(10)])))
    rendered = dz.render_report(results)
    assert "2 dataset(s), 20 views" in rendered
    assert "rewrote 2 view table(s)" in rendered


def test_report_names_the_datasets_it_could_not_migrate(tmp_path: Path):
    out = tmp_path / "113197_model0"
    out.mkdir()
    _views(10).to_parquet(out / "views.parquet")
    rendered = dz.render_report([dz.migrate_folder(out, _poses(["key0"]))])
    assert "1 dataset(s) not migrated" in rendered
    assert "113197_model0" in rendered


# ------------------------------------------------------------------- invertibility

def test_radius_grows_as_the_lens_gets_wider():
    """Same sensor, shorter focal length: the corner sits further off axis."""
    wide = _views(1); wide["fx"] = wide["fy"] = 350.0
    tele = _views(1); tele["fx"] = tele["fy"] = 2000.0
    assert dz.max_radius_squared(wide)[0] > dz.max_radius_squared(tele)[0]


def test_radius_uses_the_furthest_corner_when_the_centre_is_offset():
    """A cropped sensor has an off-centre principal point; the near corner is not
    the one that matters."""
    centred = _views(1)
    offset = _views(1); offset["cx"] = 50.0          # optical axis near the left edge
    assert dz.max_radius_squared(offset)[0] > dz.max_radius_squared(centred)[0]


def test_a_zero_focal_length_does_not_blow_up():
    broken = _views(1); broken["fx"] = 0.0
    assert not np.isfinite(dz.max_radius_squared(broken)[0])


def test_mild_distortion_is_invertible():
    r2 = np.array([0.5])
    assert dz.is_invertible(np.array([-0.05]), np.array([0.0]), r2)[0]


def test_severe_barrel_distortion_is_not():
    """1 + 3*k1*r2 goes negative, so the radial map folds and cannot be undone."""
    r2 = np.array([0.5])
    assert not dz.is_invertible(np.array([-0.9]), np.array([0.0]), r2)[0]


def test_the_bound_depends_on_the_camera_not_a_global_threshold():
    """The same coefficient is fine on a long lens and ruinous on a wide one.

    This is the reason for computing r_max per camera: a single cutoff on |k1| would
    reject good telephoto rows and accept bad wide-angle ones.
    """
    k1, k2 = np.array([-0.9]), np.array([0.0])
    assert dz.is_invertible(k1, k2, np.array([0.1]))[0]        # narrow field of view
    assert not dz.is_invertible(k1, k2, np.array([0.8]))[0]    # wide field of view


def test_k2_participates_in_the_bound():
    r2 = np.array([0.5])
    assert not dz.is_invertible(np.array([0.0]), np.array([-2.0]), r2)[0]


def test_a_non_finite_coefficient_is_not_invertible():
    r2 = np.array([0.5])
    assert not dz.is_invertible(np.array([np.nan]), np.array([0.0]), r2)[0]


def test_uninvertible_rows_are_dropped_and_counted():
    views = _views(10)
    poses = pd.concat([_poses([f"key{i}" for i in range(7)], k1=-0.05),
                       _poses([f"key{i}" for i in range(7, 10)], k1=-3.0)],
                      ignore_index=True)
    out, result = dz.attach_distortion(views, poses)
    assert result.implausible == 3
    assert result.matched == 7
    assert result.dropped == 3
    assert len(out) == 7
    assert not out["file_key"].isin(["key7", "key8", "key9"]).any()
    assert out["k1"].eq(-0.05).all()                     # the survivors keep their real value


def test_uninvertible_rows_can_be_kept_deliberately():
    views = _views(5)
    poses = _poses(list(views["file_key"]), k1=-3.0)
    out, result = dz.attach_distortion(views, poses, drop_uninvertible=False)
    assert result.implausible == 0
    assert out["k1"].eq(-3.0).all()


def test_a_table_whose_only_gaps_are_uninvertible_rows_still_writes(folder: Path):
    """Those rows are an intended outcome, not a half-finished migration, so they
    do not need --allow-partial."""
    poses = pd.concat([_poses([f"key{i}" for i in range(8)], k1=-0.05),
                       _poses([f"key{i}" for i in range(8, 10)], k1=-3.0)],
                      ignore_index=True)
    result = dz.migrate_folder(folder, poses)
    assert result.written
    written = pd.read_parquet(folder / "views.parquet")
    assert len(written) == 8
    assert np.isfinite(written[["k1", "k2"]].to_numpy()).all()
    # The dropped views are still recoverable from the backup.
    assert len(pd.read_parquet(folder / ("views" + dz.BACKUP_SUFFIX))) == 10


def test_a_genuinely_partial_table_is_still_refused(folder: Path):
    """Rows the poses table never covered are different from rows we rejected."""
    poses = pd.concat([_poses([f"key{i}" for i in range(6)], k1=-0.05),
                       _poses([f"key{i}" for i in range(6, 8)], k1=-3.0)],
                      ignore_index=True)
    result = dz.migrate_folder(folder, poses)
    assert not result.written
    assert "--allow-partial" in result.note


def test_report_mentions_the_dropped_rows(folder: Path):
    poses = pd.concat([_poses([f"key{i}" for i in range(8)], k1=-0.05),
                       _poses([f"key{i}" for i in range(8, 10)], k1=-3.0)],
                      ignore_index=True)
    rendered = dz.render_report([dz.migrate_folder(folder, poses)])
    assert "cannot be inverted" in rendered
    assert "dropped from the table" in rendered
    assert "skips" not in rendered                       # load_view raises; it never skipped


def test_report_shows_the_median_not_only_the_maximum():
    """A maximum alone cannot distinguish three bad cameras from half the data."""
    views = _views(10)
    poses = pd.concat([_poses([f"key{i}" for i in range(9)], k1=-0.02),
                       _poses(["key9"], k1=-0.3)], ignore_index=True)
    _, result = dz.attach_distortion(views, poses)
    assert result.k1_median == pytest.approx(0.02)
    assert result.k1_max == pytest.approx(0.3)


def test_dry_run_writes_nothing_even_when_rows_are_rejected(folder: Path):
    """The uninvertible branch writes; it must still honour --dry-run.

    This went wrong once: the rejection path returned before the dry-run check, so
    a run meant only to report would have rewritten every table.
    """
    poses = pd.concat([_poses([f"key{i}" for i in range(8)], k1=-0.05),
                       _poses([f"key{i}" for i in range(8, 10)], k1=-3.0)],
                      ignore_index=True)
    result = dz.migrate_folder(folder, poses, dry_run=True)
    assert result.dry_run and not result.written
    assert "k1" not in pd.read_parquet(folder / "views.parquet").columns
    assert not (folder / f"views{dz.BACKUP_SUFFIX}").exists()


def test_a_report_of_a_dry_run_does_not_claim_to_have_written(folder: Path):
    poses = _poses([f"key{i}" for i in range(10)], k1=-0.05)
    rendered = dz.render_report([dz.migrate_folder(folder, poses, dry_run=True)])
    assert "rewrote" not in rendered


def test_rejected_rows_do_not_make_a_migration_a_failure(folder: Path):
    poses = pd.concat([_poses([f"key{i}" for i in range(8)], k1=-0.05),
                       _poses([f"key{i}" for i in range(8, 10)], k1=-3.0)],
                      ignore_index=True)
    result = dz.migrate_folder(folder, poses)
    assert result.ok and result.written
    assert result.line().strip().startswith("ok")
