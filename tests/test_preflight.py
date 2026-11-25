"""Tests for the training preflight.

The preflight exists to be trusted when it says something is broken, so these
tests care as much about it staying quiet on a healthy setup as about it catching a
broken one. A check that cries wolf gets ignored, and then it may as well not run.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from timemachine.training import preflight as pf


# --------------------------------------------------------------------------- fixtures

def _views(n: int = 20, *, old: int = 0, root: Path | None = None, drop: str = "") -> pd.DataFrame:
    """A view table carrying every column the dataset requires, plus the usual extras.

    The required columns are generated from ``required_view_columns()`` rather than
    listed here, because a hardcoded fixture is the same mistake as a hardcoded
    requirement: when the dataset gained ``image_path`` and friends, a fixture built
    from an assumed list declared perfectly good tables broken.
    """
    rows = []
    for i in range(n):
        year = 1900 + (i % 44) if i < old else 2015   # always < 1945
        image = str((root / f"img{i}.jpg") if root else f"images/img{i}.jpg")
        rows.append({
            "scene_id": "Scene_A", "model": 0, "file_key": f"k{i}",
            "path": f"commons/img{i}.jpg",            # bucket key, not a local file
            "image_path": image, "title": f"img{i}.jpg",
            "image_width": 800, "image_height": 600,
            "camera_width": 800, "camera_height": 600,
            "fx": 700.0, "fy": 700.0, "cx": 400.0, "cy": 300.0, "k1": 0.0, "k2": 0.0,
            "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "tx": 0.0, "ty": 0.0, "tz": 1.0,
            "date_lo": float(year), "date_hi": float(year), "date_mid": float(year),
            "date_confidence": 1.0, "era": "old" if year < 1945 else "modern",
            "medium": "photo", "color_hint": "color", "is_monochrome": year < 1945,
            "pose_source": "colmap",
        })
    frame = pd.DataFrame(rows)

    # Anything the dataset requires that this fixture does not know about still has to
    # be present, or every test fails for a reason that has nothing to do with it.
    for column in pf.required_view_columns():
        if column not in frame.columns:
            frame[column] = 0.0
    return frame.drop(columns=[drop]) if drop else frame


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A dataset folder whose table points at images that are really there."""
    folder = tmp_path / "Scene_A_model0"
    images = folder / "images"
    images.mkdir(parents=True)
    for i in range(20):
        (images / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")
    _views(20, old=12, root=images).to_parquet(folder / "views.parquet")
    return folder


@pytest.fixture
def hub(tmp_path: Path) -> Path:
    """A hub cache holding a filled snapshot for every required repo."""
    root = tmp_path / "hub"
    for repo in pf.REQUIRED_REPOS:
        snap = root / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc123def456"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"\x00")
    return root


# --------------------------------------------------------------------------- report

def test_report_is_ok_when_every_check_passed():
    report = pf.Report()
    report.add(pf.Check("a", True), pf.Check("b", True))
    assert report.ok
    assert "Ready to submit." in report.render()


def test_skipped_checks_do_not_block():
    report = pf.Report()
    report.add(pf.Check("gpu", False, skipped=True))
    assert report.ok
    assert report.failures == []


def test_report_lists_the_fix_for_each_failure():
    report = pf.Report()
    report.add(pf.Check("weights", False, "missing", "download them"))
    rendered = report.render()
    assert not report.ok
    assert "1 blocking problem" in rendered
    assert "download them" in rendered


def test_marks_are_distinguishable():
    assert pf.Check("x", True).mark == "ok"
    assert pf.Check("x", False).mark == "FAIL"
    assert pf.Check("x", False, skipped=True).mark == "--"


# --------------------------------------------------------------------------- weights

def test_healthy_hub_cache_passes_silently(hub: Path):
    checks = pf.check_weights(hub)
    assert all(c.ok for c in checks), [c.detail for c in checks if not c.ok]
    assert len(checks) == 1 + len(pf.REQUIRED_REPOS)


def test_missing_hub_cache_reports_once_and_stops(tmp_path: Path):
    checks = pf.check_weights(tmp_path / "nope")
    assert len(checks) == 1
    assert not checks[0].ok


def test_each_missing_repo_is_named_separately(hub: Path):
    import shutil

    shutil.rmtree(hub / "models--stabilityai--stable-virtual-camera")
    failures = [c for c in pf.check_weights(hub) if not c.ok]
    assert len(failures) == 1
    assert "stable-virtual-camera" in failures[0].name
    assert "gated" in failures[0].detail


def test_an_interrupted_download_is_not_mistaken_for_a_real_one(hub: Path):
    """The folder and an empty snapshot survive a cancelled download."""
    snap = hub / "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K" / "snapshots" / "abc123def456"
    for f in snap.iterdir():
        f.unlink()
    failures = [c for c in pf.check_weights(hub) if not c.ok]
    assert [c.name for c in failures] == ["weights CLIP-ViT-H-14-laion2B-s32B-b79K"]


def test_explicit_hub_beats_the_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TIMEMACHINE_HF_HUB", str(tmp_path / "from-env"))
    assert pf.hub_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_hub_root_appends_hub_to_a_bare_cache_dir(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TIMEMACHINE_HF_HUB", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
    assert pf.hub_root().name == "hub"


# --------------------------------------------------------------------------- datasets

def test_healthy_dataset_passes(dataset: Path):
    checks = pf.check_datasets([dataset], min_old=10)
    assert all(c.ok for c in checks), [c.detail for c in checks if not c.ok]
    assert "20 views" in checks[0].detail


def test_no_datasets_is_a_failure_not_a_pass():
    assert not pf.check_datasets([])[0].ok


def test_unbuilt_dataset_says_how_to_build_it(tmp_path: Path):
    folder = tmp_path / "Scene_B_model0"
    folder.mkdir()
    check = pf.check_datasets([folder])[0]
    assert not check.ok
    assert "build_scene_dataset" in check.fix


def test_clean_table_is_preferred_over_the_raw_one(dataset: Path):
    _views(5, old=5, root=dataset / "images").to_parquet(dataset / "views_clean.parquet")
    assert "5 views" in pf.check_datasets([dataset])[0].detail


def test_missing_required_column_is_named(dataset: Path):
    _views(20, old=12, root=dataset / "images", drop="k1").to_parquet(dataset / "views.parquet")
    check = pf.check_datasets([dataset])[0]
    assert not check.ok
    assert "k1" in check.detail


def test_empty_table_is_caught(dataset: Path):
    _views(0).to_parquet(dataset / "views.parquet")
    assert not pf.check_datasets([dataset])[0].ok


def test_corrupt_table_is_reported_not_raised(dataset: Path):
    (dataset / "views.parquet").write_bytes(b"not a parquet file")
    check = pf.check_datasets([dataset])[0]
    assert not check.ok
    assert "unreadable" in check.detail


def test_images_missing_on_this_machine_are_caught(tmp_path: Path):
    """The failure mode of copying a dataset built elsewhere."""
    folder = tmp_path / "Scene_C_model0"
    folder.mkdir()
    _views(20, old=12, root=Path("/somewhere/that/does/not/exist")).to_parquet(
        folder / "views.parquet")
    check = pf.check_datasets([folder])[0]
    assert not check.ok
    assert "not found" in check.detail
    assert "--image-root" in check.fix


def test_image_root_rescues_a_relocated_dataset(tmp_path: Path):
    folder = tmp_path / "Scene_D_model0"
    folder.mkdir()
    elsewhere = tmp_path / "shared-images"
    elsewhere.mkdir()
    for i in range(3):
        (elsewhere / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")
    _views(20, old=12, root=Path("/old/machine/path")).to_parquet(folder / "views.parquet")

    assert not pf.check_datasets([folder])[0].ok
    rescued = pf.check_datasets([folder], image_root=elsewhere, min_old=10)[0]
    assert rescued.ok, rescued.detail


def test_image_root_rescues_absolute_paths_from_another_machine(tmp_path: Path):
    """The cluster writes absolute paths, so this is the real relocation case.

    Joining a root onto an absolute path is a no-op in pathlib, so the leading
    directories have to be stripped first or --image-root silently does nothing.
    """
    folder = tmp_path / "Scene_J_model0"
    folder.mkdir()
    elsewhere = tmp_path / "shared" / "images"
    elsewhere.mkdir(parents=True)
    for i in range(3):
        (elsewhere / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")
    _views(20, old=12, root=Path("/fs/class-projects/fall2026/gone/images")).to_parquet(
        folder / "views.parquet")

    assert not pf.check_datasets([folder], min_old=10)[0].ok
    rescued = pf.check_datasets([folder], image_root=elsewhere, min_old=10)[0]
    assert rescued.ok, rescued.detail


def test_re_rooting_keeps_the_parent_folder_apart(tmp_path: Path):
    """images/ and thumbnails/ hold the same file names; the tail must disambiguate."""
    folder = tmp_path / "Scene_K_model0"
    folder.mkdir()
    root = tmp_path / "shared"
    (root / "images").mkdir(parents=True)
    (root / "thumbnails").mkdir(parents=True)
    for i in range(3):
        (root / "images" / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff" + b"full")
        (root / "thumbnails" / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")
    _views(20, old=12, root=Path("/gone/images")).to_parquet(folder / "views.parquet")

    found = pf._resolve("/gone/images/img0.jpg", folder, root)
    assert found is not None and found.parent.name == "images"


def test_relative_paths_resolve_against_the_dataset_folder(tmp_path: Path):
    folder = tmp_path / "Scene_E_model0"
    (folder / "images").mkdir(parents=True)
    for i in range(3):
        (folder / "images" / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")
    _views(20, old=12).to_parquet(folder / "views.parquet")   # paths are images/imgN.jpg
    assert pf.check_datasets([folder], min_old=10)[0].ok


def test_too_few_old_photos_is_flagged_while_the_folders_still_pass(tmp_path: Path):
    folder = tmp_path / "Scene_F_model0"
    images = folder / "images"
    images.mkdir(parents=True)
    for i in range(20):
        (images / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")
    _views(20, old=0, root=images).to_parquet(folder / "views.parquet")

    checks = pf.check_datasets([folder])
    by_name = {c.name: c for c in checks}
    assert by_name["dataset Scene_F_model0"].ok
    assert not by_name["old photos"].ok
    assert "0 of 20" in by_name["old photos"].detail


def test_old_photo_count_is_summed_across_folders(tmp_path: Path):
    folders = []
    for n in ("G", "H"):
        folder = tmp_path / f"Scene_{n}_model0"
        images = folder / "images"
        images.mkdir(parents=True)
        for i in range(80):
            (images / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")
        _views(80, old=60, root=images).to_parquet(folder / "views.parquet")
        folders.append(folder)
    old = next(c for c in pf.check_datasets(folders) if c.name == "old photos")
    assert old.ok           # 60 + 60 clears the threshold that 60 alone would not
    assert "120 of 160" in old.detail


def test_one_broken_folder_does_not_hide_a_healthy_one(dataset: Path, tmp_path: Path):
    broken = tmp_path / "Scene_I_model0"
    broken.mkdir()
    checks = pf.check_datasets([dataset, broken])
    assert checks[0].ok and not checks[1].ok


# --------------------------------------------------------------------------- signatures

def test_only_declared_keywords_are_passed():
    def factory(views, *, resolution=64):
        return resolution, "with_appearance" in locals()

    # with_appearance would be a TypeError if it were forwarded blindly
    assert pf._call_with_supported(factory, [], resolution=576, with_appearance=True)[0] == 576


def test_kwargs_factories_receive_everything():
    def factory(views, **kw):
        return kw

    assert pf._call_with_supported(factory, [], resolution=576, with_appearance=True) == {
        "resolution": 576, "with_appearance": True}


# --------------------------------------------------------------------------- smoke train

def _fake_script(tmp_path: Path, *, flags: str, body: str = "print('loss: 0.1234')",
                 code: int = 0) -> Path:
    script = tmp_path / "train_fake.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        if "--help" in sys.argv:
            print("usage: train_fake.py {flags}")
            raise SystemExit(0)
        {body}
        raise SystemExit({code})
    """))
    return script


def test_flags_are_read_from_help(tmp_path: Path):
    script = _fake_script(tmp_path, flags="--max-steps N --dataset D --no-save")
    assert {"--max-steps", "--dataset", "--no-save"} <= pf.script_flags(script, sys.executable)


def test_missing_script_is_reported(tmp_path: Path):
    check = pf.check_smoke_train(tmp_path / "absent.py", [], python=sys.executable)
    assert not check.ok
    assert "not found" in check.detail


def test_a_script_that_cannot_even_print_help_is_reported(tmp_path: Path):
    script = tmp_path / "broken.py"
    script.write_text("import nonexistent_module_xyz\n")
    check = pf.check_smoke_train(script, [], python=sys.executable)
    assert not check.ok
    assert "--help" in check.detail


def test_a_script_without_a_step_limit_is_refused(tmp_path: Path):
    """Running without a step cap would start the full job, not smoke-test it."""
    script = _fake_script(tmp_path, flags="--dataset D")
    check = pf.check_smoke_train(script, [], python=sys.executable)
    assert not check.ok
    assert "--max-steps" in check.fix or "--max-steps" in check.detail


def test_successful_step_reports_the_loss(tmp_path: Path, dataset: Path):
    script = _fake_script(tmp_path, flags="--max-steps N --dataset D --no-save")
    check = pf.check_smoke_train(script, [dataset], python=sys.executable)
    assert check.ok, check.detail
    assert "0.1234" in check.detail


def test_failing_step_gives_back_a_runnable_command(tmp_path: Path, dataset: Path):
    script = _fake_script(tmp_path, flags="--max-steps N --dataset D",
                          body="import sys; print('CUDA out of memory', file=sys.stderr)", code=1)
    check = pf.check_smoke_train(script, [dataset], python=sys.executable)
    assert not check.ok
    assert "CUDA out of memory" in check.detail
    assert "--max-steps 1" in check.fix and str(dataset) in check.fix


def test_unsupported_flags_are_dropped_rather_than_crashing_the_run(tmp_path: Path, dataset: Path):
    """The caller's extra flags may target a newer script than the one on disk."""
    script = _fake_script(tmp_path, flags="--max-steps N --dataset D",
                          body="import sys; "
                               "raise SystemExit('unrecognized' if '--appearance-remove-era' "
                               "in sys.argv else 0)")
    check = pf.check_smoke_train(script, [dataset], extra=["--appearance-remove-era"],
                                 python=sys.executable)
    assert check.ok, check.detail


def test_supported_extra_flags_are_kept(tmp_path: Path, dataset: Path):
    script = _fake_script(tmp_path, flags="--max-steps N --dataset D --appearance-remove-era",
                          body="import sys; "
                               "print('ok' if '--appearance-remove-era' in sys.argv else 'MISSING')")
    check = pf.check_smoke_train(script, [dataset], extra=["--appearance-remove-era"],
                                 python=sys.executable)
    assert check.ok
    assert "MISSING" not in check.detail


def test_a_hanging_step_times_out_instead_of_hanging_the_preflight(tmp_path: Path):
    script = _fake_script(tmp_path, flags="--max-steps N", body="import time; time.sleep(30)")
    check = pf.check_smoke_train(script, [], python=sys.executable, timeout=2)
    assert not check.ok
    assert "no step finished" in check.detail


# --------------------------------------------------------------------------- end to end

def test_run_checks_passes_on_a_healthy_setup(dataset: Path, hub: Path, tmp_path: Path):
    """Nothing we can control should fail; the rest skips for want of torch/seva."""
    script = _fake_script(tmp_path, flags="--max-steps N --dataset D --no-save")
    report = pf.run_checks([dataset], hub=hub, train_script=script,
                           python=sys.executable, min_old=10)
    controllable = [c for c in report.failures
                    if not c.name.startswith(("import", "gpu", "sample"))]
    assert controllable == [], [(c.name, c.detail) for c in controllable]


def test_run_checks_collects_every_failure_rather_than_the_first(tmp_path: Path):
    report = pf.run_checks([tmp_path / "unbuilt"], hub=tmp_path / "no-hub",
                           train_script=tmp_path / "no-script", python=sys.executable)
    names = {c.name for c in report.failures}
    assert {"hub cache", "dataset unbuilt", "smoke train"} <= names


def test_no_smoke_skips_the_slow_check(dataset: Path, hub: Path):
    report = pf.run_checks([dataset], hub=hub, train_script=None, min_old=10)
    smoke = next(c for c in report.checks if c.name == "smoke train")
    assert smoke.skipped


@pytest.fixture
def fake_dataset_module(monkeypatch):
    """A stand-in for timemachine.data.dataset, so the load path can be exercised.

    The real one needs torch and the full package; the preflight only depends on its
    constructor accepting a frame and its items being indexable, which is exactly
    what this pins down.
    """
    import types

    module = types.ModuleType("timemachine.data.dataset")
    module._REQUIRED = list(pf._FALLBACK_VIEW_COLUMNS)

    class DatedViewsDataset:
        def __init__(self, views, resolution=576, with_appearance=False):
            missing = [c for c in module._REQUIRED if c not in views]
            if missing:
                raise ValueError(f"View table is missing columns: {missing}")
            self.views, self.with_appearance = views, with_appearance

        def __len__(self):
            return len(self.views)

        def __getitem__(self, i):
            keys = ["image", "date", "is_input", "c2w", "intrinsics"]
            return dict.fromkeys(keys + (["appearance"] if self.with_appearance else []), 0)

    module.DatedViewsDataset = DatedViewsDataset
    monkeypatch.setitem(sys.modules, "timemachine.data.dataset", module)
    return module


def test_sample_builds_on_a_healthy_dataset(dataset: Path, fake_dataset_module):
    check = pf.check_dataset_loads([dataset])
    assert check.ok, check.detail
    assert "20 samples" in check.detail


def test_sample_builds_concatenates_several_folders(dataset: Path, tmp_path: Path,
                                                    fake_dataset_module):
    """The trainer concatenates the folders, so the preflight must too - a column
    present in one table and absent in another only shows up after the concat."""
    second = tmp_path / "Scene_Z_model0"
    images = second / "images"
    images.mkdir(parents=True)
    for i in range(5):
        (images / f"img{i}.jpg").write_bytes(b"\xff\xd8\xff")
    _views(5, old=5, root=images).to_parquet(second / "views.parquet")

    check = pf.check_dataset_loads([dataset, second])
    assert check.ok, check.detail
    assert "25 samples" in check.detail


def test_sample_builds_accepts_a_parquet_path_directly(dataset: Path, fake_dataset_module):
    """--dataset takes a folder or a view table; the trainer supports both."""
    check = pf.check_dataset_loads([dataset / "views.parquet"])
    assert check.ok, check.detail


def test_sample_builds_reports_a_dataset_that_rejects_the_table(dataset: Path,
                                                                fake_dataset_module):
    """The real failure mode of the stale distortion columns."""
    _views(20, old=12, root=dataset / "images", drop="k1").to_parquet(
        dataset / "views.parquet")
    check = pf.check_dataset_loads([dataset])
    assert not check.ok
    assert "k1" in check.detail


def test_appearance_flag_reaches_the_dataset(dataset: Path, fake_dataset_module):
    assert "appearance" in pf.check_dataset_loads([dataset], appearance=True).detail
    assert "appearance" not in pf.check_dataset_loads([dataset], appearance=False).detail


# ------------------------------------------------------------------ drift and defects

def test_required_columns_match_the_dataset():
    """The preflight must require exactly what DatedViewsDataset requires.

    Two hand-maintained lists drift, and a preflight that demands a column nothing
    reads reports healthy datasets as broken - which is how 34 good datasets were
    once declared unusable.
    """
    try:
        from timemachine.data.dataset import _REQUIRED
    except Exception:                                  # noqa: BLE001
        pytest.skip("the dataset module is not importable here")
    assert set(pf.required_view_columns()) == set(_REQUIRED)


def test_fallback_columns_are_a_subset_of_the_real_ones():
    """The fallback is used when the repo cannot be imported, so it must not invent
    requirements the dataset does not have."""
    try:
        from timemachine.data.dataset import _REQUIRED
    except Exception:                                  # noqa: BLE001
        pytest.skip("the dataset module is not importable here")
    assert set(pf._FALLBACK_VIEW_COLUMNS) <= set(_REQUIRED)


def test_an_uninstalled_package_skips():
    exc = ModuleNotFoundError("No module named 'seva'", name="seva")
    check = pf.import_failure("sample builds", exc)
    assert check.skipped and check.ok
    assert "seva" in check.detail


def test_a_missing_name_in_an_installed_module_fails():
    """The bug this file shipped with: a function that did not exist, skipped away.

    ``pathlib`` is certainly installed, so an ImportError naming it cannot be an
    environment gap - it means the caller asked for something that is not there.
    """
    exc = ImportError("cannot import name 'load_views' from 'pathlib'", name="pathlib")
    check = pf.import_failure("sample builds", exc)
    assert not check.ok and not check.skipped
    assert "load_views" in check.detail


def test_a_nameless_import_error_fails_rather_than_hiding():
    check = pf.import_failure("sample builds", ImportError("something went wrong"))
    assert not check.ok
