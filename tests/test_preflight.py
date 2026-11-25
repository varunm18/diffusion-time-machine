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
    """A view table with the columns the real builder emits."""
    rows = []
    for i in range(n):
        year = 1900 + (i % 44) if i < old else 2015   # always < 1945
        rows.append({
            "scene_id": "Scene_A", "model": 0, "file_key": f"k{i}",
            "path": str((root / f"img{i}.jpg") if root else f"images/img{i}.jpg"),
            "title": f"img{i}.jpg",
            "camera_width": 800, "camera_height": 600,
            "fx": 700.0, "fy": 700.0, "cx": 400.0, "cy": 300.0, "k1": 0.0, "k2": 0.0,
            "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "tx": 0.0, "ty": 0.0, "tz": 1.0,
            "date_lo": float(year), "date_hi": float(year), "date_mid": float(year),
            "date_confidence": 1.0, "era": "old" if year < 1945 else "modern",
            "medium": "photo", "color_hint": "color", "pose_source": "colmap",
        })
    frame = pd.DataFrame(rows)
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


def test_an_uninstalled_repo_does_not_masquerade_as_a_dataset_failure(dataset: Path):
    """A missing package is an environment problem, and the import checks own it.

    Calling it a dataset failure sends someone rebuilding datasets that are fine.
    """
    check = pf.check_dataset_loads([dataset])
    assert check.ok
    if check.skipped and "cannot import" in check.detail:
        assert "timemachine" in check.detail or "seva" in check.detail
