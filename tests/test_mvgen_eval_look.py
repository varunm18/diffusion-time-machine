"""CPU tests for experiment look1's evaluation: its sets, read-outs R1-R6, the state read-out and the scripts.

Covered: ``timemachine.mvgen.evaluation`` (its "Experiment look1" part), ``timemachine.mvgen.readout``,
``scripts/mvgen/make_eval_sets.py``, ``scripts/mvgen/fit_readout.py`` (its helpers), ``scripts/mvgen/eval_dates.py
--eval-sets`` and ``scripts/mvgen/render.py --look/--tone``. Nothing of the released model is loaded. The stand-ins are
those of ``tests/test_mvgen_trainer.py`` (its tiny UNet with the real date conditioner, its sampling pipeline, its
folder of 24 tiny views: rows 0-7 old, 8-15 modern, 16-23 after the fire), plus, defined here:

* a changed-region file for that folder (the top-left quarter of every view) and a look-code file and class maps;
* ``TinyLookMVGen``: the tiny model with a look, whose ``dates`` takes per-view looks as ``MVGen.dates`` does and
  records them (the look embedding itself belongs to the training side and has its own tests);
* ``FakeReadout`` for ``readout.StateReadout``: the read-out of a picture is its mean red level in the region.

One test runs the real DINOv2 weights on the appearance study's stored crops where the cache and the study's files
are present. Run from MVGenMaster's checkout, in its environment (see ``timemachine/mvgen/__init__.py``).
"""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("my_diffusers", reason="runs in MVGenMaster's environment, from its checkout")

from tests.test_mvgen_trainer import (  # noqa: E402, F401
    EDGES, N, SIZE, TinyMVGen, adapted_model, config, fake_conditioning, folder, load_script, tiny_model,
)

from timemachine.data.holdout import HoldoutConfig  # noqa: E402
from timemachine.mvgen import evaluation as ev  # noqa: E402
from timemachine.mvgen import look as looks  # noqa: E402
from timemachine.mvgen import readout  # noqa: E402
from timemachine.mvgen.data import MVGenScene, ViewSamplingConfig, write_region_file  # noqa: E402
from timemachine.mvgen.probes import RenderProbe  # noqa: E402
from timemachine.mvgen.step import LossConfig  # noqa: E402
from timemachine.mvgen.trainer import Trainer  # noqa: E402

FIRE = EDGES[-1]
SMALL = ViewSamplingConfig(n_ref=2, n_targets=3, min_targets_seen=2, top_k=4)      # view sets of 2 + 3 views
WITHIN = {"modern": [2013.4, 2016.5], "after": [2019.45, 2021.5]}
STUDY = Path(__file__).resolve().parents[1] / "data/appearance_study/design_learned"


# ----------------------------------------------------------------------------- stand-ins
def write_regions(root: Path) -> None:
    """The top-left quarter of every view is its changed region (whole latent cells of the 32 px crops)."""
    keys = pd.read_parquet(root / "views.parquet")["file_key"].tolist()
    cells = np.zeros((N, SIZE // 8, SIZE // 8), dtype=np.uint8)
    cells[:, :SIZE // 16, :SIZE // 16] = 64
    write_region_file(root, cells, keys, np.full(N, 0.25), resolution=SIZE, min_share=0.02, edge=FIRE, config={})


def write_looks(root: Path) -> dict:
    """A look-code file and class maps for the folder: each view's code from its own (plain) image."""
    views = pd.read_parquet(root / "views.parquet")
    scene = MVGenScene(root, resolution=SIZE, require_depth=False)
    labels = np.ones((N, 4, 4), dtype=np.uint8)              # ADE20K "building" ...
    labels[:, 0] = 2                                         # ... under a row of "sky"
    groups = looks.class_groups(labels)
    region = np.zeros((N, SIZE // 8, SIZE // 8), dtype=bool)
    region[:, :3, :3] = True                                 # the stored quarter grown by one cell
    codes, shown = zip(*(looks.look_code(looks.to_image(scene.make_sample([row], 1)["images"][0]), groups[row],
                                         looks.excluded_pixels(region[row], SIZE), tone="colour") for row in range(N)))
    codes, shown = np.stack(codes), np.stack(shown)
    stats = looks.LookStats.from_codes(codes)
    np.savez(root / looks.CLASS_MAP_FILE, labels=labels, file_keys=views["file_key"].to_numpy(dtype=str),
             model="test", revision="test")
    looks.write_look_file(root, {"codes": codes, "shown": shown, "tone": np.array(["colour"] * N),
                                 "tone_source": np.array(["labels"] * N), "computed": np.ones(N, bool), "region": region,
                                 "toned_curves": np.zeros((0, 256, 3), np.float32), "toned_rows": np.zeros(0, np.int64),
                                 "stats_mean": np.array(stats.mean), "stats_sd": np.array(stats.sd),
                                 "file_keys": views["file_key"].to_numpy(dtype=str)}, {"test": True})
    return looks.read_look(root)


class TinyLookMVGen(TinyMVGen):
    """The tiny model with a look: ``dates`` takes per-view looks as ``MVGen.dates`` does and records every call."""

    look_stats = None

    def dates(self, year, valid=None, uncond="same", look=None, look_known=None):
        self.seen_looks = [*getattr(self, "seen_looks", []),
                           (year.clone(), None if look is None else torch.as_tensor(look).clone(), look_known)]
        return super().dates(year, valid, uncond)


def look_model(stats) -> TinyLookMVGen:
    base = adapted_model()
    model = TinyLookMVGen(unet=base.unet, vae=base.vae, config=base.config, scheduler=base.scheduler, date=base.date)
    model.look_stats = stats
    return model


class FakeReadout:
    """Stand-in for ``readout.StateReadout``: a picture's logit is its mean red level in the region, centred."""

    fitted_on = frozenset()

    def readout_logit(self, images, regions):
        return np.array([(image[..., 0][region].mean() - 128) / 16 for image, region in zip(images, regions)])

    def check_unseen(self, file_keys):
        pass


@pytest.fixture
def regions_folder(folder) -> Path:
    write_regions(folder)
    return folder


@pytest.fixture
def run(regions_folder, tmp_path) -> Trainer:
    """A finished three-step run of the tiny trainer with a hold-out and a checkpoint per step."""
    cfg = config(regions_folder, tmp_path / "run", sampling=SMALL, ema_decay=0.9, save_every=1, latest_every=1,
                 loss=LossConfig(precision="fp32"), holdout=HoldoutConfig(fraction=0.5, seed=1, era_edges=EDGES))
    trainer = Trainer(cfg, model=tiny_model())
    trainer.train()
    return trainer


def tiny_plan(scene: MVGenScene, path: Path) -> Path:
    """An evaluation-sets file for the scene's hold-out: one set per cell, two scored targets and one training target
    each, every target its own unit, donors from views outside the set."""
    held, keys = np.flatnonzero(scene.heldout), scene.views["file_key"].tolist()
    period = {name: p for p, name in enumerate(ev.LOOK1_PERIODS)}
    cells = {}
    for i, (ref_name, target_name) in enumerate([("modern", "modern"), ("after", "modern"), ("modern", "after"),
                                                 ("after", "after")]):
        held_targets = [r for r in held if scene.period[r] == period[target_name] and scene.usable[r]][:2]
        training = [r for r in range(N) if not scene.heldout[r] and scene.period[r] == period[target_name]]
        refs = [r for r in range(N) if not scene.heldout[r] and scene.period[r] == period[ref_name]
                and r not in training[:1]][:2]
        targets = [*held_targets, training[0]] if ref_name != target_name else [*held_targets, training[-1]]
        rows = [int(r) for r in (*refs, *targets)]
        targets = rows[2:]
        others = [r for r in range(N) if r not in rows]
        other_period = 2 if target_name == "modern" else 1
        donors = {"other": [r for r in others if scene.period[r] == other_period][:3],
                  "same": [r for r in others if scene.period[r] == period[target_name]][-3:]}
        cells[f"{ref_name}->{target_name}"] = {"reference_period": ref_name, "target_period": target_name, "sets": [{
            "name": f"{ref_name[:3]}-{target_name[:3]}/0", "n_ref": 2, "rows": rows, "file_keys": [keys[r] for r in rows],
            "anchor": rows[2], "scored": [True, True, False], "units": [100 + t for t in targets], "strict": False,
            "donors": {**donors, "other_file_keys": [keys[r] for r in donors["other"]],
                       "same_file_keys": [keys[r] for r in donors["same"]]}}], "strict_sets": []}
    content = {"holdout": {"seed": 7, "rows": held.tolist(), "file_keys": [keys[r] for r in held]},
               "within_years": WITHIN, "cells": cells,
               "yardsticks": {"look_gap": {"sd": [1.0] * 6, "unit": [1.0, 0, 0, 0, 0, 0]},
                              "chroma": {name: {"mean": mean, "se": 0.5, "photos": 10, "units": 8}
                                         for name, mean in (("grey", 0.0), ("toned", 12.0), ("colour", 13.0))}}}
    path.write_text(json.dumps(content))
    return path


# ----------------------------------------------------------------------------- pixel measures
def test_look_measures_of_plain_pictures():
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    image[:] = (200, 100, 50)
    measures = dict(zip(ev.LOOK_MEASURES, ev.look_measures(image, np.ones((30, 30), bool))))
    assert measures["brightness"] == pytest.approx((0.299 * 200 + 0.587 * 100 + 0.114 * 50) / 255)
    assert measures["contrast"] == pytest.approx(0.0, abs=1e-12)
    assert measures["saturation"] == pytest.approx(150 / 200)
    assert measures["warmth"] == pytest.approx(150 / 255) and measures["sky_blueness"] == pytest.approx(-150 / 255)
    assert measures["ground_green_share"] == 0.0                         # hue 20 degrees: not green
    green = np.zeros_like(image)
    green[:] = (40, 160, 60)
    keep = np.ones((30, 30), bool)
    keep[:12] = False                                                    # no pixel of the top third is kept
    measured = ev.look_measures(green, keep)
    assert np.isnan(measured[4]) and measured[5] == 1.0
    assert np.isnan(ev.look_measures(image, np.zeros((30, 30), bool))).all()


def test_hsv_and_cielab_agree_with_opencv():
    cv2 = pytest.importorskip("cv2")
    rgb = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    hue, saturation, value = ev.hsv(rgb.astype(np.float64))
    reference = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    np.testing.assert_allclose(saturation, reference[..., 1], atol=1e-5)
    np.testing.assert_allclose(value, reference[..., 2], atol=1e-6)
    np.testing.assert_allclose(hue, reference[..., 0], atol=1e-2)
    np.testing.assert_allclose(ev.srgb_to_lab(rgb.astype(np.float64)), cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB), atol=0.5)
    assert ev.srgb_to_lab(np.array([1.0, 0.0, 0.0])) == pytest.approx([53.24, 80.09, 67.20], abs=0.02)


def test_chroma_is_zero_for_grey_pictures():
    rng = np.random.default_rng(1)
    grey = np.repeat(rng.integers(0, 256, (20, 20, 1), dtype=np.uint8), 3, axis=-1)
    keep = np.ones((20, 20), bool)
    assert ev.mean_chroma(grey, keep) == pytest.approx(0.0, abs=1e-3)
    red = np.zeros((20, 20, 3))
    red[..., 0] = 1.0
    assert ev.mean_chroma(red, keep) == pytest.approx(np.hypot(80.09, 67.20), abs=0.05)


def test_colour_matching_moves_each_channels_mean_and_spread_to_the_photos():
    rng = np.random.default_rng(2)
    render, photo = rng.uniform(0.3, 0.7, (24, 24, 3)), rng.uniform(0.35, 0.6, (24, 24, 3))
    keep = np.ones((24, 24), bool)
    keep[:4] = False
    matched = ev.match_moments(render, photo, keep)
    np.testing.assert_allclose(matched[keep].mean(0), photo[keep].mean(0), atol=1e-9)
    np.testing.assert_allclose(matched[keep].std(0), photo[keep].std(0), atol=1e-9)
    assert ev.psnr_over(photo, photo) == pytest.approx(100.0)            # mse clamped at 1e-10
    assert ev.psnr_over(render, photo, keep) == pytest.approx(ev.to_psnr(((render - photo)[keep] ** 2).mean()))


# ----------------------------------------------------------------------------- statistics over units
def test_clustered_mean_is_the_ordinary_error_with_one_target_per_unit_and_larger_within_units():
    values = {f"t{i}": v for i, v in enumerate([1.0, 2.0, 4.0, 7.0])}
    alone = ev.clustered_mean(values, {key: i for i, key in enumerate(values)})
    assert alone["mean"] == pytest.approx(3.5) and (alone["targets"], alone["units"]) == (4, 4)
    assert alone["se"] == pytest.approx(np.std(list(values.values()), ddof=1) / 2)
    paired = ev.clustered_mean(values, {"t0": 0, "t1": 0, "t2": 1, "t3": 1})
    assert paired["units"] == 2 and paired["se"] > alone["se"]
    assert ev.clustered_mean({"a": 1.0}, {"a": 0})["se"] is None
    assert ev.clustered_mean({"a": float("nan")}, {"a": 0}) == {"mean": None, "se": None, "targets": 0, "units": 0}


def test_clustered_slope_and_the_bootstrap_bound_of_a_ratio():
    x = np.array([1.0, -2.0, 0.5, 3.0])
    slope, se = ev.clustered_slope(x, 0.5 * x, np.array([0, 0, 1, 2]))
    assert slope == pytest.approx(0.5) and se == pytest.approx(0.0, abs=1e-12)
    noisy, _ = ev.clustered_slope(x, 0.5 * x + np.array([0.1, -0.1, 0.2, 0.0]), np.array([0, 1, 2, 3]))
    assert noisy == pytest.approx((x * (0.5 * x + np.array([0.1, -0.1, 0.2, 0.0]))).sum() / (x * x).sum())
    numerator = {f"k{i}": v for i, v in enumerate([0.2, 0.5, 0.1, 0.4])}
    denominator = {key: 4.0 for key in numerator}
    units = {key: i for i, key in enumerate(numerator)}
    bound = ev.bootstrap_ratio_bound(numerator, denominator, units)
    assert bound == ev.bootstrap_ratio_bound(numerator, denominator, units)          # seeded
    assert 0.3 / 4 < bound <= 0.5 / 4


# ----------------------------------------------------------------------------- conditions, dates, noise
def names(conditions):
    return [(c.name, c.primary) for c in conditions]


def test_conditions_follow_the_readouts_and_what_the_model_has():
    everything = ev.LOOK1_READOUTS
    assert names(ev.look1_conditions(everything, False, False, "modern")) == [("none/true", True)]
    assert names(ev.look1_conditions(everything, True, False, "after")) == [
        ("none/true", True), ("none/swapped", True), ("none/within", True)]
    look = names(ev.look1_conditions(everything, True, True, "modern"))
    assert look[:3] == [("references/true", True), ("references/swapped", True), ("references/within", True)]
    assert set(look[3:]) == {("references/old", False), ("neutral/true", False), ("neutral/swapped", False),
                             ("own/true", False), ("other/true", False), ("same/true", False), ("grey/true", False)}
    assert ("references/old", False) not in names(ev.look1_conditions(everything, True, True, "after"))
    gate = names(ev.look1_conditions(["R1", "R3"], True, True, "after"))
    assert set(gate) == {("references/true", True), ("references/swapped", True), ("references/within", True),
                         ("neutral/true", False), ("neutral/swapped", False), ("other/true", False),
                         ("same/true", False), ("grey/true", False)}
    assert names(ev.look1_conditions(["R6"], True, False, "after")) == [("none/true", True)]


def test_every_set_gets_its_own_noise_and_every_condition_of_a_set_the_same():
    assert ev.render_seed((1, 2, 3), 0) == ev.render_seed((1, 2, 3), 0)
    assert len({ev.render_seed((1, 2, 3), k) for k in range(4)} | {ev.render_seed((1, 2, 4), 0)}) == 5


def test_within_period_and_old_dates(regions_folder, tmp_path):
    scene = MVGenScene(regions_folder, SMALL, resolution=SIZE, require_depth=False)
    probe = RenderProbe(scene, (8, 9, 10, 16, 23), 2, "set", ("true", "swapped"), (0,), LossConfig(), (1890.0, 2015.0,
                        2021.0))
    view_set = SimpleNamespace(target_period="after")
    plan = SimpleNamespace(within_years=WITHIN)
    year, valid = ev.look1_years("within", probe, view_set, plan, (1890.0, 2015.0, 2021.0))
    assert year[:2].tolist() == probe.sample["date_mid"][:2].tolist() and bool(valid.all())
    assert year[2:].tolist() == pytest.approx([2021.5, 2021.5, 2019.45])          # 2012.2 is not after; 2022.3 -> 2019.45
    year, _ = ev.look1_years("old", probe, view_set, plan, (1890.0, 2015.0, 2021.0))
    assert year[2:].tolist() == [1890.0] * 3
    assert torch.equal(ev.look1_years("swapped", probe, view_set, plan, ())[0], probe.view_dates("swapped")[0])
    with pytest.raises(ValueError, match="look1 date"):
        ev.look1_years("tomorrow", probe, view_set, plan, ())


# ----------------------------------------------------------------------------- the evaluation-sets file
def test_eval_sets_are_read_by_file_key_and_checked_against_the_runs_hold_out(run, tmp_path):
    scene = run.scene
    path = tiny_plan(scene, tmp_path / "sets.json")
    plan = ev.read_eval_sets(path, scene)
    assert [s.cell for s in plan.sets] == ["modern->modern", "after->modern", "modern->after", "after->after"]
    assert plan.seed == 7 and plan.within_years["modern"] == (2013.4, 2016.5)
    first = plan.sets[0]
    assert first.n_ref == 2 and first.scored == (True, True, False) and first.sign == 1 and plan.sets[2].sign == -1
    assert len(plan.scored_targets()) == 8 and first.folder == "mod-mod_0"
    ev.check_eval_sets(plan, scene)
    assert [s.cell for s in ev.read_eval_sets(path, scene, cells=["modern->after"]).sets] == ["modern->after"]
    with pytest.raises(ValueError, match="no cells"):
        ev.read_eval_sets(path, scene, cells=["old->old"])

    content = json.loads(path.read_text())
    content["cells"]["modern->modern"]["sets"][0]["file_keys"][0] = "another.png"
    (tmp_path / "renamed.json").write_text(json.dumps(content))
    with pytest.raises(ValueError, match="another table"):
        ev.read_eval_sets(tmp_path / "renamed.json", scene)
    other = MVGenScene(run.cfg.dataset, SMALL, resolution=SIZE, require_depth=False)
    other.hold_out(np.flatnonzero(scene.heldout)[1:])
    with pytest.raises(ValueError, match="hold-out of"):
        ev.check_eval_sets(plan, other)
    unscored = replace(first, scored=(True, True, True))                  # a training view counted as scored
    with pytest.raises(ValueError, match="not held out"):
        ev.check_eval_sets(replace(plan, sets=(unscored,)), scene)


def test_stored_masks_are_the_latent_cells_with_a_masked_pixel(regions_folder):
    scene = MVGenScene(regions_folder, resolution=SIZE, require_depth=False)
    masks = ev.stored_masks(scene, [3, 17])
    assert masks.shape == (2, SIZE, SIZE) and masks[:, :SIZE // 2, :SIZE // 2].all()
    assert masks.sum() == 2 * (SIZE // 2) ** 2


# ----------------------------------------------------------------------------- looks
def test_looks_give_references_their_own_codes_and_targets_the_conditions(regions_folder):
    look = write_looks(regions_folder)
    scene = MVGenScene(regions_folder, resolution=SIZE, require_depth=False)
    stats = look["stats"]
    inputs = ev.LookInputs(look=look, stats=stats, class_maps=looks.read_class_maps(regions_folder))
    rows = (8, 9, 10, 16, 17)
    sample = scene.make_sample(np.asarray(rows), 2)
    donors = {"other": [20, 21, 22], "same": [11, 12, 13]}

    def targets_of(kind):
        raw, look_in, known = ev.view_looks(kind, rows, 2, sample, inputs, donors)
        assert bool(known.all()) and look_in.shape == (5, looks.LOOK_DIM)
        np.testing.assert_allclose(look_in[:2].numpy(), stats.standardise(look["codes"][[8, 9]]), atol=1e-6)
        return raw, look_in[2:].numpy()

    raw, standardised = targets_of("references")
    np.testing.assert_allclose(raw, np.repeat(looks.references_code(look["codes"][[8, 9]], look["shown"][[8, 9]])[None],
                                              3, 0))
    assert np.abs(targets_of("neutral")[1]).max() < 1e-5                # the population mean
    np.testing.assert_allclose(targets_of("own")[0], look["codes"][[10, 16, 17]])
    np.testing.assert_allclose(targets_of("other")[0], look["codes"][[20, 21, 22]])
    np.testing.assert_allclose(targets_of("same")[0], look["codes"][[11, 12, 13]])
    grey, _ = targets_of("grey")
    assert np.allclose(grey[:, 0], grey[:, 1]) and np.allclose(grey[:, 1], grey[:, 2])     # one grey level
    assert (grey[:, looks.TONE] == [1, 0, 0]).all()
    toned = ev.view_looks("own", rows, 2, sample, replace(inputs, tone="toned"))[0]
    assert (toned[:, looks.TONE] == [0, 1, 0]).all() and np.allclose(toned[:, :21], look["codes"][[10, 16, 17], :21],
                                                                     equal_nan=True)
    assert ev.view_looks("none", rows, 2, sample, None) == (None, None, None)
    with pytest.raises(ValueError, match="one donor row per target"):
        ev.view_looks("donor", rows, 2, sample, inputs, {"donor": [1]})


def test_looks_reach_every_dated_pass_through_the_look_arguments(run, tmp_path):
    """Renders and losses hand a condition's looks to the model as ``render_views`` and the step's loss functions
    take them (their ``look`` and ``look_known`` arguments, on to ``MVGen.dates``)."""
    scene, years = run.scene, (1890.0, 2015.0, 2021.0)
    look = write_looks(Path(run.cfg.dataset))
    plan = ev.read_eval_sets(tiny_plan(scene, tmp_path / "sets.json"), scene)
    view_set = plan.sets[0]
    model = look_model(look["stats"])
    inputs = ev.LookInputs(look=look, stats=look["stats"])
    probe = RenderProbe(scene, view_set.rows, 2, view_set.name, ("true", "swapped"), (0,), LossConfig(precision="fp32"),
                        years, steps=1)
    masks = ev.stored_masks(scene, view_set.targets)
    _, expected, known = ev.view_looks("references", view_set.rows, 2, probe.sample, inputs)
    ev.render_look1_set(model, model.sampling_pipeline(), probe, view_set, plan, [ev.Condition("references", "true", True)],
                        {True: 1, False: 1}, years, masks, inputs)
    ev.measure_look1_losses(model, probe, view_set, plan, masks, [500], [0], years, inputs)
    assert len(model.seen_looks) == 1 + 4                               # one render, four loss dates
    for _, given, given_known in model.seen_looks:
        assert torch.equal(given, expected) and torch.equal(torch.as_tensor(given_known), known)


# ----------------------------------------------------------------------------- rendering a set
def test_a_set_is_rendered_under_every_condition_and_its_scored_targets_scored(run, tmp_path):
    scene = run.scene
    look = write_looks(Path(run.cfg.dataset))
    plan = ev.read_eval_sets(tiny_plan(scene, tmp_path / "sets.json"), scene)
    view_set = plan.sets[0]                                             # modern targets from modern references
    model = look_model(look["stats"])
    inputs = ev.LookInputs(look=look, stats=look["stats"], class_maps=looks.read_class_maps(run.cfg.dataset))
    probe = RenderProbe(scene, view_set.rows, 2, view_set.name, ("true", "swapped"), (0,), LossConfig(precision="fp32"),
                        (1890.0, 2015.0, 2021.0), steps=2)
    conditions = ev.look1_conditions(ev.LOOK1_READOUTS, True, True, "modern")
    masks = ev.stored_masks(scene, view_set.targets)
    records, pairs = ev.render_look1_set(model, model.sampling_pipeline(), probe, view_set, plan, conditions,
                                         {True: 2, False: 1}, (1890.0, 2015.0, 2021.0), masks, inputs, FakeReadout(),
                                         tmp_path / "renders")
    renders = sum(2 if c.primary else 1 for c in conditions)
    assert len(records) == 2 * renders and len(model.seen_looks) == renders      # one dated block per render
    assert {r["row"] for r in records} == set(view_set.targets[:2])
    by = {(r["condition"], r["seed"], r["row"]): r for r in records}
    assert by["references/true", 0, view_set.targets[0]]["requested"] == pytest.approx(
        looks.references_code(look["codes"][list(view_set.rows[:2])], look["shown"][list(view_set.rows[:2])])[:3].tolist())
    assert by["other/true", 0, view_set.targets[1]]["requested"] == pytest.approx(
        look["codes"][view_set.donors["other"][1], :3].tolist())
    first = by["references/true", 0, view_set.targets[0]]
    assert first["logit"] is not None and first["tone"] in looks.TONES and len(first["measures"]) == 6
    assert first["unit"] == view_set.units[0] and first["cell"] == "modern->modern"
    # the dates the model was given: each condition's, the references at their own
    def rounded(years):
        return tuple(round(float(year), 2) for year in years)

    seen = {rounded(year) for year, _, _ in model.seen_looks}
    own = probe.sample["date_mid"].tolist()
    assert rounded(own) in seen and rounded([*own[:2], 1890.0, 1890.0, 1890.0]) in seen
    assert rounded([*own[:2], 2016.5, 2016.5, 2016.5]) in seen
    # the renders and photos of the scored targets are kept, of the unscored target none
    kept = sorted(path.name for path in (tmp_path / "renders").iterdir())
    assert "real_view2.jpg" in kept and "mask_view3.png" in kept and "references_swapped_seed1_view3.jpg" in kept
    assert not any(name.endswith("view4.jpg") for name in kept)
    assert {(p["swap"], p["seed"]) for p in pairs} == {("swapped", 0), ("swapped", 1), ("within", 0), ("within", 1)}
    assert all(0 <= p["difference_outside"] <= 1 for p in pairs)

    no_look = tiny_model()
    plain, _ = ev.render_look1_set(no_look, no_look.sampling_pipeline(), probe, view_set, plan,
                                   ev.look1_conditions(["R2"], True, False, "modern"), {True: 1, False: 1},
                                   (1890.0, 2015.0, 2021.0), masks, None, FakeReadout(), None)
    assert len(plain) == 3 * 2 and all(r["requested"] is None for r in plain)


def test_losses_are_measured_at_the_base_look_for_four_dates(run, tmp_path):
    scene = run.scene
    look = write_looks(Path(run.cfg.dataset))
    plan = ev.read_eval_sets(tiny_plan(scene, tmp_path / "sets.json"), scene)
    view_set = plan.sets[2]
    model = look_model(look["stats"])
    probe = RenderProbe(scene, view_set.rows, 2, view_set.name, ("true", "swapped"), (0,), LossConfig(precision="fp32"),
                        (1890.0, 2015.0, 2021.0))
    records = ev.measure_look1_losses(model, probe, view_set, plan, ev.stored_masks(scene, view_set.targets), [200, 800],
                                      [0], (1890.0, 2015.0, 2021.0), ev.LookInputs(look=look, stats=look["stats"]))
    assert [(r["variant"], r["timestep"]) for r in records] == [
        (v, t) for t in (200, 800) for v in ("true", "swapped", "within", "unknown")]
    assert all(given is not None for _, given, _ in model.seen_looks)          # every pass had the references' code
    gaps = ev.r2_loss_gaps(records, plan)
    assert set(gaps) == {f"{view_set.name}|{row}" for row in view_set.targets[:2]}


# ----------------------------------------------------------------------------- read-outs from records
def synthetic_plan(n_targets: int = 4) -> ev.Look1Plan:
    """Two sets of two scored targets, modern and after-fire, every target its own unit."""
    sets = tuple(ev.Look1Set(name=f"{cell}/0", cell=cell, reference_period="modern", target_period=period,
                             rows=(0, 1, 10 + 2 * i, 11 + 2 * i), n_ref=2, scored=(True, True),
                             units=(10 + 2 * i, 11 + 2 * i), donors={"other": (90, 91), "same": (92, 93)}, strict=False)
                 for i, (cell, period) in enumerate([("modern->modern", "modern"), ("modern->after", "after")]))
    yardsticks = {"look_gap": {"sd": [1.0] * 6, "unit": [0.6, 0.8, 0, 0, 0, 0]},
                  "chroma": {"grey": {"mean": 0.5, "se": 0.2, "photos": 30, "units": 20},
                             "toned": {"mean": 12.0, "se": 1.0, "photos": 30, "units": 20},
                             "colour": {"mean": 13.0, "se": 1.0, "photos": 30, "units": 20}}}
    return ev.Look1Plan(path=Path("sets.json"), sets=sets, heldout=(), seed=0, within_years=WITHIN,
                        yardsticks=yardsticks)


def record(view_set, row, condition, seed, **values):
    base = {"set": view_set.name, "cell": view_set.cell, "condition": condition, "seed": seed, "row": row,
            "means": [0.5, 0.5, 0.5], "requested": [0.5, 0.5, 0.5], "measures": [0.0] * 6, "chroma": 10.0,
            "tone": "colour", "logit": 0.0, "psnr_whole": 12.0, "psnr_outside": 12.5, "psnr_matched_whole": 13.0,
            "psnr_matched_outside": 13.5}
    return {**base, **values}


def test_steering_is_the_slope_of_render_means_on_requested_means():
    plan = synthetic_plan()
    records = []
    for view_set in plan.sets:
        for row in view_set.targets:
            for seed in (0, 1):
                records.append(record(view_set, row, "references/true", seed))
                for donor, shift in (("other", 0.1 + 0.01 * row), ("same", -0.05 - 0.01 * seed)):
                    requested = [0.5 + shift, 0.5 - shift, 0.5 + 2 * shift]
                    records.append(record(view_set, row, f"{donor}/true", seed, requested=requested,
                                          means=[0.5 + 0.6 * (r - 0.5) for r in requested]))
                records.append(record(view_set, row, "grey/true", seed, chroma=0.8))
    r1 = ev.summarise_r1(ev.Look1Records(records, plan), plan.yardsticks)
    assert r1["points"] == 16 and r1["slope"] == pytest.approx([0.6] * 3) and r1["gate"]
    assert r1["grey_chroma"]["all"]["mean"] == pytest.approx(0.8) and r1["grey_flag"]       # 0.3 < 2 x 0.2
    lines = ev._r1_lines(r1, plan.yardsticks)
    assert "PASS" in lines[0] and "grey flag" in lines[1]
    weak = [dict(r, means=[0.5 + 0.2 * (v - 0.5) for v in r["requested"]]) if r["condition"] != "references/true"
            else r for r in records]
    r1 = ev.summarise_r1(ev.Look1Records(weak, plan), plan.yardsticks)
    assert r1["slope"] == pytest.approx([0.2] * 3) and not r1["gate"]
    assert "FAIL" in ev._r1_lines(r1, plan.yardsticks)[0] and "film" in ev._r1_lines(r1, plan.yardsticks)[0]


def state_records(plan, flip_modern=8.0, flip_after=6.0, within=0.5, leak=0.4):
    """Read-outs of renders: modern targets -5 at the true date, after-fire +5; the cross-fire swap moves them towards
    the other period, the within-period swap a little, the other-period donor by ``leak``."""
    records = []
    for view_set in plan.sets:
        sign = view_set.sign
        flip = flip_modern if sign == 1 else flip_after
        for row in view_set.targets:
            for seed in (0, 1):
                true = -5.0 * sign
                for condition, logit in (("references/true", true), ("references/swapped", true + sign * flip),
                                         ("references/within", true + sign * within), ("same/true", true),
                                         ("other/true", true + sign * leak)):
                    records.append(record(view_set, row, condition, seed, logit=logit))
    return records


def test_the_state_readouts_flip_and_leak():
    plan = synthetic_plan()
    recs = ev.Look1Records(state_records(plan), plan)
    r2 = ev.summarise_r2(recs, "references", [])
    assert r2["flip"]["modern"]["mean"] == pytest.approx(7.5) and r2["flip"]["after"]["mean"] == pytest.approx(5.5)
    assert r2["flip"]["all"]["mean"] == pytest.approx(6.5) and r2["within"]["all"]["mean"] == pytest.approx(0.5)
    assert r2["flip"]["modern->modern"]["targets"] == 2 and r2["flip"]["all"]["units"] == 4
    r4 = ev.summarise_r4(recs, "references")
    assert r4["leak"]["modern"]["mean"] == pytest.approx(0.4) and r4["share_modern"] == pytest.approx(0.4 / 8.0)
    assert r4["pass"] and "PASS" in ev._r4_lines(r4)[0]
    leaky = ev.summarise_r4(ev.Look1Records(state_records(plan, leak=3.0), plan), "references")
    assert not leaky["pass"] and leaky["share_modern"] == pytest.approx(3.0 / 8.0)


def test_the_look_readout_projects_the_swap_on_the_real_gap():
    plan = synthetic_plan()
    records = []
    for view_set in plan.sets:
        for row in view_set.targets:
            for seed in (0, 1):
                shift = np.array([0.3, 0.4, 0.0, 0.0, 0.0, 0.0]) * view_set.sign      # towards "after" for both
                records += [record(view_set, row, "references/true", seed),
                            record(view_set, row, "references/swapped", seed, measures=shift.tolist()),
                            record(view_set, row, "references/within", seed,
                                   measures=[0.1 * (-1) ** seed, None, 0, 0, 0, None]),
                            record(view_set, row, "neutral/true", seed),
                            record(view_set, row, "neutral/swapped", seed, measures=(shift / 2).tolist())]
    pairs = [{"set": s.name, "row": row, "seed": 0, "swap": "swapped", "difference_outside": 0.05,
              "difference_inside": 0.2} for s in plan.sets for row in s.targets]
    r3 = ev.summarise_r3(ev.Look1Records(records, plan), "references", pairs, plan.yardsticks)
    assert r3["cross"]["all"]["mean"] == pytest.approx(0.5) and r3["neutral"]["all"]["mean"] == pytest.approx(0.25)
    assert r3["within"]["all"]["mean"] == pytest.approx(0.0, abs=1e-12)              # seeds cancel; NaN counts 0
    assert r3["absolute_outside"]["all"]["mean"] == pytest.approx(0.05)
    assert "R3 null" in ev._r3_lines(r3)[1]


def test_medium_and_quality_readouts():
    plan = synthetic_plan()
    records = []
    for view_set in plan.sets:
        for row in view_set.targets:
            records += [record(view_set, row, "references/old", 0, chroma=9.0, tone="colour"),
                        record(view_set, row, "grey/true", 0, chroma=1.0, tone="grey"),
                        record(view_set, row, "references/true", 0), record(view_set, row, "own/true", 0, psnr_whole=13.0)]
    recs = ev.Look1Records(records, plan)
    r5 = ev.summarise_r5(recs, "references")
    assert r5["old_date_colour_code"]["chroma"]["mean"] == pytest.approx(9.0)
    assert r5["grey_code"]["tone_shares"] == {"grey": 1.0, "toned": 0.0, "colour": 0.0}
    assert set(r5["per_target"]["grey_code"]) == {f"modern->modern/0|{row}" for row in plan.sets[0].targets}
    r6 = ev.summarise_r6(recs, "references")
    assert r6["base_whole"]["all"]["mean"] == pytest.approx(12.0) and r6["own_whole"]["all"]["mean"] == pytest.approx(13)
    assert r6["matched_outside"]["all"]["mean"] == pytest.approx(13.5)
    summary = ev.summarise_look1(records, [], [], plan, ["R5", "R6"], True, True)
    lines = ev.look1_lines(summary, plan.yardsticks)
    assert any(line.startswith("R5 modern targets at an old date") for line in lines)
    assert any(line.startswith("R6 PSNR with the own code") for line in lines)
    released = ev.summarise_look1(records, [], [], plan, ev.LOOK1_READOUTS, False, False)
    assert set(released) == {"base_look", "units", "R6"} and "base_whole" not in released["R6"]   # no none/true


def test_arms_are_compared_by_target_with_the_specs_pass_lines():
    plan = synthetic_plan()

    def results(flip, look):
        records = state_records(plan, flip_modern=flip, flip_after=flip)
        for r in records:
            if r["condition"] == "references/swapped":
                r["measures"] = (np.array([0.6, 0.8, 0, 0, 0, 0]) * look * (1 if "mod" in r["cell"].split("->")[1]
                                                                             else -1)).tolist()
        return {"look1": json.loads(json.dumps(ev.summarise_look1(records, [], [], plan, ["R2", "R3"], True, True)))}

    b, a, a2 = results(7.0, 0.1), results(8.0, 0.5), results(8.2, 0.45)
    lines = ev.compare_look1(b, {"A": a, "A2": a2}, "B")
    assert any(line.startswith("R2 non-inferiority") and "PASS" in line for line in lines)    # 7 - 0.5 >= 8 - 1.5
    assert any(line.startswith("R3 primary") and "PASS" in line for line in lines)            # 0.1 <= 0.25, 0.4 >= 0.15
    assert any(line.startswith("R2 read-out flip, this - A") for line in lines)
    worse = ev.compare_look1(results(5.0, 0.4), {"A": a, "A2": a2}, "B")
    assert any(line.startswith("R2 non-inferiority") and "FAIL" in line for line in worse)
    assert any(line.startswith("R3 primary") and "FAIL" in line for line in worse)
    assert any("need --compare A=" in line for line in ev.compare_look1(b, {}, "B"))


# ----------------------------------------------------------------------------- the state read-out
def test_the_probe_is_the_studys_pipeline_as_arrays(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    rng = np.random.default_rng(3)
    features = rng.normal(size=(60, 12)) + 0.5
    after = features[:, 0] + 0.3 * rng.normal(size=60) > 0.5
    probe = readout.fit_probe(features, after)
    reference = make_pipeline(Normalizer(), StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)).fit(features,
                                                                                                           after)
    np.testing.assert_allclose(probe.logit(features), reference.decision_function(features), rtol=1e-6, atol=1e-8)
    path = readout.save_probe(tmp_path / "probe.npz", probe, {"fitted_on": ["a.jpg"], "min_region_share": 0.1})
    loaded, record = readout.load_probe(path)
    np.testing.assert_allclose(loaded.logit(features), probe.logit(features))
    assert record["fitted_on"] == ["a.jpg"]
    with pytest.raises(ValueError, match="fitted_on"):
        readout.save_probe(tmp_path / "x.npz", probe, {})


def test_crops_are_grey_contrast_normalised_squares_around_the_region():
    pytest.importorskip("cv2")
    plain = np.full((256, 256, 3), 90, dtype=np.uint8)
    region = np.zeros((256, 256), bool)
    region[100:110, 30:40] = True
    crop = readout.region_crop(plain, region)
    assert crop.shape == (224, 224, 3) and len(np.unique(crop)) == 1      # no contrast: the normalised mean, 0.5
    assert abs(float(crop[0, 0, 0]) - 127.5) <= 0.5
    ramp = np.repeat(np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 256, 0), 3, 2)
    crop = readout.region_crop(ramp, region).astype(float)
    assert (crop[..., 0] == crop[..., 2]).all() and crop[:, :, 0].mean() == pytest.approx(128, abs=2)
    assert crop[:, 0, 0].mean() < crop[:, -1, 0].mean()                 # the ramp survives, normalised
    big = np.repeat(np.repeat(ramp, 2, 0), 2, 1)
    cells = np.repeat(np.repeat(np.kron(np.eye(8, dtype=bool), np.ones((8, 8), bool))[:64, :64], 8, 0), 8, 1)
    small_cells = np.repeat(np.repeat(cells[::8, ::8], 4, 0), 4, 1)
    assert np.array_equal(readout.readout_crop(big, cells), readout.region_crop(ramp, small_cells))
    with pytest.raises(ValueError, match="at least one pixel"):
        readout.region_crop(plain, np.zeros((256, 256), bool))


def test_the_instrument_reads_regions_large_enough_and_refuses_views_it_was_fitted_on():
    pytest.importorskip("cv2")

    class MeanEncoder(torch.nn.Module):
        """Features: the mean of each channel and of its square."""

        def __init__(self):
            super().__init__()
            self.layernorm = torch.nn.LayerNorm(6)

        def forward(self, pixels):
            return torch.cat([pixels.mean((2, 3)), pixels.pow(2).mean((2, 3))], 1)

    probe = readout.StateProbe(mean=np.zeros(6), scale=np.ones(6), coef=np.arange(6.0), intercept=1.0)
    instrument = readout.StateReadout(probe, {"fitted_on": ["seen.jpg"], "min_region_share": 0.05},
                                      encoder=MeanEncoder())
    images = np.random.default_rng(4).integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
    regions = np.zeros((2, 64, 64), bool)
    regions[0, :32, :32] = True
    regions[1, 0, :8] = True                                             # under 5% of the frame
    logits = instrument.readout_logit(images, regions)
    assert np.isfinite(logits[0]) and np.isnan(logits[1])
    instrument.check_unseen(["unseen.jpg"])
    with pytest.raises(ValueError, match="fitted on 1"):
        instrument.check_unseen(["seen.jpg", "other.jpg"])


@pytest.mark.skipif(not (STUDY / "readout_features.npz").exists(), reason="needs the appearance study's stored features")
def test_the_written_out_dinov2_computes_the_studys_features():
    try:
        folder = readout.dinov2_folder()
    except FileNotFoundError:
        pytest.skip("facebook/dinov2-base is not in the Hugging Face cache")
    crops = np.load(STUDY / "readout_crops.npz", allow_pickle=True)["crops"][[0, 300]]
    stored = np.load(STUDY / "readout_features.npz", allow_pickle=True)["feats"][[0, 300]]
    torch.set_num_threads(4)
    features = readout.encode(readout.DinoV2.from_folder(folder), crops)
    np.testing.assert_allclose(features, stored, atol=1e-4)


# ----------------------------------------------------------------------------- the weight-level date profile
def test_the_date_profile_of_an_untrained_embedding_is_flat_and_of_a_trained_one_not():
    model = adapted_model(trained=False)
    state = {f"embedding.{key}": value for key, value in model.date.embedding.state_dict().items()}
    flat = ev.date_profile(state, 4, 16, FIRE, WITHIN, (1890.0, 2015.0, 2021.0))
    assert set(flat) == {"spread_modern", "spread_after", "between_means", "across", "within_modern", "within_after"}
    assert all(value == 0 for value in flat.values())
    trained = adapted_model(trained=True)
    state = {f"embedding.{key}": value for key, value in trained.date.embedding.state_dict().items()}
    profile = ev.date_profile(state, 4, 16, FIRE, WITHIN, (1890.0, 2015.0, 2021.0))
    assert all(value > 0 for value in profile.values())


# ----------------------------------------------------------------------------- scripts/mvgen/make_eval_sets.py
def test_donors_come_from_other_units_match_the_targets_labels_and_are_never_shared():
    script = load_script("make_eval_sets")
    labels = pd.DataFrame({"tone_checked": ["colour"] * 10, "season": ["spring"] * 6 + ["summer"] * 4,
                           "sky_class": ["clear"] * 10})
    units = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7])
    period = np.array([1, 1, 1, 1, 1, 2, 2, 2, 1, 1])
    donors = script.choose_donors([2, 3], [0, 1, 2, 3], 1, units, np.ones(10, bool), period, labels,
                                  np.random.default_rng(0))
    assert all(units[d] not in (0, 1) for d in donors["other"] + donors["same"])
    assert all(period[d] == 2 for d in donors["other"]) and all(period[d] == 1 for d in donors["same"])
    assert len(set(donors["other"] + donors["same"])) == 4
    full = ["tone_checked", "season", "sky_class"]
    assert (donors["other"][0], donors["other_match"][0]) == (5, full)             # the only spring view after
    assert (donors["same"][0], donors["same_match"][0]) == (4, full)               # the only free spring view before
    assert donors["other_match"][1] == donors["same_match"][1] == ["tone_checked"]    # no spring view is left


def test_the_script_writes_sets_that_the_evaluation_reads_back_for_the_runs_hold_out(regions_folder, tmp_path,
                                                                                    monkeypatch):
    script = load_script("make_eval_sets")
    views = pd.read_parquet(regions_folder / "views.parquet")
    # one photographer took rows 8-11 and another 16-19; every other view is its own unit
    views["author"] = ["x" if 8 <= row < 12 else "y" if 16 <= row < 20 else f"a{row}" for row in range(N)]
    views.to_parquet(regions_folder / "views.parquet", index=False)
    labels = views[["file_key"]].assign(tone_checked=["bw"] * 2 + ["toned"] * 2 + ["colour"] * (N - 4),
                                        season="spring", sky_class="clear")
    labels.to_parquet(tmp_path / "labels.parquet", index=False)
    scene = MVGenScene(regions_folder, ViewSamplingConfig(), resolution=SIZE)
    seed = next(s for s in range(500) if script.heldout_views(scene, s)[[*range(8, 12), *range(16, 20)]].all())
    # 24 views are too few for donors from other units of the same period (they are tested above)
    monkeypatch.setattr(script, "choose_donors", lambda targets, *args: {
        "other": list(targets), "same": list(targets), "other_match": [[]] * len(targets),
        "same_match": [[]] * len(targets)})
    out = tmp_path / "sets.json"
    script.main(["--dataset", str(regions_folder), "--out", str(out), "--labels", str(tmp_path / "labels.parquet"),
                 "--resolution", str(SIZE), "--seed", str(seed)])
    content = json.loads(out.read_text())
    assert content["holdout"]["seed"] == seed and content["holdout"]["config"]["by"] == "component"
    assert content["holdout"]["critique_check"]["seed"] == 4 and set(content["yardsticks"]) == {"look_gap", "chroma"}
    assert content["yardsticks"]["look_gap"]["views"] == {"modern": 4, "after": 4}
    with pytest.raises(SystemExit, match="--overwrite"):
        script.main(["--dataset", str(regions_folder), "--out", str(out)])
    scene.hold_out(scene.holdout_rows(HoldoutConfig(**{**content["holdout"]["config"], "era_edges": EDGES})))
    plan = ev.read_eval_sets(out, scene)
    ev.check_eval_sets(plan, scene)
    assert plan.heldout == tuple(np.flatnonzero(scene.heldout)) and plan.sets
    for view_set in plan.sets:
        assert sum(view_set.scored) >= 3 and all(scene.heldout[t] == s for t, s in zip(view_set.targets, view_set.scored))
        assert not scene.heldout[list(view_set.rows[:view_set.n_ref])].any()


# ----------------------------------------------------------------------------- scripts/mvgen/fit_readout.py helpers
def test_the_readout_check_measures_within_period_cues():
    script = load_script("fit_readout")
    logit = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 2.0, 2.2, 1.8, 2.1, 1.9, -1, -1, -1, -1, -1, -1])
    after = np.array([True] * 10 + [False] * 6)
    cue = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=float)
    result = script.within_period(logit, after, cue)
    assert result["after"]["difference"] == pytest.approx(-1.0) and result["modern"] is None     # 3 / 3: too few


# ----------------------------------------------------------------------------- scripts/mvgen/eval_dates.py --eval-sets
def stand_in_readout(monkeypatch):
    monkeypatch.setattr(readout.StateReadout, "load", classmethod(lambda cls, path, device="cpu", encoder=None:
                                                                  FakeReadout()))


def test_look1_options_are_parsed_and_contradictions_refused(tmp_path, capsys):
    script = load_script("eval_dates")
    out = ["--out", str(tmp_path / "out")]
    args = script.parse_args([*out, "--run", "/r", "--eval-sets", "/s.json", "--readouts", "R1", "R3", "--weights",
                              "raw", "--expect-step", "2000", "--compare", "A=/a.json", "--arm", "B", "--cells",
                              "modern->after"])
    assert (args.readouts, args.weights, args.expect_step, args.compare, args.arm) == (["R1", "R3"], "raw", 2000,
                                                                                      ["A=/a.json"], "B")
    assert args.keys_file == [] and (args.primary_seeds, args.other_seeds) == (4, 2)
    assert script.parse_args(out).weights == "ema"
    assert script.parse_args([*out, "--from-results", "/x.json"]).from_results == Path("/x.json")
    refused = {"--eval-sets needs --run": ["--eval-sets", "/s.json"],
               "brings its own view sets": ["--run", "/r", "--eval-sets", "/s.json", "--heldout", "2"],
               "at least 1": ["--run", "/r", "--eval-sets", "/s.json", "--primary-seeds", "0"],
               "made from renders": ["--run", "/r", "--eval-sets", "/s.json", "--no-renders"],
               "go with --eval-sets": ["--run", "/r", "--heldout", "2", "--strict"]}
    for message, argv in refused.items():
        with pytest.raises(SystemExit):
            script.parse_args([*out, *argv])
        assert message in capsys.readouterr().err


def test_arm_a_is_evaluated_on_the_eval_sets_without_the_look_axes(run, tmp_path, monkeypatch, capsys):
    script = load_script("eval_dates")
    stand_in_readout(monkeypatch)
    monkeypatch.setattr(script, "load_finetuned", lambda checkpoint, model_dir: mvgen_load(checkpoint))
    sets = tiny_plan(run.scene, tmp_path / "sets.json")
    out = tmp_path / "eval"
    script.main(["--run", str(run.run_dir), "--checkpoint", "3", "--eval-sets", str(sets), "--out", str(out),
                 "--primary-seeds", "2", "--other-seeds", "1", "--steps", "2", "--noise-seeds", "0", "--arm", "A"])
    printed = capsys.readouterr().out
    results = json.loads((out / "results.json").read_text())
    assert results["model"]["has_date"] and not results["model"]["has_look"] and results["model"]["step"] == 3
    assert set(results["look1"]) == {"base_look", "units", "R2", "R3", "R6"} and results["look1"]["base_look"] == "none"
    assert {r["condition"] for r in results["records"]} == {"none/true", "none/swapped", "none/within"}
    assert len(results["records"]) == 4 * 2 * 3 * 2                     # sets x scored targets x conditions x seeds
    assert "skipped for this model (no look input): ['R1', 'R4', 'R5']" in printed
    assert "R2 state at the base look" in printed and "R3 look of the cross-fire swap" in printed
    assert "WEIGHT-LEVEL DATE PROFILE" in printed and set(results["date_profile"]) >= {"1", "2", "3"}
    assert (out / "losses.jsonl").exists() and results["look1"]["R2"]["loss_inside_minus_outside"]["all"]["targets"] == 8
    assert "about" in printed and "GPU-h" in printed
    reprinted = tmp_path / "again"
    capsys.readouterr()
    script.main(["--from-results", str(out / "results.json"), "--compare", f"A2={out / 'results.json'}", "--out",
                 str(reprinted)])
    again = capsys.readouterr().out
    assert "R2 read-out flip, this - A2: +0.00" in again and not reprinted.exists()


def mvgen_load(checkpoint):
    """``load_finetuned`` on the tiny model."""
    from timemachine.mvgen import model as mvgen_model

    real = mvgen_model.load_mvgen
    mvgen_model.load_mvgen = lambda model_dir, device: tiny_model()
    try:
        return mvgen_model.load_finetuned(checkpoint, device="cpu")
    finally:
        mvgen_model.load_mvgen = real


def test_a_look_arm_gets_every_condition_and_the_released_model_only_its_quality(run, tmp_path, monkeypatch, capsys):
    script = load_script("eval_dates")
    stand_in_readout(monkeypatch)
    look = write_looks(Path(run.cfg.dataset))
    model = look_model(look["stats"])
    ckpt = torch.load(run.run_dir / "checkpoints" / "step_000003.pt", map_location="cpu", weights_only=False)
    ckpt["finetune_config"]["look"] = True
    monkeypatch.setattr(script, "load_model", lambda *args: (model, ckpt))
    real_settings = script.settings

    def with_look_config(args):
        cfg, years, checkpoint = real_settings(args)
        return replace(cfg, finetune=SimpleNamespace(look=True)), years, checkpoint

    monkeypatch.setattr(script, "settings", with_look_config)
    sets = tiny_plan(run.scene, tmp_path / "sets.json")
    out = tmp_path / "b"
    script.main(["--run", str(run.run_dir), "--checkpoint", "3", "--eval-sets", str(sets), "--out", str(out),
                 "--primary-seeds", "1", "--other-seeds", "1", "--steps", "2", "--no-losses", "--arm", "B"])
    printed = capsys.readouterr().out
    results = json.loads((out / "results.json").read_text())
    assert results["model"]["has_look"] and results["look1"]["base_look"] == "references"
    assert {"R1", "R2", "R3", "R4", "R5", "R6"} <= set(results["look1"])
    conditions = {r["condition"] for r in results["records"]}
    assert {"references/old", "grey/true", "other/true", "same/true", "neutral/swapped", "own/true"} <= conditions
    assert {r["set"] for r in results["records"] if r["condition"] == "references/old"} == {"mod-mod/0", "aft-mod/0"}
    assert "R1 steering" in printed and "R4 look carries the state?" in printed and "R5 real photos" in printed

    monkeypatch.setattr(script, "settings", real_settings)
    monkeypatch.setattr(script, "load_model", lambda *args: (tiny_model(), {}))
    released = tmp_path / "released"
    script.main(["--run", str(run.run_dir), "--checkpoint", "released", "--eval-sets", str(sets), "--out",
                 str(released), "--primary-seeds", "2", "--steps", "2", "--arm", "released",
                 "--compare", f"B={out / 'results.json'}"])
    printed = capsys.readouterr().out
    results = json.loads((released / "results.json").read_text())
    assert set(results["look1"]) == {"base_look", "units", "R6"} and not results["model"]["has_date"]
    assert {r["condition"] for r in results["records"]} == {"none/true"} and not (released / "losses.jsonl").exists()
    assert "colour matched to the photo after the fact" in printed


def test_raw_weights_are_those_of_latest_pt_and_another_step_is_refused(run):
    script = load_script("eval_dates")
    from timemachine.mvgen import model as mvgen_model

    real = mvgen_model.load_mvgen
    mvgen_model.load_mvgen = lambda model_dir, device="cuda": tiny_model()
    try:
        script.load_finetuned = lambda checkpoint, model_dir: mvgen_model.load_finetuned(checkpoint, model_dir, "cpu")
        latest = run.run_dir / "checkpoints" / "latest.pt"
        model, ckpt = script.load_model(latest, "unused", "raw", 3)
        raw = torch.load(latest, map_location="cpu", weights_only=False)["raw_trainable"]
        torch.testing.assert_close(model.date.embedding.mlp[2].weight, raw["date"]["embedding.mlp.2.weight"])
        assert ckpt["weights"] == "raw"
        with pytest.raises(SystemExit, match="not 2"):
            script.load_model(latest, "unused", "ema", 2)
        with pytest.raises(SystemExit, match="no raw weights"):
            script.load_model(run.run_dir / "checkpoints" / "step_000003.pt", "unused", "raw", None)
        with pytest.raises(SystemExit, match="released model"):
            script.load_model(None, "unused", "raw", None)
    finally:
        mvgen_model.load_mvgen = real


def test_the_list_of_sets_says_how_long_the_evaluation_takes(run, tmp_path, capsys):
    script = load_script("eval_dates")
    sets = tiny_plan(run.scene, tmp_path / "sets.json")
    script.main(["--run", str(run.run_dir), "--checkpoint", "3", "--eval-sets", str(sets), "--out", str(tmp_path / "x"),
                 "--list-sets", "--readouts", "R2", "R3"])
    printed = capsys.readouterr().out
    timesteps = len(run.cfg.probe_timesteps)
    hours = (4 * 3 * 4 * 22.25 + 4 * 4 * timesteps * 2 * 0.233) / 3600
    assert f"4 sets, 48 renders, {4 * 4 * timesteps * 2} loss passes: about {hours:.2f} GPU-h" in printed
    assert not (tmp_path / "x").exists()


# ----------------------------------------------------------------------------- scripts/mvgen/render.py --look, --tone
def test_render_looks_are_parsed_and_need_a_checkpoint_with_a_look(regions_folder, tmp_path, monkeypatch):
    script = load_script("render")
    assert script.parse_look("own") == ("own", None) and script.parse_look("donor:17") == ("donor", 17)
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        script.parse_look("donor:x")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([[8, 9, 10, 16]]))
    monkeypatch.setattr(script, "load_mvgen", lambda model_dir: tiny_model())
    common = ["--out", str(tmp_path / "out"), "--dataset", str(regions_folder), "--rows-file", str(rows), "--n-ref", "2",
              "--resolution", str(SIZE), "--steps", "2", "--seeds", "0"]
    with pytest.raises(SystemExit, match="need a checkpoint"):
        script.main([*common, "--look", "own"])

    look = write_looks(regions_folder)
    model = look_model(look["stats"])
    monkeypatch.setattr(script, "load_finetuned", lambda checkpoint, model_dir: (model, {
        "finetune_config": {"look": True}, "scale": "scene", "scale_factor": 1.0, "step": 5}))
    script.main([*common, "--checkpoint", str(tmp_path / "unused.pt"), "--look", "donor:20", "--tone", "grey"])
    results = json.loads((tmp_path / "out" / "results.json").read_text())
    assert results[0]["look"] == {"targets": "donor", "donor": 20, "tone": "grey"}
    given = [look_in for _, look_in, _ in model.seen_looks if look_in is not None]
    expected = look["codes"][20].copy()
    expected[looks.TONE] = [1, 0, 0]
    np.testing.assert_allclose(given[-1][2:].numpy(), np.repeat(look["stats"].standardise(expected)[None], 2, 0),
                               atol=1e-6)
