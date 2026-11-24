"""CPU tests for the three training options that strengthen the date's signal, and for what they must not touch.

* a loss weight inside the changed region (``LossConfig.region_weight``; the regions file of a dataset folder,
  ``scripts/mvgen/prepare_regions.py``; the date-loss probe's numbers inside and outside the region);
* camera-balanced anchors (``ViewSamplingConfig.balance_cameras``);
* starting a run from another run's trained weights (``TrainConfig.init_from``).

All three are off by default, and with them off the code must do exactly what it did before they existed:
the same samples, the same loss, the same probe numbers, and runs saved by the older code must resume. The
tests of that compare with copies of the older sampler kept in this file and with run folders from which the
new settings were removed.

Nothing of the released model is loaded. The stand-ins are those of ``tests/test_mvgen_trainer.py`` (its tiny
UNet, its folder of 24 tiny views: rows 0-7 old, 8-15 modern, 16-23 after the fire), the model of
``tests/test_mvgen_eval.py`` whose output depends on the date in one quarter of the frame, and the synthetic
scene of ``tests/test_mvgen_regions.py`` for the real changed-region masks.

Run from MVGenMaster's checkout, in its environment (see ``timemachine/mvgen/__init__.py``).
"""
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("my_diffusers", reason="runs in MVGenMaster's environment, from its checkout")

# The stand-ins, and the fixtures ``fake_conditioning`` (autouse), ``folder`` and ``scene``, which pytest finds
# here by name. Importing that module also lets the ``src`` package find MVGenMaster's modules, so it comes
# before anything of ``timemachine.mvgen``.
from tests.test_mvgen_trainer import (  # noqa: E402, F401
    AFTER_FROM_MODERN, EDGES, LATENT, MODERN_FROM_MODERN, N, SIZE, SWAP_YEARS, adapted_model, config,
    fake_conditioning, flat_weights, folder, load_script, read_log, run_files, scene, tiny_model, watch_training,
    write_folder,
)
from tests.test_mvgen_data import two_sided_scene  # noqa: E402
from tests.test_mvgen_eval import region_dated_model, stand_in_models  # noqa: E402
from tests.test_mvgen_regions import AWAY, FRONT, RES  # noqa: E402
from tests.test_mvgen_regions import write_scene as write_ring_scene  # noqa: E402

from timemachine.data.holdout import HoldoutConfig, era_index  # noqa: E402
from timemachine.mvgen import data as data_module  # noqa: E402
from timemachine.mvgen import model as mvgen_model  # noqa: E402
from timemachine.mvgen import trainer as trainer_module  # noqa: E402
from timemachine.mvgen.data import (  # noqa: E402
    BALANCE_NEIGHBOURS, REGIONS_FILE, MVGenScene, ViewSamplingConfig, anchor_probabilities, camera_balance_weights,
    camera_features, draw_n_ref, draw_references, eligible_anchors, nearest_views, period_vote_accuracy,
    read_region_weights, sample_rows, target_partners, write_region_file,
)
from timemachine.mvgen.evaluation import check_run_records, open_scene, run_config  # noqa: E402
from timemachine.mvgen.model import FinetuneConfig, adapter_settings  # noqa: E402
from timemachine.mvgen.probes import (  # noqa: E402
    DateLossProbe, region_gap_line, region_loss_summary, scene_view_sets,
)
from timemachine.mvgen.regions import (  # noqa: E402
    FIRE, ChangedRegions, RegionConfig, latent_weights, view_region_cells,
)
from timemachine.mvgen.step import (  # noqa: E402
    LossConfig, build_inputs, encode_images, region_means, region_weighted_loss, target_region, training_loss,
    v_prediction_error_map, v_prediction_loss, v_prediction_loss_and_map,
)
from timemachine.mvgen.trainer import (  # noqa: E402
    ADDED_FIELDS, CONFIG_RECORDS, TrainConfig, Trainer, checkpoint_to_resume, flat_config, resume_differences,
    start_differences,
)

FP32 = LossConfig(precision="fp32")
# A training step in which nothing but the VAE's posterior, the timestep and the noise is random.
PLAIN_STEP = dict(date_dropout=0.0, camera_dropout=0.0, prior_dropout=0.0, prior_view_dropout=0.0, date_jitter=False)
SMALL = ViewSamplingConfig(n_ref=2, n_targets=3, min_targets_seen=2, top_k=4)      # view sets of 2 + 3 views
OLD_SAMPLE_FIELDS = {"images", "K", "w2c", "depth", "n_ref", "scale", "date", "date_mid", "rows", "file_keys"}
# What gate3_keep uses, in small: an EMA, a hold-out with the renders' targets, both probes on training and
# held-out views. Every interval divides the step at which the tests cut a run.
LIKE_GATE3 = dict(sampling=SMALL, grad_accum=2, warmup_steps=2, ema_decay=0.9, probe_every=2, render_every=2,
                  save_every=2, latest_every=2, render_rows=((8, 9, 16, 17),), render_seeds=(0,),
                  holdout=HoldoutConfig(fraction=0.5, probe_targets=True, seed=1, era_edges=EDGES))


# ----------------------------------------------------------------------------- helpers
def seeded(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def write_regions(folder: Path, rows=range(8, N), half_cell: bool = True) -> np.ndarray:
    """Give the tiny folder a regions file and return its weights ``[N, LATENT, LATENT]``.

    For the given rows the top-left quarter of the frame lies in the changed region (whole latent cells) and,
    with ``half_cell``, half of one more cell; the other views show no region.
    """
    cells = np.zeros((N, LATENT, LATENT), dtype=np.uint8)
    for row in rows:
        cells[row, :LATENT // 2, :LATENT // 2] = 64
        if half_cell:
            cells[row, LATENT // 2, LATENT // 2] = 32
    keys = pd.read_parquet(folder / "views.parquet")["file_key"].tolist()
    write_region_file(folder, cells, keys, cells.reshape(N, -1).mean(axis=1) / 64, resolution=SIZE, min_share=0.02,
                      edge=EDGES[-1], config={"min_days": 3})
    return cells.astype(np.float32) / 64


def measured(run_dir: Path, probes: bool | None = None) -> list[dict]:
    """A run's log without what depends on the machine (speed, memory); ``probes``: only the probe lines (True)
    or only the training lines (False)."""
    return [{name: value for name, value in record.items() if name not in ("sec_per_step", "max_mem_gb")}
            for record in read_log(run_dir) if probes is None or ("probe" in record) == probes]


def flat_state(state: dict) -> torch.Tensor:
    """A checkpoint's trainable weights in one vector, in the order of ``flat_weights``."""
    return torch.cat([state[part][key].flatten() for part in ("date", "lora") for key in sorted(state[part])])


def without_added_settings(saved: dict) -> dict:
    """A saved configuration (``asdict(TrainConfig)``, also as read from JSON) as the code wrote it before the
    settings of ``ADDED_FIELDS`` existed: without them. Changed in place and returned."""
    for name in ADDED_FIELDS:
        *parents, field = name.split(".")
        part = saved
        for parent in parents:
            part = part[parent]
        del part[field]
    return saved


def as_saved_by_the_older_code(run_dir: Path) -> None:
    """Rewrite a run folder as the code before the added settings left it: ``config.json`` and the configuration
    in every checkpoint lose those settings."""
    path = run_dir / "config.json"
    path.write_text(json.dumps(without_added_settings(json.loads(path.read_text())), indent=2))
    for path in (run_dir / "checkpoints").glob("*.pt"):
        checkpoint = torch.load(path, weights_only=False)
        without_added_settings(checkpoint["train_config"])
        torch.save(checkpoint, path)


def finished_run(folder: Path, run_dir: Path, **changes) -> TrainConfig:
    """Train a three-step run with an EMA, a date-loss probe and both kinds of checkpoint; returns its config."""
    cfg = config(folder, run_dir, **{**dict(ema_decay=0.9, probe_every=3, save_every=3, latest_every=3), **changes})
    Trainer(cfg, model=tiny_model()).train()
    return cfg


# ----------------------------------------------------------------------------- N5: runs saved by the older code
def test_a_saved_configuration_without_the_added_settings_compares_as_their_defaults():
    base = TrainConfig(dataset="d", run_dir="r")
    current = asdict(base)
    flat = flat_config(current)
    # every added setting is a setting of the configuration, and the recorded value is its default
    assert {name: flat[name] for name in ADDED_FIELDS} == ADDED_FIELDS
    assert (base.loss.region_weight, base.sampling.balance_cameras, base.init_from) == (1.0, 0.0, None)

    older = without_added_settings(json.loads(json.dumps(current)))        # as an older latest.pt holds it
    assert set(flat_config(current)) - set(flat_config(older)) == set(ADDED_FIELDS)
    assert resume_differences(older, current) == ([], [])
    assert resume_differences(older, asdict(replace(base, steps=8000))) == ([], ["steps: 10000 -> 8000"])

    # a launch that switches an option on is not that run: the option is named, with the default it was saved under
    relaunch = TrainConfig(dataset="d", run_dir="r", loss=LossConfig(region_weight=10.0), init_from="/runs/a/latest.pt",
                           sampling=ViewSamplingConfig(balance_cameras=5.0))
    assert resume_differences(older, asdict(relaunch)) == (
        ["init_from: None -> '/runs/a/latest.pt'", "loss.region_weight: 1.0 -> 10.0",
         "sampling.balance_cameras: 0.0 -> 5.0"], [])
    assert resume_differences(asdict(relaunch), current)[0] == [
        "init_from: '/runs/a/latest.pt' -> None", "loss.region_weight: 10.0 -> 1.0",
        "sampling.balance_cameras: 5.0 -> 0.0"]
    # only those settings: any other field that the saved configuration lacks is still a conflict
    del older["loss"]["precision"]
    assert resume_differences(older, current) == (["loss.precision: '(not recorded)' -> 'bf16'"], [])


def test_a_run_saved_by_the_older_code_resumes_and_continues_bit_for_bit(folder, tmp_path, capsys):
    """What the queued relaunch of gate3_keep does: a run that finished its steps under the older code (no new
    setting in its config.json, its checkpoints or its baselines) is launched again with more steps."""
    whole = Trainer(config(folder, tmp_path / "whole", steps=4, **LIKE_GATE3), model=tiny_model())
    whole.train()

    cut = config(folder, tmp_path / "cut", steps=2, **LIKE_GATE3)
    Trainer(cut, model=tiny_model()).train()
    run_dir = Path(cut.run_dir)
    as_saved_by_the_older_code(run_dir)
    saved = json.loads((run_dir / "config.json").read_text())
    assert "region_weight" not in saved["loss"] and "balance_cameras" not in saved["sampling"]
    assert "init_from" not in saved and "init_weights" not in saved
    assert run_config(run_dir) == cut                           # read back as the same settings (eval_dates.py)
    baselines = {path.name: path.read_bytes() for path in (run_dir / "probes").glob("*_baseline.json")}
    assert set(baselines) == {"train_baseline.json", "heldout_baseline.json"}
    capsys.readouterr()

    longer = config(folder, run_dir, steps=4, **LIKE_GATE3)
    resumed = Trainer(longer, model=tiny_model(), resume=checkpoint_to_resume(run_dir, "auto"))
    printed = capsys.readouterr().out
    assert "resuming with changed settings: steps: 2 -> 4\n" in printed and resumed.step == 2
    assert "camera-balanced" not in printed and "changed region" not in printed and "starting at step 0" not in printed
    resumed.train()

    assert torch.equal(flat_weights(resumed.model), flat_weights(whole.model))          # bit for bit
    assert all(torch.equal(a, b) for a, b in zip(resumed.ema.shadow, whole.ema.shadow))
    assert measured(run_dir) == measured(tmp_path / "whole")    # the same lines: keys, losses, probe numbers
    mine, theirs = run_files(run_dir), run_files(tmp_path / "whole")
    kept = [name for name in theirs if name.startswith(("probes/", "renders/"))]
    assert len(kept) == 4 + 3 * 2 and all(mine[name] == theirs[name] for name in kept)
    assert all(mine[f"probes/{name}"] == content for name, content in baselines.items())     # baselines untouched
    # and the checkpoints it writes now hold the settings, at their defaults
    latest = torch.load(run_dir / "checkpoints" / "latest.pt", weights_only=False)
    assert latest["train_config"] == asdict(longer) and "init_weights" not in latest
    assert "region_fingerprint" not in latest
    assert latest["train_config"]["loss"]["region_weight"] == 1.0
    assert set(json.loads((run_dir / "config.json").read_text())) == set(asdict(longer)) | {"scale_factor"}


# ----------------------------------------------------------------------------- N1: the regions file and the scene
def test_a_scene_hands_the_stored_region_weights_to_its_samples(folder):
    rows = np.array(AFTER_FROM_MODERN)
    plain = MVGenScene(folder, resolution=SIZE)
    before = plain.make_sample(rows, 3)
    assert not plain.has_regions and plain.regions is None and plain.region_fingerprint() is None
    assert set(before) == OLD_SAMPLE_FIELDS                     # no regions file: a sample is what it always was
    zeros = plain.region_weights(rows)
    assert zeros.shape == (6, 1, LATENT, LATENT) and zeros.dtype == torch.float32 and not zeros.any()

    weights = write_regions(folder, rows=range(12, N))
    scene = MVGenScene(folder, resolution=SIZE)
    assert scene.has_regions and len(scene.region_fingerprint()) == 64
    region = scene.region_weights(rows)
    assert region.shape == (6, 1, LATENT, LATENT) and region.dtype == torch.float32
    assert torch.equal(region[:, 0], torch.from_numpy(weights[rows]))
    assert not region[:3].any()                                 # rows 8-10 show no region
    assert region[3, 0, :2, :2].eq(1).all() and region[3, 0, 2, 2] == 0.5 and region[3, 0, 3, 3] == 0
    sample = scene.make_sample(rows, 3)
    assert set(sample) == OLD_SAMPLE_FIELDS | {"region"} and torch.equal(sample["region"], region)
    region[:] = 7.0                                             # a sample's weights are its own copy
    assert torch.equal(scene.region_weights(rows)[:, 0], torch.from_numpy(weights[rows]))

    # everything else is as without the file: the other fields, the views that are drawn, the fingerprint
    for name in OLD_SAMPLE_FIELDS - {"n_ref", "scale", "file_keys"}:
        assert torch.equal(sample[name], before[name]), name
    assert scene.fingerprint() == plain.fingerprint()
    assert all(np.array_equal(scene.indices(i), plain.indices(i)) for i in range(50))

    with np.load(folder / REGIONS_FILE) as stored:              # what the file holds
        assert stored["cells"].dtype == np.uint8 and stored["cells"].shape == (N, LATENT, LATENT)
        assert (int(stored["factor"]), int(stored["resolution"])) == (8, SIZE)
        assert stored["file_keys"].tolist() == scene.views["file_key"].tolist()
        assert float(stored["edge"]) == EDGES[-1] and float(stored["min_share"]) == 0.02
        assert json.loads(str(stored["config"])) == {"min_days": 3} and stored["share"].dtype == np.float32
        assert stored["share"][12] == pytest.approx((4 + 0.5) / 16) and stored["share"][0] == 0
    other = write_regions(folder, rows=range(12, N), half_cell=False)
    assert MVGenScene(folder, resolution=SIZE).region_fingerprint() != scene.region_fingerprint()
    assert other.sum() < weights.sum()
    with pytest.raises(ValueError, match="cells must be uint8"):
        write_region_file(folder, np.zeros((N, LATENT, LATENT)), ["a"] * N, np.zeros(N), resolution=SIZE,
                          min_share=0.02, edge=2019.0, config={})
    assert not (folder / (REGIONS_FILE + ".tmp")).exists()      # written under another name, then renamed


@pytest.mark.parametrize("problem, reason", [
    ("another view table", "made for another view table"), ("another resolution", "made for crops of side 32"),
    ("not an archive", "cannot be read"), ("empty", "cannot be read")])
def test_a_regions_file_that_does_not_fit_is_left_aside_with_a_note(folder, capsys, problem, reason):
    """The regions are optional: a file that cannot be used must not stop a run that does not need it."""
    write_regions(folder)
    keys = pd.read_parquet(folder / "views.parquet")["file_key"].tolist()
    resolution = SIZE
    if problem == "another view table":                         # the table was rebuilt: its rows are other views now
        write_region_file(folder, np.zeros((N, LATENT, LATENT), dtype=np.uint8), keys[::-1], np.zeros(N),
                          resolution=SIZE, min_share=0.02, edge=EDGES[-1], config={})
    elif problem == "another resolution":
        resolution = 2 * SIZE
    else:
        (folder / REGIONS_FILE).write_bytes(b"" if problem == "empty" else b"not a zip archive")
    capsys.readouterr()
    scene = MVGenScene(folder, resolution=resolution)
    printed = capsys.readouterr().out
    assert not scene.has_regions and scene.region_fingerprint() is None
    assert f"{folder / REGIONS_FILE} is not used: it" in printed and reason in printed
    assert "prepare_regions.py --overwrite" in printed
    assert read_region_weights(folder, np.array(keys), resolution) is None
    if resolution == SIZE:
        assert set(scene[0]) == OLD_SAMPLE_FIELDS
    capsys.readouterr()
    assert read_region_weights(folder / "nowhere", np.array(keys), SIZE) is None        # no file: no note either
    assert capsys.readouterr().out == ""


def test_view_region_cells_reduces_every_mask_and_empties_the_small_ones():
    quarter = np.zeros((SIZE, SIZE), dtype=bool)
    quarter[:SIZE // 2, :SIZE // 2] = True
    ragged = np.zeros((SIZE, SIZE), dtype=bool)
    ragged[:11, :13] = True                                     # cuts through latent cells
    specks = np.zeros((SIZE, SIZE), dtype=bool)
    specks[5, 5:8] = True                                       # 3 pixels of 1,024

    class Masks:
        scene = SimpleNamespace(views=range(5))

        def mask(self, row, resolution):
            assert resolution == SIZE
            if row == 3:
                raise OSError("no such image")
            return [quarter, ragged, specks, None, np.zeros_like(quarter)][row]

    cells, share = view_region_cells(Masks(), SIZE, min_share=0.02)
    assert cells.dtype == np.uint8 and cells.shape == (5, LATENT, LATENT)
    for row, mask in ((0, quarter), (1, ragged)):               # the masked pixels per cell, exactly
        assert np.array_equal(cells[row] / 64, latent_weights(mask)) and share[row] == mask.mean()
        assert cells[row].sum() == mask.sum()
    assert cells[1, 1, 1] == 3 * 5 and cells[1, 0, 0] == 64
    assert share[2] == pytest.approx(3 / 1024) and not cells[2].any()       # too small: stored empty, share kept
    assert np.isnan(share[3]) and not cells[3].any()                        # no image: no mask
    assert share[4] == 0 and not cells[4].any()
    every, _ = view_region_cells(Masks(), SIZE)                 # without a smallest share the specks are kept
    assert every[2].sum() == 3 and np.array_equal(every[[0, 1]], cells[[0, 1]])


def test_prepare_regions_writes_what_a_scene_then_hands_out(tmp_path, capsys):
    """The script on the synthetic scene of the regions tests, with the real ``ChangedRegions``."""
    ring = write_ring_scene(tmp_path / "ring")
    regions = ChangedRegions(MVGenScene(ring.dir, resolution=RES, require_depth=False))
    shares = np.array([regions.share(row, RES) for row in range(len(regions.scene.views))])
    smallest = float(np.median(shares[shares > 0]))             # about half of the masks are smaller than this
    kept = shares >= smallest
    assert 10 < kept.sum() < len(kept) - 10 and (shares[~kept] > 0).any()

    script = load_script("prepare_regions")
    arguments = ["--dataset", str(ring.dir), "--resolution", str(RES), "--min-share", repr(smallest)]
    script.main(arguments)
    printed = capsys.readouterr().out
    assert "changed points at 2019.2877: gone:" in printed and "with_region" in printed
    assert f"wrote {ring.dir / REGIONS_FILE}" in printed
    assert "usable views with a region (a mask of at least " in printed

    scene = MVGenScene(ring.dir, resolution=RES, require_depth=False)
    assert scene.has_regions and scene.regions.shape == (len(scene.views), RES // 8, RES // 8)
    for row in range(len(scene.views)):
        expected = latent_weights(regions.mask(row, RES)) if kept[row] else np.zeros((RES // 8, RES // 8))
        assert np.array_equal(scene.regions[row], expected), row
    assert kept[FRONT] and scene.regions[FRONT].sum() * 64 == regions.mask(FRONT, RES).sum()
    assert not scene.regions[AWAY].any()                        # a view that looks away shows no region
    sample = scene.make_sample(np.array([FRONT, AWAY]), 1)
    assert sample["region"].shape == (2, 1, RES // 8, RES // 8) and sample["region"][0].sum() > 0
    with np.load(ring.dir / REGIONS_FILE) as stored:
        assert np.allclose(stored["share"], shares) and float(stored["edge"]) == FIRE
        assert json.loads(str(stored["config"])) == asdict(RegionConfig()) and float(stored["min_share"]) == smallest
    table = script.region_table(scene, shares, smallest)
    modern, after = table.loc["1970 to 2019.2877"], table.loc["from 2019.2877"]
    assert table.loc["before 1970", "usable"] == 0 and modern["usable"] + after["usable"] == scene.usable.sum()
    assert modern["with_region"] + after["with_region"] == (kept & scene.usable).sum()
    assert (table["with_region"] + table["too_small"] + table["no_mask"]).tolist() == table["usable"].tolist()

    # an existing file is kept unless --overwrite is given
    content = (ring.dir / REGIONS_FILE).read_bytes()
    with pytest.raises(SystemExit, match="exists: give --overwrite"):
        script.main([*arguments[:-1], "0.5"])
    assert (ring.dir / REGIONS_FILE).read_bytes() == content
    script.main([*arguments[:-1], "0.5", "--overwrite"])        # no mask is half the frame: every view stored empty
    emptied = MVGenScene(ring.dir, resolution=RES, require_depth=False)
    assert emptied.has_regions and not emptied.regions.any()
    assert emptied.region_fingerprint() != scene.region_fingerprint()


# ----------------------------------------------------------------------------- N2: the weighted loss
def test_region_weighted_loss_is_the_weighted_mean_and_its_gradient_is_the_weighting():
    torch.manual_seed(0)
    error = torch.rand(3, 4, 4, dtype=torch.float64, requires_grad=True)
    region = torch.zeros(3, 4, 4, dtype=torch.float64)
    region[0, :2, :2] = 1.0                                     # four cells inside
    region[1, 0, 0] = 0.5                                       # a cell on the border; the third target: no region
    loss = region_weighted_loss(error, region, 10.0)
    by_hand, total = 0.0, 0.0
    for t in range(3):
        for i in range(4):
            for j in range(4):
                w = 1 + 9 * float(region[t, i, j])
                by_hand, total = by_hand + w * float(error[t, i, j]), total + w
    assert total == 48 + 9 * 4.5 and float(loss) == pytest.approx(by_hand / total, rel=1e-12)

    # the gradient is the plain mean's, cell by cell, times the cell's weight over the mean weight
    (weighted,) = torch.autograd.grad(loss, error)
    (plain,) = torch.autograd.grad(error.mean(), error)
    weights = 1 + 9 * region
    torch.testing.assert_close(weighted, plain * weights / weights.mean(), rtol=1e-12, atol=0)
    assert weighted[0, 0, 0] == pytest.approx(10 * float(weighted[2, 3, 3])) and weighted.sum() == pytest.approx(1.0)

    # a weight of 1, or targets without any region, give the plain mean
    assert float(region_weighted_loss(error, region, 1.0)) == pytest.approx(float(error.mean()), rel=1e-12)
    assert float(region_weighted_loss(error, torch.zeros_like(region), 10.0)) == pytest.approx(float(error.mean()))

    inside, outside = region_means(error.detach(), region)
    assert float(inside) == pytest.approx(float((error[0, :2, :2].sum() + 0.5 * error[1, 0, 0]) / 4.5), rel=1e-12)
    assert float(outside) == pytest.approx(float((error.sum() - 4.5 * inside) / 43.5), rel=1e-12)
    nowhere, everywhere = region_means(error.detach(), torch.zeros_like(region))
    assert nowhere is None and float(everywhere) == pytest.approx(float(error.mean()))
    assert region_means(error.detach(), torch.ones_like(region))[1] is None
    for bad in (0.0, -2.0, float("nan")):
        with pytest.raises(ValueError, match="region_weight must be positive"):
            LossConfig(region_weight=bad)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_loss_and_map_of_one_forward_pass_are_those_of_the_two_functions(scene, precision):
    model = adapted_model(lora_rank=2)
    cfg = LossConfig(precision=precision)
    sample = scene.make_sample(np.array(AFTER_FROM_MODERN), 3)
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    noise = torch.randn(latents.shape, generator=seeded(3))
    year, valid = sample["date_mid"], torch.ones(6, dtype=torch.bool)
    arguments = (model, latents, inputs, 600, noise, year, valid, cfg, 3)
    model.unet.seen.clear()
    loss, error = v_prediction_loss_and_map(*arguments)
    assert len(model.unet.seen) == 1                            # one forward pass for both
    reference, per_view = v_prediction_loss(*arguments)
    assert torch.equal(loss, reference)                         # the same operations: the same bits
    assert torch.equal(error, v_prediction_error_map(*arguments)) and error.shape == (3, LATENT, LATENT)
    assert loss.dtype == error.dtype == torch.float32 and loss.requires_grad and error.requires_grad
    torch.testing.assert_close(error.flatten(1).mean(1), per_view, rtol=1e-5, atol=0)
    model.scheduler.config.prediction_type = "epsilon"
    with pytest.raises(ValueError, match="velocity"):
        v_prediction_loss_and_map(*arguments)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_a_training_step_with_weight_one_is_the_step_it_was(folder, precision):
    """With or without a regions file in the folder: the same loss to the bit, the same record."""
    plain = MVGenScene(folder, resolution=SIZE)
    write_regions(folder)
    with_regions = MVGenScene(folder, resolution=SIZE)
    default = LossConfig(precision=precision)
    assert default.region_weight == 1.0 and LossConfig(precision=precision, region_weight=1.0) == default
    for index in range(4):
        rows, n_ref = plain.draw(index)
        results = []
        for source in (plain, with_regions):
            model = adapted_model(lora_rank=2)
            loss, info = training_loss(model, source.make_sample(rows, n_ref), default, seeded(index), EDGES)
            model.unet.zero_grad()
            loss.backward()
            results.append((loss.detach(), info, flat_gradient(model)))
        (loss, info, gradient), (same_loss, same_info, same_gradient) = results
        assert torch.equal(loss, same_loss) and info == same_info and torch.equal(gradient, same_gradient)
        assert set(info) == {"timestep", "loss", "camera_dropped", "priors_dropped", "prior_views_dropped",
                             "pixel_views_dropped", "dates_known", "date_rms", "date_temb_ratio"}

        # and it is what the plain loss function gives for the step's own draws (the code path before the option)
        model = adapted_model(lora_rank=2)
        weighted = training_loss(model, with_regions.make_sample(rows, n_ref),
                                 LossConfig(precision=precision, region_weight=10.0), seeded(index), EDGES)[1]
        assert weighted["loss_unweighted"] == info["loss"]      # the same forward pass, the plain loss to the bit
        assert weighted["timestep"] == info["timestep"] and weighted["dates_known"] == info["dates_known"]


def flat_gradient(model) -> torch.Tensor:
    """The gradient of every trainable weight of the model in one vector."""
    params = [*model.date.parameters(), *(p for p in model.unet.parameters() if p.requires_grad)]
    return torch.cat([p.grad.flatten() for p in params])


def step_error_map(model, sample: dict, cfg: LossConfig, seed: int) -> torch.Tensor:
    """The error map ``[T, h, w]`` of the training step with this seed, from the step's own pieces.

    ``cfg`` must be a ``PLAIN_STEP``: the only random things that count are then the VAE's posterior, the
    timestep and the noise, which ``training_loss`` draws first and in this order.
    """
    generator = seeded(seed)
    latents = encode_images(model, sample["images"], generator=generator)
    timestep = int(torch.randint(model.scheduler.config.num_train_timesteps, (), generator=generator))
    noise = torch.randn(latents.shape, generator=generator)
    inputs = build_inputs(model, sample, latents)
    valid = torch.ones(len(sample["date_mid"]), dtype=torch.bool)
    with torch.no_grad():
        return v_prediction_error_map(model, latents, inputs, timestep, noise, sample["date_mid"], valid, cfg,
                                      int(sample["n_ref"]))


def test_a_weighted_training_step_weights_the_cells_and_reports_the_plain_errors(folder):
    weights = write_regions(folder)                             # rows 8-23 show the region, the old rows do not
    scene = MVGenScene(folder, resolution=SIZE)
    sample = scene.make_sample(np.array(AFTER_FROM_MODERN), 3)
    plain_cfg = LossConfig(precision="fp32", **PLAIN_STEP)
    cfg = replace(plain_cfg, region_weight=10.0)
    model = adapted_model(lora_rank=2)
    outputs = []

    def keep_output(module, args, output) -> None:
        output.retain_grad()
        outputs.append(output)

    hook = model.unet.conv_out.register_forward_hook(keep_output)
    plain_loss, plain_info = training_loss(model, sample, plain_cfg, seeded(5), EDGES)
    plain_loss.backward()
    loss, info = training_loss(model, sample, cfg, seeded(5), EDGES)
    loss.backward()
    hook.remove()
    assert torch.equal(outputs[0], outputs[1])                  # the same draws, the same forward pass

    # the loss by hand, from the error map of the step's own draws
    error = step_error_map(model, sample, plain_cfg, seed=5).double().numpy()
    region = weights[list(AFTER_FROM_MODERN[3:])].astype(np.float64)
    w = 1 + 9 * region
    assert region.sum() == 3 * 4.5 and w.sum() == 3 * 16 + 9 * 3 * 4.5
    assert float(loss) == pytest.approx((w * error).sum() / w.sum(), rel=1e-6)
    assert float(plain_loss) == pytest.approx(error.mean(), rel=1e-6)
    assert float(loss) != pytest.approx(float(plain_loss), rel=1e-3)
    assert set(info) - set(plain_info) == {"loss_unweighted", "loss_inside", "loss_outside"}
    assert info["loss"] == float(loss) and info["loss_unweighted"] == plain_info["loss"]
    assert info["loss_inside"] == pytest.approx((region * error).sum() / region.sum(), rel=1e-6)
    assert info["loss_outside"] == pytest.approx(((1 - region) * error).sum() / (1 - region).sum(), rel=1e-6)
    assert json.loads(json.dumps(info)) == info

    # the gradient at the UNet's output differs from the plain one by the weighting alone: nothing for the
    # references, and for every cell of a target its weight over the mean weight
    plain_grad, grad = outputs[0].grad, outputs[1].grad         # [F, 4, h, w]
    assert not grad[:3].any() and not plain_grad[:3].any() and plain_grad[3:].abs().min() > 0
    scale = torch.from_numpy(w / w.mean()).float()[:, None]
    torch.testing.assert_close(grad[3:], plain_grad[3:] * scale, rtol=1e-5, atol=0)

    # targets that do not show the region: nothing inside, and the weighted loss is the plain one
    old = scene.make_sample(np.arange(8), 3)
    loss, info = training_loss(adapted_model(lora_rank=2), old, cfg, seeded(5), EDGES)
    assert info["loss_inside"] is None
    assert info["loss_outside"] == pytest.approx(info["loss_unweighted"], rel=1e-6)
    assert float(loss) == pytest.approx(info["loss_unweighted"], rel=1e-6)

    # a sample without weights, or with weights of another size, cannot be weighted
    bare = {name: value for name, value in sample.items() if name != "region"}
    with pytest.raises(ValueError, match="no changed-region weights.*prepare_regions.py"):
        training_loss(model, bare, cfg, seeded(5), EDGES)
    with pytest.raises(ValueError, match="made for crops of another size"):
        training_loss(model, {**bare, "region": torch.zeros(6, 1, 2 * LATENT, 2 * LATENT)}, cfg, seeded(5), EDGES)
    error = torch.zeros(3, LATENT, LATENT, dtype=torch.float64)
    moved = target_region(sample, 3, error)                     # on the device and in the dtype of the error map
    assert moved.dtype == torch.float64 and torch.equal(moved, sample["region"][3:, 0].double())


@pytest.mark.parametrize("dropped, entry, value", [
    (dict(date_dropout=1.0), "dates_known", False), (dict(camera_dropout=1.0), "camera_dropped", True),
    (dict(prior_dropout=1.0), "priors_dropped", True), (dict(prior_view_dropout=1.0), "prior_views_dropped", 3),
    (dict(cross_date_pixels="drop_all"), "prior_views_dropped", 3),
    (dict(date_dropout=1.0, camera_dropout=1.0, prior_dropout=1.0), "priors_dropped", True)])
def test_the_region_weight_acts_the_same_whatever_a_step_drops(folder, dropped, entry, value):
    """Dropout decides what the UNet is given; the weight acts on its output. So a step that hides the dates,
    the camera rays or the priors draws what the plain step draws and weights every cell of a target as any
    other step does. Also with gradient checkpointing, where ``backward()`` runs the blocks again."""
    weights = write_regions(folder)
    sample = MVGenScene(folder, resolution=SIZE).make_sample(np.array(AFTER_FROM_MODERN), 3)
    steps = {}
    for weight in (1.0, 10.0):
        model = adapted_model(lora_rank=2)
        model.unet.train()                                      # its blocks then checkpoint, as MVGenMaster's do
        outputs = []

        def keep_output(module, args, output) -> None:
            output.retain_grad()
            outputs.append(output)

        hook = model.unet.conv_out.register_forward_hook(keep_output)
        generator = seeded(7)
        cfg = LossConfig(precision="fp32", region_weight=weight, **{**PLAIN_STEP, **dropped})
        loss, info = training_loss(model, sample, cfg, generator, EDGES)
        loss.backward()
        hook.remove()
        assert model.unet.gradient_checkpointing and model.unet.seen[-1]["training"] and info[entry] == value
        steps[weight] = (outputs[0].grad, info, generator.get_state())
    (plain_grad, plain_info, plain_state), (grad, info, state) = steps[1.0], steps[10.0]
    assert torch.equal(state, plain_state)                      # as many random numbers drawn
    assert info["loss_unweighted"] == plain_info["loss"] and info["loss_inside"] is not None
    assert {name: info[name] for name in plain_info if name != "loss"} == \
        {name: plain_info[name] for name in plain_info if name != "loss"}
    w = torch.from_numpy(1 + 9 * weights[list(AFTER_FROM_MODERN[3:])])
    assert not grad[:3].any() and plain_grad[3:].abs().min() > 0
    torch.testing.assert_close(grad[3:], plain_grad[3:] * (w / w.mean())[:, None], rtol=1e-5, atol=0)


def test_a_run_with_a_region_weight_logs_the_plain_loss_inside_and_outside(folder, tmp_path, monkeypatch, capsys):
    write_regions(folder)
    with monkeypatch.context() as patch:
        plain_samples = watch_training(patch)
        plain = Trainer(config(folder, tmp_path / "plain", latest_every=1), model=tiny_model())
        plain.train()
    capsys.readouterr()
    cfg = config(folder, tmp_path / "run", latest_every=1, loss=LossConfig(precision="fp32", region_weight=10.0))
    with monkeypatch.context() as patch:
        samples = watch_training(patch)
        trainer = Trainer(cfg, model=tiny_model())
        trainer.train()
    printed = capsys.readouterr().out
    assert (f"the loss counts the changed region 10 times: 16 of the 24 training views show it ({REGIONS_FILE})"
            in printed)
    assert samples == plain_samples                             # the weight changes the loss, not what is drawn

    log, plain_log = measured(cfg.run_dir), measured(tmp_path / "plain")
    extra = {"loss_unweighted", "loss_inside", "loss_outside"}
    assert all(set(record) - set(before) == extra for record, before in zip(log, plain_log))
    assert all(not extra & set(record) for record in plain_log)             # an unweighted run logs what it did
    # before the first update both runs have the same weights: the first step's plain loss is the plain run's
    assert log[0]["loss_unweighted"] == plain_log[0]["loss"] and log[0]["loss"] != plain_log[0]["loss"]
    assert log[1]["loss_unweighted"] != plain_log[1]["loss"]                # then they train on different losses
    for record in log:
        assert record["loss_inside"] > 0 and record["loss_outside"] > 0
        between = sorted([record["loss_inside"], record["loss"], record["loss_outside"]])
        assert between[1] == record["loss"] or record["loss_inside"] is None    # a weighted mean of the two
    assert re.search(r"\[train\] step 1 loss \d\.\d{4} \(unweighted \d\.\d{4}; inside the changed region \d\.\d{4}, "
                     r"outside \d\.\d{4}\) gradient norm", printed)

    # every checkpoint says which weights the loss used; an unweighted run's does not
    fingerprint = trainer.scene.region_fingerprint()
    for name in ("latest.pt", "step_000003.pt"):
        saved = torch.load(Path(cfg.run_dir) / "checkpoints" / name, weights_only=False)
        assert saved["region_fingerprint"] == fingerprint and saved["train_config"]["loss"]["region_weight"] == 10.0
    assert "region_fingerprint" not in torch.load(tmp_path / "plain" / "checkpoints" / "latest.pt", weights_only=False)


def test_a_region_weight_needs_the_regions_file_and_the_file_it_started_with(folder, tmp_path, monkeypatch):
    weighted = LossConfig(precision="fp32", region_weight=10.0)
    cfg = config(folder, tmp_path / "run", steps=4, latest_every=2, loss=weighted)
    with pytest.raises(ValueError, match="has no changed-region weights.*prepare_regions.py"):
        Trainer(cfg, model=tiny_model())                        # refused before anything is written
    assert not (tmp_path / "run").exists()

    write_regions(folder)
    whole = Trainer(config(folder, tmp_path / "whole", steps=4, latest_every=2, loss=weighted), model=tiny_model())
    whole.train()
    with monkeypatch.context() as patch:
        watch_training(patch, stop_after=5)                     # the job dies in its third step
        with pytest.raises(RuntimeError, match="job killed"):
            Trainer(cfg, model=tiny_model()).train()
    latest = checkpoint_to_resume(cfg.run_dir, "auto")
    before = run_files(cfg.run_dir)

    # the regions were made again with other masks, or are gone: the run would go on with another loss
    content = (folder / REGIONS_FILE).read_bytes()
    write_regions(folder, half_cell=False)
    with pytest.raises(ValueError, match="changed-region weights that the run's loss uses differ"):
        Trainer(cfg, model=tiny_model(), resume=latest)
    (folder / REGIONS_FILE).unlink()
    with pytest.raises(ValueError, match="changed-region weights that the run's loss uses differ"):
        Trainer(cfg, model=tiny_model(), resume=latest)
    assert run_files(cfg.run_dir) == before
    (folder / REGIONS_FILE).write_bytes(content)

    resumed = Trainer(cfg, model=tiny_model(), resume=latest)   # the file is back: the run goes on, exactly
    resumed.train()
    assert torch.equal(flat_weights(resumed.model), flat_weights(whole.model))
    assert [r["loss"] for r in measured(cfg.run_dir)] == [r["loss"] for r in measured(tmp_path / "whole")]
    # a relaunch with another weight is another run
    other = config(folder, tmp_path / "run", steps=4, latest_every=2, loss=replace(weighted, region_weight=5.0))
    with pytest.raises(ValueError, match="loss.region_weight: 10.0 -> 5.0"):
        Trainer(other, model=tiny_model(), resume=latest)


def test_a_regions_file_leaves_a_run_without_region_weight_as_it_was(folder, tmp_path, capsys):
    """The file is new in the dataset folder of runs that exist. With the options off such a run must train on
    the same samples to the same weights and log the same numbers; its probes may say more, not something else."""
    before = Trainer(config(folder, tmp_path / "before", steps=4, **LIKE_GATE3), model=tiny_model())
    before.train()
    cut = config(folder, tmp_path / "cut", steps=2, **LIKE_GATE3)
    Trainer(cut, model=tiny_model()).train()                    # a run that stops before the file exists ...
    baselines = {path.name: path.read_bytes() for path in (tmp_path / "cut" / "probes").glob("*_baseline.json")}
    capsys.readouterr()

    write_regions(folder)
    after = Trainer(config(folder, tmp_path / "after", steps=4, **LIKE_GATE3), model=tiny_model())
    after.train()
    printed = capsys.readouterr().out
    assert torch.equal(flat_weights(after.model), flat_weights(before.model))
    assert all(torch.equal(a, b) for a, b in zip(after.ema.shadow, before.ema.shadow))
    assert measured(tmp_path / "after", probes=False) == measured(tmp_path / "before", probes=False)
    mine, theirs = measured(tmp_path / "after", probes=True), measured(tmp_path / "before", probes=True)
    assert len(mine) == len(theirs) == 3 * 3
    for record, old in zip(mine, theirs):
        is_loss_probe = record["probe"].startswith("date_loss")
        assert set(record) - set(old) == ({"region"} if is_loss_probe else set())
        assert {name: value for name, value in record.items() if name != "region"} == old
    files, old_files = run_files(tmp_path / "after"), run_files(tmp_path / "before")
    assert set(files) == set(old_files)                         # no file more, none less
    same = [name for name in files if name.startswith(("probes/", "renders/")) or name == "heldout.json"]
    assert len(same) == 4 + 3 * 2 + 1 and all(files[name] == old_files[name] for name in same)
    assert set(torch.load(tmp_path / "after" / "checkpoints" / "latest.pt", weights_only=False)) == \
        set(torch.load(tmp_path / "before" / "checkpoints" / "latest.pt", weights_only=False))
    # what is new: one more line per date-loss probe
    assert len(re.findall(r"\[probe\] step \d (train|heldout), inside the changed region \(\d of 2 view sets show "
                          r"it\): ", printed)) == 3 * 2
    assert "changed region" not in "".join(line for line in printed.splitlines() if not line.startswith("[probe]"))

    # ... and is resumed after it was written: its baselines are accepted, and it arrives where the others did
    resumed = Trainer(config(folder, tmp_path / "cut", steps=4, **LIKE_GATE3), model=tiny_model(),
                      resume=checkpoint_to_resume(cut.run_dir, "auto"))
    resumed.train()
    assert torch.equal(flat_weights(resumed.model), flat_weights(before.model))
    assert measured(tmp_path / "cut", probes=False) == measured(tmp_path / "before", probes=False)
    for record, old in zip(measured(tmp_path / "cut", probes=True), theirs):
        assert {name: value for name, value in record.items() if name != "region"} == old
        assert ("region" in record) == (record["probe"].startswith("date_loss") and record["step"] > 2)
    assert {path.name: path.read_bytes() for path in (tmp_path / "cut" / "probes").glob("*_baseline.json")} == baselines


# ----------------------------------------------------------------------------- N2: the probe inside and outside
def test_the_date_loss_probe_also_measures_inside_and_outside_the_changed_region(folder, tmp_path):
    sets = [AFTER_FROM_MODERN, MODERN_FROM_MODERN]
    model = adapted_model(lora_rank=2)

    def probe(name: str) -> DateLossProbe:
        return DateLossProbe(MVGenScene(folder, resolution=SIZE), sets, 3, FP32, SWAP_YEARS, tmp_path / name,
                             timesteps=(200, 800))

    plain_probe = probe("plain.json")
    plain = plain_probe(model)
    passes = len(model.unet.seen)
    assert "region" not in plain and not plain_probe.has_regions

    weights = write_regions(folder, rows=range(16, N))          # the after-fire views show the region
    model.unet.seen.clear()
    region_probe = probe("regions.json")
    result = region_probe(model)
    assert region_probe.has_regions and len(model.unet.seen) == passes == 2 * 2 * 3    # no forward pass more
    region = result.pop("region")
    assert result == plain                                      # every other number, to the bit
    assert region_probe.last_losses == plain_probe.last_losses
    assert (tmp_path / "regions.json").read_bytes() == (tmp_path / "plain.json").read_bytes()
    assert region_probe.identity() == plain_probe.identity() and region_probe.describe() == plain_probe.describe()
    # a baseline that was written before the folder had regions is the baseline still
    assert probe("plain.json")(model) == {**plain, "region": region}

    # by hand, for the first view set: the error maps of the probe's own noise, averaged with the region's weights
    inside, outside = region["inside"], region["outside"]
    assert (inside["sets"], outside["sets"]) == (1, 2)          # the modern targets show no region
    sample = region_probe.samples[0]
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    generator, m = seeded(region_probe.seed), weights[list(AFTER_FROM_MODERN[3:])].astype(np.float64)
    valid = torch.ones(6, dtype=torch.bool)
    means = {"true": [], "swapped": []}
    for timestep in (200, 800):
        noise = torch.randn(latents.shape, generator=generator)
        for variant, year in (("true", sample["date_mid"]), ("swapped", region_probe.swapped[0])):
            with torch.no_grad():
                error = v_prediction_error_map(model, latents, inputs, timestep, noise, year, valid, FP32, 3)
            error = error.double().numpy()
            means[variant].append(((m * error).sum() / m.sum(), ((1 - m) * error).sum() / (1 - m).sum()))
    true, swapped = np.array(means["true"]), np.array(means["swapped"])
    assert inside["loss_true"] == pytest.approx(true[:, 0].mean(), rel=1e-6)
    assert inside["rel_gap_swapped"] == pytest.approx((swapped[:, 0] / true[:, 0] - 1).mean(), rel=1e-4, abs=1e-7)
    assert inside["rel_gap_swapped_by_timestep"] == pytest.approx(swapped[:, 0] / true[:, 0] - 1, rel=1e-4, abs=1e-7)
    assert inside["sample_rel_gap_swapped"] == [inside["rel_gap_swapped"], None]
    assert outside["sample_rel_gap_swapped"][0] == pytest.approx((swapped[:, 1] / true[:, 1] - 1).mean(), rel=1e-4,
                                                                  abs=1e-7)
    # a view set without the region is all outside: its numbers there are the whole frame's
    assert outside["sample_rel_gap_swapped"][1] == pytest.approx(plain["sample_rel_gap_swapped"][1], rel=1e-4, abs=1e-7)
    assert outside["sample_rel_gap_unknown"][1] == pytest.approx(plain["sample_rel_gap_unknown"][1], rel=1e-4, abs=1e-7)
    assert json.loads(json.dumps(region)) == region


def test_the_probe_finds_a_date_that_acts_inside_the_region_only(folder, tmp_path):
    """The model of the evaluation tests: its output depends on the date in the top-left quarter and nowhere
    else. The frame as a whole shows a small gap; split by the region it is all inside."""
    write_regions(folder, rows=range(16, N), half_cell=False)   # the region is that quarter, for the after-fire views
    scene = MVGenScene(folder, resolution=SIZE)
    probe = DateLossProbe(scene, [AFTER_FROM_MODERN], 3, FP32, SWAP_YEARS, tmp_path / "b.json", timesteps=(200, 800))
    result = probe(region_dated_model(probe.samples[0]["date_mid"]))
    region = result["region"]
    assert region["inside"]["rel_gap_swapped"] > 3 * result["rel_gap_swapped"] > 0      # a quarter of the frame
    assert region["inside"]["rel_gap_unknown"] > 3 * result["rel_gap_unknown"] > 0
    assert region["outside"]["rel_gap_swapped"] == region["outside"]["rel_gap_unknown"] == 0.0
    line = region_gap_line(region, result["samples"])
    assert re.fullmatch(r"inside the changed region \(1 of 1 view sets show it\): swapped dates \+\d+\.\d\d%, unknown "
                        r"dates \+\d+\.\d\d%; outside: swapped dates \+0\.00%, unknown dates \+0\.00%", line)

    # no view set shows the region: nothing inside, and the line says so
    nan = np.full((2, 2), np.nan)
    whole = {"true": np.full((2, 2), 2.0), "swapped": np.full((2, 2), 3.0), "unknown": np.full((2, 2), 2.5)}
    summary = region_loss_summary({"inside": {variant: nan for variant in whole}, "outside": whole})
    assert summary["inside"] == {"sets": 0, "loss_true": None, "rel_gap_swapped": None, "rel_gap_unknown": None,
                                 "sample_rel_gap_swapped": [None, None], "sample_rel_gap_unknown": [None, None],
                                 "rel_gap_swapped_by_timestep": [None, None],
                                 "rel_gap_unknown_by_timestep": [None, None]}
    assert summary["outside"]["rel_gap_swapped"] == 0.5 and summary["outside"]["rel_gap_unknown"] == 0.25
    assert region_gap_line(summary, 2) == ("inside the changed region (0 of 2 view sets show it): no view set has "
                                           "it; outside: swapped dates +50.00%, unknown dates +25.00%")


# ----------------------------------------------------------------------------- N3: camera-balanced anchors
def anchor_probabilities_before(anchors, period, cfg):
    """``data.anchor_probabilities`` as it was before camera-balanced anchors (2026-11-20), word for word."""
    if len(anchors) == 0:
        return None
    if cfg.anchor_period_weights is None:
        return np.full(len(anchors), 1.0 / len(anchors))
    anchor_period = period[anchors]
    per_period = np.bincount(anchor_period, minlength=len(cfg.anchor_period_weights))
    weights = np.asarray(cfg.anchor_period_weights, dtype=np.float64)[anchor_period] / per_period[anchor_period]
    return weights / weights.sum() if weights.sum() > 0 else None


def sample_rows_before(dates, covis, cfg, rng, target_ok, ref_ok, anchors=None, n_ref=None):
    """``data.sample_rows`` as it was before camera-balanced anchors (2026-11-20), word for word."""
    dates = np.asarray(dates, dtype=np.float64)
    period = era_index(dates, cfg.period_edges)
    n_ref = cfg.n_ref if n_ref is None else n_ref
    n_targets = cfg.n_views - n_ref
    if anchors is None:
        anchors = eligible_anchors(dates, covis, cfg, target_ok, n_targets)
    probabilities = anchor_probabilities_before(anchors, period, cfg)
    if probabilities is None:
        return None
    anchor = int(rng.choice(anchors, p=probabilities))

    partners = np.flatnonzero(target_partners(dates, covis, cfg, target_ok, np.array([anchor]))[0])
    most_shared = partners[np.argsort(-covis[anchor, partners].astype(np.int64), kind="stable")[:cfg.top_k]]
    targets = np.concatenate([[anchor], rng.choice(most_shared, n_targets - 1, replace=False)])

    sees = covis[:, targets] >= cfg.min_shared
    candidate = np.asarray(ref_ok, dtype=bool) & (sees.sum(axis=1) >= cfg.min_targets_seen)
    candidate[targets] = False
    pool = np.flatnonzero(candidate)
    refs = None
    if rng.random() < cfg.cross_period_prob:
        refs = draw_references(pool[period[pool] != period[anchor]], sees, n_ref, rng)
    if refs is None:
        refs = draw_references(pool, sees, n_ref, rng)
    if refs is None:
        return None
    return np.concatenate([refs, targets]).astype(np.int64)


def draw_before(scene: MVGenScene, index: int, epoch: int = 0) -> tuple[np.ndarray, int]:
    """``MVGenScene.draw`` as it was before camera-balanced anchors: the same generator, the older sampler."""
    cfg = scene.config
    rng = np.random.default_rng((scene.seed, epoch, index))
    n_ref = draw_n_ref(cfg, rng)
    anchors = eligible_anchors(scene.dates, scene.covis, cfg, scene.target_ok, cfg.n_views - n_ref)
    while True:
        rows = sample_rows_before(scene.dates, scene.covis, cfg, rng, scene.target_ok, scene.ref_ok, anchors, n_ref)
        if rows is not None:
            return rows, n_ref


@pytest.mark.parametrize("settings", [
    dict(), dict(anchor_period_weights=(0.2, 0.3, 0.5)), dict(cross_period_prob=0.0), dict(cross_period_prob=1.0),
    dict(n_ref=2, n_targets=4, min_targets_seen=2), dict(anchor_period_weights=(0.0, 1.0, 1.0), top_k=6)])
def test_with_balance_off_the_sampler_draws_what_it_drew_before(settings):
    """The same seed, the same sample and the same random stream after it, as the sampler before the option."""
    dates, covis = two_sided_scene()
    cfg = ViewSamplingConfig(**settings)
    assert cfg.balance_cameras == 0.0
    ok = np.ones(60, dtype=bool)
    completed = 0
    for seed in range(300):
        rng, rng_before = np.random.default_rng(seed), np.random.default_rng(seed)
        rows, rows_before = sample_rows(dates, covis, cfg, rng, ok, ok), sample_rows_before(dates, covis, cfg,
                                                                                            rng_before, ok, ok)
        assert (rows is None) == (rows_before is None)
        if rows is not None:
            completed += 1
            assert np.array_equal(rows, rows_before) and rows.dtype == rows_before.dtype
        assert rng.random() == rng_before.random()              # as many random numbers were drawn
    assert completed > 200
    period = era_index(dates, cfg.period_edges)
    anchors = eligible_anchors(dates, covis, cfg, ok)
    assert np.array_equal(anchor_probabilities(anchors, period, cfg), anchor_probabilities_before(anchors, period, cfg))


@pytest.mark.parametrize("sampling", [ViewSamplingConfig(), SMALL, replace(SMALL, n_ref_choices=(1, 2, 2)),
                                      replace(SMALL, anchor_period_weights=(1.0, 2.0, 1.0))])
def test_with_balance_off_a_scene_draws_what_it_drew_before(folder, sampling):
    scene = MVGenScene(folder, sampling, resolution=SIZE, seed=3)
    assert scene.anchor_weights is None
    for held in ([], [1, 9, 17, 20]):
        scene.hold_out(held)
        for index in range(60):
            for epoch in (0, 2):
                rows, n_ref = scene.draw(index, epoch=epoch)
                rows_before, n_ref_before = draw_before(scene, index, epoch)
                assert np.array_equal(rows, rows_before) and n_ref == n_ref_before
                assert np.array_equal(scene.draw(index, epoch=epoch, balanced=False)[0], rows)   # changes nothing


def test_camera_features_and_nearest_views():
    c2ws = np.tile(np.eye(4), (3, 1, 1))
    c2ws[:, 0, 3] = [0.0, 2.0, 4.0]                             # centres on a line; all look along z
    c2ws[2, :3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]        # the third looks along x
    features = camera_features(c2ws)
    # the median centre is (2, 0, 0) and the centres are 2, 0 and 2 away from it: the unit is 2
    assert features.tolist() == [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [2, 0, 0, 1, 0, 0]]
    assert camera_features(10 * c2ws)[:, :3].tolist() == features[:, :3].tolist()       # COLMAP's unit drops out
    assert camera_features(np.tile(np.eye(4), (2, 1, 1)))[:, :3].tolist() == [[0, 0, 0]] * 2    # no spread: as it is

    line = np.array([[0.0], [1.0], [3.0], [7.0]])
    assert nearest_views(line, 2).tolist() == [[1, 2], [0, 2], [1, 0], [2, 1]]
    assert nearest_views(line, 9).tolist() == [[1, 2, 3], [0, 2, 3], [1, 0, 3], [2, 1, 0]]       # all the others
    assert nearest_views(np.array([[0.0], [1.0], [2.0]]), 1).tolist() == [[1], [0], [1]]         # a tie: lower index
    assert nearest_views(line[:1], 3).shape == (1, 0)
    # direction counts like position: next to each other but looking apart is not near
    assert nearest_views(features, 1).tolist() == [[1], [0], [1]]
    assert np.linalg.norm(features[2] - features[1]) > np.linalg.norm(features[0] - features[1])


def test_camera_balance_weights_and_the_vote_by_hand():
    neighbours = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    period = np.array([0, 0, 0, 1])
    # the first three have 2 of 3 neighbours of their period; the last one has none: 1 / (1 / 3), capped
    assert camera_balance_weights(neighbours, period, 10.0).tolist() == pytest.approx([1.5, 1.5, 1.5, 3.0])
    assert camera_balance_weights(neighbours, period, 2.0).tolist() == pytest.approx([1.5, 1.5, 1.5, 2.0])
    assert camera_balance_weights(neighbours, period, 1.0).tolist() == [1.0] * 4
    assert camera_balance_weights(neighbours, np.zeros(4, dtype=int), 10.0).tolist() == [1.0] * 4    # one period
    assert camera_balance_weights(np.zeros((2, 0), dtype=int), np.array([0, 1]), 5.0).tolist() == [1.0, 1.0]

    neighbours = np.array([[1, 2], [0, 2], [0, 1], [4, 0], [3, 0]])
    period = np.array([0, 0, 0, 1, 1])
    # the two views of period 1 each have one neighbour of either period: the tie goes to period 0, wrongly
    assert period_vote_accuracy(neighbours, period) == (pytest.approx(3 / 5), pytest.approx(3 / 5))
    # with their weight of 3 their neighbour of period 1 outvotes the other, and they count three times
    right, guess = period_vote_accuracy(neighbours, period, np.array([1.0, 1.0, 1.0, 3.0, 3.0]))
    assert (right, guess) == (pytest.approx(1.0), pytest.approx(6 / 9))


def test_anchor_probabilities_follow_the_views_own_weights():
    anchors, period = np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1, 1])
    weights = np.array([1.0, 3.0, 1.0, 1.0, 100.0])             # the last view is no anchor
    assert anchor_probabilities(anchors, period, ViewSamplingConfig(), weights).tolist() == \
        pytest.approx([1 / 6, 3 / 6, 1 / 6, 1 / 6])
    # with period weights the periods keep their probabilities; inside a period the views' weights decide
    by_period = ViewSamplingConfig(period_edges=(2000.0,), anchor_period_weights=(0.2, 0.8))
    assert anchor_probabilities(anchors, period, by_period, weights).tolist() == pytest.approx([0.05, 0.15, 0.4, 0.4])
    assert anchor_probabilities(anchors, period, by_period).tolist() == pytest.approx([0.1, 0.1, 0.4, 0.4])
    assert anchor_probabilities(np.array([], dtype=int), period, by_period, weights) is None
    for bad in (0.5, -1.0):
        with pytest.raises(ValueError, match="balance_cameras must be 0"):
            ViewSamplingConfig(balance_cameras=bad)
    assert ViewSamplingConfig(balance_cameras=1.0).balance_cameras == 1.0


def write_interleaved(root: Path) -> tuple[Path, np.ndarray]:
    """The tiny folder with its cameras moved along x, and their positions ``[N]``.

    The old views (rows 0-7) stand by themselves around x = 1; the modern views (8-15) and four views from
    after the fire (16-19) share the stretch from 10 to 12; the other four from after the fire (20-23) stand
    by themselves around x = 31. Every camera looks along z.
    """
    folder = write_folder(root)
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(0, 2, 8), rng.uniform(10, 12, 8), rng.uniform(10, 12, 4), rng.uniform(30, 32, 4)])
    views = pd.read_parquet(folder / "views.parquet")
    views["tx"] = -x                                            # no rotation: the camera centre is -t
    views.to_parquet(folder / "views.parquet", index=False)
    return folder, x


def weights_by_hand(x: np.ndarray, period: np.ndarray, views: np.ndarray, k: int, cap: float) -> np.ndarray:
    """The camera-balance weights of ``views`` among themselves, with plain loops; 1 for every other view."""
    weights = np.ones(len(x))
    for i in views:
        nearest = sorted((abs(x[j] - x[i]), j) for j in views if j != i)[:k]
        own = sum(period[j] == period[i] for _, j in nearest) / k
        weights[i] = min(cap, 1 / max(own, 1 / k))
    return weights


def test_balanced_anchors_draw_views_in_mixed_neighbourhoods_more_often(tmp_path, monkeypatch, capsys):
    folder, x = write_interleaved(tmp_path / "scene")
    monkeypatch.setattr(data_module, "BALANCE_NEIGHBOURS", 3)   # of 24 views; the real scene judges by 15 of 1,000
    assert BALANCE_NEIGHBOURS == 15
    capsys.readouterr()
    plain = MVGenScene(folder, SMALL, resolution=SIZE)
    assert plain.anchor_weights is None and "camera" not in capsys.readouterr().out
    scene = MVGenScene(folder, replace(SMALL, balance_cameras=2.5), resolution=SIZE)
    printed = capsys.readouterr().out
    period, everyone = scene.period, np.arange(N)
    expected = weights_by_hand(x, period, everyone, k=3, cap=2.5)
    assert scene.anchor_weights.tolist() == pytest.approx(expected.tolist())
    # views among their own period keep 1; where the two periods share the ground the weights go up to the cap
    assert expected[:8].tolist() == [1.0] * 8 and expected[20:].tolist() == [1.0] * 4
    assert expected[8:20].max() == 2.5 and (expected[16:20] > 1).all() and set(expected) <= {1.0, 1.5, 2.5}
    assert len(scene.anchors) == N

    # the anchors are drawn in proportion: exactly in the probabilities, and in what is drawn
    probabilities = anchor_probabilities(scene.anchors, period, scene.config, scene.anchor_weights)
    assert probabilities.tolist() == pytest.approx((expected / expected.sum()).tolist())
    draws, balanced_share = 6000, expected / expected.sum()
    for drawn_from, share in ((scene, balanced_share), (plain, np.full(N, 1 / N))):
        anchors = np.array([drawn_from.draw(index)[0][2] for index in range(draws)])     # the first target
        counted = np.bincount(anchors, minlength=N) / draws
        bound = 4 * np.sqrt(share.max() * (1 - share.max()) / draws)
        assert np.abs(counted - share).max() < bound < np.abs(balanced_share - 1 / N).max() / 2

    # the probes' view sets do not follow the balance: a balanced run is measured on the plain run's sets
    assert all(np.array_equal(scene.draw(i, balanced=False)[0], plain.draw(i)[0]) for i in range(200))
    assert any(not np.array_equal(scene.draw(i)[0], plain.draw(i)[0]) for i in range(200))
    assert all(np.array_equal(a, b) for a, b in zip(scene_view_sets(scene, 6), scene_view_sets(plain, 6)))

    # what is printed under the period table, and the numbers behind it
    summary = scene.camera_balance_summary()
    assert "[scene] camera-balanced anchors (largest weight 2.5): the weights of the 24 anchors are 1.00 to 2.50" \
        in printed and "an anchor's period predicted from its camera by a vote of its 3 nearest anchors: " in printed
    assert "the other targets of a sample are not weighted" in printed
    assert printed.count("camera-balanced anchors") == 1
    table = scene.period_table()
    shares = [expected[period == p].sum() / expected.sum() for p in range(3)]
    assert table["anchor_share"].tolist() == pytest.approx(shares) and summary["anchor_share"] == pytest.approx(shares)
    assert summary["anchor_share_unbalanced"] == pytest.approx([1 / 3] * 3)
    assert plain.period_table()["anchor_share"].tolist() == pytest.approx([1 / 3] * 3)
    assert shares[0] < 1 / 3 < shares[2]                        # the old views stand alone; the others are mixed
    assert (summary["cap"], summary["neighbours"], summary["anchors"]) == (2.5, 3, N)
    assert summary["weight_max"] == 2.5 and summary["weight_min"] == 1.0
    assert summary["share_above_1"] == pytest.approx(np.mean(expected > 1))
    # the vote is among the anchors as they are drawn; here every view is an anchor, drawn in proportion to its weight
    neighbours = nearest_views(camera_features(scene.c2ws), 3)
    assert (summary["vote_right"], summary["vote_guess"]) == pytest.approx(period_vote_accuracy(neighbours, period))
    assert (summary["weighted_vote_right"], summary["weighted_vote_guess"]) == \
        pytest.approx(period_vote_accuracy(neighbours, period, expected))
    assert summary["vote_guess"] == pytest.approx(1 / 3)
    assert summary["weighted_vote_guess"] == pytest.approx(max(shares))
    assert summary["weighted_vote_right"] < summary["vote_right"] < 1       # the camera says less under the weights
    assert json.loads(json.dumps(summary)) == summary
    with pytest.raises(ValueError, match="not camera-balanced"):
        plain.camera_balance_summary()

    # the weights are those of the views that train: held-out views are nobody's neighbours and are never drawn
    capsys.readouterr()
    scene.hold_out([16, 17, 18, 19])                            # the after-fire views that stood among the modern
    training = np.setdiff1d(everyone, [16, 17, 18, 19])
    assert scene.anchor_weights.tolist() == pytest.approx(weights_by_hand(x, period, training, 3, 2.5).tolist())
    assert scene.anchor_weights.tolist() == [1.0] * N           # now every period stands by itself
    assert "camera-balanced anchors (largest weight 2.5): the weights of the 20 anchors are 1.00 to 1.00" \
        in capsys.readouterr().out
    assert scene.camera_balance_summary()["vote_right"] == 1.0

    # with period weights the periods keep their shares, and the balance acts inside each period
    weighted = MVGenScene(folder, replace(SMALL, balance_cameras=2.5, anchor_period_weights=(1.0, 1.0, 2.0)),
                          resolution=SIZE)
    assert weighted.period_table()["anchor_share"].tolist() == pytest.approx([0.25, 0.25, 0.5])
    probabilities = anchor_probabilities(weighted.anchors, period, weighted.config, weighted.anchor_weights)
    assert probabilities[16:].tolist() == pytest.approx((0.5 * expected[16:] / expected[16:].sum()).tolist())


def test_the_balance_summary_is_about_the_anchors_as_they_are_drawn(tmp_path, monkeypatch, capsys):
    """A training view that cannot anchor a sample has a weight, but nothing is drawn by it: it neither votes nor
    counts in the summary that a scene prints."""
    folder, x = write_interleaved(tmp_path / "scene")
    monkeypatch.setattr(data_module, "BALANCE_NEIGHBOURS", 3)
    # The after-fire views among the modern ones (rows 16-19) share no point with any view: they still train
    # (their cameras are the modern views' neighbours), but they can neither anchor a sample nor be drawn into one.
    covis = np.load(folder / "covisibility.npy")
    covis[16:20, :] = covis[:, 16:20] = 0
    np.save(folder / "covisibility.npy", covis)
    capsys.readouterr()
    scene = MVGenScene(folder, replace(SMALL, balance_cameras=2.5), resolution=SIZE)
    printed = capsys.readouterr().out
    anchors = np.r_[0:16, 20:24]
    assert scene.anchors.tolist() == anchors.tolist() and scene.target_ok.all()
    expected = weights_by_hand(x, scene.period, np.arange(N), k=3, cap=2.5)
    assert scene.anchor_weights.tolist() == pytest.approx(expected.tolist())
    assert expected[8:16].max() > 1 and expected[16:20].min() > 1      # judged by all training views, as before

    # among the anchors every period stands by itself: the camera tells the period with and without the weights
    summary = scene.camera_balance_summary()
    assert summary["anchors"] == 20 and summary["neighbours"] == 3
    assert summary["vote_right"] == 1.0 and summary["weighted_vote_right"] == 1.0
    assert summary["vote_guess"] == pytest.approx(8 / 20)
    drawn = expected[anchors] / expected[anchors].sum()
    shares = [drawn[scene.period[anchors] == p].sum() for p in range(3)]
    assert summary["anchor_share"] == pytest.approx(shares)
    assert summary["weighted_vote_guess"] == pytest.approx(max(shares))
    assert shares[1] > 8 / 20                                   # the modern anchors are the ones with weights above 1
    assert "by a vote of its 3 nearest anchors: 100.0% right while the anchors are drawn without the weights (40.0% " \
        "by always guessing the period drawn most often), 100.0% with them" in printed
    # counting all training views by their weights, as if the four were drawn, would call the cameras mixed
    everyone = nearest_views(camera_features(scene.c2ws), 3)
    assert period_vote_accuracy(everyone, scene.period, expected)[0] < 0.9


def test_a_balanced_run_trains_on_other_samples_and_is_probed_on_the_same_view_sets(tmp_path, monkeypatch, capsys):
    folder, _ = write_interleaved(tmp_path / "scene")
    monkeypatch.setattr(data_module, "BALANCE_NEIGHBOURS", 3)
    balanced = replace(SMALL, balance_cameras=2.5)
    settings = dict(steps=4, probe_every=4, latest_every=2,
                    holdout=HoldoutConfig(fraction=0.5, seed=1, era_edges=EDGES))
    drawn, probed = {}, {}
    for name, sampling in (("plain", SMALL), ("balanced", balanced)):
        with monkeypatch.context() as patch:
            drawn[name] = watch_training(patch)
            trainer = Trainer(config(folder, tmp_path / name, sampling=sampling, **settings), model=tiny_model())
            trainer.train()
        probed[name] = {probe: trainer.loss_probes[probe].rows for probe in trainer.loss_probes}
    printed = capsys.readouterr().out
    assert drawn["plain"] != drawn["balanced"] and len(drawn["balanced"]) == 8
    assert probed["plain"] == probed["balanced"] and list(probed["plain"]) == ["train", "heldout"]
    assert printed.count("camera-balanced anchors (largest weight 2.5)") == 2   # at set-up, and under the hold-out
    saved = json.loads((tmp_path / "balanced" / "config.json").read_text())
    assert saved["sampling"]["balance_cameras"] == 2.5
    assert json.loads((tmp_path / "plain" / "config.json").read_text())["sampling"]["balance_cameras"] == 0.0
    assert run_config(tmp_path / "balanced").sampling == balanced

    # the cap is the run's: another one, or none, is another run
    latest = checkpoint_to_resume(tmp_path / "balanced", "auto")
    for sampling, named in ((replace(SMALL, balance_cameras=5.0), "sampling.balance_cameras: 2.5 -> 5.0"),
                            (SMALL, "sampling.balance_cameras: 2.5 -> 0.0")):
        with pytest.raises(ValueError, match=named):
            Trainer(config(folder, tmp_path / "balanced", sampling=sampling, **settings), resume=latest)
    with pytest.raises(ValueError, match="sampling.balance_cameras: 0.0 -> 2.5"):
        Trainer(config(folder, tmp_path / "plain", sampling=balanced, **settings),
                resume=checkpoint_to_resume(tmp_path / "plain", "auto"))


def test_a_balanced_run_resumes_on_its_own_stream_of_samples(tmp_path, monkeypatch):
    folder, _ = write_interleaved(tmp_path / "scene")
    monkeypatch.setattr(data_module, "BALANCE_NEIGHBOURS", 3)
    settings = dict(steps=4, latest_every=2, ema_decay=0.9, sampling=replace(SMALL, balance_cameras=2.5))
    with monkeypatch.context() as patch:
        samples_whole = watch_training(patch)
        whole = Trainer(config(folder, tmp_path / "whole", **settings), model=tiny_model())
        whole.train()
    cfg = config(folder, tmp_path / "cut", **settings)
    with monkeypatch.context() as patch:
        samples_cut = watch_training(patch, stop_after=5)
        with pytest.raises(RuntimeError, match="job killed"):
            Trainer(cfg, model=tiny_model()).train()
    with monkeypatch.context() as patch:
        samples_resumed = watch_training(patch)
        resumed = Trainer(cfg, model=tiny_model(), resume=checkpoint_to_resume(cfg.run_dir, "auto"))
        resumed.train()
    assert samples_cut[:4] + samples_resumed == samples_whole
    assert torch.equal(flat_weights(resumed.model), flat_weights(whole.model))


# ----------------------------------------------------------------------------- N4: starting from another run
def test_a_run_starts_from_the_raw_weights_of_another_runs_latest(folder, tmp_path, capsys):
    source = finished_run(folder, tmp_path / "source")
    latest = Path(source.run_dir) / "checkpoints" / "latest.pt"
    saved = torch.load(latest, weights_only=False)
    raw, averaged = flat_state(saved["raw_trainable"]), flat_state(saved["trainable"])
    assert saved["weights"] == "ema" and not torch.equal(raw, averaged)
    capsys.readouterr()

    cfg = config(folder, tmp_path / "next", steps=2, warmup_steps=2, ema_decay=0.9, latest_every=1, save_every=2,
                 init_from=latest)
    assert cfg.init_from == str(latest)                         # a path is kept as text, as config.json has it
    trainer = Trainer(cfg, model=tiny_model())
    printed = capsys.readouterr().out
    assert f"[train] starting at step 0 from the raw (non-EMA) weights of {latest} (step 3 of its run)" in printed
    assert "note:" not in printed                               # the same dataset, scale and hold-out
    assert trainer.step == 0 and not trainer.started            # a new run: step 0 is still to be evaluated
    assert torch.equal(flat_weights(trainer.model), raw)
    assert trainer.ema.updates == 0                             # the average starts from the loaded weights
    assert all(torch.equal(shadow, param) for shadow, param in zip(trainer.ema.shadow, trainer.params))
    assert not trainer.optimizer.state_dict()["state"] and len(trainer.grad_norm_history) == 0     # nothing else
    record = {"checkpoint": str(latest), "step": 3, "weights": "raw"}
    assert trainer.init_weights == record
    run_dir = Path(cfg.run_dir)
    written = json.loads((run_dir / "config.json").read_text())
    assert written["init_weights"] == record and written["init_from"] == str(latest)
    assert set(written) == set(asdict(cfg)) | set(CONFIG_RECORDS)
    assert run_config(run_dir) == cfg                           # the record does not get in the way of reading it

    trainer.train()
    log = [r for r in read_log(run_dir) if "probe" not in r]
    assert [r["step"] for r in log] == [1, 2] and log[0]["lr_date"] == pytest.approx(0.5 * cfg.lr_date)    # warm-up
    assert not torch.equal(flat_weights(trainer.model), raw)
    for name in ("latest.pt", "step_000002.pt"):                # every checkpoint says where the run started
        checkpoint = torch.load(run_dir / "checkpoints" / name, weights_only=False)
        assert checkpoint["init_weights"] == record and checkpoint["step"] == 2
        assert checkpoint["train_config"]["init_from"] == str(latest)
    assert "init_weights" not in saved and "init_weights" not in json.loads(
        (Path(source.run_dir) / "config.json").read_text())     # a run of its own start records nothing

    # a step_XXXXXX.pt holds only what is evaluated: the EMA, if the run kept one; otherwise the raw weights
    capsys.readouterr()
    step_file = Path(source.run_dir) / "checkpoints" / "step_000003.pt"
    from_average = Trainer(config(folder, tmp_path / "average", init_from=step_file), model=tiny_model())
    assert torch.equal(flat_weights(from_average.model), averaged)
    assert from_average.init_weights == {"checkpoint": str(step_file), "step": 3, "weights": "ema"}
    assert "from the EMA weights (the file holds no raw ones) of" in capsys.readouterr().out
    assert from_average.ema is None                             # this run keeps no average itself
    plain = finished_run(folder, tmp_path / "plain", ema_decay=0.0)
    for name in ("latest.pt", "step_000003.pt"):
        path = Path(plain.run_dir) / "checkpoints" / name
        started = Trainer(config(folder, tmp_path / f"plain_{name}", init_from=path), model=tiny_model())
        assert started.init_weights["weights"] == "raw"
        assert torch.equal(flat_weights(started.model), flat_state(torch.load(path, weights_only=False)["trainable"]))


def test_a_run_started_from_a_checkpoint_and_not_trained_reproduces_its_probe_numbers(folder, tmp_path):
    numbers = ("loss_true", "loss_swapped", "loss_unknown", "rel_gap_swapped", "rel_gap_unknown",
               "loss_true_by_timestep", "sample_rel_gap_swapped", "date_rms", "date_temb_ratio")

    def last_probe(run_dir) -> dict:
        return [record for record in read_log(run_dir) if record.get("probe") == "date_loss:train"][-1]

    def first_probe_of_a_run_from(checkpoint: Path, name: str, ema_decay: float) -> dict:
        trainer = Trainer(config(folder, tmp_path / name, probe_every=3, ema_decay=ema_decay, init_from=checkpoint),
                          model=tiny_model())
        trainer.run_loss_probes()                               # step 0, before any training
        (record,) = read_log(trainer.run_dir)
        assert record["step"] == 0 and record["change"] == 0.0  # its own baseline: the loaded model
        return record

    # a run without an EMA: its probes measured the raw weights, which its latest.pt holds
    source = finished_run(folder, tmp_path / "raw", ema_decay=0.0)
    theirs = last_probe(source.run_dir)
    assert theirs["step"] == 3 and theirs["change"] != 0 and theirs["rel_gap_swapped"] != 0
    mine = first_probe_of_a_run_from(Path(source.run_dir) / "checkpoints" / "latest.pt", "from_raw", 0.0)
    assert {name: mine[name] for name in numbers} == {name: theirs[name] for name in numbers}

    # a run with an EMA: its probes measured the average, which its step file holds; its latest.pt gives the
    # raw weights, which are another model
    source = finished_run(folder, tmp_path / "averaged")
    theirs = last_probe(source.run_dir)
    checkpoints = Path(source.run_dir) / "checkpoints"
    mine = first_probe_of_a_run_from(checkpoints / "step_000003.pt", "from_average", 0.9)
    assert {name: mine[name] for name in numbers} == {name: theirs[name] for name in numbers}
    raw = first_probe_of_a_run_from(checkpoints / "latest.pt", "from_latest", 0.9)
    assert raw["loss_true"] != theirs["loss_true"]


def test_init_from_refuses_other_adapters_before_anything_is_loaded_or_written(folder, tmp_path, monkeypatch):
    source = finished_run(folder, tmp_path / "source")          # date 4 x 16, LoRA of rank 2 on the default layers
    targets = sorted(source.finetune.lora_targets)
    latest = Path(source.run_dir) / "checkpoints" / "latest.pt"
    monkeypatch.setattr(trainer_module, "load_mvgen", lambda model_dir: pytest.fail("the model was loaded"))
    base = dict(date_freqs=4, date_hidden=16, lora_rank=2)
    for changes, named in ((dict(lora_rank=4), "lora_rank: 2 -> 4"), (dict(lora_rank=0), "lora: True -> False"),
                           (dict(date_hidden=32), "date_hidden: 16 -> 32"), (dict(date_freqs=8), "date_freqs: 4 -> 8"),
                           (dict(lora_alpha=4.0), "lora_alpha: 2.0 -> 4.0"), (dict(date=False), "date: True -> False"),
                           (dict(lora_targets=("attn1.to_q", "time_emb_proj")),
                            f"lora_targets: {targets} -> ['attn1.to_q', 'time_emb_proj']")):
        with pytest.raises(ValueError, match=f"adapters are not this run's.*{re.escape(named)}"):
            Trainer(config(folder, tmp_path / "next", finetune=FinetuneConfig(**{**base, **changes}), init_from=latest))
    with pytest.raises(ValueError, match="init_from: there is no checkpoint"):
        Trainer(config(folder, tmp_path / "next", init_from=tmp_path / "missing.pt"))
    assert not (tmp_path / "next").exists()

    # what does not change the weights or what they compute is no difference
    same = FinetuneConfig(**base, lora_alpha=2.0, grad_checkpointing=False)        # alpha written out; no checkpointing
    assert adapter_settings(same) == adapter_settings(source.finetune) == {
        "date": True, "lora": True, "date_freqs": 4, "date_hidden": 16, "lora_rank": 2, "lora_alpha": 2.0,
        "lora_targets": targets}
    assert adapter_settings(replace(same, lora_targets=tuple(reversed(targets)))) == adapter_settings(same)
    assert adapter_settings(FinetuneConfig(date=False, lora_rank=0)) == {"date": False, "lora": False}
    started = Trainer(config(folder, tmp_path / "same", finetune=same, init_from=latest), model=tiny_model())
    assert started.init_weights["weights"] == "raw"


def test_a_run_started_from_another_run_resumes_like_any_run(folder, tmp_path, monkeypatch, capsys):
    source = finished_run(folder, tmp_path / "source")
    latest = Path(source.run_dir) / "checkpoints" / "latest.pt"
    record = {"checkpoint": str(latest), "step": 3, "weights": "raw"}
    settings = dict(steps=4, ema_decay=0.9, latest_every=2, init_from=latest)
    whole = Trainer(config(folder, tmp_path / "whole", **settings), model=tiny_model())
    whole.train()
    cfg = config(folder, tmp_path / "cut", **settings)
    with monkeypatch.context() as patch:
        watch_training(patch, stop_after=5)                     # the job dies in its third step
        with pytest.raises(RuntimeError, match="job killed"):
            Trainer(cfg, model=tiny_model()).train()
    own_latest = checkpoint_to_resume(cfg.run_dir, "auto")
    latest.unlink()                                             # the other run has moved on: its file is gone
    capsys.readouterr()

    resumed = Trainer(cfg, model=tiny_model(), resume=own_latest)       # init_from is not looked at again
    printed = capsys.readouterr().out
    assert "starting at step 0" not in printed and f"resumed from {own_latest} at step 2" in printed
    assert resumed.step == 2 and resumed.init_weights == record
    resumed.train()
    assert torch.equal(flat_weights(resumed.model), flat_weights(whole.model))          # bit for bit
    assert all(torch.equal(a, b) for a, b in zip(resumed.ema.shadow, whole.ema.shadow))
    assert [r["loss"] for r in measured(cfg.run_dir)] == [r["loss"] for r in measured(tmp_path / "whole")]
    assert torch.load(own_latest, weights_only=False)["init_weights"] == record
    assert json.loads((Path(cfg.run_dir) / "config.json").read_text())["init_weights"] == record

    # where a run started is one of its settings: another start, or none, is another run
    for changed, named in ((None, f"init_from: '{latest}' -> None"), (tmp_path / "x.pt", "init_from: ")):
        with pytest.raises(ValueError, match=re.escape(named)):
            Trainer(config(folder, tmp_path / "cut", **{**settings, "init_from": changed}), resume=own_latest)
    # a fresh launch without the file cannot start
    with pytest.raises(ValueError, match="init_from: there is no checkpoint"):
        Trainer(config(folder, tmp_path / "new", **settings), model=tiny_model())


def test_starting_from_a_run_with_another_hold_out_is_pointed_out(folder, tmp_path, capsys):
    base = TrainConfig(dataset="d", run_dir="r")
    assert start_differences(asdict(base), asdict(replace(base, run_dir="s", steps=5, seed=3, lr_date=1e-3))) == []
    other = replace(base, dataset="e", scale=0.5, holdout=HoldoutConfig(fraction=0.1, seed=2),
                    render_rows=((0, 1, 2, 3),))
    assert start_differences(asdict(base), asdict(other)) == [
        "dataset: 'd' -> 'e'", "holdout.fraction: 0.0 -> 0.1", "holdout.seed: 0 -> 2", "scale: 'scene' -> 0.5"]
    held = replace(other, holdout=HoldoutConfig(probe_targets=True))    # the renders' targets are held out
    assert "render_rows: [] -> [[0, 1, 2, 3]]" in start_differences(asdict(base), asdict(held))
    assert start_differences({}, asdict(base))[0] == "dataset: '(not recorded)' -> 'd'"

    source = finished_run(folder, tmp_path / "source", holdout=HoldoutConfig(fraction=0.25, era_edges=EDGES))
    latest = Path(source.run_dir) / "checkpoints" / "latest.pt"
    capsys.readouterr()
    Trainer(config(folder, tmp_path / "next", init_from=latest), model=tiny_model())      # holds nothing out
    printed = capsys.readouterr().out
    assert f"[train] note: {latest} comes from a run with another dataset, scale or hold-out (holdout.era_edges: " \
        in printed and "holdout.fraction: 0.25 -> 0.0" in printed
    assert "may be views those weights were trained on" in printed
    Trainer(config(folder, tmp_path / "same", init_from=latest, holdout=source.holdout), model=tiny_model())
    assert "note:" not in capsys.readouterr().out


# ----------------------------------------------------------------------------- N6: the command lines
def test_the_train_script_takes_the_three_options(tmp_path, capsys):
    script = load_script("train")
    cfg = script.build_config(script.parse_args([
        "--run-name", "a", "--region-weight", "10", "--balance-cameras", "5",
        "--init-from", "/runs/gate3_keep/checkpoints/latest.pt"]))
    assert cfg.loss == LossConfig(region_weight=10.0) and cfg.sampling == ViewSamplingConfig(balance_cameras=5.0)
    assert cfg.init_from == "/runs/gate3_keep/checkpoints/latest.pt"
    default = script.build_config(script.parse_args(["--run-name", "a"]))
    assert (default.loss.region_weight, default.sampling.balance_cameras, default.init_from) == (1.0, 0.0, None)
    with pytest.raises(SystemExit):
        script.parse_args(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    for option in ("--region-weight", "--balance-cameras", "--init-from", "changed_regions.npz",
                   "the largest weight an anchor may get", "start this NEW run at step 0"):
        assert option in text, option
    with pytest.raises(SystemExit):
        load_script("prepare_regions").parse_args(["--help"])
    assert "--min-share" in capsys.readouterr().out


def test_the_scripts_run_with_the_options_and_read_older_and_newer_runs(folder, tmp_path, monkeypatch, capsys):
    """train.py with the three options; then render.py and eval_dates.py on a checkpoint of that run and on one of
    a run as the older code saved it."""
    monkeypatch.setattr(trainer_module, "load_mvgen", lambda model_dir: tiny_model())
    write_regions(folder)
    train, render, evaluate = load_script("train"), load_script("render"), load_script("eval_dates")
    common = ["--runs-root", str(tmp_path), "--dataset", str(folder), "--resolution", str(SIZE), "--grad-accum", "2",
              "--lora-rank", "2", "--lora-targets", "attn1.to_q", "--date-freqs", "4", "--date-hidden", "16",
              "--precision", "fp32", "--ema-decay", "0.9", "--num-workers", "0", "--log-every", "1",
              "--probe-every", "2", "--probe-timesteps", "200", "800", "--probe-samples", "2", "--render-every", "0",
              "--latest-every", "2", "--save-every", "2", "--warmup", "0", "--lr-schedule", "constant",
              "--n-ref", "2", "--n-targets", "3", "--min-targets-seen", "2", "--top-k", "4",
              "--holdout-fraction", "0.5", "--holdout-seed", "1", "--resume", "auto"]
    train.main(["--run-name", "older", *common, "--steps", "2"])
    older = tmp_path / "older"
    as_saved_by_the_older_code(older)
    capsys.readouterr()

    options = ["--region-weight", "10", "--balance-cameras", "3", "--init-from", str(older / "checkpoints/latest.pt")]
    train.main(["--run-name", "newer", *common, *options, "--steps", "2"])
    printed = capsys.readouterr().out
    for said in ("starting at step 0 from the raw (non-EMA) weights of", "camera-balanced anchors (largest weight 3)",
                 "the loss counts the changed region 10 times", "inside the changed region (",
                 "(unweighted "):
        assert said in printed, said
    newer = tmp_path / "newer"
    train.main(["--run-name", "newer", *common, *options, "--steps", "4"])      # the same line again: it resumes
    assert "resumed from" in capsys.readouterr().out
    assert [r["step"] for r in read_log(newer) if "probe" not in r] == [1, 2, 3, 4]
    saved = json.loads((newer / "config.json").read_text())
    assert saved["init_weights"]["weights"] == "raw" and saved["loss"]["region_weight"] == 10.0

    for run_dir in (older, newer):                              # what eval_dates.py asks of a run folder
        cfg = run_config(run_dir)
        checkpoint = torch.load(run_dir / "checkpoints" / "latest.pt", weights_only=False)
        check_run_records(open_scene(cfg), run_dir, checkpoint)
    assert run_config(older).loss.region_weight == 1.0 and run_config(newer).sampling.balance_cameras == 3.0

    monkeypatch.setattr(mvgen_model, "load_mvgen", lambda model_dir, device: tiny_model())
    real_load = render.load_finetuned
    monkeypatch.setattr(render, "load_finetuned", lambda *args: real_load(*args, device="cpu"))
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([[8, 9, 16, 17, 18]]))
    stand_in_models(evaluate, monkeypatch)
    for name, step in (("older", 2), ("newer", 4)):
        out = tmp_path / f"render_{name}"
        render.main(["--checkpoint", str(tmp_path / name / "checkpoints" / f"step_{step:06d}.pt"), "--rows-file",
                     str(rows), "--out", str(out), "--dates", "true", "swapped", "--seeds", "0", "--steps", "1",
                     "--thumb", "16"])
        assert len(json.loads((out / "results.json").read_text())[0]["renders"]) == 2
        out = tmp_path / f"eval_{name}"
        evaluate.main(["--run", str(tmp_path / name), "--checkpoint", str(step), "--keys-file", "none", "--heldout",
                       "1", "--no-renders", "--noise-seeds", "0", "--min-mask-pixels", "4", "--out", str(out)])
        results = json.loads((out / "results.json").read_text())
        assert results["model"]["step"] == step and results["settings"]["loss"]["region_weight"] == (
            1.0 if name == "older" else 10.0)
