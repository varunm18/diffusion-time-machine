"""CPU tests for the date evaluation by area (``timemachine.mvgen.evaluation``, ``scripts/mvgen/eval_dates.py``).

Nothing of the released model is loaded. The stand-ins are those of ``tests/test_mvgen_trainer.py``:
its tiny UNet with the real date conditioner, its sampling pipeline, its folder of 24 tiny views
(rows 0-7 old, 8-15 modern, 16-23 after the fire) and runs of the real ``Trainer`` on them. Two more
are defined here:

* ``FakeRegions`` stands in for ``regions.ChangedRegions`` (which has its own tests): the region of a
  view is the top-left quarter of its frame, which is whole latent cells, or a few pixels of its
  first image row;
* ``RegionDatedUNet`` is a model whose output depends on the date in that quarter and nowhere else:
  the case the evaluation exists to detect.

Run from MVGenMaster's checkout, in its environment (see ``timemachine/mvgen/__init__.py``).
"""
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

pytest.importorskip("my_diffusers", reason="runs in MVGenMaster's environment, from its checkout")

# The stand-ins, and the fixtures ``fake_conditioning`` (autouse), ``folder`` and ``scene``, which pytest finds
# here by name. Importing that module also lets the ``src`` package find MVGenMaster's modules, so it comes
# before anything of ``timemachine.mvgen``.
from tests.test_mvgen_trainer import (  # noqa: E402, F401
    AFTER_FROM_MODERN, EDGES, LATENT, MODERN_FROM_MODERN, N, SCHEDULER, SIZE, SWAP_YEARS, TinyMVGen, adapted_model,
    config, fake_conditioning, flat_weights, folder, get_3d_priors, get_camera_embedding, is_zero, load_script,
    record_dates, run_files, scene, small_mvgenmaster, tiny_model, write_folder,
)

from my_diffusers.schedulers import DDPMScheduler  # noqa: E402

from timemachine.data.holdout import HoldoutConfig  # noqa: E402
from timemachine.mvgen import model as mvgen_model  # noqa: E402
from timemachine.mvgen import step  # noqa: E402
from timemachine.mvgen.data import MVGenScene, ViewSamplingConfig  # noqa: E402
from timemachine.mvgen.evaluation import (  # noqa: E402
    AREAS, HELDOUT_SOURCE, RENDER_DATES, TargetMasks, ViewSet, area_means, check_run_records, checkpoint_path,
    collect_view_sets, crosses_edge, describe_set, format_table, groups_of, heldout_sets, loss_noise, mean_and_error,
    measure_losses, open_scene, priors_differ, render_at, render_set, run_config, run_swap_years, score_renders,
    summarise_groups, summarise_set_losses, summarise_set_renders, target_masks, to_psnr,
)
from timemachine.mvgen.model import FinetuneConfig, prepare_for_finetuning  # noqa: E402
from timemachine.mvgen.probes import DateLossProbe, RenderProbe, heldout_view_sets, psnr  # noqa: E402
from timemachine.mvgen.regions import latent_weights  # noqa: E402
from timemachine.mvgen.step import (  # noqa: E402
    LossConfig, build_inputs, cross_date_drops, encode_images, v_prediction_error_map, v_prediction_loss,
)
from timemachine.mvgen.trainer import Trainer  # noqa: E402

FIRE = EDGES[-1]
SMALL = ViewSamplingConfig(n_ref=2, n_targets=3, min_targets_seen=2, top_k=4)      # view sets of 2 + 3 views
FP32 = LossConfig(precision="fp32")


# ----------------------------------------------------------------------------- stand-ins
class FakeRegions:
    """Stand-in for ``ChangedRegions``: a view's region is the top-left quarter of its frame, or, for the rows
    in ``sizes``, that many pixels at the start of its first image row."""

    def __init__(self, scene=None, edge=None, sizes=None):
        self.scene, self.edge, self.sizes = scene, edge, dict(sizes or {})
        self.points = SimpleNamespace(counts={"gone": {"kept": 5}, "new": {"kept": 2}})

    def mask(self, row: int, resolution: int = 512) -> np.ndarray:
        mask = np.zeros((resolution, resolution), dtype=bool)
        if row in self.sizes:
            mask[0, :self.sizes[row]] = True
        else:
            mask[:resolution // 2, :resolution // 2] = True
        return mask


class RegionDatedUNet(nn.Module):
    """A UNet that predicts zero, plus ``gain`` in the top-left quarter of every view whose date is unknown or
    is not the view's true year: the date acts inside the region and nowhere else."""

    def __init__(self, true_years: torch.Tensor, gain: float = 5.0):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))          # gives the module a device, as the renderer asks
        self.true_years, self.gain = true_years.clone(), gain
        self.year = self.valid = None

    @contextmanager
    def dated(self, year: torch.Tensor, valid: torch.Tensor | None):
        self.year, self.valid = year, torch.ones_like(year, dtype=torch.bool) if valid is None else valid
        try:
            yield
        finally:
            self.year = self.valid = None

    def forward(self, sample, timestep, encoder_hidden_states=None, add_inputs=None, class_labels=None, coords=None,
                return_dict=True, **unused):
        batch, views, _, height, width = sample.shape
        wrong = ~self.valid | ((self.year - self.true_years).abs() > 0.01)
        block = torch.zeros(height, width)
        block[:height // 2, :width // 2] = 1.0
        offset = self.gain * wrong.float().repeat(batch).view(-1, 1, 1, 1) * block
        return (torch.zeros(batch * views, 4, height, width) + offset,)


class RegionDatedMVGen(TinyMVGen):
    """The tiny model around a ``RegionDatedUNet``: its dates go to that UNet instead of a date conditioner."""

    def dates(self, year, valid=None, uncond="same"):
        return self.unet.dated(year, valid)


def region_dated_model(true_years: torch.Tensor) -> RegionDatedMVGen:
    base = tiny_model()
    return RegionDatedMVGen(unet=RegionDatedUNet(true_years).eval(), vae=base.vae, config=base.config,
                            scheduler=base.scheduler)


def quarter_masks(rows, sizes=None, min_pixels: int = 1) -> TargetMasks:
    """The fake regions of the given target rows."""
    return target_masks(FakeRegions(sizes=sizes), rows, SIZE, min_pixels)


def fire_probe(scene, loss_cfg: LossConfig = FP32, seeds=(0,), rows=AFTER_FROM_MODERN) -> RenderProbe:
    """The view set the evaluation is about: targets after the fire from references before it."""
    return RenderProbe(scene, rows, 3, "fire", RENDER_DATES, seeds, loss_cfg, SWAP_YEARS, steps=2)


@pytest.fixture
def run(folder, tmp_path) -> Trainer:
    """A finished three-step run of the tiny trainer: a hold-out, an EMA, both probes, a checkpoint per step."""
    cfg = config(folder, tmp_path / "run", sampling=SMALL, ema_decay=0.9, probe_every=3, render_every=3,
                 render_rows=((8, 9, 16, 17),), render_seeds=(0,), save_every=1, latest_every=1,
                 loss=LossConfig(precision="fp32", class_label=1),
                 holdout=HoldoutConfig(fraction=0.5, probe_targets=True, seed=1, era_edges=EDGES))
    trainer = Trainer(cfg, model=tiny_model())
    trainer.train()
    return trainer


def stand_in_models(script, monkeypatch) -> list:
    """Let the script load the tiny model instead of the released one; returns the list its models are put in,
    as ``(model, checkpoint)``."""
    models = []
    monkeypatch.setattr(mvgen_model, "load_mvgen", lambda model_dir, device: tiny_model())
    real_load = script.load_finetuned
    monkeypatch.setattr(script, "load_finetuned",
                        lambda *args: models.append(real_load(*args, device="cpu")) or models[-1])
    monkeypatch.setattr(script, "load_mvgen", lambda model_dir: models.append((tiny_model(), {})) or models[-1][0])
    monkeypatch.setattr(script, "ChangedRegions", lambda scene, edge: FakeRegions(scene, edge, sizes={17: 3}))
    return models


def read_results(out: Path) -> dict:
    return json.loads((out / "results.json").read_text())


# ----------------------------------------------------------------------------- the error map of a training step
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_the_mean_of_the_error_map_is_the_loss(scene, precision):
    model = adapted_model(lora_rank=2)
    sample = scene.make_sample(np.array(AFTER_FROM_MODERN), 3)
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    cfg = LossConfig(precision=precision, class_label=1)
    noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(3))
    valid = torch.ones(6, dtype=torch.bool)
    maps = []
    for timestep in (50, 500, 950):
        with torch.no_grad():
            loss, per_view = v_prediction_loss(model, latents, inputs, timestep, noise, sample["date_mid"], valid,
                                               cfg, 3)
            error = v_prediction_error_map(model, latents, inputs, timestep, noise, sample["date_mid"], valid, cfg, 3)
        assert error.shape == (3, LATENT, LATENT) and error.dtype == torch.float32 and error.min() >= 0
        torch.testing.assert_close(error.flatten(1).mean(1), per_view, rtol=1e-6, atol=0)       # per target view
        assert float(error.mean()) == pytest.approx(float(loss), rel=1e-6)
        assert error.std() > 0.01 * error.mean()            # a map: the error is not the same everywhere
        maps.append(error)
    assert not torch.equal(maps[0], maps[1])
    # the same forward pass as the loss: another date, or no date, moves the map as it moves the loss
    with torch.no_grad():
        other = v_prediction_error_map(model, latents, inputs, 500, noise, sample["date_mid"] - 30, valid, cfg, 3)
        unknown = v_prediction_error_map(model, latents, inputs, 500, noise, sample["date_mid"], ~valid, cfg, 3)
        loss_unknown, _ = v_prediction_loss(model, latents, inputs, 500, noise, sample["date_mid"], ~valid, cfg, 3)
    assert not torch.equal(other, maps[1]) and not torch.equal(unknown, maps[1])
    assert float(unknown.mean()) == pytest.approx(float(loss_unknown), rel=1e-6)

    model.scheduler = DDPMScheduler.from_config({**SCHEDULER, "prediction_type": "epsilon"})
    with pytest.raises(ValueError, match="not the velocity"):
        v_prediction_error_map(model, latents, inputs, 500, noise, None, None, cfg, 3)


def test_the_error_map_is_the_loss_on_mvgenmasters_own_unet(tmp_path, monkeypatch):
    monkeypatch.setattr(step, "get_camera_embedding", get_camera_embedding)        # the real ones, not the fakes
    monkeypatch.setattr(step, "get_3d_priors", get_3d_priors)
    model = small_mvgenmaster(tmp_path)
    sample = MVGenScene(write_folder(tmp_path / "scene64", 64), resolution=64).make_sample(
        np.array(AFTER_FROM_MODERN), 3)
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        loss, per_view = v_prediction_loss(model, latents, inputs, 600, noise, None, None, FP32, 3)
        error = v_prediction_error_map(model, latents, inputs, 600, noise, None, None, FP32, 3)
    assert error.shape == (3, 8, 8)
    torch.testing.assert_close(error.flatten(1).mean(1), per_view, rtol=1e-6, atol=0)
    assert float(error.mean()) == pytest.approx(float(loss), rel=1e-6)


# ----------------------------------------------------------------------------- areas
def test_area_means_split_the_whole_into_inside_and_outside():
    rng = np.random.default_rng(0)
    values, weights = rng.random((3, 4, 4)), rng.random((3, 4, 4))
    means = area_means(values, weights)
    share = weights.mean()
    assert list(means) == list(AREAS) and means["whole"] == pytest.approx(values.mean())
    assert share * means["inside"] + (1 - share) * means["outside"] == pytest.approx(means["whole"])
    assert means["inside"] == pytest.approx((values * weights).sum() / weights.sum())

    mask = values > 0.5                                     # a bool mask: plain means over the two sets of pixels
    means = area_means(values, mask)
    assert means["inside"] == pytest.approx(values[mask].mean())
    assert means["outside"] == pytest.approx(values[~mask].mean())
    nowhere = area_means(values, np.zeros_like(mask))
    assert nowhere["inside"] is None and nowhere["outside"] == nowhere["whole"] == pytest.approx(values.mean())
    assert area_means(values, np.ones_like(mask))["outside"] is None
    with pytest.raises(ValueError, match="one shape"):
        area_means(values, weights[0])
    assert to_psnr(None) is None and to_psnr(0.01) == pytest.approx(20.0) and to_psnr(0.0) == pytest.approx(100.0)


def test_regions_that_are_too_small_are_not_scored():
    masks = quarter_masks([16, 17, 18], sizes={17: 3}, min_pixels=4)
    assert masks.pixels.tolist() == [256, 3, 256] and masks.counted.tolist() == [True, False, True]
    assert masks.masks.shape == (3, SIZE, SIZE) and masks.masks.dtype == bool
    assert masks.masks[0].sum() == 256 and not masks.masks[1].any()        # emptied: its whole frame is outside
    assert quarter_masks([17], sizes={17: 3}, min_pixels=3).counted.tolist() == [True]
    assert quarter_masks([17], sizes={17: 0}, min_pixels=0).counted.tolist() == [False]     # nothing to score


def test_loss_noise_depends_on_the_seed_the_timestep_and_the_view_set_alone():
    noise = loss_noise(0, 500, AFTER_FROM_MODERN, (1, 6, 4, 4, 4))
    assert noise.shape == (1, 6, 4, 4, 4) and noise.std() == pytest.approx(1.0, abs=0.2)
    assert torch.equal(noise, loss_noise(0, 500, list(AFTER_FROM_MODERN), (1, 6, 4, 4, 4)))
    others = [loss_noise(1, 500, AFTER_FROM_MODERN, noise.shape), loss_noise(0, 501, AFTER_FROM_MODERN, noise.shape),
              loss_noise(0, 500, MODERN_FROM_MODERN, noise.shape)]
    assert all(not torch.equal(noise, other) for other in others)


# ----------------------------------------------------------------------------- loss with fixed noise
def test_losses_are_measured_as_the_date_loss_probe_measures_them(scene, tmp_path):
    cfg = LossConfig(precision="fp32", class_label=1, cross_date_pixels="drop", cross_date_gap=20.0)
    model = adapted_model()
    dates = record_dates(model)
    reference = DateLossProbe(scene, [AFTER_FROM_MODERN], 3, cfg, SWAP_YEARS, tmp_path / "b.json", timesteps=(200, 800))
    reference(model)
    probe_dates = list(dates)
    dates.clear()
    model.unet.seen.clear()

    probe = fire_probe(scene, cfg)
    masks = quarter_masks(AFTER_FROM_MODERN[3:], sizes={16: 20})       # 20 pixels: parts of three latent cells
    records = measure_losses(model, probe, masks, (200, 800), (0, 1))
    assert [(r["set"], r["timestep"], r["noise_seed"], r["variant"]) for r in records] == [
        ("fire", t, s, v) for t in (200, 800) for s in (0, 1) for v in ("true", "swapped", "unknown")]
    assert json.loads(json.dumps(records)) == records

    # the swapped years, and the dates the model is given for each variant, are the probe's
    assert torch.equal(probe.swapped, reference.swapped[0]) and probe.swapped[3:].tolist() == [2015.0] * 3
    assert len(dates) == 12 and len(probe_dates) == 6
    for (year, valid), (probe_year, probe_valid) in zip(dates[:3] + dates[6:9], probe_dates):
        assert torch.equal(year, probe_year) and torch.equal(valid, probe_valid)
    # the three variants of a measurement see the same noisy latents; another seed or timestep sees others
    noisy = [call["sample"] for call in model.unet.seen]
    assert all(torch.equal(noisy[i], noisy[i + 1]) and torch.equal(noisy[i], noisy[i + 2]) for i in range(0, 12, 3))
    assert len({tuple(noisy[i].flatten().tolist()) for i in range(0, 12, 3)}) == 4
    assert all(torch.equal(sample[0, :3], noisy[0][0, :3]) for sample in noisy)        # the references stay clean
    # a training step without dropout, in eval mode and without gradients; the run's domain label; and the
    # cross-date rule by the TRUE dates in every variant (at the swapped date the targets would keep their pixels)
    for call in model.unet.seen:
        assert not call["grad"] and not call["training"] and call["class_labels"].tolist() == [1]
        assert is_zero(call["coords"][1]).tolist() == [False] * 3 + [True] * 3
        assert not is_zero(call["coords"][0]).any() and call["add_inputs"][0, :, 1:].eq(2.0).all()

    # "whole" is the loss of step.v_prediction_loss on those inputs, and the areas split it by the latent weights
    sample = probe.sample
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents,
                          dropped_pixel_views=cross_date_drops(sample["date_mid"], 3, cfg, EDGES)[1])
    with torch.no_grad():
        loss, per_view = v_prediction_loss(model, latents, inputs, 800, loss_noise(1, 800, AFTER_FROM_MODERN,
                                           latents.shape), probe.swapped, torch.ones(6, dtype=torch.bool), cfg, 3)
    record = next(r for r in records if (r["timestep"], r["noise_seed"], r["variant"]) == (800, 1, "swapped"))
    assert record["whole"] == pytest.approx(float(loss), rel=1e-6)
    assert record["per_target"]["whole"] == pytest.approx(per_view.tolist(), rel=1e-6)
    weights = np.stack([latent_weights(mask, 8) for mask in masks.masks])
    assert weights[0].sum() * 64 == 20 and 0 < weights[0].max() < 1         # fractional cells
    shares = weights.mean(axis=(1, 2))
    for record in records:
        share = weights.mean()
        assert share * record["inside"] + (1 - share) * record["outside"] == pytest.approx(record["whole"])
        per_target = record["per_target"]
        assert shares * per_target["inside"] + (1 - shares) * per_target["outside"] == pytest.approx(
            per_target["whole"])
        assert np.mean(per_target["whole"]) == pytest.approx(record["whole"])
        # the size of the date embedding is reported for the true dates, as by the probe
        assert ("date_temb_ratio" in record) == (record["variant"] == "true")
    assert records[0]["date_temb_ratio"] > 0 and records[0]["date_rms"] > 0

    with pytest.raises(ValueError, match="outside the scheduler"):
        measure_losses(model, probe, masks, (1000,), (0,))


def test_a_view_set_without_a_scored_region_has_no_inside_numbers(scene):
    model = adapted_model()
    probe = fire_probe(scene)
    nowhere = quarter_masks(AFTER_FROM_MODERN[3:], sizes={16: 3, 17: 3, 18: 0}, min_pixels=4)
    records = measure_losses(model, probe, nowhere, (500,), (0,))
    assert all(r["inside"] is None and r["outside"] == r["whole"] for r in records)
    summary = summarise_set_losses(records)
    assert summary["inside"] is None and summary["outside"] == summary["whole"]
    assert summary["whole"]["gap_swapped"] != 0

    one = quarter_masks(AFTER_FROM_MODERN[3:], sizes={16: 3, 17: 3}, min_pixels=4)      # only the third target counts
    records = measure_losses(model, probe, one, (500,), (0,))
    assert [r["per_target"]["inside"][:2] for r in records] == [[None, None]] * 3
    assert all(r["inside"] == pytest.approx(r["per_target"]["inside"][2]) for r in records)


# ----------------------------------------------------------------------------- a date that acts inside the region
def test_a_date_that_acts_inside_the_region_only_shows_inside_only(scene, tmp_path):
    probe = fire_probe(scene, seeds=(0, 1))
    model = region_dated_model(probe.sample["date_mid"])
    masks = quarter_masks(AFTER_FROM_MODERN[3:])

    losses = summarise_set_losses(measure_losses(model, probe, masks, (200, 800), (0, 1)))
    for variant in ("swapped", "unknown"):
        inside, outside, whole = (losses[area] for area in AREAS)
        assert inside[f"gap_{variant}"] > 1 and outside[f"gap_{variant}"] == 0.0
        assert 0 < whole[f"gap_{variant}"] < inside[f"gap_{variant}"]         # diluted by the three quarters outside
        assert outside[f"gap_{variant}_by_timestep"] == [0.0, 0.0] and min(inside[f"gap_{variant}_by_timestep"]) > 1
        assert outside[f"loss_{variant}"] == outside["loss_true"]

    out_dir, sheet = tmp_path / "renders" / "fire", tmp_path / "sheets" / "fire.jpg"
    records = render_set(model, model.sampling_pipeline(), probe, masks, out_dir, sheet, thumb=16)
    assert [(record["set"], record["seed"]) for record in records] == [("fire", 0), ("fire", 1)]
    assert json.loads(json.dumps(records)) == records
    for record in records:
        difference, scores = record["difference"], record["psnr"]
        assert difference["outside"] == [0.0] * 3 and min(difference["inside"]) > 0.05
        assert difference["whole"] == pytest.approx([d / 4 for d in difference["inside"]])     # a quarter of the frame
        assert scores["true"]["outside"] == scores["swapped"]["outside"]
        assert all(a != b for a, b in zip(scores["true"]["inside"], scores["swapped"]["inside"]))
        assert all(a != b for a, b in zip(scores["true"]["whole"], scores["swapped"]["whole"]))
    assert records[0]["psnr"]["true"]["whole"] != records[1]["psnr"]["true"]["whole"]       # another seed
    renders = summarise_set_renders(records)
    assert renders["outside"]["psnr_gain"] == 0.0 and renders["outside"]["difference"] == 0.0
    assert renders["inside"]["psnr_gain"] != 0 and renders["inside"]["difference"] > 0.05
    assert abs(renders["whole"]["psnr_gain"]) < abs(renders["inside"]["psnr_gain"])

    # the pictures are kept: the rendered targets of every date and seed, the real photos and the regions
    assert sorted(path.name for path in out_dir.iterdir()) == sorted(
        [f"{date}_seed{seed}_view{j}.jpg" for date in RENDER_DATES for seed in (0, 1) for j in (3, 4, 5)]
        + [f"real_view{j}.jpg" for j in (3, 4, 5)] + [f"mask_view{j}.png" for j in (3, 4, 5)])
    assert np.array_equal(np.asarray(Image.open(out_dir / "mask_view3.png")), masks.masks[0])
    real = np.asarray(Image.open(out_dir / "real_view4.jpg")).astype(int)
    assert np.abs(real - np.array([170, 100, 200])).max() <= 3                 # the photo of row 17, as a JPEG
    for date in RENDER_DATES:
        rendered = render_at(model, model.sampling_pipeline(), probe, date, 1)[4].astype(int)
        kept = np.asarray(Image.open(out_dir / f"{date}_seed1_view4.jpg")).astype(int)
        assert kept.shape == (SIZE, SIZE, 3) and np.abs(kept - rendered).mean() < 4
    true, swapped = (np.asarray(Image.open(out_dir / f"{date}_seed0_view5.jpg")).astype(int) for date in RENDER_DATES)
    assert np.abs(true - swapped)[masks.masks[2]].mean() > 10 * np.abs(true - swapped)[~masks.masks[2]].mean()
    # the sheet of the first seed: a label column, a header, and four rows of three targets
    assert Image.open(sheet).size == (230 + 3 * 16, 18 + 14 + 4 * 16)


def test_renders_are_those_of_the_render_probe(scene, tmp_path):
    cfg = LossConfig(precision="fp32", class_label=1, cross_date_pixels="drop", cross_date_gap=20.0)
    model = adapted_model()
    probe = RenderProbe(scene, AFTER_FROM_MODERN, 3, "fire", RENDER_DATES, (0, 1), cfg, SWAP_YEARS, steps=2,
                        guidance=3.0, uncond_dates="unknown")
    reference = probe(model, tmp_path / "probe")["renders"]             # true 0, true 1, swapped 0, swapped 1
    calls = len(model.unet.seen)
    masks = quarter_masks(AFTER_FROM_MODERN[3:], sizes={17: 3}, min_pixels=4)
    records = render_set(model, model.sampling_pipeline(), probe, masks, tmp_path / "kept", tmp_path / "sheet.jpg",
                         thumb=16)
    assert len(model.unet.seen) == 2 * calls            # the same renders, once more
    for seed, record in enumerate(records):
        assert record["psnr"]["true"]["whole"] == pytest.approx(reference[seed]["psnr_per_target"], rel=1e-5)
        assert record["psnr"]["swapped"]["whole"] == pytest.approx(reference[2 + seed]["psnr_per_target"], rel=1e-5)
        assert np.mean(record["difference"]["whole"]) == pytest.approx(reference[2 + seed]["diff_vs_first"], rel=1e-5)
        assert record["difference"]["inside"][1] is None and record["psnr"]["true"]["inside"][1] is None
        assert record["difference"]["outside"][1] == record["difference"]["whole"][1]
    # with these settings the two dates differ in more than the date: the swapped one keeps its warped pixels
    pipeline = model.pipelines[-1]
    assert [(call["steps"], call["guidance"], call["class_label"]) for call in pipeline.calls] == [(2, 3.0, 1)] * 4
    assert all(not call["grad"] for call in model.unet.seen)

    # the scores of a pair of renders, against plain arithmetic
    real = np.full((1, 4, 4, 3), 100, dtype=np.uint8)
    true, swapped = real.copy(), real.copy()
    true[0, :2] += 10                                   # the true-date render is off in the top half ...
    swapped[0, :, :2] += 30                             # ... and the swapped one in the left half
    mask = np.zeros((1, 4, 4), dtype=bool)
    mask[0, :, :2] = True
    scores = score_renders(real, true, swapped, mask)
    assert scores["psnr"]["true"]["whole"] == [pytest.approx(psnr(true[0], real[0]), rel=1e-6)]
    assert scores["psnr"]["swapped"]["inside"] == [pytest.approx(to_psnr((30 / 255) ** 2))]
    assert scores["psnr"]["swapped"]["outside"] == [pytest.approx(100.0)]          # identical there: the cap
    assert scores["psnr"]["true"]["inside"] == scores["psnr"]["true"]["outside"]
    assert scores["difference"] == {"inside": [pytest.approx(25 / 255)], "outside": [pytest.approx(5 / 255)],
                                    "whole": [pytest.approx(15 / 255)]}


def test_losses_and_renders_on_mvgenmasters_own_unet_and_pipeline(tmp_path, monkeypatch):
    """The path of a GPU job, as far as a CPU goes: MVGenMaster's UNet class, camera rays, 3D priors and sampling
    pipeline, under bf16 autocast, with a date embedding that is no longer zero."""
    monkeypatch.setattr(step, "get_camera_embedding", get_camera_embedding)        # the real ones, not the fakes
    monkeypatch.setattr(step, "get_3d_priors", get_3d_priors)
    model = small_mvgenmaster(tmp_path)
    prepare_for_finetuning(model, FinetuneConfig(date_freqs=4, date_hidden=16))
    torch.manual_seed(1)
    nn.init.normal_(model.date.embedding.mlp[-1].weight, std=0.5)       # as after training
    size = 64
    scene = MVGenScene(write_folder(tmp_path / "scene64", size), resolution=size)
    cfg = LossConfig(precision="bf16")
    probe = RenderProbe(scene, AFTER_FROM_MODERN, 3, "fire", RENDER_DATES, (0,), cfg, SWAP_YEARS, steps=2)
    masks = target_masks(FakeRegions(sizes={17: 20}), AFTER_FROM_MODERN[3:], size, min_pixels=1)

    records = measure_losses(model, probe, masks, (300, 700), (0,))
    weights = np.stack([latent_weights(mask, 8) for mask in masks.masks])
    assert weights.shape == (3, 8, 8) and len(records) == 2 * 3
    for record in records:
        assert all(np.isfinite(record[area]) and record[area] > 0 for area in AREAS)
        share = weights.mean()
        assert share * record["inside"] + (1 - share) * record["outside"] == pytest.approx(record["whole"])
    losses = summarise_set_losses(records)
    assert all(losses[area][f"gap_{variant}"] != 0 for area in AREAS for variant in ("swapped", "unknown"))
    # "whole" at the true dates is the loss of a training step on the same inputs and noise
    latents = encode_images(model, probe.sample["images"], sample_posterior=False)
    with torch.no_grad():
        loss, _ = v_prediction_loss(model, latents, build_inputs(model, probe.sample, latents), 300,
                                    loss_noise(0, 300, AFTER_FROM_MODERN, latents.shape), probe.sample["date_mid"],
                                    torch.ones(6, dtype=torch.bool), cfg, 3)
    assert records[0]["whole"] == pytest.approx(float(loss), rel=1e-6)

    pipeline = model.sampling_pipeline()
    pipeline.set_progress_bar_config(disable=True)
    reference = probe(model, tmp_path / "probe")["renders"]             # the render probe itself: true, swapped
    record, = render_set(model, pipeline, probe, masks, tmp_path / "kept", tmp_path / "sheets" / "fire.jpg", thumb=32)
    assert record["psnr"]["true"]["whole"] == pytest.approx(reference[0]["psnr_per_target"], rel=1e-5)
    assert record["psnr"]["swapped"]["whole"] == pytest.approx(reference[1]["psnr_per_target"], rel=1e-5)
    assert np.mean(record["difference"]["whole"]) == pytest.approx(reference[1]["diff_vs_first"], rel=1e-5)
    assert all(np.isfinite(value) and value > 0 for area in AREAS for value in record["difference"][area])
    assert np.asarray(Image.open(tmp_path / "kept" / "swapped_seed0_view5.jpg")).shape == (size, size, 3)
    assert Image.open(tmp_path / "sheets" / "fire.jpg").size == (230 + 3 * 32, 18 + 14 + 4 * 32)


# ----------------------------------------------------------------------------- summaries
def loss_records(true, swapped, unknown, inside=True) -> list[dict]:
    """Loss records of one view set at timesteps 200 and 800: per variant ``(inside, outside)`` at each timestep.
    The region is half of the frame."""
    records = []
    for i, timestep in enumerate((200, 800)):
        for variant, values in (("true", true), ("swapped", swapped), ("unknown", unknown)):
            inner, outer = values[i]
            records.append({"variant": variant, "timestep": timestep, "noise_seed": 0,
                            "inside": inner if inside else None, "outside": outer,
                            "whole": (inner + outer) / 2 if inside else outer})
    return records


def test_a_view_sets_losses_are_summarised_as_mean_ratios():
    summary = summarise_set_losses(loss_records(true=[(1.0, 4.0), (2.0, 4.0)], swapped=[(1.1, 4.0), (2.4, 4.0)],
                                                unknown=[(1.0, 4.0), (3.0, 4.0)]))
    inside, outside = summary["inside"], summary["outside"]
    assert inside["loss_true"] == 1.5 and inside["loss_swapped"] == pytest.approx(1.75)
    assert inside["gap_swapped"] == pytest.approx(0.15)            # the mean of the ratios 0.1 and 0.2, not 1.75 / 1.5
    assert inside["gap_swapped_by_timestep"] == pytest.approx([0.1, 0.2])
    assert inside["gap_unknown"] == pytest.approx(0.25) and inside["gap_unknown_by_timestep"] == pytest.approx([0, 0.5])
    assert outside["gap_swapped"] == outside["gap_unknown"] == 0.0 and outside["loss_true"] == 4.0
    assert summary["whole"]["gap_swapped"] == pytest.approx(np.mean([0.05 / 2.5, 0.2 / 3.0]))
    assert json.loads(json.dumps(summary)) == summary
    without = summarise_set_losses(loss_records([(1.0, 4.0)] * 2, [(1.0, 5.0)] * 2, [(1.0, 4.0)] * 2, inside=False))
    assert without["inside"] is None and without["outside"]["gap_swapped"] == pytest.approx(0.25)


def test_a_view_sets_renders_are_summarised_over_seeds_and_scored_targets():
    def record(true, swapped, difference):
        spread = lambda inside: {"inside": inside, "outside": [20.0, 20.0], "whole": [20.0, 21.0]}     # noqa: E731
        return {"psnr": {"true": spread(true), "swapped": spread(swapped)},
                "difference": {"inside": difference, "outside": [0.01, 0.03], "whole": [0.02, 0.04]}}

    summary = summarise_set_renders([record([10.0, None], [8.0, None], [0.2, None]),
                                     record([14.0, None], [9.0, None], [0.4, None])])
    assert summary["inside"] == {"psnr_true": 12.0, "psnr_swapped": 8.5, "psnr_gain": 3.5,
                                 "difference": pytest.approx(0.3)}
    assert summary["outside"]["psnr_gain"] == 0.0 and summary["outside"]["difference"] == pytest.approx(0.02)
    assert summary["whole"]["psnr_true"] == 20.5
    assert summarise_set_renders([record([None, None], [None, None], [None, None])])["inside"] is None


def test_groups_average_over_their_view_sets_with_a_standard_error():
    view_sets = [ViewSet("a", "keys.json", (0, 1, 2), cross=True),
                 ViewSet("b/0", HELDOUT_SOURCE, (3, 4, 5), cross=True),
                 ViewSet("c", "keys.json", (6, 7, 8), cross=False)]
    assert [view_set.group for view_set in view_sets] == ["cross", "cross", "other"]
    assert view_sets[1].folder == "b_0"
    assert groups_of(view_sets) == {"cross": ["a", "b/0"], "other": ["c"], "all": ["a", "b/0", "c"]}
    assert groups_of(view_sets[:2]) == {"cross": ["a", "b/0"]} and groups_of(view_sets[2:]) == {"other": ["c"]}
    assert mean_and_error([1.0, 3.0]) == (2.0, 1.0) and mean_and_error([4.0]) == (4.0, None)

    def numbers(gap, by_timestep):
        return {"gap": gap, "by_timestep": by_timestep}

    per_set = {"a": {"inside": numbers(0.1, [0.0, 0.2]), "outside": numbers(0.0, [0, 0]),
                     "whole": numbers(0.02, [0.0, 0.04])},
               "b/0": {"inside": numbers(0.3, [0.2, 0.4]), "outside": numbers(0.02, [0, 0]),
                       "whole": numbers(0.04, [0.0, 0.08])},
               "c": {"inside": None, "outside": numbers(0.5, [1, 0]), "whole": numbers(0.5, [1, 0])}}
    summary = summarise_groups(view_sets, per_set)
    assert list(summary) == ["cross", "other", "all"] and [summary[g]["sets"] for g in summary] == [2, 1, 3]
    cross = summary["cross"]["inside"]
    assert cross["sets"] == 2 and cross["gap"] == pytest.approx(0.2) and cross["gap_se"] == pytest.approx(0.1)
    assert cross["by_timestep"] == pytest.approx([0.1, 0.3]) and "by_timestep_se" not in cross
    assert summary["other"]["inside"] == {"sets": 0}                # no set of the group has a region
    assert summary["other"]["outside"] == {"sets": 1, "gap": 0.5, "gap_se": None, "by_timestep": [1.0, 0.0]}
    assert summary["all"]["inside"]["sets"] == 2 and summary["all"]["outside"]["sets"] == 3
    assert json.loads(json.dumps(summary)) == summary

    table = format_table("gaps", summary, [("swapped date", "gap", "+.1%")]).splitlines()
    assert table[0] == "gaps" and table[1].split() == ["view", "sets", "area", "sets", "swapped", "date"]
    assert len(table) == 2 + 3 * 3
    assert table[2].split() == ["cross", "(2)", "inside", "2", "+20.0%", "+-", "10.0%"]
    assert table[3].split() == ["outside", "2", "+1.0%", "+-", "1.0%"]
    assert table[5].split() == ["other", "(1)", "inside", "0", "-"]
    assert table[6].split() == ["outside", "1", "+50.0%", "+-", "n/a"]


# ----------------------------------------------------------------------------- view sets
def test_crossing_the_edge_means_targets_after_it_and_references_before():
    assert crosses_edge([2012.0, 2015.0, 2019.3, 2021.0], 2, FIRE)
    assert crosses_edge([1890.0, 2015.0, FIRE], 2, FIRE)                    # a date on the edge is after it
    assert not crosses_edge([2012.0, 2019.3, 2019.3, 2021.0], 2, FIRE)      # a reference after it
    assert not crosses_edge([2012.0, 2015.0, 2019.2, 2021.0], 2, FIRE)      # a target before it
    assert not crosses_edge([2020.0, 2021.0, 2012.0, 2015.0], 2, FIRE)      # the other way round
    assert crosses_edge([1900.0, 1960.0, 1975.0], 2, 1970.0)


def test_heldout_sets_follow_the_sequence_of_the_runs_held_out_probe(folder):
    small = MVGenScene(folder, SMALL, resolution=SIZE)
    assert heldout_sets(small, 3) == []                                 # nothing is held out
    small.hold_out(np.arange(0, N, 2))                                  # every other view: 4 per period
    sequence = heldout_view_sets(small, 64)
    first = heldout_sets(small, 3)
    assert [i for i, _ in first] == [0, 1, 2]
    assert all(np.array_equal(rows, sequence[i]) for i, rows in first)
    assert heldout_sets(small, 0) == [] and heldout_sets(small, 0, FIRE) == []

    crossing = [i for i, rows in enumerate(sequence) if crosses_edge(small.dates[rows], 2, FIRE)]
    assert 3 <= len(crossing) < len(sequence) and crossing[:3] != [0, 1, 2]     # some sets cross, not the first three
    chosen = heldout_sets(small, 3, FIRE)
    assert [i for i, _ in chosen] == crossing[:3]
    for i, rows in chosen:
        assert np.array_equal(rows, sequence[i])
        assert (small.dates[rows[2:]] >= FIRE).all() and (small.dates[rows[:2]] < FIRE).all()
        assert small.heldout[rows[2:]].all() and not small.heldout[rows[:2]].any()
    # an edge that no held-out set crosses: the search gives up
    assert heldout_sets(small, 2, 2050.0) == []


def test_view_sets_are_collected_from_keys_files_and_the_held_out_sampler(folder, tmp_path, capsys):
    small = MVGenScene(folder, SMALL, resolution=SIZE)
    small.hold_out(np.arange(0, N, 2))
    gate, more = tmp_path / "gate.json", tmp_path / "more.json"
    first_heldout = [f"view_{row:02d}.png" for row in heldout_view_sets(small, 1)[0]]
    gate.write_text(json.dumps([{"refs": ["view_08.png", "view_09.png"], "targets": ["view_16.png", "view_17.png"]},
                                {"name": "modern", "refs": ["view_08.png", "view_09.png"],
                                 "targets": ["view_10.png", "view_11.png", "view_12.png"]}]))
    more.write_text(json.dumps([{"refs": ["view_08.png", "view_09.png"], "targets": ["view_16.png", "view_17.png"]},
                                {"refs": first_heldout[:2], "targets": first_heldout[2:]}]))
    view_sets = collect_view_sets(small, [gate, more], n_heldout=2)
    assert [(view_set.name, view_set.source) for view_set in view_sets] == [
        ("gate/0", "gate.json"), ("modern", "gate.json"), ("more/1", "more.json"), ("heldout/1", HELDOUT_SOURCE)]
    assert view_sets[0].rows == (8, 9, 16, 17) and view_sets[0].cross and not view_sets[1].cross
    printed = capsys.readouterr().out
    assert "more/0 (more.json) is the view set gate/0: evaluated once" in printed
    assert f"heldout/0 ({HELDOUT_SOURCE}) is the view set more/1: evaluated once" in printed
    # --cross-only limits the held-out sets, not those of the keys files; the edge decides what crosses
    crossing = collect_view_sets(small, [gate], n_heldout=2, cross_only=True)
    assert [view_set.name for view_set in crossing[:2]] == ["gate/0", "modern"]
    assert all(view_set.cross and view_set.source == HELDOUT_SOURCE for view_set in crossing[2:]) and len(crossing) == 4
    assert not any(view_set.cross for view_set in collect_view_sets(small, [gate], edge=2050.0))
    assert [view_set.name for view_set in collect_view_sets(small, [], n_heldout=1)] == ["heldout/0"]
    with pytest.raises(ValueError, match="2 references"):
        collect_view_sets(MVGenScene(folder, resolution=SIZE), [gate])           # a scene of three references


def test_a_view_set_is_described_for_the_records(folder):
    small = MVGenScene(folder, SMALL, resolution=SIZE)
    view_set = ViewSet("gate/0", "gate.json", (8, 9, 16, 17, 18), cross=True)
    probe = RenderProbe(small, view_set.rows, 2, view_set.name, RENDER_DATES, (0,), FP32, SWAP_YEARS)
    masks = quarter_masks(view_set.rows[2:], sizes={17: 3}, min_pixels=4)
    record = describe_set(small, view_set, probe, masks)
    assert (record["name"], record["folder"], record["source"], record["group"]) == ("gate/0", "gate_0", "gate.json",
                                                                                       "cross")
    assert record["rows"] == [8, 9, 16, 17, 18] and record["n_ref"] == 2 and record["scale_factor"] == 1.0
    assert record["years"] == pytest.approx([2012.0, 2012.1, 2020.0, 2020.1, 2020.2])
    assert record["swapped_years"] == [2015.0] * 3 and record["held_out"] is None       # nothing is held out
    assert record["region_pixels"] == [256, 3, 256] and record["region_scored"] == [True, False, True]
    assert record["region_share"] == pytest.approx([0.25, 3 / 1024, 0.25])
    assert json.loads(json.dumps(record)) == record
    small.hold_out([9, 16, 18])
    assert describe_set(small, view_set, probe, masks)["held_out"] == [False, True, True, False, True]


# ----------------------------------------------------------------------------- a run's settings
def test_a_runs_settings_scene_and_hold_out_are_rebuilt_from_its_folder(run, tmp_path):
    run_dir, trained = run.run_dir, run.cfg
    cfg = run_config(run_dir)
    assert cfg == trained and isinstance(cfg.render_rows[0], tuple) and isinstance(cfg.sampling.period_edges, tuple)
    assert run_swap_years(run_dir) == pytest.approx(run.swap_years)
    assert (trained.ema_decay, trained.holdout.probe_targets) == (0.9, True)       # the run the fixture describes

    scene = open_scene(cfg)
    assert scene.heldout.sum() >= 12 and np.array_equal(scene.heldout, run.scene.heldout)
    assert scene.heldout[[16, 17]].all()                                # the targets of the run's render probe
    assert (scene.scale, scene.scale_factor, scene.seed) == (run.scene.scale, run.scene.scale_factor, run.scene.seed)
    assert scene.fingerprint() == run.dataset_fingerprint
    # the held-out sets are those the run's own held-out date-loss probe measures
    assert [rows.tolist() for _, rows in heldout_sets(scene, 2)] == run.loss_probes["heldout"].rows
    checkpoint = torch.load(run_dir / "checkpoints" / "step_000002.pt", weights_only=False)
    check_run_records(scene, run_dir)
    check_run_records(scene, run_dir, checkpoint)

    # a dataset or a hold-out that is not the run's is refused, with what differs
    other_holdout = open_scene(replace(cfg, holdout=replace(cfg.holdout, seed=2)))
    with pytest.raises(ValueError, match="held-out views are not those of heldout.json.*cannot be told"):
        check_run_records(other_holdout, run_dir)
    halved = open_scene(replace(cfg, scale=0.5))
    with pytest.raises(ValueError, match="scale factor is 0.5, config.json has 1.0"):
        check_run_records(halved, run_dir)
    with pytest.raises(ValueError, match="scale factor is 0.5, config.json has 1.0; .*0.5, the checkpoint has 1.0"):
        check_run_records(halved, run_dir, checkpoint)
    with pytest.raises(ValueError, match="dataset fingerprint is not the checkpoint's"):
        check_run_records(scene, run_dir, {**checkpoint, "dataset_fingerprint": "of another folder"})
    check_run_records(scene, run_dir, {"step": 2})                      # a checkpoint that recorded neither
    # MVGenMaster's limit on the extent of the cameras: by default MVGenScene's own, the released model's
    assert scene.camera_longest_side == 5.0
    assert open_scene(cfg, camera_longest_side=2.3).scale_factor == pytest.approx(0.5)      # the extent is 4.6

    # a run without a hold-out: nothing is held out, and nothing recorded to compare with
    plain_dir = tmp_path / "plain"
    Trainer(config(run.cfg.dataset, plain_dir, pinned_rows=(AFTER_FROM_MODERN,)), model=tiny_model())
    plain = open_scene(run_config(plain_dir))
    assert not plain.heldout.any() and not plain.require_depth and heldout_sets(plain, 2) == []
    check_run_records(plain, plain_dir)


def test_a_checkpoint_is_named_by_step_latest_file_or_released(run):
    folder = run.run_dir / "checkpoints"
    assert checkpoint_path(run.run_dir, "latest") == folder / "latest.pt"
    assert checkpoint_path(run.run_dir, "2") == folder / "step_000002.pt"
    assert checkpoint_path(run.run_dir, str(folder / "step_000001.pt")) == folder / "step_000001.pt"
    assert checkpoint_path(run.run_dir, "released") is None
    with pytest.raises(FileNotFoundError, match="step_000007.pt.*holds \\['latest.pt', 'step_000001.pt'"):
        checkpoint_path(run.run_dir, "7")
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        checkpoint_path(run.run_dir, "newest")


# ----------------------------------------------------------------------------- the command line
def test_the_documented_queue_lines_parse():
    script = load_script("eval_dates")
    lines = [line.split() for line in script.__doc__.splitlines() if line.strip().startswith("/")]
    assert len(lines) == 2                                  # one task per line: the queue splits it at spaces
    for line in lines:
        assert Path(line[0]).is_absolute() and Path(line[0]).name == "eval_dates.py"
    at_step, released = (script.parse_args(line[1:]) for line in lines)
    assert at_step.checkpoint == "1000" and released.checkpoint == "released" and at_step.out != released.out
    for args in (at_step, released):
        assert args.run.is_absolute() and args.run.name == "gate3_keep" and args.out.is_absolute()
        assert (args.heldout, args.cross_only, args.keys_file) == (8, True, [script.DEFAULT_KEYS_FILE])
        assert (args.noise_seeds, args.min_mask_pixels, args.thumb) == ([0, 1], 256, 192)
        assert (args.timesteps, args.seeds, args.steps, args.guidance, args.uncond_dates) == (None,) * 5   # the run's
        assert not (args.no_losses or args.no_renders or args.list_sets) and args.region_edge is None
    assert script.DEFAULT_KEYS_FILE.name == "mvgen_gate2.json" and script.DEFAULT_KEYS_FILE.exists()


def test_options_are_parsed_and_contradictions_refused(tmp_path, capsys):
    script = load_script("eval_dates")
    out = ["--out", str(tmp_path / "out")]
    args = script.parse_args([*out, "--run", "/runs/a", "--checkpoint", "latest", "--keys-file", "/k/a.json",
                              "--keys-file", "/k/b.json", "--keys-file", "/k/a.json", "--heldout", "4", "--no-renders",
                              "--seeds", "3", "4", "--timesteps", "100", "900", "--noise-seeds", "5",
                              "--region-edge", "1970", "--min-mask-pixels", "64", "--steps", "20", "--guidance", "3",
                              "--uncond-dates", "unknown"])
    assert args.keys_file == [Path("/k/a.json"), Path("/k/b.json")] and args.run == Path("/runs/a")
    assert (args.seeds, args.timesteps, args.noise_seeds, args.region_edge) == ([3, 4], [100, 900], [5], 1970.0)
    assert (args.min_mask_pixels, args.steps, args.guidance, args.uncond_dates) == (64, 20, 3.0, "unknown")
    assert args.no_renders and not args.no_losses and not args.cross_only
    assert script.parse_args([*out, "--run", "/runs/a", "--keys-file", "none", "--heldout", "2"]).keys_file == []

    released = script.parse_args([*out, "--scale", "0.5", "--class-label", "1", "--precision", "fp32", "--dataset",
                                  "/data/x", "--resolution", "256", "--swap-years", "1900", "2000", "2020"])
    cfg, swap_years, checkpoint = script.settings(released)
    assert (cfg.dataset, cfg.resolution, cfg.scale, cfg.swap_years) == ("/data/x", 256, 0.5, (1900.0, 2000.0, 2020.0))
    assert cfg.loss == LossConfig(class_label=1, precision="fp32") and swap_years == cfg.swap_years
    assert checkpoint is None and not cfg.holdout.active
    cfg, swap_years, _ = script.settings(script.parse_args(out))        # the defaults, as scripts/mvgen/render.py
    assert (cfg.dataset, cfg.scale, cfg.resolution) == (str(script.DEFAULT_DATASET), "scene", 512)
    assert cfg.loss == LossConfig() and swap_years is None and cfg.sampling == ViewSamplingConfig()

    refused = {"--checkpoint needs --run": ["--checkpoint", "1000"],
               "--heldout needs --run": ["--heldout", "8"],
               "--scale, --class-label cannot be given with --run": ["--run", "/r", "--scale", "1",
                                                                     "--class-label", "1"],
               "--cross-only limits the --heldout sets": ["--run", "/r", "--cross-only"],
               "leave nothing to measure": ["--no-losses", "--no-renders"],
               "no view sets": ["--keys-file", "none"]}
    for message, argv in refused.items():
        with pytest.raises(SystemExit):
            script.parse_args([*out, *argv])
        assert message in capsys.readouterr().err
    assert script.parse_args([*out, "--no-losses", "--no-renders", "--list-sets"]).list_sets


def test_the_script_refuses_tf32_matrix_products_before_doing_anything(tmp_path):
    script = load_script("eval_dates")
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with pytest.raises(AssertionError, match="TF32"):
            script.main(["--out", str(tmp_path / "out")])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------------- the script, end to end
def keys_file(tmp_path: Path) -> Path:
    """Two view sets of two references: the run's render set (after the fire), and modern targets."""
    path = tmp_path / "keys.json"
    path.write_text(json.dumps([{"refs": ["view_08.png", "view_09.png"], "targets": ["view_16.png", "view_17.png"]},
                                {"name": "modern", "refs": ["view_08.png", "view_09.png"],
                                 "targets": ["view_10.png", "view_11.png", "view_12.png"]}]))
    return path


def test_a_run_of_the_trainer_is_evaluated_at_a_checkpoint(run, tmp_path, monkeypatch, capsys):
    script = load_script("eval_dates")
    models = stand_in_models(script, monkeypatch)
    run_dir, out = run.run_dir, tmp_path / "eval"
    before = run_files(run_dir)
    capsys.readouterr()
    script.main(["--run", str(run_dir), "--checkpoint", "2", "--keys-file", str(keys_file(tmp_path)), "--heldout", "2",
                 "--cross-only", "--out", str(out), "--min-mask-pixels", "4", "--thumb", "16"])
    printed = capsys.readouterr().out
    assert run_files(run_dir) == before                     # the run folder is read, never written to
    results = read_results(out)

    # the model: the checkpoint's weights, which are the EMA the run's probes measure; eval mode, no gradients
    model, ckpt = models[-1]
    assert len(models) == 1 and ckpt["step"] == 2 and ckpt["weights"] == "ema"
    saved = ckpt["trainable"]
    torch.testing.assert_close(flat_weights(model), torch.cat(
        [saved[part][key].flatten() for part in ("date", "lora") for key in sorted(saved[part])]))
    ratio = results["model"].pop("date_temb_ratio")         # the date embedding's size, in the true-date passes
    assert 0 < ratio < 1
    assert results["model"] == {"run": str(run_dir), "checkpoint": str(run_dir / "checkpoints" / "step_000002.pt"),
                                "step": 2, "weights": "ema",
                                "finetune": json.loads(json.dumps(ckpt["finetune_config"]))}
    assert model.unet.seen and all(not call["grad"] and not call["training"] for call in model.unet.seen)
    assert "step_000002.pt (step 2, the EMA of the weights)" in printed

    # the settings are the run's: its loss settings, swap years, probe timesteps and sampler settings
    settings = results["settings"]
    assert settings["loss"]["class_label"] == 1 and settings["loss"]["precision"] == "fp32"
    assert settings["swap_years"] == pytest.approx(run.swap_years) and settings["timesteps"] == [200, 800]
    assert (settings["render_seeds"], settings["render_steps"], settings["render_guidance"]) == ([0], 2, 2.0)
    assert (settings["render_uncond_dates"], settings["noise_seeds"], settings["n_ref"]) == ("same", [0, 1], 2)
    assert (settings["scale"], settings["scale_factor"], settings["dataset"]) == ("scene", 1.0, run.cfg.dataset)
    assert results["region"] == {"edge": FIRE, "min_mask_pixels": 4, "targets": 11, "targets_not_scored": sum(
        17 in entry["rows"][2:] for entry in results["sets"]), "points": {"gone": {"kept": 5}, "new": {"kept": 2}}}
    pipeline = model.pipelines[-1]
    assert len(model.pipelines) == 1 and len(pipeline.calls) == 4 * 2      # 4 sets x 2 dates x 1 seed
    assert {(call["steps"], call["guidance"], call["class_label"], call["cond_num"]) for call in pipeline.calls} == {
        (2, 2.0, 1, 2)}

    # the view sets: the two of the keys file, then the first two held-out sets of the run that cross the fire
    sequence = heldout_view_sets(run.scene, 64)
    crossing = [i for i, rows in enumerate(sequence) if crosses_edge(run.scene.dates[rows], 2, FIRE)][:2]
    sets = results["sets"]
    assert [entry["name"] for entry in sets] == ["keys/0", "modern", *(f"heldout/{i}" for i in crossing)]
    assert [entry["group"] for entry in sets] == ["cross", "other", "cross", "cross"]
    assert [entry["source"] for entry in sets] == ["keys.json"] * 2 + [HELDOUT_SOURCE] * 2
    assert [entry["rows"] for entry in sets[2:]] == [sequence[i].tolist() for i in crossing]
    assert sets[0]["rows"] == [8, 9, 16, 17] and sets[0]["held_out"][2:] == [True, True]
    assert all(all(entry["held_out"][2:]) and not any(entry["held_out"][:2]) for entry in sets[2:])
    assert sets[0]["swapped_years"] == pytest.approx([run.swap_years[1]] * 2, abs=1e-3)
    assert sets[0]["region_pixels"] == [256, 3] and sets[0]["region_scored"] == [True, False]
    assert results["sets_evaluated"] == {"loss": 4, "renders": 4, "of": 4}
    assert "4 view sets: 3 cross (every target dated from 2019.2877 on, every reference before it), 1 other" in printed
    assert "keys/0" in printed and "(2 of 2 held out of training)" in printed and f"from {HELDOUT_SOURCE}" in printed
    assert "0.3%*" in printed and "* a region of fewer than 4 pixels: the target is not scored inside (" in printed
    assert f"({results['region']['targets_not_scored']} of 11 targets)" in printed

    # losses: one line per set, timestep, noise seed and variant; per set and per group the gaps by area
    lines = [json.loads(line) for line in (out / "losses.jsonl").read_text().splitlines()]
    assert len(lines) == 4 * 2 * 2 * 3 and {line["set"] for line in lines} == {entry["name"] for entry in sets}
    for entry in sets:
        assert set(entry["loss"]) == set(AREAS)
        assert entry["loss"] == summarise_set_losses([line for line in lines if line["set"] == entry["name"]])
        assert entry["loss"]["inside"]["gap_swapped"] != 0
        assert len(entry["loss"]["whole"]["gap_swapped_by_timestep"]) == 2
    assert list(results["loss"]) == ["cross", "other", "all"]
    cross = results["loss"]["cross"]
    assert cross["sets"] == 3 and cross["inside"]["sets"] == 3
    gaps = [entry["loss"]["inside"]["gap_swapped"] for entry in sets if entry["group"] == "cross"]
    assert (cross["inside"]["gap_swapped"], cross["inside"]["gap_swapped_se"]) == pytest.approx(mean_and_error(gaps))
    assert results["loss"]["other"]["whole"]["gap_unknown_se"] is None and results["loss"]["all"]["sets"] == 4

    # renders: the pictures, the regions and a sheet per set; the scores per seed and target, and the tables
    files = set(run_files(out))
    for entry in sets:
        positions = range(2, len(entry["rows"]))
        assert {f"renders/{entry['folder']}/{kind}_view{j}.{kind_ext}" for j in positions for kind, kind_ext in (
            ("true_seed0", "jpg"), ("swapped_seed0", "jpg"), ("real", "jpg"), ("mask", "png"))} <= files
        assert f"sheets/{entry['folder']}.jpg" in files
        assert [record["seed"] for record in entry["render_records"]] == [0]
        assert entry["renders"] == summarise_set_renders(entry["render_records"])
        assert entry["renders"]["inside"]["difference"] > 0
    assert sets[0]["folder"] == "keys_0" and sets[2]["folder"] == f"heldout_{crossing[0]}"
    assert sets[0]["render_records"][0]["psnr"]["true"]["inside"][1] is None            # the target of 3 pixels
    assert len(files) == 2 + 4 + sum(4 * (len(entry["rows"]) - 2) for entry in sets)    # nothing else is written
    assert results["renders"]["cross"]["inside"]["sets"] == 3 and "psnr_gain_se" in results["renders"]["all"]["whole"]
    assert Image.open(out / "sheets" / "modern.jpg").size == (230 + 3 * 16, 18 + 14 + 4 * 16)

    # what a person reads: the two tables, a block per group, a line per area
    assert "LOSS with fixed noise" in printed and "RENDERS: PSNR against the real photos" in printed
    assert "date embedding / timestep embedding" in printed
    loss_table = printed.split("LOSS with fixed noise")[1].split("RENDERS")[0].splitlines()
    assert "timesteps [200, 800], noise seeds [0, 1]" in loss_table[1]
    assert loss_table[2].split()[:4] == ["view", "sets", "area", "sets"]
    assert [line.split()[:2] for line in loss_table[3:9]] == [["cross", "(3)"], ["outside", "3"], ["whole", "3"],
                                                              ["other", "(1)"], ["outside", "1"], ["whole", "1"]]
    assert "seeds [0], 2 sampler steps, guidance 2, unconditional dates same" in printed
    assert any(line.startswith("inside the region, swapped date, by timestep (cross sets): t=200")
               for line in loss_table)


def test_latest_and_the_released_baseline_of_a_run(run, tmp_path, monkeypatch, capsys):
    script = load_script("eval_dates")
    models = stand_in_models(script, monkeypatch)
    common = ["--run", str(run.run_dir), "--keys-file", "none", "--heldout", "2", "--no-renders",
              "--timesteps", "500", "--noise-seeds", "0"]
    script.main([*common, "--out", str(tmp_path / "latest")])               # --checkpoint defaults to latest
    latest = read_results(tmp_path / "latest")
    model, ckpt = models[-1]
    assert latest["model"]["step"] == 3 and latest["model"]["weights"] == "ema"
    assert latest["model"]["checkpoint"].endswith("checkpoints/latest.pt")
    ema, raw = (torch.cat([ckpt[name][part][key].flatten() for part in ("date", "lora") for key in sorted(
        ckpt[name][part])]) for name in ("trainable", "raw_trainable"))
    torch.testing.assert_close(flat_weights(model), ema)                    # the EMA, as the run's probes ...
    assert not torch.allclose(ema, raw)                                     # ... not the raw weights beside it
    # the held-out sets are the first of the run's held-out probe, measured at the given timestep and seed
    assert [entry["name"] for entry in latest["sets"]] == ["heldout/0", "heldout/1"]
    assert [entry["rows"] for entry in latest["sets"]] == run.loss_probes["heldout"].rows
    assert latest["settings"]["timesteps"] == [500] and latest["settings"]["noise_seeds"] == [0]
    assert sorted(path.name for path in (tmp_path / "latest").iterdir()) == ["losses.jsonl", "results.json"]
    assert "renders" not in latest and "RENDERS" not in capsys.readouterr().out
    assert any(entry["loss"]["whole"]["gap_swapped"] != 0 for entry in latest["sets"])

    # the released model under the run's settings: the same view sets, no date input, so no gap anywhere
    script.main([*common, "--checkpoint", "released", "--out", str(tmp_path / "released")])
    released = read_results(tmp_path / "released")
    assert models[-1][0].date is None and models[-1][1] == {}
    assert released["model"] == {"run": str(run.run_dir), "checkpoint": None, "step": None, "weights": "released",
                                 "finetune": None, "date_temb_ratio": None}
    assert [entry["rows"] for entry in released["sets"]] == [entry["rows"] for entry in latest["sets"]]
    assert released["settings"] == latest["settings"]
    for entry in released["sets"]:
        for area in AREAS:
            assert entry["loss"][area]["gap_swapped"] == entry["loss"][area]["gap_unknown"] == 0.0
    assert "the released model (no date input)" in capsys.readouterr().out

    # renders only, with the sampler settings given here instead of the run's
    script.main(["--run", str(run.run_dir), "--checkpoint", "3", "--no-losses", "--seeds", "4", "5", "--steps", "1",
                 "--guidance", "1.5", "--uncond-dates", "unknown", "--out", str(tmp_path / "renders"),
                 "--keys-file", str(keys_file(tmp_path)), "--thumb", "16"])
    rendered = read_results(tmp_path / "renders")
    calls = models[-1][0].pipelines[-1].calls
    assert len(calls) == 2 * 2 * 2 and {(call["steps"], call["guidance"]) for call in calls} == {(1, 1.5)}
    assert "loss" not in rendered and not (tmp_path / "renders" / "losses.jsonl").exists()
    assert [record["seed"] for record in rendered["sets"][0]["render_records"]] == [4, 5]
    assert rendered["settings"]["render_uncond_dates"] == "unknown" and rendered["model"]["step"] == 3
    assert (tmp_path / "renders" / "renders" / "modern" / "swapped_seed5_view4.jpg").exists()


def test_the_released_model_without_a_run(folder, tmp_path, monkeypatch, capsys):
    script = load_script("eval_dates")
    models = stand_in_models(script, monkeypatch)
    keys = tmp_path / "keys3.json"
    keys.write_text(json.dumps([{"refs": [f"view_{row:02d}.png" for row in rows[:3]],
                                 "targets": [f"view_{row:02d}.png" for row in rows[3:]]}
                                for rows in (AFTER_FROM_MODERN, MODERN_FROM_MODERN)]))
    out = tmp_path / "out"
    script.main(["--dataset", str(folder), "--resolution", str(SIZE), "--precision", "fp32", "--class-label", "1",
                 "--keys-file", str(keys), "--out", str(out), "--seeds", "0", "--steps", "2", "--timesteps", "500",
                 "--min-mask-pixels", "4", "--thumb", "16"])
    results = read_results(out)
    model, ckpt = models[-1]
    assert model.date is None and ckpt == {} and results["model"]["weights"] == "released"
    assert results["model"]["run"] is None and results["model"]["checkpoint"] is None
    assert results["settings"]["swap_years"] == pytest.approx([1901.75, 2012.35, 2020.35])      # the periods' medians
    assert (results["settings"]["scale"], results["settings"]["loss"]["class_label"]) == ("scene", 1)
    assert [(entry["name"], entry["group"], entry["held_out"]) for entry in results["sets"]] == [
        ("keys3/0", "cross", None), ("keys3/1", "other", None)]
    assert "held out of training" not in capsys.readouterr().out        # no run, no hold-out to speak of
    for entry in results["sets"]:                                       # a model without a date input
        assert all(entry["loss"][area]["gap_swapped"] == 0.0 for area in AREAS)
        assert all(entry["renders"][area]["difference"] == 0.0 and entry["renders"][area]["psnr_gain"] == 0.0
                   for area in AREAS)
    assert list(results["loss"]) == ["cross", "other", "all"] and (out / "sheets" / "keys3_1.jpg").exists()


def test_list_sets_needs_no_model_and_writes_nothing(run, tmp_path, monkeypatch, capsys):
    script = load_script("eval_dates")
    stand_in_models(script, monkeypatch)

    def no_model(*args, **kwargs):
        raise AssertionError("--list-sets must not load a model")

    monkeypatch.setattr(script, "load_finetuned", no_model)
    monkeypatch.setattr(script, "load_mvgen", no_model)
    out = tmp_path / "listed"
    common = ["--run", str(run.run_dir), "--keys-file", str(keys_file(tmp_path)), "--out", str(out), "--list-sets"]
    script.main([*common, "--heldout", "3", "--cross-only"])
    printed = capsys.readouterr().out
    assert not out.exists() and "LOSS" not in printed and "note:" not in printed
    assert "5 view sets: 4 cross" in printed and "other from keys.json" in printed and "swapped to" in printed
    assert printed.count(f"cross from {HELDOUT_SOURCE}") == 3 and "(3 of 3 held out of training)" in printed
    # fewer held-out sets than asked for: said, and the others are listed
    script.main([*common, "--heldout", "2", "--cross-only", "--region-edge", "2050"])
    printed = capsys.readouterr().out
    assert "note: 0 of the 2 held-out view sets asked for were found" in printed and "2 view sets: 0 cross" in printed
    # a dataset that is not the run's is refused before any model is loaded
    heldout = run.run_dir / "heldout.json"
    heldout.write_text(json.dumps(json.loads(heldout.read_text())[1:]))
    with pytest.raises(ValueError, match="not the dataset the run .* trained on"):
        script.main(["--run", str(run.run_dir), "--out", str(out), "--list-sets"])


# ----------------------------------------------------------------------------- added by the review of 2026-11-20
def test_view_sets_that_would_share_a_name_are_refused(folder, tmp_path):
    """Records, pictures and group means are kept by a view set's name (and its folder): two different sets under
    one name would silently share one set's numbers."""
    small = MVGenScene(folder, SMALL, resolution=SIZE)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, second = tmp_path / "a" / "gate.json", tmp_path / "b" / "gate.json"       # the same stem: gate/0 twice
    first.write_text(json.dumps([{"refs": ["view_08.png", "view_09.png"], "targets": ["view_16.png", "view_17.png"]}]))
    second.write_text(json.dumps([{"refs": ["view_08.png", "view_09.png"], "targets": ["view_18.png", "view_19.png"]}]))
    with pytest.raises(ValueError, match=r"share a name or a folder of renders: \['gate/0'\]"):
        collect_view_sets(small, [first, second])
    # names that differ but give one folder of renders ("a/b" and "a_b") clash as well
    second.write_text(json.dumps([{"name": "gate_0", "refs": ["view_08.png", "view_09.png"],
                                   "targets": ["view_18.png", "view_19.png"]}]))
    with pytest.raises(ValueError, match=r"share a name or a folder of renders: \['gate/0', 'gate_0'\]"):
        collect_view_sets(small, [first, second])
    # the same view set twice is no clash: it is evaluated once, under its first name
    second.write_text(first.read_text())
    assert [view_set.name for view_set in collect_view_sets(small, [first, second])] == ["gate/0"]


class CornerDatedUNet(RegionDatedUNet):
    """As ``RegionDatedUNet``, but the date acts in the top-RIGHT quarter: a place that a mask with rows and
    columns exchanged would miss (the top-left quarter of the other stand-ins is its own mirror image)."""

    def forward(self, sample, timestep, encoder_hidden_states=None, add_inputs=None, class_labels=None, coords=None,
                return_dict=True, **unused):
        batch, views, _, height, width = sample.shape
        wrong = ~self.valid | ((self.year - self.true_years).abs() > 0.01)
        block = torch.zeros(height, width)
        block[:height // 2, width // 2:] = 1.0
        offset = self.gain * wrong.float().repeat(batch).view(-1, 1, 1, 1) * block
        return (torch.zeros(batch * views, 4, height, width) + offset,)


def test_a_mask_is_laid_on_the_error_map_and_the_renders_the_right_way_round(scene, tmp_path):
    probe = fire_probe(scene)
    base = tiny_model()
    model = RegionDatedMVGen(unet=CornerDatedUNet(probe.sample["date_mid"]).eval(), vae=base.vae, config=base.config,
                             scheduler=base.scheduler)

    def quarter(rows: slice, columns: slice) -> TargetMasks:
        masks = np.zeros((3, SIZE, SIZE), dtype=bool)
        masks[:, rows, columns] = True
        return TargetMasks(masks=masks, pixels=masks.sum(axis=(1, 2)))

    first, second = slice(None, SIZE // 2), slice(SIZE // 2, None)             # the two halves of a frame's side
    top_right, bottom_left = quarter(first, second), quarter(second, first)
    for masks, on_the_date in ((top_right, True), (bottom_left, False)):
        losses = summarise_set_losses(measure_losses(model, probe, masks, (200, 800), (0,)))
        assert (losses["inside"]["gap_swapped"] > 1) == on_the_date
        assert (losses["outside"]["gap_swapped"] == 0) == on_the_date
        renders = summarise_set_renders(render_set(model, model.sampling_pipeline(), probe, masks, tmp_path / "kept",
                                                   tmp_path / "sheet.jpg", thumb=16))
        assert (renders["inside"]["difference"] > 0.05) == on_the_date
        assert (renders["outside"]["difference"] == 0) == on_the_date


@pytest.mark.parametrize("rule", ["keep", "drop_all"])
def test_the_script_reproduces_the_numbers_of_the_runs_own_probes(folder, tmp_path, monkeypatch, rule):
    """The whole chain against the trainer's records: a run with both probes, then the script on its last
    checkpoint. Over the whole frame the evaluation must measure what the run's probes logged at that step: the
    render probe's PSNR per target and, once it is given the date-loss probe's noise, that probe's three losses.
    That holds only if the scene, the hold-out, the EMA weights, the domain label, the dates shown, the swap
    years and the cross-date rule are all the run's."""
    cfg = config(folder, tmp_path / "run", sampling=SMALL, ema_decay=0.9, probe_every=3, render_every=3,
                 render_rows=((8, 9, 16, 17),), render_seeds=(0, 1), save_every=3, latest_every=3,
                 loss=LossConfig(precision="fp32", class_label=1, cross_date_pixels=rule, date_views="targets"),
                 holdout=HoldoutConfig(fraction=0.5, probe_targets=True, seed=1, era_edges=EDGES))
    trainer = Trainer(cfg, model=tiny_model())
    trainer.train()
    run_dir, probe = trainer.run_dir, trainer.loss_probes["heldout"]
    logged = json.loads((run_dir / "probes" / "heldout_losses.jsonl").read_text().splitlines()[-1])
    rendered = json.loads((run_dir / "renders" / "step_000003" / "set0.json").read_text())["renders"]
    assert logged["step"] == 3

    def probe_noise(noise_seed, timestep, rows, shape):
        """The noise ``DateLossProbe`` draws for one of its view sets: seed + position, one draw per timestep."""
        generator = torch.Generator().manual_seed(probe.seed + probe.rows.index([int(row) for row in rows]))
        draws = [torch.randn(tuple(shape), generator=generator) for _ in probe.timesteps]
        return draws[probe.timesteps.index(timestep)]

    monkeypatch.setattr("timemachine.mvgen.evaluation.loss_noise", probe_noise)
    script = load_script("eval_dates")
    stand_in_models(script, monkeypatch)
    common = ["--run", str(run_dir), "--checkpoint", "3", "--noise-seeds", "0", "--min-mask-pixels", "4"]
    script.main([*common, "--keys-file", "none", "--heldout", "2", "--no-renders", "--out", str(tmp_path / "losses")])
    lines = [json.loads(line) for line in (tmp_path / "losses" / "losses.jsonl").read_text().splitlines()]
    for i in range(2):
        for variant in ("true", "swapped", "unknown"):
            measured = [line["whole"] for line in lines if (line["set"], line["variant"]) == (f"heldout/{i}", variant)]
            assert measured == pytest.approx(logged[variant][i], rel=1e-5)
    assert logged["true"] != logged["swapped"] != logged["unknown"]         # three different measurements

    keys = tmp_path / "keys.json"                                           # the run's render set, by file key
    keys.write_text(json.dumps([{"refs": ["view_08.png", "view_09.png"], "targets": ["view_16.png", "view_17.png"]}]))
    script.main([*common, "--keys-file", str(keys), "--no-losses", "--thumb", "16", "--out", str(tmp_path / "renders")])
    records = read_results(tmp_path / "renders")["sets"][0]["render_records"]
    for theirs in rendered:
        mine = next(record for record in records if record["seed"] == theirs["seed"])
        assert mine["psnr"][theirs["date"]]["whole"] == pytest.approx(theirs["psnr_per_target"], rel=1e-5)
    assert len(rendered) == 4 and rendered[0]["psnr_per_target"] != rendered[2]["psnr_per_target"]


def test_renders_can_be_given_the_true_dates_priors_at_both_dates(scene, tmp_path):
    """Under a cross-date rule the render probe gives a swapped-date render other priors than the true-date one;
    ``priors_by="true"`` holds them fixed, so that the two renders differ in the date alone."""
    cfg = LossConfig(precision="fp32", cross_date_pixels="drop_all", cross_date_gap=1000.0)     # the period alone
    probe = fire_probe(scene, cfg)                          # targets after the fire, references before it
    assert priors_differ(probe) and not priors_differ(fire_probe(scene))            # never under "keep"
    assert not priors_differ(fire_probe(scene, replace(cfg, cross_date_gap=1.0)))   # cross-date at 2015 as well
    model = tiny_model()                                    # the released model: it has no date input at all
    pipeline = model.sampling_pipeline()

    def targets_without_priors() -> list[bool]:
        """Per target of the latest UNet call, in the conditional half: are both of its priors zeros?"""
        coordinates, pixels = (is_zero(prior)[6:] for prior in model.unet.seen[-1]["coords"])
        assert torch.equal(coordinates, pixels)
        return pixels[3:].tolist()

    true = render_at(model, pipeline, probe, "true", 0)
    assert targets_without_priors() == [True] * 3
    as_the_probe = render_at(model, pipeline, probe, "swapped", 0)
    assert targets_without_priors() == [False] * 3          # at 2015 the references are of the targets' period
    fixed = render_at(model, pipeline, probe, "swapped", 0, priors_by="true")
    assert targets_without_priors() == [True] * 3
    assert np.array_equal(render_at(model, pipeline, probe, "true", 0, priors_by="true"), true)
    assert np.array_equal(fixed, true) and not np.array_equal(as_the_probe, true)   # without a date: the priors alone
    with pytest.raises(ValueError, match="priors_by must be one of"):
        render_at(model, pipeline, probe, "true", 0, priors_by="swapped")

    masks = quarter_masks(AFTER_FROM_MODERN[3:])
    by_rule = {rule: summarise_set_renders(render_set(model, pipeline, probe, masks, tmp_path / rule,
                                                      tmp_path / f"{rule}.jpg", thumb=16, priors_by=rule))
               for rule in ("rendered", "true")}
    assert by_rule["rendered"]["whole"]["difference"] > 0 and by_rule["true"]["whole"]["difference"] == 0.0


def test_the_script_says_when_the_two_renders_differ_in_priors_and_can_hold_them_fixed(folder, tmp_path, monkeypatch,
                                                                                         capsys):
    cfg = config(folder, tmp_path / "run", sampling=SMALL,
                 loss=LossConfig(precision="fp32", cross_date_pixels="drop_all", cross_date_gap=1000.0))
    Trainer(cfg, model=tiny_model())                        # writes the run's records; its baseline needs no more
    script = load_script("eval_dates")
    stand_in_models(script, monkeypatch)
    common = ["--run", cfg.run_dir, "--checkpoint", "released", "--keys-file", str(keys_file(tmp_path)),
              "--no-losses", "--seeds", "0", "--thumb", "16", "--min-mask-pixels", "4"]
    capsys.readouterr()
    script.main([*common, "--out", str(tmp_path / "as_the_probe")])
    printed = capsys.readouterr().out
    assert ("note: the run's cross-date rule (drop_all) gives the targets of 2 of the 2 view sets other priors"
            in printed and "(keys/0 modern). --priors-by true" in printed)
    assert "unconditional dates same, cross-date priors by the rendered dates" in printed
    as_the_probe = read_results(tmp_path / "as_the_probe")
    assert as_the_probe["settings"]["render_priors_by"] == "rendered"
    assert as_the_probe["settings"]["sets_with_other_priors_when_swapped"] == ["keys/0", "modern"]
    # the released model has no date input, and still its two renders differ: by their priors
    assert all(entry["renders"]["whole"]["difference"] > 0 for entry in as_the_probe["sets"])

    script.main([*common, "--priors-by", "true", "--out", str(tmp_path / "fixed")])
    printed = capsys.readouterr().out
    assert "note:" not in printed and "cross-date priors by the true dates" in printed
    fixed = read_results(tmp_path / "fixed")
    assert fixed["settings"]["render_priors_by"] == "true"
    assert all(entry["renders"][area]["difference"] == 0.0 for entry in fixed["sets"] for area in AREAS)
    for entry, other in zip(fixed["sets"], as_the_probe["sets"]):           # the true-date render is the same
        assert entry["renders"]["whole"]["psnr_true"] == other["renders"]["whole"]["psnr_true"]
    # --list-sets gives the note as well; a run that keeps all priors never gets it (the other tests of this file)
    script.main([*common, "--out", str(tmp_path / "listed"), "--list-sets"])
    assert "note: the run's cross-date rule (drop_all)" in capsys.readouterr().out
