"""CPU tests for experiment look1: the per-view look code, its conditioner, its training draws and the hold-out of
whole days and photographers.

* the code's arithmetic, its standardisation, the references' code, the medium conversion and the files
  (``timemachine.mvgen.look``, ``scripts/mvgen/prepare_look.py``);
* the look inside the one date conditioner (``timemachine.mvgen.date``): an exact no-op at step 0 in fp32 and
  bf16, the same gradients with and without gradient checkpointing, a forward pass between a loss and its
  ``backward()`` that changes nothing, the same look in both guidance halves; also with the per-block
  scale-and-shift fallback (``look_injection="film"``);
* training (``timemachine.mvgen.step``, ``.trainer``): the look's draws come from generators of their own, so a run
  with the look starts from the same date and LoRA weights and draws the same view sets, timesteps, noise and
  dropouts as the run without it, and a run without it does what it did before the look existed;
* the hold-out by component and the 2 x 2 evaluation sets (``timemachine.data.holdout``), the probes' new
  variants (``timemachine.mvgen.probes``) and the command line (``scripts/mvgen/train.py``).

Nothing of the released model is loaded: the stand-ins are those of ``tests/test_mvgen_trainer.py`` (its tiny
UNet and its folder of 24 tiny views: rows 0-7 old, 8-15 modern, 16-23 after the fire) and the small UNet of
MVGenMaster's own class of ``tests/test_mvgen_model.py``. Run from MVGenMaster's checkout, in its environment.
"""
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

pytest.importorskip("my_diffusers", reason="runs in MVGenMaster's environment, from its checkout")

# The stand-ins, and the fixtures ``fake_conditioning`` (autouse) and ``folder``, which pytest finds here by name.
# Importing that module also lets the ``src`` package find MVGenMaster's modules, so it comes first.
from tests.test_mvgen_trainer import (  # noqa: E402, F401
    AFTER_FROM_MODERN, EDGES, LATENT, MODERN_FROM_MODERN, N, SIZE, SWAP_YEARS, FakePipeline, config,
    fake_conditioning, flat_weights, folder, load_script, read_log, run_files, tiny_model,
)
from tests.test_mvgen_model import YEARS, _predict, _tiny_model  # noqa: E402
from tests.test_mvgen_model import TinyUNet as BlockUNet  # noqa: E402
from tests.test_mvgen_signal import write_regions  # noqa: E402

from timemachine.data.dataset import load_view  # noqa: E402
from timemachine.data.holdout import (  # noqa: E402
    EVAL_SEED_OFFSET, HoldoutConfig, disjoint_sets, era_index, evaluation_cells, evaluation_sets, first_holdout_seed,
    heldout_rows, holdout_units, unit_labels, view_authors,
)
from timemachine.mvgen import model as mvgen_model  # noqa: E402
from timemachine.mvgen import step  # noqa: E402
from timemachine.mvgen import trainer as trainer_module  # noqa: E402
from timemachine.mvgen.data import MVGenScene, read_region_weights  # noqa: E402
from timemachine.mvgen.date import DateConditioner, DateEmbedding, LookEmbedding, dated_blocks  # noqa: E402
from timemachine.mvgen.look import (  # noqa: E402
    CLASS_MAP_FILE, LOOK_CLASSES, LOOK_DIM, LOOK_FIELDS, LOOK_FILE, SEGFORMER, SEGFORMER_REVISION, TONES, LookStats,
    TrainingLooks, class_groups, class_slice, convert_medium, excluded_pixels, grown_cells, look_code, matched_donors,
    read_class_maps, read_look, references_code, season_of, to_image, tone_rule, toned_curve, write_look_file,
)
from timemachine.mvgen.model import (  # noqa: E402
    FinetuneConfig, adapter_settings, load_finetuned, load_trainable_state_dict, prepare_for_finetuning,
    save_checkpoint, trainable_state_dict,
)
from timemachine.mvgen.probes import (  # noqa: E402
    DATE_VARIANTS, DateLossProbe, default_looks, render_views, within_swapped_years, within_years,
)
from timemachine.mvgen.step import LossConfig, training_loss  # noqa: E402
from timemachine.mvgen.trainer import ADDED_FIELDS, TrainConfig, Trainer, resume_differences  # noqa: E402

LOOK_TUNE = dict(date_freqs=4, date_hidden=16, lora_rank=2, look=True, look_hidden=16)
TONED, GREY = 3, 5                          # rows of the tiny folder that are prints


# ----------------------------------------------------------------------------- the tiny folder with look codes
def paint(folder: Path) -> None:
    """Give the tiny folder's views some texture: each its own colour plus noise; row GREY grey, TONED sepia."""
    views = pd.read_parquet(folder / "views.parquet")
    from PIL import Image
    for row, path in enumerate(views["image_path"]):
        rng = np.random.default_rng(row)
        image = np.array([10 * row, 100, 200], dtype=np.float64) + rng.normal(0, 25, (SIZE, 3 * SIZE // 2, 3))
        image[: SIZE // 4] += 40                                    # a lighter "sky"
        if row == GREY:
            image = np.repeat(image.mean(axis=2, keepdims=True), 3, axis=2)
        if row == TONED:
            light = image.mean(axis=2, keepdims=True) / 255
            image = 255 * light * np.array([1.0, 0.85, 0.6])
        Image.fromarray(image.clip(0, 255).astype(np.uint8)).save(path)


def write_look(folder: Path, size: int = SIZE) -> dict:
    """Write class maps (8 x 8 cells: sky on top, vegetation in every second view, building elsewhere) and the look
    codes of every view at crops of ``size``, as prepare_look.py makes them; returns ``read_look`` of the folder."""
    views = pd.read_parquet(folder / "views.parquet")
    keys = views["file_key"].to_numpy().astype(str)
    labels = np.full((N, 8, 8), 1, dtype=np.uint8)                  # ADE20K 1: building
    labels[:, :2] = 2                                               # 2: sky
    labels[::2, 6:, :4] = 4                                         # 4: tree (vegetation)
    np.savez(folder / CLASS_MAP_FILE, labels=labels, file_keys=keys, model=SEGFORMER, revision=SEGFORMER_REVISION,
             computed=np.ones(N, dtype=bool), resolution=size)
    regions = read_region_weights(folder, keys, size)
    region = np.zeros((N, size // 8, size // 8), dtype=bool)
    if regions is not None:
        for row in range(N):
            if (regions[row] > 0).any():
                region[row] = grown_cells(regions[row] > 0)
    tones = np.array(["colour"] * N, dtype=object)
    tones[TONED], tones[GREY] = "toned", "grey"
    codes, shown = np.zeros((N, LOOK_DIM), dtype=np.float32), np.zeros((N, len(LOOK_CLASSES)), dtype=bool)
    curves = []
    for row in range(N):
        image = to_image(load_view(views.iloc[row], size)[0])
        codes[row], shown[row] = look_code(image, class_groups(labels[row]), excluded_pixels(region[row], size),
                                           tones[row])
        if tones[row] == "toned":
            curves.append(toned_curve(image))
    stats = LookStats.from_codes(codes)
    write_look_file(folder, {
        "codes": codes, "shown": shown, "tone": tones.astype(str), "tone_source": np.array(["labels"] * N),
        "computed": np.ones(N, dtype=bool), "region": region, "file_keys": keys,
        "toned_curves": np.asarray(curves, dtype=np.float32), "toned_rows": np.array([TONED]),
        "stats_mean": np.asarray(stats.mean), "stats_sd": np.asarray(stats.sd)}, {"resolution": size})
    return read_look(folder, keys)


@pytest.fixture
def look_folder(folder) -> Path:
    """The tiny folder with textured views, changed regions on rows 8-23 and look codes."""
    paint(folder)
    write_regions(folder, half_cell=False)
    write_look(folder)
    return folder


@pytest.fixture
def look_scene(look_folder) -> MVGenScene:
    return MVGenScene(look_folder, resolution=SIZE)


def look_model(**finetune) -> nn.Module:
    """The tiny stand-in with date, LoRA and look adapters, the look's output layers made non-zero."""
    model = tiny_model()
    prepare_for_finetuning(model, FinetuneConfig(**{**LOOK_TUNE, **finetune}))
    torch.manual_seed(1)
    nn.init.normal_(model.date.embedding.mlp[-1].weight, std=0.5)
    perturb_look(model.date.look)
    return model


def perturb_look(look: LookEmbedding, std: float = 0.5) -> None:
    """Make a look embedding's zero-initialised output layers non-zero, as after some training."""
    torch.manual_seed(4)
    layers = [look.mlp[-1]] if look.injection == "temb" else list(look.film)
    for layer in layers:
        nn.init.normal_(layer.weight, std=std)
        nn.init.normal_(layer.bias, std=std)


# ----------------------------------------------------------------------------- the code
def test_look_code_is_the_frame_outside_the_region():
    rng = np.random.default_rng(0)
    image = rng.random((16, 16, 3)).astype(np.float32)
    groups = np.full((4, 4), 1, dtype=np.uint8)                     # building
    groups[0] = 0                                                   # sky: the top quarter
    groups[3, 3] = 3                                                # ground: one cell, 1/16 of the frame
    region = np.zeros((16, 16), dtype=bool)
    region[8:, :8] = True
    code, shown = look_code(image, groups, region, "toned")
    keep = ~region
    assert code.shape == (LOOK_DIM,) and len(LOOK_FIELDS) == LOOK_DIM == 24 and code.dtype == np.float32
    np.testing.assert_allclose(code[:3], image[keep].mean(0), rtol=1e-6)
    np.testing.assert_allclose(code[3:6], np.log(image[keep].std(0) + 1e-3), rtol=1e-6)
    np.testing.assert_allclose(code[class_slice(0)], image[:4][keep[:4]].mean(0), rtol=1e-6)
    assert shown.tolist() == [True, True, False, True, False]
    assert np.isnan(code[class_slice(2)]).all() and np.isnan(code[class_slice(4)]).all()
    assert code[21:].tolist() == [0.0, 1.0, 0.0]                    # one-hot of TONES: grey, toned, colour

    tiny = np.full((4, 4), 1, dtype=np.uint8)
    tiny_image = rng.random((100, 100, 3)).astype(np.float32)
    big = np.repeat(np.repeat(tiny, 25, 0), 25, 1)
    big[0, 0] = 0                                                   # 1 pixel of 10,000: under 1% of the frame
    assert not look_code(tiny_image, big, None, "colour")[1][0]
    everything = look_code(image, groups, np.ones((16, 16), dtype=bool), "colour")[0]
    np.testing.assert_allclose(everything[:3], image.reshape(-1, 3).mean(0), rtol=1e-6)    # nothing left: the frame
    with pytest.raises(ValueError, match="tone"):
        look_code(image, groups, None, "sepia")


def test_grown_region_and_class_groups():
    cells = np.zeros((4, 4), dtype=bool)
    cells[0, 0] = True
    assert grown_cells(cells).sum() == 4 and grown_cells(cells)[1, 1]
    assert excluded_pixels(np.zeros((4, 4)), 32) is None
    assert excluded_pixels(cells, 32).sum() == 64
    assert class_groups(np.array([2, 1, 4, 6, 21, 7, 0], dtype=np.uint8)).tolist() == [0, 1, 2, 3, 4, 255, 1]


def test_look_stats_standardise_and_fill_the_population_mean():
    codes = np.full((3, LOOK_DIM), 0.5, dtype=np.float32)
    codes[:, 0] = [0.2, 0.4, 0.6]
    codes[0, class_slice(1)] = np.nan
    codes[:, class_slice(4)] = np.nan                               # a class no view shows
    stats = LookStats.from_codes(codes)
    assert stats.mean[0] == pytest.approx(0.4) and stats.sd[0] == pytest.approx(np.std([0.2, 0.4, 0.6]))
    assert stats.sd[1] == 1.0 and stats.mean[class_slice(4).start] == 0.0       # no spread / no value
    z = stats.standardise(codes)
    assert z.dtype == np.float32 and np.isfinite(z).all()
    assert z[0, class_slice(1)].tolist() == [0.0, 0.0, 0.0]          # unshown -> the population mean
    np.testing.assert_allclose(z[:, 0], [-1.2247449, 0.0, 1.2247449], atol=1e-6)
    assert LookStats.from_dict(json.loads(json.dumps(stats.to_dict()))) == stats
    with pytest.raises(ValueError):
        LookStats(mean=(0.0,) * 23, sd=(1.0,) * 23)


def test_references_code_is_the_inference_default():
    codes = np.zeros((3, LOOK_DIM), dtype=np.float32)
    codes[:, :6] = [[0.1] * 6, [0.2] * 6, [0.6] * 6]
    codes[:, class_slice(0)] = [[1, 1, 1], [np.nan] * 3, [0, 0, 0]]
    codes[:, class_slice(1)] = [[0.5] * 3] * 3
    codes[:, class_slice(2)] = np.nan
    codes[:, 21:] = [[0, 0, 1], [1, 0, 0], [1, 0, 0]]               # colour, grey, grey
    shown = np.isfinite(codes[:, 6:21:3])
    out = references_code(codes, shown)
    np.testing.assert_allclose(out[:6], 0.3, rtol=1e-6)
    assert out[class_slice(0)].tolist() == [0.5] * 3                # only the references that show the sky
    assert np.isnan(out[class_slice(2)]).all()                       # none does: the population mean later
    assert out[21:].tolist() == [1, 0, 0]                            # grey, by majority
    codes[1, 21:] = [0, 1, 0]                                        # a tie of all three: the earliest reference's
    assert references_code(codes, shown)[21:].tolist() == [0, 0, 1]


def test_medium_conversion_tone_curve_and_tone_rule():
    rng = np.random.default_rng(1)
    image = np.stack([rng.random((64, 64)), rng.random((64, 64)) * 0.5, rng.random((64, 64)) * 0.2], -1)
    image = image.astype(np.float32)
    grey = convert_medium(image, "grey")
    assert np.array_equal(grey[..., 0], grey[..., 1]) and np.array_equal(grey[..., 1], grey[..., 2])
    np.testing.assert_allclose(grey[..., 0], image @ np.array([0.299, 0.587, 0.114], np.float32), rtol=1e-6)
    sepia = np.array([1.0, 0.8, 0.55], dtype=np.float32)
    print_ = np.clip(grey * sepia, 0, 1)                             # a toned print: colour a function of lightness
    curve = toned_curve(print_)
    assert curve.shape == (256, 3) and (curve[:, 0] >= curve[:, 2]).all()
    toned = convert_medium(image, "toned", curve)
    assert toned.shape == image.shape and (toned[..., 0] >= toned[..., 2] - 1e-6).all()
    assert tone_rule(grey) == "grey" and tone_rule(image) == "colour" and tone_rule(print_) == "toned"
    with pytest.raises(ValueError):
        convert_medium(image, "toned")
    with pytest.raises(ValueError):
        convert_medium(image, "infrared")


def test_matched_donors_prefer_tone_then_sky_then_season():
    codes = np.zeros((5, LOOK_DIM), dtype=np.float32)
    codes[:, class_slice(0)] = [[0.5] * 3, [0.5] * 3, [0.55] * 3, [0.51] * 3, [np.nan] * 3]
    shown = np.isfinite(codes[:, 6:21:3])
    look = {"codes": codes, "shown": shown, "tone": np.array(["colour", "grey", "colour", "colour", "colour"])}
    season = np.array([0.5, 0.5, 0.5, 0.0, 0.5])
    stats = LookStats(mean=(0.0,) * LOOK_DIM, sd=(0.1,) * LOOK_DIM)
    candidates = np.array([False, True, True, True, True])
    assert matched_donors([0], candidates, look, season, stats).tolist() == [2]    # 1: grey; 3: winter; 4: no sky
    assert matched_donors([0], np.array([False, True, False, False, False]), look, season, stats).tolist() == [1]
    assert matched_donors([0], np.zeros(5, dtype=bool), look, season, stats).tolist() == [-1]
    assert np.isnan(season_of(np.array([2019.0]), np.array([2020.0])))[0]
    assert season_of(np.array([2019.5]), np.array([2019.502]))[0] == pytest.approx(0.501)


def test_look_files_are_tied_to_their_view_table(look_folder, capsys):
    look = read_look(look_folder)
    assert look["codes"].shape == (N, LOOK_DIM) and look["stats"] == LookStats.from_codes(look["codes"])
    assert set(look["tone"]) == set(TONES) and look["toned_curves"].shape == (1, 256, 3)
    assert read_look(look_folder, ["other"] * N) is None and "another view table" in capsys.readouterr().out
    assert read_class_maps(look_folder).shape == (N, 8, 8) and read_class_maps(look_folder, ["other"] * N) is None
    assert read_look(look_folder.parent) is None
    scene = MVGenScene(look_folder, resolution=SIZE)
    sample = scene.make_sample(np.array(AFTER_FROM_MODERN), 3)
    torch.testing.assert_close(sample["look_raw"], torch.from_numpy(look["codes"][list(AFTER_FROM_MODERN)]),
                               rtol=0, atol=0, equal_nan=True)
    assert sample["look_shown"].shape == (6, 5) and sample["tone"] == ["colour"] * 6
    assert scene.look_fingerprint() == MVGenScene(look_folder, resolution=SIZE).look_fingerprint()
    other_size = MVGenScene(look_folder, resolution=2 * SIZE, require_depth=False)
    assert other_size.look is None and "not used" in capsys.readouterr().out


# ----------------------------------------------------------------------------- the conditioner
@pytest.mark.parametrize("injection", ["temb", "film"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_look_is_an_exact_noop_at_init(injection, precision):
    """MVGenMaster's own UNet class: with the date and the look at their zero start the output is the base model's,
    bit for bit, with looks known or unknown and in a guidance-doubled batch."""
    model = _tiny_model()
    context = (torch.autocast("cpu", dtype=torch.bfloat16) if precision == "bf16"
               else torch.autocast("cpu", enabled=False))
    with torch.no_grad(), context:
        base = _predict(model, batch=2)
    prepare_for_finetuning(model, FinetuneConfig(look=True, look_injection=injection))
    looks = torch.randn(3, LOOK_DIM)
    with torch.no_grad(), context:
        with model.dates(YEARS[:3], look=looks, look_known=torch.tensor([True, False, True])):
            assert torch.equal(_predict(model, batch=2), base)
        with model.dates(YEARS[:3], uncond="unknown"):              # every look unknown
            assert torch.equal(_predict(model, batch=2), base)


def _look_gradient(injection: str, checkpointing: bool, between: bool) -> torch.Tensor:
    """Gradient of every adapter (date, LoRA, look) through MVGenMaster's forward pass; ``between``: another dated
    forward pass, with other dates and looks, runs between the loss and its backward()."""
    model = _tiny_model()
    groups = prepare_for_finetuning(model, FinetuneConfig(lora_rank=2, look=True, look_injection=injection))
    torch.manual_seed(2)
    nn.init.normal_(model.date.embedding.mlp[-1].weight, std=0.5)
    perturb_look(model.date.look, std=0.2)
    model.unet.train(checkpointing)
    looks = torch.randn(3, LOOK_DIM, generator=torch.Generator().manual_seed(5))
    with model.dates(YEARS[:3], torch.tensor([True, True, False]), look=looks,
                     look_known=torch.tensor([True, False, True])):
        out = _predict(model)
    if between:
        with model.dates(YEARS[3:6], look=-looks):
            _predict(model, batch=2).pow(2).mean().backward()       # a probe or a second loss term
        for parameters in groups.values():
            for p in parameters:
                p.grad = None
    out.pow(2).mean().backward()
    parameters = groups["date"] + groups["lora"] + groups["look"]
    assert all(p.grad is not None for p in parameters) and groups["look"]
    assert all(p.grad.abs().sum() > 0 for p in groups["look"] if p is not model.date.look.unknown) \
        and model.date.look.unknown.grad.abs().sum() > 0            # also the unknown look of the second view
    return torch.cat([p.grad.flatten() for p in parameters])


@pytest.mark.parametrize("injection", ["temb", "film"])
def test_look_gradients_survive_checkpointing_and_other_passes(injection):
    plain = _look_gradient(injection, checkpointing=False, between=False)
    checkpointed = _look_gradient(injection, checkpointing=True, between=False)
    torch.testing.assert_close(checkpointed, plain, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(_look_gradient(injection, checkpointing=True, between=True), checkpointed,
                               rtol=0, atol=0)


def _block_unet(injection: str, checkpointed: bool = False) -> tuple[BlockUNet, DateConditioner]:
    """Two real ResBlocks whose views do not interact, with a date conditioner that carries a trained look."""
    torch.manual_seed(0)
    unet = BlockUNet()
    unet.checkpointed = checkpointed
    conditioner = DateConditioner(unet, DateEmbedding(32))
    look = LookEmbedding(32, LOOK_DIM, 16, injection, conditioner.film_channels() if injection == "film" else ())
    conditioner.attach_look(look)
    perturb_look(look)
    return unet, conditioner


@pytest.mark.parametrize("injection", ["temb", "film"])
def test_each_view_gets_its_own_look_and_both_guidance_halves_the_same(injection):
    unet, conditioner = _block_unet(injection)
    torch.manual_seed(1)
    sample, t = torch.randn(1, 8, 8, 4, 4), torch.full((1,), 500)
    looks = torch.randn(8, LOOK_DIM)
    with torch.no_grad():
        with conditioner.dates(YEARS, look=looks):
            a = unet(sample, t)
        changed = looks.clone()
        changed[5] += 1.0
        with conditioner.dates(YEARS, look=changed):
            b = unet(sample, t)
    assert (~torch.isclose(a, b, rtol=0, atol=1e-6)).flatten(1).any(1).tolist() == [i == 5 for i in range(8)]

    seen = []
    conditioner.look.register_forward_hook(lambda module, args, output: seen.append((args[0].clone(), args[1].clone())))
    doubled = sample.repeat(2, 1, 1, 1, 1)
    with torch.no_grad():
        for uncond in ("same", "unknown"):
            with conditioner.dates(YEARS, look=looks, look_known=torch.arange(8) != 2, uncond=uncond):
                out = unet(doubled, t.repeat(2))
            code, known = seen[-1]
            assert torch.equal(code[:8], code[8:]) and torch.equal(known[:8], known[8:])
            assert torch.equal(code[:8], looks) and known[:8].tolist() == [i != 2 for i in range(8)]
            if uncond == "same":
                assert torch.equal(out[:8], out[8:])
    assert conditioner.last_stats is not None and set(conditioner.last_stats) == {"date_rms", "temb_rms", "look_rms",
                                                                                  "sum_rms"}


def test_looks_are_validated_and_a_model_without_a_look_ignores_them():
    unet, conditioner = _block_unet("temb")
    for bad in (torch.zeros(7, LOOK_DIM), torch.zeros(8, LOOK_DIM - 1), torch.full((8, LOOK_DIM), float("nan"))):
        with pytest.raises(ValueError, match="look"), conditioner.dates(YEARS, look=bad):
            pass
    with pytest.raises(ValueError, match="no look was given"), \
            conditioner.dates(YEARS, look_known=torch.ones(8, dtype=bool)):
        pass
    with pytest.raises(ValueError, match="already"):
        conditioner.attach_look(LookEmbedding(32, LOOK_DIM, 16))
    plain = DateConditioner(BlockUNet(), DateEmbedding(32))
    with pytest.raises(ValueError, match="carries no look"), plain.dates(YEARS, look=torch.zeros(8, LOOK_DIM)):
        pass
    with pytest.raises(ValueError, match="width"):
        plain.attach_look(LookEmbedding(16, LOOK_DIM, 16))
    with pytest.raises(ValueError, match="channels"):
        plain.attach_look(LookEmbedding(32, LOOK_DIM, 16, "film", [4, 4]))

    model = tiny_model()                                             # no look: MVGen.dates drops the looks
    prepare_for_finetuning(model, FinetuneConfig(date_freqs=4, date_hidden=16))
    with model.dates(YEARS[:6], look=torch.zeros(6, LOOK_DIM)):
        assert not model.has_look
    with pytest.raises(ValueError, match="look needs date"):
        FinetuneConfig(date=False, look=True)
    with pytest.raises(ValueError, match="look_injection"):
        FinetuneConfig(look=True, look_injection="tokens")


def test_look_sizes_are_logged_per_pass():
    model = look_model()
    with torch.no_grad(), model.dates(YEARS[:6], look=torch.ones(6, LOOK_DIM)):
        model.unet(torch.zeros(1, 6, 4, 4, 4), torch.tensor([500]), None, add_inputs=torch.zeros(1, 6, 7, 32, 32),
                   class_labels=torch.zeros(1, dtype=torch.long),
                   coords=[torch.zeros(6, 193, 4, 4), torch.zeros(6, 4, 32, 32)])
    size, date = step.look_embedding_size(model), step.date_embedding_size(model)
    stats = model.date.last_stats
    assert size["look_rms"] == stats["look_rms"] > 0 and size["look_temb_ratio"] == pytest.approx(
        stats["look_rms"] / stats["temb_rms"])
    assert size["sum_temb_ratio"] == pytest.approx(stats["sum_rms"] / stats["temb_rms"])
    assert date["date_temb_ratio"] == pytest.approx(stats["date_rms"] / stats["temb_rms"])
    assert step.look_embedding_size(tiny_model())["look_rms"] is None


# ----------------------------------------------------------------------------- adapters and checkpoints
@pytest.mark.parametrize("injection", ["temb", "film"])
def test_look_is_created_after_lora_and_round_trips_through_a_checkpoint(injection, tmp_path, monkeypatch):
    weights = []
    for look in (False, True):
        model = _tiny_model()
        torch.manual_seed(7)
        groups = prepare_for_finetuning(model, FinetuneConfig(lora_rank=2, look=look, look_injection=injection))
        state = trainable_state_dict(model)
        weights.append(torch.cat([v.flatten() for part in ("date", "lora") for _, v in sorted(state[part].items())]))
    assert torch.equal(weights[0], weights[1])                       # the same date and LoRA weights with the look
    assert set(state) == {"date", "lora", "look"} and all(k.startswith("embedding.") for k in state["date"])
    assert len(groups["look"]) == len(list(model.date.look.parameters()))
    count = sum(p.numel() for p in groups["look"])
    assert count == (24 * 512 + 512 + 24 + (512 * 128 + 128 if injection == "temb" else
                                           sum(2 * c * 513 for c in model.date.film_channels())))

    cfg = FinetuneConfig(lora_rank=2, look=True, look_injection=injection)
    perturb_look(model.date.look, std=0.2)
    with pytest.raises(ValueError, match="look_stats"):
        save_checkpoint(tmp_path / "x.pt", model, cfg, step=1)
    model.look_stats = LookStats(mean=(0.1,) * LOOK_DIM, sd=(2.0,) * LOOK_DIM)
    looks = torch.randn(3, LOOK_DIM)
    with torch.no_grad(), model.dates(YEARS[:3], look=looks):
        trained = _predict(model)
    save_checkpoint(tmp_path / "step_1.pt", model, cfg, step=1)
    monkeypatch.setattr(mvgen_model, "load_mvgen", lambda model_dir, device: _tiny_model())
    restored, ckpt = load_finetuned(tmp_path / "step_1.pt", device="cpu")
    assert restored.look_stats == model.look_stats and ckpt["look_stats"] == model.look_stats.to_dict()
    with torch.no_grad(), restored.dates(YEARS[:3], look=looks):
        torch.testing.assert_close(_predict(restored), trained, rtol=0, atol=0)
    no_look = _tiny_model()
    prepare_for_finetuning(no_look, FinetuneConfig(lora_rank=2))
    with pytest.raises(ValueError, match="disagree on the look"):
        load_trainable_state_dict(no_look, trainable_state_dict(model))
    with pytest.raises(ValueError, match="disagree on the look"):
        load_trainable_state_dict(model, trainable_state_dict(no_look))
    assert adapter_settings(cfg) != adapter_settings(FinetuneConfig(lora_rank=2))
    assert "look" not in adapter_settings(FinetuneConfig(lora_rank=2))          # unchanged for runs without it


# ----------------------------------------------------------------------------- training draws
def training_looks(scene: MVGenScene, **options) -> TrainingLooks:
    stats = LookStats.from_codes(scene.look["codes"][scene.target_ok])
    settings = {"dropout": 0.5, "unspecified": 0.3, **options}
    converts = settings.get("grey", 0) + settings.get("toned", 0) > 0
    return TrainingLooks(scene.look, scene.target_ok, stats, seed=0,
                         class_maps=read_class_maps(scene.dataset_dir) if converts else None, **settings)


def test_training_looks_fill_specify_and_drop_from_their_own_generators(look_scene):
    looks = training_looks(look_scene)
    stats, codes = looks.stats, look_scene.look["codes"]
    shown = look_scene.look["shown"]
    vegetation = class_slice(2)
    unspecified, drops, filled, fills = [], [], 0, set()
    for index in range(200):
        sample = look_scene.make_sample(*look_scene.draw(index))
        before = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in sample.items()}
        out = looks(sample, 0, index)
        for key, value in before.items():                             # the sample given is not changed
            if torch.is_tensor(value):
                torch.testing.assert_close(value, sample[key], rtol=0, atol=0, equal_nan=True)
            else:
                assert value == sample[key]
        assert out["look"].shape == (len(sample["rows"]), LOOK_DIM) and torch.isfinite(out["look"]).all()
        again = looks(look_scene.make_sample(*look_scene.draw(index)), 0, index)
        assert torch.equal(again["look"], out["look"]) and again["look_drop"] == out["look_drop"]
        z = out["look"].numpy()
        own = stats.standardise(sample["look_raw"].numpy())
        for view, row in enumerate(sample["rows"].tolist()):
            for c in np.flatnonzero(shown.any(axis=0)):              # classes no view shows stay at the mean
                entry = z[view, class_slice(c)]
                if (entry == 0).all():
                    unspecified.append(True)
                    continue
                unspecified.append(False)
                if shown[row, c]:
                    np.testing.assert_allclose(entry, own[view, class_slice(c)], rtol=1e-6)
                elif c == 2:                                          # an unshown class: some donor's entry
                    filled += 1
                    fills.add(tuple(np.round(entry, 5)))
                    shifted = np.zeros((1, LOOK_DIM), np.float32)
                    sd, mean = np.asarray(stats.sd)[vegetation], np.asarray(stats.mean)[vegetation]
                    shifted[0, vegetation] = entry * sd + mean
                    assert any(np.allclose(shifted[0, vegetation], codes[d, vegetation], atol=1e-5)
                               for d in np.flatnonzero(shown[:, 2] & look_scene.target_ok))
            np.testing.assert_allclose(z[view, :6], own[view, :6], rtol=1e-6)     # global and tone untouched
            np.testing.assert_allclose(z[view, 21:], own[view, 21:], rtol=1e-6)
        drops.append(out["look_drop"])
    assert filled > 50 and len(fills) > 1
    assert 0.25 < np.mean(unspecified) < 0.35 and 0.4 < np.mean(drops) < 0.6


def test_medium_conversion_converts_colour_views_only_and_leaves_the_other_draws(look_scene):
    plain = training_looks(look_scene)
    converting = training_looks(look_scene, grey=0.5, toned=0.5)    # every colour view converted
    class_maps, region = read_class_maps(look_scene.dataset_dir), look_scene.look["region"]
    kinds = []
    for index in range(30):
        rows, n_ref = look_scene.draw(index)
        rows = np.concatenate([rows[:-1], [TONED]]) if TONED not in rows else rows
        sample = look_scene.make_sample(rows, n_ref)
        a, b = plain(sample, 1, index), converting(sample, 1, index)
        assert a["look_drop"] == b["look_drop"]
        for view, row in enumerate(rows.tolist()):
            tone = sample["tone"][view]
            if tone != "colour":                                     # prints are left as they are
                assert torch.equal(b["images"][view], sample["images"][view]) and b["tone"][view] == tone
                assert torch.equal(a["look"][view], b["look"][view])
                continue
            kind = b["tone"][view]
            kinds.append(kind)
            converted = to_image(b["images"][view])
            if kind == "grey":
                assert np.allclose(converted[..., 0], converted[..., 1], atol=1e-6)
            code, shown = look_code(converted, class_maps[row], excluded_pixels(region[row], SIZE), kind)
            np.testing.assert_allclose(b["look_raw"][view].numpy(), code, rtol=1e-6, equal_nan=True)
            assert b["look_shown"][view].tolist() == shown.tolist() == sample["look_shown"][view].tolist()
            # the class entries the conversion does not change are specified, filled or not, as without it
            unspecified_a = (a["look"][view, 6:21].reshape(5, 3) == 0).all(1)
            unspecified_b = (b["look"][view, 6:21].reshape(5, 3) == 0).all(1)
            assert torch.equal(unspecified_a & torch.from_numpy(~shown), unspecified_b & torch.from_numpy(~shown))
    assert {"grey", "toned"} <= set(kinds) and "colour" not in kinds
    with pytest.raises(ValueError, match="class maps"):
        training_looks(look_scene, grey=0.1).__class__(look_scene.look, look_scene.target_ok, plain.stats, seed=0,
                                                       dropout=0.5, unspecified=0.3, grey=0.1)


def test_training_step_ties_the_look_to_the_date_and_draws_nothing_more(look_scene):
    sample = training_looks(look_scene)(look_scene.make_sample(np.array(AFTER_FROM_MODERN), 3), 0, 0)
    sample["look_drop"] = True
    for dropout, known in ((1.0, False), (0.0, True)):
        model = look_model()
        cfg = LossConfig(precision="fp32", date_dropout=dropout)
        seen = []
        model.date.look.register_forward_hook(lambda module, args, output: seen.append(args[1].clone()))
        generator = torch.Generator().manual_seed(3)
        loss, info = training_loss(model, sample, cfg, generator)
        assert seen[-1].tolist() == [known] * 6 and info["looks_known"] == known
        assert info["look_rms"] is not None and "sum_temb_ratio" in info
        plain = tiny_model()                                         # the same draws without the look
        prepare_for_finetuning(plain, FinetuneConfig(date_freqs=4, date_hidden=16, lora_rank=2))
        other = torch.Generator().manual_seed(3)
        _, plain_info = training_loss(plain, sample, cfg, other)
        assert torch.equal(other.get_state(), generator.get_state()) and plain_info["timestep"] == info["timestep"]
        assert "look_rms" not in plain_info and "looks_known" not in plain_info
    with pytest.raises(ValueError, match="no look input"):
        training_loss(look_model(), look_scene.make_sample(np.array(AFTER_FROM_MODERN), 3),
                      LossConfig(precision="fp32"), torch.Generator().manual_seed(0))


# ----------------------------------------------------------------------------- the trainer
def look_config(folder: Path, run_dir: Path, **changes) -> TrainConfig:
    return config(folder, run_dir, finetune=FinetuneConfig(**LOOK_TUNE), **changes)


def watch_draws(monkeypatch) -> list[dict]:
    """Record, per training sample, its rows, what the step drew and the state of the step's generator after it."""
    seen, real = [], trainer_module.training_loss

    def watched(model, sample, cfg, generator, *args):
        loss, info = real(model, sample, cfg, generator, *args)
        seen.append({"rows": sample["rows"].tolist(), "state": generator.get_state().clone(),
                     **{key: info[key] for key in ("timestep", "camera_dropped", "priors_dropped",
                                                   "prior_views_dropped", "dates_known")}})
        return loss, info

    monkeypatch.setattr(trainer_module, "training_loss", watched)
    return seen


def test_arms_with_and_without_the_look_start_alike_and_draw_the_same_samples(look_folder, tmp_path, monkeypatch):
    """Arms A (no look), B (the look) and D (the look and medium conversion), one seed: the same date and LoRA
    weights at step 0, and for the first 100 samples the same view sets, timesteps, noise, camera and prior
    dropout (the step's generator ends each sample in the same state)."""
    arms = {"A": config(look_folder, tmp_path / "A", steps=50, seed=5, lr_date=1e-3, lr_lora=1e-3),
            "B": look_config(look_folder, tmp_path / "B", steps=50, seed=5, lr_date=1e-3, lr_lora=1e-3),
            "D": look_config(look_folder, tmp_path / "D", steps=50, seed=5, lr_date=1e-3, lr_lora=1e-3,
                             loss=LossConfig(precision="fp32", medium_grey=0.3, medium_toned=0.2))}
    start, draws = {}, {}
    for name, cfg in arms.items():
        trainer = Trainer(cfg, model=tiny_model())
        start[name] = flat_weights(trainer.model)
        with monkeypatch.context() as patch:
            draws[name] = watch_draws(patch)
            trainer.train()
        assert len(draws[name]) == 100
    assert torch.equal(start["A"], start["B"]) and torch.equal(start["A"], start["D"])
    for name in ("B", "D"):
        for a, b in zip(draws["A"], draws[name]):
            assert a["rows"] == b["rows"] and torch.equal(a["state"], b["state"])
            assert {k: a[k] for k in a if k != "state"} == {k: b[k] for k in b if k != "state"}
    assert any(not d["dates_known"] for d in draws["A"])             # date dropout happened
    log = read_log(tmp_path / "B")
    assert {"look_rms", "look_temb_ratio", "sum_temb_ratio", "looks_known"} <= set(log[-1])
    assert "look_rms" not in read_log(tmp_path / "A")[-1]


def test_a_run_without_the_look_is_not_changed_by_a_look_file(folder, tmp_path):
    """The old path: the same run, with probes, renders and checkpoints, gives the same files with and without the
    folder's look codes (and the samples carry the look entries only with them)."""
    settings = dict(steps=4, seed=2, ema_decay=0.9, probe_every=2, render_every=4, save_every=2, latest_every=2,
                    render_rows=(AFTER_FROM_MODERN,), render_seeds=(0,), holdout=HoldoutConfig(
                        fraction=0.2, seed=1, era_edges=EDGES))
    Trainer(config(folder, tmp_path / "before", **settings), model=tiny_model()).train()
    write_look(folder)
    scene = MVGenScene(folder, resolution=SIZE)
    assert scene.has_look and "look_raw" in scene[0]
    Trainer(config(folder, tmp_path / "after", **settings), model=tiny_model()).train()
    before, after = run_files(tmp_path / "before"), run_files(tmp_path / "after")
    assert set(before) == set(after)
    for name in before:
        if name.endswith(".pt"):
            a, b = (torch.load(tmp_path / run / name, weights_only=False) for run in ("before", "after"))
            assert set(a) == set(b) and "look_stats" not in b and "look_fingerprint" not in b
            for part in ("trainable",):
                assert all(torch.equal(a[part][k][n], b[part][k][n]) for k in a[part] for n in a[part][k])
        elif name == "log.jsonl":
            strip = lambda run: [{k: v for k, v in r.items() if k not in ("sec_per_step", "max_mem_gb")}
                                 for r in read_log(tmp_path / run)]
            assert strip("before") == strip("after")
        elif not name.endswith((".jpg",)):
            text = lambda run: (tmp_path / run / name).read_text().replace(str(tmp_path / run), "RUN")
            assert text("before") == text("after"), name


def test_a_look_run_probes_renders_saves_and_resumes(look_folder, tmp_path, capsys):
    cfg = look_config(look_folder, tmp_path / "run", steps=2, probe_every=2, render_every=2, latest_every=1,
                      probe_within=True, render_rows=(AFTER_FROM_MODERN,), render_seeds=(0,),
                      holdout=HoldoutConfig(fraction=0.2, seed=1, era_edges=EDGES))
    trainer = Trainer(cfg, model=tiny_model())
    assert trainer.model.look_stats == LookStats.from_codes(trainer.scene.look["codes"][trainer.scene.target_ok])
    trainer.train()
    probes = [r for r in read_log(tmp_path / "run") if r.get("probe", "").startswith("date_loss")]
    first = probes[0]
    for key in ("rel_gap_swapped", "rel_gap_unknown", "rel_gap_within", "rel_gap_look"):
        assert first[key] == 0.0, key                                 # step 0: date and look are no-ops
    assert {"loss_within", "loss_look_other", "loss_look_same", "look_rms", "sum_temb_ratio"} <= set(first)
    assert "within-period dates" in capsys.readouterr().out
    described = json.loads((tmp_path / "run" / "probes.json").read_text())
    donors = described["date_loss"]["train"][0]["look_donors"]
    assert len(donors["look_other"]) == len(donors["look_same"]) == 5
    renders = json.loads(next((tmp_path / "run" / "renders").rglob("set0.json")).read_text())
    assert len(renders["look"]) == len(AFTER_FROM_MODERN)
    latest = torch.load(tmp_path / "run" / "checkpoints" / "latest.pt", weights_only=False)
    assert latest["look_stats"] == trainer.model.look_stats.to_dict()
    assert latest["look_fingerprint"] == trainer.scene.look_fingerprint() and "look" in latest["trainable"]

    resumed = Trainer(replace(cfg, steps=3), model=tiny_model(), resume=tmp_path / "run" / "checkpoints" / "latest.pt")
    assert resumed.step == 2
    look = read_look(look_folder)
    codes = look["codes"].copy()
    codes[0, 0] += 0.01
    arrays = {k: v for k, v in look.items() if k not in ("config", "stats")}
    arrays.update(codes=codes, stats_mean=np.asarray(look["stats"].mean), stats_sd=np.asarray(look["stats"].sd))
    write_look_file(look_folder, arrays, look["config"])
    with pytest.raises(ValueError, match="look codes"):
        Trainer(replace(cfg, steps=3), model=tiny_model(), resume=tmp_path / "run" / "checkpoints" / "latest.pt")


def test_a_look_run_needs_look_codes_and_a_medium_conversion_needs_the_look(folder, tmp_path):
    with pytest.raises(ValueError, match="look codes"):
        Trainer(look_config(folder, tmp_path / "run"), model=tiny_model())
    with pytest.raises(ValueError, match="needs finetune.look"):
        config(folder, tmp_path / "run", loss=LossConfig(medium_grey=0.1))
    with pytest.raises(ValueError):
        LossConfig(medium_grey=0.7, medium_toned=0.5)


def test_a_saved_config_without_the_look1_settings_resumes():
    current = asdict(TrainConfig(dataset="d", run_dir="r"))
    saved = json.loads(json.dumps(current))
    for name in ADDED_FIELDS:
        part, _, field = name.rpartition(".")
        (saved[part] if part else saved).pop(field)
    assert resume_differences(saved, current) == ([], [])
    flat_defaults = {"holdout.unit_periods": [0]}
    for name, default in ADDED_FIELDS.items():
        part, _, field = name.rpartition(".")
        value = (current[part] if part else current)[field]
        assert (list(value) if isinstance(value, tuple) else value) == flat_defaults.get(name, default), name


# ----------------------------------------------------------------------------- hold-out by component
def unit_table() -> pd.DataFrame:
    """Twelve views: rows 0-5 modern, 6-11 after the fire; days, authors and intervals chosen to link as noted."""
    day = lambda year: dict(date_lo=year - 0.001, date_hi=year + 0.001, date_mid=year)
    rows = [day(2012.5) | {"author": "A"}, day(2012.5) | {"author": "B"},      # 0, 1: the same day
            day(2013.5) | {"author": "B"},                                    # 2: B on another day -> with 0, 1
            dict(date_lo=2010.0, date_hi=2011.0, date_mid=2010.5, author=""),  # 3, 4: unknown author, same interval
            dict(date_lo=2010.0, date_hi=2011.0, date_mid=2010.5, author=""),
            day(2014.5) | {"author": "C"},                                    # 5: alone
            day(2019.4) | {"author": "C"},                                    # 6: C again, but in another period
            day(2019.4) | {"author": "D"}, day(2020.1) | {"author": "D"},     # 7, 8: with 6 by day, D
            day(2021.0) | {"author": ""}, day(2021.0) | {"author": "E"},      # 9, 10: the same day
            day(2022.0) | {"author": "E"}]                                    # 11: E -> with 9, 10
    table = pd.DataFrame(rows)
    table["usable"] = True
    table["file_key"] = [f"v{i}" for i in range(len(table))]
    return table


def partition(units: np.ndarray) -> list[list[int]]:
    """The rows of each unit, the units in the order of their first row."""
    groups: dict[int, list[int]] = {}
    for row, unit in enumerate(units.tolist()):
        groups.setdefault(unit, []).append(row)
    return sorted(groups.values())


def test_holdout_units_link_days_authors_and_intervals_within_a_period():
    table = unit_table()
    units = holdout_units(table, EDGES)
    assert partition(units) == [[0, 1, 2], [3, 4], [5], [6, 7, 8], [9, 10, 11]]
    labels = unit_labels(table, table["author"].to_numpy())
    assert labels[0] == "D:2012-07-02" and labels[3] == "I:|2010.0000|2011.0000"
    # numbered in the order of their labels ("C<period>:<union-find root>"): the order a seed shuffles them in
    assert np.unique(units).tolist() == list(range(5)) and units[0] < units[3] < units[5] < units[6] < units[9]
    table.loc[4, "usable"] = False                                   # an unusable view is a unit of its own
    assert partition(holdout_units(table, EDGES))[:3] == [[0, 1, 2], [3], [4]]
    old = table.assign(date_mid=table["date_mid"] - 100, date_lo=table["date_lo"] - 100, date_hi=table["date_hi"] - 100)
    assert partition(holdout_units(old, EDGES))[:4] == [[0, 1], [2], [3, 4], [5]]    # old: days and intervals only
    assert partition(holdout_units(old, EDGES, day_only_periods=()))[0] == [0, 1, 2]


def test_view_authors_come_from_the_metadata_table(tmp_path, monkeypatch):
    from timemachine import config as config_module
    meta = tmp_path / "tables" / "image_metadata"
    meta.mkdir(parents=True)
    pd.DataFrame({"file_key": ["v0", "v1"], "artist": ["  Jane   Doe ", None]}).to_parquet(meta / "7.parquet")
    real = config_module.DataPaths
    monkeypatch.setattr(config_module, "DataPaths", lambda: real(root=tmp_path))
    views = pd.DataFrame({"file_key": ["v0", "v1", "v2"], "scene_id": 7})
    assert view_authors(views).tolist() == ["Jane Doe", "", ""]
    assert view_authors(views.assign(author=["x", None, "y  z"])).tolist() == ["x", "", "y z"]
    with pytest.raises(FileNotFoundError):
        view_authors(views.assign(scene_id=8))


def test_component_holdout_takes_whole_units_until_views_and_side_views_are_covered():
    dates = np.concatenate([np.full(10, 1900.0), np.full(20, 2012.0), np.full(20, 2020.0)])
    units = np.concatenate([np.arange(10), 10 + np.repeat(np.arange(10) * 2, 2),
                            np.array([30] * 13 + list(range(43, 50)))])             # after: one unit of 13 views
    usable = np.ones(50, dtype=bool)
    side = np.zeros(50, dtype=bool)
    side[[10, 11, 30, 43, 44]] = True
    cfg = HoldoutConfig(fraction=0.1, seed=3, era_edges=EDGES, by="component")
    rows = heldout_rows(dates, cfg, units=units, usable=usable, side=side)
    held = np.isin(np.arange(50), rows)
    assert np.array_equal(rows, heldout_rows(dates, cfg, units=units, usable=usable, side=side))
    for unit in np.unique(units):                                    # whole units only
        assert held[units == unit].all() or not held[units == unit].any()
    assert not held[units == 30].any()                               # larger than max_unit
    assert held[:10].sum() == 1                                      # old: 10% of its units
    modern, after = held[10:30], held[30:]
    assert modern.sum() >= 2 and (held & side)[10:30].sum() >= 1     # >= 10% of views and of side views
    assert after.sum() >= 2 and (held & side)[30:].sum() >= 1
    assert 5 in heldout_rows(dates, cfg, [5], units=units, usable=usable, side=side)
    with pytest.raises(ValueError, match="units"):
        heldout_rows(dates, cfg)
    with pytest.raises(ValueError):
        HoldoutConfig(by="day")
    assert np.array_equal(heldout_rows(dates, replace(cfg, by="view")),
                          heldout_rows(dates, HoldoutConfig(fraction=0.1, seed=3, era_edges=EDGES)))


def test_evaluation_sets_score_held_out_targets_seen_by_every_reference():
    n = 30
    dates = np.concatenate([np.full(15, 2012.0), np.full(15, 2020.0)])
    period = era_index(dates, EDGES)
    covis = np.full((n, n), 100, dtype=np.uint16)
    covis[0, 20:] = covis[20:, 0] = 0                                # view 0 is not seen by after-fire views 20-29
    heldout = np.zeros(n, dtype=bool)
    heldout[[0, 1, 2, 3, 4, 15, 16, 17]] = True
    side = heldout.copy()
    usable = ref_ok = np.ones(n, dtype=bool)
    arrays = dict(heldout=heldout, usable=usable, ref_ok=ref_ok, dates=dates, period=period, covis=covis, side=side)
    sets = evaluation_sets(1, 2, tolerance=5.0, rng=np.random.default_rng(0), **arrays)
    assert sets
    for refs, targets in sets:
        scored = targets[heldout[targets]]
        assert heldout[targets[0]] and side[targets[0]] and len(scored) >= 3 and len(targets) == 5
        assert (period[targets] == 1).all() and (period[refs] == 2).all() and not heldout[refs].any()
        assert (covis[np.ix_(refs, scored)] >= 30).all()
        assert 0 not in scored or not (refs >= 20).any()
        assert (covis[np.ix_(refs, targets)] >= 30).any(axis=0).all()     # together they see every target
    kept = disjoint_sets(sets, heldout)
    used = [t for i in kept for t in sets[i][1] if heldout[t]]
    assert len(used) == len(set(used)) and len(kept) >= 1
    cells = evaluation_cells(4, **arrays)                            # one generator, target period first
    rng = np.random.default_rng(EVAL_SEED_OFFSET + 4)
    for target in (1, 2):
        for reference in (1, 2):
            drawn = evaluation_sets(target, reference, rng=rng, **arrays)
            expected = [drawn[i] for i in disjoint_sets(drawn, heldout)]
            assert [(a.tolist(), b.tolist()) for a, b in cells[reference, target]] == \
                [(a.tolist(), b.tolist()) for a, b in expected]
    seed, records = first_holdout_seed(HoldoutConfig(fraction=0.25, era_edges=EDGES, by="component"), dates=dates,
                                       units=np.arange(n), usable=usable, side=np.ones(n, dtype=bool), ref_ok=ref_ok,
                                       covis=covis, seeds=range(3), min_sets=1, max_share=0.5)
    assert seed == 0 and records[0]["1->2"] >= 1 and records[0]["share_1"] <= 0.5


def component_folder(look_folder: Path) -> Path:
    """The look folder with authors: modern rows 8-9 and 10-11 by one photographer each, 16-19 by another."""
    views = pd.read_parquet(look_folder / "views.parquet")
    views["author"] = [f"p{row}" for row in range(N)]
    views.loc[[8, 9], "author"] = "a"
    views.loc[[10, 11], "author"] = "b"
    views.loc[[16, 17, 18, 19], "author"] = "c"
    views.to_parquet(look_folder / "views.parquet", index=False)
    return look_folder


def test_the_trainer_holds_out_components_records_units_and_refuses_another_holdout(look_folder, tmp_path):
    folder = component_folder(look_folder)
    holdout = HoldoutConfig(fraction=0.1, seed=2, era_edges=EDGES, by="component")
    cfg = look_config(folder, tmp_path / "run", steps=1, latest_every=1, holdout=holdout, probe_every=1,
                      probe_samples=2)
    trainer = Trainer(cfg, model=tiny_model())
    scene = trainer.scene
    units = scene.holdout_units()
    for unit in np.unique(units[scene.heldout]):
        assert scene.heldout[units == unit].all()
    written = json.loads((tmp_path / "run" / "heldout.json").read_text())
    assert [entry["unit"] for entry in written] == units[scene.heldout].tolist()
    assert trainer.model.look_stats == LookStats.from_codes(scene.look["codes"][scene.target_ok])
    trainer.train()
    latest = tmp_path / "run" / "checkpoints" / "latest.pt"
    assert torch.load(latest, weights_only=False)["heldout_rows"] == np.flatnonzero(scene.heldout).tolist()
    Trainer(replace(cfg, steps=2), model=tiny_model(), resume=latest)
    views = pd.read_parquet(folder / "views.parquet")
    views["author"] = [f"q{row}" for row in range(N)]                # the units change under the run
    views.to_parquet(folder / "views.parquet", index=False)
    changed = MVGenScene(folder, resolution=SIZE).holdout_rows(holdout)
    assert not np.array_equal(changed, np.flatnonzero(scene.heldout))
    with pytest.raises(ValueError, match="hold-out by component"):
        Trainer(replace(cfg, steps=2), model=tiny_model(), resume=latest)
    with pytest.raises(ValueError, match="era_edges"):
        MVGenScene(folder, resolution=SIZE).holdout_rows(replace(holdout, era_edges=(2000.0,)))


def test_the_heldout_probe_can_read_its_view_sets_from_a_keys_file(look_folder, tmp_path, capsys):
    keys = pd.read_parquet(look_folder / "views.parquet")["file_key"].tolist()
    sets = [{"refs": [keys[r] for r in (8, 9, 10)], "targets": [keys[r] for r in (16, 17, 18)]}]
    path = tmp_path / "keys.json"
    path.write_text(json.dumps(sets))
    cfg = config(look_folder, tmp_path / "run", probe_every=1, heldout_probe_keys=str(path),
                 holdout=HoldoutConfig(fraction=0.2, seed=1, era_edges=EDGES))
    trainer = Trainer(cfg, model=tiny_model())
    assert trainer.loss_probes["heldout"].rows == [[8, 9, 10, 16, 17, 18]]
    assert "held-out probe: 1 view sets" in capsys.readouterr().out


# ----------------------------------------------------------------------------- probes
def test_within_period_years():
    pairs = ((1880.0, 1940.0), (2013.4, 2016.5), (2019.45, 2021.5))
    dates = np.array([2013.0, 2014.0, 2018.0, 1990.0, 2019.4, 2022.0, 1900.0])
    moved = within_swapped_years(dates, 1, EDGES, pairs)
    assert moved.tolist() == [2013.0, 2016.5, 2013.4, 2016.5, 2021.5, 2019.45, 1940.0]
    years = within_years(np.array([2010.0, 2012.0, 2014.0, 2016.0, 2020.0, 2021.0, 1900.0]), EDGES)
    assert years[1] == (2011.5, 2014.5) and years[0] == (1900.0, 1900.0)
    with pytest.raises(ValueError):
        within_swapped_years(dates, 1, EDGES, pairs[:2])
    with pytest.raises(ValueError, match="two years"):
        TrainConfig(dataset="d", run_dir="r", within_years=((1.0, 2.0),))


def test_date_loss_probe_at_the_own_look_with_within_and_look_variants(look_scene, tmp_path):
    stats = LookStats.from_codes(look_scene.look["codes"][look_scene.target_ok])
    pairs = within_years(look_scene.dates, EDGES)
    sets = [MODERN_FROM_MODERN, AFTER_FROM_MODERN]
    probe = DateLossProbe(look_scene, sets, 3, LossConfig(precision="fp32"), SWAP_YEARS, tmp_path / "b.json",
                          (200, 800), within_years=pairs, look_stats=stats, look_axis=True)
    assert probe.variants == (*DATE_VARIANTS, "within", "look_other", "look_same")
    for donors, rows in zip(probe.donors, sets):
        for target, other, same in zip(rows[3:], donors["look_other"], donors["look_same"]):
            assert look_scene.period[other] != look_scene.period[target] and other not in rows
            assert look_scene.period[same] == look_scene.period[target] and same not in rows
    model = look_model()
    seen = []
    model.date.look.register_forward_hook(lambda module, args, output: seen.append(args[0].clone()))
    result = probe(model)
    own = torch.from_numpy(stats.standardise(probe.samples[0]["look_raw"]))
    assert torch.equal(seen[0], own) and torch.equal(seen[1], own) and torch.equal(seen[2], own)
    assert not torch.equal(seen[4][3:], own[3:]) and torch.equal(seen[4][:3], own[:3])    # a donor's look
    assert result["rel_gap_look"] != 0 and "rel_gap_within" in result and result["region"]["outside"]["sets"] == 2
    assert "rel_gap_look" in result["region"]["outside"] and result["look_rms"] > 0
    assert json.loads((tmp_path / "b.json").read_text())["look_stats"] == stats.to_dict()

    zero = tiny_model()                                              # at step 0 every variant is the same loss
    prepare_for_finetuning(zero, FinetuneConfig(**LOOK_TUNE))
    flat = DateLossProbe(look_scene, sets, 3, LossConfig(precision="fp32"), SWAP_YEARS, tmp_path / "z.json", (200,),
                         within_years=pairs, look_stats=stats, look_axis=True)(zero)
    assert all(flat[key] == 0 for key in ("rel_gap_swapped", "rel_gap_unknown", "rel_gap_within", "rel_gap_look"))
    plain = DateLossProbe(look_scene, sets, 3, LossConfig(precision="fp32"), SWAP_YEARS, tmp_path / "p.json", (200,))
    keys = set(plain(tiny_model()))
    assert not any("within" in key or "look" in key for key in keys)
    with pytest.raises(ValueError, match="look_stats"):
        DateLossProbe(look_scene, sets, 3, LossConfig(), SWAP_YEARS, tmp_path / "q.json", look_axis=True)


def test_renders_give_both_guidance_halves_the_references_code(look_scene):
    stats = LookStats.from_codes(look_scene.look["codes"])
    sample = look_scene.make_sample(np.array(AFTER_FROM_MODERN), 3)
    looks = default_looks(sample, 3, stats)
    expected = stats.standardise(references_code(sample["look_raw"][:3].numpy(), sample["look_shown"][:3].numpy()))
    assert torch.equal(looks[:3], torch.from_numpy(stats.standardise(sample["look_raw"][:3])))
    assert all(torch.equal(looks[view], torch.from_numpy(expected)) for view in range(3, 6))
    model = look_model()
    seen = []
    model.date.look.register_forward_hook(lambda module, args, output: seen.append(args[0].clone()))
    year = sample["date_mid"]
    frames = render_views(model, model.sampling_pipeline(), sample, year, torch.ones(6, dtype=torch.bool), seed=0,
                          steps=2, precision="fp32", uncond_dates="unknown", look=looks)
    assert frames.shape == (6, SIZE, SIZE, 3) and len(seen) == 2
    assert all(torch.equal(code, torch.cat([looks, looks])) for code in seen)


# ----------------------------------------------------------------------------- scripts
def test_train_py_builds_the_look1_settings(look_folder, tmp_path):
    script = load_script("train")
    keys = tmp_path / "keys.json"
    keys.write_text("[]")
    args = script.parse_args([
        "--run-name", "x", "--runs-root", str(tmp_path), "--dataset", str(look_folder), "--look",
        "--look-injection", "film", "--look-hidden", "64", "--lr-look", "2e-4", "--look-dropout", "0.4",
        "--look-class-unspecified", "0.2", "--medium-grey", "0.1", "--medium-toned", "0.05",
        "--holdout-fraction", "0.1",
        "--holdout-by", "component", "--holdout-max-unit", "12", "--holdout-seed", "7", "--holdout-side-share", "0.05",
        "--holdout-unit-periods", "0", "--heldout-probe-keys-file", str(keys), "--probe-within",
        "--within-years", "1880", "1940", "2013.4", "2016.5", "2019.45", "2021.5"])
    cfg = script.build_config(args)
    assert cfg.finetune.look and cfg.finetune.look_injection == "film" and cfg.finetune.look_hidden == 64
    assert (cfg.lr_look, cfg.loss.look_dropout, cfg.loss.look_class_unspecified) == (2e-4, 0.4, 0.2)
    assert (cfg.loss.medium_grey, cfg.loss.medium_toned) == (0.1, 0.05)
    assert cfg.holdout == HoldoutConfig(fraction=0.1, seed=7, era_edges=EDGES, by="component", max_unit=12)
    assert cfg.heldout_probe_keys == str(keys) and cfg.probe_within
    assert cfg.within_years == ((1880.0, 1940.0), (2013.4, 2016.5), (2019.45, 2021.5))
    default = script.build_config(script.parse_args(["--run-name", "x", "--dataset", str(look_folder)]))
    assert not default.finetune.look and default.loss == LossConfig() and default.lr_look == TrainConfig.lr_look
    assert default.holdout.by == "view" and default.heldout_probe_keys is None and not default.probe_within
    with pytest.raises(ValueError, match="two years"):
        script.build_config(script.parse_args(["--run-name", "x", "--within-years", "1", "2", "3"]))


def test_prepare_look_writes_codes_from_cached_class_maps_and_refuses_to_overwrite(folder, tmp_path, capsys):
    paint(folder)
    views = pd.read_parquet(folder / "views.parquet")
    keys = views["file_key"].to_numpy().astype(str)
    labels = np.full((N, 128, 128), 1, dtype=np.uint8)
    labels[:, :32] = 2
    np.savez(folder / CLASS_MAP_FILE, labels=labels, file_keys=keys, model=SEGFORMER, revision=SEGFORMER_REVISION,
             computed=np.ones(N, dtype=bool), resolution=512)
    tones = tmp_path / "tones.npz"
    np.savez(tones, rows=np.array([0, TONED, GREY]), tone=np.array(["colour", "toned", "bw"]))
    script = load_script("prepare_look")
    script.main(["--dataset", str(folder), "--tone-labels", str(tones)])
    look = read_look(folder, keys)
    assert "using the stored class maps" in capsys.readouterr().out
    assert look["computed"].all() and look["tone"][TONED] == "toned" and look["tone"][GREY] == "grey"
    assert look["tone_source"][[0, TONED, GREY]].tolist() == ["labels"] * 3 and look["tone_source"][1] == "rule"
    assert look["toned_rows"].tolist() == [TONED] or TONED in look["toned_rows"]
    assert look["config"]["resolution"] == 512 and look["config"]["regions"] is None
    image = to_image(load_view(views.iloc[7], 512)[0])
    expected, shown = look_code(image, class_groups(labels[7]), None, look["tone"][7])
    np.testing.assert_allclose(look["codes"][7], expected, rtol=1e-6, equal_nan=True)
    assert look["shown"][7].tolist() == shown.tolist() == [True, True, False, False, False]
    with pytest.raises(SystemExit, match="overwrite"):
        script.main(["--dataset", str(folder)])
