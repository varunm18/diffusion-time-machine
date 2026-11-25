#!/usr/bin/env python
"""Render view sets with date-conditioned MVGenMaster at chosen dates and noise seeds (single GPU).

For every view set (its references first) the targets are generated from the references alone, once
per date and seed, and compared with the real target photos. Per set this writes a contact sheet
``<out>/set<i>.jpg`` (real views on top, one row per date and seed) and ``set<i>.json`` (PSNR per date
and seed); ``<out>/results.json`` collects them (``timemachine.mvgen.probes.RenderProbe``).

A date is a year, or ``true`` (each target at its own date), ``swapped`` (each target at its date in
another period) or ``unknown`` (no view is dated).

With ``--checkpoint`` the model is the fine-tuned one, and the dataset, the scene-scale rule, the
number of references, how dates are shown, the domain label, the cross-date rule, the period edges,
the precision and the years for swapped dates default to those of its training run. Without it the
released model is rendered, which has no date input: every date then gives the same pictures, the
baseline to compare with.

A checkpoint of a run with a look code (experiment look1, ``timemachine.mvgen.look``) also gives every
view a look: the references their own codes, the targets the one ``--look`` names: ``references`` (the
default, as at inference: the references' code, ``look.references_code``), ``own`` (each target's own
code), ``neutral`` (the population mean of the training views) or ``donor:<row>`` (the code of view
``<row>`` of the view table, for every target). ``--tone grey|toned|colour`` replaces the targets'
tone in their codes and keeps the rest. The codes come from the dataset folder's ``look_codes.npz``
(``scripts/mvgen/prepare_look.py``) and are standardised with the checkpoint's statistics. Without a look
in the checkpoint both options are refused.

The UNet is given what a training step gives it (``timemachine.mvgen.probes.render_views``): camera
translations and depth scaled by ``--scale`` (``scene``: one factor for the whole scene, as in the
authors' training and the default for the released model; ``sample``: MVGenMaster's demo rule; or a
number), 3D priors computed in fp32 from depth whose far values are contracted once. The rule and
the factor of every view set are written to its record.

Runs in MVGenMaster's environment from its checkout, so give absolute paths:

    RUN=$PWD/data/experiments/mvgen/train/gate2_date_lora
    SCRIPT=$PWD/scripts/mvgen/render.py ARGS="--checkpoint $RUN/checkpoints/latest.pt --out $RUN/render_years \\
        --keys-file $PWD/configs/mvgen_gate2.json --dates true swapped 2015 2021" \\
        sbatch --time=00:30:00 --job-name=dtm_render slurm/mvgen_gpu.sbatch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from timemachine.mvgen.data import SCALE_RULES, MVGenScene, ViewSamplingConfig
from timemachine.mvgen.date import UNCOND_DATES
from timemachine.mvgen.evaluation import LookInputs, view_looks
from timemachine.mvgen.multiscene import scene_edges
from timemachine.mvgen.model import DEFAULT_MODEL_DIR, assert_exact_matmul, load_finetuned, load_mvgen
from timemachine.mvgen.probes import (
    RenderProbe, parse_render_date, read_keys_file, read_rows_file, representative_years,
)
from timemachine.mvgen.step import CROSS_DATE_PIXELS, PRECISIONS, LossConfig
from timemachine.mvgen.trainer import parse_scale
from timemachine.training.loss import DATE_VIEWS

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / "data/datasets/25512_model0_v2"
# Options that default to the checkpoint's training run; each sets the LossConfig field of the same name.
LOSS_OPTIONS = ("date_views", "class_label", "cross_date_pixels", "cross_date_gap", "precision")
LOOKS = ("references", "own", "neutral")            # --look words; "donor:<row>" besides
TONES = ("grey", "toned", "colour")                 # as timemachine.mvgen.look.TONES


def parse_look(text: str) -> tuple[str, int | None]:
    """A ``--look`` value: one of ``LOOKS``, as ``(word, None)``, or ``donor:<row>``, as ``("donor", row)``."""
    if text in LOOKS:
        return text, None
    kind, _, row = text.partition(":")
    if kind == "donor" and row.isdigit():
        return "donor", int(row)
    raise argparse.ArgumentTypeError(f"a look is one of {LOOKS} or donor:<row>, not {text!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="output folder")
    p.add_argument("--checkpoint", type=Path,
                   help="a checkpoint of scripts/mvgen/train.py (default: the released model)")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="the released model, relative to the checkout")
    p.add_argument("--dataset", type=Path, help=f"dataset folder (default: the checkpoint's, else {DEFAULT_DATASET})")
    sets = p.add_mutually_exclusive_group(required=True)
    sets.add_argument("--rows-file", type=Path, help="view sets as a JSON list of row lists, references first")
    sets.add_argument("--keys-file", type=Path,
                      help='view sets as a JSON list of {"refs": [file_key, ...], "targets": [file_key, ...]}')
    p.add_argument("--n-ref", type=int, help="references per view set (default: the checkpoint's, else 3)")
    p.add_argument("--dates", type=parse_render_date, nargs="+", default=["true"],
                   help="years, or: true, swapped, unknown")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, default=50, help="sampler steps")
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--uncond-dates", choices=UNCOND_DATES, default="same",
                   help="what the unconditional half of the guidance batch sees")
    p.add_argument("--resolution", type=int, help="default: the checkpoint's, else 512")
    p.add_argument("--scale", type=parse_scale,
                   help=f"scene-scale rule: one of {SCALE_RULES}, or a factor (default: the checkpoint's, else scene)")
    p.add_argument("--period-edges", type=float, nargs="*", help="default: the checkpoint's, else Notre-Dame's")
    p.add_argument("--swap-years", type=float, nargs="*",
                   help="one year per period for 'swapped' (default: the checkpoint's, else each period's median date)")
    p.add_argument("--date-views", choices=DATE_VIEWS)
    p.add_argument("--class-label", type=int, help="MVGenMaster's domain label: 0 regular datasets, 1 MegaScenes")
    p.add_argument("--cross-date-pixels", choices=CROSS_DATE_PIXELS)
    p.add_argument("--cross-date-gap", type=float)
    p.add_argument("--precision", choices=PRECISIONS)
    p.add_argument("--look", type=parse_look,
                   help="for a checkpoint with a look: the targets' look, one of references (default), own, neutral or "
                        "donor:<row>")
    p.add_argument("--tone", choices=TONES, help="for a checkpoint with a look: this tone in the targets' codes")
    p.add_argument("--thumb", type=int, default=256, help="side of the contact sheet's pictures, in pixels")
    return p.parse_args(argv)


def first_given(*values):
    """The first value that is not None."""
    return next(value for value in values if value is not None)


def main(argv: list[str] | None = None) -> None:
    assert_exact_matmul()
    args = parse_args(argv)
    if args.checkpoint is not None:
        model, ckpt = load_finetuned(args.checkpoint, args.model_dir)
    else:
        model, ckpt = load_mvgen(args.model_dir), {}
    model.unet.eval()
    trained = ckpt.get("train_config", {})
    trained_sampling = trained.get("sampling", {})
    given = {name: getattr(args, name) for name in LOSS_OPTIONS if getattr(args, name) is not None}
    loss_cfg = LossConfig(**{**trained.get("loss", {}), **given})
    n_ref = first_given(args.n_ref, trained_sampling.get("n_ref"), ViewSamplingConfig.n_ref)
    dataset = Path(first_given(args.dataset, trained.get("dataset"), DEFAULT_DATASET))
    # A multi-scene checkpoint (scripts/mvgen/train_multi.py) names no edges: the scene's own (scene.json).
    edges = tuple(first_given(args.period_edges, trained_sampling.get("period_edges"),
                              scene_edges(dataset) if trained.get("multi_scene") else None,
                              ViewSamplingConfig.period_edges))
    resolution = first_given(args.resolution, trained.get("resolution"), 512)
    scale = parse_scale(first_given(args.scale, trained.get("scale"), "scene"))

    scene = MVGenScene(dataset, ViewSamplingConfig(n_ref=n_ref, period_edges=edges), resolution=resolution,
                       camera_longest_side=model.config.camera_longest_side, require_depth=False, scale=scale)
    if args.rows_file is not None:
        view_sets = read_rows_file(args.rows_file)
    else:
        table_keys = pd.read_parquet(dataset / "views.parquet", columns=["file_key"])["file_key"].tolist()
        view_sets = read_keys_file(args.keys_file, table_keys, n_ref)
    swap_years = first_given(args.swap_years or None, ckpt.get("swap_years"),
                             representative_years(scene.dates[scene.usable], edges))
    factor = "a factor per sample" if scene.scale_factor is None else f"factor {scene.scale_factor:.6g}"
    print(f"{len(view_sets)} view sets of {dataset}; model: {args.checkpoint or 'released (no date input)'}, "
          f"step {ckpt.get('step', 0)}; dates {args.dates}, seeds {args.seeds}, guidance {args.guidance:g}, "
          f"{loss_cfg.precision}; scene scale {scale!r} ({factor}); swapped dates use the years {list(swap_years)}",
          flush=True)
    per_scene = ckpt.get("scale") == "scene" and ckpt.get("scale_factor") is None   # multi-scene: each scene's own
    if ckpt and not (per_scene and scale == "scene") and (scale, scene.scale_factor) != (ckpt["scale"], ckpt["scale_factor"]):
        print(f"note: the checkpoint was trained with scene scale {ckpt['scale']!r} (factor {ckpt['scale_factor']}); "
              "these renders use another, so the cameras are not at the size the adapters were trained on",
              flush=True)

    has_look = bool(ckpt.get("finetune_config", {}).get("look", False))
    if (args.look is not None or args.tone is not None) and not has_look:
        raise SystemExit("--look and --tone need a checkpoint of a run with a look code")
    inputs, (kind, donor) = None, args.look or ("references", None)
    if has_look:
        from timemachine.mvgen.look import read_look

        look = read_look(dataset, scene.views["file_key"].tolist())
        if look is None:
            raise SystemExit(f"{dataset} has no look codes for this view table: run scripts/mvgen/prepare_look.py")
        inputs = LookInputs(look=look, stats=model.look_stats, tone=args.tone)
        print(f"targets' look: {kind}{'' if donor is None else f' of row {donor}'}"
              f"{'' if args.tone is None else f', tone {args.tone}'}", flush=True)

    results = []
    for i, rows in enumerate(view_sets):
        # The targets' codes come from the look file alone (no look of render.py needs the pictures: sample None).
        donors = None if donor is None else {"donor": [donor] * (len(rows) - n_ref)}
        _, look, known = view_looks(kind if has_look else "none", rows, n_ref, None, inputs, donors)
        probe = RenderProbe(scene, rows, n_ref, f"set{i}", args.dates, args.seeds, loss_cfg, swap_years, args.steps,
                            args.guidance, args.uncond_dates, look=look, look_known=known)
        results.append(probe(model, args.out, thumb=args.thumb))
        if has_look:        # which look was asked for; the standardised codes per view stay in set<i>.json
            results[-1]["look"] = {"targets": kind, "donor": donor, "tone": args.tone}
        scores = ", ".join(f"{date} {value:.2f} dB" for date, value in results[-1]["psnr_by_date"].items())
        print(f"   set{i} (targets dated {results[-1]['years'][n_ref:]}): PSNR at date {scores}", flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {len(results)} contact sheets and results.json to {args.out}")


if __name__ == "__main__":
    main()
