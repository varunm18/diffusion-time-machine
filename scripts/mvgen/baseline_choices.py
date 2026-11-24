#!/usr/bin/env python
"""Which scene scale and domain label does the RELEASED MVGenMaster prefer on our views? (single GPU)

Two choices change every baseline of the fine-tuning runs and cannot be read off the authors' code:

* the scene scale. MVGenMaster's training scaled a scene once, by the extent of all its cameras; for
  Notre-Dame that is a factor of about 0.087, far smaller than for most of its training scenes, and
  the factor only reaches the model through the camera-ray channels. ``--scales`` lists the settings
  to compare: ``scene`` (that rule, from the dataset's ``scene.json``), ``sample`` (the demo's
  per-sample rule) or constant factors;
* the domain label: 0 (regular datasets) or 1 (MegaScenes, the model's own name for photo
  collections like ours).

A third option measures what a run's rule for cross-date targets costs the released model:

* ``--cross-date-pixels`` lists what is taken from the targets that no reference shows as they
  looked on their date: ``keep`` (nothing: the default, and the only value measured unless others
  are given), ``drop`` (their warped pixels) or ``drop_all`` (both their priors), which are the
  settings of a training run (``timemachine.mvgen.step.CROSS_DATE_PIXELS``); or ``no_priors``, where
  every prior of every view is zeros, the references' too, as under the authors' whole-sample prior
  dropout. Which targets are cross-date is decided by the function a training step uses
  (``step.cross_date_drops``): those without a reference within ``--cross-date-gap`` years in their
  own period.

For every view set, scale, label and cross-date value this measures what a training step measures
before any training: the released model's v-prediction loss on the targets, in fp32, with no dates,
no dropout, far depth contracted once and the mode of the VAE's posterior, at fixed timesteps and
with fixed noise (``timemachine.mvgen.step``; the same noisy latents for every scale, label and
cross-date value of a view set). A lower loss means the released model finds these inputs more
familiar. The PSNR of a sampled render is a much noisier measure (one render of such a set scores 10
to 14 dB), but ``--render`` adds it: every set is rendered once per scale and label with noise seed
0, through the code path of ``scripts/mvgen/render.py``, and with the FIRST cross-date value only
(the renderer knows the three settings of a training run, not ``no_priors``).

View sets are those of the key files (``--keys-file``, repeatable; default: the three gate files and
``configs/mvgen_nvs_test_sets.json``) plus ``--sampled`` sets drawn by the default sampler with a
fixed seed. A view set that several files name is measured once. Each is classed by how its
references relate to its targets in time: ``same`` (all views in one period), ``cross`` (no reference
shares a period with a target: they lie on different sides of a period edge) or ``mixed``. The class
is not the cross-date rule: a target of a ``same`` set is cross-date when every reference is dated
more than the gap away, so the number of cross-date targets is recorded per view set.

A hand-picked view set can break rules that the sampler keeps. Two cases are reported per view set,
in the printed list and in ``sets.json``: views that the dataset excludes (``usable`` is false, e.g.
a sideways photo), and targets that no reference sees (fewer than the sampler's ``min_shared``
COLMAP points in common with every reference: by the sampler's measure the references do not show
them). The key files are not changed for this: earlier baselines were measured on these sets.

Depth maps of the references come from the dataset folder. Where it has none yet, they are computed
here with Depth-Pro exactly as ``scripts/mvgen/prepare_depth.py`` computes them (its ``view_depth``)
and kept under ``<out>/dataset``, a folder of links to the dataset with its own ``depth``; the
dataset folder is never written to.

Output under ``--out``:

* ``sets.json``: the view sets (rows, file keys, dates, periods, class, ``cross_date_targets``, and
  what to be warned of: ``not_usable``, ``references_without_reliable_depth`` and
  ``targets_no_reference_sees``);
* ``losses.jsonl``: one line per (view set, scale, label, cross-date value, timestep, noise seed),
  with the value as ``cross_date_pixels`` and the view set's ``cross_date_targets``;
* ``summary.json`` and a printed table: the mean loss per scale, label and cross-date value over the
  ``same`` sets, the ``cross`` sets (and the ``mixed`` ones, if any) and all sets, with the standard
  error over view sets, and the mean relative difference to the first scale, label and cross-date
  value on the same view sets (paired, so the differences between view sets do not blur it);
  ``summary.json`` also holds the ``cross_date_gap``;
* with ``--render``: ``renders/scale_<s>_label_<l>/<set>.jpg`` and ``.json``, ``renders.jsonl`` and
  the same table for the PSNR.

Every run writes these files anew and appends to none of them, so a folder never mixes the
measurements of two invocations: a run restarted with other settings replaces what was there. Only
contact sheets of scales or labels that the new run does not render stay behind in ``renders/``;
``renders.jsonl`` lists the ones that belong to the run.

``--list-sets`` prints the view sets and stops; it needs no GPU and writes nothing.

Runs in MVGenMaster's environment from its checkout, so give absolute paths:

    SCRIPT=$PWD/scripts/mvgen/baseline_choices.py ARGS="--out $PWD/data/experiments/mvgen/baseline_choices" \\
        sbatch --time=01:00:00 --job-name=dtm_choices slurm/mvgen_gpu.sbatch
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from timemachine.mvgen.data import MVGenScene, ViewSamplingConfig, depth_path, period_names
from timemachine.mvgen.model import DEFAULT_MODEL_DIR, MVGen, assert_exact_matmul, load_mvgen
from timemachine.mvgen.probes import (
    DEFAULT_TIMESTEPS, RenderProbe, check_view_sets, describe_views, read_keys_file, scene_view_sets,
)
from timemachine.mvgen.step import (
    CROSS_DATE_PIXELS, LossConfig, build_inputs, cross_date_drops, cross_date_views, encode_images, v_prediction_loss,
)
from timemachine.mvgen.trainer import parse_scale

REPO = Path(__file__).resolve().parents[2]
DEFAULT_KEYS_FILES = tuple(REPO / "configs" / name for name in (
    "mvgen_gate1a_post.json", "mvgen_gate1b_pre.json", "mvgen_gate2.json", "mvgen_nvs_test_sets.json"))
DEFAULT_SCALES = ("scene", 1.0, 0.5, 0.2)
KINDS = ("same", "cross", "mixed")                  # how a view set's references relate to its targets in time
NOT_LINKED = ("depth", "depth_stats.parquet")       # what the folder of links keeps for itself
NO_PRIORS = "no_priors"                             # every prior of every view is zeros, the references' too
PRIOR_RULES = (*CROSS_DATE_PIXELS, NO_PRIORS)       # the values of --cross-date-pixels
SETTINGS = ("scale", "class_label", "cross_date_pixels")    # what is compared on a view set: a row of the table


@dataclass(frozen=True)
class ViewSet:
    """A view set to measure: its name, where it comes from, its rows (references first) and its class; how
    many of its targets the cross-date rule marks; and what breaks the sampler's rules (:meth:`warnings`)."""

    name: str
    source: str
    rows: tuple[int, ...]
    kind: str
    cross_date_targets: int = 0                                 # targets that step.cross_date_views marks
    not_usable: dict[str, str] = field(default_factory=dict)    # views the dataset excludes: file key -> reason
    targets_no_reference_sees: tuple[str, ...] = ()             # file keys (unseen_targets)

    def warnings(self) -> list[str]:
        """What a reader of this set's numbers should be warned of, as text; nothing for most sets."""
        found = []
        if self.not_usable:
            found.append(f"excluded by the dataset (not usable): {self.not_usable}")
        if self.targets_no_reference_sees:
            found.append(f"no reference sees target(s) {list(self.targets_no_reference_sees)}: each shares fewer "
                         "COLMAP points with every reference than the sampler asks for (min_shared)")
        return found


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="output folder")
    p.add_argument("--dataset", type=Path, default=REPO / "data/datasets/25512_model0_v2",
                   help="dataset folder (views.parquet, covisibility.npy, scene.json, sparse_points.npz, depth/)")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="the released model, relative to the checkout")
    p.add_argument("--keys-file", type=Path, action="append",
                   help='view sets as a JSON list of {"refs": [file_key, ...], "targets": [file_key, ...]}; '
                        "repeatable (default: the three gate files and mvgen_nvs_test_sets.json of configs/)")
    p.add_argument("--sampled", type=int, default=6, help="view sets drawn by the default sampler, with a fixed seed")
    p.add_argument("--scales", type=parse_scale, nargs="+", default=list(DEFAULT_SCALES),
                   help="scene-scale settings to compare: scene, sample, or constant factors")
    p.add_argument("--class-labels", type=int, nargs="+", default=[0, 1],
                   help="domain labels to compare: 0 regular datasets, 1 MegaScenes")
    p.add_argument("--cross-date-pixels", nargs="+", choices=PRIOR_RULES, default=["keep"],
                   help="what to take from the targets that no reference shows as they looked, to compare: keep "
                        "(nothing), drop (their warped pixels), drop_all (both their priors), as a training run's "
                        "option of this name; or no_priors: every prior of every view, the references' too")
    p.add_argument("--cross-date-gap", type=float, default=LossConfig.cross_date_gap,
                   help="years between date midpoints beyond which a reference does not show a target as it looked")
    p.add_argument("--timesteps", type=int, nargs="+", default=list(DEFAULT_TIMESTEPS))
    p.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--n-ref", type=int, default=ViewSamplingConfig.n_ref, help="references per view set")
    p.add_argument("--resolution", type=int, default=512, help="side of the crops; the depth maps' size")
    p.add_argument("--period-edges", type=float, nargs="*", default=list(ViewSamplingConfig.period_edges),
                   help="years that cut the scene's history into periods")
    p.add_argument("--render", action="store_true",
                   help="also render every set once per scale and label (noise seed 0; the first value of "
                        "--cross-date-pixels) and report the PSNR")
    p.add_argument("--render-steps", type=int, default=50, help="sampler steps of --render")
    p.add_argument("--list-sets", action="store_true", help="print the view sets and stop (no GPU, nothing written)")
    args = p.parse_args(argv)
    args.cross_date_pixels = list(dict.fromkeys(args.cross_date_pixels))        # a value given twice is measured once
    if args.render and args.cross_date_pixels[0] == NO_PRIORS:
        p.error(f"--render uses the first value of --cross-date-pixels, and the renderer cannot leave out every "
                f"prior ({NO_PRIORS}): put another value first")
    return args


# ----------------------------------------------------------------------------
# View sets
# ----------------------------------------------------------------------------
def set_kind(periods: Sequence[int], n_ref: int) -> str:
    """One of ``KINDS`` for a view set with these period indices (references first).

    ``"same"``: every view lies in one period. ``"cross"``: no reference shares a period with a
    target, so references and targets lie on different sides of a period edge. ``"mixed"``: some
    references share a target's period and some do not.
    """
    references, targets = set(periods[:n_ref]), set(periods[n_ref:])
    if len(references | targets) == 1:
        return "same"
    return "mixed" if references & targets else "cross"


def unseen_targets(covis: np.ndarray, rows: Sequence[int], n_ref: int, min_shared: int) -> list[int]:
    """The targets of a view set (rows of the view table) that share fewer than ``min_shared`` COLMAP points with
    every one of its references.

    ``covis`` is the dataset's co-visibility matrix. The sampler never draws such a set
    (``timemachine.mvgen.data.draw_references``): by its measure no reference shows that target, so
    the model has to invent it instead of carrying it over.
    """
    rows = np.asarray(rows)
    seen = (covis[np.ix_(rows[:n_ref], rows[n_ref:])] >= min_shared).any(axis=0)
    return rows[n_ref:][~seen].tolist()


def keyed_view_sets(paths: Sequence[Path], table_keys: Sequence[str], n_ref: int) -> list[tuple[str, str, tuple]]:
    """``(name, source, rows)`` of every sample of the key files, in order.

    A sample is named by its ``"name"`` entry, or ``<file stem>/<position>``. A file key that is not
    in the view table raises ``KeyError`` (``timemachine.mvgen.probes.read_keys_file``).
    """
    found = []
    for path in paths:
        rows = read_keys_file(path, table_keys, n_ref)
        names = [sample.get("name", f"{path.stem}/{i}") for i, sample in enumerate(json.loads(path.read_text()))]
        found += [(name, path.name, tuple(view_rows)) for name, view_rows in zip(names, rows)]
    return found


def collect_view_sets(dataset: Path, keys_files: Sequence[Path], n_sampled: int, sampling: ViewSamplingConfig,
                      resolution: int, cross_date_gap: float = LossConfig.cross_date_gap) -> list[ViewSet]:
    """The view sets to measure: those of the key files, then ``n_sampled`` drawn by the sampler.

    The sampled sets are the ones the trainer's date-loss probe would use
    (``timemachine.mvgen.probes.scene_view_sets``: ``sampling``'s rules, a fixed seed); their
    references have reliable depth if the folder's depth maps have been prepared. A view set that
    was already collected is left out, with a note. Needs no depth maps and no GPU.

    Each set also gets the number of its targets that the cross-date rule marks
    (``step.cross_date_views`` with ``cross_date_gap`` and ``sampling.period_edges``, on the dates as
    a sample carries them), the views the dataset excludes, and the targets that none of its
    references sees (:func:`unseen_targets` with ``sampling.min_shared``).
    """
    scene = MVGenScene(dataset, sampling, resolution=resolution, scale="sample",
                       require_depth=(dataset / "depth_stats.parquet").exists())
    n_ref = sampling.n_ref
    candidates = keyed_view_sets(keys_files, scene.views["file_key"].tolist(), n_ref)
    candidates += [(f"sampled/{i}", "sampler", tuple(int(row) for row in rows))
                   for i, rows in enumerate(scene_view_sets(scene, n_sampled))]
    check_view_sets([rows for _, _, rows in candidates], n_ref, len(scene.views))
    view_sets: dict[tuple, ViewSet] = {}
    for name, source, rows in candidates:
        if rows in view_sets:
            print(f"   {name} ({source}) is the view set {view_sets[rows].name}: measured once", flush=True)
            continue
        date_mid = torch.from_numpy(scene.dates[list(rows)]).float()            # float32, as in a sample
        cross_date = cross_date_views(date_mid, n_ref, cross_date_gap, sampling.period_edges)
        unseen = unseen_targets(scene.covis, rows, n_ref, sampling.min_shared)
        view_sets[rows] = ViewSet(
            name, source, rows, set_kind(scene.period[list(rows)].tolist(), n_ref),
            cross_date_targets=int(cross_date.sum()), not_usable=describe_views(scene, rows, n_ref)["not_usable"],
            targets_no_reference_sees=tuple(scene.views["file_key"][row] for row in unseen))
    return list(view_sets.values())


def print_view_sets(view_sets: Sequence[ViewSet], dates: np.ndarray, n_ref: int, edges: Sequence[float]) -> None:
    """One line per view set: its class, the dates of its references and of its targets, and how many of the
    targets are cross-date; then a line for each thing to be warned of (``ViewSet.warnings``)."""
    counts = ", ".join(f"{sum(view_set.kind == kind for view_set in view_sets)} {kind}" for kind in KINDS)
    print(f"{len(view_sets)} view sets ({counts}); periods: {period_names(edges)}", flush=True)
    for view_set in view_sets:
        years = [round(float(year), 2) for year in dates[list(view_set.rows)]]
        print(f"   {view_set.name:28s} {view_set.kind:5s} references {years[:n_ref]} targets {years[n_ref:]}; "
              f"{view_set.cross_date_targets} of {len(years) - n_ref} targets cross-date", flush=True)
        for warning in view_set.warnings():
            print(f"      warning: {warning}", flush=True)


# ----------------------------------------------------------------------------
# Depth
# ----------------------------------------------------------------------------
def link(target: Path, link_path: Path) -> None:
    """Make ``link_path`` a symbolic link to ``target``, replacing what is there."""
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target.resolve())


def folder_with_depth(dataset: Path, work: Path, rows: Sequence[int], size: int, device: str = "cuda") -> Path:
    """A dataset folder in which every view of ``rows`` has a depth map: ``dataset`` itself, if it has them all.

    Otherwise ``work``: links to everything in ``dataset`` except its depth, the depth maps that
    ``dataset`` has for ``rows`` (linked), and for the other rows maps computed now with Depth-Pro
    by ``view_depth`` of ``scripts/mvgen/prepare_depth.py``, stored as that script stores them, with
    a ``depth_stats.parquet`` for ``rows``. ``dataset`` is not written to. A view without any
    COLMAP point in its crop has no depth and cannot be a reference (``ValueError``).
    """
    rows = sorted({int(row) for row in rows})
    missing = [row for row in rows if not depth_path(dataset, row).exists()]
    if not missing:
        return dataset
    if str(Path(__file__).parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent))
    import prepare_depth                # here, not at the top: it loads Depth-Pro, which only this branch needs

    (work / "depth").mkdir(parents=True, exist_ok=True)
    for item in dataset.iterdir():
        if item.name not in NOT_LINKED:
            link(item, work / item.name)
    views = pd.read_parquet(dataset / "views.parquet")
    prepared = prepare_depth.finished_views(dataset, views)
    records = {row: prepared[row] for row in rows if row not in missing and row in prepared}
    for row in rows:
        if row not in missing:
            link(depth_path(dataset, row), depth_path(work, row))
    print(f"{len(missing)} of the {len(rows)} references have no depth map in {dataset}: computing them with "
          f"Depth-Pro into {work}", flush=True)
    depth_model, transform = prepare_depth.create_model_and_transforms(device=torch.device(device))
    depth_model.eval()
    sparse = prepare_depth.load_sparse(dataset)
    for row in missing:
        depth, records[row] = prepare_depth.view_depth(views, row, sparse, depth_model, transform, size)
        if depth is None:
            raise ValueError(f"view {row} ({views.file_key[row]}) has no COLMAP point in its crop, so its depth "
                             "cannot be scaled: it cannot be a reference")
        link_path = depth_path(work, row)
        if link_path.is_symlink():      # a link of an earlier run, when the dataset still had the map
            link_path.unlink()
        prepare_depth.save_depth(link_path, depth)
        print(f"   view {row}: scaled to {records[row]['n_points']} COLMAP points, median disagreement "
              f"{records[row]['median_rel_error']:.1%}{'' if records[row]['ok'] else ' (NOT reliable)'}", flush=True)
    prepare_depth.write_stats(work, records)
    del depth_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return work


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------
def scale_name(scale: str | float) -> str:
    """A scale setting as text, for tables and folder names."""
    return scale if isinstance(scale, str) else f"{scale:g}"


def dropped_priors(rule: str, date_mid: torch.Tensor, n_ref: int, gap: float, period_edges: Sequence[float]) -> dict:
    """What ``step.build_inputs`` is told to zero under one value of ``--cross-date-pixels``, as its keyword arguments.

    ``"no_priors"`` zeroes every prior of every view (``drop_priors``, the authors' whole-sample prior
    dropout). The other values are a training run's ``LossConfig.cross_date_pixels``, and the views
    they mark come from the function the training step calls, ``step.cross_date_drops``, with the
    view set's date midpoints, ``gap`` and the scene's ``period_edges``: nothing for ``"keep"``, the
    cross-date targets' warped pixels for ``"drop"``, both their priors for ``"drop_all"``.
    """
    if rule == NO_PRIORS:
        return {"drop_priors": True}
    cfg = LossConfig(cross_date_pixels=rule, cross_date_gap=gap)
    prior_views, pixel_views = cross_date_drops(date_mid, n_ref, cfg, period_edges)
    return {"dropped_prior_views": prior_views, "dropped_pixel_views": pixel_views}


@torch.no_grad()
def measure_losses(model: MVGen, scenes: dict[str, MVGenScene], view_sets: Sequence[ViewSet], n_ref: int,
                   class_labels: Sequence[int], timesteps: Sequence[int], noise_seeds: Sequence[int],
                   out_file: Path, cross_date_pixels: Sequence[str] = ("keep",),
                   cross_date_gap: float = LossConfig.cross_date_gap) -> list[dict]:
    """The model's target loss for every (view set, scale, domain label, cross-date value, timestep, noise seed).

    Args:
        scenes: the dataset opened once per scale setting, by :func:`scale_name`.
        out_file: gets one JSON line per measurement, as it is made; what it held is replaced.
        cross_date_pixels: values of ``PRIOR_RULES``, each measured (:func:`dropped_priors`).
        cross_date_gap: the gap (years) of the cross-date rule.

    The conditions are those of a training step at step 0 without dropout: fp32, no dates, the mode
    of the VAE's posterior (``step.build_inputs`` and ``step.v_prediction_loss``). The noise depends
    on the noise seed and the timestep alone, and the latents on the view set alone, so the scales,
    labels and cross-date values of a view set are compared on the very same noisy latents.
    """
    records = []
    with open(out_file, "w") as f:
        for number, view_set in enumerate(view_sets, start=1):
            latents = None
            for name, scene in scenes.items():
                sample = scene.make_sample(np.asarray(view_set.rows), n_ref)
                if latents is None:             # the images are the same at every scale
                    latents = encode_images(model, sample["images"], sample_posterior=False)
                inputs = {rule: build_inputs(model, sample, latents, **dropped_priors(
                    rule, sample["date_mid"], n_ref, cross_date_gap, scene.config.period_edges))
                    for rule in cross_date_pixels}
                for label, rule, timestep, seed in product(class_labels, cross_date_pixels, timesteps, noise_seeds):
                    cfg = LossConfig(precision="fp32", class_label=label)
                    generator = torch.Generator().manual_seed(1000 * seed + timestep)
                    noise = torch.randn(latents.shape, generator=generator)
                    loss, per_view = v_prediction_loss(model, latents, inputs[rule], timestep, noise, None, None, cfg,
                                                       n_ref)
                    records.append({"set": view_set.name, "kind": view_set.kind, "scale": name,
                                    "scale_factor": sample["scale"], "class_label": label, "cross_date_pixels": rule,
                                    "cross_date_targets": view_set.cross_date_targets, "timestep": timestep,
                                    "noise_seed": seed, "loss": float(loss), "loss_per_target": per_view.tolist()})
                    f.write(json.dumps(records[-1]) + "\n")
                f.flush()
            print(f"   {number} / {len(view_sets)} {view_set.name} ({view_set.kind}) measured", flush=True)
    return records


def render_sets(model: MVGen, scenes: dict[str, MVGenScene], view_sets: Sequence[ViewSet], n_ref: int,
                class_labels: Sequence[int], steps: int, out_dir: Path, cross_date_pixels: str = "keep",
                cross_date_gap: float = LossConfig.cross_date_gap) -> list[dict]:
    """Render every view set once per scale and domain label, and return the PSNR against the real targets.

    Uses ``timemachine.mvgen.probes.RenderProbe``, as ``scripts/mvgen/render.py`` does: noise seed 0,
    each target at its own date (which the released model cannot read), fp32, guidance 2, and one
    cross-date setting of a training run (``cross_date_pixels``, one of ``step.CROSS_DATE_PIXELS``,
    with ``cross_date_gap``). Writes a contact sheet and a record per set into
    ``out_dir/scale_<s>_label_<l>``, and one JSON line per render into ``out_dir.jsonl``, replacing
    what it held.
    """
    records = []
    with open(out_dir.with_suffix(".jsonl"), "w") as f:
        for name, scene in scenes.items():
            for label in class_labels:
                cfg = LossConfig(precision="fp32", class_label=label, cross_date_pixels=cross_date_pixels,
                                 cross_date_gap=cross_date_gap)
                for view_set in view_sets:
                    probe = RenderProbe(scene, view_set.rows, n_ref, re.sub(r"[^\w.-]+", "_", view_set.name),
                                        dates=("true",), seeds=(0,), loss_cfg=cfg, steps=steps)
                    result = probe(model, out_dir / f"scale_{name}_label_{label}")
                    records.append({"set": view_set.name, "kind": view_set.kind, "scale": name,
                                    "scale_factor": result["scale_factor"], "class_label": label,
                                    "cross_date_pixels": cross_date_pixels,
                                    "cross_date_targets": view_set.cross_date_targets,
                                    "psnr": result["psnr_by_date"]["true"]})
                    f.write(json.dumps(records[-1]) + "\n")
                    f.flush()
                    print(f"   scale {name} label {label} {view_set.name}: {records[-1]['psnr']:.2f} dB", flush=True)
    return records


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def mean_and_error(values: np.ndarray) -> tuple[float, float | None]:
    """The mean of the values and its standard error (None for fewer than two values)."""
    error = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return float(np.mean(values)), error


def setting_of(record: dict) -> tuple:
    """The setting a measurement or a table row belongs to: its scale, domain label and cross-date value."""
    return tuple(record[name] for name in SETTINGS)


def summarise(records: Sequence[dict], value: str, relative: bool) -> list[dict]:
    """Per setting (scale, domain label and cross-date value) and per class of view sets, the mean of
    ``records[...][value]``.

    A view set counts once, with the mean of its measurements, so the standard error is that over
    view sets. ``vs_first`` compares with the first setting of ``records`` (the first scale, label
    and cross-date value) on the same view sets: the mean and standard error of each set's relative
    difference (``relative``: ratio - 1, for a loss) or difference (for a PSNR in dB). Returns one
    row per (setting, class) in the order of ``records``, with the class ``"all"`` last; classes
    without a view set are left out.
    """
    settings = list(dict.fromkeys(setting_of(record) for record in records))
    kind_of = {record["set"]: record["kind"] for record in records}
    measured = {setting: {name: [] for name in kind_of} for setting in settings}
    for record in records:
        measured[setting_of(record)][record["set"]].append(record[value])
    per_set = {setting: np.array([np.mean(values) for values in measured[setting].values()]) for setting in settings}
    first = per_set[settings[0]]
    rows = []
    for setting, values in per_set.items():
        change = values / first - 1 if relative else values - first
        for kind in (*KINDS, "all"):
            chosen = np.array([kind in ("all", kind_of_set) for kind_of_set in kind_of.values()])
            if not chosen.any():
                continue
            mean, error = mean_and_error(values[chosen])
            mean_change, change_error = mean_and_error(change[chosen])
            rows.append({**dict(zip(SETTINGS, setting)), "kind": kind, "sets": int(chosen.sum()), "mean": mean,
                         "standard_error": error, "vs_first": mean_change, "vs_first_standard_error": change_error})
    return rows


def format_table(rows: Sequence[dict], title: str, digits: int, relative: bool) -> str:
    """The rows of :func:`summarise` as text: one line per setting (scale, label, and ``priors``: the cross-date
    value), one column per class of view sets."""
    def with_error(mean: float, error: float | None, form: str) -> str:
        return f"{mean:{form}} +- " + ("n/a" if error is None else f"{error:{form.lstrip('+')}}")

    def cell(row: dict) -> str:
        change = with_error(row["vs_first"], row["vs_first_standard_error"], "+.1%" if relative else "+.2f")
        return f"{with_error(row['mean'], row['standard_error'], f'.{digits}f')} [{change}]"

    kinds = {row["kind"]: row["sets"] for row in rows}
    first = rows[0]
    lines = [title, f"mean +- standard error over view sets [change from scale {first['scale']}, label "
                    f"{first['class_label']}, priors {first['cross_date_pixels']} on the same view sets]",
             f"{'scale':>8} {'label':>5} {'priors':>9} | "
             + " | ".join(f"{f'{kind} ({n} sets)':<36}" for kind, n in kinds.items())]
    for setting in dict.fromkeys(setting_of(row) for row in rows):
        scale, label, rule = setting
        cells = [cell(row) for row in rows if setting_of(row) == setting]
        lines.append(f"{scale:>8} {label:>5} {rule:>9} | " + " | ".join(f"{text:<36}" for text in cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    assert_exact_matmul()
    args = parse_args(argv)
    edges, n_ref, rules = tuple(args.period_edges), args.n_ref, args.cross_date_pixels
    sampling = ViewSamplingConfig(n_ref=n_ref, period_edges=edges)
    view_sets = collect_view_sets(args.dataset, args.keys_file or DEFAULT_KEYS_FILES, args.sampled, sampling,
                                  args.resolution, args.cross_date_gap)
    dates = pd.read_parquet(args.dataset / "views.parquet", columns=["date_mid"])["date_mid"].to_numpy()
    print_view_sets(view_sets, dates, n_ref, edges)
    if args.list_sets:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    references = [row for view_set in view_sets for row in view_set.rows[:n_ref]]
    folder = folder_with_depth(args.dataset, args.out / "dataset", references, args.resolution)
    model = load_mvgen(args.model_dir)
    if not 0 <= min(args.timesteps) <= max(args.timesteps) < model.scheduler.config.num_train_timesteps:
        raise ValueError(f"timesteps {args.timesteps} are outside the scheduler's training steps")
    scenes = {scale_name(scale): MVGenScene(folder, sampling, resolution=args.resolution, require_depth=False,
                                            camera_longest_side=model.config.camera_longest_side, scale=scale)
              for scale in args.scales}
    described = next(iter(scenes.values()))
    (args.out / "sets.json").write_text(json.dumps(
        [{**asdict(view_set), **describe_views(described, view_set.rows, n_ref)} for view_set in view_sets], indent=1))

    losses = measure_losses(model, scenes, view_sets, n_ref, args.class_labels, args.timesteps, args.noise_seeds,
                            args.out / "losses.jsonl", rules, args.cross_date_gap)
    summary = {"cross_date_gap": args.cross_date_gap, "loss": summarise(losses, "loss", relative=True)}
    print("\n" + format_table(summary["loss"], "target loss of the released model (fp32, no dates; lower: more "
                              "familiar inputs)", digits=4, relative=True), flush=True)
    if args.render:
        if len(rules) > 1:
            print(f"\nrendering with --cross-date-pixels {rules[0]} only (the first value)", flush=True)
        renders = render_sets(model, scenes, view_sets, n_ref, args.class_labels, args.render_steps,
                              args.out / "renders", rules[0], args.cross_date_gap)
        summary["psnr"] = summarise(renders, "psnr", relative=False)
        print("\n" + format_table(summary["psnr"], "PSNR of one render against the real targets, in dB (noise seed "
                                  "0; a noisy measure)", digits=2, relative=False), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote sets.json, losses.jsonl and summary.json to {args.out}")


if __name__ == "__main__":
    main()
