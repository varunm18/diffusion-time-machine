"""How much the date matters inside and outside the region that changed at a period edge.

The probes of :mod:`timemachine.mvgen.probes` score whole frames, and a whole frame barely registers
Notre-Dame's fire: the spire and the roofs are a few percent of a picture. A model that draws the
right building for a date and one that ignores the date then differ by 0.1 dB of PSNR and 0.15% of
loss. This module makes the probes' two measurements again, split into three *areas* of every target
view with the masks of :mod:`timemachine.mvgen.regions`:

* ``inside``: where the view shows (or showed, or would show) what changed at the edge;
* ``outside``: the rest of the frame;
* ``whole``: the frame, which is what the probes report.

**Loss with fixed noise** (:func:`measure_losses`), as ``probes.DateLossProbe`` measures it: the
conditioning of a training step without dropout, the mode of the VAE's posterior, the run's
:class:`~timemachine.mvgen.step.LossConfig` (precision, domain label, which views show their date,
the cross-date rule by the TRUE dates), and the same noisy latents for the three date variants:
``true``, ``swapped`` (each target at the representative year of the period across the nearest
edge, ``probes.swapped_years``) and ``unknown``. The squared error of every latent cell
(``step.v_prediction_error_map``) is averaged with the mask reduced to the latents' resolution as
weights (``regions.latent_weights``: a cell half inside the region counts half for each area), over
the cells of all targets of a view set together. A set's *gap* is ``loss_variant / loss_true - 1``,
taken per timestep and noise seed and then averaged: above zero, the true date gives the lower loss.

**Renders** (:func:`render_set`), as ``probes.RenderProbe`` makes them (:func:`render_at`): a view set
at its true dates and at its swapped dates, with the same noise seed. Per target and area: the PSNR
against the real photo, from the mean squared error over the pixels of the area; and the mean
absolute difference between the two renders (0-1), which needs no real photo. If the date moves
the building and nothing else, that difference is large inside the region and small outside it.
On a GPU it is not exactly zero for a model without a date input: the prior warp scatters with
duplicate indices and is not reproducible to the bit.

A run whose ``cross_date_pixels`` rule is not ``"keep"`` takes priors from cross-date targets, and
``RenderProbe`` applies the rule by the date it renders at. A target from after the fire, rendered
from references from before it, has no priors at its true date and can have all of them at its
swapped date: the two renders then differ in their priors as well as in the date, also for the
released model (:func:`priors_differ` says for which view sets; where the rule looks at the period
alone, as with a ``cross_date_gap`` of 1000 years, for every set whose references all lie on one
side of the edge that its targets are swapped across). ``priors_by="true"`` (:func:`render_at`)
applies the rule by the targets' own dates in both renders, as the loss measurements do, so that
only the date differs; the true-date render is the same either way.

**Small regions.** A target whose region has fewer than ``min_pixels`` pixels has too little of it to
score (frontal views of the west facade show a sliver of roof). Its region is treated as empty
(:func:`target_masks`): the target has no ``inside`` numbers, and its whole frame is ``outside``.
A view set without any scored region has no ``inside`` numbers and is left out of that row of the
tables, whose ``sets`` column counts the sets that are in it. The rule only removes what is too
small to average. It does not tell a region from the specks that the masks leave in frontal views
(see :mod:`~timemachine.mvgen.regions`): over half of those reach 256 pixels, a tenth 2% of the
frame, which nearly all views from the south, south-east and apse exceed. For view sets with frontal
targets, set ``min_pixels`` to 2% of the frame (5,243 pixels at 512) or read ``inside`` with care.

**View sets** (:func:`collect_view_sets`) come from keys files and from the held-out probe of a run:
``probes.heldout_view_sets`` on the run's scene, whose first sets are the ones the run's own
held-out date-loss probe measures (:func:`heldout_sets`). Held-out sets can be limited to those
that *cross* the edge: every target dated from the edge on, every reference before it. Only there
does the date decide what the building looks like, because the references show its other state.
Every set is marked ``cross`` or ``other`` and the tables give the two groups apart
(:func:`groups_of`). The standard errors are over view sets; held-out sets share targets (a run
holds out some 40 views from after the fire), so they are not independent and the errors are
somewhat too small.

**A run's settings** (:func:`run_config`, :func:`open_scene`, :func:`run_swap_years`,
:func:`checkpoint_path`). A run folder's ``config.json`` gives back its ``TrainConfig``; the scene is
opened and its views held out as the run's ``Trainer`` does, and :func:`check_run_records` refuses a
dataset folder that no longer gives the run's hold-out, scale factor or fingerprint: the "held-out"
views would then be views the model trained on. A checkpoint's weights are the ones the trainer's
probes measure: its ``trainable`` entry, which is the EMA when the run keeps one (the checkpoint
says which under ``weights``).

The summaries (:func:`summarise_set_losses`, :func:`summarise_set_renders`,
:func:`summarise_groups`) and :func:`format_table` turn the raw records into what
``scripts/mvgen/eval_dates.py`` writes and prints.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from timemachine.data.holdout import HoldoutConfig, heldout_rows
from timemachine.mvgen.data import MVGenScene, ViewSamplingConfig
from timemachine.mvgen.model import FinetuneConfig
from timemachine.mvgen.probes import (
    DATE_VARIANTS, RenderProbe, check_view_sets, contact_sheet, describe_views, heldout_view_sets, read_keys_file,
    render_views, to_uint8,
)
from timemachine.mvgen.regions import ChangedRegions, latent_weights
from timemachine.mvgen.step import (
    LossConfig, build_inputs, cross_date_drops, date_embedding_size, encode_images, v_prediction_error_map,
)
from timemachine.mvgen.trainer import CONFIG_RECORDS, TrainConfig

if TYPE_CHECKING:
    from timemachine.mvgen.model import MVGen

AREAS = ("inside", "outside", "whole")
RENDER_DATES = ("true", "swapped")          # the two renders of a view set that are compared
# By which dates a run's cross-date rule takes priors from a render's targets: the dates rendered at (as
# probes.RenderProbe does), or the targets' own dates whatever the render's (as the loss measurements do).
PRIORS_BY = ("rendered", "true")
RELEASED = "released"                       # the checkpoint name of the released model (no adapters)
HELDOUT_SOURCE = "held-out sampler"         # ViewSet.source of the sets drawn from a run's held-out views
_CROSS_DRAWS = 64                           # held-out sets drawn per requested set, at most, when only some count
_SHEET_COLUMNS = 5                          # targets shown on a view set's sheet
_TINT = np.array([255.0, 40.0, 40.0])       # the colour of the region on a sheet
# How the kept pictures are stored: full-resolution colour, so that scores can be made again from the files.
_KEPT_JPEG = {"quality": 95, "subsampling": 0}
# The sub-configs of a TrainConfig, by field name: what config.json stores as nested dicts.
_CONFIG_PARTS = {"sampling": ViewSamplingConfig, "loss": LossConfig, "finetune": FinetuneConfig,
                 "holdout": HoldoutConfig}


# ----------------------------------------------------------------------------
# A run's settings
# ----------------------------------------------------------------------------
def _as_tuples(value: object) -> object:
    """A value read from JSON as the config dataclasses hold it: lists become tuples, at every depth."""
    return tuple(_as_tuples(item) for item in value) if isinstance(value, list) else value


def run_config(run_dir: str | Path) -> TrainConfig:
    """The ``TrainConfig`` of the run in ``run_dir``, from its ``config.json``.

    The file is ``asdict(TrainConfig)`` plus the run's records (``trainer.CONFIG_RECORDS``: the scene-scale
    factor and, for a run started from another run's weights, which weights); the result equals the
    config the run was launched with. A setting that the file lacks because the run was saved before
    the setting existed takes its default, under which the code does what it did then
    (``trainer.ADDED_FIELDS``). A file with fields that this version of the code does not know raises
    ``TypeError``, as it should: its settings are not known.
    """
    saved = json.loads((Path(run_dir) / "config.json").read_text())
    for record in CONFIG_RECORDS:
        saved.pop(record, None)
    fields = {name: _as_tuples(value) for name, value in saved.items()}
    for name, part in _CONFIG_PARTS.items():
        fields[name] = part(**{key: _as_tuples(value) for key, value in saved[name].items()})
    return TrainConfig(**fields)


def run_swap_years(run_dir: str | Path) -> tuple[float, ...]:
    """The years the run's probes move a date to, one per period, from its ``probes.json`` (NaN: no view
    stands for the period)."""
    years = json.loads((Path(run_dir) / "probes.json").read_text())["swap_years"]
    return tuple(float("nan") if year is None else float(year) for year in years)


def checkpoint_path(run_dir: str | Path, checkpoint: str) -> Path | None:
    """The checkpoint file that a ``--checkpoint`` value names; None for the released model.

    ``"latest"`` is the run's ``checkpoints/latest.pt``, a whole number its ``step_XXXXXX.pt``,
    ``"released"`` no file at all (the released model under the run's settings: the run's baseline),
    and anything else a path. A file that does not exist raises ``FileNotFoundError`` with the
    checkpoints the run has.
    """
    if checkpoint == RELEASED:
        return None
    folder = Path(run_dir) / "checkpoints"
    if checkpoint == "latest":
        path = folder / "latest.pt"
    elif checkpoint.isdigit():
        path = folder / f"step_{int(checkpoint):06d}.pt"
    else:
        path = Path(checkpoint)
    if not path.is_file():
        found = sorted(file.name for file in folder.glob("*.pt"))
        raise FileNotFoundError(f"no checkpoint {path}; {folder} holds {found}")
    return path


def open_scene(cfg: TrainConfig, camera_longest_side: float | None = None) -> MVGenScene:
    """The scene of a run as its ``Trainer`` opens it, with the run's views held out.

    The same arguments as ``Trainer.__init__`` gives ``MVGenScene``, and the hold-out of
    ``Trainer._hold_out_views``: random views per period and, if the run asks for it, the targets of its
    render probes. ``camera_longest_side`` is MVGenMaster's ``config.camera_longest_side``; by default
    ``MVGenScene``'s own default, which is the released model's value, so that the scene can be
    opened and checked before the model is loaded.
    """
    options = {} if camera_longest_side is None else {"camera_longest_side": camera_longest_side}
    scene = MVGenScene(cfg.dataset, cfg.sampling, resolution=cfg.resolution, seed=cfg.seed,
                       require_depth=not cfg.pinned_rows, scale=cfg.scale, **options)
    if cfg.holdout.active:
        targets = [row for rows in cfg.render_rows for row in rows[cfg.sampling.n_ref:]]
        scene.hold_out(heldout_rows(scene.dates, cfg.holdout, targets if cfg.holdout.probe_targets else []))
    return scene


def check_run_records(scene: MVGenScene, run_dir: str | Path, checkpoint: Mapping | None = None) -> None:
    """Raise ``ValueError`` unless ``scene`` is what the run in ``run_dir`` trained on.

    Compared, wherever the run recorded them: the held-out views (``heldout.json``: rows and file keys),
    the scene-scale factor (``config.json``) and, with a loaded ``checkpoint``, its scale factor and
    its fingerprint of the dataset folder (``MVGenScene.fingerprint``). If one differs, the folder
    was rebuilt or this version of the code draws the hold-out differently; "held-out" views would
    then include views the run trained on, and the cameras could be at another scale.
    """
    run_dir = Path(run_dir)
    differing = []
    recorded = run_dir / "heldout.json"
    if recorded.exists():
        held = [(int(row), scene.views.file_key[row]) for row in np.flatnonzero(scene.heldout)]
        if held != [(view["row"], view["file_key"]) for view in json.loads(recorded.read_text())]:
            differing.append("the held-out views are not those of heldout.json")
    records = {"config.json": json.loads((run_dir / "config.json").read_text()), "the checkpoint": checkpoint or {}}
    differing += [f"the scene-scale factor is {scene.scale_factor!r}, {where} has {record['scale_factor']!r}"
                  for where, record in records.items()
                  if "scale_factor" in record and record["scale_factor"] != scene.scale_factor]
    if records["the checkpoint"].get("dataset_fingerprint", scene.fingerprint()) != scene.fingerprint():
        differing.append("the dataset fingerprint is not the checkpoint's (the view table, the views with reliable "
                         "depth or the co-visibility changed)")
    if differing:
        raise ValueError(f"{scene.dataset_dir} is not the dataset the run in {run_dir} trained on: "
                         f"{'; '.join(differing)}. Its held-out views cannot be told from its training views.")


# ----------------------------------------------------------------------------
# View sets
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ViewSet:
    """A view set to evaluate: its name, where it comes from, its rows (references first), and whether it
    crosses the edge (:func:`crosses_edge`)."""

    name: str
    source: str
    rows: tuple[int, ...]
    cross: bool

    @property
    def group(self) -> str:
        """``"cross"`` or ``"other"``: the group whose numbers the set is averaged into."""
        return "cross" if self.cross else "other"

    @property
    def folder(self) -> str:
        """The name as a file or folder name."""
        return re.sub(r"[^\w.-]+", "_", self.name)


def crosses_edge(dates: np.ndarray, n_ref: int, edge: float) -> bool:
    """Whether a view set with these dates (references first) has every target dated from ``edge`` on and
    every reference before it: the references then show the scene's other state."""
    dates = np.asarray(dates, dtype=np.float64)
    return bool((dates[n_ref:] >= edge).all() and (dates[:n_ref] < edge).all())


def heldout_sets(scene: MVGenScene, n: int, cross_edge: float | None = None) -> list[tuple[int, np.ndarray]]:
    """Up to ``n`` view sets with held-out targets, each with its position among the sets that
    ``probes.heldout_view_sets`` draws.

    That function draws one fixed sequence of sets for a scene and its hold-out, and a run's held-out
    date-loss probe measures the first ``probe_samples`` of it. Without ``cross_edge`` the result is
    the first ``n`` of the sequence. With it, only sets that cross that edge count
    (:func:`crosses_edge`), and the sequence is followed until ``n`` of them are found or
    ``_CROSS_DRAWS * n`` sets were looked at. The position names a set (``heldout/<position>``), so a
    set keeps its name whichever others are picked.
    """
    n_ref = scene.config.n_ref
    asked, chosen = n, []
    while n > 0:
        drawn = heldout_view_sets(scene, asked)
        chosen = [(i, rows) for i, rows in enumerate(drawn)
                  if cross_edge is None or crosses_edge(scene.dates[rows], n_ref, cross_edge)]
        if len(chosen) >= n or len(drawn) < asked or asked >= _CROSS_DRAWS * n:
            break
        asked *= 2
    return chosen[:n]


def collect_view_sets(scene: MVGenScene, keys_files: Sequence[Path], n_heldout: int = 0, cross_only: bool = False,
                      edge: float | None = None) -> list[ViewSet]:
    """The view sets to evaluate: those of the keys files, then ``n_heldout`` held-out sets of the scene.

    A set of a keys file is named by its ``"name"`` entry, or ``<file stem>/<position>``; a held-out set
    ``heldout/<position>`` (:func:`heldout_sets`; with ``cross_only``, only sets that cross ``edge``,
    which does not limit the sets of keys files). A set that was already collected is left out, with
    a note. ``edge`` (default: the scene's last period edge) also decides which sets are ``cross``.

    Two different sets under one name, or under names that give one folder of renders (two keys files
    with the same file name, a ``"name"`` used twice), raise ``ValueError``: records, pictures and the
    group means are kept by name, so one of the two sets would silently stand for both.
    """
    n_ref = scene.config.n_ref
    edge = scene.config.period_edges[-1] if edge is None else edge
    table_keys = scene.views["file_key"].tolist()
    candidates: list[tuple[str, str, tuple[int, ...]]] = []
    for path in keys_files:
        names = [sample.get("name", f"{path.stem}/{i}") for i, sample in enumerate(json.loads(path.read_text()))]
        candidates += [(name, path.name, rows) for name, rows in zip(names, read_keys_file(path, table_keys, n_ref))]
    candidates += [(f"heldout/{i}", HELDOUT_SOURCE, tuple(int(row) for row in rows))
                   for i, rows in heldout_sets(scene, n_heldout, edge if cross_only else None)]
    check_view_sets([rows for _, _, rows in candidates], n_ref, len(scene.views))
    view_sets: dict[tuple[int, ...], ViewSet] = {}
    for name, source, rows in candidates:
        if rows in view_sets:
            print(f"   {name} ({source}) is the view set {view_sets[rows].name}: evaluated once", flush=True)
            continue
        view_sets[rows] = ViewSet(name, source, rows, crosses_edge(scene.dates[list(rows)], n_ref, edge))
    folders = [view_set.folder for view_set in view_sets.values()]
    clashing = sorted({view_set.name for view_set in view_sets.values() if folders.count(view_set.folder) > 1})
    if clashing:
        raise ValueError(f"different view sets would share a name or a folder of renders: {clashing}. Give the "
                         'samples of the keys files distinct "name" entries.')
    return list(view_sets.values())


def groups_of(view_sets: Sequence[ViewSet]) -> dict[str, list[str]]:
    """The names of the view sets by group: ``cross``, ``other`` and, when both exist, ``all``."""
    groups = {group: [view_set.name for view_set in view_sets if view_set.group == group]
              for group in ("cross", "other")}
    if groups["cross"] and groups["other"]:
        groups["all"] = [view_set.name for view_set in view_sets]
    return {group: names for group, names in groups.items() if names}


# ----------------------------------------------------------------------------
# Areas
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetMasks:
    """The changed region in each target of a view set (T targets, crops of side S)."""

    masks: np.ndarray       # bool [T, S, S], as scored: all False for a target whose region is too small
    pixels: np.ndarray      # int [T]: the size of each target's region, before that rule

    @property
    def counted(self) -> np.ndarray:
        """Bool [T]: the targets that have ``inside`` numbers."""
        return self.masks.any(axis=(1, 2))


def target_masks(regions: ChangedRegions, target_rows: Sequence[int], resolution: int, min_pixels: int
                 ) -> TargetMasks:
    """The masks of ``regions`` for the given target views, with regions of fewer than ``min_pixels`` pixels
    emptied (see "Small regions" in the module docstring)."""
    masks = np.stack([regions.mask(int(row), resolution) for row in target_rows])
    pixels = masks.sum(axis=(1, 2))
    masks[pixels < min_pixels] = False
    return TargetMasks(masks=masks, pixels=pixels)


def area_means(values: np.ndarray, weights: np.ndarray) -> dict[str, float | None]:
    """The mean of ``values`` inside a region, outside it and over everything.

    ``weights`` has the shape of ``values`` and says how much of each value lies inside the region: a
    bool mask, or shares in [0, 1] (a mask at the latents' resolution). ``inside`` weights the values
    by it and ``outside`` by its complement, so with ``share = weights.mean()``,
    ``whole == share * inside + (1 - share) * outside``. An area without any weight is None.
    """
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError(f"values {values.shape} and weights {weights.shape} must have one shape")
    means: dict[str, float | None] = {}
    for area, weight in (("inside", weights), ("outside", 1.0 - weights)):
        total = weight.sum()
        means[area] = float((values * weight).sum() / total) if total > 0 else None
    means["whole"] = float(values.mean())
    return means


def to_psnr(mse: float | None) -> float | None:
    """PSNR in dB of a mean squared error of values in [0, 1] (as ``probes.psnr``); None stays None."""
    return None if mse is None else float(10 * np.log10(1.0 / max(mse, 1e-10)))


def _by_area(per_target: Sequence[Mapping[str, float | None]]) -> dict[str, list[float | None]]:
    """Per-target area dicts as one list per area."""
    return {area: [values[area] for values in per_target] for area in AREAS}


# ----------------------------------------------------------------------------
# Loss with fixed noise
# ----------------------------------------------------------------------------
def loss_noise(noise_seed: int, timestep: int, rows: Sequence[int], shape: Sequence[int]) -> torch.Tensor:
    """The noise of one loss measurement: it depends on the noise seed, the timestep and the view set's rows
    and on nothing else, so a view set's numbers do not depend on which other sets are evaluated with
    it, and every date variant is measured on the same noisy latents."""
    entropy = [int(noise_seed), int(timestep), *(int(row) for row in rows)]
    seed = int(np.random.SeedSequence(entropy).generate_state(1)[0])
    return torch.randn(tuple(shape), generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def measure_losses(model: MVGen, probe: RenderProbe, masks: TargetMasks, timesteps: Sequence[int],
                   noise_seeds: Sequence[int]) -> list[dict]:
    """The loss of one view set by area, for every timestep, noise seed and date variant; gradient-free.

    ``probe`` carries the view set: its sample, the run's ``LossConfig``, the period edges and the dates
    of each variant (``RenderProbe.view_dates``, which gives the model what ``DateLossProbe`` gives it).
    The conditions are those of ``DateLossProbe`` (see the module docstring).

    Returns one record per (timestep, noise seed, variant), in that order: ``set``, ``variant``,
    ``timestep``, ``noise_seed``; ``inside``, ``outside`` and ``whole``, the mean squared error over
    the latent cells of all targets in each area (``whole`` is ``step.v_prediction_loss``; ``inside``
    is None if no target has a scored region); ``per_target``, the same per target; and, for the
    true dates, ``date_rms`` and ``date_temb_ratio`` (``step.date_embedding_size``).
    """
    cfg, n_ref, sample = probe.loss_cfg, probe.n_ref, probe.sample
    steps = model.scheduler.config.num_train_timesteps
    if min(timesteps) < 0 or max(timesteps) >= steps:
        raise ValueError(f"timesteps {list(timesteps)} are outside the scheduler's {steps} training steps")
    latents = encode_images(model, sample["images"], sample_posterior=False)
    prior_views, pixel_views = cross_date_drops(sample["date_mid"], n_ref, cfg, probe.period_edges)
    inputs = build_inputs(model, sample, latents, dropped_prior_views=prior_views, dropped_pixel_views=pixel_views)
    weights = np.stack([latent_weights(mask, mask.shape[-1] // latents.shape[-1]) for mask in masks.masks])
    records = []
    for timestep, seed in product(timesteps, noise_seeds):
        noise = loss_noise(seed, timestep, probe.rows, latents.shape)
        for variant in DATE_VARIANTS:
            year, valid = probe.view_dates(variant)
            error = v_prediction_error_map(model, latents, inputs, timestep, noise, year, valid, cfg, n_ref)
            error = error.double().cpu().numpy()
            record = {"set": probe.name, "variant": variant, "timestep": int(timestep), "noise_seed": int(seed),
                      **area_means(error, weights),
                      "per_target": _by_area([area_means(e, w) for e, w in zip(error, weights)])}
            if variant == "true":
                record.update(date_embedding_size(model))
            records.append(record)
    return records


# ----------------------------------------------------------------------------
# Renders
# ----------------------------------------------------------------------------
def priors_differ(probe: RenderProbe) -> bool:
    """Whether the run's cross-date rule gives the probe's targets other priors at their swapped dates than at
    their true dates. The two renders of the view set then differ in more than the date, unless they are made
    with ``priors_by="true"`` (:func:`render_at`). Never so under ``cross_date_pixels="keep"``."""
    true, swapped = (cross_date_drops(probe.view_dates(date)[0], probe.n_ref, probe.loss_cfg, probe.period_edges)
                     for date in RENDER_DATES)
    return any(views is not None and not torch.equal(views, other) for views, other in zip(true, swapped))


@torch.no_grad()
def render_at(model: MVGen, pipeline, probe: RenderProbe, date: str | float, seed: int,
              priors_by: str = "rendered") -> np.ndarray:
    """One render of the probe's view set at one of its dates, uint8 ``[F, H, W, 3]``: what
    ``RenderProbe.__call__`` renders for that date and seed, handed back instead of being put on a contact sheet.

    ``priors_by`` (one of ``PRIORS_BY``) says by which dates the run's cross-date rule takes priors from the
    targets: ``"rendered"``, the dates rendered at, as ``RenderProbe`` does; or ``"true"``, the targets' own
    dates, so that a render at the swapped dates gets the priors of the render at the true dates.
    """
    if priors_by not in PRIORS_BY:
        raise ValueError(f"priors_by must be one of {PRIORS_BY}, got {priors_by!r}")
    cfg = probe.loss_cfg
    year, valid = probe.view_dates(date)
    rule_dates = year if priors_by == "rendered" else probe.sample["date_mid"]
    prior_views, pixel_views = cross_date_drops(rule_dates, probe.n_ref, cfg, probe.period_edges)
    return render_views(model, pipeline, probe.sample, year, valid, seed=seed, steps=probe.steps,
                        guidance=probe.guidance, class_label=cfg.class_label, precision=cfg.precision,
                        uncond_dates=probe.uncond_dates, dropped_prior_views=prior_views,
                        dropped_pixel_views=pixel_views)


def score_renders(real: np.ndarray, true: np.ndarray, swapped: np.ndarray, masks: np.ndarray) -> dict:
    """The scores of one pair of renders of a view set's targets (uint8 ``[T, H, W, 3]`` each; ``masks`` bool
    ``[T, H, W]``).

    Returns ``psnr``: per render (``"true"``, ``"swapped"``) and area, the PSNR in dB of each target
    against its real photo; and ``difference``: per area, the mean absolute difference (0-1) between
    the two renders of each target. An entry is None where a target has no such area.
    """
    def per_pixel(a: np.ndarray, b: np.ndarray, power: int) -> np.ndarray:
        return ((np.abs(a.astype(np.float64) - b.astype(np.float64)) / 255) ** power).mean(axis=-1)

    psnr = {}
    for date, render in zip(RENDER_DATES, (true, swapped)):
        errors = [area_means(per_pixel(image, photo, 2), mask) for image, photo, mask in zip(render, real, masks)]
        psnr[date] = {area: [to_psnr(mse) for mse in values] for area, values in _by_area(errors).items()}
    difference = _by_area([area_means(per_pixel(a, b, 1), mask) for a, b, mask in zip(true, swapped, masks)])
    return {"psnr": psnr, "difference": difference}


def tinted(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """``images`` (uint8 ``[T, H, W, 3]``) with the pixels of ``masks`` (bool ``[T, H, W]``) tinted red."""
    marked = images.astype(np.float32)
    marked[masks] = 0.5 * marked[masks] + 0.5 * _TINT
    return marked.astype(np.uint8)


@torch.no_grad()
def render_set(model: MVGen, pipeline, probe: RenderProbe, masks: TargetMasks, out_dir: Path, sheet_path: Path,
               thumb: int = 192, priors_by: str = "rendered") -> list[dict]:
    """Render one view set at its true and at its swapped dates with each of the probe's seeds, keep the
    pictures and score them; gradient-free. ``priors_by``: as for :func:`render_at`.

    Writes into ``out_dir``, with ``j`` the position of a target in the view set:
    ``real_view<j>.jpg`` (the real photo), ``mask_view<j>.png`` (its region as scored) and, per seed
    ``k``, ``true_seed<k>_view<j>.jpg`` and ``swapped_seed<k>_view<j>.jpg``. ``sheet_path`` gets a small
    sheet of the first seed: the real photos, the two renders and the photos with the region tinted,
    for at most five targets, each row labelled with its PSNR inside the region.

    Returns one record per seed: ``set``, ``seed`` and the scores of :func:`score_renders`.
    """
    n_ref = probe.n_ref
    positions = range(n_ref, len(probe.rows))
    real = to_uint8(probe.sample["images"])[n_ref:]
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, photo, mask in zip(positions, real, masks.masks):
        Image.fromarray(photo).save(out_dir / f"real_view{j}.jpg", **_KEPT_JPEG)
        Image.fromarray(mask).save(out_dir / f"mask_view{j}.png")
    records = []
    for seed in probe.seeds:
        frames = {date: render_at(model, pipeline, probe, date, seed, priors_by)[n_ref:] for date in RENDER_DATES}
        for date, images in frames.items():
            for j, image in zip(positions, images):
                Image.fromarray(image).save(out_dir / f"{date}_seed{seed}_view{j}.jpg", **_KEPT_JPEG)
        records.append({"set": probe.name, "seed": int(seed),
                        **score_renders(real, frames["true"], frames["swapped"], masks.masks)})
        if seed == probe.seeds[0]:
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            _sheet(probe, masks, real, frames, records[-1], thumb).save(sheet_path, quality=88)
    return records


def _sheet(probe: RenderProbe, masks: TargetMasks, real: np.ndarray, frames: Mapping[str, np.ndarray], record: dict,
           thumb: int) -> Image.Image:
    """The sheet of one seed's renders (:func:`render_set`): ``probes.contact_sheet`` of the targets alone."""
    shown = slice(0, _SHEET_COLUMNS)
    target_rows, years = probe.rows[probe.n_ref:], probe.sample["date_mid"][probe.n_ref:]
    captions = [f"row {row}  {float(year):.2f}  {mask.mean():.0%}"
                for row, year, mask in zip(target_rows, years, masks.masks)]
    swapped_to = ", ".join(dict.fromkeys(f"{float(year):.1f}" for year in probe.swapped[probe.n_ref:]))
    labels = {"true": "true dates", "swapped": f"dates swapped to {swapped_to}"}
    rows = []
    for date in RENDER_DATES:
        inside = _mean(record["psnr"][date]["inside"])
        score = "no region scored" if inside is None else f"inside the region {inside:.1f} dB"
        rows.append((f"{labels[date]}, seed {record['seed']}\n{score}", frames[date][shown]))
    rows.append(("changed region (red)", tinted(real, masks.masks)[shown]))
    title = (f"{probe.name}: targets rendered from {probe.n_ref} references at their true and at their swapped dates "
             "(above each: its row, its date, the region's share of its frame)")
    return contact_sheet(title, real[shown], captions[shown], rows, 0, thumb)


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------
def _mean(values: Sequence[float | None]) -> float | None:
    """The mean of the values that are not None; None if there is none."""
    known = [value for value in values if value is not None]
    return float(np.mean(known)) if known else None


def mean_and_error(values: Sequence[float]) -> tuple[float, float | None]:
    """The mean of the values and its standard error (None for fewer than two values)."""
    error = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return float(np.mean(values)), error


def summarise_set_losses(records: Sequence[dict]) -> dict[str, dict | None]:
    """One view set's loss records (:func:`measure_losses`) as numbers per area.

    Per area: ``loss_true``, ``loss_swapped`` and ``loss_unknown``, the means over timesteps and noise
    seeds; ``gap_swapped`` and ``gap_unknown``, the means of ``loss_variant / loss_true - 1`` over the
    same (above zero: the true date gives the lower loss); and ``gap_swapped_by_timestep`` and
    ``gap_unknown_by_timestep``, those ratios per timestep, in the order the timesteps were measured.
    An area the set does not have is None.
    """
    def measured(variant: str, area: str) -> np.ndarray:
        return np.array([record[area] for record in records if record["variant"] == variant], dtype=np.float64)

    timestep = np.array([record["timestep"] for record in records if record["variant"] == "true"])
    summary: dict[str, dict | None] = {}
    for area in AREAS:
        true = measured("true", area)
        if np.isnan(true).any():                # None in the records: the set has no such area
            summary[area] = None
            continue
        summary[area] = {"loss_true": float(true.mean())}
        for variant in DATE_VARIANTS[1:]:
            loss = measured(variant, area)
            gap = loss / true - 1
            summary[area].update({
                f"loss_{variant}": float(loss.mean()), f"gap_{variant}": float(gap.mean()),
                f"gap_{variant}_by_timestep": [float(gap[timestep == t].mean()) for t in dict.fromkeys(timestep)]})
    return summary


def summarise_set_renders(records: Sequence[dict]) -> dict[str, dict | None]:
    """One view set's render records (:func:`render_set`) as numbers per area.

    Per area, as means over the seeds and the targets that have the area: ``psnr_true`` and
    ``psnr_swapped`` (dB against the real photos), ``psnr_gain`` (their difference: above zero, the
    true date gives the better picture) and ``difference`` (between the two renders, 0-1). An area
    that no target has is None.
    """
    summary: dict[str, dict | None] = {}
    for area in AREAS:
        psnr = {date: _mean([value for record in records for value in record["psnr"][date][area]])
                for date in RENDER_DATES}
        if psnr["true"] is None:
            summary[area] = None
            continue
        summary[area] = {"psnr_true": psnr["true"], "psnr_swapped": psnr["swapped"],
                         "psnr_gain": psnr["true"] - psnr["swapped"],
                         "difference": _mean([value for record in records for value in record["difference"][area]])}
    return summary


def summarise_groups(view_sets: Sequence[ViewSet], per_set: Mapping[str, Mapping[str, dict | None]]) -> dict:
    """The view sets' summaries averaged per group of sets (:func:`groups_of`) and area.

    ``per_set`` gives, by view-set name, :func:`summarise_set_losses` or :func:`summarise_set_renders`.
    Returns ``{group: {"sets": n, area: {...}}}``. An area's dict holds ``sets``, the number of the
    group's sets that have the area, and for each of their numbers its mean over those sets under
    the same name and the standard error of that mean under ``<name>_se`` (None for a single set);
    a list (the gaps by timestep) is averaged entry by entry, without an error.
    """
    summary = {}
    for group, names in groups_of(view_sets).items():
        summary[group] = {"sets": len(names)}
        for area in AREAS:
            entries = [per_set[name][area] for name in names if per_set[name][area] is not None]
            cell: dict = {"sets": len(entries)}
            for key in (entries[0] if entries else ()):
                values = np.array([entry[key] for entry in entries], dtype=np.float64)
                if values.ndim == 1:
                    cell[key], cell[f"{key}_se"] = mean_and_error(values)
                else:
                    cell[key] = values.mean(axis=0).tolist()
            summary[group][area] = cell
    return summary


def format_table(title: str, summary: Mapping[str, Mapping], columns: Sequence[tuple[str, str, str]]) -> str:
    """A summary of :func:`summarise_groups` as text under ``title``: one block per group of view sets, one line
    per area.

    ``columns`` are ``(heading, name of the number, its format)``; a cell reads ``mean +- standard error``
    over the view sets counted under ``sets``, ``n/a`` for the error of a single set, and ``-`` where
    no set has the area.
    """
    def cell(numbers: Mapping, key: str, form: str) -> str:
        if key not in numbers:
            return "-"
        error = numbers[f"{key}_se"]
        return f"{numbers[key]:{form}} +- " + ("n/a" if error is None else f"{error:{form.lstrip('+')}}")

    width = 2 + max(20, *(len(heading) for heading, _, _ in columns))
    lines = [title, f"{'view sets':<12}{'area':<9}{'sets':>4}  " + "".join(f"{h:<{width}}" for h, _, _ in columns)]
    for group, areas in summary.items():
        for i, area in enumerate(AREAS):
            label = f"{group} ({areas['sets']})" if i == 0 else ""
            cells = "".join(f"{cell(areas[area], key, form):<{width}}" for _, key, form in columns)
            lines.append(f"{label:<12}{area:<9}{areas[area]['sets']:>4}  {cells}".rstrip())
    return "\n".join(lines)


def describe_set(scene: MVGenScene, view_set: ViewSet, probe: RenderProbe, masks: TargetMasks) -> dict:
    """A view set for the records: what ``probes.describe_views`` says of its views, plus its name, the name of
    its folder of renders, its source and group, the years its targets are swapped to, its scene-scale
    factor, which of its views the scene holds out of training (``held_out``, per view; None when the
    scene holds out nothing), and per target the size of its region in pixels (``region_pixels``), as
    a share of the frame (``region_share``) and whether it is scored (``region_scored``)."""
    n_ref = probe.n_ref
    rows = list(view_set.rows)
    return {"name": view_set.name, "folder": view_set.folder, "source": view_set.source, "group": view_set.group,
            **describe_views(scene, rows, n_ref),
            "swapped_years": [round(float(year), 4) for year in probe.swapped[n_ref:]],
            "scale_factor": float(probe.sample["scale"]),
            "held_out": scene.heldout[rows].tolist() if scene.heldout.any() else None,
            "region_pixels": masks.pixels.tolist(),
            "region_share": [float(pixels) / masks.masks[0].size for pixels in masks.pixels],
            "region_scored": masks.counted.tolist()}
