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

**Experiment look1** (``eval_dates.py --eval-sets``; nothing above changes with it). Does an explicit,
untrained per-view look code let the date learn the place's state instead of the photos' look? The
evaluation-sets file (``configs/look1_eval_sets.json``, ``scripts/mvgen/make_eval_sets.py``;
:func:`read_eval_sets`, :func:`check_eval_sets`) holds a 2x2 of reference period x target period
(modern / after the fire) of sets whose scored targets are held out by whole days and photographers,
each scored target's hold-out unit, two donors per target and yardsticks measured on real training
photos. Regions are the dataset folder's stored masks (:func:`stored_masks`).

A set is rendered under *conditions* (:func:`look1_conditions`): a look for the targets (``LOOK1_LOOKS``;
the references always keep their own codes) at a date (``LOOK1_DATES``). The *base* look is the
references' code (``look.references_code``) in a model with a look, no look input otherwise. Primary
conditions, the base at the true, the cross-fire (``probes.swapped_years``) and the within-period date
(each target to the farther of its period's two ``within_years``), get 4 seeds; the others (the neutral
code, the two donors, the grey code: the references' code of the references converted to grey as arm D
converts a view, the own code, an old date) 2. Every set has its own noise, and all conditions of a set
share it at a seed (:func:`render_seed`), so the swaps are paired. The looks reach the model as every
other caller gives them: the ``look`` and ``look_known`` arguments of ``probes.render_views`` and of the
step's loss functions (:func:`view_looks` makes them; ``MVGen.dates`` takes them). Each scored
target of each render is scored (:func:`measure_target`): channel means, the six look measures of the
appearance study (:func:`look_measures`) and the CIELAB chroma outside its region, the PSNR, the tone
rule's verdict and the colour-blind state read-out (``timemachine.mvgen.readout``).

The read-outs, each with the spec's pass line (:func:`summarise_look1`, :func:`look1_lines`):

* R1, steering: the slope of the render's channel means outside the region on the requested ones, donor
  render minus base render against donor code minus references' code (:func:`clustered_slope`). Pass and
  early gate: the mean over channels at least 0.4, every channel at least 0.25. Grey flag: the chroma with
  the grey code within 2 SE of real grey photos';
* R2, the date decides the state at the base look: the read-out's flip for the cross-fire swap minus the
  within-period swap; the loss version, the wrong-date cost inside minus outside the region
  (:func:`measure_look1_losses`). A look arm passes if it is at least A's minus max(1.5, 2 |A - A2|);
* R3, the date lets go of the look (primary): the cross-fire swap's shift of the six look measures, in the
  real photos' within-period sd, projected on the real after-fire-minus-modern gap (:func:`look_projection`);
  the within-period swap is the null, the neutral code a variant, the absolute outside difference
  secondary. A look arm passes if it is at most half of A's and A minus it is at least max(0.15, 3 |A - A2|);
* R4, the look does not carry the state: on modern targets, the read-out with the after-fire donor's code
  minus with the modern donor's, as a share of the date flip; pass under 1 logit and 15%, with an upper 95%
  bound (units resampled, :func:`bootstrap_ratio_bound`) under 25%;
* R5, medium: modern targets at an old date with the references' code and at their own date with the grey
  code: chroma and tone rule;
* R6, quality: PSNR with the base look, with the own code, and (the released model) after matching the
  render's colour to the photo; differences under 0.8 dB are not interpretable.

Means are over scored targets (each distinct target once per cell) and their standard errors are
clustered by hold-out unit (:func:`clustered_mean`). The cross-arm lines (:func:`compare_look1`) pair the
arms by target. :func:`date_profile` is the weight-level read-out: how far the date embedding moves inside
a period against across the fire.
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

from timemachine.data.holdout import HoldoutConfig
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
    ``Trainer._hold_out_views`` (``MVGenScene.holdout_rows``): random views per period, or whole days and
    photographers (``holdout.by="component"``), and, if the run asks for it, the targets of its render
    probes. ``camera_longest_side`` is MVGenMaster's ``config.camera_longest_side``; by default
    ``MVGenScene``'s own default, which is the released model's value, so that the scene can be
    opened and checked before the model is loaded.
    """
    options = {} if camera_longest_side is None else {"camera_longest_side": camera_longest_side}
    scene = MVGenScene(cfg.dataset, cfg.sampling, resolution=cfg.resolution, seed=cfg.seed,
                       require_depth=not cfg.pinned_rows, scale=cfg.scale, **options)
    if cfg.holdout.active:
        targets = [row for rows in cfg.render_rows for row in rows[cfg.sampling.n_ref:]]
        scene.hold_out(scene.holdout_rows(cfg.holdout, targets if cfg.holdout.probe_targets else []))
    return scene


def check_run_records(scene: MVGenScene, run_dir: str | Path, checkpoint: Mapping | None = None) -> None:
    """Raise ``ValueError`` unless ``scene`` is what the run in ``run_dir`` trained on.

    Compared, wherever the run recorded them: the held-out views (``heldout.json``: rows and file keys),
    the scene-scale factor (``config.json``) and, with a loaded ``checkpoint``, its scale factor and
    its fingerprint of the dataset folder (``MVGenScene.fingerprint``); a multi-scene checkpoint
    (``scripts/mvgen/train_multi.py``) records no single factor (None). If one differs, the folder
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
                  if record.get("scale_factor") is not None and record["scale_factor"] != scene.scale_factor]
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


# ============================================================================
# Experiment look1 (scripts/mvgen/eval_dates.py --eval-sets): sets, conditions, measures, read-outs R1-R6
# ============================================================================
LOOK1_PERIODS = ("old", "modern", "after")      # the period names of an evaluation-sets file
LOOK1_READOUTS = ("R1", "R2", "R3", "R4", "R5", "R6", "profile")
# What the targets are rendered with: no look input (a model without a look), the references' code, the
# population mean, their own codes, their donors' codes from the other period or from their own period, or the
# references' code of the references converted to grey.
LOOK1_LOOKS = ("none", "references", "neutral", "own", "other", "same", "grey")
# At which date: their own, across the fire (probes.swapped_years), across years inside their period, or old.
LOOK1_DATES = ("true", "swapped", "within", "old")
LOOK_MEASURES = ("brightness", "contrast", "saturation", "warmth", "sky_blueness", "ground_green_share")
_LUMA_601 = np.array([0.299, 0.587, 0.114])
_MIN_MEASURE_PIXELS = 16                # fewer pixels in a part of the frame: its measure is NaN
_RENDER_STREAM = 7_200                  # marks the per-set noise seeds of the renders
_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])
_D65 = np.array([0.95047, 1.0, 1.08883])
# The pass lines of the look1 spec (scratchpad look1_spec.md, "Read-outs").
GATE_MEAN_SLOPE, GATE_CHANNEL_SLOPE = 0.4, 0.25         # R1 and the early gate at step 2,000
R2_MARGIN = 1.5                                         # logits: the floor of B's non-inferiority margin against A
R3_MIN_DROP = 0.15                                      # sd: A - B must be at least this (or 3 x |A - A2|)
R4_MAX_LOGIT, R4_MAX_SHARE, R4_MAX_UPPER = 1.0, 0.15, 0.25
R6_NOISE_DB = 0.8                                       # PSNR differences under this are not interpretable
BOOTSTRAP_DRAWS = 2000


# ----------------------------------------------------------------------------
# look1: pixel measures (photos and renders alike; uint8 or float [H, W, 3] in [0, 1])
# ----------------------------------------------------------------------------
def _unit_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    return image.astype(np.float64) / 255 if image.dtype == np.uint8 else image.astype(np.float64)


def hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue in degrees [0, 360), saturation and value in [0, 1] of RGB in [0, 1] (as OpenCV's float HSV)."""
    value, low = rgb.max(-1), rgb.min(-1)
    delta = value - low
    saturation = np.where(value > 0, delta / np.where(value > 0, value, 1.0), 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    safe = np.where(delta > 0, delta, 1.0)
    hue = np.where(value == r, 60 * (g - b) / safe, np.where(value == g, 120 + 60 * (b - r) / safe,
                                                             240 + 60 * (r - g) / safe))
    return np.where(delta > 0, hue % 360, 0.0), saturation, value


def look_measures(image: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """The six look measures of the appearance study (``confounds/measures.py``, the ones its look-gap
    projection keeps) over the pixels of ``keep``: float ``[6]`` in the order of ``LOOK_MEASURES``.

    brightness and contrast: mean and sd of the Rec. 601 luma; saturation: mean HSV saturation; warmth:
    red minus blue mean; sky_blueness: blue minus red mean in the top third of the frame; ground_green_share:
    the share of green pixels (hue 60-170 degrees, saturation above 0.2, value above 0.15) in its lower
    half. NaN where the part has fewer than 16 kept pixels.
    """
    rgb, keep = _unit_rgb(image), np.asarray(keep, dtype=bool)
    out = np.full(len(LOOK_MEASURES), np.nan)
    if keep.sum() < _MIN_MEASURE_PIXELS:
        return out
    luma = rgb @ _LUMA_601
    hue, saturation, value = hsv(rgb)
    means = rgb[keep].mean(0)
    out[:4] = luma[keep].mean(), luma[keep].std(), saturation[keep].mean(), means[0] - means[2]
    top = keep.copy()
    top[rgb.shape[0] // 3:] = False
    if top.sum() >= _MIN_MEASURE_PIXELS:
        sky = rgb[top].mean(0)
        out[4] = sky[2] - sky[0]
    low = keep.copy()
    low[:rgb.shape[0] // 2] = False
    if low.sum() >= _MIN_MEASURE_PIXELS:
        out[5] = ((hue[low] >= 60) & (hue[low] <= 170) & (saturation[low] > 0.2) & (value[low] > 0.15)).mean()
    return out


def channel_means(image: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """float ``[3]``: the mean of each sRGB channel (0-1) over the pixels of ``keep``."""
    return _unit_rgb(image)[np.asarray(keep, dtype=bool)].mean(0)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (D65; L 0-100) of sRGB values in [0, 1], ``[..., 3]``."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (linear @ _SRGB_TO_XYZ.T) / _D65
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def mean_chroma(image: np.ndarray, keep: np.ndarray) -> float:
    """The mean CIELAB chroma (sqrt(a^2 + b^2)) over the pixels of ``keep``: about 0 for a grey picture."""
    lab = srgb_to_lab(_unit_rgb(image)[np.asarray(keep, dtype=bool)])
    return float(np.hypot(lab[:, 1], lab[:, 2]).mean())


def match_moments(render: np.ndarray, photo: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``render`` with each channel's mean and spread over ``keep`` moved to ``photo``'s (no pixel correspondence;
    the appearance study's after-the-fact colour matching of R6), float in [0, 1]."""
    render, photo, keep = _unit_rgb(render), _unit_rgb(photo), np.asarray(keep, dtype=bool)
    out = render.copy()
    for c in range(3):
        r, p = render[..., c][keep], photo[..., c][keep]
        out[..., c] = (render[..., c] - r.mean()) / max(float(r.std()), 1e-4) * float(p.std()) + float(p.mean())
    return np.clip(out, 0, 1)


def psnr_over(a: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None) -> float:
    """PSNR in dB of two pictures over the pixels of ``keep`` (all pixels without it)."""
    squared = ((_unit_rgb(a) - _unit_rgb(b)) ** 2).mean(-1)
    return float(to_psnr(float(squared.mean() if keep is None else squared[np.asarray(keep, dtype=bool)].mean())))


def _finite(value: float) -> float | None:
    """A number for the records: None for NaN or infinity."""
    return float(value) if value is not None and np.isfinite(value) else None


# ----------------------------------------------------------------------------
# look1: the evaluation sets (configs/look1_eval_sets.json, scripts/mvgen/make_eval_sets.py)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Look1Set:
    """One view set of a look1 cell: its references first, which targets are scored (held out), each target's
    hold-out unit and two donors per target (``"other"``: of the other period, ``"same"``: of its own)."""

    name: str
    cell: str
    reference_period: str
    target_period: str
    rows: tuple[int, ...]
    n_ref: int
    scored: tuple[bool, ...]
    units: tuple[int, ...]
    donors: Mapping[str, tuple[int, ...]]
    strict: bool

    @property
    def targets(self) -> tuple[int, ...]:
        return self.rows[self.n_ref:]

    @property
    def folder(self) -> str:
        return re.sub(r"[^\w.-]+", "_", self.name)

    @property
    def sign(self) -> int:
        """+1 for modern targets, -1 for after-fire ones: a shift times the sign is positive "towards the other
        period" for a swap across the fire, and towards "after" for the look measures."""
        return 1 if self.target_period == "modern" else -1


@dataclass(frozen=True)
class Look1Plan:
    """An evaluation-sets file: its sets, the hold-out it was made for, the within-period years and the
    yardsticks measured on real photos."""

    path: Path
    sets: tuple[Look1Set, ...]
    heldout: tuple[int, ...]
    seed: int
    within_years: Mapping[str, tuple[float, float]]
    yardsticks: Mapping[str, object]

    def target_key(self, view_set: Look1Set, row: int) -> str:
        return f"{view_set.name}|{row}"

    def scored_targets(self) -> dict[str, tuple[Look1Set, int, int]]:
        """``{"<set>|<row>": (set, row, unit)}`` for every scored target, in the order of the sets."""
        return {self.target_key(s, row): (s, row, unit)
                for s in self.sets for row, unit, scored in zip(s.targets, s.units, s.scored) if scored}


def read_eval_sets(path: str | Path, scene: MVGenScene, strict: bool = False, cells: Sequence[str] | None = None
                   ) -> Look1Plan:
    """The sets of an evaluation-sets file for ``scene``: its main sets, or with ``strict`` its strict ones, of all
    cells or of ``cells``. Every row the file names is checked by its file key: ``ValueError`` if the scene's view
    table is not the one the file was made for."""
    path = Path(path)
    data = json.loads(path.read_text())
    keys = scene.views["file_key"].tolist()

    def checked(rows: Sequence[int], file_keys: Sequence[str], what: str) -> tuple[int, ...]:
        rows = tuple(int(row) for row in rows)
        if any(row >= len(keys) for row in rows) or [keys[row] for row in rows] != list(file_keys):
            raise ValueError(f"{path}: the rows of {what} are not those of {scene.dataset_dir}'s view table (made "
                             "for another table?)")
        return rows

    heldout = checked(data["holdout"]["rows"], data["holdout"]["file_keys"], "the hold-out")
    unknown = sorted(set(cells or ()) - set(data["cells"]))
    if unknown:
        raise ValueError(f"{path} has no cells {unknown}; it has {sorted(data['cells'])}")
    sets = []
    for cell, content in data["cells"].items():
        if cells and cell not in cells:
            continue
        for entry in content["strict_sets" if strict else "sets"]:
            donors = {kind: checked(entry["donors"][kind], entry["donors"][f"{kind}_file_keys"], f"{entry['name']}'s "
                                    f"{kind} donors") for kind in ("other", "same")}
            sets.append(Look1Set(name=entry["name"], cell=cell, reference_period=content["reference_period"],
                                 target_period=content["target_period"],
                                 rows=checked(entry["rows"], entry["file_keys"], entry["name"]), n_ref=entry["n_ref"],
                                 scored=tuple(entry["scored"]), units=tuple(entry["units"]), donors=donors,
                                 strict=bool(entry["strict"])))
    return Look1Plan(path=path, sets=tuple(sets), heldout=heldout, seed=int(data["holdout"]["seed"]),
                     within_years={period: tuple(years) for period, years in data["within_years"].items()},
                     yardsticks=data["yardsticks"])


def check_eval_sets(plan: Look1Plan, scene: MVGenScene) -> None:
    """``ValueError`` unless ``scene`` holds out every view the sets were made for, every scored target is held out,
    and every reference and unscored target is a usable training view. A run may hold out more views (its render
    probes' targets), which is printed: the sets stay valid as long as their references and unscored targets
    train."""
    held = set(np.flatnonzero(scene.heldout).tolist())
    missing = set(plan.heldout) - held
    if missing:
        raise ValueError(f"{plan.path} was made for a hold-out of {len(plan.heldout)} views (seed {plan.seed}); the "
                         f"run trains on {len(missing)} of them. Its scored targets would not be the run's held-out "
                         "views.")
    if held - set(plan.heldout):
        print(f"note: the run holds out {len(held - set(plan.heldout))} views more than {plan.path.name} (its render "
              "probes' targets?)", flush=True)
    for view_set in plan.sets:
        training = [row for row, scored in zip(view_set.targets, view_set.scored) if not scored]
        if not all(scene.heldout[row] for row, scored in zip(view_set.targets, view_set.scored) if scored):
            raise ValueError(f"{view_set.name}: a scored target is not held out")
        if any(scene.heldout[row] or not scene.usable[row] for row in (*view_set.rows[:view_set.n_ref], *training)):
            raise ValueError(f"{view_set.name}: a reference or an unscored target is not a usable training view")


def stored_masks(scene: MVGenScene, rows: Sequence[int]) -> np.ndarray:
    """bool ``[T, S, S]``: the changed region of each view as the dataset folder stores it (``changed_regions.npz``:
    every latent cell with a masked pixel), at the scene's resolution. look1 scores renders, photos, the look
    gap and the read-out with these masks."""
    if scene.regions is None:
        raise ValueError(f"{scene.dataset_dir} has no changed-region file: look1's read-outs need one")
    cells = scene.regions[np.asarray(rows, dtype=int)] > 0
    factor = scene.resolution // cells.shape[-1]
    return np.repeat(np.repeat(cells, factor, axis=1), factor, axis=2)


# ----------------------------------------------------------------------------
# look1: conditions (what a set is rendered with), dates, looks and noise
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """A render of a set's targets: with which look (``LOOK1_LOOKS``) at which date (``LOOK1_DATES``); primary
    conditions get the larger number of seeds."""

    look: str
    date: str
    primary: bool

    @property
    def name(self) -> str:
        return f"{self.look}/{self.date}"


def look1_conditions(readouts: Sequence[str], has_date: bool, has_look: bool, target_period: str) -> list[Condition]:
    """The conditions a set needs for ``readouts`` (see the module docstring, "Experiment look1"), primary first.

    The references' code is the targets' look in a model with a look, no look input otherwise ("base"). A
    model without a date (the released model) needs only the base at the true date (R6). A model without a
    look skips what needs one: the donors (R1, R4), the neutral code (R3), the own code (R6) and R5. The old
    date of R5 is rendered for sets of modern targets only.
    """
    readouts = set(readouts)
    base = "references" if has_look else "none"
    wanted: dict[str, Condition] = {}

    def add(look: str, date: str, primary: bool = False) -> None:
        wanted.setdefault(f"{look}/{date}", Condition(look, date, primary))

    if not has_date:
        if "R6" in readouts:
            add("none", "true", True)
        return list(wanted.values())
    if readouts & {"R2", "R3", "R4"}:
        for date in ("true", "swapped", "within"):
            add(base, date, True)
    if readouts & {"R1", "R6"}:
        add(base, "true", True)
    if has_look:
        if "R3" in readouts:
            add("neutral", "true")
            add("neutral", "swapped")
        if readouts & {"R1", "R4"}:
            add("other", "true")
            add("same", "true")
        if readouts & {"R1", "R5"}:
            add("grey", "true")
        if "R5" in readouts and target_period == "modern":
            add(base, "old")
        if "R6" in readouts:
            add("own", "true")
    return sorted(wanted.values(), key=lambda c: (not c.primary, LOOK1_LOOKS.index(c.look), LOOK1_DATES.index(c.date)))


def render_seed(rows: Sequence[int], index: int) -> int:
    """The noise seed of a set's ``index``-th render seed: every set gets other noise, and every condition of a set
    the same noise at a seed (the date swaps are paired at a seed)."""
    return int(np.random.SeedSequence([_RENDER_STREAM, int(index), *(int(row) for row in rows)]).generate_state(1)[0])


def look1_years(date: str, probe: RenderProbe, view_set: Look1Set, plan: Look1Plan, swap_years: Sequence[float]
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(year, valid)`` ``[F]`` of one of ``LOOK1_DATES`` for the probe's set.

    ``"within"`` moves each target to the one of its period's two within-period years (the file's
    ``within_years``, e.g. 2013.4 and 2016.5 for modern targets) that is farther from its own date;
    ``"old"`` to the representative year of the oldest period (``swap_years[0]``).
    """
    if date in ("true", "swapped"):
        return probe.view_dates(date)
    if date not in LOOK1_DATES:
        raise ValueError(f"a look1 date is one of {LOOK1_DATES}, not {date!r}")
    year, valid = probe.view_dates("true")
    n_ref = probe.n_ref
    if date == "old":
        year[n_ref:] = float(swap_years[0])
        return year, valid
    pair = np.asarray(plan.within_years[view_set.target_period], dtype=np.float64)
    own = year[n_ref:].double().numpy()
    farther = np.abs(own[:, None] - pair[None, :]).argmax(axis=1)
    year[n_ref:] = torch.from_numpy(pair[farther]).float()
    return year, valid


@dataclass
class LookInputs:
    """What the looks of a model with a look are made from: the scene's codes (``look.read_look``), the model's
    standardisation (``MVGen.look_stats``), the scene's class maps (``look.read_class_maps``, for the grey code)
    and, for ``scripts/mvgen/render.py``, a tone the targets' codes are given instead of their own."""

    look: Mapping[str, np.ndarray]
    stats: object                       # timemachine.mvgen.look.LookStats
    class_maps: np.ndarray | None = None
    tone: str | None = None


def grey_references_code(sample: dict, rows: Sequence[int], n_ref: int, inputs: LookInputs) -> np.ndarray:
    """raw ``[24]``: the references' code (``look.references_code``) of the references converted to grey as arm D
    converts a colour view (``look.convert_medium``; codes from the converted pixels, tone grey by construction)."""
    from timemachine.mvgen import look as looks

    if inputs.class_maps is None:
        raise ValueError(f"the grey code needs the scene's class maps ({looks.CLASS_MAP_FILE})")
    codes, shown = [], []
    for j in range(n_ref):
        row = int(rows[j])
        grey = looks.convert_medium(looks.to_image(sample["images"][j]), "grey")
        code, seen = looks.look_code(grey, inputs.class_maps[row],
                                     looks.excluded_pixels(inputs.look["region"][row], grey.shape[0]), tone="grey")
        codes.append(code)
        shown.append(seen)
    return looks.references_code(np.stack(codes), np.stack(shown))


def target_codes(kind: str, rows: Sequence[int], n_ref: int, sample: dict | None, inputs: LookInputs,
                 donors: Mapping[str, Sequence[int]] | None = None) -> np.ndarray:
    """raw float64 ``[T, 24]``: the codes the targets of a set (``rows``, references first) are given under one of
    ``LOOK1_LOOKS`` (not ``"none"``), or ``"donor"``. A donor look (``"other"``, ``"same"``, ``"donor"``) takes the
    codes of the rows ``donors[kind]``, one per target. Only the grey code reads the set's pictures (``sample``,
    its ``images``); every other look comes from the look file, and ``sample`` may then be None."""
    from timemachine.mvgen import look as looks

    codes, shown = np.asarray(inputs.look["codes"]), np.asarray(inputs.look["shown"], dtype=bool)
    refs, targets = list(rows[:n_ref]), list(rows[n_ref:])
    if kind == "references":
        one = looks.references_code(codes[refs], shown[refs])
    elif kind == "neutral":
        one = np.asarray(inputs.stats.mean)                 # in float64: standardised, exactly 0
    elif kind == "grey":
        one = grey_references_code(sample, rows, n_ref, inputs)
    elif kind == "own":
        return codes[targets].astype(np.float64)
    elif kind in ("other", "same", "donor"):
        chosen = list((donors or {}).get(kind, ()))
        if len(chosen) != len(targets):
            raise ValueError(f"the look {kind!r} needs one donor row per target ({len(targets)}), got {chosen}")
        return codes[chosen].astype(np.float64)
    else:
        raise ValueError(f"no codes for the look {kind!r}")
    return np.repeat(np.asarray(one, dtype=np.float64)[None], len(targets), axis=0)


def view_looks(kind: str, rows: Sequence[int], n_ref: int, sample: dict | None, inputs: LookInputs | None,
               donors: Mapping[str, Sequence[int]] | None = None
               ) -> tuple[np.ndarray | None, torch.Tensor | None, torch.Tensor | None]:
    """``(raw target codes [T, 24], look [F, 24], known [F])`` of a set under a look: the references keep their own
    codes, the targets get :func:`target_codes` (with ``inputs.tone``, if given, as their tone); all standardised
    with the model's statistics and known. ``(None, None, None)`` for ``"none"``."""
    if kind == "none":
        return None, None, None
    if inputs is None:
        raise ValueError(f"the look {kind!r} needs a model with a look and the scene's look codes")
    from timemachine.mvgen import look as looks

    raw = target_codes(kind, rows, n_ref, sample, inputs, donors).copy()
    if inputs.tone is not None:
        raw[:, looks.TONE] = np.eye(len(looks.TONES))[looks.TONES.index(inputs.tone)]
    references = np.asarray(inputs.look["codes"])[list(rows[:n_ref])]
    look = inputs.stats.standardise(np.concatenate([references, raw]))
    return raw, torch.from_numpy(look), torch.ones(len(rows), dtype=torch.bool)


# ----------------------------------------------------------------------------
# look1: rendering and measuring a set
# ----------------------------------------------------------------------------
def measure_target(render: np.ndarray, photo: np.ndarray, mask: np.ndarray) -> dict:
    """What a render of one target is scored by, outside its region unless named ``whole``: the channel means
    (``means``), the six look measures (``measures``), the mean chroma, the PSNR against the photo over the whole
    frame and outside the region, and the same after its colour is matched to the photo (:func:`match_moments`)."""
    keep = ~np.asarray(mask, dtype=bool)
    matched = match_moments(render, photo, keep)
    return {"means": channel_means(render, keep).tolist(),
            "measures": [_finite(value) for value in look_measures(render, keep)],
            "chroma": mean_chroma(render, keep),
            "psnr_whole": psnr_over(render, photo), "psnr_outside": psnr_over(render, photo, keep),
            "psnr_matched_whole": psnr_over(matched, photo), "psnr_matched_outside": psnr_over(matched, photo, keep)}


@torch.no_grad()
def render_look1_set(model: MVGen, pipeline, probe: RenderProbe, view_set: Look1Set, plan: Look1Plan,
                     conditions: Sequence[Condition], seeds: Mapping[bool, int], swap_years: Sequence[float],
                     masks: np.ndarray, inputs: LookInputs | None, readout=None, out_dir: Path | None = None,
                     priors_by: str = "rendered") -> tuple[list[dict], list[dict]]:
    """Render a set under every condition with its seeds (``seeds[condition.primary]`` of them) and score its scored
    targets; gradient-free.

    ``masks``: bool ``[T, S, S]`` of the targets (:func:`stored_masks`); ``readout``: a
    ``readout.StateReadout`` or None; ``out_dir``: where the scored targets' renders are kept
    (``<look>_<date>_seed<k>_view<j>.jpg``, ``real_view<j>.jpg``, ``mask_view<j>.png``), or None.

    Returns ``(records, pairs)``. A record per condition, seed and scored target: ``set``, ``cell``,
    ``condition``, ``seed``, ``row``, ``position``, ``unit``, the scores of :func:`measure_target`, ``tone`` (the
    tone rule's verdict on the whole render), ``logit`` (the state read-out; None if not read) and
    ``requested`` (the channel means of the code the target was given; None without a look). A pair per seed,
    scored target and swap (``swapped``, ``within``) of the base look: the mean absolute difference (0-1) of the
    two renders outside and inside the region.
    """
    from timemachine.mvgen.look import tone_rule

    cfg, n_ref = probe.loss_cfg, probe.n_ref
    real = to_uint8(probe.sample["images"])[n_ref:]
    scored = [t for t, is_scored in enumerate(view_set.scored) if is_scored]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for t in scored:
            Image.fromarray(real[t]).save(out_dir / f"real_view{n_ref + t}.jpg", **_KEPT_JPEG)
            Image.fromarray(masks[t]).save(out_dir / f"mask_view{n_ref + t}.png")
    records, frames_of = [], {}
    for condition in conditions:
        year, valid = look1_years(condition.date, probe, view_set, plan, swap_years)
        raw, look, known = view_looks(condition.look, view_set.rows, n_ref, probe.sample, inputs, view_set.donors)
        rule_dates = year if priors_by == "rendered" else probe.sample["date_mid"]
        prior_views, pixel_views = cross_date_drops(rule_dates, n_ref, cfg, probe.period_edges)
        for k in range(seeds[condition.primary]):
            frames = render_views(model, pipeline, probe.sample, year, valid, seed=render_seed(view_set.rows, k),
                                  steps=probe.steps, guidance=probe.guidance, class_label=cfg.class_label,
                                  precision=cfg.precision, uncond_dates=probe.uncond_dates,
                                  dropped_prior_views=prior_views, dropped_pixel_views=pixel_views, look=look,
                                  look_known=known)[n_ref:]
            frames_of[condition.name, k] = frames
            logits = (readout.readout_logit(frames[scored], masks[scored]) if readout is not None
                      else np.full(len(scored), np.nan))
            for t, logit in zip(scored, logits):
                if out_dir is not None:
                    name = f"{condition.look}_{condition.date}_seed{k}_view{n_ref + t}.jpg"
                    Image.fromarray(frames[t]).save(out_dir / name, **_KEPT_JPEG)
                records.append({"set": view_set.name, "cell": view_set.cell, "condition": condition.name, "seed": k,
                                "row": int(view_set.targets[t]), "position": n_ref + t, "unit": int(view_set.units[t]),
                                **measure_target(frames[t], real[t], masks[t]),
                                "tone": tone_rule(frames[t].astype(np.float32) / 255), "logit": _finite(logit),
                                "requested": None if raw is None else raw[t, :3].tolist()})
    base = "references" if inputs is not None else "none"
    pairs = []
    for (name, k), frames in frames_of.items():
        if not name.startswith(f"{base}/") or name.split("/")[1] not in ("swapped", "within"):
            continue
        true = frames_of.get((f"{base}/true", k))
        if true is None:
            continue
        for t in scored:
            difference = (np.abs(frames[t].astype(np.float64) - true[t].astype(np.float64)) / 255).mean(-1)
            pairs.append({"set": view_set.name, "row": int(view_set.targets[t]), "seed": k, "swap": name.split("/")[1],
                          "difference_outside": float(difference[~masks[t]].mean()),
                          "difference_inside": _finite(difference[masks[t]].mean() if masks[t].any() else np.nan)})
    return records, pairs


@torch.no_grad()
def measure_look1_losses(model: MVGen, probe: RenderProbe, view_set: Look1Set, plan: Look1Plan, masks: np.ndarray,
                         timesteps: Sequence[int], noise_seeds: Sequence[int], swap_years: Sequence[float],
                         inputs: LookInputs | None) -> list[dict]:
    """The loss of a set by area at the base look (the references' code, or no look) for the dates ``true``,
    ``swapped``, ``within`` and ``unknown``, as :func:`measure_losses` measures it (the same noise, conditioning
    and cross-date rule by the true dates); gradient-free. One record per timestep, noise seed and date, with
    ``per_target`` areas."""
    cfg, n_ref, sample = probe.loss_cfg, probe.n_ref, probe.sample
    base = "references" if inputs is not None else "none"
    _, look, known = view_looks(base, view_set.rows, n_ref, sample, inputs)
    variants = {date: look1_years(date, probe, view_set, plan, swap_years) for date in ("true", "swapped", "within")}
    variants["unknown"] = probe.view_dates("unknown")
    latents = encode_images(model, sample["images"], sample_posterior=False)
    prior_views, pixel_views = cross_date_drops(sample["date_mid"], n_ref, cfg, probe.period_edges)
    inputs_ = build_inputs(model, sample, latents, dropped_prior_views=prior_views, dropped_pixel_views=pixel_views)
    weights = np.stack([latent_weights(mask, mask.shape[-1] // latents.shape[-1]) for mask in masks])
    records = []
    for timestep, seed in product(timesteps, noise_seeds):
        noise = loss_noise(seed, timestep, probe.rows, latents.shape)
        for variant, (year, valid) in variants.items():
            error = v_prediction_error_map(model, latents, inputs_, timestep, noise, year, valid, cfg, n_ref,
                                           look=look, look_known=known)
            error = error.double().cpu().numpy()
            records.append({"set": view_set.name, "variant": variant, "timestep": int(timestep),
                            "noise_seed": int(seed), **area_means(error, weights),
                            "per_target": _by_area([area_means(e, w) for e, w in zip(error, weights)])})
    return records


# ----------------------------------------------------------------------------
# look1: statistics over hold-out units
# ----------------------------------------------------------------------------
def clustered_mean(values: Mapping[str, float], units: Mapping[str, int]) -> dict:
    """The mean of per-target values and its standard error clustered by hold-out unit.

    ``SE^2 = G / (G - 1) * sum_g (sum_{t in g} (v_t - mean))^2 / n^2`` over the ``G`` units of the ``n``
    targets: targets of one day or photographer are not counted as independent (with one target per unit it
    is the ordinary standard error). Non-finite values are left out. Returns ``mean``, ``se`` (None for fewer
    than two units), ``targets`` and ``units``; ``mean`` None without any value.
    """
    keys = [key for key, value in values.items() if value is not None and np.isfinite(value)]
    if not keys:
        return {"mean": None, "se": None, "targets": 0, "units": 0}
    x = np.array([values[key] for key in keys], dtype=np.float64)
    groups = np.array([units[key] for key in keys])
    mean, n = float(x.mean()), len(x)
    sums = np.array([(x[groups == g] - mean).sum() for g in np.unique(groups)])
    n_units = len(sums)
    se = float(np.sqrt(n_units / (n_units - 1) * (sums ** 2).sum()) / n) if n_units > 1 else None
    return {"mean": mean, "se": se, "targets": n, "units": n_units}


def clustered_slope(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float | None]:
    """The least-squares slope of ``y`` on ``x`` through the origin and its standard error clustered by ``groups``."""
    x, y, groups = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(groups)
    sxx = float((x * x).sum())
    if sxx == 0:
        return float("nan"), None
    slope = float((x * y).sum() / sxx)
    scores = np.array([(x[groups == g] * (y[groups == g] - slope * x[groups == g])).sum() for g in np.unique(groups)])
    n_units = len(scores)
    se = float(np.sqrt(n_units / (n_units - 1) * (scores ** 2).sum()) / sxx) if n_units > 1 else None
    return slope, se


def bootstrap_ratio_bound(numerator: Mapping[str, float], denominator: Mapping[str, float], units: Mapping[str, int],
                          quantile: float = 0.95, draws: int = BOOTSTRAP_DRAWS, seed: int = 0) -> float | None:
    """The ``quantile`` of ``mean(numerator) / mean(denominator)`` over targets, with the hold-out units resampled
    (one-sided bound; None without targets that have both)."""
    keys = [key for key in numerator if key in denominator and np.isfinite(numerator[key])
            and np.isfinite(denominator[key])]
    if not keys:
        return None
    groups = sorted({units[key] for key in keys})
    sums = np.array([[sum(numerator[k] for k in keys if units[k] == g), sum(denominator[k] for k in keys if units[k] == g)]
                     for g in groups])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), size=(draws, len(groups)))
    totals = sums[picks].sum(axis=1)
    ratios = totals[:, 0] / np.where(totals[:, 1] != 0, totals[:, 1], np.nan)
    return _finite(np.nanquantile(ratios, quantile))


class Look1Records:
    """The records of :func:`render_look1_set` indexed by scored target (``"<set>|<row>"``), condition and seed."""

    def __init__(self, records: Sequence[Mapping], plan: Look1Plan):
        self.plan = plan
        self.targets = plan.scored_targets()
        self.units = {key: unit for key, (_, _, unit) in self.targets.items()}
        self._by: dict[tuple[str, str], dict[int, Mapping]] = {}
        for record in records:
            key = f"{record['set']}|{record['row']}"
            self._by.setdefault((key, record["condition"]), {})[record["seed"]] = record

    def has(self, condition: str) -> bool:
        return any(name == condition for _, name in self._by)

    def sign(self, key: str) -> int:
        return self.targets[key][0].sign

    def period(self, key: str) -> str:
        return self.targets[key][0].target_period

    def single(self, condition: str, value) -> dict[str, float]:
        """Per target: the mean over its seeds of ``value(record)`` (non-finite values left out)."""
        out = {}
        for key in self.targets:
            values = [value(r) for r in self._by.get((key, condition), {}).values()]
            values = [v for v in values if v is not None and np.isfinite(v)]
            if values:
                out[key] = float(np.mean(values))
        return out

    def paired(self, first: str, second: str, value) -> dict[str, float]:
        """Per target: the mean over the seeds both conditions have of ``value(second) - value(first)``."""
        out = {}
        for key in self.targets:
            a, b = self._by.get((key, first), {}), self._by.get((key, second), {})
            differences = [value(b[s]) - value(a[s]) for s in sorted(set(a) & set(b))
                           if value(a[s]) is not None and value(b[s]) is not None]
            differences = [d for d in differences if np.isfinite(d)]
            if differences:
                out[key] = float(np.mean(differences))
        return out

    def pairs(self, first: str, second: str) -> tuple[list[str], list[tuple[Mapping, Mapping]]]:
        """The targets and their record pairs ``(first, second)``, one per seed both conditions have."""
        keys, pairs = [], []
        for key in self.targets:
            a, b = self._by.get((key, first), {}), self._by.get((key, second), {})
            for seed in sorted(set(a) & set(b)):
                keys.append(key)
                pairs.append((a[seed], b[seed]))
        return keys, pairs

    def records(self, condition: str, period: str | None = None) -> list[Mapping]:
        return [r for (key, name), by_seed in self._by.items() if name == condition
                and (period is None or self.period(key) == period) for r in by_seed.values()]

    def groups(self, values: Mapping[str, float]) -> dict[str, dict]:
        """:func:`clustered_mean` of per-target values over all targets, per target period and per cell."""
        out = {"all": clustered_mean(values, self.units)}
        for period in ("modern", "after"):
            out[period] = clustered_mean({k: v for k, v in values.items() if self.period(k) == period}, self.units)
        for cell in dict.fromkeys(s.cell for s in self.plan.sets):
            out[cell] = clustered_mean({k: v for k, v in values.items() if self.targets[k][0].cell == cell}, self.units)
        return out


def _logit(record: Mapping) -> float | None:
    return record["logit"]


# ----------------------------------------------------------------------------
# look1: read-outs
# ----------------------------------------------------------------------------
def _fmt(stats: Mapping | None, form: str = "+.2f") -> str:
    """``mean +- se (targets / units)`` of a :func:`clustered_mean`."""
    if not stats or stats.get("mean") is None:
        return "n/a"
    se = "n/a" if stats["se"] is None else f"{stats['se']:{form.lstrip('+')}}"
    return f"{stats['mean']:{form}} +- {se} ({stats['targets']} targets / {stats['units']} units)"


def _verdict(passed: bool | None) -> str:
    return "n/a" if passed is None else ("PASS" if passed else "FAIL")


def _real(stats: Mapping) -> str:
    """``mean +- se (photos)`` of a yardstick of real photos."""
    if stats.get("mean") is None:
        return "n/a"
    return f"{stats['mean']:.2f} +- {stats['se'] or 0:.2f} ({stats['photos']} photos)"


def summarise_r1(recs: Look1Records, yardsticks: Mapping) -> dict:
    """R1, steering: the slope of the render's channel means outside the region on the requested ones, donor render
    minus references'-code render against donor code minus references' code, paired at a seed (both donors of
    every scored target), per channel with a clustered error; and the grey flag: the render's chroma outside the
    region with the grey code against real grey photos'."""
    x, y, groups = [], [], []
    for donor in ("other", "same"):
        keys, pairs = recs.pairs("references/true", f"{donor}/true")
        for key, (references, donated) in zip(keys, pairs):
            x.append(np.subtract(donated["requested"], references["requested"]))
            y.append(np.subtract(donated["means"], references["means"]))
            groups.append(recs.units[key])
    out: dict = {"points": len(x)}
    if x:
        x, y = np.array(x), np.array(y)
        slopes = [clustered_slope(x[:, c], y[:, c], np.array(groups)) for c in range(3)]
        out["slope"] = [s for s, _ in slopes]
        out["slope_se"] = [e for _, e in slopes]
        out["mean_slope"] = float(np.mean(out["slope"]))
        out["gate"] = bool(out["mean_slope"] >= GATE_MEAN_SLOPE and min(out["slope"]) >= GATE_CHANNEL_SLOPE)
    if recs.has("grey/true"):
        grey = recs.groups(recs.single("grey/true", lambda r: r["chroma"]))
        real = yardsticks["chroma"]["grey"]
        out["grey_chroma"] = grey
        mean, se = grey["all"]["mean"], grey["all"]["se"] or 0.0
        out["grey_flag"] = (None if mean is None or real["mean"] is None else
                            bool(abs(mean - real["mean"]) <= 2 * np.hypot(se, real["se"] or 0.0)))
    return out


def _r1_lines(r1: Mapping, yardsticks: Mapping) -> list[str]:
    if "slope" not in r1:
        return ["R1 steering: no donor renders (a model without a look, or R1 not asked)"]
    channels = ", ".join(f"{c} {s:+.3f} +- {e if e is None else round(e, 3)}"
                         for c, s, e in zip("RGB", r1["slope"], r1["slope_se"]))
    lines = [f"R1 steering: slope of the render's channel means on the requested ones ({r1['points']} paired points): "
             f"{channels}; mean {r1['mean_slope']:+.3f} | pass and early gate (mean >= {GATE_MEAN_SLOPE}, every "
             f"channel >= {GATE_CHANNEL_SLOPE}): {_verdict(r1['gate'])}"
             + ("" if r1["gate"] else " -> switch this arm to the per-block injection (look_injection film)")]
    if "grey_chroma" in r1:
        real = yardsticks["chroma"]["grey"]
        lines.append(f"R1 grey flag: render chroma outside the region with the grey code {_fmt(r1['grey_chroma']['all'])}"
                     f"; real grey photos {_real(real)} | within 2 SE: {_verdict(r1['grey_flag'])}")
    return lines


def r2_loss_gaps(loss_records: Sequence[Mapping], plan: Look1Plan) -> dict[str, float]:
    """Per scored target with a region: the mean over timesteps and noise seeds of the cross-fire swap's loss gap
    inside the region minus outside it (``(swapped / true - 1)`` per area)."""
    by_set: dict[str, dict] = {}
    for record in loss_records:
        by_set.setdefault(record["set"], {}).setdefault((record["timestep"], record["noise_seed"]), {})[
            record["variant"]] = record["per_target"]
    sets = {s.name: s for s in plan.sets}
    out = {}
    for name, measured in by_set.items():
        view_set = sets[name]
        for t, (row, scored) in enumerate(zip(view_set.targets, view_set.scored)):
            gaps = []
            for variants in measured.values():
                true, swapped = variants["true"], variants["swapped"]
                if not scored or true["inside"][t] is None:
                    continue
                gaps.append((swapped["inside"][t] / true["inside"][t] - 1) - (swapped["outside"][t] / true["outside"][t] - 1))
            if gaps:
                out[f"{name}|{row}"] = float(np.mean(gaps))
    return out


def summarise_r2(recs: Look1Records, base: str, loss_records: Sequence[Mapping]) -> dict:
    """R2, the date decides the state at a fixed look: the read-out's flip for the cross-fire swap minus the
    within-period swap (signed towards the other period), per target; and the loss version."""
    cross = recs.paired(f"{base}/true", f"{base}/swapped", _logit)
    within = recs.paired(f"{base}/true", f"{base}/within", _logit)
    cross = {k: v * recs.sign(k) for k, v in cross.items()}
    within = {k: v * recs.sign(k) for k, v in within.items()}
    flip = {k: cross[k] - within[k] for k in cross if k in within}
    out = {"flip": recs.groups(flip), "cross": recs.groups(cross), "within": recs.groups(within),
           "per_target": {"flip": flip, "cross": cross}}
    if loss_records:
        gaps = r2_loss_gaps(loss_records, recs.plan)
        out["loss_inside_minus_outside"] = recs.groups(gaps)
        out["per_target"]["loss"] = gaps
    return out


def _r2_lines(r2: Mapping) -> list[str]:
    lines = [f"R2 state at the base look, read-out flip cross-fire minus within-period swap (logits): all "
             f"{_fmt(r2['flip']['all'])}; modern targets {_fmt(r2['flip']['modern'])}; after-fire "
             f"{_fmt(r2['flip']['after'])} | cross-fire alone {_fmt(r2['cross']['all'])}, within alone "
             f"{_fmt(r2['within']['all'])} | non-inferiority of a look arm against A (B >= A - max({R2_MARGIN}, "
             "2 |A - A2|)): with --compare"]
    if "loss_inside_minus_outside" in r2:
        lines.append(f"R2 loss version, wrong-date cost inside minus outside the region: all "
                     f"{_fmt(r2['loss_inside_minus_outside']['all'], '+.2%')}; modern "
                     f"{_fmt(r2['loss_inside_minus_outside']['modern'], '+.2%')}; after-fire "
                     f"{_fmt(r2['loss_inside_minus_outside']['after'], '+.2%')} (secondary)")
    return lines


def look_projection(recs: Look1Records, first: str, second: str, yardsticks: Mapping) -> dict[str, float]:
    """Per target: the shift of the six look measures from ``first`` to ``second`` (signed: modern targets +,
    after-fire -), in the real photos' within-period sd, averaged over the paired seeds (a measure NaN at every seed
    counts 0), projected on the unit direction of the real after-fire-minus-modern gap."""
    gap = yardsticks["look_gap"]
    sd, unit = np.asarray(gap["sd"]), np.asarray(gap["unit"])
    keys, pairs = recs.pairs(first, second)
    shifts: dict[str, list[np.ndarray]] = {}
    for key, (a, b) in zip(keys, pairs):
        shifts.setdefault(key, []).append((np.array(b["measures"], dtype=float) - np.array(a["measures"], dtype=float))
                                          / sd)
    out = {}
    for key, of_key in shifts.items():
        stacked = np.stack(of_key)
        known = np.isfinite(stacked)
        counts = known.sum(0)
        mean = np.where(counts > 0, np.where(known, stacked, 0.0).sum(0) / np.maximum(counts, 1), 0.0)
        out[key] = float(recs.sign(key) * mean @ unit)
    return out


def summarise_r3(recs: Look1Records, base: str, pairs: Sequence[Mapping], yardsticks: Mapping) -> dict:
    """R3, the date lets go of the look (primary): the projection of the cross-fire swap's shift on the real look gap
    at the base look; the within-period swap (null) and, with a look, the neutral code; the absolute outside
    difference of the swap (secondary)."""
    cross = look_projection(recs, f"{base}/true", f"{base}/swapped", yardsticks)
    within = look_projection(recs, f"{base}/true", f"{base}/within", yardsticks)
    out = {"cross": recs.groups(cross), "within": recs.groups(within), "per_target": {"cross": cross, "within": within}}
    if recs.has("neutral/swapped"):
        neutral = look_projection(recs, "neutral/true", "neutral/swapped", yardsticks)
        out["neutral"] = recs.groups(neutral)
        out["per_target"]["neutral"] = neutral
    absolute: dict[str, list[float]] = {}
    for pair in pairs:
        if pair["swap"] == "swapped":
            absolute.setdefault(f"{pair['set']}|{pair['row']}", []).append(pair["difference_outside"])
    out["absolute_outside"] = recs.groups({k: float(np.mean(v)) for k, v in absolute.items() if k in recs.units})
    return out


def _r3_lines(r3: Mapping) -> list[str]:
    within = r3["within"]["all"]
    null_ok = (None if within["mean"] is None or within["se"] is None else abs(within["mean"]) <= 2 * within["se"])
    lines = [f"R3 look of the cross-fire swap projected on the real look gap (sd; primary): all {_fmt(r3['cross']['all'], '+.3f')}"
             f"; modern targets {_fmt(r3['cross']['modern'], '+.3f')}; after-fire {_fmt(r3['cross']['after'], '+.3f')} | "
             f"pass for a look arm (B <= A / 2 and A - B >= max({R3_MIN_DROP}, 3 |A - A2|)): with --compare",
             f"R3 null, within-period swap: {_fmt(within, '+.3f')} | about 0 (within 2 SE): {_verdict(null_ok)}"]
    if "neutral" in r3:
        lines.append(f"R3 at the neutral code (population mean): {_fmt(r3['neutral']['all'], '+.3f')}")
    lines.append(f"R3 secondary, absolute outside difference of the swap (0-1): {_fmt(r3['absolute_outside']['all'], '.4f')}"
                 " (a global code can remove at most 20-30% of it)")
    return lines


def summarise_r4(recs: Look1Records, base: str) -> dict:
    """R4, the look does not carry the state: the read-out with the other-period donor's code minus with the
    same-period donor's code at the true date (signed towards the donor's period), as a share of the cross-fire
    date flip of the same targets; modern targets are the test."""
    leak = recs.paired("same/true", "other/true", _logit)
    leak = {k: v * recs.sign(k) for k, v in leak.items()}
    flip = {k: v * recs.sign(k) for k, v in recs.paired(f"{base}/true", f"{base}/swapped", _logit).items()}
    out = {"leak": recs.groups(leak), "per_target": {"leak": leak}}
    for period in ("modern", "after"):
        keys = [k for k in leak if k in flip and recs.period(k) == period]
        if keys:
            share = float(np.mean([leak[k] for k in keys]) / np.mean([flip[k] for k in keys]))
            upper = bootstrap_ratio_bound({k: leak[k] for k in keys}, {k: flip[k] for k in keys}, recs.units)
            out[f"share_{period}"], out[f"share_upper_{period}"] = share, upper
    stats, share, upper = out["leak"]["modern"], out.get("share_modern"), out.get("share_upper_modern")
    out["pass"] = (None if stats["mean"] is None or share is None or upper is None else
                   bool(stats["mean"] < R4_MAX_LOGIT and share < R4_MAX_SHARE and upper < R4_MAX_UPPER))
    return out


def _r4_lines(r4: Mapping) -> list[str]:
    def share(period: str) -> str:
        if f"share_{period}" not in r4:
            return "n/a"
        upper = r4[f"share_upper_{period}"]
        return f"{r4[f'share_{period}']:.0%} of the date flip (upper 95% {'n/a' if upper is None else f'{upper:.0%}'})"

    return [f"R4 look carries the state? read-out, other-period donor minus same-period donor (logits): modern targets "
            f"{_fmt(r4['leak']['modern'])}, {share('modern')}; after-fire {_fmt(r4['leak']['after'])}, {share('after')} "
            f"| pass (modern: < {R4_MAX_LOGIT:g} logit, < {R4_MAX_SHARE:.0%}, upper bound < {R4_MAX_UPPER:.0%}): "
            f"{_verdict(r4['pass'])}"]


def summarise_r5(recs: Look1Records, base: str) -> dict:
    """R5, medium: modern targets at an old date with the references' (colour) code, and at their true date with the
    grey code: the render's chroma outside the region and the tone rule's verdicts."""
    out = {}
    for name, condition in (("old_date_colour_code", f"{base}/old"), ("grey_code", "grey/true")):
        if not recs.has(condition):
            continue
        chroma = {k: v for k, v in recs.single(condition, lambda r: r["chroma"]).items() if recs.period(k) == "modern"}
        tones = [r["tone"] for r in recs.records(condition, "modern")]
        out[name] = {"chroma": clustered_mean(chroma, recs.units),
                     "tone_shares": {tone: tones.count(tone) / len(tones) for tone in ("grey", "toned", "colour")}
                     if tones else {}}
        out.setdefault("per_target", {})[name] = chroma
    return out


def _r5_lines(r5: Mapping, yardsticks: Mapping) -> list[str]:
    if not r5:
        return ["R5 medium: not measured (a model without a look, or R5 not asked)"]
    real = yardsticks["chroma"]
    lines = []
    for name, label in (("old_date_colour_code", "modern targets at an old date, colour code"),
                        ("grey_code", "modern targets at their date, grey code")):
        if name in r5:
            shares = ", ".join(f"{tone} {share:.0%}" for tone, share in r5[name]["tone_shares"].items())
            lines.append(f"R5 {label}: chroma outside {_fmt(r5[name]['chroma'])}; tone rule {shares}")
    lines.append(f"R5 real photos' chroma: grey {_real(real['grey'])}, toned {_real(real['toned'])}, colour "
                 f"{_real(real['colour'])} | D against B: with --compare")
    return lines


def summarise_r6(recs: Look1Records, base: str) -> dict:
    """R6, quality: the PSNR (dB) over the whole frame and outside the region at the true date with the base look,
    with the own code, and (for the released model) after matching the render's colour to the photo."""
    out = {"per_target": {}}
    fields = {"whole": "psnr_whole", "outside": "psnr_outside"}
    for name, condition, prefix in (("base", f"{base}/true", "psnr_"), ("own", "own/true", "psnr_"),
                                    ("matched", f"{base}/true", "psnr_matched_")):
        if not recs.has(condition):
            continue
        for area in fields:
            values = recs.single(condition, lambda r, field=f"{prefix}{area}": r[field])
            out[f"{name}_{area}"] = recs.groups(values)
            out["per_target"][f"{name}_{area}"] = values
    return out


def _r6_lines(r6: Mapping) -> list[str]:
    lines = []
    for name, label in (("base", "at the base look"), ("own", "with the own code"),
                        ("matched", "colour matched to the photo after the fact")):
        if f"{name}_whole" in r6:
            lines.append(f"R6 PSNR {label}: whole {_fmt(r6[f'{name}_whole']['all'])} dB, outside "
                         f"{_fmt(r6[f'{name}_outside']['all'])} dB")
    lines.append(f"R6 against the released model: with --compare released=... (differences under {R6_NOISE_DB} dB are "
                 "not interpretable)")
    return lines


def summarise_look1(records: Sequence[Mapping], pairs: Sequence[Mapping], loss_records: Sequence[Mapping],
                    plan: Look1Plan, readouts: Sequence[str], has_date: bool, has_look: bool) -> dict:
    """The look1 read-outs of an evaluation, as numbers: ``{"R1": ..., ...}`` for those asked and measurable."""
    recs = Look1Records(records, plan)
    base = "references" if has_look else "none"
    out: dict = {"base_look": base, "units": recs.units}
    if not has_date:
        return {**out, "R6": summarise_r6(recs, base)} if "R6" in readouts else out
    summarise = {"R1": lambda: summarise_r1(recs, plan.yardsticks) if has_look else None,
                 "R2": lambda: summarise_r2(recs, base, loss_records),
                 "R3": lambda: summarise_r3(recs, base, pairs, plan.yardsticks),
                 "R4": lambda: summarise_r4(recs, base) if has_look else None,
                 "R5": lambda: summarise_r5(recs, base) if has_look else None,
                 "R6": lambda: summarise_r6(recs, base)}
    for readout in readouts:
        if readout in summarise:
            result = summarise[readout]()
            if result is not None:
                out[readout] = result
    return out


def look1_lines(summary: Mapping, yardsticks: Mapping) -> list[str]:
    """The read-outs of :func:`summarise_look1` as text, each with the spec's pass line beside it."""
    lines = []
    writers = {"R1": lambda s: _r1_lines(s, yardsticks), "R2": _r2_lines, "R3": _r3_lines, "R4": _r4_lines,
               "R5": lambda s: _r5_lines(s, yardsticks), "R6": _r6_lines}
    for readout, write in writers.items():
        if readout in summary:
            lines += write(summary[readout])
    return lines


# ----------------------------------------------------------------------------
# look1: verdicts across arms (the same sets and seeds, so paired by target)
# ----------------------------------------------------------------------------
def paired_difference(this: Mapping[str, float], other: Mapping[str, float], units: Mapping[str, int]) -> dict:
    """:func:`clustered_mean` of ``this - other`` over the targets both have."""
    return clustered_mean({k: this[k] - other[k] for k in this if k in other}, units)


def _per_target(results: Mapping, readout: str, name: str) -> dict[str, float] | None:
    return results.get("look1", {}).get(readout, {}).get("per_target", {}).get(name)


def _pooled(results: Mapping, readout: str, name: str) -> float | None:
    part = results.get("look1", {}).get(readout, {}).get(name)
    return None if part is None else part["all"]["mean"]


def compare_look1(this: Mapping, others: Mapping[str, Mapping], arm: str | None = None) -> list[str]:
    """Lines comparing this evaluation with others of the same checkpoint step (``others`` by arm name: ``A``,
    ``A2``, ``B``, ``released``, ...), with the spec's cross-arm pass lines where the arms they need are given."""
    units = this["look1"]["units"]
    lines = []
    for name, other in others.items():
        for readout, measure, label, form in (("R2", "flip", "R2 read-out flip", "+.2f"),
                                              ("R3", "cross", "R3 look projection", "+.3f"),
                                              ("R4", "leak", "R4 leak", "+.2f")):
            mine, theirs = _per_target(this, readout, measure), _per_target(other, readout, measure)
            if mine and theirs:
                lines.append(f"{label}, this - {name}: {_fmt(paired_difference(mine, theirs, units), form)}")
    if "A" in others and "A2" in others:
        a, a2 = others["A"], others["A2"]
        flips = [_pooled(r, "R2", "flip") for r in (this, a, a2)]
        if None not in flips:
            margin = max(R2_MARGIN, 2 * abs(flips[1] - flips[2]))
            lines.append(f"R2 non-inferiority: this {flips[0]:+.2f}, A {flips[1]:+.2f}, A2 {flips[2]:+.2f}; margin "
                         f"{margin:.2f} | this >= A - margin: {_verdict(flips[0] >= flips[1] - margin)}")
        looks = [_pooled(r, "R3", "cross") for r in (this, a, a2)]
        if None not in looks:
            needed = max(R3_MIN_DROP, 3 * abs(looks[1] - looks[2]))
            passed = looks[0] <= looks[1] / 2 and looks[1] - looks[0] >= needed
            lines.append(f"R3 primary: this {looks[0]:+.3f}, A {looks[1]:+.3f}, A2 {looks[2]:+.3f} | this <= A / 2 and "
                         f"A - this >= {needed:.3f}: {_verdict(passed)}")
    elif any(_pooled(this, r, m) is not None for r, m in (("R2", "flip"), ("R3", "cross"))) and arm not in ("A", "A2"):
        lines.append("R2 and R3 verdicts need --compare A=<results.json> A2=<results.json> (same step)")
    if "released" in others:
        released = others["released"]
        for mine_name, theirs_name, label in (("base", "base", "base look - released raw"),
                                              ("own", "matched", "own code - released colour matched")):
            for area in ("whole", "outside"):
                mine = _per_target(this, "R6", f"{mine_name}_{area}")
                theirs = _per_target(released, "R6", f"{theirs_name}_{area}")
                if mine and theirs:
                    lines.append(f"R6 {label}, {area}: {_fmt(paired_difference(mine, theirs, units))} dB (under "
                                 f"{R6_NOISE_DB} dB: not interpretable)")
    if "B" in others and arm == "D":
        for name in ("old_date_colour_code", "grey_code"):
            mine, theirs = _per_target(this, "R5", name), _per_target(others["B"], "R5", name)
            if mine and theirs:
                lines.append(f"R5 {name.replace('_', ' ')}, chroma D - B: {_fmt(paired_difference(mine, theirs, units))}")
    return lines


# ----------------------------------------------------------------------------
# look1: the weight-level date profile
# ----------------------------------------------------------------------------
def date_profile(date_state: Mapping[str, torch.Tensor], date_freqs: int, date_hidden: int, edge: float,
                 within_years: Mapping[str, tuple[float, float]], swap_years: Sequence[float],
                 span: tuple[float, float] = (2005.0, 2023.0), spacing: float = 0.05) -> dict:
    """How far the date embedding moves inside a period against across the fire, from a checkpoint's weights.

    ``date_state`` is a checkpoint's ``trainable["date"]`` (the ``embedding.`` entries are used). Distances are
    the root mean square over the embedding's entries of a difference. ``spread_<period>``: the mean distance of
    the embeddings of a grid of years (every ``spacing`` over ``span``, cut at the ``edge``) from their period's
    mean; ``between_means``: the distance between the two periods' means; ``across``: between the representative
    years of the two periods (``swap_years[1:3]``, the dates of the cross-fire swap); ``within_<period>``: between
    the period's two within-period years (the dates of the within-period swap).
    """
    from timemachine.mvgen.date import DateEmbedding

    state = {key[len("embedding."):]: value for key, value in date_state.items() if key.startswith("embedding.")}
    embedding = DateEmbedding(state["mlp.2.weight"].shape[0], date_freqs, date_hidden)
    embedding.load_state_dict(state)
    embedding.eval()

    def embed(years: Sequence[float]) -> torch.Tensor:
        years = torch.as_tensor(np.asarray(years, dtype=np.float32))
        with torch.no_grad():
            return embedding(years, torch.ones_like(years, dtype=torch.bool)).double()

    def distance(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).pow(2).mean().sqrt())

    grid = np.arange(span[0], span[1] + spacing / 2, spacing)
    out = {}
    means = {}
    for period, years in (("modern", grid[grid < edge]), ("after", grid[grid >= edge])):
        vectors = embed(years)
        means[period] = vectors.mean(0)
        out[f"spread_{period}"] = float((vectors - means[period]).pow(2).mean(1).sqrt().mean())
    out["between_means"] = distance(means["modern"], means["after"])
    across = embed(swap_years[1:3])
    out["across"] = distance(across[0], across[1])
    for period, pair in within_years.items():
        vectors = embed(pair)
        out[f"within_{period}"] = distance(vectors[0], vectors[1])
    return out
