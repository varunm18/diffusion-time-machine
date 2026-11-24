"""Fixed evaluations of date-conditioned MVGenMaster: the loss at true / wrong / unknown dates, and renders.

Both probes use the same view sets, noise and timesteps at every evaluation, so their numbers and
pictures are comparable across training steps, and both are read against the first evaluation
(normally step 0, where the adapters are exact no-ops and the model is the released one).

**Date loss** (:class:`DateLossProbe`, cheap). For each fixed view set, at each fixed timestep and
with the same noise, the training loss on the targets is computed three times: with the TRUE dates,
with the targets' dates SWAPPED to another period, and with every date UNKNOWN. A model that uses
the date has a lower loss with the true dates: the relative gaps ``swapped / true - 1`` and
``unknown / true - 1`` rise above zero. At step 0 the three losses are equal. The probe also
reports how much each sample's true-date loss changed since the first evaluation, which is the
number an overfitting test watches, and how large the date embedding is next to the timestep
embedding it is added to (``date_temb_ratio``): a gate result is only worth reading while that
ratio is well below 1. Where the dataset folder says where the scene changed (``changed_regions.npz``,
``timemachine.mvgen.data``), the probe also gives the two gaps inside that region and outside it:
the whole frame barely registers a change that covers a few percent of it.

**Renders** (:class:`RenderProbe`, slow). A fixed view set is generated with MVGenMaster's sampling
pipeline (the references' pixels and depth are given, the targets' are not) at several dates and
with several noise seeds. It writes a contact sheet (the real views on top, then one row per date
and seed) and the PSNR against the real target photos. The UNet is given what a training step gives
it: only the UNet runs at the run's precision, and the 3D priors in particular are computed in fp32
(:func:`unet_precision`) from depth whose far values are contracted once
(``step.depth_for_priors``); cross-date targets lose the priors that the run's
``cross_date_pixels`` rule takes from them in training (:func:`view_priors_dropped`).

**Scene scale.** Both probes load their view sets with the scene's own rule
(``MVGenScene(scale=...)``: one factor for the scene, or MVGenMaster's demo rule per sample), so
they see the cameras and depth that training sees; their records name the rule and the factor.

**Which date is "wrong"** (:func:`swapped_years`). A scene's history is cut into periods by
``ViewSamplingConfig.period_edges`` (Notre-Dame: 1970 and the day after the fire). A target's date
is moved across the edge nearest to it, to one year that stands for the period on the other side
(:func:`representative_years`: the median date of the views the run trains on in that period). So
a photo from 2021 is tested at about 2015, one from 2015 at the after-fire dates, one from 1890 at
about 2015.

**Which views** come either from the caller (row lists, or file keys through
:func:`read_keys_file`, for runs pinned to chosen views) or from the scene's sampler with a fixed
seed (:func:`scene_view_sets`; :func:`heldout_view_sets` draws sets whose targets are all held out
of training).

The probes never take gradients and draw their noise from their own seeded generators. They do not
change the model's mode: the caller puts it in ``eval()`` (``Trainer.evaluation`` does).
"""
from __future__ import annotations

import copy
import json
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from timemachine.data.holdout import era_index
from timemachine.mvgen.data import MVGenScene, eligible_anchors, sample_rows
from timemachine.mvgen.step import (
    LossConfig, autocast, build_inputs, cross_date_drops, date_embedding_size, depth_for_priors, encode_images,
    region_means, target_region, v_prediction_loss, v_prediction_loss_and_map,
)
from timemachine.training.loss import shown_dates

if TYPE_CHECKING:
    from timemachine.mvgen.model import MVGen

PROBE_SAMPLE_SEED = 0           # sampler seed of the probes' view sets, whatever the training seed
PROBE_EPOCH = 1_000_000         # ... drawn in an "epoch" that training never reaches
DEFAULT_TIMESTEPS = (100, 300, 500, 700, 900)
DATE_VARIANTS = ("true", "swapped", "unknown")
REGION_AREAS = ("inside", "outside")    # of the changed region: where the date-loss probe measures besides the frame
# What a render can be dated with, besides a year: each target's own date, its swapped date, or no date at all.
RENDER_DATE_WORDS = DATE_VARIANTS
_HELDOUT_ATTEMPTS = 200         # sampler draws tried per requested held-out view set

# Contact sheet layout, in pixels.
_LABEL_WIDTH, _HEADER, _CAPTION = 230, 18, 14


# ----------------------------------------------------------------------------
# Which views
# ----------------------------------------------------------------------------
def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def rows_of_file_keys(table_keys: Sequence[str], wanted: Sequence[str]) -> list[int]:
    """Rows of a view table for the given file keys, in the order given.

    ``table_keys`` is the table's ``file_key`` column. Keys are compared in Unicode NFC form, so an
    accented name matches however it was typed. A key that is not in the table, or that names several
    rows, raises ``KeyError``.
    """
    index: dict[str, list[int]] = {}
    for row, key in enumerate(table_keys):
        index.setdefault(_nfc(key), []).append(row)
    missing = [key for key in wanted if _nfc(key) not in index]
    if missing:
        raise KeyError(f"file keys that are not in the view table: {missing}")
    repeated = [key for key in wanted if len(index[_nfc(key)]) > 1]
    if repeated:
        raise KeyError(f"file keys that name several rows of the view table: {repeated}")
    return [index[_nfc(key)][0] for key in wanted]


def read_rows_file(path: str | Path) -> tuple[tuple[int, ...], ...]:
    """View sets from a JSON file: a list of row lists, each with its references first."""
    view_sets = json.loads(Path(path).read_text())
    if not isinstance(view_sets, list) or not all(isinstance(rows, list) for rows in view_sets):
        raise ValueError(f"{path}: expected a JSON list of row lists")
    return tuple(tuple(int(row) for row in rows) for rows in view_sets)


def read_keys_file(path: str | Path, table_keys: Sequence[str], n_ref: int) -> tuple[tuple[int, ...], ...]:
    """View sets from a JSON file that names the views by file key, as rows of the view table.

    The file holds a list of samples, each ``{"refs": [file_key, ...], "targets": [file_key, ...]}``.
    This is how views are pinned before the table's rows are known, and it survives a rebuild of the
    table. Every sample must have ``n_ref`` references: a run uses one number of references for all
    its view sets.
    """
    samples = json.loads(Path(path).read_text())
    if not isinstance(samples, list) or not all(isinstance(s, dict) and {"refs", "targets"} <= set(s)
                                                for s in samples):
        raise ValueError(f'{path}: expected a JSON list of {{"refs": [...], "targets": [...]}} samples')
    view_sets = []
    for i, sample in enumerate(samples):
        if len(sample["refs"]) != n_ref:
            raise ValueError(f"{path}: sample {i} has {len(sample['refs'])} references, the run uses {n_ref}")
        view_sets.append(tuple(rows_of_file_keys(table_keys, [*sample["refs"], *sample["targets"]])))
    return tuple(view_sets)


def check_view_sets(view_sets: Sequence[Sequence[int]], n_ref: int, n_views: int | None = None) -> None:
    """Raise ``ValueError`` unless every view set has ``n_ref`` references, a target, and no view twice.

    With ``n_views`` (the size of the view table) the rows must also exist.
    """
    for i, rows in enumerate(view_sets):
        if len(rows) <= n_ref:
            raise ValueError(f"view set {i} has {len(rows)} views: it needs {n_ref} references and at least one target")
        if len(set(rows)) != len(rows):
            raise ValueError(f"view set {i} names a view twice: {list(rows)}")
        if min(rows) < 0 or (n_views is not None and max(rows) >= n_views):
            raise ValueError(f"view set {i} has rows that are not in the view table: {list(rows)}")


def scene_view_sets(scene: MVGenScene, n: int) -> list[np.ndarray]:
    """``n`` view sets drawn as training draws them, but with a fixed seed and in an epoch training never reaches.

    They come from the views the scene trains on (neither held out nor unusable), and are the same in
    every run with the same sampling settings and hold-out, whatever its training seed. Each has the
    config's ``n_ref`` references, also when training draws that number per sample (``n_ref_choices``).
    Their anchors are drawn without the camera-balance weights (``ViewSamplingConfig.balance_cameras``),
    so a run with balanced anchors is measured on the view sets of the same run without them.
    """
    n_ref = scene.config.n_ref
    return [scene.indices(i, epoch=PROBE_EPOCH, seed=PROBE_SAMPLE_SEED, n_ref=n_ref, balanced=False)
            for i in range(n)]


def heldout_view_sets(scene: MVGenScene, n: int) -> list[np.ndarray]:
    """Up to ``n`` view sets whose targets are all held-out views and whose references are training views.

    Drawn with the scene's sampler and a fixed seed. Fewer than ``n`` (or none) come back when too few
    held-out views share a date and enough 3D points to form a target group.
    """
    target_ok = scene.usable & scene.heldout
    anchors = eligible_anchors(scene.dates, scene.covis, scene.config, target_ok)
    rng = np.random.default_rng((PROBE_SAMPLE_SEED, PROBE_EPOCH))
    view_sets: list[np.ndarray] = []
    for _ in range(_HELDOUT_ATTEMPTS * n if len(anchors) else 0):
        if len(view_sets) == n:
            break
        rows = sample_rows(scene.dates, scene.covis, scene.config, rng, target_ok, scene.ref_ok, anchors)
        if rows is not None:
            view_sets.append(rows)
    return view_sets


def describe_views(scene: MVGenScene, rows: Sequence[int], n_ref: int) -> dict:
    """The views of a set for a run's records: rows, file keys, dates and periods, and what a reader should be
    warned of.

    ``periods``: each view's period index by the scene's ``period_edges`` (0 = the oldest);
    ``not_usable``: the views the dataset marks as not usable, with the reason where the table gives
    one; ``references_without_reliable_depth``: the references that ``depth_stats.parquet`` does not
    mark ``ok`` (their depth map, which warps them into the targets, disagrees with COLMAP's points,
    or the folder has no such table yet).
    """
    rows = [int(row) for row in rows]
    views = scene.views.iloc[rows]
    keys = views["file_key"].tolist()
    excluded = views["excluded"].tolist() if "excluded" in views else ["" if ok else "not usable"
                                                                        for ok in scene.usable[rows]]
    return {"rows": rows, "n_ref": n_ref, "file_keys": keys,
            "years": [round(float(year), 4) for year in scene.dates[rows]],
            "periods": [int(period) for period in scene.period[rows]],
            "not_usable": {key: why for key, why in zip(keys, excluded) if why},
            "references_without_reliable_depth": [key for key, ok in zip(keys[:n_ref], scene.depth_ok[rows[:n_ref]])
                                                  if not ok]}


# ----------------------------------------------------------------------------
# Wrong dates
# ----------------------------------------------------------------------------
def representative_years(dates: np.ndarray, edges: Sequence[float], preferred: np.ndarray | None = None
                         ) -> tuple[float, ...]:
    """One year per period (``len(edges) + 1`` of them), to stand for the period when a probe moves a date there.

    The median of the ``preferred`` dates that lie in the period; without any, the median of ``dates``
    in it; NaN if the period has no view at all. A run pinned to a few view sets passes its targets'
    dates as ``preferred``, so that a swapped date is one the model has trained on.
    """
    pools = (np.asarray([] if preferred is None else preferred, dtype=np.float64), np.asarray(dates, dtype=np.float64))
    years = []
    for period in range(len(edges) + 1):
        inside = [pool[era_index(pool, tuple(edges)) == period] for pool in pools]
        chosen = next((pool for pool in inside if len(pool)), None)
        years.append(float("nan") if chosen is None else float(np.median(chosen)))
    return tuple(years)


def swapped_years(dates: np.ndarray, n_ref: int, edges: Sequence[float], years: Sequence[float]) -> np.ndarray:
    """The views' dates with every target moved to another period; the references keep theirs.

    A target goes across the period edge nearest to its date, to the representative year
    (``years``, one per period: :func:`representative_years`) of the period on the other side. With
    Notre-Dame's edges (1970, 2019.2877) and years (1890, 2015, 2021): 2021 -> 2015, 2015 -> 2021,
    1890 -> 2015, 1975 -> 1890.

    Args:
        dates: ``[F]`` decimal years (float64, as the view table has them: a float32 date can land on
            the wrong side of an edge).
        n_ref: the first ``n_ref`` views are references.

    Raises ``ValueError`` without an edge, or when the period a target would move to has no
    representative year (NaN).
    """
    dates, edges = np.asarray(dates, dtype=np.float64), np.asarray(edges, dtype=np.float64)
    if len(edges) == 0:
        raise ValueError("dates are swapped across a period edge, and there is none (period_edges is empty)")
    if len(years) != len(edges) + 1:
        raise ValueError(f"{len(edges) + 1} periods need as many representative years, got {len(years)}")
    nearest = np.abs(dates[:, None] - edges[None, :]).argmin(axis=1)
    other_side = np.where(dates < edges[nearest], nearest + 1, nearest)     # edge k separates periods k and k + 1
    swapped = dates.copy()
    swapped[n_ref:] = np.asarray(years, dtype=np.float64)[other_side[n_ref:]]
    if np.isnan(swapped[n_ref:]).any():
        empty = sorted(set(other_side[n_ref:][np.isnan(swapped[n_ref:])].tolist()))
        raise ValueError(f"no view stands for period(s) {empty}: give their years explicitly (swap_years)")
    return swapped


# ----------------------------------------------------------------------------
# Date loss
# ----------------------------------------------------------------------------
def date_loss_summary(losses: dict[str, np.ndarray], baseline: np.ndarray, timesteps: Sequence[int]) -> dict:
    """What :class:`DateLossProbe` reports, from the losses ``[samples, timesteps]`` of each date variant.

    * ``loss_true`` / ``loss_swapped`` / ``loss_unknown``: means over samples and timesteps;
    * ``rel_gap_swapped`` / ``rel_gap_unknown``: mean of ``loss_variant / loss_true - 1`` (above zero:
      the true date gives the lower loss);
    * ``change``: mean of ``loss_true / baseline - 1``, the change since the first evaluation;
    * ``sample_*``: the same three ratios per sample (mean over its timesteps);
    * ``*_by_timestep``: the true-date loss and the three ratios per timestep (mean over samples).
    """
    true = losses["true"]
    ratios = {"rel_gap_swapped": losses["swapped"] / true - 1, "rel_gap_unknown": losses["unknown"] / true - 1,
              "change": true / baseline - 1}
    summary = {"samples": int(true.shape[0]), "timesteps": [int(t) for t in timesteps]}
    summary.update({f"loss_{variant}": float(losses[variant].mean()) for variant in DATE_VARIANTS})
    summary.update({name: float(ratio.mean()) for name, ratio in ratios.items()})
    summary.update({f"sample_{name}": ratio.mean(axis=1).tolist() for name, ratio in ratios.items()})
    summary["loss_true_by_timestep"] = true.mean(axis=0).tolist()
    summary.update({f"{name}_by_timestep": ratio.mean(axis=0).tolist() for name, ratio in ratios.items()})
    return summary


def region_loss_summary(losses: dict[str, dict[str, np.ndarray]]) -> dict:
    """What :class:`DateLossProbe` reports about the changed region: per area of ``REGION_AREAS`` a dict, from
    the area's losses ``[samples, timesteps]`` of each date variant (NaN for a view set without the area).

    * ``sets``: how many view sets have the area (``inside``: at least one target shows the region);
    * ``loss_true``: the mean true-date loss in the area, over those sets and the timesteps;
    * ``rel_gap_swapped`` / ``rel_gap_unknown``: the mean of ``loss_variant / loss_true - 1`` over the same
      (above zero: the true date gives the lower loss in that area);
    * ``sample_rel_gap_*``: the two gaps per view set (None for a set without the area);
    * ``rel_gap_*_by_timestep``: the two gaps per timestep (mean over the sets that have the area).

    The numbers of an area that no view set has are None, and its lists hold only None.
    """
    summary = {}
    for area, by_variant in losses.items():
        true = by_variant["true"]
        has_area = ~np.isnan(true).any(axis=1)
        n_sets = int(has_area.sum())
        of_area: dict = {"sets": n_sets, "loss_true": float(true[has_area].mean()) if n_sets else None}
        for variant in DATE_VARIANTS[1:]:
            gap = by_variant[variant] / true - 1                    # NaN in the rows of the sets without the area
            of_area[f"rel_gap_{variant}"] = float(gap[has_area].mean()) if n_sets else None
            of_area[f"sample_rel_gap_{variant}"] = [float(value) if has else None
                                                    for value, has in zip(gap.mean(axis=1), has_area)]
            of_area[f"rel_gap_{variant}_by_timestep"] = (gap[has_area].mean(axis=0).tolist() if n_sets
                                                         else [None] * true.shape[1])
        summary[area] = of_area
    return summary


def region_gap_line(region: dict, samples: int) -> str:
    """One line for the log on a :func:`region_loss_summary` of ``samples`` view sets: what a wrong and an
    unknown date cost inside the changed region, and outside it."""
    def gaps(of_area: dict) -> str:
        if of_area["sets"] == 0:
            return "no view set has it"
        return f"swapped dates {of_area['rel_gap_swapped']:+.2%}, unknown dates {of_area['rel_gap_unknown']:+.2%}"

    inside, outside = region["inside"], region["outside"]
    return (f"inside the changed region ({inside['sets']} of {samples} view sets show it): {gaps(inside)}; "
            f"outside: {gaps(outside)}")


class DateLossProbe:
    """The targets' loss with true, swapped and unknown dates on fixed view sets, timesteps and noise.

    Every view set is measured at every timestep, with noise that depends only on ``seed``, the set's
    position and the timestep's position. The conditioning is the one of a training step without
    dropout (and with ``loss_cfg``'s rule for cross-date targets, applied to the true dates in all
    three variants: only the date given to the model differs between them). The latents are the mode
    of the VAE's posterior. With ``date_views="targets"`` the references' dates are hidden, as in
    training.

    The first evaluation's true-date losses are written to ``baseline_path`` and every later
    evaluation, also after a resume, is compared with them. The file also records everything that
    decides those losses (:meth:`identity`), and a baseline measured with anything else is refused.

    On a scene with changed-region weights (``MVGenScene.has_regions``) the three losses are also
    measured inside and outside the changed region, from the same forward passes
    (``step.v_prediction_loss_and_map``), and the result has one more entry, ``region``
    (:func:`region_loss_summary`). Everything else is as on a scene without them: the same losses to
    the bit, the same files, the same baseline. The probe always measures the plain loss, also for
    a run that trains with a region weight.

    Args:
        scene: where the view sets are loaded from, with its scene-scale rule; they are kept in memory
            (about 35 MB each at 512). Its ``config.period_edges`` are the periods of the swapped
            dates and of the cross-date rule.
        view_sets: row lists, each with its ``n_ref`` references first.
        swap_years: one representative year per period of ``scene.config.period_edges``.
    """

    def __init__(self, scene: MVGenScene, view_sets: Sequence[Sequence[int]], n_ref: int, loss_cfg: LossConfig,
                 swap_years: Sequence[float], baseline_path: str | Path,
                 timesteps: Sequence[int] = DEFAULT_TIMESTEPS, seed: int = 1234):
        if len(view_sets) == 0 or len(timesteps) == 0:
            raise ValueError("the date-loss probe needs at least one view set and one timestep")
        check_view_sets(view_sets, n_ref, len(scene.views))
        self.n_ref, self.loss_cfg, self.seed = n_ref, loss_cfg, seed
        self.timesteps = tuple(int(t) for t in timesteps)
        self.baseline_path = Path(baseline_path)
        self.period_edges = tuple(float(edge) for edge in scene.config.period_edges)
        self.rows = [[int(row) for row in rows] for rows in view_sets]
        self.samples = [scene.make_sample(np.asarray(rows), n_ref) for rows in self.rows]
        self.has_regions = scene.has_regions                    # the samples then carry changed-region weights
        self.swapped = [torch.from_numpy(swapped_years(scene.dates[rows], n_ref, self.period_edges,
                                                       swap_years)).float() for rows in self.rows]
        self.last_losses: dict[str, list[list[float]]] = {}     # the latest evaluation's raw losses, per variant

    def describe(self) -> list[dict]:
        """Per view set: its rows, file keys, dates, scene-scale factor and the dates its targets are swapped to."""
        return [{"rows": rows, "file_keys": sample["file_keys"],
                 "years": [round(float(year), 4) for year in sample["date_mid"]],
                 "scale_factor": float(sample["scale"]),
                 "swapped_years": [round(float(year), 4) for year in swapped[self.n_ref:]]}
                for rows, sample, swapped in zip(self.rows, self.samples, self.swapped)]

    def identity(self) -> dict:
        """Everything that decides the true-date losses, as JSON types: two probes with the same identity
        measure the same thing on the same model.

        The view sets, the timesteps and the noise seed; the number of references; the loss settings that
        change the loss (precision, domain label, which views show their date, the cross-date rule with
        its gap and the period edges); and the scene-scale factor of each view set.
        """
        cfg = self.loss_cfg
        return {"rows": self.rows, "timesteps": list(self.timesteps), "seed": self.seed, "n_ref": self.n_ref,
                "precision": cfg.precision, "class_label": cfg.class_label, "date_views": cfg.date_views,
                "cross_date_pixels": cfg.cross_date_pixels, "cross_date_gap": cfg.cross_date_gap,
                "period_edges": list(self.period_edges),
                "scale_factors": [float(sample["scale"]) for sample in self.samples]}

    def _baseline(self, true: np.ndarray) -> np.ndarray:
        """The first evaluation's true-date losses: read from the run folder, or written there now."""
        identity = self.identity()
        if self.baseline_path.exists():
            stored = json.loads(self.baseline_path.read_text())
            differing = [key for key, value in identity.items() if stored.get(key) != value]
            if differing:
                raise ValueError(f"{self.baseline_path} holds a first evaluation made with other settings "
                                 f"({', '.join(differing)}), so this run cannot be compared with it: use a new "
                                 "run folder, or delete the file to start a new baseline")
            return np.asarray(stored["loss_true"], dtype=np.float64)
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.baseline_path.write_text(json.dumps({**identity, "loss_true": true.tolist()}))
        return true

    @torch.no_grad()
    def __call__(self, model: MVGen) -> dict:
        """Evaluate ``model``; gradient-free.

        Returns :func:`date_loss_summary` of the losses, plus ``date_rms`` and ``date_temb_ratio``: the
        size of the date embedding in the true-date passes, averaged over them
        (``step.date_embedding_size``; None for a model without a date conditioner). On a scene with
        changed-region weights also ``region``: :func:`region_loss_summary` of the losses inside and
        outside the changed region.
        """
        cfg, n_ref = self.loss_cfg, self.n_ref
        if max(self.timesteps) >= model.scheduler.config.num_train_timesteps or min(self.timesteps) < 0:
            raise ValueError(f"probe timesteps {self.timesteps} are outside the scheduler's training steps")
        losses = {variant: np.zeros((len(self.samples), len(self.timesteps))) for variant in DATE_VARIANTS}
        region_losses = {area: {variant: np.full_like(losses[variant], np.nan) for variant in DATE_VARIANTS}
                         for area in REGION_AREAS}
        sizes = []
        for i, (sample, swapped) in enumerate(zip(self.samples, self.swapped)):
            is_ref = torch.arange(len(swapped)) < n_ref
            shown = shown_dates(torch.ones_like(is_ref), is_ref, cfg.date_views)
            dates = {"true": (sample["date_mid"], shown), "swapped": (swapped, shown),
                     "unknown": (sample["date_mid"], torch.zeros_like(shown))}
            latents = encode_images(model, sample["images"], sample_posterior=False)
            prior_views, pixel_views = cross_date_drops(sample["date_mid"], n_ref, cfg, self.period_edges)
            inputs = build_inputs(model, sample, latents, dropped_prior_views=prior_views,
                                  dropped_pixel_views=pixel_views)
            generator = torch.Generator().manual_seed(self.seed + i)
            for j, timestep in enumerate(self.timesteps):
                noise = torch.randn(latents.shape, generator=generator)
                for variant, (year, valid) in dates.items():
                    if not self.has_regions:
                        loss, _ = v_prediction_loss(model, latents, inputs, timestep, noise, year, valid, cfg, n_ref)
                    else:               # the same loss, and where it lies, from the one forward pass
                        loss, error = v_prediction_loss_and_map(model, latents, inputs, timestep, noise, year, valid,
                                                                cfg, n_ref)
                        for area, mean in zip(REGION_AREAS, region_means(error, target_region(sample, n_ref, error))):
                            if mean is not None:
                                region_losses[area][variant][i, j] = float(mean)
                    losses[variant][i, j] = float(loss)
                    if variant == "true":
                        sizes.append(date_embedding_size(model))
        self.last_losses = {variant: values.tolist() for variant, values in losses.items()}
        summary = date_loss_summary(losses, self._baseline(losses["true"]), self.timesteps)
        for name in ("date_rms", "date_temb_ratio"):
            measured = [size[name] for size in sizes if size[name] is not None]
            summary[name] = float(np.mean(measured)) if measured else None
        if self.has_regions:
            summary["region"] = region_loss_summary(region_losses)
        return summary


# ----------------------------------------------------------------------------
# Renders
# ----------------------------------------------------------------------------
def parse_render_date(text: str | float) -> str | float:
    """A render date from the command line or a config: one of ``RENDER_DATE_WORDS``, or a decimal year."""
    if isinstance(text, str) and text in RENDER_DATE_WORDS:
        return text
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ValueError(f"a render date is a year or one of {RENDER_DATE_WORDS}, not {text!r}") from None


def date_label(date: str | float) -> str:
    """How a render date is written in file contents and on contact sheets (a year: to about an hour)."""
    return date if isinstance(date, str) else f"{date:.8g}"


def to_uint8(images: torch.Tensor) -> np.ndarray:
    """Images ``[F, 3, H, W]`` in [-1, 1] as uint8 ``[F, H, W, 3]``."""
    return ((images.permute(0, 2, 3, 1).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two uint8 images (as ``scripts/mvgen/nvs_test.py`` computes it)."""
    mse = float((((a.astype(np.float32) - b.astype(np.float32)) / 255) ** 2).mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


@contextmanager
def view_priors_dropped(unet: nn.Module, n_ref: int, prior_views: torch.Tensor | None = None,
                        pixel_views: torch.Tensor | None = None) -> Iterator[None]:
    """Inside the block, the UNet gets no 3D priors for the views marked in ``prior_views`` and no warped-pixel
    prior for those marked in ``pixel_views`` (bool ``[F]`` each, or None).

    The sampling pipeline computes the 3D priors itself, so this is how a run's ``cross_date_pixels``
    rule is applied while sampling: a forward pre-hook zeroes the marked views' rows at every UNet
    call, in each copy of the views of a guidance-doubled batch. It does to the UNet's input exactly
    what ``step.build_inputs`` does with its masks of the same names: ``prior_views`` lose every
    prior tensor, ``pixel_views`` only the last one (the warped pixels), and the first ``n_ref``
    entries (the references) are ignored. The masks come from ``step.cross_date_drops``, in training
    and here.
    """
    def zeroed(prior: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        if prior.shape[0] % len(views):
            raise ValueError(f"a prior has {prior.shape[0]} rows, which {len(views)} views cannot fill")
        keep = ~views.to(prior.device)
        keep[:n_ref] = True
        return prior * keep.to(prior.dtype).repeat(prior.shape[0] // len(views)).view(-1, 1, 1, 1)

    def drop(module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        coords = kwargs.get("coords")
        if coords is None:
            raise ValueError("the UNet was not given any 3D priors: there is nothing to drop")
        if pixel_views is not None and not isinstance(coords, list):
            raise ValueError("the UNet was not given a list of priors: there is no warped-pixel prior to drop")
        priors = list(coords) if isinstance(coords, list) else [coords]             # the warped pixels come last
        if prior_views is not None:
            priors = [zeroed(prior, prior_views) for prior in priors]
        if pixel_views is not None:
            priors[-1] = zeroed(priors[-1], pixel_views)
        return args, {**kwargs, "coords": priors if isinstance(coords, list) else priors[0]}

    if prior_views is None and pixel_views is None:
        yield
        return
    handle = unet.register_forward_pre_hook(drop, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def unet_precision(unet: nn.Module, precision: str) -> Iterator[None]:
    """Inside the block, the UNet's forward passes run under ``precision``'s autocast and return fp32
    (when they return a tuple, as the sampling pipeline asks them to).

    This is how the sampling pipeline gets the numerics of a training step (``step.predict``): bf16
    or fp32 inside the UNet, fp32 everywhere else. Autocast must not be opened around the whole
    pipeline: the pipeline also warps the references into the targets (the 3D priors), with matrix
    products that autocast would round to bf16. Pixel coordinates up to 512 do not survive 8 bits:
    measured on a real view set (CPU), the warped pixels then cover 24% of the targets instead of
    70% and half of the validity mask flips (under fp16 autocast, 6% of it).
    """
    device_type = next(unet.parameters()).device.type
    open_contexts = []

    def enter(module: nn.Module, args: tuple) -> None:
        open_contexts.append(autocast(precision, device_type))
        open_contexts[-1].__enter__()

    def leave(module: nn.Module, args: tuple, output: object) -> tuple | None:
        open_contexts.pop().__exit__(None, None, None)
        if not isinstance(output, tuple):               # also None: the forward pass failed
            return None
        return tuple(part.float() if torch.is_tensor(part) else part for part in output)

    handles = [unet.register_forward_pre_hook(enter), unet.register_forward_hook(leave, always_call=True)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def render_views(model: MVGen, pipeline, sample: dict, year: torch.Tensor, valid: torch.Tensor, *, seed: int,
                 steps: int = 50, guidance: float = 2.0, class_label: int = 0, precision: str = "bf16",
                 uncond_dates: str = "same", dropped_prior_views: torch.Tensor | None = None,
                 dropped_pixel_views: torch.Tensor | None = None) -> np.ndarray:
    """A view set as MVGenMaster's sampling pipeline generates it at the given dates: uint8 ``[F, H, W, 3]``.

    The pipeline is called as ``scripts/mvgen/nvs_test.py`` calls it, except that the UNet is given
    what a training step gives it (``step.build_inputs``). The weights are fp32, the UNet runs under
    ``precision``'s autocast, and the VAE, the camera rays, the 3D priors and the sampler's arithmetic
    are fp32 (:func:`unet_precision`): no autocast context is opened around the pipeline, and none
    may be. Far depth is contracted once for every prior (``step.depth_for_priors``). The pipeline
    is given the references' images and depth; the targets' images are replaced by zeros, so nothing
    of the real targets can leak into the result.

    Args:
        pipeline: ``model.sampling_pipeline()``.
        sample: a view set (section 2.1 of the implementation spec); not changed.
        year, valid: ``[F]`` the dates to condition on and which of them are known.
        seed: of the sampler's initial noise (a CPU generator, as in ``nvs_test.py``).
        uncond_dates: what the unconditional half of the guidance batch sees: the ``"same"`` dates, or
            all ``"unknown"`` (guidance then also amplifies the effect of the date).
        dropped_prior_views, dropped_pixel_views: bool ``[F]``, targets that get no 3D priors at all, or no
            warped-pixel prior (:func:`view_priors_dropped`; as the arguments of ``step.build_inputs``).
    """
    n_views, n_ref = sample["images"].shape[0], int(sample["n_ref"])
    height, width = sample["images"].shape[-2:]
    device = model.device
    images = sample["images"].clone()
    images[n_ref:] = 0
    config = copy.deepcopy(model.config)
    config.nframe = n_views
    depth = depth_for_priors(sample["depth"], device)
    drop = view_priors_dropped(model.unet, n_ref, dropped_prior_views, dropped_pixel_views)
    with drop, model.dates(year, valid, uncond_dates), unet_precision(model.unet, precision):
        frames = pipeline(images=images.to(device), nframe=n_views, cond_num=n_ref, key_rescale=None, height=height,
                          width=width, intrinsics=sample["K"].to(device), extrinsics=sample["w2c"].to(device),
                          num_inference_steps=steps, guidance_scale=guidance, output_type="np", config=config,
                          tag=["custom"] * n_views, class_label=class_label, depth=depth, vae=model.vae,
                          generator=torch.Generator().manual_seed(seed)).images
    return (np.asarray(frames) * 255).astype(np.uint8)


def contact_sheet(title: str, real: np.ndarray, captions: Sequence[str], rows: Sequence[tuple[str, np.ndarray]],
                  n_ref: int, thumb: int) -> Image.Image:
    """Row 0: the real views under their captions (references green, targets red). Then one row per render:
    its label on the left and its targets below the real ones (``rows``: label, uint8 ``[F, H, W, 3]``)."""
    n_views = len(real)
    top = _HEADER + _CAPTION
    sheet = Image.new("RGB", (_LABEL_WIDTH + n_views * thumb, top + (1 + len(rows)) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 3), title, fill="black")
    draw.text((4, top + thumb // 2 - 6), "real photos", fill="black")
    for j in range(n_views):
        left = _LABEL_WIDTH + j * thumb
        draw.text((left + 3, _HEADER), captions[j], fill=(0, 130, 0) if j < n_ref else (190, 0, 0))
        sheet.paste(Image.fromarray(real[j]).resize((thumb, thumb)), (left, top))
    for r, (label, frames) in enumerate(rows, start=1):
        draw.multiline_text((4, top + r * thumb + thumb // 2 - 12), label, fill="black")
        for j in range(n_ref, n_views):
            sheet.paste(Image.fromarray(frames[j]).resize((thumb, thumb)), (_LABEL_WIDTH + j * thumb, top + r * thumb))
    return sheet


class RenderProbe:
    """One view set generated at several dates and with several noise seeds.

    A date is a year (every target is rendered at it), ``"true"`` (each target at its own date),
    ``"swapped"`` (each target at its date in another period, :func:`swapped_years`) or ``"unknown"``
    (no view has a date, as under date dropout). The references always keep their own dates, unless
    ``loss_cfg.date_views`` is ``"targets"`` and they are hidden as in training. With
    ``loss_cfg.cross_date_pixels`` set to ``"drop"`` or ``"drop_all"`` a target loses its warped
    pixels, or all its priors, when it is a cross-date target at the date it is rendered at
    (``step.cross_date_views`` with the scene's ``period_edges``; for ``"unknown"``: at its true
    date, as in a training step with date dropout).

    Calling the probe writes ``<out_dir>/<name>.jpg`` (the contact sheet) and ``<name>.json`` and
    returns the JSON's content: per date and seed the PSNR against the real target photos (mean and
    per target) and ``diff_vs_first``, the mean absolute difference (0-1) of the targets from the
    render at the first date with the same seed. While the date embedding has no effect it is 0,
    with two reservations: the cross-date rule can give the two dates different priors, which changes
    the picture at step 0 already; and on a GPU the prior warp is not bit-reproducible (it scatters
    with duplicate indices), so there "no effect" is the difference between two renders of one date.

    Args:
        scene: where the view set is loaded from, with its scene-scale rule and ``period_edges``.
        rows: the view set's rows, its ``n_ref`` references first.
        swap_years: one representative year per period; only needed for the date ``"swapped"``.
        loss_cfg: the run's :class:`LossConfig` (dates shown, domain label, cross-date rule, precision).
    """

    def __init__(self, scene: MVGenScene, rows: Sequence[int], n_ref: int, name: str,
                 dates: Sequence[str | float] = ("true", "swapped"), seeds: Sequence[int] = (0, 1),
                 loss_cfg: LossConfig = LossConfig(), swap_years: Sequence[float] | None = None, steps: int = 50,
                 guidance: float = 2.0, uncond_dates: str = "same"):
        check_view_sets([rows], n_ref, len(scene.views))
        self.name, self.n_ref, self.loss_cfg = name, n_ref, loss_cfg
        self.dates = tuple(dict.fromkeys(parse_render_date(date) for date in dates))      # unique, in order
        self.seeds = tuple(int(seed) for seed in seeds)
        if not self.dates or not self.seeds:
            raise ValueError("a render probe needs at least one date and one seed")
        self.steps, self.guidance, self.uncond_dates = steps, guidance, uncond_dates
        self.period_edges = tuple(float(edge) for edge in scene.config.period_edges)
        self.scale = scene.scale
        self.rows = [int(row) for row in rows]
        self.sample = scene.make_sample(np.asarray(self.rows), n_ref)
        self.swapped = None
        if "swapped" in self.dates:
            if swap_years is None:
                raise ValueError('the render date "swapped" needs swap_years')
            self.swapped = torch.from_numpy(swapped_years(scene.dates[self.rows], n_ref, self.period_edges,
                                                          swap_years)).float()

    def describe(self) -> dict:
        """The probe's views and settings, for the run's records (``scale``: the scene-scale rule, and the
        factor it gave this view set)."""
        described = {"name": self.name, "rows": self.rows, "n_ref": self.n_ref, "file_keys": self.sample["file_keys"],
                     "years": [round(float(year), 4) for year in self.sample["date_mid"]],
                     "scale": self.scale, "scale_factor": float(self.sample["scale"]),
                     "dates": [date_label(date) for date in self.dates], "seeds": list(self.seeds),
                     "steps": self.steps, "guidance": self.guidance, "uncond_dates": self.uncond_dates}
        if self.swapped is not None:
            described["swapped_years"] = [round(float(year), 4) for year in self.swapped[self.n_ref:]]
        return described

    def view_dates(self, date: str | float) -> tuple[torch.Tensor, torch.Tensor]:
        """``(year, valid)``, both ``[F]``, for one of the probe's dates."""
        year = self.sample["date_mid"].clone()
        if date == "swapped":
            year = self.swapped.clone()
        elif not isinstance(date, str):
            year[self.n_ref:] = date
        is_ref = torch.arange(len(year)) < self.n_ref
        valid = shown_dates(torch.full_like(is_ref, date != "unknown"), is_ref, self.loss_cfg.date_views)
        return year, valid

    @torch.no_grad()
    def __call__(self, model: MVGen, out_dir: str | Path, thumb: int = 256) -> dict:
        """Render every date and seed with ``model``; gradient-free. ``thumb``: side of the sheet's pictures."""
        cfg, n_ref = self.loss_cfg, self.n_ref
        pipeline = model.sampling_pipeline()
        pipeline.set_progress_bar_config(disable=True)
        real = to_uint8(self.sample["images"])
        renders, records, sheet_rows = {}, [], []
        for date in self.dates:
            year, valid = self.view_dates(date)
            prior_views, pixel_views = cross_date_drops(year, n_ref, cfg, self.period_edges)
            for seed in self.seeds:
                frames = render_views(model, pipeline, self.sample, year, valid, seed=seed, steps=self.steps,
                                      guidance=self.guidance, class_label=cfg.class_label, precision=cfg.precision,
                                      uncond_dates=self.uncond_dates, dropped_prior_views=prior_views,
                                      dropped_pixel_views=pixel_views)
                renders[date, seed] = frames
                per_target = [psnr(frames[j], real[j]) for j in range(n_ref, len(real))]
                first = renders[self.dates[0], seed]
                difference = np.abs(frames[n_ref:].astype(np.float32) - first[n_ref:].astype(np.float32)).mean() / 255
                records.append({"date": date_label(date), "seed": seed, "psnr": float(np.mean(per_target)),
                                "psnr_per_target": per_target, "diff_vs_first": float(difference)})
                sheet_rows.append((f"date {date_label(date)}, seed {seed}\n{records[-1]['psnr']:.1f} dB", frames))
        by_date = {date_label(date): float(np.mean([r["psnr"] for r in records if r["date"] == date_label(date)]))
                   for date in self.dates}
        result = {**self.describe(), "renders": records, "psnr_by_date": by_date}
        captions = [f"{'ref' if j < n_ref else 'target'} {float(year):.2f}"
                    for j, year in enumerate(self.sample["date_mid"])]
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title = f"{self.name}: {n_ref} references, {len(real) - n_ref} targets, guidance {self.guidance:g}"
        contact_sheet(title, real, captions, sheet_rows, n_ref, thumb).save(out_dir / f"{self.name}.jpg", quality=90)
        (out_dir / f"{self.name}.json").write_text(json.dumps(result, indent=1))
        return result
