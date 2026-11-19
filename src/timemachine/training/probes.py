"""Fixed evaluation probes for date-conditioned SEVA.

Two probes, both run on the *same* fixed inputs at every evaluation, so
numbers and images are comparable across training steps:

**Date-swap loss gap** (:class:`DateGapProbe`, cheap). A fixed set of training
samples, each with a fixed noise level and noise. We compute the denoising loss
with the true dates, with the targets' dates swapped to another era, and with
all dates marked unknown. If the model uses dates, the relative gap
``loss_swapped / loss_true - 1`` grows above zero. At step 0 (zero-init date module) all three are equal.
It also reports ``x0_mse_high_noise``, the denoised-latent error at high noise levels: the
noise-prediction loss barely moves when the global look of the renders drifts, this does.

**Renders at many dates** (:class:`RenderProbe`, slow). One fixed set of input
views and target cameras, chosen by a :class:`RenderProbeSpec` (e.g. "targets
from after the 2019 fire, inputs from before it"). The targets are generated at
several years with the same seed, so differences between rows are caused by
the date alone. Saved as a contact sheet, with colour statistics per year and
PSNR against the real targets.

**Held-out vs trained views** (:class:`HeldoutProbe`, cheap; only when the run holds
views out, see :mod:`timemachine.data.holdout`). The denoising loss on views the model
never trained on, next to the loss on trained views of the same eras, each relative to
the first evaluation. Memorising the scene's photos shows as the trained views' loss
falling while the held-out views' does not.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.holdout import era_index
from timemachine.data.images import color_stats
from timemachine.data.sampling import tolerance
from timemachine.model.inference import SEVA_FRAMES, generate_targets
from timemachine.model.loading import SevaComponents
from timemachine.training.conditioning import sample_conditions
from timemachine.training.loss import (  # noqa: F401  (swap constants re-exported for older scripts)
    MODERN_SWAP, NUM_LEVELS, OLD_SWAP, SWAP_BOUNDARY, TRAIN_AUTOCAST_DTYPE, autocast, noise_prediction_loss,
    predict_noise, shown_dates, swapped_years,
)

PROBE_SAMPLE_SEED = 0    # dataset sampling seed for probe samples, independent of the training seed


# ----------------------------------------------------------------------------
# Date-swap loss gap
# ----------------------------------------------------------------------------
class DateGapProbe:
    """Loss with true / swapped / unknown dates on fixed samples, noise levels and noise."""

    def __init__(self, components: SevaComponents, dataset: DatedViewsDataset, n_samples: int = 16,
                 seed: int = 1234, device: str = "cuda", autocast_dtype: torch.dtype | None = TRAIN_AUTOCAST_DTYPE,
                 date_views: str = "all"):
        self.autocast_dtype = autocast_dtype            # match the training run's precision
        self.date_views = date_views                    # ... and which views see their date
        gen = torch.Generator().manual_seed(seed)
        # Evenly spread noise levels (skip the extremes, where dates cannot matter much).
        levels = np.linspace(100, NUM_LEVELS - 100, n_samples).round().astype(int)
        self.items = []
        for i in range(n_samples):
            # An epoch training never uses, and a fixed sampling seed: runs with different training
            # seeds see the same probe samples (seed-0 runs: the same samples as before 2026-11-18).
            rows = dataset.indices(i, epoch=10_000 + seed, seed=PROBE_SAMPLE_SEED)
            sample = dataset.make_sample(rows, dataset.config.n_inputs)
            latents, cond, _ = sample_conditions(components, sample, device, camera_scale=dataset.camera_scale)
            self.items.append({
                "latents": latents, "cond": cond, "level": int(levels[i]),
                "noise": torch.randn(latents.shape, generator=gen).to(device),
                "year": sample["date_mid"].to(device), "is_input": sample["is_input"].to(device),
                "old_targets": bool(sample["date_mid"][~sample["is_input"]].max() < SWAP_BOUNDARY),
            })

    @torch.no_grad()
    def __call__(self, model, sigmas: torch.Tensor) -> dict:
        rows = []
        for it in self.items:
            ones = shown_dates(torch.ones_like(it["is_input"]), it["is_input"], self.date_views)
            with autocast(self.autocast_dtype):
                def loss(year, valid):
                    return float(noise_prediction_loss(model, sigmas, it["latents"], it["cond"], it["level"],
                                                       it["noise"], year, valid))
                rows.append({"true": loss(it["year"], ones),
                             "swapped": loss(swapped_years(it["year"], it["is_input"]), ones),
                             "unknown": loss(it["year"], torch.zeros_like(ones)),
                             "old_targets": it["old_targets"]})
        true, swapped, unknown = (np.array([r[k] for r in rows]) for k in ("true", "swapped", "unknown"))
        old = np.array([r["old_targets"] for r in rows])
        sigma = np.array([float(sigmas[it["level"]]) for it in self.items])
        high = sigma > 10
        # Relative gaps (loss ratio - 1) are comparable across noise levels, whose losses differ ~100x.
        rel_swapped, rel_unknown = swapped / true - 1, unknown / true - 1
        out = {"loss_true": true.mean(), "loss_swapped": swapped.mean(), "loss_unknown": unknown.mean(),
               "rel_gap_swapped": rel_swapped.mean(), "rel_gap_unknown": rel_unknown.mean()}
        if high.any():
            # Error of the denoised latent (sigma^2 x noise error) at high noise, where colours and
            # layout are decided. The noise error itself is tiny there and hides drift (docs/findings.md).
            out["x0_mse_high_noise"] = (sigma[high] ** 2 * true[high]).mean()
        for name, mask in (("old_targets", old), ("modern_targets", ~old),
                           ("high_noise", high), ("low_noise", ~high)):
            if mask.any():
                out[f"rel_gap_swapped_{name}"] = rel_swapped[mask].mean()
        return {k: float(v) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Held-out vs trained views
# ----------------------------------------------------------------------------
def measured_view_rows(dataset: DatedViewsDataset, view: int, pool: np.ndarray) -> np.ndarray | None:
    """Rows of a sample that measures one view: inputs, then ``view``, then its same-date siblings.

    The siblings (the other targets) are the ``pool`` views of the same scene/model nearest by
    camera within the sampling date tolerance; the inputs are the nearest remaining ones, of
    any date (as in training samples without the input gap). ``None`` if ``view`` has too few
    siblings.
    """
    cfg = dataset.config
    pool = pool[(pool != view) & (dataset.group[pool] == dataset.group[view])]
    dist = np.linalg.norm(dataset.camera_feats[pool] - dataset.camera_feats[view], axis=1)
    order = pool[np.argsort(dist, kind="stable")]
    same = np.abs(dataset.dates[order] - dataset.dates[view]) <= tolerance(cfg, dataset.dates[view])
    siblings = order[same][: cfg.n_targets - 1]
    if len(siblings) < cfg.n_targets - 1:
        return None
    inputs = order[~np.isin(order, siblings)][: cfg.n_inputs]
    return np.concatenate([inputs, [view], siblings])


def heldout_pairs(dataset: DatedViewsDataset, n_pairs: int, rng: np.random.Generator,
                  era_edges: tuple[float, ...]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Up to ``n_pairs`` (held-out sample rows, trained sample rows, kind), each pair from one era.

    Two kinds of held-out view: ``"view"``, in a scene the model trains on (the sample's other
    views are trained views), and ``"scene"``, in a held-out scene (all its views are unseen,
    so the sample is built from them). Kinds alternate when both exist. Within a kind, old
    views come first, up to half of the kind's share (they are few and matter most), then
    modern ones, then any remaining old ones. Each is paired with a random trained view of
    its era, preferably from the same scene.
    """
    train = dataset.train_rows
    scene_held = dataset.heldout_groups
    held = np.flatnonzero(dataset.heldout)
    held = held[rng.permutation(len(held))]
    is_old = dataset.dates[held] < dataset.config.old_before
    queues = [held[mask] for mask in (~scene_held[dataset.group[held]], scene_held[dataset.group[held]]) if mask.any()]
    cap = n_pairs // (2 * max(len(queues), 1))  # old views first, up to half of each kind's share
    queues = [np.concatenate([q[is_old_q][:cap], q[~is_old_q], q[is_old_q][cap:]])
              for q, is_old_q in ((q, np.isin(q, held[is_old])) for q in queues)]
    order = [q[i] for i in range(max(map(len, queues), default=0)) for q in queues if i < len(q)]
    train_era = era_index(dataset.dates[train], era_edges)
    used, pairs = set(), []
    for h in order:
        if len(pairs) == n_pairs:
            break
        kind = "scene" if scene_held[dataset.group[h]] else "view"
        rows_h = measured_view_rows(dataset, h, dataset.group_rows[dataset.group[h]] if kind == "scene" else train)
        if rows_h is None:
            continue
        era = era_index(dataset.dates[[h]], era_edges)[0]
        same_era = train[train_era == era]
        same_scene = dataset.group[same_era] == dataset.group[h]
        candidates = np.concatenate([rng.permutation(same_era[same_scene]), rng.permutation(same_era[~same_scene])])
        for t in candidates:
            rows_t = None if t in used else measured_view_rows(dataset, t, train)
            if rows_t is not None:
                used.add(t)
                pairs.append((rows_h, rows_t, kind))
                break
    return pairs


class HeldoutProbe:
    """Denoising loss on held-out views vs trained views, relative to the first evaluation.

    Each held-out view is paired with a trained view of the same era (:func:`heldout_pairs`).
    Both are measured in the same kind of sample, at the same noise level and with the same
    noise, and the loss is taken on the measured view alone. Per group it reports the loss
    with true dates, its change since the first evaluation (``*_change``: the mean per-sample
    ratio - 1, so the noise level's scale cancels), how much true dates help over unknown
    ones (``*_rel_unknown`` > 0: they help) and the high-noise denoised-latent error.
    ``memorisation_gap`` = held-out change - trained change: above zero and growing means
    the model improves on the photos it trained on but not on the scene. With held-out
    scenes, the held-out change is also reported per kind (``heldout_change_view`` /
    ``heldout_change_scene``, see :func:`heldout_pairs`).

    ``*_rel_swapped`` (> 0: the true date beats a wrong-era date) is the test of temporal
    learning: the targets' dates are moved to the other era (old -> 2015, modern -> 1890),
    with the same noise; reported per era and kind. A model without dates scores 0.

    The first evaluation's losses are stored at ``baseline_path``, so a resumed run is
    compared with the same reference (normally step 0, i.e. plain SEVA). The mean of
    per-sample ratios weighs every noise level equally, so it is dominated by the high-noise
    samples (tiny noise errors, large relative changes); ``*_change_of_mean`` (ratio of mean
    losses) and ``*_change_median`` are given too, and ``last_per_sample`` keeps the raw losses.
    """

    GROUPS = ("heldout", "train")

    def __init__(self, components: SevaComponents, dataset: DatedViewsDataset, baseline_path: Path,
                 n_pairs: int = 48, seed: int = 4321, device: str = "cuda",
                 autocast_dtype: torch.dtype | None = TRAIN_AUTOCAST_DTYPE, date_views: str = "all",
                 era_edges: tuple[float, ...] = (1900.0, 1970.0, 2000.0)):
        if not dataset.heldout.any():
            raise ValueError("HeldoutProbe needs a dataset with held-out views (DatedViewsDataset.hold_out).")
        self.autocast_dtype, self.date_views = autocast_dtype, date_views
        self.baseline_path = Path(baseline_path)
        rng = np.random.default_rng(seed)
        gen = torch.Generator().manual_seed(seed)
        pairs = heldout_pairs(dataset, n_pairs, rng, era_edges)
        if not pairs:
            raise ValueError("No held-out view has enough same-date trained views to be measured.")
        levels = rng.permutation(np.linspace(100, NUM_LEVELS - 100, len(pairs)).round().astype(int))
        n_in = dataset.config.n_inputs
        self.items = []
        for (rows_h, rows_t, kind), level in zip(pairs, levels):
            item = {"level": int(level), "kind": kind,
                    "old": bool(dataset.dates[rows_h[n_in]] < dataset.config.old_before)}
            for group, rows in zip(self.GROUPS, (rows_h, rows_t)):
                sample = dataset.make_sample(rows, n_in)
                latents, cond, _ = sample_conditions(components, sample, device, camera_scale=dataset.camera_scale)
                item[group] = {"latents": latents, "cond": cond, "year": sample["date_mid"].to(device),
                               "is_input": sample["is_input"].to(device), "file_key": sample["file_keys"][n_in]}
            item["noise"] = torch.randn(item["heldout"]["latents"].shape, generator=gen).to(device)
            self.items.append(item)
        self.measured = n_in                            # index of the measured view in every sample

    def describe(self) -> list[dict]:
        return [{"level": it["level"], "kind": it["kind"], "old": it["old"],
                 **{g: it[g]["file_key"] for g in self.GROUPS}} for it in self.items]

    def _view_loss(self, model, sigmas: torch.Tensor, s: dict, level: int, noise: torch.Tensor,
                   valid: torch.Tensor) -> float:
        latents = s["latents"]
        t = torch.full((latents.shape[0],), level, device=latents.device, dtype=torch.long)
        sigma = sigmas[t].to(latents.device)
        noisy = latents.float() + sigma.view(-1, 1, 1, 1) * noise
        with autocast(self.autocast_dtype):
            eps_hat, _ = predict_noise(model, noisy, sigma, t, s["cond"], s["year"], valid)
        m = self.measured
        return float((eps_hat[m].float() - noise[m]).pow(2).mean())

    def _baseline(self, true: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.baseline_path.exists():
            base = {g: np.array(v) for g, v in json.loads(self.baseline_path.read_text()).items()}
            if any(len(base[g]) != len(true[g]) for g in self.GROUPS):
                raise ValueError(f"{self.baseline_path} does not match this probe's samples.")
            return base
        self.baseline_path.write_text(json.dumps({g: true[g].tolist() for g in self.GROUPS}))
        return true

    @torch.no_grad()
    def __call__(self, model, sigmas: torch.Tensor) -> dict:
        true = {g: [] for g in self.GROUPS}
        unknown = {g: [] for g in self.GROUPS}
        swapped = {g: [] for g in self.GROUPS}
        for it in self.items:
            for g in self.GROUPS:
                s = it[g]
                shown = shown_dates(torch.ones_like(s["is_input"]), s["is_input"], self.date_views)
                true[g].append(self._view_loss(model, sigmas, s, it["level"], it["noise"], shown))
                unknown[g].append(self._view_loss(model, sigmas, s, it["level"], it["noise"], torch.zeros_like(shown)))
                wrong = {**s, "year": swapped_years(s["year"], s["is_input"])}
                swapped[g].append(self._view_loss(model, sigmas, wrong, it["level"], it["noise"], shown))
        true, unknown, swapped = ({g: np.array(v) for g, v in d.items()} for d in (true, unknown, swapped))
        base = self._baseline(true)
        sigma = np.array([float(sigmas[it["level"]]) for it in self.items])
        high, old = sigma > 10, np.array([it["old"] for it in self.items])
        kinds = np.array([it["kind"] for it in self.items])
        self.last_per_sample = {f"{g}_{kind}": v[g].tolist()
                                for kind, v in (("true", true), ("unknown", unknown), ("swapped", swapped))
                                for g in self.GROUPS}
        out = {"pairs": len(self.items), "old_pairs": int(old.sum())}
        for g in self.GROUPS:
            change = true[g] / base[g] - 1
            out[f"{g}_loss"] = true[g].mean()
            out[f"{g}_change"] = change.mean()
            out[f"{g}_change_of_mean"] = true[g].mean() / base[g].mean() - 1
            out[f"{g}_change_median"] = np.median(change)
            out[f"{g}_rel_unknown"] = (unknown[g] / true[g] - 1).mean()
            rel_swapped = swapped[g] / true[g] - 1
            out[f"{g}_rel_swapped"] = rel_swapped.mean()
            for name, mask in (("old", old), ("modern", ~old), ("view", kinds == "view"), ("scene", kinds == "scene")):
                if mask.any() and not mask.all():
                    out[f"{g}_rel_swapped_{name}"] = rel_swapped[mask].mean()
            for name, mask in (("high_noise", high), ("old", old), ("modern", ~old)):
                if mask.any():
                    out[f"{g}_change_{name}"] = change[mask].mean()
            if high.any():
                out[f"{g}_x0_high_noise"] = (sigma[high] ** 2 * true[g][high]).mean()
        out["memorisation_gap"] = out["heldout_change"] - out["train_change"]
        if len(set(kinds)) > 1:
            change = true["heldout"] / base["heldout"] - 1
            for kind in ("view", "scene"):
                out[f"heldout_change_{kind}"] = change[kinds == kind].mean()
                out[f"memorisation_gap_{kind}"] = out[f"heldout_change_{kind}"] - (
                    true["train"][kinds == kind] / base["train"][kinds == kind] - 1).mean()
        return {k: float(v) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Renders at many dates
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderProbeSpec:
    """Which views to render, and at which years. ``None`` in ``render_years`` = all dates unknown."""

    name: str
    target_range: tuple[float, float]           # real targets dated in [lo, hi)
    input_range: tuple[float, float]            # inputs dated in [lo, hi)
    render_years: tuple[float | None, ...]
    n_inputs: int = 3
    n_targets: int = 5
    seed: int = 0
    file_keys: tuple[str, ...] = field(default=())   # pin exact views (inputs first); overrides the ranges
    scene_id: int | None = None                 # needed when the dataset has several scenes

    @classmethod
    def from_dict(cls, d: dict) -> "RenderProbeSpec":
        d = dict(d)
        for key in ("target_range", "input_range", "render_years", "file_keys"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)


def select_probe_rows(dataset: DatedViewsDataset, spec: RenderProbeSpec) -> np.ndarray:
    """View-table rows for a probe: inputs first. Targets are a tight cluster of cameras."""
    if spec.scene_id is None and dataset.n_groups > 1:
        raise ValueError(f"Probe {spec.name}: give a scene_id, the dataset has several scenes.")
    if spec.scene_id is None:
        in_scene = np.ones(len(dataset.views), dtype=bool)
    else:
        rows = np.flatnonzero((dataset.views["scene_id"] == spec.scene_id).to_numpy())
        if len(rows) == 0:
            raise ValueError(f"Probe {spec.name}: scene {spec.scene_id} is not in the dataset.")
        in_scene = dataset.group == dataset.group[rows[0]]      # one coordinate frame: the scene's first model
    if spec.file_keys:
        index = {k: i for i, k in enumerate(dataset.views["file_key"]) if in_scene[i]}
        return np.array([index[k] for k in spec.file_keys])
    dates, feats = dataset.dates, dataset.camera_feats
    tgt_pool = np.flatnonzero(in_scene & (dates >= spec.target_range[0]) & (dates < spec.target_range[1]))
    inp_pool = np.flatnonzero(in_scene & (dates >= spec.input_range[0]) & (dates < spec.input_range[1]))
    if len(tgt_pool) < spec.n_targets or len(inp_pool) < spec.n_inputs:
        raise ValueError(f"Probe {spec.name}: not enough views in the date ranges.")
    # Anchor: a random target-pool view among those with the tightest target neighbourhood.
    dist = np.linalg.norm(feats[tgt_pool][:, None] - feats[tgt_pool][None], axis=-1)
    spread = np.sort(dist, axis=1)[:, spec.n_targets - 1]
    rng = np.random.default_rng(spec.seed)
    tight = tgt_pool[np.argsort(spread)[: max(1, len(tgt_pool) // 10)]]
    anchor = rng.choice(tight)
    to_anchor = np.linalg.norm(feats - feats[anchor], axis=1)
    targets = tgt_pool[np.argsort(to_anchor[tgt_pool])[: spec.n_targets]]
    inputs = inp_pool[np.argsort(to_anchor[inp_pool])[: spec.n_inputs]]
    return np.concatenate([inputs, targets])


def _pil(x: torch.Tensor, size: int) -> Image.Image:
    arr = ((x.permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size))


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.float() - b.float()) / 2).pow(2).mean().item()
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def _year_label(year: float | None) -> str:
    return "unknown" if year is None else f"{year:g}"


class RenderProbe:
    def __init__(self, dataset: DatedViewsDataset, spec: RenderProbeSpec, date_views: str = "all",
                 uncond_dates: str = "same", frames: int | None = SEVA_FRAMES):
        self.spec = spec
        self.date_views = date_views                    # as in training
        self.uncond_dates = uncond_dates                # what the unconditional CFG branch sees
        self.frames = frames                            # pad each forward pass to this many frames (None: no padding)
        self.camera_scale = dataset.camera_scale
        self.rows = select_probe_rows(dataset, spec)
        self.sample = dataset.make_sample(self.rows, spec.n_inputs)

    def describe(self) -> dict:
        s = self.sample
        return {**asdict(self.spec), "views": [
            {"file_key": k, "input": bool(i), "year": round(float(y), 2)}
            for k, i, y in zip(s["file_keys"], s["is_input"], s["date_mid"])]}

    @torch.no_grad()
    def __call__(self, components: SevaComponents, out_dir: Path, num_steps: int = 25, seed: int = 0,
                 thumb: int = 192, cfg: float = 2.0) -> dict:
        s = self.sample
        is_input = s["is_input"]
        targets = (~is_input).nonzero().flatten().tolist()
        real_year = float(s["date_mid"][targets].median())
        renders, metrics = {}, []
        for year in dict.fromkeys(self.spec.render_years):   # unique, in order (renders is keyed by year)
            # "Unknown" marks every view unknown, as date dropout does in training.
            dates = torch.where(is_input, s["date_mid"], torch.full_like(s["date_mid"], year or 0.0))
            valid = shown_dates(torch.full_like(is_input, year is not None), is_input, self.date_views)
            out = generate_targets(components, s["images"].cuda(), s["K"].cuda(), s["c2w"].cuda(),
                                   is_input.cuda(), dates=dates, date_valid=valid, camera_scale=self.camera_scale,
                                   num_steps=num_steps, seed=seed, cfg=cfg, uncond_dates=self.uncond_dates,
                                   pad_to=self.frames).cpu()
            renders[year] = out
            stats = [color_stats(_pil(out[j], 96)) for j in targets]
            metrics.append({"year": year,
                            "saturation": float(np.mean([st["saturation"] for st in stats])),
                            "chroma_minor": float(np.mean([st["chroma_minor"] for st in stats])),
                            "psnr_vs_real": float(np.mean([_psnr(out[j], s["images"][j]) for j in targets]))})
        # Date sensitivity: how much the targets change relative to the render at the real date.
        ref_year = min((y for y in renders if y is not None), key=lambda y: abs(y - real_year), default=None)
        for m in metrics:
            if ref_year is not None:
                diff = (renders[m["year"]][targets] - renders[ref_year][targets]).abs().mean()
                m["mean_abs_diff_vs_real_date"] = float(diff) / 2   # in [0, 1] pixel units
        out_dir.mkdir(parents=True, exist_ok=True)
        self._sheet(renders, metrics, thumb).save(out_dir / f"{self.spec.name}.jpg", quality=90)
        result = {"probe": self.spec.name, "real_target_year": real_year, "per_year": metrics}
        (out_dir / f"{self.spec.name}.json").write_text(json.dumps(result, indent=2))
        return result

    def _sheet(self, renders: dict, metrics: list[dict], thumb: int) -> Image.Image:
        """Row 0: real views (inputs green, targets red). Then one row of generated targets per year."""
        s = self.sample
        n, label_h = len(s["file_keys"]), 18
        rows = 1 + len(renders)
        sheet = Image.new("RGB", (thumb * n, rows * (thumb + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i in range(n):
            y0 = 0
            inp = bool(s["is_input"][i])
            sheet.paste(_pil(s["images"][i], thumb), (i * thumb, y0 + label_h))
            draw.text((i * thumb + 3, y0 + 3), f"{'IN' if inp else 'REAL'} {float(s['date_mid'][i]):.0f}",
                      fill=(0, 130, 0) if inp else (190, 0, 0))
        for r, (year, m) in enumerate(zip(renders, metrics), start=1):
            y0 = r * (thumb + label_h)
            draw.text((3, y0 + 3), f"date = {_year_label(year)}   sat {m['saturation']:.2f}  "
                      f"chroma {m['chroma_minor']:.1f}", fill=(0, 0, 0))
            for i in range(n):
                if not bool(s["is_input"][i]):
                    sheet.paste(_pil(renders[year][i], thumb), (i * thumb, y0 + label_h))
        return sheet


def load_probe_specs(path: Path) -> list[RenderProbeSpec]:
    return [RenderProbeSpec.from_dict(d) for d in json.loads(Path(path).read_text())]
