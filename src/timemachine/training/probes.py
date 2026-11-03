"""Fixed evaluation probes for date-conditioned SEVA.

Two probes, both run on the *same* fixed inputs at every evaluation, so
numbers and images are comparable across training steps:

**Date-swap loss gap** (:class:`DateGapProbe`, cheap). A fixed set of training
samples, each with a fixed noise level and noise. We compute the denoising loss
with the true dates, with the targets' dates swapped to another era, and with
all dates marked unknown. If the model uses dates, the relative gap
``loss_swapped / loss_true - 1`` grows above zero. At step 0 (zero-init date module) all three are equal.

**Renders at many dates** (:class:`RenderProbe`, slow). One fixed set of input
views and target cameras, chosen by a :class:`RenderProbeSpec` (e.g. "targets
from after the 2019 fire, inputs from before it"). The targets are generated at
several years with the same seed, so differences between rows are caused by
the date alone. Saved as a contact sheet, with colour statistics per year and
PSNR against the real targets.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.images import color_stats
from timemachine.model.inference import generate_targets
from timemachine.model.loading import SevaComponents
from timemachine.training.conditioning import sample_conditions
from timemachine.training.loss import NUM_LEVELS, noise_prediction_loss

SWAP_BOUNDARY = 1980.0   # targets dated before this are swapped to MODERN_SWAP, others to OLD_SWAP
MODERN_SWAP = 2015.0
OLD_SWAP = 1890.0


def swapped_years(year: torch.Tensor, is_input: torch.Tensor) -> torch.Tensor:
    """Targets' years moved to the other era (old -> 2015, modern -> 1890); inputs unchanged."""
    swapped = torch.where(year < SWAP_BOUNDARY, torch.full_like(year, MODERN_SWAP), torch.full_like(year, OLD_SWAP))
    return torch.where(is_input, year, swapped)


# ----------------------------------------------------------------------------
# Date-swap loss gap
# ----------------------------------------------------------------------------
class DateGapProbe:
    """Loss with true / swapped / unknown dates on fixed samples, noise levels and noise."""

    def __init__(self, components: SevaComponents, dataset: DatedViewsDataset, n_samples: int = 16,
                 seed: int = 1234, device: str = "cuda"):
        gen = torch.Generator().manual_seed(seed)
        # Evenly spread noise levels (skip the extremes, where dates cannot matter much).
        levels = np.linspace(100, NUM_LEVELS - 100, n_samples).round().astype(int)
        self.items = []
        saved_epoch = dataset.epoch
        dataset.set_epoch(10_000 + seed)                     # an epoch the training run never uses
        for i in range(n_samples):
            sample = dataset[i]
            latents, cond, _ = sample_conditions(components, sample, device)
            self.items.append({
                "latents": latents, "cond": cond, "level": int(levels[i]),
                "noise": torch.randn(latents.shape, generator=gen).to(device),
                "year": sample["date_mid"].to(device), "is_input": sample["is_input"].to(device),
                "old_targets": bool(sample["date_mid"][~sample["is_input"]].max() < SWAP_BOUNDARY),
            })
        dataset.set_epoch(saved_epoch)

    @torch.no_grad()
    def __call__(self, model, sigmas: torch.Tensor) -> dict:
        rows = []
        for it in self.items:
            ones = torch.ones_like(it["is_input"])
            with torch.autocast("cuda"):
                def loss(year, valid):
                    return float(noise_prediction_loss(model, sigmas, it["latents"], it["cond"], it["level"],
                                                       it["noise"], year, valid))
                rows.append({"true": loss(it["year"], ones),
                             "swapped": loss(swapped_years(it["year"], it["is_input"]), ones),
                             "unknown": loss(it["year"], ~ones),
                             "old_targets": it["old_targets"]})
        true, swapped, unknown = (np.array([r[k] for r in rows]) for k in ("true", "swapped", "unknown"))
        old = np.array([r["old_targets"] for r in rows])
        high = np.array([float(sigmas[it["level"]]) > 10 for it in self.items])
        # Relative gaps (loss ratio - 1) are comparable across noise levels, whose losses differ ~100x.
        rel_swapped, rel_unknown = swapped / true - 1, unknown / true - 1
        out = {"loss_true": true.mean(), "loss_swapped": swapped.mean(), "loss_unknown": unknown.mean(),
               "rel_gap_swapped": rel_swapped.mean(), "rel_gap_unknown": rel_unknown.mean()}
        for name, mask in (("old_targets", old), ("modern_targets", ~old),
                           ("high_noise", high), ("low_noise", ~high)):
            if mask.any():
                out[f"rel_gap_swapped_{name}"] = rel_swapped[mask].mean()
        return {k: float(v) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Renders at many dates
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderProbeSpec:
    """Which views to render, and at which years. ``None`` in ``render_years`` = date unknown."""

    name: str
    target_range: tuple[float, float]           # real targets dated in [lo, hi)
    input_range: tuple[float, float]            # inputs dated in [lo, hi)
    render_years: tuple[float | None, ...]
    n_inputs: int = 3
    n_targets: int = 5
    seed: int = 0
    file_keys: tuple[str, ...] = field(default=())   # pin exact views (inputs first); overrides the ranges

    @classmethod
    def from_dict(cls, d: dict) -> "RenderProbeSpec":
        d = dict(d)
        for key in ("target_range", "input_range", "render_years", "file_keys"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)


def select_probe_rows(dataset: DatedViewsDataset, spec: RenderProbeSpec) -> np.ndarray:
    """View-table rows for a probe: inputs first. Targets are a tight cluster of cameras."""
    if spec.file_keys:
        index = {k: i for i, k in enumerate(dataset.views["file_key"])}
        return np.array([index[k] for k in spec.file_keys])
    dates, feats = dataset.dates, dataset.camera_feats
    tgt_pool = np.flatnonzero((dates >= spec.target_range[0]) & (dates < spec.target_range[1]))
    inp_pool = np.flatnonzero((dates >= spec.input_range[0]) & (dates < spec.input_range[1]))
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
    def __init__(self, dataset: DatedViewsDataset, spec: RenderProbeSpec):
        self.spec = spec
        self.rows = select_probe_rows(dataset, spec)
        self.sample = dataset.make_sample(self.rows, spec.n_inputs)

    def describe(self) -> dict:
        s = self.sample
        return {**asdict(self.spec), "views": [
            {"file_key": k, "input": bool(i), "year": round(float(y), 2)}
            for k, i, y in zip(s["file_keys"], s["is_input"], s["date_mid"])]}

    @torch.no_grad()
    def __call__(self, components: SevaComponents, out_dir: Path, num_steps: int = 25, seed: int = 0,
                 thumb: int = 192) -> dict:
        s = self.sample
        is_input = s["is_input"]
        targets = (~is_input).nonzero().flatten().tolist()
        real_year = float(s["date_mid"][targets].median())
        renders, metrics = {}, []
        for year in self.spec.render_years:
            dates = torch.where(is_input, s["date_mid"], torch.full_like(s["date_mid"], year or 0.0))
            valid = is_input | (year is not None)
            out = generate_targets(components, s["images"].cuda(), s["K"].cuda(), s["c2w"].cuda(),
                                   is_input.cuda(), dates=dates, date_valid=valid,
                                   num_steps=num_steps, seed=seed).cpu()
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
