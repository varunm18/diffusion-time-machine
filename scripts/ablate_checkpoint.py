#!/usr/bin/env python
"""Which trained part changed the model? Date-gap probe with parts of a checkpoint switched off.

Runs the training run's date-gap probe (same 16 samples, noise levels and noise;
see ``timemachine.training.probes.DateGapProbe``) four times:

    plain SEVA   LoRA off, date conditioning off   (= step 0)
    date only    LoRA off
    LoRA only    date module off
    full         the checkpoint as trained

If the probe loss is broken only when the LoRA is on, the adapters did the
damage; if only when the date module is on, the date embedding did.

``--date-scales 0.5 0.25`` adds settings with the date embedding shrunk by
those factors (LoRA off): if the damage fades as the embedding shrinks, its
size, not what it encodes, is the problem.

The loss barely sees damage at high noise levels (where colours and layout are
decided), so ``--render`` also renders one probe (``configs/probes_*.json``) in
each setting, plus the full model without classifier-free guidance (``cfg 1``),
which shows whether guidance exaggerates a drift.

    python scripts/ablate_checkpoint.py data/experiments/train/<run>/checkpoints/step_002500.pt \
        [--render fire --probes configs/probes_notre_dame.json --years 1880 2015] [--date-scales 0.5]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch

from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.sampling import SamplingConfig
from timemachine.model.date_film import date_film_layers
from timemachine.model.finetune import load_finetuned
from timemachine.model.lora import lora_layers
from timemachine.training.probes import DateGapProbe, RenderProbe, load_probe_specs

# (name, LoRA factor, date-embedding factor); 1 = as trained, 0 = switched off.
SETTINGS = (("plain SEVA", 0.0, 0.0), ("date only", 0.0, 1.0), ("LoRA only", 1.0, 0.0), ("full", 1.0, 1.0))


def settings(date_scales: list[float], has_lora: bool = True) -> list[tuple[str, float, float]]:
    """SETTINGS plus the shrunk-date ones; without LoRA, the LoRA settings would only repeat the others."""
    base = SETTINGS if has_lora else [x for x in SETTINGS if x[1] == 0.0]
    return list(base) + [(f"date x{s:g}", 0.0, s) for s in date_scales]


@contextmanager
def scaled(model, lora: float, date: float):
    """Temporarily multiply the LoRA updates and the date conditioning by these factors.

    The date conditioning is the date embedding's output and, with date FiLM, every
    block's scale and shift. Their last layers are linear, so scaling weight and bias
    scales them exactly (approximately if the date embedding has ``max_abs``).
    """
    layers = lora_layers(model.module).values()
    scales = [layer.scale for layer in layers]
    linears = [model.date_module.date_embedding.mlp[-1], *(l.film for l in date_film_layers(model.module).values())]
    saved = [(m.weight.detach().clone(), m.bias.detach().clone()) for m in linears]
    try:
        for layer in layers:
            layer.scale = layer.scale * lora
        with torch.no_grad():
            for m in linears:
                m.weight.mul_(date)
                m.bias.mul_(date)
        yield
    finally:
        for layer, scale in zip(layers, scales):
            layer.scale = scale
        with torch.no_grad():
            for m, (w, b) in zip(linears, saved):
                m.weight.copy_(w)
                m.bias.copy_(b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--render", help="name of a render probe to render in every setting")
    parser.add_argument("--probes", type=Path, help="render probe specs (JSON)")
    parser.add_argument("--years", type=float, nargs="+", default=[1880.0, 2015.0])
    parser.add_argument("--render-steps", type=int, default=25)
    parser.add_argument("--date-scales", type=float, nargs="*", default=[],
                        help="also test the date embedding shrunk by these factors (LoRA off)")
    args = parser.parse_args()

    run_cfg = json.loads((args.checkpoint.parent.parent / "config.json").read_text())
    components, ckpt = load_finetuned(args.checkpoint)
    model = components.model
    views = pd.read_parquet(Path(run_cfg["dataset"]) / "views.parquet")
    dataset = DatedViewsDataset(views, SamplingConfig(**run_cfg["sampling"]), resolution=run_cfg["resolution"],
                                samples_per_epoch=1000, seed=run_cfg["seed"], camera_scale=run_cfg["camera_scale"])
    probe = DateGapProbe(components, dataset, run_cfg["gap_samples"])
    print(f"checkpoint {args.checkpoint} (step {ckpt['step']})", flush=True)
    results = {}
    has_lora = bool(lora_layers(model.module))
    for name, lora, date in settings(args.date_scales, has_lora):
        with scaled(model, lora, date):
            results[name] = probe(model, components.denoiser.sigmas)
        r = results[name]
        print(f"{name:>10}: loss_true {r['loss_true']:.4f}  unknown {r['loss_unknown']:.4f}  "
              f"rel_gap_swapped {r['rel_gap_swapped']:+.4f}  x0_mse_high_noise {r.get('x0_mse_high_noise', 0):.4f}",
              flush=True)
    out = args.checkpoint.with_suffix(".ablation.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    if args.render:
        render_ablation(args, components, dataset, has_lora)


def render_ablation(args, components, dataset, has_lora: bool) -> None:
    spec = next(s for s in load_probe_specs(args.probes) if s.name == args.render)
    probe = RenderProbe(dataset, dataclasses.replace(spec, render_years=tuple(args.years)))
    root = args.checkpoint.parent / f"{args.checkpoint.stem}_ablation"
    runs = [(name, lora, date, 2.0) for name, lora, date in settings(args.date_scales, has_lora)]
    runs.append(("full cfg 1", 1.0, 1.0, 1.0))
    for name, lora, date, cfg in runs:
        with scaled(components.model, lora, date):
            result = probe(components, root / name.replace(" ", "_"), num_steps=args.render_steps, cfg=cfg)
        print(f"{name:>10}: " + "; ".join(f"{m['year']:g}: sat {m['saturation']:.2f} psnr {m['psnr_vs_real']:.1f}"
                                         for m in result["per_year"]), flush=True)
    print(f"renders in {root}")


if __name__ == "__main__":
    main()
