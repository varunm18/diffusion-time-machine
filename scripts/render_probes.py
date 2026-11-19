#!/usr/bin/env python
"""Render the date probes with a trained checkpoint (or plain SEVA), after training.

Each probe renders one fixed set of target cameras at several years with the same
seed; see ``timemachine.training.probes``. With ``--checkpoint``, the dataset(s),
sampling, date settings and probes come from the run's ``config.json`` (override the
probes with ``--probes``). Without it, the dated model is untouched SEVA (zero date
module): every row should then be identical, the baseline to compare against.

``--uncond-dates unknown`` (default) leaves the unconditional CFG branch undated; the
training probes of runs before 2026-11-19 used ``same``, which breaks renders
(docs/findings.md). Several ``--seeds`` show how much a render depends on sampler noise.

    python scripts/render_probes.py --checkpoint data/experiments/train/<run>/checkpoints/step_005000.pt
    python scripts/render_probes.py --dataset data/datasets/25512_model0 --probes configs/probes_notre_dame.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.sampling import SamplingConfig
from timemachine.model.dated_seva import UNCOND_DATES
from timemachine.model.finetune import load_finetuned
from timemachine.model.loading import load_components
from timemachine.training.probes import RenderProbe, RenderProbeSpec, load_probe_specs
from timemachine.training.trainer import load_views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="trained checkpoint (default: plain SEVA)")
    parser.add_argument("--dataset", type=Path, nargs="+", help="dataset folder(s)/table(s); default: the run's")
    parser.add_argument("--probes", type=Path, help="probe specs (JSON); default: the run's")
    parser.add_argument("--only", nargs="+", help="render only these probe names")
    parser.add_argument("--uncond-dates", choices=UNCOND_DATES, default="unknown",
                        help="what the unconditional CFG branch sees")
    parser.add_argument("--steps", type=int, default=25, help="sampler steps (training probes use 25)")
    parser.add_argument("--frames", type=int, default=21,
                        help="pad each forward pass to this many frames, as SEVA does (0: unpadded, as before 2026-11-19)")
    parser.add_argument("--cfg", type=float, default=2.0, help="guidance scale")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out", type=Path, help="default: <run>/renders_offline/<ckpt>_uncond-<mode>")
    args = parser.parse_args()

    run_cfg = {}
    if args.checkpoint:
        run_cfg = json.loads((args.checkpoint.parent.parent / "config.json").read_text())
        components, ckpt = load_finetuned(args.checkpoint)
        out = args.out or (args.checkpoint.parent.parent / "renders_offline"
                           / f"{args.checkpoint.stem}_uncond-{args.uncond_dates}_frames{args.frames}")
        print(f"checkpoint {args.checkpoint} (step {ckpt['step']})", flush=True)
    else:
        if not (args.dataset and args.probes):
            parser.error("without --checkpoint, give --dataset and --probes")
        components = load_components(dated=True)
        out = args.out or Path("data/experiments/probes_seva")
    components.model.eval()

    views = load_views([str(d) for d in args.dataset] if args.dataset else run_cfg["dataset"])
    sampling = SamplingConfig(**run_cfg["sampling"]) if run_cfg else SamplingConfig()
    dataset = DatedViewsDataset(views, sampling, resolution=run_cfg.get("resolution", 576),
                                camera_scale=run_cfg.get("camera_scale", 2.0))
    specs = (load_probe_specs(args.probes) if args.probes
             else [RenderProbeSpec.from_dict(d) for d in run_cfg["render_probes"]])
    if args.only:
        specs = [s for s in specs if s.name in args.only]
    date_views = run_cfg.get("loss", {}).get("date_views", "all")
    results = []
    for spec in specs:
        probe = RenderProbe(dataset, spec, date_views=date_views, uncond_dates=args.uncond_dates,
                            frames=args.frames or None)
        for seed in args.seeds:
            result = probe(components, out / f"seed{seed}", num_steps=args.steps, seed=seed, cfg=args.cfg)
            results.append({**result, "seed": seed})
            print(f"{spec.name} seed {seed} done", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
