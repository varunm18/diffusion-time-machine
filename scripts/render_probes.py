#!/usr/bin/env python
"""Render the date probes with a trained checkpoint (or plain SEVA), at full sampler quality.

Each probe renders one fixed set of target cameras at several years with the same
seed; see ``timemachine.training.probes``. Without ``--checkpoint`` the dated
model is untouched SEVA (zero date module): every row should then be identical,
which is the baseline to compare against.

    python scripts/render_probes.py --dataset data/datasets/25512_model0 \\
        --probes configs/probes_notre_dame.json \\
        --checkpoint data/experiments/train/notre_dame_overfit/checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from timemachine.data.dataset import DatedViewsDataset
from timemachine.model.finetune import load_finetuned
from timemachine.model.loading import load_components
from timemachine.training.probes import RenderProbe, load_probe_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="trained checkpoint (default: plain SEVA)")
    parser.add_argument("--out", type=Path, help="default: <run>/renders_final/<ckpt> or data/experiments/probes_seva")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    if args.checkpoint:
        components, ckpt = load_finetuned(args.checkpoint)
        out = args.out or args.checkpoint.parent.parent / "renders_final" / args.checkpoint.stem
        print(f"checkpoint {args.checkpoint} (step {ckpt['step']})", flush=True)
    else:
        components = load_components(dated=True)
        out = args.out or Path("data/experiments/probes_seva")
    components.model.eval()

    dataset = DatedViewsDataset(pd.read_parquet(args.dataset / "views.parquet"))
    results = []
    for spec in load_probe_specs(args.probes):
        probe = RenderProbe(dataset, spec)
        for seed in args.seeds:
            result = probe(components, out / f"seed{seed}", num_steps=args.steps, seed=seed)
            results.append({**result, "seed": seed})
            print(json.dumps(result), flush=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
