#!/usr/bin/env python
"""Score checkpoints (and plain SEVA) on a run's held-out views, all on identical samples.

Rebuilds the run's hold-out exactly (``trainer.run_heldout_rows``; checked against the run's
``heldout.json``), builds one :class:`~timemachine.training.probes.HeldoutProbe` and
evaluates every model on the same pairs, noise levels and noise. The first model is the
reference for the ``*_change`` numbers, so put ``seva`` first. Checkpoints from other runs
can be compared as long as they used the same data and hold-out (e.g. a no-date control).

Key numbers per model (see ``HeldoutProbe``): denoising-loss change on held-out / trained
views vs the reference, and ``rel_swapped`` (> 0: the true date beats a wrong-era date),
the test of temporal learning.

    python scripts/eval_heldout.py --run data/experiments/train/ms13_lora_f4_holdout \\
        --models seva data/experiments/train/ms13_lora_f4_holdout/checkpoints/step_005000.pt \\
                 data/experiments/train/ms13_lora_nodate_holdout/checkpoints/step_005000.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.holdout import HoldoutConfig
from timemachine.data.sampling import SamplingConfig
from timemachine.model.finetune import load_finetuned
from timemachine.model.loading import load_components
from timemachine.training.probes import HeldoutProbe, RenderProbeSpec, select_probe_rows
from timemachine.training.trainer import load_views, run_heldout_rows

SUMMARY = ("heldout_change_of_mean", "train_change_of_mean", "heldout_change_scene", "heldout_rel_unknown",
           "heldout_rel_swapped", "heldout_rel_swapped_old", "heldout_rel_swapped_modern",
           "heldout_rel_swapped_scene", "train_rel_swapped")


def build_dataset(run_dir: Path) -> tuple[DatedViewsDataset, dict, HoldoutConfig]:
    cfg = json.loads((run_dir / "config.json").read_text())
    dataset = DatedViewsDataset(load_views(cfg["dataset"]), SamplingConfig(**cfg["sampling"]),
                                resolution=cfg["resolution"], seed=cfg["seed"], camera_scale=cfg["camera_scale"],
                                group_weight_power=cfg.get("group_weight_power", 0.5))
    h = cfg["holdout"]
    holdout = HoldoutConfig(fraction=h["fraction"], probe_targets=h["probe_targets"], scenes=tuple(h["scenes"]),
                            seed=h["seed"], era_edges=tuple(h["era_edges"]))
    probes = [SimpleNamespace(spec=s, rows=select_probe_rows(dataset, s))
              for s in (RenderProbeSpec.from_dict(d) for d in cfg["render_probes"])]
    rows = run_heldout_rows(dataset, holdout, probes)
    recorded = {v["file_key"] for v in json.loads((run_dir / "heldout.json").read_text())["views"]}
    if set(dataset.views.file_key[rows]) != recorded:
        raise ValueError(f"Rebuilt hold-out differs from {run_dir / 'heldout.json'}")
    dataset.hold_out(rows)
    return dataset, cfg, holdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="run whose data and hold-out define the samples")
    p.add_argument("--models", nargs="+", required=True, help="'seva' (plain SEVA) and/or checkpoint paths")
    p.add_argument("--pairs", type=int, default=64)
    p.add_argument("--out", type=Path, help="default: <run>/eval_heldout.json")
    args = p.parse_args()

    dataset, cfg, holdout = build_dataset(args.run)
    out_path = args.out or args.run / "eval_heldout.json"
    baseline = out_path.with_suffix(".baseline.json")
    baseline.unlink(missing_ok=True)                    # the first model of this call is the reference
    probe, results = None, {}
    for name in args.models:
        components = load_components(dated=True) if name == "seva" else load_finetuned(Path(name))[0]
        components.model.eval()
        if probe is None:
            probe = HeldoutProbe(components, dataset, baseline, args.pairs, date_views=cfg["loss"]["date_views"],
                                 era_edges=holdout.era_edges)
            print(f"{len(probe.items)} pairs ({sum(it['old'] for it in probe.items)} old)", flush=True)
        results[name] = {**probe(components.model, components.denoiser.sigmas), "per_sample": probe.last_per_sample}
        print(name, {k: round(results[name][k], 4) for k in SUMMARY if k in results[name]}, flush=True)
        del components
        torch.cuda.empty_cache()
    out_path.write_text(json.dumps({"reference": args.models[0], "pairs": probe.describe(), "results": results},
                                   indent=1))
    print(f"\n{'model':70s} " + " ".join(f"{k.replace('heldout_', 'h_').replace('train_', 't_'):>22s}" for k in SUMMARY))
    for name, r in results.items():
        print(f"{name[-70:]:70s} " + " ".join(f"{r.get(k, np.nan):+22.4f}" for k in SUMMARY))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
