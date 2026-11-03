#!/usr/bin/env python
"""Fine-tune date-conditioned SEVA on one scene's dataset (single GPU).

Trains the date module + LoRA adapters (SEVA frozen) with SEVA's denoising
objective; see ``timemachine.training``. Writes a run folder with logs,
checkpoints and periodic probe renders (``configs/probes_*.json``).

    python scripts/train_dated_seva.py --dataset data/datasets/25512_model0 \\
        --run-name notre_dame_overfit --probes configs/probes_notre_dame.json --steps 10000

Needs SEVA (scripts/setup_seva.sh) and TIMEMACHINE_HF_HUB; run through
slurm/seva_gpu.sbatch (SCRIPT=scripts/train_dated_seva.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from timemachine.data.sampling import SamplingConfig
from timemachine.model.finetune import FinetuneConfig
from timemachine.training.loss import LossConfig
from timemachine.training.probes import load_probe_specs
from timemachine.training.trainer import TrainConfig, Trainer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="folder with views.parquet")
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-root", type=Path, default=Path("data/experiments/train"))
    p.add_argument("--probes", type=Path, help="JSON list of render probe specs")
    p.add_argument("--resume", type=Path, help="checkpoint to resume from (e.g. <run>/checkpoints/latest.pt)")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-date", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lora-rank", type=int, default=32, help="0: train the date module only")
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--n-inputs", type=int, default=3)
    p.add_argument("--n-targets", type=int, default=5)
    p.add_argument("--max-neighbors", type=int, default=24, help="0: no spatial restriction")
    p.add_argument("--old-target-fraction", type=float, default=0.5)
    p.add_argument("--uncond-prob", type=float, default=0.1)
    p.add_argument("--date-dropout", type=float, default=0.1)
    p.add_argument("--no-date-jitter", action="store_true")
    p.add_argument("--weighting", choices=["eps", "edm", "min_snr"], default="eps",
                   help="per-noise-level loss weight (see timemachine.training.loss)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--gap-every", type=int, default=500)
    p.add_argument("--render-every", type=int, default=2500)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--render-steps", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = TrainConfig(
        dataset=str(args.dataset), run_dir=str(args.runs_root / args.run_name), steps=args.steps,
        lr_lora=args.lr_lora, lr_date=args.lr_date, warmup_steps=args.warmup, grad_accum=args.grad_accum,
        seed=args.seed, num_workers=args.num_workers, log_every=args.log_every, gap_every=args.gap_every,
        render_every=args.render_every, save_every=args.save_every, render_steps=args.render_steps,
        sampling=SamplingConfig(n_inputs=args.n_inputs, n_targets=args.n_targets,
                                old_target_fraction=args.old_target_fraction or None,
                                max_neighbors=args.max_neighbors or None),
        loss=LossConfig(uncond_prob=args.uncond_prob, date_dropout=args.date_dropout,
                        date_jitter=not args.no_date_jitter, weighting=args.weighting),
        finetune=FinetuneConfig(lora_rank=args.lora_rank, grad_checkpointing=not args.no_grad_checkpointing),
        render_probes=tuple(load_probe_specs(args.probes)) if args.probes else (),
    )
    trainer = Trainer(cfg)
    if args.resume:
        trainer.resume(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
