#!/usr/bin/env python
"""Fine-tune date-conditioned SEVA on one or more scenes' datasets (single GPU).

Trains the date conditioning (default: per-block date FiLM) + LoRA adapters (SEVA frozen) with SEVA's denoising
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

from timemachine.data.holdout import HoldoutConfig
from timemachine.data.sampling import SamplingConfig
from timemachine.model.finetune import FinetuneConfig
from timemachine.training.loss import AUTOCAST_DTYPES, DATE_VIEWS, LossConfig
from timemachine.training.probes import load_probe_specs
from timemachine.training.trainer import TrainConfig, Trainer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, nargs="+", required=True,
                   help="per scene, a folder with views.parquet or a view-table .parquet file "
                        "(several: multi-scene training)")
    p.add_argument("--group-weight-power", type=float, default=TrainConfig.group_weight_power,
                   help="multi-scene: sample a scene with probability ~ its training views ** this")
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-root", type=Path, default=Path("data/experiments/train"))
    p.add_argument("--probes", type=Path, help="JSON list of render probe specs")
    p.add_argument("--resume", help="checkpoint to resume from (e.g. <run>/checkpoints/latest.pt), or 'auto': "
                   "the run's latest.pt if it exists (for requeued / preempted jobs), else start fresh")
    p.add_argument("--steps", type=int, default=TrainConfig.steps)
    p.add_argument("--lr-lora", type=float, default=TrainConfig.lr_lora)
    p.add_argument("--lr-date", type=float, default=TrainConfig.lr_date)
    p.add_argument("--lr-film", type=float, default=TrainConfig.lr_film)
    p.add_argument("--date-injection", choices=["film", "timestep", "none"], default=FinetuneConfig.date_injection,
                   help="film: per-block date scale-and-shift; timestep: date vector added to the timestep embedding")
    p.add_argument("--appearance-injection", choices=["film", "none"],
                   default=FinetuneConfig.appearance_injection,
                   help="how a view's appearance (colour, tone, grain) reaches the model: a per-block "
                        "scale-and-shift, or not at all. Our dates are largely readable from the medium "
                        "alone (docs/findings.md), so without this the date path can learn film stock.")
    p.add_argument("--appearance-dim", type=int, default=FinetuneConfig.appearance_dim)
    p.add_argument("--appearance-dropout", type=float, default=LossConfig.appearance_dropout,
                   help="share of samples whose appearance is marked unknown")
    p.add_argument("--lr-appearance", type=float, default=TrainConfig.lr_appearance)
    p.add_argument("--film-dim", type=int, default=FinetuneConfig.film_dim)
    p.add_argument("--film-freqs", type=int, default=FinetuneConfig.film_freqs,
                   help="Fourier octaves of the FiLM date encoder (shortest period 200/2^(n-1) years)")
    p.add_argument("--warmup", type=int, default=TrainConfig.warmup_steps)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default=TrainConfig.lr_schedule,
                   help="after warmup: constant, or cosine decay to --lr-min-ratio x lr at the last step")
    p.add_argument("--lr-min-ratio", type=float, default=TrainConfig.lr_min_ratio)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--autocast", choices=sorted(AUTOCAST_DTYPES), default=TrainConfig.autocast,
                   help="training precision (fp16, without a gradient scaler, reproduces pre-audit runs)")
    p.add_argument("--grad-clip", type=float, default=0.0, help="absolute gradient-norm clip (0: off)")
    p.add_argument("--grad-clip-factor", type=float, default=TrainConfig.grad_clip_factor,
                   help="clip only spikes above this x the median of recent gradient norms (0: off)")
    p.add_argument("--lora-rank", type=int, default=32, help="0: train the date module only")
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--date-max-abs", type=float, default=None,
                   help="soft bound per date-embedding dimension (SEVA's timestep embedding RMS is ~0.3)")
    p.add_argument("--n-inputs", type=int, default=3)
    p.add_argument("--n-targets", type=int, default=5)
    p.add_argument("--max-neighbors", type=int, default=24, help="0: no spatial restriction")
    p.add_argument("--old-target-fraction", type=float, default=0.5)
    p.add_argument("--min-input-gap", type=float, default=0.0,
                   help="inputs must be more than this many years from the target date (0: any date), so the "
                        "model cannot copy the targets' look from a same-date input and has to use the date")
    p.add_argument("--input-gap-fraction", type=float, default=1.0,
                   help="share of samples that enforce --min-input-gap; the rest take inputs from any date")
    p.add_argument("--gap-old-targets-only", action="store_true",
                   help="enforce --min-input-gap only for old targets (no modern-from-old samples by force)")
    p.add_argument("--event", action="append", default=[], metavar="SCENE_ID:YEAR",
                   help="a physical change in a scene (repeatable), e.g. 25512:2019.29 (Notre-Dame fire)")
    p.add_argument("--event-prob", type=float, default=0.0,
                   help="share of samples drawn across an event: targets one side, inputs the other")
    p.add_argument("--event-window", type=float, default=15.0,
                   help="event samples: inputs within this many years of the change when enough exist (0: any)")
    p.add_argument("--rank-weight", type=float, default=0.0, help="date-discrimination ranking loss weight")
    p.add_argument("--rank-margin", type=float, default=LossConfig.rank_margin)
    p.add_argument("--rank-prob", type=float, default=LossConfig.rank_prob)
    p.add_argument("--rank-min-sigma", type=float, default=LossConfig.rank_min_sigma)
    p.add_argument("--rank-reference", choices=["wrong", "unknown"], default=LossConfig.rank_reference,
                   help="compare the true date with a wrong-era date or with the unknown date")
    p.add_argument("--rank-detach", action="store_true",
                   help="no gradient through the reference branch, so it cannot be made worse")
    p.add_argument("--uncond-prob", type=float, default=0.1)
    p.add_argument("--date-views", choices=DATE_VIEWS, default="all",
                   help="targets: only target views see their date (inputs always unknown)")
    p.add_argument("--date-dropout", type=float, default=0.1)
    p.add_argument("--uncond-dates", choices=["same", "unknown"], default="same",
                   help="unconditional-branch training steps: dates as drawn, or all unknown")
    p.add_argument("--render-frames", type=int, default=21,
                   help="render probes: pad each forward pass to this many frames, as SEVA does (0: unpadded)")
    p.add_argument("--render-uncond-dates", choices=["same", "unknown"], default="same",
                   help="render probes: what the unconditional CFG branch sees (unknown: avoids broken renders)")
    p.add_argument("--no-date-jitter", action="store_true")
    p.add_argument("--weighting", choices=["eps", "edm", "min_snr"], default="eps",
                   help="per-noise-level loss weight (see timemachine.training.loss)")
    p.add_argument("--abort-loss-ratio", type=float, default=TrainConfig.abort_loss_ratio,
                   help="stop when the date-gap probe loss exceeds this multiple of its first value (0: never)")
    p.add_argument("--abort-x0-ratio", type=float, default=TrainConfig.abort_x0_ratio,
                   help="stop when the probe's high-noise denoised-latent error (colour drift) exceeds this "
                        "multiple of its first value (0: never)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--gap-every", type=int, default=500)
    p.add_argument("--gap-samples", type=int, default=TrainConfig.gap_samples,
                   help="fixed samples in the date-gap probe (split by era, so 32+ for per-era numbers)")
    p.add_argument("--render-every", type=int, default=2500)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--latest-every", type=int, default=TrainConfig.latest_every,
                   help="overwrite <run>/checkpoints/latest.pt this often (resume point after preemption)")
    p.add_argument("--render-steps", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--holdout-fraction", type=float, default=0.0,
                   help="hold this share of each era's views out of training (evaluated by the held-out probe)")
    p.add_argument("--holdout-probe-targets", action="store_true",
                   help="also hold out the render probes' target views")
    p.add_argument("--holdout-scenes", type=int, nargs="*", default=(),
                   help="scene ids held out entirely (evaluation only)")
    p.add_argument("--holdout-seed", type=int, default=HoldoutConfig.seed)
    p.add_argument("--heldout-every", type=int, default=TrainConfig.heldout_every)
    p.add_argument("--heldout-pairs", type=int, default=TrainConfig.heldout_pairs)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = TrainConfig(
        dataset=str(args.dataset[0]) if len(args.dataset) == 1 else tuple(map(str, args.dataset)),
        run_dir=str(args.runs_root / args.run_name), steps=args.steps, group_weight_power=args.group_weight_power,
        lr_lora=args.lr_lora, lr_date=args.lr_date, lr_film=args.lr_film, lr_appearance=args.lr_appearance,
        warmup_steps=args.warmup, grad_accum=args.grad_accum,
        lr_schedule=args.lr_schedule, lr_min_ratio=args.lr_min_ratio,
        autocast=args.autocast,
        grad_clip=args.grad_clip or None, grad_clip_factor=args.grad_clip_factor or None,
        seed=args.seed, num_workers=args.num_workers, log_every=args.log_every, gap_every=args.gap_every,
        gap_samples=args.gap_samples,
        render_every=args.render_every, save_every=args.save_every, latest_every=args.latest_every,
        render_steps=args.render_steps, render_uncond_dates=args.render_uncond_dates,
        render_frames=args.render_frames or None,
        abort_loss_ratio=args.abort_loss_ratio or None, abort_x0_ratio=args.abort_x0_ratio or None,
        sampling=SamplingConfig(n_inputs=args.n_inputs, n_targets=args.n_targets,
                                old_target_fraction=args.old_target_fraction or None,
                                min_input_gap=args.min_input_gap, input_gap_fraction=args.input_gap_fraction,
                                gap_old_targets_only=args.gap_old_targets_only,
                                events=tuple((int(e.split(":")[0]), float(e.split(":")[1])) for e in args.event),
                                event_prob=args.event_prob, event_window=args.event_window or None,
                                max_neighbors=args.max_neighbors or None),
        loss=LossConfig(uncond_prob=args.uncond_prob, date_dropout=args.date_dropout,
                        date_jitter=not args.no_date_jitter, weighting=args.weighting,
                        date_views=args.date_views, uncond_dates=args.uncond_dates,
                        rank_weight=args.rank_weight, rank_margin=args.rank_margin, rank_prob=args.rank_prob,
                        rank_min_sigma=args.rank_min_sigma, rank_reference=args.rank_reference,
                        rank_detach=args.rank_detach, appearance_dropout=args.appearance_dropout),
        finetune=FinetuneConfig(lora_rank=args.lora_rank, grad_checkpointing=not args.no_grad_checkpointing,
                                date_max_abs=args.date_max_abs, date_injection=args.date_injection,
                                film_dim=args.film_dim, film_freqs=args.film_freqs,
                                appearance_injection=args.appearance_injection,
                                appearance_dim=args.appearance_dim),
        heldout_every=args.heldout_every, heldout_pairs=args.heldout_pairs,
        holdout=HoldoutConfig(fraction=args.holdout_fraction, probe_targets=args.holdout_probe_targets,
                              scenes=tuple(args.holdout_scenes), seed=args.holdout_seed),
        render_probes=tuple(load_probe_specs(args.probes)) if args.probes else (),
    )
    trainer = Trainer(cfg)
    resume = Path(args.resume) if args.resume and args.resume != "auto" else None
    if args.resume == "auto" and (Path(cfg.run_dir) / "checkpoints" / "latest.pt").exists():
        resume = Path(cfg.run_dir) / "checkpoints" / "latest.pt"
    if resume:
        trainer.resume(resume)
    trainer.train()


if __name__ == "__main__":
    main()
