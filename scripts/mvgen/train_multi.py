#!/usr/bin/env python
"""Train date (and look-code) conditioning on many scenes at once, on one or several GPUs.

A lean sibling of ``scripts/mvgen/train.py`` for the first multi-scene run. It reuses the parts that are
tested (the released model and its adapters, ``timemachine.mvgen.model``; the training step,
``timemachine.mvgen.step.training_loss``; the samples of each scene, ``MVGenScene``; the look inputs,
``TrainingLooks``; the date-loss probe, ``DateLossProbe``; the weight average, ``trainer.WeightEMA``) and
adds what many scenes and many GPUs need:

* the samples come from :class:`timemachine.mvgen.multiscene.SceneMixture`, numbered so that rank ``r`` of
  ``W`` takes the numbers ``r, r + W, ...``: every GPU trains on different samples and a resumed run
  continues the same streams;
* data parallelism without a wrapper: after each backward pass the adapters' gradients are summed over
  the ranks (``all_reduce``) and divided by the number of ranks, so every rank takes the same optimiser
  step and the weights stay identical (checked at every log line);
* rank 0 alone writes logs, runs the probes (Notre-Dame's held-out and training view sets, the date-loss
  probe of the single-scene runs) and saves checkpoints; the other ranks wait.

Checkpoints: ``checkpoints/step_XXXXXX.pt`` holds the averaged weights (for ``load_finetuned``,
``render.py``, ``eval_dates.py --checkpoint``) and ``checkpoints/latest.pt`` also the raw weights and the
optimiser, from which ``--resume`` continues (the per-rank random generators are re-seeded from the
step, so a resumed run is not bit-identical to an uninterrupted one, but draws the same samples).

Launch (one node, N GPUs), from MVGenMaster's checkout with its environment:

    torchrun --standalone --nproc_per_node N $REPO/scripts/mvgen/train_multi.py --run-dir <folder> \\
        --scenes-file <file with one dataset folder per line> [--look] ...

Test of the data-parallel logic on ONE GPU: with ``DTM_DDP_SAME_GPU=1`` every rank uses ``cuda:0`` and the
ranks talk through gloo on CPU copies (NCCL refuses two ranks on one device). Slow; only for tests.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from timemachine.mvgen.data import ViewSamplingConfig
from timemachine.mvgen.model import (
    DEFAULT_MODEL_DIR, FinetuneConfig, assert_exact_matmul, load_mvgen, load_trainable_state_dict,
    prepare_for_finetuning, save_checkpoint, trainable_state_dict,
)
from timemachine.mvgen.multiscene import LookSettings, SceneMixture
from timemachine.mvgen.probes import DateLossProbe, heldout_view_sets, representative_years, scene_view_sets
from timemachine.mvgen.step import LossConfig, training_loss
from timemachine.mvgen.trainer import WeightEMA

LORA_TARGETS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "ff.net.0.proj", "ff.net.2",
                "time_emb_proj")
NOTRE_DAME = "25512_model0_v2"


class Numbered(Dataset):
    """The mixture's samples by number (picklable for data-loading processes)."""

    def __init__(self, mixture: SceneMixture):
        self.mixture = mixture

    def __len__(self) -> int:
        return 2 ** 62

    def __getitem__(self, number: int) -> dict:
        return self.mixture[number]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--scenes-file", type=Path, required=True, help="one dataset folder per line ('#' comments)")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--steps", type=int, default=100_000, help="optimiser steps")
    p.add_argument("--grad-accum", type=int, default=1, help="samples per GPU per optimiser step")
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate of the date, LoRA and look adapters")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--look", action="store_true", help="give every view its look code (experiment look1, arm D)")
    p.add_argument("--medium-grey", type=float, default=0.1)
    p.add_argument("--medium-toned", type=float, default=0.05)
    p.add_argument("--notre-dame-share", type=float, default=0.15)
    p.add_argument("--old-anchor-share", type=float, default=None,
                   help="draw anchors from before the first period edge at least this often (default: off)")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--probe-every", type=int, default=2000)
    p.add_argument("--probe-samples", type=int, default=8)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--latest-every", type=int, default=500)
    p.add_argument("--resume", action="store_true", help="continue from <run-dir>/checkpoints/latest.pt if it exists")
    return p.parse_args()


def all_reduce_mean(flat: torch.Tensor, world: int, on_cpu: bool) -> torch.Tensor:
    """The mean of ``flat`` over the ranks (through a CPU copy in the one-GPU test mode)."""
    if on_cpu:
        host = flat.cpu()
        dist.all_reduce(host)
        return host.to(flat.device) / world
    dist.all_reduce(flat)
    return flat / world


def main() -> None:
    args = parse_args()
    assert_exact_matmul()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    same_gpu = os.environ.get("DTM_DDP_SAME_GPU") == "1"      # test mode, see the module docstring
    if same_gpu:
        local = 0
    if world > 1:
        dist.init_process_group("gloo" if same_gpu else "nccl", timeout=timedelta(minutes=90))
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    lead = rank == 0
    say = (lambda *a: print(*a, flush=True)) if lead else (lambda *a: None)

    folders = [Path(line.strip()) for line in args.scenes_file.read_text().splitlines()
               if line.strip() and not line.startswith("#")]
    model = load_mvgen(args.model_dir, device=device)
    look = LookSettings(grey=args.medium_grey, toned=args.medium_toned) if args.look else None
    mixture = SceneMixture(folders, ViewSamplingConfig(), seed=args.seed,
                           camera_longest_side=model.config.camera_longest_side,
                           fixed_shares={NOTRE_DAME: args.notre_dame_share}, look=look,
                           old_anchor_share=args.old_anchor_share)
    say(f"[multi] {len(mixture.scenes)} scenes, {sum(int(s.target_ok.sum()) for s in mixture.scenes)} training views; "
        f"{world} GPU(s)\n{mixture.summary()}")

    finetune = FinetuneConfig(lora_rank=args.lora_rank, lora_targets=LORA_TARGETS, look=args.look)
    loss_cfg = LossConfig(precision="bf16", look_dropout=0.5, look_class_unspecified=0.3,
                          medium_grey=args.medium_grey if args.look else 0.0,
                          medium_toned=args.medium_toned if args.look else 0.0)
    torch.manual_seed(args.seed)                    # the same initial adapters on every rank
    groups = prepare_for_finetuning(model, finetune)
    if args.look:
        model.look_stats = mixture.look_stats
    params = [p for name in ("date", "lora", "look") for p in groups.get(name, [])]
    optimizer = torch.optim.AdamW([{"params": groups[n], "lr": args.lr, "name": n}
                                   for n in ("date", "lora", "look") if groups.get(n)], weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))
    ema = WeightEMA(params, args.ema_decay) if args.ema_decay > 0 else None
    step, norms = 0, deque(maxlen=200)

    run_dir, latest = args.run_dir, args.run_dir / "checkpoints" / "latest.pt"
    if args.resume and latest.exists():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        load_trainable_state_dict(model, ckpt["trainable"])
        if ema is not None:
            ema.reset(updates=ckpt["ema_updates"])
            load_trainable_state_dict(model, ckpt["raw_trainable"])
        optimizer.load_state_dict(ckpt["optimizer"])
        schedule.load_state_dict(ckpt["lr_scheduler"])
        norms.extend(ckpt["grad_norms"])
        step = ckpt["step"]
        say(f"[multi] resumed from {latest} at step {step}")
    generator = torch.Generator().manual_seed(args.seed * 100_003 + rank * 7919 + step)

    # train_config / scale / scale_factor in the form render.py reads (single-scene checkpoints have them):
    # no dataset, no period edges and no factor, which are the rendered scene's own (scene.json).
    extra_base = {"train_args": {k: str(v) for k, v in vars(args).items()}, "scenes": mixture.names,
                  "shares": mixture.shares.tolist(), "precision": "bf16", "world_size": world,
                  "train_config": {"loss": asdict(loss_cfg), "sampling": {"n_ref": mixture.sampling.n_ref},
                                   "resolution": mixture.resolution, "scale": "scene", "multi_scene": True},
                  "scale": "scene", "scale_factor": None}

    def save(path: Path, full: bool) -> None:
        extra = dict(extra_base, weights="ema" if ema is not None else "raw")
        if full:
            extra.update(optimizer=optimizer.state_dict(), lr_scheduler=schedule.state_dict(),
                         grad_norms=list(norms))
            if ema is not None:
                extra.update(raw_trainable=trainable_state_dict(model), ema_updates=ema.updates)
        if ema is not None:
            with ema.applied():
                save_checkpoint(path, model, finetune, step, extra=extra)
        else:
            save_checkpoint(path, model, finetune, step, extra=extra)

    # probes (rank 0): Notre-Dame's held-out and training view sets, as in the single-scene runs
    probes = {}
    if lead and args.probe_every > 0 and NOTRE_DAME in mixture.names:
        nd = mixture.scenes[mixture.names.index(NOTRE_DAME)]
        swap = representative_years(nd.dates[nd.target_ok], nd.config.period_edges)
        options = dict(look_stats=model.look_stats, look_axis=False) if args.look else {}
        # the probe's view sets are drawn with the scene's natural anchor weights, so that a run resumed with
        # another --old-anchor-share keeps the same sets (and its first evaluation stays the baseline)
        sampling = nd.config
        nd.config = replace(sampling, anchor_period_weights=None)
        drawn = (("nd_heldout", heldout_view_sets(nd, args.probe_samples)),
                 ("nd_train", scene_view_sets(nd, args.probe_samples)))
        nd.config = sampling
        for name, sets in drawn:
            if sets:
                probes[name] = DateLossProbe(nd, sets, 3, loss_cfg, swap, run_dir / "probes" / f"{name}_baseline.json",
                                             **options)

    def run_probes() -> None:
        if not lead:
            return
        model.unet.eval()
        with torch.no_grad(), (ema.applied() if ema is not None else _nothing()):
            for name, probe in probes.items():
                started = time.time()
                result = probe(model)
                log({"step": step, "probe": name, **{k: v for k, v in result.items() if np.isscalar(v)}})
                say(f"[probe] step {step} {name} ({time.time() - started:.0f}s): loss {result['loss_true']:.4f} "
                    f"({result['change']:+.1%}); wrong date {result['rel_gap_swapped']:+.2%}, unknown date "
                    f"{result['rel_gap_unknown']:+.2%}")
        model.unet.train(True)

    def log(record: dict) -> None:
        if lead:
            with open(run_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

    if lead:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps({**extra_base, "finetune": asdict(finetune),
                                                         "loss": asdict(loss_cfg)}, indent=1, default=str))
        sizes = {n: sum(p.numel() for p in groups[n]) / 1e6 for n in ("date", "lora", "look") if groups.get(n)}
        say(f"[multi] trainable (M): {sizes}; lr {args.lr}; EMA {args.ema_decay}; look {args.look}")
    if step == 0:
        run_probes()
        if lead:
            save(latest, full=True)
    if world > 1:
        dist.barrier()

    numbers = (n for n in range(step * args.grad_accum * world + rank, 2 ** 62, world))
    loader = iter(DataLoader(Numbered(mixture), batch_size=None, sampler=numbers, num_workers=args.num_workers,
                             persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers else None))
    model.unet.train(True)
    window, started = [], time.time()
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            sample = next(loader)
            loss, info = training_loss(model, sample, loss_cfg, generator, sample["period_edges"])
            (loss / args.grad_accum).backward()
            window.append({"loss": float(loss), "scene": int(sample["scene"]), "t": info["timestep"],
                           "date_temb_ratio": info.get("date_temb_ratio"), "look_temb_ratio": info.get("look_temb_ratio")})
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        if world > 1:
            flat = torch.cat([g.reshape(-1) for g in grads])
            flat = all_reduce_mean(flat, world, same_gpu)
            offset = 0
            for p, g in zip(params, grads):
                p.grad = flat[offset:offset + g.numel()].view_as(g)
                offset += g.numel()
        limit = 10 * float(np.median(norms)) if len(norms) >= 20 else math.inf
        norm = float(torch.nn.utils.clip_grad_norm_(params, limit))
        if math.isfinite(norm):
            norms.append(norm)
            optimizer.step()
            if ema is not None:
                ema.update()
        else:
            say(f"[multi] step {step + 1}: gradient not finite, update skipped")
        schedule.step()
        step += 1

        if step % args.log_every == 0:
            losses = [w["loss"] for w in window]
            ratio = [w["date_temb_ratio"] for w in window if w["date_temb_ratio"] is not None]
            check = float(sum(p.detach().float().sum() for p in params))
            if world > 1:           # the weights must be the same on every rank
                t = torch.tensor([check], device="cpu" if same_gpu else device)
                gathered = [torch.zeros_like(t) for _ in range(world)]
                dist.all_gather(gathered, t)
                if max(abs(float(g) - check) for g in gathered) > 1e-3 * max(1.0, abs(check)):
                    raise RuntimeError(f"weights differ between ranks at step {step}")
            speed = (time.time() - started) / args.log_every
            say(f"[multi] step {step} loss {np.mean(losses):.4f} grad norm {norm:.3g} date/temb "
                f"{np.mean(ratio) if ratio else float('nan'):.3f} {speed:.2f} s/step "
                f"{torch.cuda.max_memory_allocated(device) / 2**30:.1f} GB")
            log({"step": step, "loss": float(np.mean(losses)), "grad_norm": norm, "s_per_step": speed,
                 "date_temb_ratio": float(np.mean(ratio)) if ratio else None})
            window, started = [], time.time()
        busy = time.time()
        if args.probe_every and step % args.probe_every == 0:
            run_probes()
        if lead and (step % args.save_every == 0 or step == args.steps):
            save(run_dir / "checkpoints" / f"step_{step:06d}.pt", full=False)
        if lead and (step % args.latest_every == 0 or step == args.steps):
            save(latest, full=True)
        if world > 1 and (args.probe_every and step % args.probe_every == 0):
            dist.barrier()
        started += time.time() - busy
    if world > 1:
        dist.destroy_process_group()


class _nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
