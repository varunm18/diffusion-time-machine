"""Training loop for date-conditioned SEVA (single GPU, one 8-view sample per step).

Each step: draw a sample from :class:`~timemachine.data.dataset.DatedViewsDataset`,
encode it (VAE latents, CLIP embedding, Plücker rays; all frozen), and take one
optimizer step on :func:`~timemachine.training.loss.training_loss`. Only the date
module and the LoRA adapters train (:mod:`timemachine.model.finetune`).

A run folder holds::

    config.json        the full TrainConfig
    log.jsonl          one line per logged step (loss, lr, grad norm, speed) and per probe
    probes.json        which views each render probe uses
    checkpoints/       step_XXXXXX.pt (trainable weights), latest.pt (+ optimizer, to resume)
    renders/step_XXXXXX/<probe>.jpg|json
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.sampling import SamplingConfig
from timemachine.model.finetune import FinetuneConfig, load_trainable_state_dict, prepare_for_finetuning, save_checkpoint
from timemachine.model.loading import SevaComponents, load_components
from timemachine.training.conditioning import sample_conditions
from timemachine.training.loss import LossConfig, training_loss
from timemachine.training.probes import DateGapProbe, RenderProbe, RenderProbeSpec


@dataclass(frozen=True)
class TrainConfig:
    dataset: str                                    # folder with views.parquet
    run_dir: str
    steps: int = 10_000
    lr_lora: float = 1e-4
    lr_date: float = 1e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    grad_accum: int = 1
    resolution: int = 576
    camera_scale: float = 2.0
    seed: int = 0
    num_workers: int = 4
    log_every: int = 25
    gap_every: int = 500
    render_every: int = 2500
    save_every: int = 2500
    render_steps: int = 25                          # sampler steps for the render probes
    gap_samples: int = 16
    sampling: SamplingConfig = SamplingConfig()
    loss: LossConfig = LossConfig()
    finetune: FinetuneConfig = FinetuneConfig()
    render_probes: tuple[RenderProbeSpec, ...] = field(default=())


def _infinite(loader: DataLoader, dataset: DatedViewsDataset, start_epoch: int):
    epoch = start_epoch
    while True:
        dataset.set_epoch(epoch)
        for sample in loader:
            yield epoch, sample
        epoch += 1


def _lr_lambda(warmup: int):
    return lambda step: min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0


class Trainer:
    def __init__(self, cfg: TrainConfig, components: SevaComponents | None = None):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        torch.manual_seed(cfg.seed)

        views = pd.read_parquet(Path(cfg.dataset) / "views.parquet")
        self.dataset = DatedViewsDataset(views, cfg.sampling, resolution=cfg.resolution,
                                         samples_per_epoch=1000, seed=cfg.seed, camera_scale=cfg.camera_scale)
        self.components = components or load_components(dated=True)
        self.model = self.components.model
        groups = prepare_for_finetuning(self.model, cfg.finetune)
        self.model.train()
        param_groups = [{"params": groups["date"], "lr": cfg.lr_date, "name": "date"}]
        if groups["lora"]:
            param_groups.append({"params": groups["lora"], "lr": cfg.lr_lora, "name": "lora"})
        self.params = [p for g in param_groups for p in g["params"]]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda(cfg.warmup_steps))
        self.sigmas = self.components.denoiser.sigmas
        self.step = 0
        self.epoch = 0
        n_train = sum(p.numel() for p in self.params)
        print(f"[train] {len(views)} views, {len(self.dataset.anchors)} anchors; trainable params "
              f"{n_train / 1e6:.1f}M (date {sum(p.numel() for p in groups['date']) / 1e6:.2f}M)", flush=True)

        self.gap_probe = None
        self.render_probes = [RenderProbe(self.dataset, spec) for spec in cfg.render_probes]
        (self.run_dir / "probes.json").write_text(json.dumps([p.describe() for p in self.render_probes], indent=2))

    # ------------------------------------------------------------------ state
    def save(self, tag: str | None = None) -> Path:
        """``step_XXXXXX.pt``: trainable weights only. ``latest.pt`` (tag) also has the optimizer, to resume."""
        path = self.run_dir / "checkpoints" / (tag or f"step_{self.step:06d}.pt")
        extra = {"epoch": self.epoch}
        if tag is not None:
            extra.update(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict())
        save_checkpoint(path, self.model, self.cfg.finetune, self.step, extra=extra)
        return path

    def resume(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        load_trainable_state_dict(self.model, ckpt["trainable"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step, self.epoch = ckpt["step"], ckpt.get("epoch", 0) + 1
        print(f"[train] resumed from {path} at step {self.step}", flush=True)

    def log(self, record: dict) -> None:
        with open(self.run_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ probes
    def run_gap_probe(self) -> None:
        if self.gap_probe is None:
            self.gap_probe = DateGapProbe(self.components, self.dataset, self.cfg.gap_samples)
        self.model.eval()
        result = self.gap_probe(self.model, self.sigmas)
        self.model.train()
        self.log({"step": self.step, "probe": "date_gap", **result})
        print(f"[gap] step {self.step}: " + ", ".join(f"{k} {v:.4f}" for k, v in result.items()), flush=True)

    def run_render_probes(self) -> None:
        if not self.render_probes:
            return
        self.model.eval()
        out_dir = self.run_dir / "renders" / f"step_{self.step:06d}"
        for probe in self.render_probes:
            t0 = time.time()
            result = probe(self.components, out_dir, num_steps=self.cfg.render_steps, seed=self.cfg.seed)
            self.log({"step": self.step, "probe": f"render:{probe.spec.name}", **result})
            summary = "; ".join(f"{m['year']}: sat {m['saturation']:.2f} diff {m.get('mean_abs_diff_vs_real_date', 0):.3f}"
                                for m in result["per_year"])
            print(f"[render] step {self.step} {probe.spec.name} ({time.time() - t0:.0f}s): {summary}", flush=True)
        self.model.train()

    # ------------------------------------------------------------------ loop
    def train(self) -> None:
        cfg = self.cfg
        loader = DataLoader(self.dataset, batch_size=None, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True, prefetch_factor=4 if cfg.num_workers else None)
        stream = _infinite(loader, self.dataset, self.epoch)
        gen = torch.Generator().manual_seed(cfg.seed + self.step)
        if self.step == 0:
            self.run_gap_probe()
            self.run_render_probes()
        window, t_last = [], time.time()
        while self.step < cfg.steps:
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum):
                self.epoch, sample = next(stream)
                latents, cond, uncond = sample_conditions(self.components, sample, camera_scale=cfg.camera_scale)
                with torch.autocast("cuda"):
                    loss, info = training_loss(self.model, self.sigmas, latents, cond, uncond,
                                               sample["date"], cfg.loss, gen)
                (loss / cfg.grad_accum).backward()
                window.append({"loss": float(loss), **info})
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip))
            window[-1]["grad_norm"] = grad_norm
            if not math.isfinite(grad_norm):
                print(f"[train] step {self.step}: non-finite grad norm, skipping update", flush=True)
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.step % cfg.log_every == 0:
                elapsed = time.time() - t_last
                losses = np.array([w["loss"] for w in window])
                cond_losses = [w["loss"] for w in window if not w["uncond"]]
                norms = np.array([w["grad_norm"] for w in window if "grad_norm" in w])
                record = {"step": self.step, "epoch": self.epoch, "loss": float(losses.mean()),
                          "loss_cond": float(np.mean(cond_losses)) if cond_losses else None,
                          "eps_mse": float(np.mean([w["eps_mse"] for w in window])),
                          "grad_norm_median": float(np.median(norms)), "grad_norm_max": float(norms.max()),
                          "lr_lora": self.optimizer.param_groups[-1]["lr"],
                          "sec_per_step": elapsed / cfg.log_every,
                          "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
                self.log(record)
                print(f"[train] step {self.step} loss {record['loss']:.4f} eps-mse {record['eps_mse']:.4f} "
                      f"gnorm median {record['grad_norm_median']:.2g} max {record['grad_norm_max']:.2g} "
                      f"{record['sec_per_step']:.2f}s/step mem {record['max_mem_gb']:.1f}GB", flush=True)
                window, t_last = [], time.time()
            t_busy = time.time()   # probe/checkpoint time is kept out of the speed measurement
            if self.step % cfg.gap_every == 0:
                self.run_gap_probe()
            if self.step % cfg.render_every == 0 or self.step == cfg.steps:
                self.run_render_probes()
            if self.step % cfg.save_every == 0 or self.step == cfg.steps:
                self.save()
                self.save("latest.pt")
            t_last += time.time() - t_busy
