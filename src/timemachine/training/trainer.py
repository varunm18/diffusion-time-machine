"""Training loop for date-conditioned SEVA (single GPU, one 8-view sample per step).

The data is one scene's view table or several (``TrainConfig.dataset``); with several,
each step's sample comes from one scene (see :class:`DatedViewsDataset`).

Each step: draw a sample from :class:`~timemachine.data.dataset.DatedViewsDataset`,
encode it (VAE latents, CLIP embedding, Plücker rays; all frozen), and take one
optimizer step on :func:`~timemachine.training.loss.training_loss`. Only the date
module and the LoRA adapters train (:mod:`timemachine.model.finetune`).

A run folder holds::

    config.json        the full TrainConfig
    log.jsonl          one line per logged step (loss, lr, grad norm, speed) and per probe
    probes.json        which views each render probe uses
    heldout.json       views held out of training, and the held-out probe's pairs (if any)
    heldout_baseline.json, heldout_losses.jsonl   the held-out probe's per-sample losses (step 0 / each run)
    checkpoints/       step_XXXXXX.pt (trainable weights), latest.pt (+ optimizer, to resume)
    renders/step_XXXXXX/<probe>.jpg|json
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from timemachine.data.appearance import dataset_appearance_stats
from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.holdout import HoldoutConfig, heldout_rows
from timemachine.data.sampling import SamplingConfig
from timemachine.model.appearance_film import fit_era_directions
from timemachine.model.diagnostics import finetune_stats
from timemachine.model.finetune import FinetuneConfig, load_trainable_state_dict, prepare_for_finetuning, save_checkpoint
from timemachine.model.loading import SevaComponents, load_components
from timemachine.training.conditioning import sample_conditions
from timemachine.training.loss import AUTOCAST_DTYPES, LossConfig, autocast, training_loss
from timemachine.training.probes import DateGapProbe, HeldoutProbe, RenderProbe, RenderProbeSpec


@dataclass(frozen=True)
class TrainConfig:
    dataset: str | tuple[str, ...]                  # per scene: a folder with views.parquet, or a .parquet file
    run_dir: str
    steps: int = 10_000
    lr_lora: float = 1e-5                           # 1e-4 destroyed SEVA within ~1k steps (docs/findings.md)
    lr_date: float = 1e-4                           # date vector added to the timestep embedding
    lr_film: float = 1e-5                           # date FiLM (per-block layers act like LoRA: keep it low)
    lr_appearance: float = 1e-5                     # appearance FiLM (same shape as date FiLM: keep it low)
    warmup_steps: int = 200
    lr_schedule: str = "constant"                   # after warmup: "constant", or "cosine" decay to lr_min_ratio
    lr_min_ratio: float = 0.05
    # One sample per step, so a fixed clip at 1.0 (the old default) rescaled almost every update
    # to norm 1 and cancelled the per-noise-level loss weighting (docs/findings.md, code audit).
    # Now clipping only catches spikes: above grad_clip_factor x the median of recent norms.
    grad_clip: float | None = None                  # absolute clip (off by default)
    grad_clip_factor: float | None = 10.0           # spike guard, relative to the running median
    grad_accum: int = 1
    autocast: str = "fp32"                          # training/probe precision, a key of AUTOCAST_DTYPES
    resolution: int = 576
    camera_scale: float = 2.0
    group_weight_power: float = 0.5                 # scene sampling weight = training views ** this
    appearance_stats_samples: int = 64              # samples used to standardize the appearance encoder
    era_stats_samples: int = 256                    # with appearance_remove_era: samples for normalization + era fit
    seed: int = 0
    num_workers: int = 4
    log_every: int = 25
    gap_every: int = 500
    render_every: int = 2500
    save_every: int = 2500                          # step_XXXXXX.pt (kept)
    latest_every: int = 500                         # latest.pt (overwritten; to resume after preemption)
    render_steps: int = 25                          # sampler steps for the render probes
    render_uncond_dates: str = "same"               # render probes: unconditional CFG branch dated "same"/"unknown"
    render_frames: int | None = 21                  # render probes: pad to SEVA's 21 frames (None: unpadded, pre-audit)
    gap_samples: int = 16
    heldout_every: int = 1000                       # held-out probe interval (runs with a hold-out only)
    heldout_pairs: int = 48
    abort_loss_ratio: float | None = 3.0            # stop if the probe's denoising loss exceeds this x its first value
    abort_x0_ratio: float | None = 1.5              # ... or its high-noise denoised-latent error (colour drift)
    sampling: SamplingConfig = SamplingConfig()
    loss: LossConfig = LossConfig()
    finetune: FinetuneConfig = FinetuneConfig()
    holdout: HoldoutConfig = HoldoutConfig()
    render_probes: tuple[RenderProbeSpec, ...] = field(default=())


def dataset_dirs(dataset: str | tuple[str, ...] | list[str]) -> list[Path]:
    return [Path(dataset)] if isinstance(dataset, (str, Path)) else [Path(d) for d in dataset]


def load_views(dataset: str | tuple[str, ...] | list[str]) -> pd.DataFrame:
    """The view tables of one or more scenes, concatenated. Each entry is a dataset folder
    (its ``views.parquet``) or a view-table file (e.g. a cleaned ``views_clean.parquet``)."""
    tables = [pd.read_parquet(d if d.suffix == ".parquet" else d / "views.parquet") for d in dataset_dirs(dataset)]
    return pd.concat(tables, ignore_index=True)


def run_heldout_rows(dataset: DatedViewsDataset, cfg: HoldoutConfig, render_probes: list[RenderProbe]) -> np.ndarray:
    """The view rows a run holds out: random views per scene and era, the render probes'
    targets (if ``cfg.probe_targets``) and whole held-out scenes. Deterministic given the
    dataset, so evaluation scripts recover a run's hold-out exactly."""
    extra = [p.rows[p.spec.n_inputs:] for p in render_probes] if cfg.probe_targets else []
    extra.append(np.flatnonzero(dataset.views["scene_id"].isin(cfg.scenes).to_numpy()))
    return heldout_rows(dataset.dates, cfg, np.concatenate(extra).astype(int), groups=dataset.group)


def _infinite(loader: DataLoader, dataset: DatedViewsDataset, start_epoch: int):
    epoch = start_epoch
    while True:
        dataset.set_epoch(epoch)
        for sample in loader:
            yield epoch, sample
        epoch += 1


def _lr_lambda(warmup: int, schedule: str = "constant", total: int = 1, min_ratio: float = 0.0):
    """Linear warmup, then constant or cosine decay to ``min_ratio`` at step ``total``."""
    if schedule not in ("constant", "cosine"):
        raise ValueError("lr_schedule must be 'constant' or 'cosine'")

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


class Trainer:
    def __init__(self, cfg: TrainConfig, components: SevaComponents | None = None):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        torch.manual_seed(cfg.seed)

        views = load_views(cfg.dataset)
        # Appearance statistics are computed per view only when a model asks for them.
        self.with_appearance = cfg.finetune.appearance_injection != "none"
        self.dataset = DatedViewsDataset(views, cfg.sampling, resolution=cfg.resolution, samples_per_epoch=1000,
                                         seed=cfg.seed, camera_scale=cfg.camera_scale,
                                         group_weight_power=cfg.group_weight_power,
                                         with_appearance=self.with_appearance)
        self.components = components or load_components(dated=True)
        self.model = self.components.model
        groups = prepare_for_finetuning(self.model, cfg.finetune)
        self.model.train()
        if self.with_appearance:
            # The raw statistics are on very different scales; standardize the encoder on
            # the data this run will actually see. The buffers travel with the checkpoint.
            encoder = self.model.date_module.appearance_encoder
            if cfg.finetune.appearance_remove_era:
                self.fit_era_residual(encoder)
            else:
                encoder.fit_normalization(
                    dataset_appearance_stats(self.dataset, n_samples=cfg.appearance_stats_samples, seed=cfg.seed))
        lrs = {"date": cfg.lr_date, "film": cfg.lr_film, "lora": cfg.lr_lora,
               "appearance": cfg.lr_appearance}
        param_groups = [{"params": groups[name], "lr": lr, "name": name} for name, lr in lrs.items() if groups[name]]
        self.params = [p for g in param_groups for p in g["params"]]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, _lr_lambda(cfg.warmup_steps, cfg.lr_schedule, cfg.steps, cfg.lr_min_ratio))
        self.sigmas = self.components.denoiser.sigmas
        self.step = 0
        self.epoch = 0
        self.grad_norm_history: deque[float] = deque(maxlen=200)
        self.gap_probe = None
        self.heldout_probe = None
        self.probe_baseline: dict[str, float] = {}      # first value of each watched probe loss (step 0 if fresh)
        # Render probes pick their views from the full table, before any hold-out (so they can be held out).
        self.render_probes = [RenderProbe(self.dataset, spec, date_views=cfg.loss.date_views,
                                          uncond_dates=cfg.render_uncond_dates, frames=cfg.render_frames)
                              for spec in cfg.render_probes]
        (self.run_dir / "probes.json").write_text(json.dumps([p.describe() for p in self.render_probes], indent=2))
        if cfg.holdout.active:
            self.hold_out_views()

        n_train = sum(p.numel() for p in self.params)
        sizes = ", ".join(f"{g['name']} {sum(p.numel() for p in g['params']) / 1e6:.2f}M" for g in param_groups)
        print(f"[train] {len(views)} views ({int(self.dataset.heldout.sum())} held out), "
              f"{len(self.dataset.anchors)} anchors; trainable params {n_train / 1e6:.1f}M ({sizes})", flush=True)
        if self.dataset.n_groups > 1:
            self.log_scenes()

    def log_scenes(self) -> None:
        """Per scene: views, held-out views, anchors (old ones) and sampling weight."""
        ds, weights = self.dataset, dict(zip(self.dataset.sample_groups, self.dataset.group_weights))
        anchors = set(ds.anchors.tolist())
        scenes = []
        for g, rows in enumerate(ds.group_rows):
            a = np.array([r for r in rows if r in anchors], dtype=int)
            scenes.append({"scene_id": int(ds.views.scene_id[rows[0]]), "model": int(ds.views.model[rows[0]]),
                           "views": len(rows), "heldout": int(ds.heldout[rows].sum()), "anchors": len(a),
                           "old_anchors": int((ds.dates[a] < ds.config.old_before).sum()),
                           "weight": round(float(weights.get(g, 0.0)), 4)})
            print(f"[train]   scene {scenes[-1]}", flush=True)
        (self.run_dir / "scenes.json").write_text(json.dumps(scenes, indent=1))

    def fit_era_residual(self, encoder) -> dict:
        """Standardize the appearance encoder and install the era subspace, from one draw.

        One draw for both, because the era directions are only in the encoder's basis
        on the sample its normalization was fitted on. More samples than plain
        normalization uses: the stopping rule is a held-out AUC, and with the default
        64 it would rest on a few dozen old views. Writes ``era_directions.json``.
        """
        cfg = self.cfg
        stats, years = dataset_appearance_stats(
            self.dataset, n_samples=max(cfg.appearance_stats_samples, cfg.era_stats_samples),
            seed=cfg.seed, with_dates=True)
        encoder.fit_normalization(stats)
        report = fit_era_directions(encoder, stats, years.numpy(), seed=cfg.seed)
        (self.run_dir / "era_directions.json").write_text(json.dumps(
            {**report, "directions": encoder.era_directions.cpu().tolist()}, indent=1))
        print(f"[train] era residual: removed {report['dims_removed']} of {report['n_features']} "
              f"appearance dimensions ({report['n_old']} old of {report['n_rows']} views); era AUC "
              f"{report['era_auc_full']:.3f} -> {report['era_auc_residual']:.3f}, "
              f"{report['variance_kept']:.0%} of variance kept", flush=True)
        if not report["era_removed"]:
            print(f"[train] WARNING: era is still readable from the residual appearance "
                  f"(AUC {report['era_auc_residual']:.3f} >= {report['target_auc']}) after "
                  f"{report['dims_removed']} removals; this run does not test an era-free "
                  f"appearance pathway", flush=True)
        return report

    def hold_out_views(self) -> None:
        """Remove the hold-out from training: random views per scene and era, the render probes'
        targets (if asked) and whole held-out scenes."""
        rows = run_heldout_rows(self.dataset, self.cfg.holdout, self.render_probes)
        self.dataset.hold_out(rows)
        views = self.dataset.views
        self.heldout_info = {"views": [{"file_key": views.file_key[r], "year": round(float(views.date_mid[r]), 2)}
                                       for r in rows]}
        self._write_heldout_info()

    def _write_heldout_info(self) -> None:
        (self.run_dir / "heldout.json").write_text(json.dumps(self.heldout_info, indent=1))

    def clip_limit(self) -> float:
        """Gradient-norm limit for this step: the absolute clip and/or the spike guard (inf = none)."""
        limit = self.cfg.grad_clip or math.inf
        if self.cfg.grad_clip_factor and len(self.grad_norm_history) >= 50:
            limit = min(limit, self.cfg.grad_clip_factor * float(np.median(self.grad_norm_history)))
        return limit

    # ------------------------------------------------------------------ state
    def save(self, tag: str | None = None) -> Path:
        """``step_XXXXXX.pt``: trainable weights only. ``latest.pt`` (tag) also has the optimizer, to resume."""
        path = self.run_dir / "checkpoints" / (tag or f"step_{self.step:06d}.pt")
        extra = {"epoch": self.epoch, "probe_baseline": self.probe_baseline, "autocast": self.cfg.autocast}
        if tag is not None:
            extra.update(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                         grad_norm_history=list(self.grad_norm_history))
        save_checkpoint(path, self.model, self.cfg.finetune, self.step, extra=extra)
        return path

    def resume(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        trained_at = ckpt.get("autocast", "bf16")          # checkpoints before 2026-11-19 are all bf16 (or fp16)
        if trained_at != self.cfg.autocast:
            raise ValueError(f"{path} was trained at {trained_at}; resuming at {self.cfg.autocast} would mix "
                             f"precisions in one run. Pass --autocast {trained_at}, or start a new run.")
        load_trainable_state_dict(self.model, ckpt["trainable"])
        self.grad_norm_history.extend(ckpt.get("grad_norm_history", []))
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step, self.epoch = ckpt["step"], ckpt.get("epoch", 0) + 1
        baseline = ckpt.get("probe_baseline") or {}
        self.probe_baseline = baseline if isinstance(baseline, dict) else {"loss_true": baseline}  # older runs
        print(f"[train] resumed from {path} at step {self.step}", flush=True)

    def log(self, record: dict) -> None:
        with open(self.run_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ probes
    def run_gap_probe(self) -> str | None:
        """Date-gap probe + weight statistics. Returns a message if the model has diverged.

        Diverged: the denoising loss rose above ``abort_loss_ratio`` x its first value, or the
        high-noise denoised-latent error above ``abort_x0_ratio`` x (the renders' colours and
        layout break long before the denoising loss moves; docs/findings.md).
        """
        if self.gap_probe is None:
            self.gap_probe = DateGapProbe(self.components, self.dataset, self.cfg.gap_samples,
                                          autocast_dtype=AUTOCAST_DTYPES[self.cfg.autocast],
                                          date_views=self.cfg.loss.date_views)
        self.model.eval()
        result = self.gap_probe(self.model, self.sigmas)
        stats = finetune_stats(self.model)
        self.model.train()
        self.log({"step": self.step, "probe": "date_gap", **result})
        self.log({"step": self.step, "probe": "weights", **stats})
        print(f"[gap] step {self.step}: " + ", ".join(f"{k} {v:.4f}" for k, v in result.items()), flush=True)
        print(f"[weights] step {self.step}: timestep rms {stats['timestep_rms']:.3f}, date rms 2015 "
              f"{stats['date_rms_2015']:.3f} / unknown {stats['date_rms_unknown']:.3f} / spread "
              f"{stats['date_rms_spread']:.3f}, lora max rel {stats.get('lora_rel_max', 0):.4f}"
              + (f", film scale {stats['film_scale_rms_max']:.3f} / shift {stats['film_shift_rms_max']:.3f}"
                 f" / spread {stats['film_spread_max']:.3f}" if "film_scale_rms_max" in stats else ""), flush=True)
        limits = {"loss_true": self.cfg.abort_loss_ratio, "x0_mse_high_noise": self.cfg.abort_x0_ratio}
        for key, limit in limits.items():
            if key not in result:
                continue
            first = self.probe_baseline.setdefault(key, result[key])
            if limit is not None and result[key] > limit * first:
                return f"{key} {result[key]:.4f} is above {limit}x its first value {first:.4f}"
        return None

    def run_heldout_probe(self) -> None:
        if not self.cfg.holdout.active:
            return
        if self.heldout_probe is None:
            self.heldout_probe = HeldoutProbe(self.components, self.dataset, self.run_dir / "heldout_baseline.json",
                                              self.cfg.heldout_pairs, autocast_dtype=AUTOCAST_DTYPES[self.cfg.autocast],
                                              date_views=self.cfg.loss.date_views, era_edges=self.cfg.holdout.era_edges)
            self.heldout_info["probe_pairs"] = self.heldout_probe.describe()
            self._write_heldout_info()
        self.model.eval()
        t0 = time.time()
        result = self.heldout_probe(self.model, self.sigmas)
        self.model.train()
        self.log({"step": self.step, "probe": "heldout", **result})
        with open(self.run_dir / "heldout_losses.jsonl", "a") as f:    # per-sample losses, pair order of heldout.json
            f.write(json.dumps({"step": self.step, **self.heldout_probe.last_per_sample}) + "\n")
        print(f"[heldout] step {self.step} ({time.time() - t0:.0f}s, {result['pairs']:.0f} pairs, "
              f"{result['old_pairs']:.0f} old): change held-out {result['heldout_change']:+.4f} / trained "
              f"{result['train_change']:+.4f}, memorisation gap {result['memorisation_gap']:+.4f}; dates help "
              f"held-out {result['heldout_rel_unknown']:+.4f} / trained {result['train_rel_unknown']:+.4f}", flush=True)

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
            self.run_heldout_probe()
            self.run_render_probes()
        window, t_last = [], time.time()
        while self.step < cfg.steps:
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum):
                self.epoch, sample = next(stream)
                latents, cond, uncond = sample_conditions(self.components, sample, camera_scale=cfg.camera_scale)
                with autocast(AUTOCAST_DTYPES[cfg.autocast]):
                    loss, info = training_loss(self.model, self.sigmas, latents, cond, uncond,
                                               sample["date"], cfg.loss, gen, is_input=sample["is_input"],
                                               appearance=sample.get("appearance"))
                (loss / cfg.grad_accum).backward()
                window.append({"loss": float(loss), **info})
            limit = self.clip_limit()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, limit))
            window[-1]["grad_norm"] = grad_norm
            window[-1]["clipped"] = grad_norm > limit
            if math.isfinite(grad_norm):
                self.grad_norm_history.append(grad_norm)
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
                          "clipped_fraction": float(np.mean([w["clipped"] for w in window if "clipped" in w])),
                          **{f"lr_{g['name']}": g["lr"] for g in self.optimizer.param_groups},
                          "sec_per_step": elapsed / cfg.log_every,
                          "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
                ranked = [w for w in window if "rank_log_ratio" in w]
                if ranked:          # date discrimination: log(loss with wrong-era date / loss with true date)
                    record.update(rank_log_ratio=float(np.mean([w["rank_log_ratio"] for w in ranked])),
                                  rank_loss=float(np.mean([w["rank_loss"] for w in ranked])),
                                  rank_steps=len(ranked))
                self.log(record)
                print(f"[train] step {self.step} loss {record['loss']:.4f} eps-mse {record['eps_mse']:.4f} "
                      f"gnorm median {record['grad_norm_median']:.2g} max {record['grad_norm_max']:.2g} "
                      f"clipped {record['clipped_fraction']:.0%} "
                      f"{record['sec_per_step']:.2f}s/step mem {record['max_mem_gb']:.1f}GB", flush=True)
                window, t_last = [], time.time()
            t_busy = time.time()   # probe/checkpoint time is kept out of the speed measurement
            if self.step % cfg.gap_every == 0 and (diverged := self.run_gap_probe()):
                print(f"[train] step {self.step}: {diverged}; the model has diverged, stopping", flush=True)
                self.log({"step": self.step, "event": "diverged", "reason": diverged})
                self.save()
                return
            if self.step % cfg.heldout_every == 0 or self.step == cfg.steps:
                self.run_heldout_probe()
            if self.step % cfg.render_every == 0 or self.step == cfg.steps:
                self.run_render_probes()
            if self.step % cfg.save_every == 0 or self.step == cfg.steps:
                self.save()
            if self.step % cfg.latest_every == 0 or self.step % cfg.save_every == 0 or self.step == cfg.steps:
                self.save("latest.pt")
            t_last += time.time() - t_busy
