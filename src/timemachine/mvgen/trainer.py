"""Training loop for date-conditioned MVGenMaster on one scene (single GPU).

Only the adapters train: the per-view date embedding and, optionally, LoRA
(:func:`timemachine.mvgen.model.prepare_for_finetuning`); the released model stays frozen.

**One optimiser step** is ``grad_accum`` samples. Each sample is one view set (references and
targets); its loss (:func:`timemachine.mvgen.step.training_loss`) is divided by ``grad_accum`` and
back-propagated at once, so only one sample's graph is alive at a time. Then: the spike guard
(the gradient is clipped only when its norm exceeds ``grad_clip_factor`` times the median of
recent norms; there is no fixed clip), AdamW, the learning-rate schedule, and the EMA.

**Samples** come from the scene's co-visibility sampler (:class:`timemachine.mvgen.data.MVGenScene`),
or, with ``pinned_rows``, from exactly the given view sets, cycled in order: with two pinned sets and
``grad_accum = 2`` every step sees both. Pinned runs are the overfitting tests of the plan (gates 1
and 2 of ``docs/mvgenmaster_time_plan.md``). Samples are numbered from the start of the run, and
everything random in a step is drawn from one generator whose state is saved, so a resumed run
continues exactly where it stopped.

**Scene scale** is ``scale``, the rule by which ``MVGenScene`` scales camera translations and depth:
``"scene"`` (one factor for the whole scene, from the extent of all its cameras, as in the authors'
training), ``"sample"`` (per sample, as in MVGenMaster's demo), or a constant factor. Training
samples, probes and renders all come from the one scene object, so they share the rule; the factor
is printed at start-up and recorded in ``config.json`` and in every checkpoint.

**Mode and memory.** MVGenMaster's blocks only use gradient checkpointing when the UNet is in
``train()`` mode (they test ``self.training``); in ``eval()`` mode the setting is silently ignored
and the activations of every block and view are kept for ``backward()``. So with
``finetune.grad_checkpointing`` the UNet is in ``train()`` mode during training steps, where only
the blocks' inputs and outputs are kept and each block is run a second time in ``backward()``; without
it the UNet stays in ``eval()``. ``train()`` must then change nothing else: the released UNet's
dropout rates are zero and it has no batch normalisation, and :func:`mode_dependent_modules` checks
this before the first step. Probes and renders always run in ``eval()`` mode.

**Precision** is ``loss.precision``: bf16 autocast or fp32 for the UNet's forward pass, fp32 weights
and loss either way (``timemachine.mvgen.step``). The trainer opens no autocast context of its own.

**Starting and resuming.** A run folder belongs to one run. A launch without ``resume`` is refused if
the folder already holds a log or checkpoints: its probes would be compared with another run's first
evaluation. A run that has a log must therefore have something to resume from, so a fresh run writes
its first ``latest.pt`` at step 0, right after the first evaluation of its probes (which starts the
log) and before the first optimiser step. Resuming from it gives the run that an uninterrupted launch
gives, without evaluating step 0 a second time. Only a job that dies during that first evaluation
leaves a log and no checkpoint: nothing has been trained then, and the folder is to be deleted.

A launch with ``resume`` continues the run saved in a ``latest.pt`` and must have that
run's configuration: the optimiser and the schedule come from the checkpoint, so a changed learning
rate would be ignored without a word, and any other change (dropout rates, the domain label, the
cross-date rule, the view sets) would take effect mid-run against a baseline measured under the old
settings. Only ``RESUME_MAY_CHANGE`` may differ: the number of steps, the logging, probe, render and
checkpoint intervals, the data-loading processes and what the renders show. Both checks run before
the model is loaded and before anything is written, so a refused launch leaves the folder as it was.

A resumed launch must also find the run's dataset. The configuration names a folder, not what is in
it: under ``scale="scene"`` the scale factor comes from the folder's ``scene.json``, and which views
are drawn from its view table. Every checkpoint therefore holds the scene-scale factor and a
fingerprint of the folder (``MVGenScene.fingerprint``: the view table's rows with their ``usable``
flags, cameras and dates, the views with reliable depth, the co-visibility). A launch whose scene
gives another factor or another fingerprint is refused, with the name of what differs, and so is a
checkpoint without a fingerprint. This check needs the scene, so the model is loaded by then, but
nothing has been written.

**EMA.** With ``ema_decay > 0`` an exponential moving average of the trainable weights is kept
(:class:`WeightEMA`); the probes measure it and the checkpoints' ``trainable`` entry holds it.

**Probes** (:mod:`timemachine.mvgen.probes`) run under :meth:`Trainer.evaluation`: without gradients,
in ``eval()`` mode, on the EMA weights. They draw from their own generators and the global random
state is put back afterwards, so a run with probes trains exactly as the same run without them.

**Size of the date embedding.** Nothing bounds the date embedding, and it is added to the vector
that carries the timestep to every ResBlock. Every log line therefore has ``date_rms``, its root
mean square in the logged training steps, and ``date_temb_ratio``, its ratio to that of the timestep
embedding (``timemachine.mvgen.step.date_embedding_size``); the date-loss probe reports the same for
the weights it evaluates. Read a gate's result only while the ratio is well below 1.

A run folder holds::

    config.json                 the TrainConfig, and scale_factor: the scene-scale factor (null: per sample)
    log.jsonl                   one line per logged step and per probe evaluation
    pinned.json                 the pinned view sets (pinned runs): rows, file keys, dates, periods
    heldout.json                the views held out of training (runs with a hold-out)
    probes.json                 the probes' view sets and the years used for swapped dates
    probes/<name>_baseline.json, <name>_losses.jsonl    date-loss probe: first evaluation, and all raw losses
    renders/step_XXXXXX/<probe>.jpg, .json
    checkpoints/step_XXXXXX.pt  the trainable weights to evaluate (the EMA, if kept)
    checkpoints/latest.pt       the same, plus what resuming needs (raw weights, optimiser, random state);
                                first written at step 0
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from timemachine.data.holdout import HoldoutConfig, era_index, heldout_rows
from timemachine.mvgen.data import SCALE_RULES, MVGenScene, ViewSamplingConfig, checked_scale, period_names
from timemachine.mvgen.date import UNCOND_DATES
from timemachine.mvgen.model import (
    DEFAULT_MODEL_DIR, FinetuneConfig, MVGen, load_mvgen, load_trainable_state_dict, prepare_for_finetuning,
    save_checkpoint, trainable_state_dict,
)
from timemachine.mvgen.probes import (
    DEFAULT_TIMESTEPS, DateLossProbe, RenderProbe, check_view_sets, describe_views, heldout_view_sets,
    parse_render_date, representative_years, scene_view_sets,
)
from timemachine.mvgen.step import LossConfig, training_loss

LR_SCHEDULES = ("constant", "cosine")
GRAD_NORM_HISTORY = 200         # recent gradient norms the spike guard takes its median over
SPIKE_GUARD_MIN_HISTORY = 50    # ... and how many it needs before it clips anything
EMA_WARMUP = 10                 # the EMA's decay at update n is at most (1 + n) / (EMA_WARMUP + n)
# The TrainConfig fields that a resumed launch may change (see "Starting and resuming" in the module docstring).
# render_rows is one of them only while the renders' targets are not held out of training (holdout.probe_targets).
RESUME_MAY_CHANGE = frozenset({
    "steps", "log_every", "probe_every", "render_every", "save_every", "latest_every", "num_workers",
    "render_rows", "render_dates", "render_seeds", "render_steps", "render_guidance", "render_uncond_dates"})


def parse_scale(value: str | float) -> str | float:
    """A scene-scale setting from the command line, a config or a checkpoint, for ``MVGenScene(scale=...)``:
    one of ``timemachine.mvgen.data.SCALE_RULES``, or a positive factor (a number, also given as text)."""
    if isinstance(value, str) and value not in SCALE_RULES:
        try:
            value = float(value)
        except ValueError:
            pass                        # not a number either: checked_scale says what a scale can be
    return checked_scale(value)


@dataclass(frozen=True)
class TrainConfig:
    """A training run. The intervals count optimiser steps; 0 switches a probe or a periodic checkpoint off."""

    dataset: str                                    # dataset folder (the layout is in timemachine.mvgen.data)
    run_dir: str
    model_dir: str = DEFAULT_MODEL_DIR              # the released model, relative to MVGenMaster's checkout
    resolution: int = 512                           # side of the square crops; the depth maps' size
    scale: str | float = "scene"                    # scene-scale rule: one of data.SCALE_RULES, or a constant factor
    steps: int = 10_000                             # optimiser steps
    lr_date: float = 1e-4
    lr_lora: float = 1e-4
    warmup_steps: int = 100                         # linear warm-up of both learning rates
    lr_schedule: str = "cosine"                     # after the warm-up, one of LR_SCHEDULES
    lr_min_ratio: float = 0.05                      # "cosine" decays to this share of the learning rate
    grad_accum: int = 4                             # samples per optimiser step
    grad_clip_factor: float | None = 10.0           # spike guard: clip above this x the median recent norm (None: off)
    ema_decay: float = 0.999                        # EMA of the trainable weights (0: off)
    seed: int = 0
    num_workers: int = 4                            # data-loading processes (not used by pinned runs)
    log_every: int = 25
    probe_every: int = 500                          # date-loss probes; also at step 0 and at the last step
    render_every: int = 2500                        # render probes; also at step 0 and at the last step
    save_every: int = 2500                          # step_XXXXXX.pt (kept); the last step is always saved
    latest_every: int = 500                         # latest.pt (overwritten; also at step 0): where a job resumes
    probe_samples: int = 8                          # view sets per date-loss probe (pinned runs: the pinned sets)
    probe_timesteps: tuple[int, ...] = DEFAULT_TIMESTEPS
    swap_years: tuple[float, ...] | None = None     # one year per period for swapped dates (None: from the data)
    render_rows: tuple[tuple[int, ...], ...] = ()   # view sets to render (pinned runs: default, the pinned sets)
    render_dates: tuple[str | float, ...] = ("true", "swapped")    # years, or words of probes.RENDER_DATE_WORDS
    render_seeds: tuple[int, ...] = (0, 1)
    render_steps: int = 50                          # sampler steps
    render_guidance: float = 2.0
    render_uncond_dates: str = "same"               # dates of the unconditional guidance half: date.UNCOND_DATES
    sampling: ViewSamplingConfig = ViewSamplingConfig()
    loss: LossConfig = LossConfig()
    finetune: FinetuneConfig = FinetuneConfig()
    holdout: HoldoutConfig = HoldoutConfig()        # its era_edges are the strata; use sampling.period_edges
    pinned_rows: tuple[tuple[int, ...], ...] = ()   # train on exactly these view sets (references first), cycled

    def __post_init__(self):
        for name in ("pinned_rows", "render_rows"):
            object.__setattr__(self, name, tuple(tuple(int(row) for row in rows) for rows in getattr(self, name)))
            check_view_sets(getattr(self, name), self.sampling.n_ref)
        object.__setattr__(self, "scale", parse_scale(self.scale))
        object.__setattr__(self, "probe_timesteps", tuple(int(t) for t in self.probe_timesteps))
        object.__setattr__(self, "render_seeds", tuple(int(seed) for seed in self.render_seeds))
        object.__setattr__(self, "render_dates", tuple(parse_render_date(date) for date in self.render_dates))
        if self.swap_years is not None:
            object.__setattr__(self, "swap_years", tuple(float(year) for year in self.swap_years))
        if self.steps < 1 or self.grad_accum < 1 or self.resolution < 1 or self.log_every < 1:
            raise ValueError("steps, grad_accum, resolution and log_every must be at least 1")
        if self.lr_date <= 0 or self.lr_lora <= 0:
            raise ValueError("lr_date and lr_lora must be positive")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}")
        if self.warmup_steps < 0 or not 0.0 <= self.lr_min_ratio <= 1.0:
            raise ValueError("warmup_steps must not be negative and lr_min_ratio must be in [0, 1]")
        if self.grad_clip_factor is not None and self.grad_clip_factor <= 1.0:
            raise ValueError("grad_clip_factor must be above 1 (it multiplies the median gradient norm), or None")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1); 0 switches the EMA off")
        intervals = ("probe_every", "render_every", "save_every", "latest_every", "num_workers")
        if any(getattr(self, name) < 0 for name in intervals):
            raise ValueError(f"{', '.join(intervals)} must not be negative")
        if self.probe_samples < 1 or not self.probe_timesteps or min(self.probe_timesteps) < 0:
            raise ValueError("probe_samples must be at least 1 and probe_timesteps non-negative and not empty")
        if not self.render_dates or not self.render_seeds or self.render_steps < 1 or self.render_guidance < 0:
            raise ValueError("renders need a date, a seed, at least one sampler step and a non-negative guidance")
        if self.render_uncond_dates not in UNCOND_DATES:
            raise ValueError(f"render_uncond_dates must be one of {UNCOND_DATES}")
        edges = self.sampling.period_edges
        if self.swap_years is not None and era_index(self.swap_years, edges).tolist() != list(range(len(edges) + 1)):
            raise ValueError(f"swap_years needs one year inside each of the {len(edges) + 1} periods, in order")
        if self.holdout.scenes:
            raise ValueError("holdout.scenes does not apply: a run trains on one scene")
        if self.pinned_rows and self.holdout.active:
            raise ValueError("a pinned run trains on exactly its view sets: a hold-out has no meaning there")


def lr_factor(step: int, warmup: int, schedule: str, total: int, min_ratio: float) -> float:
    """Learning-rate multiplier at optimiser step ``step``: linear warm-up, then constant or cosine decay.

    The warm-up reaches 1 at step ``warmup - 1``; ``"cosine"`` then falls to ``min_ratio`` at step ``total``.
    """
    if step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def mode_dependent_modules(module: nn.Module) -> list[str]:
    """Names of the submodules whose output differs between ``train()`` and ``eval()``.

    These are dropout layers with a positive rate and batch-normalisation layers. The released UNet
    has neither, which is what allows putting it in ``train()`` mode only to switch gradient
    checkpointing on (see the module docstring).
    """
    return [name for name, child in module.named_modules()
            if (isinstance(child, nn.modules.dropout._DropoutNd) and child.p > 0)
            or isinstance(child, nn.modules.batchnorm._BatchNorm)]


class WeightEMA:
    """An exponential moving average of a list of parameters.

    The decay grows towards ``decay``: update ``n`` (from 1) uses ``min(decay, (1 + n) / (EMA_WARMUP + n))``.
    Early on the average therefore follows the weights closely, instead of staying near their
    zero-initialised start for the first ``1 / (1 - decay)`` updates.
    """

    def __init__(self, params: list[nn.Parameter], decay: float):
        self.params, self.decay = params, decay
        self.reset()

    def reset(self, updates: int = 0) -> None:
        """Take the parameters' current values as the average, after ``updates`` updates."""
        self.shadow = [param.detach().clone() for param in self.params]
        self.updates = updates

    def update(self) -> None:
        """Move the average towards the parameters' current values; call after every optimiser step."""
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (EMA_WARMUP + self.updates))
        for shadow, param in zip(self.shadow, self.params):
            shadow.lerp_(param.detach(), 1.0 - decay)

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Inside the block the parameters hold the average; afterwards their own values again."""
        raw = [param.detach().clone() for param in self.params]
        self._assign(self.shadow)
        try:
            yield
        finally:
            self._assign(raw)

    @torch.no_grad()
    def _assign(self, values: list[torch.Tensor]) -> None:
        for param, value in zip(self.params, values):
            param.copy_(value)


class NumberedSamples(Dataset):
    """A scene's training samples by their running number in the run.

    Sample ``n`` is index ``n mod len(scene)`` of epoch ``n // len(scene)`` of the scene's sampler, so
    the stream depends on nothing but the scene's seed and ``n``: a resumed run asks for the numbers
    that follow, and data-loading processes need no shared state. The sampler also decides the
    sample's number of references (``ViewSamplingConfig.n_ref_choices``).
    """

    def __init__(self, scene: MVGenScene):
        self.scene = scene

    def __getitem__(self, number: int) -> dict:
        epoch, index = divmod(number, len(self.scene))
        return self.scene.make_sample(*self.scene.draw(index, epoch=epoch))


def checkpoint_to_resume(run_dir: str | Path, resume: str | None) -> Path | None:
    """The checkpoint a launch should resume from.

    ``resume`` is a checkpoint path, ``None`` (start fresh), or ``"auto"``: the run's ``latest.pt`` if it
    exists, for jobs that are requeued or submitted again, otherwise a fresh start (which
    :class:`Trainer` refuses if the folder holds a log or other checkpoints). A run writes its first
    ``latest.pt`` at step 0, as soon as its probes' first evaluation is done, so the same command
    also continues a job that died before its first optimiser step.
    """
    if resume is None:
        return None
    if resume != "auto":
        return Path(resume)
    latest = Path(run_dir) / "checkpoints" / "latest.pt"
    return latest if latest.exists() else None


def flat_config(config: dict, prefix: str = "") -> dict[str, object]:
    """A nested config dict (``asdict(TrainConfig)``) as one level: dotted names such as ``loss.precision``,
    and sequences as lists, so that a tuple equals the list it becomes in a saved file."""
    def plain(value: object) -> object:
        return [plain(item) for item in value] if isinstance(value, (list, tuple)) else value

    flat: dict[str, object] = {}
    for name, value in config.items():
        if isinstance(value, dict):
            flat.update(flat_config(value, f"{prefix}{name}."))
        else:
            flat[f"{prefix}{name}"] = plain(value)
    return flat


def resume_differences(saved: dict, current: dict) -> tuple[list[str], list[str]]:
    """How the configuration of a launch differs from the one its checkpoint was written under.

    Both are ``asdict(TrainConfig)``. Returns ``(conflicts, changes)``, each a list of
    ``"name: saved value -> current value"``: the fields that differ and must not, and those that
    differ and may (``RESUME_MAY_CHANGE``). A field that only one of the two has is a conflict:
    the checkpoint was written by another version of the code, under a setting that is not known.
    """
    saved, current = flat_config(saved), flat_config(current)
    may_change = set(RESUME_MAY_CHANGE)
    if saved.get("holdout.probe_targets") or current.get("holdout.probe_targets"):
        may_change.discard("render_rows")           # the renders' targets are part of the hold-out
    missing = "(not recorded)"
    conflicts, changes = [], []
    for name in sorted(set(saved) | set(current)):
        before, now = saved.get(name, missing), current.get(name, missing)
        if before != now:
            (changes if name.split(".")[0] in may_change else conflicts).append(f"{name}: {before!r} -> {now!r}")
    return conflicts, changes


def describe_pinned(views: dict, edges: tuple[float, ...]) -> str:
    """One line on a pinned view set for the start-up printout, from its :func:`describe_views` record:
    the references' and the targets' rows, dates and periods."""
    names, n_ref = period_names(edges), views["n_ref"]

    def part(role: str, chosen: slice) -> str:
        periods = list(dict.fromkeys(names[period] for period in views["periods"][chosen]))
        return f"{role} rows {views['rows'][chosen]} dated {views['years'][chosen]} ({', '.join(periods)})"

    return f"{part('references', slice(None, n_ref))}; {part('targets', slice(n_ref, None))}"


class Trainer:
    """A training run in ``cfg.run_dir`` (see the module docstring).

    Args:
        model: the released model as :func:`timemachine.mvgen.model.load_mvgen` returns it, without
            adapters (default: loaded from ``cfg.model_dir``).
        scene: the scene's dataset, opened with ``cfg.scale`` (default: opened from ``cfg.dataset``). A
            pinned run only needs the depth maps of its references; any other run needs
            ``depth_stats.parquet``.
        resume: a ``latest.pt`` of this run to continue from (:func:`checkpoint_to_resume`); None starts
            the run. Either is refused, before the model is loaded or anything is written, when the
            folder or the checkpoint does not allow it, and a checkpoint also, once the scene is open
            and before anything is written, when the dataset folder is no longer the one it was
            trained on (``ValueError``; "Starting and resuming" in the module docstring).

    Attributes: ``model``, ``scene``, ``dataset_fingerprint`` (the scene's, when the launch opened
    it), ``step`` (optimiser steps done), ``started`` (whether step 0, with its evaluation and its
    ``latest.pt``, is behind the run), ``params`` (the trainable parameters), ``ema``
    (:class:`WeightEMA` or None), ``swap_years`` (one year per period, for swapped dates),
    ``loss_probes`` (by name) and ``render_probes``.
    """

    def __init__(self, cfg: TrainConfig, model: MVGen | None = None, scene: MVGenScene | None = None,
                 resume: str | Path | None = None):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        checkpoint = self._checkpoint_to_continue(resume)
        self.model = model if model is not None else load_mvgen(cfg.model_dir)
        self.scene = scene if scene is not None else MVGenScene(
            cfg.dataset, cfg.sampling, resolution=cfg.resolution, seed=cfg.seed,
            camera_longest_side=self.model.config.camera_longest_side, require_depth=not cfg.pinned_rows,
            scale=cfg.scale)
        if parse_scale(self.scene.scale) != cfg.scale:
            raise ValueError(f"the scene was opened with scale {self.scene.scale!r}, the run's is {cfg.scale!r}")
        self.dataset_fingerprint = self.scene.fingerprint()         # of the folder as this launch found it
        if checkpoint is not None:
            self._check_dataset(checkpoint, resume)

        torch.manual_seed(cfg.seed)                 # the adapters' initial weights depend on the seed alone
        groups = prepare_for_finetuning(self.model, cfg.finetune)
        if cfg.finetune.grad_checkpointing and (changed := mode_dependent_modules(self.model.unet)):
            raise ValueError("gradient checkpointing needs the UNet in train() mode, which would also change "
                             f"{changed[:5]}: set their dropout rates to 0 or train without checkpointing")
        rates = {"date": cfg.lr_date, "lora": cfg.lr_lora}
        param_groups = [{"params": groups[name], "lr": rate, "name": name} for name, rate in rates.items()
                        if groups[name]]
        if not param_groups:
            raise ValueError("nothing to train: the run has neither the date embedding nor LoRA")
        self.params = [param for group in param_groups for param in group["params"]]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: lr_factor(
            step, cfg.warmup_steps, cfg.lr_schedule, cfg.steps, cfg.lr_min_ratio))
        self.ema = WeightEMA(self.params, cfg.ema_decay) if cfg.ema_decay > 0 else None
        self.generator = torch.Generator().manual_seed(cfg.seed)    # everything random in the training steps
        self.step = 0
        self.started = False        # whether step 0 is behind the run: its first evaluation and its first latest.pt
        self.grad_norm_history: deque[float] = deque(maxlen=GRAD_NORM_HISTORY)

        n_ref = cfg.sampling.n_ref
        check_view_sets([*cfg.pinned_rows, *cfg.render_rows], n_ref, len(self.scene.views))
        if cfg.holdout.active:
            self._hold_out_views()
        self.pinned = [self.scene.make_sample(np.asarray(rows), n_ref) for rows in cfg.pinned_rows]
        self.swap_years = cfg.swap_years or self._swap_years_from_data()
        self.loss_probes = self._loss_probes()
        self.render_probes = self._render_probes()
        self._write_records()

        sizes = ", ".join(f"{g['name']} {sum(p.numel() for p in g['params']) / 1e6:.2f}M" for g in param_groups)
        source = (f"{len(self.pinned)} pinned view sets" if self.pinned else
                  f"{int(self.scene.target_ok.sum())} training views, {len(self.scene.anchors)} anchors, "
                  f"{int(self.scene.heldout.sum())} held out")
        factor = self.scene.scale_factor
        scale = "a factor per sample (MVGenMaster's demo rule)" if factor is None else f"factor {factor:.6g}"
        print(f"[train] {len(self.scene.views)} views in {cfg.dataset}: {source}; trainable {sizes}; "
              f"precision {cfg.loss.precision}, UNet in {'train' if cfg.finetune.grad_checkpointing else 'eval'}() "
              f"mode (gradient checkpointing {'on' if cfg.finetune.grad_checkpointing else 'off'})\n"
              f"[train] scene scale {cfg.scale!r}: {scale} on camera translations and depth", flush=True)
        if checkpoint is not None:
            self._load_state(checkpoint, resume)

    # ------------------------------------------------------------------ set-up
    def _checkpoint_to_continue(self, resume: str | Path | None) -> dict | None:
        """Decide whether this launch may use the run folder; returns the checkpoint to continue from, if any.

        Raises ``ValueError`` for a fresh start in a folder that already holds a run, and for a
        checkpoint that cannot be resumed or was written under another configuration. Nothing has
        been loaded or written when this runs; whether the dataset is still the checkpoint's is
        checked once the scene is open (:meth:`_check_dataset`).
        """
        if resume is None:
            if (self.run_dir / "log.jsonl").exists() or any((self.run_dir / "checkpoints").glob("*.pt")):
                raise ValueError(
                    f"{self.run_dir} already holds a run (log.jsonl or checkpoints) and this launch would start "
                    "at step 0: its probes would be compared with that run's first evaluation and its log would "
                    "mix the two. Resume the run from its checkpoints/latest.pt (--resume auto does so when that "
                    "file exists; a run without one died during its first evaluation, before it trained "
                    "anything), or use a new run folder, or delete this one.")
            return None
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if "optimizer" not in checkpoint:
            raise ValueError(f"{resume} holds weights only: resume from the run's latest.pt")
        conflicts, changes = resume_differences(checkpoint["train_config"], asdict(self.cfg))
        if conflicts:
            raise ValueError(
                f"{resume} was written under another configuration, and resuming would silently mix the two "
                f"(the optimiser and schedule are the checkpoint's). Differing: {'; '.join(conflicts)}. Relaunch "
                f"with the run's options (only {', '.join(sorted(RESUME_MAY_CHANGE))} may change), or start a new "
                "run folder.")
        if changes:
            print(f"[train] resuming with changed settings: {'; '.join(changes)}", flush=True)
            if self.cfg.lr_schedule == "cosine" and any(change.startswith("steps:") for change in changes):
                print("[train] note: the cosine schedule now runs to the new number of steps: after one more step at "
                      "the saved rate, the learning rate jumps to the new curve", flush=True)
        return checkpoint

    def _check_dataset(self, checkpoint: dict, path: str | Path) -> None:
        """Refuse to resume (``ValueError``) unless the dataset folder is still the one ``checkpoint`` was trained on.

        The run's settings do not say so: with ``scale="scene"`` the scale factor comes from the folder's
        ``scene.json``, and the views that are drawn from its view table, depth statistics and
        co-visibility. So the scene's ``scale_factor`` and its ``fingerprint()`` must be those in the
        checkpoint; the error names which of the two differs. A checkpoint without a fingerprint
        cannot be checked and is refused as well. Runs once the scene is open, before anything is
        written.
        """
        if "dataset_fingerprint" not in checkpoint:
            raise ValueError(
                f"{path} does not record the dataset it was trained on (it was written before checkpoints held a "
                f"dataset fingerprint), so it cannot be checked that {self.cfg.dataset} is still that dataset: "
                "start a new run folder.")
        differing = []
        saved_factor, factor = checkpoint["scale_factor"], self.scene.scale_factor
        if saved_factor != factor:
            differing.append(f"the scene-scale factor was {saved_factor!r} and is now {factor!r} (under scale "
                             f"{self.cfg.scale!r} it comes from camera_extent in the folder's scene.json)")
        if checkpoint["dataset_fingerprint"] != self.dataset_fingerprint:
            differing.append("the dataset fingerprint differs (the view table's rows, usable flags, cameras or dates, "
                             "the views with reliable depth, or the co-visibility)")
        if differing:
            raise ValueError(
                f"{self.cfg.dataset} has changed since {path} was written, and resuming would silently continue the "
                f"run on other samples or at another scale: {'; '.join(differing)}. Restore the dataset as the run "
                "found it, or start a new run folder.")

    def _write_records(self) -> None:
        """Write what this launch runs with into the run folder: the config, the pinned view sets (which are
        also printed, with what a reader should be warned of), the hold-out and the probes."""
        cfg, n_ref = self.cfg, self.cfg.sampling.n_ref
        for name in ("checkpoints", "probes"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        config = {**asdict(cfg), "scale_factor": self.scene.scale_factor}
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))
        if self.pinned:
            described = [describe_views(self.scene, rows, n_ref) for rows in cfg.pinned_rows]
            (self.run_dir / "pinned.json").write_text(json.dumps(described, indent=1))
            for i, views in enumerate(described):
                print(f"[train] pinned view set {i}: {describe_pinned(views, cfg.sampling.period_edges)}", flush=True)
                for concern in ("not_usable", "references_without_reliable_depth"):
                    if views[concern]:
                        print(f"[train] note: pinned view set {i}, {concern.replace('_', ' ')}: {views[concern]}",
                              flush=True)
        if cfg.holdout.active:
            heldout = [{"row": int(row), "file_key": self.scene.views.file_key[row],
                        "year": round(float(self.scene.dates[row]), 4)} for row in np.flatnonzero(self.scene.heldout)]
            (self.run_dir / "heldout.json").write_text(json.dumps(heldout, indent=1))
        (self.run_dir / "probes.json").write_text(json.dumps({
            "swap_years": [None if math.isnan(year) else year for year in self.swap_years],
            "date_loss": {name: probe.describe() for name, probe in self.loss_probes.items()},
            "renders": [probe.describe() for probe in self.render_probes]}, indent=1))

    def _hold_out_views(self) -> None:
        """Take the hold-out away from the sampler: random views per era and, if asked, the renders' targets."""
        cfg = self.cfg
        targets = [row for rows in cfg.render_rows for row in rows[cfg.sampling.n_ref:]]
        self.scene.hold_out(heldout_rows(self.scene.dates, cfg.holdout, targets if cfg.holdout.probe_targets else []))

    def _swap_years_from_data(self) -> tuple[float, ...]:
        """One year per period for swapped dates: the median date of what the run trains on in that period."""
        n_ref = self.cfg.sampling.n_ref
        pinned_targets = [row for rows in self.cfg.pinned_rows for row in rows[n_ref:]]
        return representative_years(self.scene.dates[self.scene.target_ok], self.cfg.sampling.period_edges,
                                    preferred=self.scene.dates[pinned_targets])

    def _loss_probes(self) -> dict[str, DateLossProbe]:
        """The date-loss probes: on the pinned sets, or on sets drawn from the training and the held-out views."""
        cfg = self.cfg
        if cfg.probe_every == 0:
            return {}
        if cfg.pinned_rows:
            view_sets = {"pinned": list(cfg.pinned_rows)}
        else:
            view_sets = {"train": scene_view_sets(self.scene, cfg.probe_samples)}
            if self.scene.heldout.any():
                view_sets["heldout"] = heldout_view_sets(self.scene, cfg.probe_samples)
                if not view_sets["heldout"]:
                    del view_sets["heldout"]
                    print("[train] no date-loss probe on held-out views: too few of them share a date and 3D points",
                          flush=True)
        return {name: DateLossProbe(self.scene, sets, cfg.sampling.n_ref, cfg.loss, self.swap_years,
                                    self.run_dir / "probes" / f"{name}_baseline.json", cfg.probe_timesteps)
                for name, sets in view_sets.items()}

    def _render_probes(self) -> list[RenderProbe]:
        """One render probe per view set of ``render_rows`` (pinned runs: by default, per pinned set)."""
        cfg = self.cfg
        if cfg.render_every == 0:
            return []
        return [RenderProbe(self.scene, rows, cfg.sampling.n_ref, f"set{i}", cfg.render_dates, cfg.render_seeds,
                            cfg.loss, self.swap_years, cfg.render_steps, cfg.render_guidance, cfg.render_uncond_dates)
                for i, rows in enumerate(cfg.render_rows or cfg.pinned_rows)]

    # ------------------------------------------------------------------ evaluation
    def _averaged_weights(self) -> AbstractContextManager[None]:
        """Context: the weights to evaluate, which are the EMA if one is kept."""
        return self.ema.applied() if self.ema is not None else nullcontext()

    @contextmanager
    def evaluation(self) -> Iterator[None]:
        """Context for probes: no gradients, ``eval()`` mode, the EMA weights (if kept).

        On leaving, the UNet's mode, the raw weights and the global random state (CPU and GPU) are put
        back, so training goes on as if the block had not run; the GPU's peak-memory counter is reset,
        so the training log reports the memory of training steps alone.
        """
        unet = self.model.unet
        was_training = unet.training
        cpu_state = torch.get_rng_state()
        gpu_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        unet.eval()
        try:
            with torch.no_grad(), self._averaged_weights():
                yield
        finally:
            unet.train(was_training)
            torch.set_rng_state(cpu_state)
            if gpu_states is not None:
                torch.cuda.set_rng_state_all(gpu_states)
                torch.cuda.reset_peak_memory_stats()

    def log(self, record: dict) -> None:
        """Append one line to the run's ``log.jsonl``."""
        with open(self.run_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def run_loss_probes(self) -> None:
        """Evaluate the date-loss probes and log them (``probe: "date_loss:<name>"``)."""
        for name, probe in self.loss_probes.items():
            started = time.time()
            with self.evaluation():
                result = probe(self.model)
            self.log({"step": self.step, "probe": f"date_loss:{name}", **result})
            with open(self.run_dir / "probes" / f"{name}_losses.jsonl", "a") as f:
                f.write(json.dumps({"step": self.step, **probe.last_losses}) + "\n")
            ratio = result["date_temb_ratio"]
            size = "no date embedding" if ratio is None else f"date embedding / timestep embedding {ratio:.3f}"
            print(f"[probe] step {self.step} {name} ({time.time() - started:.0f}s): loss with true dates "
                  f"{result['loss_true']:.4f} ({result['change']:+.1%} since the first evaluation); swapped dates "
                  f"{result['rel_gap_swapped']:+.2%}, unknown dates {result['rel_gap_unknown']:+.2%}; {size}",
                  flush=True)

    def run_render_probes(self) -> None:
        """Render the probes' view sets into ``renders/step_XXXXXX`` and log their PSNR (``probe: "render:<name>"``)."""
        out_dir = self.run_dir / "renders" / f"step_{self.step:06d}"
        for probe in self.render_probes:
            started = time.time()
            with self.evaluation():
                result = probe(self.model, out_dir)
            self.log({"step": self.step, "probe": f"render:{probe.name}", **result})
            scores = ", ".join(f"{date} {value:.2f} dB" for date, value in result["psnr_by_date"].items())
            print(f"[render] step {self.step} {probe.name} ({time.time() - started:.0f}s): PSNR at date {scores}",
                  flush=True)

    # ------------------------------------------------------------------ state
    def save(self, tag: str | None = None) -> Path:
        """Write a checkpoint and return its path.

        ``step_XXXXXX.pt`` holds the trainable weights to evaluate (the EMA, if kept), ready for
        :func:`timemachine.mvgen.model.load_finetuned`, with the run's config, its scene-scale rule and
        factor (``scale``, ``scale_factor``: None when the factor is per sample), the fingerprint of
        its dataset folder (``dataset_fingerprint``: ``MVGenScene.fingerprint``) and its swap years.
        With ``tag`` (``"latest.pt"``) it also holds what resuming needs: the raw weights, the
        optimiser, the schedule, the spike guard's history and the training generator's state.
        """
        path = self.run_dir / "checkpoints" / (tag or f"step_{self.step:06d}.pt")
        extra = {"precision": self.cfg.loss.precision, "train_config": asdict(self.cfg),
                 "scale": self.cfg.scale, "scale_factor": self.scene.scale_factor,
                 "dataset_fingerprint": self.dataset_fingerprint,
                 "swap_years": list(self.swap_years), "weights": "ema" if self.ema is not None else "raw"}
        if tag is not None:
            extra.update(optimizer=self.optimizer.state_dict(), lr_scheduler=self.lr_scheduler.state_dict(),
                         grad_norm_history=list(self.grad_norm_history), generator=self.generator.get_state())
            if self.ema is not None:
                extra.update(raw_trainable=trainable_state_dict(self.model), ema_updates=self.ema.updates)
        with self._averaged_weights():
            save_checkpoint(path, self.model, self.cfg.finetune, self.step, extra=extra)
        return path

    def _load_state(self, ckpt: dict, path: str | Path) -> None:
        """Take over the state of the run saved in a ``latest.pt`` that :meth:`_checkpoint_to_continue` accepted.

        A ``latest.pt`` is written after the evaluation of its step, the one of step 0 included, so the
        resumed run is marked as started: :meth:`train` does not evaluate that step a second time.
        """
        self.started = True
        load_trainable_state_dict(self.model, ckpt["trainable"])        # the EMA, if the run keeps one
        if self.ema is not None:
            self.ema.reset(updates=ckpt["ema_updates"])                 # the average continues from what was loaded
            load_trainable_state_dict(self.model, ckpt["raw_trainable"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.grad_norm_history.clear()
        self.grad_norm_history.extend(ckpt["grad_norm_history"])
        self.generator.set_state(ckpt["generator"])
        self.step = ckpt["step"]
        print(f"[train] resumed from {path} at step {self.step}", flush=True)

    # ------------------------------------------------------------------ loop
    def clip_limit(self) -> float:
        """This step's gradient-norm limit: the spike guard, or infinity while it is off or lacks history."""
        factor = self.cfg.grad_clip_factor
        if factor is None or len(self.grad_norm_history) < SPIKE_GUARD_MIN_HISTORY:
            return math.inf
        return factor * float(np.median(self.grad_norm_history))

    def _samples(self, start: int) -> Iterator[dict]:
        """The run's training samples from number ``start`` on (see :class:`NumberedSamples`; pinned sets cycle)."""
        numbers = range(start, self.cfg.steps * self.cfg.grad_accum)
        if self.pinned:
            return (self.pinned[number % len(self.pinned)] for number in numbers)
        return iter(DataLoader(NumberedSamples(self.scene), batch_size=None, sampler=numbers,
                               num_workers=self.cfg.num_workers))

    def _due(self, every: int) -> bool:
        """Whether something that runs every ``every`` steps (0: never) runs now; the last step always counts."""
        return every > 0 and (self.step % every == 0 or self.step == self.cfg.steps)

    def train(self) -> None:
        """Train from the current step to ``cfg.steps``, with probes, logs and checkpoints at their intervals.

        A fresh run first evaluates its probes at step 0 and then writes ``latest.pt``, whatever
        ``latest_every`` is: the probes start the run's log, and a folder with a log and nothing to
        resume from could not be relaunched ("Starting and resuming" in the module docstring). A run
        resumed from a checkpoint, also from that first one, does neither again. The UNet is left in
        ``eval()`` mode.
        """
        cfg = self.cfg
        if not self.started:
            self.run_loss_probes()
            self.run_render_probes()
            self.save("latest.pt")
            self.started = True
        self.model.unet.train(cfg.finetune.grad_checkpointing)      # see "Mode and memory" in the module docstring
        samples = self._samples(self.step * cfg.grad_accum)
        window, norms, clipped, last_log = [], [], [], time.time()
        while self.step < cfg.steps:
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum):
                loss, info = training_loss(self.model, next(samples), cfg.loss, self.generator,
                                           cfg.sampling.period_edges)
                (loss / cfg.grad_accum).backward()
                window.append(info)
            limit = self.clip_limit()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, limit))
            if math.isfinite(grad_norm):
                self.grad_norm_history.append(grad_norm)
                norms.append(grad_norm)
                clipped.append(grad_norm > limit)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update()
            else:
                print(f"[train] step {self.step + 1}: the gradient is not finite, update skipped", flush=True)
            rates = {f"lr_{group['name']}": group["lr"] for group in self.optimizer.param_groups}     # of this step
            self.lr_scheduler.step()
            self.step += 1

            if self.step % cfg.log_every == 0 or self.step == cfg.steps:
                self._log_training(window, norms, clipped, rates, time.time() - last_log)
                window, norms, clipped, last_log = [], [], [], time.time()
            busy = time.time()                      # probes and checkpoints are kept out of the speed measurement
            if self._due(cfg.probe_every):
                self.run_loss_probes()
            if self._due(cfg.render_every):
                self.run_render_probes()
            if self._due(cfg.save_every) or self.step == cfg.steps:
                self.save()
            if self._due(cfg.latest_every) or self.step == cfg.steps:
                self.save("latest.pt")
            last_log += time.time() - busy
        self.model.unet.eval()

    def _log_training(self, window: list[dict], norms: list[float], clipped: list[bool], rates: dict[str, float],
                      seconds: float) -> None:
        """Log and print the training steps since the last such line.

        ``window`` has one entry per sample, ``norms`` and ``clipped`` one per update that was made, and
        ``rates`` are the learning rates of the latest step. ``date_rms`` and ``date_temb_ratio`` are the
        means over the window's forward passes (the raw weights, not the EMA; None without a date embedding).
        """
        def mean_size(name: str) -> float | None:
            measured = [info[name] for info in window if info[name] is not None]
            return float(np.mean(measured)) if measured else None

        steps = len(window) // self.cfg.grad_accum
        record = {"step": self.step, "samples": self.step * self.cfg.grad_accum,
                  "loss": float(np.mean([info["loss"] for info in window])),
                  "grad_norm_median": float(np.median(norms)) if norms else None,
                  "grad_norm_max": float(np.max(norms)) if norms else None,
                  "clipped_fraction": float(np.mean(clipped)) if clipped else None,
                  "skipped_steps": steps - len(norms),
                  "dates_known": float(np.mean([info["dates_known"] for info in window])),
                  "date_rms": mean_size("date_rms"), "date_temb_ratio": mean_size("date_temb_ratio"),
                  **rates, "sec_per_step": seconds / steps,
                  "max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0}
        self.log(record)
        norm = "none" if not norms else f"{record['grad_norm_median']:.3g} (max {record['grad_norm_max']:.3g})"
        ratio = record["date_temb_ratio"]
        size = "" if ratio is None else f" date embedding {record['date_rms']:.3g} ({ratio:.3f} of temb)"
        print(f"[train] step {self.step} loss {record['loss']:.4f} gradient norm {norm}{size} "
              f"{record['sec_per_step']:.2f} s/step, {record['max_mem_gb']:.1f} GB", flush=True)
