"""Views a training run never sees, to tell learning a scene apart from memorising its photos.

On a single scene, every probe so far rendered views the model had trained on, so a
model that memorised each photo's look by date was indistinguishable from one that
learned the scene's eras (docs/findings.md, 2026-11-18). A hold-out removes views
from training samples (:meth:`DatedViewsDataset.hold_out`); the held-out probe
(:class:`timemachine.training.probes.HeldoutProbe`) then compares the loss on them
with the loss on trained views over the course of training.

Held-out views are drawn at random within each era of each scene, so the few old
photos are represented; optionally the render probes' target views are held out as
well, so their contact sheets show cameras and photos the model never trained on.
Whole scenes can also be held out, to measure generalisation to unseen places.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HoldoutConfig:
    fraction: float = 0.0                       # share of each era's views held out at random (0: none)
    probe_targets: bool = False                 # also hold out the render probes' target views
    scenes: tuple[int, ...] = ()                # scene ids held out entirely (evaluation only)
    seed: int = 0                               # fixed separately from the training seed
    era_edges: tuple[float, ...] = (1900.0, 1970.0, 2000.0)

    def __post_init__(self):
        if not 0.0 <= self.fraction < 1.0:
            raise ValueError("holdout fraction must be in [0, 1).")

    @property
    def active(self) -> bool:
        return self.fraction > 0 or self.probe_targets or bool(self.scenes)


def era_index(dates: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Era bin of each date: 0 before ``edges[0]``, ..., ``len(edges)`` from ``edges[-1]`` on."""
    return np.digitize(np.asarray(dates, dtype=np.float64), edges)


def heldout_rows(dates: np.ndarray, cfg: HoldoutConfig, extra_rows=(), groups: np.ndarray | None = None
                 ) -> np.ndarray:
    """Sorted view rows to hold out: ``round(fraction * n)`` random views per era (per group, if
    ``groups`` gives each view's scene/model group), plus ``extra_rows``."""
    rng = np.random.default_rng(cfg.seed)
    era = era_index(dates, cfg.era_edges)
    if groups is not None:                      # one stratum per (group, era); group 0 alone = era only
        era = np.asarray(groups) * (len(cfg.era_edges) + 1) + era
    chosen = [np.asarray(extra_rows, dtype=int)]
    for e in np.unique(era):
        members = np.flatnonzero(era == e)
        chosen.append(rng.choice(members, int(round(cfg.fraction * len(members))), replace=False))
    return np.unique(np.concatenate(chosen))
