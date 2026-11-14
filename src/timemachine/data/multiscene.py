"""Multi-scene training dataset over the dataset v1 caches.

Each item is one sample of ``V = n_inputs + n_targets`` views from a single
COLMAP model (inputs first; view 0 is SEVA's reference camera). Nothing is
decoded or encoded at load time: latents and CLIP embeddings come from the
arrays written by :mod:`timemachine.data.cache`.

=================  ================  ====================================================
latents            (V, 4, 72, 72)    VAE latents, already scaled by 0.18215
clip               (V, 1024)         OpenCLIP ViT-H embedding (SEVA averages the inputs')
K                  (V, 3, 3)         normalized intrinsics of the 576x576 crop
c2w                (V, 4, 4)         camera-to-world, SEVA-normalized (view 0 at distance 2)
year               (V,)              decimal year drawn uniformly from the view's date interval
year_valid         (V,)              False -> the model should use its "unknown date" embedding
is_input           (V,)              True for the first n_inputs views
medium             (V,)              int code (dataset.MEDIUM_CODES); not used by the model in v1
is_monochrome      (V,)              weak pixel hint; not used by the model in v1
scene_id, model    scalars
target_era         str               era the targets were drawn from
file_keys          list[str]         Commons file of each view
=================  ================  ====================================================

How a sample is drawn (deterministic per ``(seed, epoch, index)``):

1. a model, with probability proportional to ``sqrt(old anchors + modern_weight *
   other anchors)``, after capping every scene cluster at ``max_cluster_share``;
2. a target era, uniformly among the eras that model has anchors in, so old
   targets are as frequent as modern ones wherever both exist;
3. inputs and targets with :func:`timemachine.data.sampling.sample_views`,
   restricted to anchors of that era; a draw that puts two near-duplicates
   (same ``dup_cluster``) in one sample is rejected and redrawn.

Plücker rays and date dropout (classifier-free guidance) belong in the training
step, which keeps this dataset deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from timemachine.analysis.census import ERAS, OLD_ERAS
from timemachine.config import DataPaths
from timemachine.data.cache import CLIP, LATENTS, RESOLUTION, cache_is_current
from timemachine.data.cameras import (
    adjust_intrinsics, colmap_to_c2w, intrinsics_matrix, normalize_cameras,
    normalize_intrinsics, resize_and_center_crop_params,
)
from timemachine.data.dataset import MEDIUM_CODES
from timemachine.data.sampling import SamplingConfig, eligible_anchors, sample_views
from timemachine.data.selection import era_of

MIN_YEAR, MAX_YEAR = 1826.0, 2026.0     # first photograph .. now
MIN_VALID_CONFIDENCE = 0.5              # below this the date is treated as unknown (interface rule 4)


def view_intrinsics(row, size: int = RESOLUTION) -> np.ndarray:
    """Normalized intrinsics of a view's ``size`` crop, from the table alone.

    Same arithmetic as :func:`timemachine.data.dataset.load_view`, without opening the image.
    """
    scale, left, top = resize_and_center_crop_params(row.image_width, row.image_height, size)
    new_w, new_h = round(row.image_width * scale), round(row.image_height * scale)
    K = intrinsics_matrix(row.fx, row.fy, row.cx, row.cy)
    K = adjust_intrinsics(K, row.image_width / row.camera_width, row.image_height / row.camera_height)
    K = adjust_intrinsics(K, new_w / row.image_width, new_h / row.image_height, left, top)
    return normalize_intrinsics(K, size, size)


def cap_group_share(weights: np.ndarray, groups: np.ndarray, max_share: float) -> np.ndarray:
    """Probabilities proportional to ``weights`` in which no group exceeds ``max_share``.

    Groups over the cap are scaled down to exactly the cap and the freed
    probability goes to the others (repeated, since that can push another group over).
    """
    p = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    groups = np.asarray(groups)
    if len(np.unique(groups)) * max_share < 1.0:
        return p                         # too few groups for the cap to be satisfiable
    capped = np.zeros(len(p), dtype=bool)
    for _ in range(len(np.unique(groups))):
        share = pd.Series(p).groupby(groups).sum()
        over = [g for g, s in share.items() if s > max_share + 1e-12 and not capped[groups == g].all()]
        if not over:
            break
        for g in over:
            p[groups == g] *= max_share / share[g]
            capped[groups == g] = True
        free = ~capped
        p[free] *= (1.0 - p[capped].sum()) / p[free].sum()
    return p


@dataclass
class ModelViews:
    """Everything needed to draw samples from one COLMAP model."""
    scene_id: int
    model: int
    cluster: int
    folder: object
    file_keys: list[str]
    dates: np.ndarray
    date_lo: np.ndarray
    date_hi: np.ndarray
    year_valid: np.ndarray
    K: np.ndarray
    c2w: np.ndarray
    dup: np.ndarray
    medium: np.ndarray
    is_monochrome: np.ndarray
    anchors: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def weight_counts(self) -> tuple[int, int]:
        old = sum(len(a) for era, a in self.anchors.items() if era in OLD_ERAS)
        return old, sum(len(a) for a in self.anchors.values()) - old


def load_model(folder, cluster: int, cfg: SamplingConfig) -> ModelViews:
    views = pd.read_parquet(folder / "views.parquet")
    keys = views["file_key"].tolist()
    if not cache_is_current(folder, keys):
        raise ValueError(f"{folder}: latent cache missing or out of date; run scripts/cache_dataset_v1.py")
    dates = views["date_mid"].to_numpy(np.float64)
    anchors = eligible_anchors(dates, cfg)
    eras = era_of(dates[anchors])
    return ModelViews(
        scene_id=int(views["scene_id"].iloc[0]), model=int(views["model"].iloc[0]), cluster=cluster, folder=folder,
        file_keys=keys, dates=dates,
        date_lo=views["date_lo"].to_numpy(np.float64), date_hi=views["date_hi"].to_numpy(np.float64),
        year_valid=views["date_confidence"].to_numpy() >= MIN_VALID_CONFIDENCE,
        K=np.stack([view_intrinsics(r) for r in views.itertuples()]),
        c2w=colmap_to_c2w(views[["qw", "qx", "qy", "qz"]].to_numpy(), views[["tx", "ty", "tz"]].to_numpy()),
        dup=pd.factorize(views["dup_cluster"])[0],
        medium=np.array([MEDIUM_CODES.get(m, 0) for m in views["medium"]]),
        is_monochrome=views["is_monochrome"].to_numpy(bool),
        anchors={label: anchors[eras == label] for label, _, _ in ERAS if (eras == label).any()},
    )


class MultiSceneDataset(Dataset):
    """Random dated samples from every model of one split (see the module docstring)."""

    def __init__(self, paths: DataPaths | None = None, split: str = "train",
                 config: SamplingConfig = SamplingConfig(), samples_per_epoch: int = 10_000, seed: int = 0,
                 modern_weight: float = 0.2, max_cluster_share: float = 0.25, camera_scale: float = 2.0,
                 max_retries: int = 20):
        paths = paths or DataPaths()
        models = pd.read_parquet(paths.tables / "models_v1.parquet")
        splits = pd.read_parquet(paths.tables / "splits_v1.parquet")[["scene_id", "cluster", "split"]]
        models = models[models["role"] != "excluded"].merge(splits, on="scene_id")
        models = models[models["split"] == split]
        loaded = [load_model(paths.dataset_dir(int(m.scene_id), int(m.model)), int(m.cluster), config)
                  for m in models.itertuples()]
        self.models = [m for m in loaded if m.anchors]          # a model without anchors cannot give a sample
        if not self.models:
            raise ValueError(f"no model of split '{split}' can produce a sample")
        counts = np.array([m.weight_counts for m in self.models], dtype=np.float64)
        weights = np.sqrt(counts[:, 0] + modern_weight * counts[:, 1])
        self.probabilities = cap_group_share(weights, np.array([m.cluster for m in self.models]), max_cluster_share)
        self.config, self.samples_per_epoch, self.seed = config, samples_per_epoch, seed
        self.camera_scale, self.max_retries = camera_scale, max_retries
        self.epoch = 0
        self._arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arrays"] = {}            # memory maps are reopened in each worker process
        return state

    def _cached(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if i not in self._arrays:
            folder = self.models[i].folder
            self._arrays[i] = (np.load(folder / LATENTS, mmap_mode="r"), np.load(folder / CLIP, mmap_mode="r"))
        return self._arrays[i]

    def draw(self, index: int) -> tuple[int, str, np.ndarray, np.random.Generator]:
        """Model index, target era and view rows (inputs first) of sample ``index``."""
        rng = np.random.default_rng((self.seed, self.epoch, index))
        for _ in range(self.max_retries):
            i = int(rng.choice(len(self.models), p=self.probabilities))
            m = self.models[i]
            era = str(rng.choice(sorted(m.anchors)))
            inputs, targets = sample_views(m.dates, self.config, rng, m.anchors[era])
            rows = np.concatenate([inputs, targets])
            if len(np.unique(m.dup[rows])) == len(rows):
                return i, era, rows, rng
        raise RuntimeError(f"sample {index}: {self.max_retries} draws all held near-duplicate views")

    def __getitem__(self, index: int) -> dict:
        i, era, rows, rng = self.draw(index)
        m = self.models[i]
        latents, clip = self._cached(i)
        order = np.argsort(rows)                     # read the memory map in file order, then restore
        restore = np.argsort(order)
        is_input = np.zeros(len(rows), dtype=bool)
        is_input[:self.config.n_inputs] = True
        year = np.clip(rng.uniform(m.date_lo[rows], m.date_hi[rows]), MIN_YEAR, MAX_YEAR)
        return {
            "latents": torch.from_numpy(latents[rows[order]][restore].astype(np.float32)),
            "clip": torch.from_numpy(clip[rows[order]][restore].astype(np.float32)),
            "K": torch.from_numpy(m.K[rows]).float(),
            "c2w": torch.from_numpy(normalize_cameras(m.c2w[rows], camera_scale=self.camera_scale)).float(),
            "year": torch.from_numpy(year).float(),
            "year_valid": torch.from_numpy(m.year_valid[rows]),
            "is_input": torch.from_numpy(is_input),
            "medium": torch.from_numpy(m.medium[rows]),
            "is_monochrome": torch.from_numpy(m.is_monochrome[rows]),
            "scene_id": m.scene_id,
            "model": m.model,
            "target_era": era,
            "file_keys": [m.file_keys[r] for r in rows],
        }
