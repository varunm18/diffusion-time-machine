"""PyTorch dataset of dated multi-view samples for date-conditioned SEVA.

Each item is one sample drawn by :mod:`timemachine.data.sampling` from a view
table (see :mod:`timemachine.data.views`, with image columns added by
:mod:`timemachine.data.images`). Views are ordered inputs first, so view 0 (an
input) is SEVA's reference camera.

Item fields (V = n_inputs + n_targets):

=================  ===============  ==================================================
images             (V, 3, S, S)     float in [-1, 1], short side resized to S, centre crop
K                  (V, 3, 3)        normalized intrinsics (SEVA format)
c2w                (V, 4, 4)        camera-to-world, SEVA-normalized (view 0 at distance 2)
date               (V, 2)           [earliest, latest) in decimal years
date_mid           (V,)             interval midpoint
is_input           (V,)             bool, True for the first n_inputs views
is_monochrome       (V,)             bool, weak pixel-based monochrome (B&W/sepia) hint
medium             (V,)             int code, see MEDIUM_CODES
file_keys          list[str]        Commons file of each view
=================  ===============  ==================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from timemachine.data.cameras import (
    adjust_intrinsics, colmap_to_c2w, intrinsics_matrix, normalize_cameras,
    normalize_intrinsics, resize_and_center_crop_params,
)
from timemachine.data.sampling import SamplingConfig, eligible_anchors, sample_views
from timemachine.dating.medium import Medium

MEDIUM_CODES = {Medium.UNKNOWN: 0, Medium.PHOTO: 1, Medium.POSTCARD: 2, Medium.PAINTING: 3,
                Medium.PRINT: 4, Medium.DRAWING: 5, Medium.PLAN: 6, Medium.ARTWORK: 7}

_REQUIRED = ("image_path", "image_width", "image_height", "camera_width", "camera_height",
             "fx", "fy", "cx", "cy", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
             "date_lo", "date_hi", "date_mid", "medium", "is_monochrome", "file_key")


def load_view(row: pd.Series, size: int) -> tuple[torch.Tensor, np.ndarray]:
    """Image tensor (3, size, size) in [-1, 1] and its pixel intrinsics after resize + crop."""
    image = Image.open(row.image_path).convert("RGB")
    scale, left, top = resize_and_center_crop_params(image.width, image.height, size)
    new_w, new_h = round(image.width * scale), round(image.height * scale)
    image = image.resize((new_w, new_h), Image.Resampling.BICUBIC).crop((left, top, left + size, top + size))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)

    # COLMAP intrinsics refer to the camera's resolution; our cached file may be smaller.
    K = intrinsics_matrix(row.fx, row.fy, row.cx, row.cy)
    K = adjust_intrinsics(K, row.image_width / row.camera_width, row.image_height / row.camera_height)
    K = adjust_intrinsics(K, new_w / row.image_width, new_h / row.image_height, left, top)
    return pixels, K


class DatedViewsDataset(Dataset):
    """Random dated samples from one scene/model view table.

    Sampling is deterministic per (seed, epoch, index); call :meth:`set_epoch`
    each epoch to get fresh samples.
    """

    def __init__(self, views: pd.DataFrame, config: SamplingConfig = SamplingConfig(),
                 resolution: int = 576, samples_per_epoch: int = 1000, seed: int = 0,
                 camera_scale: float = 2.0):
        missing = [c for c in _REQUIRED if c not in views]
        if missing:
            raise ValueError(f"View table is missing columns: {missing}")
        if views["model"].nunique() != 1:
            raise ValueError("All views must come from one COLMAP model (one coordinate frame).")
        self.views = views.reset_index(drop=True)
        self.config = config
        self.resolution = resolution
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.camera_scale = camera_scale
        self.epoch = 0
        self.dates = self.views["date_mid"].to_numpy(dtype=np.float64)
        self.anchors = eligible_anchors(self.dates, config)
        if len(self.anchors) == 0:
            raise ValueError("No sample can be drawn: too few views share a date.")
        self.c2ws = colmap_to_c2w(self.views[["qw", "qx", "qy", "qz"]].to_numpy(),
                                  self.views[["tx", "ty", "tz"]].to_numpy())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def indices(self, index: int) -> np.ndarray:
        """View-table rows of sample ``index``: inputs first, then targets."""
        rng = np.random.default_rng((self.seed, self.epoch, index))
        inputs, targets = sample_views(self.dates, self.config, rng, self.anchors)
        return np.concatenate([inputs, targets])

    def __getitem__(self, index: int) -> dict:
        rows = self.indices(index)
        sub = self.views.iloc[rows]
        images, Ks = zip(*(load_view(row, self.resolution) for row in sub.itertuples()))
        K_norm = np.stack([normalize_intrinsics(K, self.resolution, self.resolution) for K in Ks])
        c2w = normalize_cameras(self.c2ws[rows], camera_scale=self.camera_scale)
        is_input = np.zeros(len(rows), dtype=bool)
        is_input[: self.config.n_inputs] = True
        return {
            "images": torch.stack(images),
            "K": torch.from_numpy(K_norm).float(),
            "c2w": torch.from_numpy(c2w).float(),
            "date": torch.tensor(sub[["date_lo", "date_hi"]].to_numpy(np.float32)),
            "date_mid": torch.tensor(sub["date_mid"].to_numpy(np.float32)),
            "is_input": torch.from_numpy(is_input),
            "is_monochrome": torch.tensor(sub["is_monochrome"].to_numpy(bool)),
            "medium": torch.tensor([MEDIUM_CODES.get(m, 0) for m in sub["medium"]]),
            "file_keys": sub["file_key"].tolist(),
        }
