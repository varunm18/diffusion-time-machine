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
    adjust_intrinsics, camera_features, colmap_to_c2w, intrinsics_matrix, normalize_cameras,
    normalize_intrinsics, resize_and_center_crop_params, undistorted_crop,
)
from timemachine.data.sampling import SamplingConfig, eligible_anchors, event_anchors, sample_views
from timemachine.dating.medium import Medium

MEDIUM_CODES = {Medium.UNKNOWN: 0, Medium.PHOTO: 1, Medium.POSTCARD: 2, Medium.PAINTING: 3,
                Medium.PRINT: 4, Medium.DRAWING: 5, Medium.PLAN: 6, Medium.ARTWORK: 7}

_REQUIRED = ("scene_id", "image_path", "image_width", "image_height", "camera_width", "camera_height",
             "fx", "fy", "cx", "cy", "k1", "k2", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
             "date_lo", "date_hi", "date_mid", "medium", "is_monochrome", "file_key")


def load_view(row, size: int) -> tuple[torch.Tensor, np.ndarray]:
    """Image tensor (3, size, size) in [-1, 1] and its pinhole pixel intrinsics after resize + crop.

    The short side is resized to ``size`` (antialiased), then the centre square is
    taken; views with radial distortion (``k1``/``k2``) are undistorted in that same
    step, so the returned intrinsics describe a distortion-free camera.
    """
    if not (np.isfinite(row.k1) and np.isfinite(row.k2)):
        raise ValueError(f"{row.file_key}: unsupported lens model (no radial coefficients)")
    image = Image.open(row.image_path).convert("RGB")
    scale, left, top = resize_and_center_crop_params(image.width, image.height, size)
    new_w, new_h = round(image.width * scale), round(image.height * scale)
    image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    # COLMAP intrinsics refer to the camera's resolution; our cached file may be smaller.
    K = intrinsics_matrix(row.fx, row.fy, row.cx, row.cy)
    K = adjust_intrinsics(K, row.image_width / row.camera_width, row.image_height / row.camera_height)
    K = adjust_intrinsics(K, new_w / row.image_width, new_h / row.image_height)
    if row.k1 == 0 and row.k2 == 0:
        pixels = np.asarray(image.crop((left, top, left + size, top + size)), dtype=np.float32)
        return torch.from_numpy(pixels / 127.5 - 1.0).permute(2, 0, 1), adjust_intrinsics(K, 1, 1, left, top)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
    return undistorted_crop(pixels, K, row.k1, row.k2, left, top, size)


def crop_intrinsics(row, size: int) -> np.ndarray:
    """The intrinsics :func:`load_view` returns for ``row`` at ``size``, without decoding the image.

    For code that draws into a view's crop (masks, projected points) and must land on the same
    pixels. Only the file's header is read, for the image size :func:`load_view` computes with; the
    arithmetic is the same (both branches of :func:`load_view` end in the same matrix), and
    ``tests/test_mvgen_regions.py`` compares the two.
    """
    if not (np.isfinite(row.k1) and np.isfinite(row.k2)):
        raise ValueError(f"{row.file_key}: unsupported lens model (no radial coefficients)")
    with Image.open(row.image_path) as image:
        width, height = image.size
    scale, left, top = resize_and_center_crop_params(width, height, size)
    new_w, new_h = round(width * scale), round(height * scale)
    K = intrinsics_matrix(row.fx, row.fy, row.cx, row.cy)
    K = adjust_intrinsics(K, row.image_width / row.camera_width, row.image_height / row.camera_height)
    K = adjust_intrinsics(K, new_w / row.image_width, new_h / row.image_height)
    return adjust_intrinsics(K, 1, 1, left, top)


class DatedViewsDataset(Dataset):
    """Random dated samples from the view tables of one or more scenes.

    Each COLMAP model of a scene has its own coordinate frame, so it is its own
    *group*, and a sample's views always come from one group. With several groups,
    each sample first picks a group, with probability proportional to its number
    of training views to the power ``group_weight_power`` (0: every scene equally
    often; 1: every view equally often), then draws views inside it.

    Sampling is deterministic per (seed, epoch, index); call :meth:`set_epoch`
    each epoch to get fresh samples. :meth:`hold_out` removes views from the
    samples (for evaluation on views or scenes the model never trained on);
    held-out views can still be loaded with :meth:`make_sample`.
    """

    def __init__(self, views: pd.DataFrame, config: SamplingConfig = SamplingConfig(),
                 resolution: int = 576, samples_per_epoch: int = 1000, seed: int = 0,
                 camera_scale: float = 2.0, group_weight_power: float = 0.5):
        missing = [c for c in _REQUIRED if c not in views]
        if missing:
            raise ValueError(f"View table is missing columns: {missing}")
        self.views = views.reset_index(drop=True)
        self.config = config
        self.resolution = resolution
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.camera_scale = camera_scale
        self.group_weight_power = group_weight_power
        self.epoch = 0
        self.dates = self.views["date_mid"].to_numpy(dtype=np.float64)
        # Group = (scene, COLMAP model), numbered in order of first appearance.
        self.group = self.views.groupby(["scene_id", "model"], sort=False).ngroup().to_numpy()
        self.group_rows = [np.flatnonzero(self.group == g) for g in range(self.group.max() + 1)]
        self.c2ws = colmap_to_c2w(self.views[["qw", "qx", "qy", "qz"]].to_numpy(),
                                  self.views[["tx", "ty", "tz"]].to_numpy())
        self.camera_feats = np.zeros((len(self.views), 6))
        for rows in self.group_rows:                    # positions are in units of each scene's spread
            self.camera_feats[rows] = camera_features(self.c2ws[rows])
        self.hold_out(np.array([], dtype=int))

    @property
    def n_groups(self) -> int:
        return len(self.group_rows)

    @property
    def heldout_groups(self) -> np.ndarray:
        """(n_groups,) bool: groups whose views are all held out (held-out scenes)."""
        return np.array([len(train) == 0 for train in self._group_train])

    def hold_out(self, rows: np.ndarray) -> None:
        """Draw samples only from the other views (replaces any earlier hold-out).

        Sets ``train_rows`` (all groups), ``anchors`` (rows that can anchor a sample), the
        groups samples are drawn from and their weights. A group whose views are all held
        out (a held-out scene) is never sampled. With nothing held out and one group, the
        sample stream is exactly that of the single-scene dataset before 2026-11-18.
        """
        self.heldout = np.zeros(len(self.views), dtype=bool)
        self.heldout[np.asarray(rows, dtype=int)] = True
        self.train_rows = np.flatnonzero(~self.heldout)
        self._group_train, self._group_anchors = [], []    # per group: train rows; anchors (positions in them)
        for rows_g in self.group_rows:
            train = rows_g[~self.heldout[rows_g]]
            self._group_train.append(train)
            self._group_anchors.append(eligible_anchors(self.dates[train], self.config) if len(train)
                                       else np.array([], dtype=int))
        # Change events (SamplingConfig.events): per sampled group with one, its year and the anchors
        # (positions in the group's train rows) that can be drawn across it.
        event_year = dict(self.config.events)
        self._event = {}
        for g, train in enumerate(self._group_train):
            year = event_year.get(int(self.views.scene_id[self.group_rows[g][0]]))
            if year is not None and len(train):
                anchors = event_anchors(self.dates[train], self.config, year)
                if len(anchors):
                    self._event[g] = (year, anchors)
        self.sample_groups = np.array([g for g, a in enumerate(self._group_anchors) if len(a)], dtype=int)
        if len(self.sample_groups) == 0:
            raise ValueError("No sample can be drawn: too few views share a date.")
        sizes = np.array([len(self._group_train[g]) for g in self.sample_groups], dtype=np.float64)
        self.group_weights = sizes ** self.group_weight_power / (sizes ** self.group_weight_power).sum()
        self.anchors = np.concatenate([self._group_train[g][self._group_anchors[g]] for g in self.sample_groups])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def indices(self, index: int, epoch: int | None = None, seed: int | None = None) -> np.ndarray:
        """View-table rows of sample ``index``: inputs first, then targets.

        ``epoch`` / ``seed`` default to the dataset's own; probes pass fixed values so their
        samples do not depend on the training run's seed.
        """
        rng = np.random.default_rng((self.seed if seed is None else seed,
                                     self.epoch if epoch is None else epoch, index))
        # Draw only what is configured, so older runs keep their exact sample stream.
        if self._event and self.config.event_prob > 0 and rng.random() < self.config.event_prob:
            g = rng.choice(sorted(self._event))             # a sample across a change event
            year, anchors = self._event[g]
            rows = self._group_train[g]
            inputs, targets = sample_views(self.dates[rows], self.config, rng, anchors, self.camera_feats[rows],
                                           split_year=year)
            return rows[np.concatenate([inputs, targets])]
        g = (self.sample_groups[0] if len(self.sample_groups) == 1
             else rng.choice(self.sample_groups, p=self.group_weights))
        rows = self._group_train[g]
        inputs, targets = sample_views(self.dates[rows], self.config, rng, self._group_anchors[g],
                                       self.camera_feats[rows])
        return rows[np.concatenate([inputs, targets])]

    @property
    def event_groups(self) -> dict[int, float]:
        """Groups that can be sampled across a change event, and the event's year."""
        return {g: year for g, (year, _) in self._event.items()}

    def __getitem__(self, index: int) -> dict:
        return self.make_sample(self.indices(index), self.config.n_inputs)

    def make_sample(self, rows: np.ndarray, n_inputs: int) -> dict:
        """A sample from explicit view-table rows (the first ``n_inputs`` are inputs), all from one group."""
        if len(np.unique(self.group[rows])) != 1:
            raise ValueError("A sample's views must come from one COLMAP model of one scene (one coordinate frame).")
        sub = self.views.iloc[rows]
        images, Ks = zip(*(load_view(row, self.resolution) for row in sub.itertuples()))
        K_norm = np.stack([normalize_intrinsics(K, self.resolution, self.resolution) for K in Ks])
        c2w = normalize_cameras(self.c2ws[rows], camera_scale=self.camera_scale)
        is_input = np.zeros(len(rows), dtype=bool)
        is_input[:n_inputs] = True
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
