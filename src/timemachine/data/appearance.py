"""Per-view appearance statistics for the images a sample actually contains.

How a photograph *looks* - its colour cast, tone curve, grain - is the confound
that :mod:`timemachine.analysis.shortcut` measured: across four scenes those
cues alone separate pre-1945 from modern photos at AUC 0.89, so "old" and
"sepia and flat" are nearly the same signal in our training data.

This module turns that measurement into a per-view conditioning signal. The
statistics are computed from the pixels the model sees (after resize and centre
crop), not from the file on disk, and they go through exactly the same code as
the measurement so the two cannot drift apart.

The model side lives in :mod:`timemachine.model.appearance_film`.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from timemachine.analysis.shortcut import FEATURE_NAMES, appearance_features, feature_matrix

APPEARANCE_FEATURES = FEATURE_NAMES
N_APPEARANCE_FEATURES = len(FEATURE_NAMES)


def appearance_from_tensor(image: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [-1, 1] -> the (F,) appearance statistics of that image."""
    array = (image.detach().float().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
    # Round, do not truncate: astype(uint8) floors, which loses a level on the way
    # back and makes these statistics disagree with the ones measured from the file.
    pil = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), mode="RGB")
    return feature_matrix([appearance_features(pil)])[0]


def appearance_batch(images: torch.Tensor) -> torch.Tensor:
    """(V, 3, H, W) in [-1, 1] -> (V, F) float32 appearance statistics."""
    return torch.from_numpy(np.stack([appearance_from_tensor(x) for x in images])).float()


def dataset_appearance_stats(dataset, n_samples: int = 64, seed: int = 0) -> torch.Tensor:
    """Appearance statistics of a few samples, for
    :meth:`~timemachine.model.appearance_film.AppearanceEncoder.fit_normalization`.

    The raw statistics are on very different scales, so the encoder standardizes
    them with buffers fitted once on the training data. Drawing whole samples
    (rather than random views) keeps the era mix the model will actually see.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    rows = [dataset[int(i)] for i in indices]
    if any("appearance" not in r for r in rows):
        raise KeyError("Samples carry no 'appearance' field; build the dataset with with_appearance=True.")
    return torch.cat([r["appearance"] for r in rows]).float()
