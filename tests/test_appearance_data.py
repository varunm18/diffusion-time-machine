"""Tests for per-view appearance statistics (timemachine.data.appearance)."""
import numpy as np
import pytest
import torch
from PIL import Image

from timemachine.analysis import shortcut as sc
from timemachine.data.appearance import (
    APPEARANCE_FEATURES, N_APPEARANCE_FEATURES, appearance_batch, appearance_from_tensor,
    dataset_appearance_stats,
)


def to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def noise_image(rng, size=64, brightness=0.5, contrast=0.2):
    lum = np.clip(rng.normal(brightness, contrast, (size, size)), 0, 1)
    return Image.fromarray((np.stack([lum] * 3, -1) * 255).astype(np.uint8), "RGB")


def test_features_match_the_measurement_code_exactly():
    """Conditioning and measurement must not drift apart: same numbers, same code."""
    rng = np.random.default_rng(0)
    for image in (noise_image(rng), sc.apply_sepia(noise_image(rng)),
                  Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB")):
        direct = sc.feature_matrix([sc.appearance_features(image)])[0]
        np.testing.assert_allclose(appearance_from_tensor(to_tensor(image)), direct, rtol=1e-5, atol=1e-5)
    assert APPEARANCE_FEATURES == sc.FEATURE_NAMES


def test_batch_shape_dtype_and_finiteness():
    rng = np.random.default_rng(1)
    images = torch.stack([to_tensor(noise_image(rng)) for _ in range(5)])
    feats = appearance_batch(images)
    assert feats.shape == (5, N_APPEARANCE_FEATURES)
    assert feats.dtype == torch.float32 and torch.isfinite(feats).all()


def test_extreme_pixel_values_do_not_break_the_conversion():
    """Pixels can sit exactly at the ends of [-1, 1], and slightly outside after resizing."""
    for value in (-1.0, 1.0, -1.2, 1.2):
        feats = appearance_from_tensor(torch.full((3, 16, 16), value))
        assert np.isfinite(feats).all()


def test_features_are_sensitive_to_the_medium_and_not_to_the_scene():
    rng = np.random.default_rng(2)
    scene = noise_image(rng)
    plain, toned = appearance_from_tensor(to_tensor(scene)), appearance_from_tensor(to_tensor(sc.apply_sepia(scene)))
    minor = APPEARANCE_FEATURES.index("chroma_minor")
    major = APPEARANCE_FEATURES.index("chroma_major")
    assert toned[major] > plain[major] + 1          # toning adds a strong single-axis cast
    assert toned[major] > 3 * toned[minor]          # along one axis only

    other_scene = noise_image(rng, brightness=0.5, contrast=0.2)
    same_medium = appearance_from_tensor(to_tensor(other_scene))
    # Two different grey scenes shot the same way look alike to these statistics,
    # which is the point: they describe the photograph, not what is in it.
    assert abs(same_medium[minor] - plain[minor]) < abs(toned[minor] - plain[minor]) + 1e-6


def test_dataset_stats_collects_rows_and_reports_a_missing_field():
    class FakeDataset:
        def __init__(self, n, with_appearance=True):
            self.n, self.with_appearance = n, with_appearance

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            sample = {"images": torch.zeros(8, 3, 4, 4)}
            if self.with_appearance:
                sample["appearance"] = torch.full((8, N_APPEARANCE_FEATURES), float(i))
            return sample

    stats = dataset_appearance_stats(FakeDataset(20), n_samples=5, seed=0)
    assert stats.shape == (5 * 8, N_APPEARANCE_FEATURES)        # every view of every drawn sample
    assert torch.equal(stats, dataset_appearance_stats(FakeDataset(20), n_samples=5, seed=0))
    assert len(dataset_appearance_stats(FakeDataset(3), n_samples=50)) == 3 * 8   # asking for too many

    with pytest.raises(KeyError, match="with_appearance=True"):
        dataset_appearance_stats(FakeDataset(4, with_appearance=False), n_samples=2)
