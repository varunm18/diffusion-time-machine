"""Integration test: the real DatedViewsDataset emits per-view appearance statistics."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from timemachine.analysis import shortcut as sc
from timemachine.data.appearance import N_APPEARANCE_FEATURES, appearance_from_tensor
from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.sampling import SamplingConfig


@pytest.fixture
def scene(tmp_path):
    """A tiny one-model scene: old views are sepia, modern ones are colour."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        old = i < 10
        year = rng.uniform(1880, 1900) if old else rng.uniform(2010, 2020)
        # Smooth colour regions, not per-pixel noise: noise averages to grey when
        # the dataset downsamples, which would make the "colour" photos colourless.
        yy, xx = np.mgrid[0:64, 0:80] / 64.0
        rgb = np.stack([0.3 + 0.6 * xx, 0.4 + 0.4 * yy, 0.8 - 0.5 * xx], -1)
        rgb = np.clip(rgb + rng.normal(0, 0.03, rgb.shape), 0, 1)
        image = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
        if old:
            image = sc.apply_sepia(image)
        path = tmp_path / f"v{i:03d}.jpg"
        image.save(path, quality=95)
        angle = 2 * np.pi * i / 24
        rows.append(dict(
            scene_id=1, k1=0.0, k2=0.0,       # required by the multi-scene view schema
            file_key=f"v{i}", image_path=str(path), image_width=80, image_height=64,
            camera_width=80, camera_height=64, fx=70.0, fy=70.0, cx=40.0, cy=32.0,
            qw=1.0, qx=0.0, qy=0.0, qz=0.0,
            tx=float(3 * np.cos(angle)), ty=0.0, tz=float(3 * np.sin(angle)),
            date_lo=year - 0.5, date_hi=year + 0.5, date_mid=year,
            medium="photo", is_monochrome=bool(old), model=0,
        ))
    return pd.DataFrame(rows)


def make(views, **kwargs):
    return DatedViewsDataset(views, SamplingConfig(n_inputs=2, n_targets=3, old_tolerance=30.0,
                                                   modern_tolerance=30.0),
                             resolution=32, samples_per_epoch=4, seed=0, **kwargs)


def test_appearance_is_off_by_default_and_costs_nothing(scene):
    sample = make(scene)[0]
    assert "appearance" not in sample
    assert set(sample) >= {"images", "K", "c2w", "date", "is_input"}


def test_appearance_has_one_row_per_view_and_matches_the_pixels(scene):
    dataset = make(scene, with_appearance=True)
    sample = dataset[0]
    views = sample["images"].shape[0]

    assert sample["appearance"].shape == (views, N_APPEARANCE_FEATURES)
    assert sample["appearance"].dtype == torch.float32
    assert torch.isfinite(sample["appearance"]).all()
    # Every row is the statistics of that view's own pixels, in order.
    for i in range(views):
        np.testing.assert_allclose(sample["appearance"][i].numpy(),
                                   appearance_from_tensor(sample["images"][i]), rtol=1e-5, atol=1e-5)


def test_appearance_separates_the_old_views_inside_one_sample(scene):
    """The whole premise: these numbers must carry the medium difference."""
    dataset = make(scene, with_appearance=True)
    minor = sc.FEATURE_NAMES.index("chroma_minor")
    major = sc.FEATURE_NAMES.index("chroma_major")

    old_axis_ratio, modern_axis_ratio = [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        years = sample["date_mid"].numpy()
        feats = sample["appearance"].numpy()
        for year, f in zip(years, feats):
            ratio = f[major] / (f[minor] + 1e-6)      # high = toned along one colour axis
            (old_axis_ratio if year < 1945 else modern_axis_ratio).append(ratio)

    assert old_axis_ratio and modern_axis_ratio, "sampler produced no mixed-era samples"
    assert np.median(old_axis_ratio) > np.median(modern_axis_ratio)


def test_appearance_is_deterministic_for_a_given_sample(scene):
    a, b = make(scene, with_appearance=True), make(scene, with_appearance=True)
    assert torch.equal(a[2]["appearance"], b[2]["appearance"])
    assert a[2]["file_keys"] == b[2]["file_keys"]
