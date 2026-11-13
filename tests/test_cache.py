import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from timemachine.data.cache import (  # noqa: E402
    CLIP_MEAN, CLIP_STD, cache_is_current, encode_views, psnr, write_cache,
)

GREYS = (10, 90, 170, 250, 50)


def _views(tmp_path) -> pd.DataFrame:
    """Five flat grey images of different brightness, with matching cameras."""
    rows = []
    for i, grey in enumerate(GREYS):
        path = tmp_path / f"img{i}.png"
        Image.new("RGB", (800, 600), (grey, grey, grey)).save(path)
        rows.append({"file_key": f"img{i}.png", "image_path": str(path), "image_width": 800, "image_height": 600,
                     "camera_width": 1600, "camera_height": 1200, "fx": 1500.0, "fy": 1500.0, "cx": 800.0, "cy": 600.0})
    return pd.DataFrame(rows)


def _fake_encode(images):
    """Stand-in encoders: every output value is the image's mean pixel value."""
    mean = images.mean(dim=(1, 2, 3))
    return mean[:, None, None, None].expand(-1, 4, 72, 72), mean[:, None].expand(-1, 1024)


def test_encode_views_keeps_row_order(tmp_path):
    latents, clip = encode_views(_views(tmp_path), _fake_encode, batch_size=2, workers=0)
    assert latents.shape == (5, 4, 72, 72) and clip.shape == (5, 1024)
    assert latents.dtype == np.float16 and clip.dtype == np.float16
    expected = np.array(GREYS) / 127.5 - 1.0          # images are scaled to [-1, 1]
    assert np.allclose(latents[:, 0, 0, 0], expected, atol=1e-3)
    assert np.allclose(clip[:, 0], expected, atol=1e-3)


def test_cache_roundtrip_and_key_check(tmp_path):
    keys = ["a.jpg", "Cathédrale b.jpg", "c.jpg"]
    latents = np.random.default_rng(0).normal(size=(3, 4, 72, 72))
    clip = np.random.default_rng(1).normal(size=(3, 1024))
    assert not cache_is_current(tmp_path, keys)
    write_cache(tmp_path, keys, latents, clip)
    assert cache_is_current(tmp_path, keys)
    assert not cache_is_current(tmp_path, keys[::-1])            # same files, other order
    assert not cache_is_current(tmp_path, keys[:2])              # a view was dropped since
    stored = np.load(tmp_path / "latents.npy", mmap_mode="r")
    assert stored.dtype == np.float16 and np.allclose(stored, latents, atol=2e-3)


def test_write_cache_rejects_misaligned_rows(tmp_path):
    with pytest.raises(ValueError):
        write_cache(tmp_path, ["a.jpg", "b.jpg"], np.zeros((3, 4, 72, 72)), np.zeros((3, 1024)))
    assert not cache_is_current(tmp_path, ["a.jpg", "b.jpg"])


def test_clip_preprocess_matches_seva_formula():
    pytest.importorskip("kornia")
    from timemachine.data.cache import clip_preprocess

    out = clip_preprocess(torch.zeros(2, 3, 576, 576))             # mid-grey in [-1, 1]
    assert out.shape == (2, 3, 224, 224)
    expected = (0.5 - torch.tensor(CLIP_MEAN)) / torch.tensor(CLIP_STD)
    assert torch.allclose(out[0].mean(dim=(1, 2)), expected, atol=1e-5)


def test_psnr():
    a = torch.zeros(3, 8, 8)
    assert psnr(a, a) == float("inf")
    assert abs(psnr(a, a + 0.2) - 10 * np.log10(4 / 0.04)) < 1e-4
