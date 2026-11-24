"""Tests for evaluation metrics (PSNR, SSIM, and color-corrected versions)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from timemachine.eval.metrics import (
    align_color,
    color_corrected_psnr,
    color_corrected_ssim,
    compute_metrics,
    psnr,
    ssim,
)


def test_psnr_identical():
    """Identical images should hit the 100 dB cap."""
    a = np.zeros((3, 16, 16), dtype=np.float32)
    assert psnr(a, a) == pytest.approx(100.0)


def test_psnr_known_value():
    """Known difference: delta 0.2 in [-1, 1] is 0.1 in [0, 1] -> 20 dB."""
    a = np.zeros((3, 16, 16), dtype=np.float32)
    b = np.full((3, 16, 16), 0.2, dtype=np.float32)
    assert psnr(a, b) == pytest.approx(20.0)


def test_psnr_shape_mismatch():
    """Mismatched image shapes must raise ValueError."""
    a = np.zeros((3, 16, 16), dtype=np.float32)
    b = np.zeros((3, 16, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="Image shapes must match"):
        psnr(a, b)


def test_ssim_identical():
    """Identical images must yield an SSIM of 1.0."""
    rng = np.random.default_rng(42)
    a = rng.uniform(-1.0, 1.0, (3, 32, 32)).astype(np.float32)
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-4)


def test_ssim_different():
    """Significantly different images should have low SSIM."""
    a = np.full((3, 32, 32), 1.0, dtype=np.float32)
    b = np.full((3, 32, 32), -1.0, dtype=np.float32)
    score = ssim(a, b)
    assert score < 0.1


def test_align_color_recovers_affine_shift():
    """Align color should invert affine color/brightness transformations."""
    rng = np.random.default_rng(123)
    target = rng.uniform(-0.8, 0.8, (3, 32, 32)).astype(np.float32)
    # Simulate vintage fade: scale down contrast and add tint
    pred = target * 0.6 - 0.2

    aligned = align_color(pred, target)
    # After alignment, MSE to target should be virtually zero
    assert np.mean((aligned - target) ** 2) < 1e-6
    assert psnr(aligned, target) > 60.0


def test_color_corrected_metrics_improve_on_color_shift():
    """Color-corrected PSNR and SSIM should be significantly higher than raw when color is shifted."""
    rng = np.random.default_rng(77)
    target = rng.uniform(-0.7, 0.7, (3, 32, 32)).astype(np.float32)
    pred = target * 0.5 - 0.3

    raw_p = psnr(pred, target)
    cc_p = color_corrected_psnr(pred, target)
    raw_s = ssim(pred, target)
    cc_s = color_corrected_ssim(pred, target)

    assert cc_p > raw_p + 15.0  # At least 15 dB improvement
    assert cc_s >= raw_s
    assert cc_s == pytest.approx(1.0, abs=1e-3)


def test_compute_metrics_dict():
    """compute_metrics should return dictionary with all expected keys."""
    rng = np.random.default_rng(99)
    target = rng.uniform(-1.0, 1.0, (3, 16, 16)).astype(np.float32)
    pred = rng.uniform(-1.0, 1.0, (3, 16, 16)).astype(np.float32)

    res = compute_metrics(pred, target, aligned=True)
    assert set(res.keys()) == {"psnr", "ssim", "cc_psnr", "cc_ssim"}
    for v in res.values():
        assert isinstance(v, float)

    res_no_align = compute_metrics(pred, target, aligned=False)
    assert set(res_no_align.keys()) == {"psnr", "ssim"}


def test_torch_tensor_support():
    """Metrics should work seamlessly on PyTorch tensors."""
    target = torch.rand(3, 24, 24) * 2.0 - 1.0
    pred = target * 0.8 - 0.1

    res = compute_metrics(pred, target)
    assert res["psnr"] > 0
    assert res["ssim"] > 0
    assert res["cc_psnr"] > res["psnr"]

    aligned_tensor = align_color(pred, target)
    assert isinstance(aligned_tensor, torch.Tensor)
    assert aligned_tensor.shape == pred.shape
    assert aligned_tensor.device == pred.device


def test_2d_grayscale_support():
    """Metrics should support 2D (H, W) or (1, H, W) images."""
    a = np.ones((16, 16), dtype=np.float32)
    b = np.full((16, 16), 0.5, dtype=np.float32)

    assert psnr(a, b) > 0
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-4)

    a_1ch = a[None, ...]
    b_1ch = b[None, ...]
    assert psnr(a_1ch, b_1ch) == pytest.approx(psnr(a, b))
