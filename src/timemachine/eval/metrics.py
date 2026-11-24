"""Evaluation metrics for novel view synthesis and baseline comparisons.

Images are expected to have values in [-1, 1] (the project standard),
shaped (C, H, W) or (H, W). Both NumPy arrays and PyTorch tensors are supported.

Metrics provided:
- psnr(pred, target): Peak Signal-to-Noise Ratio in dB.
- ssim(pred, target): Structural Similarity Index in [-1, 1].
- align_color(pred, target): Affine 3x3 + bias color calibration via least-squares.
- color_corrected_psnr(pred, target): PSNR after affine color alignment.
- color_corrected_ssim(pred, target): SSIM after affine color alignment.
- compute_metrics(pred, target, aligned=True): Dictionary with all scores.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import torch
except ImportError:
    torch = None


def _to_numpy(x) -> tuple[np.ndarray, bool, any, any]:
    """Convert tensor or array-like to float32 NumPy array while preserving metadata."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32), True, x.device, x.dtype
    return np.asarray(x, dtype=np.float32), False, None, None


def psnr(pred: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (dB) between images in [-1, 1].

    Dividing difference by 2 maps [-1, 1] to [0, 1] scale with peak 1.0.
    Floor on MSE is 1e-10 (capped at 100 dB for identical images).
    """
    p, _, _, _ = _to_numpy(pred)
    t, _, _, _ = _to_numpy(target)
    if p.shape != t.shape:
        raise ValueError(f"Image shapes must match: {p.shape} vs {t.shape}")
    mse = float(np.mean(((p - t) / 2.0) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-10)))


def align_color(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Affine color calibration (channels x channels matrix + bias) from pred to target.

    Solves min ||A @ pred + b - target||^2 via linear least-squares.
    Aligns overall color tint, exposure, or sepia fade while keeping structural geometry.
    Returns aligned prediction clipped to [-1, 1].
    """
    p, is_tensor, device, dtype = _to_numpy(pred)
    t, _, _, _ = _to_numpy(target)
    if p.shape != t.shape:
        raise ValueError(f"Image shapes must match: {p.shape} vs {t.shape}")

    orig_shape = p.shape
    channels = p.shape[0] if p.ndim == 3 else 1
    p_flat = p.reshape(channels, -1).T  # (N, C)
    t_flat = t.reshape(channels, -1).T  # (N, C)

    # Affine design matrix with bias column: [P, 1]
    a_mat = np.column_stack([p_flat, np.ones(len(p_flat), dtype=np.float32)])
    w_mat, _, _, _ = np.linalg.lstsq(a_mat, t_flat, rcond=None)

    aligned = np.clip((a_mat @ w_mat).T.reshape(orig_shape), -1.0, 1.0)
    if is_tensor and torch is not None:
        return torch.from_numpy(aligned).to(device=device, dtype=dtype)
    return aligned


def ssim(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    sigma: float = 1.5,
) -> float:
    """Structural Similarity Index (SSIM) between two images in [-1, 1].

    Uses separable Gaussian filtering (sigma=1.5, window standard).
    Returns mean SSIM across channels in range [-1, 1] (1.0 = identical).
    """
    p, _, _, _ = _to_numpy(pred)
    t, _, _, _ = _to_numpy(target)
    if p.shape != t.shape:
        raise ValueError(f"Image shapes must match: {p.shape} vs {t.shape}")

    # Map [-1, 1] to [0, 1]
    a = (p + 1.0) / 2.0
    b = (t + 1.0) / 2.0

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    channels = a.shape[0] if a.ndim == 3 else 1
    if a.ndim == 2:
        a, b = a[None], b[None]

    ssim_channels = []
    for c in range(channels):
        im1, im2 = a[c], b[c]
        mu1 = gaussian_filter(im1, sigma=sigma, mode="reflect")
        mu2 = gaussian_filter(im2, sigma=sigma, mode="reflect")

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = gaussian_filter(im1 * im1, sigma=sigma, mode="reflect") - mu1_sq
        sigma2_sq = gaussian_filter(im2 * im2, sigma=sigma, mode="reflect") - mu2_sq
        sigma12 = gaussian_filter(im1 * im2, sigma=sigma, mode="reflect") - mu1_mu2

        num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_channels.append(float(np.mean(num / den)))

    return float(np.mean(ssim_channels))


def color_corrected_psnr(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
) -> float:
    """PSNR (dB) after affine color calibration."""
    return psnr(align_color(pred, target), target)


def color_corrected_ssim(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    sigma: float = 1.5,
) -> float:
    """SSIM after affine color calibration."""
    return ssim(align_color(pred, target), target, sigma=sigma)


def compute_metrics(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    aligned: bool = True,
) -> dict[str, float]:
    """Compute standard metrics (PSNR, SSIM, and optionally color-corrected versions).

    Args:
        pred: Predicted image (C, H, W) in [-1, 1].
        target: Target image (C, H, W) in [-1, 1].
        aligned: Whether to also compute color-calibrated PSNR and SSIM.

    Returns:
        Dictionary with keys 'psnr', 'ssim', and (if aligned=True) 'cc_psnr', 'cc_ssim'.
    """
    res = {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
    }
    if aligned:
        al = align_color(pred, target)
        res["cc_psnr"] = psnr(al, target)
        res["cc_ssim"] = ssim(al, target)
    return res
