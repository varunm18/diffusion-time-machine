"""Evaluation tools: baselines and metrics for judging the date-conditioned model.

retrieval
    Model-free "copy a real photo" baselines (nearest in date, nearest in camera),
    scored the same way as the plain-SEVA baseline so the two can be compared.
metrics
    PSNR, SSIM, and color-corrected (affine appearance-aligned) evaluation functions.
"""
from timemachine.eval.metrics import (
    align_color,
    color_corrected_psnr,
    color_corrected_ssim,
    compute_metrics,
    psnr,
    ssim,
)
