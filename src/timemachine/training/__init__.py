"""Fine-tuning date-conditioned SEVA (SEVA's own training code is not public).

conditioning  SEVA's conditioning tensors for one sample, built exactly as at inference
loss          the denoising loss (epsilon prediction on SEVA's discrete noise levels)
probes        fixed evaluation probes: date-swap loss gap, and renders of one view set at many dates
trainer       the training loop (config, optimizer, logging, checkpoints)
"""
