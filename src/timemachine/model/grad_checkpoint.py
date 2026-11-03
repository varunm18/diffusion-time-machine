"""Gradient checkpointing for SEVA's UNet (SEVA ships without it).

Each block of ``input_blocks``, ``middle_block`` and ``output_blocks`` gets a
forward that, while training with gradients enabled, runs under
``torch.utils.checkpoint`` (non-reentrant): its activations are dropped after the
forward pass and recomputed during backward. Trades ~30% more compute for a
large cut in activation memory, which is what lets 8-view, 576x576 training fit
on a 48 GB card.

The forward is overridden on the *instance*, so module names and state-dict
keys do not change, and eval / no-grad passes run the plain forward.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _checkpoint_forward(module: nn.Module) -> None:
    plain_forward = module.forward

    def forward(*args, **kwargs):
        if module.training and torch.is_grad_enabled():
            return checkpoint(plain_forward, *args, use_reentrant=False, **kwargs)
        return plain_forward(*args, **kwargs)

    module.forward = forward
    module._grad_checkpointed = True


def enable_gradient_checkpointing(seva: nn.Module) -> int:
    """Checkpoint every UNet block of a ``seva.model.Seva``. Returns the number of blocks."""
    blocks = [*seva.input_blocks, seva.middle_block, *seva.output_blocks]
    for block in blocks:
        if not getattr(block, "_grad_checkpointed", False):
            _checkpoint_forward(block)
    return len(blocks)
