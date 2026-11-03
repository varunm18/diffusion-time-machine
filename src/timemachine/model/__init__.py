"""Date-conditioned Stable Virtual Camera (SEVA).

Requires SEVA installed via ``scripts/setup_seva.sh`` (``seva`` extra).

date_embedding   per-view decimal year -> embedding (zero at init)
dated_seva       SEVA wrapper adding the date embedding to the timestep embedding
lora             low-rank adapters for SEVA's linear layers (zero at init)
grad_checkpoint  gradient checkpointing for SEVA's UNet blocks
finetune         freeze SEVA + add adapters; save/load the trained parameters
loading          load SEVA, VAE and CLIP from a local Hugging Face cache
inference        generate target views for one sample with SEVA's sampler
"""
