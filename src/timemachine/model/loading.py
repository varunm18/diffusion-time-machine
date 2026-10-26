"""Load SEVA and its helper networks from local Hugging Face snapshots.

SEVA's own loaders download from the Hub, and the SD 2.1-base VAE repository
they point to no longer exists. Here every component is loaded from a local hub
cache folder (read-only), passed explicitly or via ``TIMEMACHINE_HF_HUB``:

* SEVA weights      models--stabilityai--stable-virtual-camera
* VAE (SD 2.1 base) models--Manojb--stable-diffusion-2-1-base   (mirror)
* OpenCLIP ViT-H-14 models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import open_clip
import torch
from diffusers.models import AutoencoderKL
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model as load_seva_weights
from torch import nn

from timemachine.model.dated_seva import DatedSGMWrapper

SEVA_REPO = "stabilityai/stable-virtual-camera"
VAE_REPO = "Manojb/stable-diffusion-2-1-base"
CLIP_REPO = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def snapshot_dir(hub: Path, repo: str) -> Path:
    """Folder of the (single/latest) snapshot of ``repo`` in a Hugging Face hub cache."""
    snapshots = sorted((Path(hub) / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"{repo} not found in {hub}")
    return snapshots[-1]


def default_hub() -> Path:
    hub = os.environ.get("TIMEMACHINE_HF_HUB")
    if hub:
        return Path(hub)
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


class LocalAutoEncoder(AutoEncoder):
    """SEVA's VAE wrapper, loading the SD 2.1-base VAE from a local folder."""

    def __init__(self, vae_dir: Path, chunk_size: int | None = None):
        nn.Module.__init__(self)
        self.module = AutoencoderKL.from_pretrained(str(vae_dir), subfolder="vae", low_cpu_mem_usage=False)
        self.module.eval().requires_grad_(False)
        self.chunk_size = chunk_size


class LocalCLIPConditioner(CLIPConditioner):
    """SEVA's CLIP image encoder, loading OpenCLIP ViT-H-14 weights from a local file."""

    def __init__(self, weights: Path):
        nn.Module.__init__(self)
        self.module = open_clip.create_model_and_transforms("ViT-H-14", pretrained=str(weights))[0]
        self.module.eval().requires_grad_(False)
        self.register_buffer("mean", torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer("std", torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)


@dataclass
class SevaComponents:
    model: SGMWrapper          # SGMWrapper, or DatedSGMWrapper when dated=True
    ae: AutoEncoder
    conditioner: CLIPConditioner
    denoiser: DiscreteDenoiser


def load_components(hub: Path | None = None, version: float = 1.1, device: str = "cuda",
                    dated: bool = False) -> SevaComponents:
    """SEVA (optionally with the date module), VAE, CLIP encoder and denoiser."""
    hub = Path(hub) if hub else default_hub()
    weight_name = "modelv1.1.safetensors" if version > 1 else "model.safetensors"
    seva = load_seva_weights(pretrained_model_name_or_path=str(snapshot_dir(hub, SEVA_REPO)),
                             weight_name=weight_name, device="cpu").eval()
    model = DatedSGMWrapper(seva) if dated else SGMWrapper(seva)
    if dated:  # the new module starts in fp32 on CPU; match SEVA's dtype
        model.date_module.date_embedding.to(next(seva.parameters()).dtype)
    ae = LocalAutoEncoder(snapshot_dir(hub, VAE_REPO), chunk_size=1)
    clip = LocalCLIPConditioner(snapshot_dir(hub, CLIP_REPO) / "open_clip_model.safetensors")
    return SevaComponents(model.to(device), ae.to(device), clip.to(device),
                          DiscreteDenoiser(num_idx=1000, device=device))
