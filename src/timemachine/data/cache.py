"""Precomputed VAE latents and CLIP embeddings for dataset views.

Training would otherwise run SEVA's frozen VAE encoder and CLIP image encoder
on every view at every step. Both are deterministic functions of the 576x576
crop, so they can be computed once and stored:

* **Latents**: SEVA encodes with the *mean* of the VAE posterior times 0.18215
  (``seva/modules/autoencoder.py``), not a random sample, so a cached latent is
  exactly what an uncached run would compute. Stored as float16 (4, 72, 72).
* **CLIP**: SEVA resizes the [-1, 1] image to 224 (bicubic, antialias,
  align_corners), normalizes with CLIP's mean/std and calls OpenCLIP ViT-H-14
  ``encode_image`` (``seva/modules/conditioner.py``). Stored as float16 (1024,).
  SEVA averages these over a sample's input views; that stays in the model.

Images are preprocessed with :func:`timemachine.data.dataset.load_view`, the same
function the pixel dataset uses, so both paths see identical crops.

Per model, next to ``views.parquet``: ``latents.npy`` (N, 4, 72, 72),
``clip.npy`` (N, 1024) and ``cache_keys.json`` (the N file names, in row order).
Row *i* of both arrays belongs to row *i* of the view table; ``cache_keys.json``
lets a reader verify that before trusting the arrays.

The trade-off: cached latents rule out pixel augmentation (random crops, color
jitter). Horizontal flips are invalid anyway: mirroring a camera is not a rotation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from timemachine.data.dataset import load_view

RESOLUTION = 576
VAE_SCALE = 0.18215
VAE_REPO = "Manojb/stable-diffusion-2-1-base"      # mirror; the original SD 2.1-base repo is gone
CLIP_MODEL, CLIP_PRETRAINED = "ViT-H-14", "laion2b_s32b_b79k"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
LATENTS, CLIP, KEYS = "latents.npy", "clip.npy", "cache_keys.json"


class ViewImages(Dataset):
    """The 576x576 crops of a view table, in row order."""

    def __init__(self, views: pd.DataFrame, resolution: int = RESOLUTION):
        self.views = views.reset_index(drop=True)
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_view(self.views.iloc[index], self.resolution)[0]


def clip_preprocess(images: torch.Tensor) -> torch.Tensor:
    """SEVA's CLIP preprocessing: (B, 3, H, W) in [-1, 1] -> normalized (B, 3, 224, 224)."""
    import kornia

    x = kornia.geometry.resize(images, (224, 224), interpolation="bicubic", align_corners=True, antialias=True)
    x = (x + 1.0) / 2.0
    return kornia.enhance.normalize(x, torch.tensor(CLIP_MEAN, device=x.device), torch.tensor(CLIP_STD, device=x.device))


class Encoders:
    """SEVA's frozen VAE encoder and CLIP image encoder (loaded from the Hugging Face cache)."""

    def __init__(self, device: str = "cuda"):
        import open_clip
        from diffusers.models import AutoencoderKL

        self.device = device
        self.vae = AutoencoderKL.from_pretrained(VAE_REPO, subfolder="vae").to(device).eval().requires_grad_(False)
        scale = self.vae.config.scaling_factor
        if abs(scale - VAE_SCALE) > 1e-6:
            raise ValueError(f"VAE scaling factor is {scale}, expected {VAE_SCALE} (wrong VAE?)")
        self.clip = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)[0]
        self.clip = self.clip.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 3, 576, 576) in [-1, 1] -> latents (B, 4, 72, 72) and CLIP embeddings (B, 1024), float32."""
        images = images.to(self.device, dtype=torch.float32)
        latents = self.vae.encode(images).latent_dist.mean * VAE_SCALE
        return latents, self.clip.encode_image(clip_preprocess(images))

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Scaled latents -> images in [-1, 1] (used only to sanity-check the cache)."""
        return self.vae.decode(latents.to(self.device, dtype=torch.float32) / VAE_SCALE).sample


def cache_is_current(folder: Path, file_keys: list[str]) -> bool:
    """Do the arrays in ``folder`` exist and belong to exactly these rows, in this order?"""
    paths = [folder / LATENTS, folder / CLIP, folder / KEYS]
    return all(p.exists() for p in paths) and json.loads(paths[2].read_text(encoding="utf-8")) == file_keys


def write_cache(folder: Path, file_keys: list[str], latents: np.ndarray, clip: np.ndarray) -> None:
    """Save the arrays and their row keys; the key file is written last, so its presence means "complete"."""
    if not (len(file_keys) == len(latents) == len(clip)):
        raise ValueError(f"row counts differ: {len(file_keys)} keys, {len(latents)} latents, {len(clip)} clip")
    (folder / KEYS).unlink(missing_ok=True)
    np.save(folder / LATENTS, latents.astype(np.float16))
    np.save(folder / CLIP, clip.astype(np.float16))
    (folder / KEYS).write_text(json.dumps(file_keys, ensure_ascii=False), encoding="utf-8")


def encode_views(views: pd.DataFrame, encode, batch_size: int = 16, workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Latents and CLIP embeddings of every row of ``views``, in row order (float16).

    ``encode`` maps a batch of images to ``(latents, clip)`` tensors (see :meth:`Encoders.encode`).
    """
    loader = DataLoader(ViewImages(views), batch_size=batch_size, shuffle=False, num_workers=workers)
    latents, clip = [], []
    for images in loader:
        z, c = encode(images)
        latents.append(z.float().cpu().numpy().astype(np.float16))
        clip.append(c.float().cpu().numpy().astype(np.float16))
    return np.concatenate(latents), np.concatenate(clip)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR in dB between images in [-1, 1]."""
    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * float(np.log10(4.0 / mse))
