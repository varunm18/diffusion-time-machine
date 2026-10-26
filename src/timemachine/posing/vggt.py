"""Thin wrapper around VGGT-Omega (feed-forward camera prediction).

VGGT-Omega is used from an existing checkout, read-only: point the
``VGGT_OMEGA_ROOT`` environment variable at the repository (it is put on
``sys.path``; nothing is installed). The checkpoint path defaults to the
Hugging Face cache (``$HF_HOME/hub/models--facebook--VGGT-Omega``).

Output convention: camera-to-world matrices in VGGT's frame (first image at the
origin, arbitrary scale) and focal lengths in pixels of the *original* image.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CHECKPOINT_NAME = "vggt_omega_1b_512.pt"


def default_checkpoint() -> Path | None:
    """The 1B/512 checkpoint in the Hugging Face cache, if present."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    hits = sorted(hub.glob(f"models--facebook--VGGT-Omega/snapshots/*/{CHECKPOINT_NAME}"))
    return hits[-1] if hits else None


def _import_vggt_omega():
    """Import VGGT-Omega from ``VGGT_OMEGA_ROOT`` without writing anything into that checkout."""
    root = os.environ.get("VGGT_OMEGA_ROOT")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # no __pycache__ files in someone else's directory
    try:
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils import load_fn
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ImportError as exc:
        raise ImportError("vggt_omega not importable; set VGGT_OMEGA_ROOT to the VGGT-Omega checkout") from exc
    finally:
        sys.dont_write_bytecode = previous
    return VGGTOmega, load_fn, encoding_to_camera


class VGGTOmegaRunner:
    """Predict cameras for a list of images in one forward pass."""

    def __init__(self, checkpoint: Path | None = None, device: str = "cuda", resolution: int = 512):
        VGGTOmega, self._load_fn, self._encoding_to_camera = _import_vggt_omega()
        checkpoint = checkpoint or default_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError(f"No {CHECKPOINT_NAME} found; pass checkpoint explicitly")
        self.device = device
        self.resolution = resolution
        self.model = VGGTOmega().to(device).eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    def _resize_factor(self, path: str) -> float:
        """Original-pixel -> network-pixel scale, mirroring VGGT-Omega's preprocessing."""
        image = Image.open(path)
        cropped = self._load_fn._crop_to_supported_aspect_ratio(image)
        _, target_w = self._load_fn._balanced_target_shape(cropped.width / cropped.height,
                                                           self.resolution, 16)
        return target_w / cropped.width

    @torch.inference_mode()
    def predict(self, image_paths: list[str]) -> dict[str, np.ndarray]:
        """Returns ``c2w`` (N, 4, 4) in VGGT's frame and ``focal`` (N,) in original pixels."""
        images = self._load_fn.load_and_preprocess_images(image_paths, image_resolution=self.resolution)
        predictions = self.model(images.to(self.device))
        extrinsics, intrinsics = self._encoding_to_camera(predictions["pose_enc"],
                                                          predictions["images"].shape[-2:])
        w2c = np.tile(np.eye(4), (len(image_paths), 1, 1))
        w2c[:, :3, :] = extrinsics.reshape(-1, 3, 4).double().cpu().numpy()
        focal_net = intrinsics.reshape(-1, 3, 3)[:, 0, 0].double().cpu().numpy()
        scale = np.array([self._resize_factor(p) for p in image_paths])
        return {"c2w": np.linalg.inv(w2c), "focal": focal_net / scale}
