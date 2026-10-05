"""Global image descriptors for picking anchors that look like the queries.

A feed-forward model can only relate images that overlap. Picking anchors by
visual similarity (rather than spreading them over the scene) gives each batch
modern views of the same part of the building as its old queries. DINOv2
features are used because they are fairly robust to color vs. black-and-white
and to film grain, which matters for old photos.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "facebook/dinov2-base"


class GlobalDescriptor:
    """L2-normalized DINOv2 embedding: [CLS token, mean of patch tokens]."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel  # optional dependency

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def embed(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """(N, D) unit-norm descriptors."""
        out = []
        for start in range(0, len(image_paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in image_paths[start:start + batch_size]]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            tokens = self.model(**inputs).last_hidden_state            # (B, 1 + patches, D)
            feat = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=-1)
            out.append(torch.nn.functional.normalize(feat, dim=-1).float().cpu().numpy())
        return np.concatenate(out)
