"""SEVA's conditioning for one multi-view sample, matching ``seva.eval.do_sample``.

At inference SEVA builds, for T views of which some are inputs:

* ``crossattn``    (T, 1, 1024)  CLIP image embedding *averaged over the inputs*, same for all views
* ``concat``       (T, 7, h, w)  [input mask, Plücker rays], concatenated to the noisy latent
* ``dense_vector`` (T, 6, h, w)  Plücker rays, modulating every ResBlock
* ``replace``      (T, 5, h, w)  [clean latent, 1] on input views, zeros elsewhere: the denoiser
                                  swaps the noisy latent of input views for their clean latent

and an unconditional version (no CLIP, zero mask, no replacement; the cameras
stay) for classifier-free guidance. Training must feed the network the same
tensors, so this module mirrors that code; :func:`camera_rays` reuses SEVA's own
``get_value_dict`` so the camera normalization is identical by construction.
"""
from __future__ import annotations

import torch
from einops import repeat
from seva.eval import get_value_dict

from timemachine.model.loading import SevaComponents


def camera_rays(c2w: torch.Tensor, K: torch.Tensor, n_inputs: int, height: int, width: int,
                camera_scale: float = 2.0) -> torch.Tensor:
    """(T, 6, h, w) Plücker rays at latent resolution, normalized as SEVA does (CPU)."""
    T = c2w.shape[0]
    value_dict = get_value_dict(
        curr_imgs=torch.zeros(T, 3, height, width), curr_input_frame_indices=list(range(n_inputs)),
        curr_c2ws=c2w.float().cpu(), curr_Ks=K.float().cpu(), curr_input_camera_indices=list(range(n_inputs)),
        all_c2ws=c2w.float().cpu(), camera_scale=camera_scale)
    return value_dict["plucker_coordinate"]


@torch.no_grad()
def encode_sample(components: SevaComponents, images: torch.Tensor, is_input: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """VAE latents of all views (T, 4, h, w) and the CLIP embedding averaged over inputs (1024,)."""
    latents = components.ae.encode(images, 1)
    clip = components.conditioner(images[is_input]).mean(0)
    return latents, clip


def build_conditions(latents: torch.Tensor, clip: torch.Tensor, plucker: torch.Tensor,
                     is_input: torch.Tensor) -> tuple[dict, dict]:
    """``(cond, uncond)`` dicts exactly as built in ``seva.eval.do_sample``."""
    T, _, h, w = latents.shape
    mask = repeat(is_input.float(), "n -> n 1 h w", h=h, w=w)       # fp32, like do_sample
    replace = torch.zeros(T, latents.shape[1] + 1, h, w, dtype=latents.dtype, device=latents.device)
    replace[is_input] = torch.cat([latents[is_input], torch.ones_like(latents[is_input, :1])], 1)
    crossattn = repeat(clip, "d -> n 1 d", n=T)
    plucker = plucker.to(latents.device).float()
    cond = {"crossattn": crossattn, "replace": replace,
            "concat": torch.cat([mask, plucker], 1), "dense_vector": plucker}
    uncond = {"crossattn": torch.zeros_like(crossattn), "replace": torch.zeros_like(replace),
              "concat": torch.cat([torch.zeros_like(mask), plucker], 1), "dense_vector": plucker}
    return cond, uncond


def sample_conditions(components: SevaComponents, sample: dict, device: str = "cuda",
                      camera_scale: float = 2.0) -> tuple[torch.Tensor, dict, dict]:
    """Latents plus ``(cond, uncond)`` for a dataset sample (see ``timemachine.data.dataset``)."""
    images = sample["images"].to(device)
    is_input = sample["is_input"].to(device)
    _, _, H, W = images.shape
    plucker = camera_rays(sample["c2w"], sample["K"], int(is_input.sum()), H, W, camera_scale)
    with torch.autocast(device):
        latents, clip = encode_sample(components, images, is_input)
    cond, uncond = build_conditions(latents, clip, plucker.to(device), is_input)
    return latents, cond, uncond

