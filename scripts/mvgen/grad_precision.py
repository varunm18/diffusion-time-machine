#!/usr/bin/env python
"""Gate test before training MVGenMaster: is its training gradient sound in half precision?

The 2026-11-19 audit found that SEVA's gradient is rounding noise in fp16 and bf16 (norm ~1,000
vs ~1 in fp32, cosine ~0), so nothing trained. MVGenMaster's authors train with fp16 autocast.
This script builds one real training step (as in MVGenMaster's ``train.py``: 3 clean reference
views, 5 noised targets, camera rays, depth-warped 3D priors, v-prediction loss on the targets)
from a view set exported by ``scripts/export_nvs_samples.py``, and compares the gradient under
fp16 and bf16 autocast with the fp32 gradient, at several timesteps. The priors are computed as the
training step computes them, from depth whose far values are contracted once
(``timemachine.mvgen.step.depth_for_priors``); the results recorded on 2026-11-19 predate this and
had the warped pixels from depth contracted twice.

Gradients are taken for three groups: the domain-switcher embedding (next to where a date
embedding will go), the time-embedding projection of every ResBlock (what a date embedding
feeds), and the self-attention output projections (typical LoRA targets).

    SCRIPT=$PWD/scripts/mvgen/grad_precision.py ARGS="--sample $PWD/data/experiments/mvgen/samples/easy0 \\
        --out $PWD/data/experiments/mvgen/grad_precision.json" sbatch slurm/mvgen_gpu.sbatch
"""
import argparse
import json
import sys
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from easydict import EasyDict
from omegaconf import OmegaConf

from depth_pro.depth_pro import create_model_and_transforms
from my_diffusers.models import UNet2DConditionModel
from src.modules.camera import get_camera_embedding
from src.modules.position_encoding import get_3d_priors
from src.modules.schedulers import get_diffusion_scheduler
from timemachine.mvgen.step import depth_for_priors

sys.path.insert(0, str(Path(__file__).parent))
from nvs_test import load_sample, normalise_scale, reference_depth  # noqa: E402

GROUPS = {"domain_embedding": lambda n: n.startswith("class_embedding"),
          "resblock_time_proj": lambda n: "time_emb_proj" in n,
          "self_attn_out": lambda n: "attn1.to_out.0" in n}
AUTOCAST = {"fp32": dict(enabled=False), "fp16": dict(dtype=torch.float16), "bf16": dict(dtype=torch.bfloat16)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-dir", default="check_points/pretrained_model")
    p.add_argument("--timesteps", type=int, nargs="+", default=[50, 300, 600, 900, 999])
    p.add_argument("--class-label", type=int, default=0)
    args = p.parse_args()
    device = "cuda"

    config = EasyDict(OmegaConf.load(f"{args.model_dir}/config.yaml"))
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet", rank=0,
                                                model_cfg=config.model_cfg, low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True, local_files_only=True)
    unet.load_state_dict(torch.load(f"{args.model_dir}/ema_unet.pt", map_location="cpu"))
    vae.requires_grad_(False).to(device)
    unet.to(device).float().train()                      # fp32 master weights, as in training
    unet.enable_gradient_checkpointing()
    scheduler = get_diffusion_scheduler(config, name="DDPM")
    print("prediction type:", scheduler.config.prediction_type, flush=True)

    params = {g: [] for g in GROUPS}
    for name, param in unet.named_parameters():
        group = next((g for g, match in GROUPS.items() if match(name)), None)
        param.requires_grad_(group is not None)
        if group is not None:
            params[group].append(param)
    print("gradient groups:", {g: f"{sum(q.numel() for q in ps) / 1e6:.2f}M" for g, ps in params.items()}, flush=True)

    # one training sample: all 8 real views; the first 3 are references
    s = load_sample(args.sample)
    n_in, f = sum(s["is_input"]), len(s["is_input"])
    H, W = s["images"].shape[-2:]
    depth_model, transform = create_model_and_transforms(device=torch.device(device))
    depth_model.eval()
    depth = torch.zeros(f, 1, H, W)
    for j in range(n_in):
        depth[j, 0] = torch.from_numpy(reference_depth(depth_model, transform, s["images"][j], s["K"][j], s["sparse"][j])[0])
    del depth_model
    torch.cuda.empty_cache()
    w2c, depth, _, _ = normalise_scale(s["w2c"], depth, config.camera_longest_side)
    image = s["images"][None].to(device)                                     # [1, f, 3, H, W]
    K, w2c = s["K"].to(device), w2c.to(device)
    with torch.no_grad():
        latents = vae.encode(image[0]).latent_dist.mode()[None] * vae.config.scaling_factor     # [1, f, 4, h, w]
        masks = torch.ones((1, f, 1, H, W), device=device)
        latent_masks = torch.ones((1, f, 1, *latents.shape[-2:]), device=device)
        masks[:, :n_in], latent_masks[:, :n_in] = 0, 0
        camera = get_camera_embedding(K, w2c, 1, f, H, W, config=config).to(device=device, dtype=latents.dtype)
        add_inputs = torch.cat([masks, camera], dim=2)
        # a private float64 copy: each prior then contracts far depth once in its own copy, as in the training step
        coords = get_3d_priors(config, depth_for_priors(depth, device), K, w2c, n_in, nframe=f, device=device,
                               colors=image[0], latents=latents, vae=vae,
                               prior_type=config.model_cfg.get("prior_type", "3dpe"))
    class_labels = torch.tensor([args.class_label], dtype=torch.long, device=device)

    def gradient(t: int, noise: torch.Tensor, precision: str):
        unet.zero_grad(set_to_none=True)
        timesteps = torch.tensor([t], device=device, dtype=torch.long)
        noisy = scheduler.add_noise(latents, noise, timesteps)
        noisy = latents * (1 - latent_masks) + noisy * latent_masks
        target = scheduler.get_velocity(latents, noise, timesteps)
        prior = [c.clone() for c in coords] if isinstance(coords, list) else coords.clone()
        with torch.autocast("cuda", **AUTOCAST[precision]):
            pred = unet(noisy, timesteps, None, add_inputs=add_inputs, class_labels=class_labels, coords=prior,
                        return_dict=False)[0]
        pred = einops.rearrange(pred, "(b f) c h w -> b f c h w", f=f)
        loss = F.mse_loss(pred.float()[:, n_in:], target.float()[:, n_in:])
        loss.backward()
        return float(loss), {g: torch.cat([q.grad.detach().float().flatten() for q in ps]) for g, ps in params.items()}

    rows = []
    for t in args.timesteps:
        noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(t)).to(device)
        ref_loss, ref = gradient(t, noise, "fp32")
        _, ref2 = gradient(t, noise, "fp32")                                   # run-to-run reproducibility of fp32 itself
        for precision in ("fp32", "fp16", "bf16"):
            loss, g = (ref_loss, ref2) if precision == "fp32" else gradient(t, noise, precision)
            row = {"timestep": t, "precision": precision, "loss": loss}
            for group in GROUPS:
                row[f"{group}_norm"] = float(g[group].norm())
                row[f"{group}_cos_fp32"] = float(F.cosine_similarity(g[group], ref[group], dim=0))
            rows.append(row)
            print(f"t {t:4d} {precision}: loss {loss:.5f} | " + " | ".join(
                f"{group}: norm {row[f'{group}_norm']:.3g}, cos with fp32 {row[f'{group}_cos_fp32']:+.3f}" for group in GROUPS),
                flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    print("\nmean cosine with the fp32 gradient:")
    for precision in ("fp16", "bf16"):
        sel = [r for r in rows if r["precision"] == precision]
        print(f"   {precision}: " + ", ".join(f"{g} {np.mean([r[f'{g}_cos_fp32'] for r in sel]):+.3f}" for g in GROUPS), flush=True)


if __name__ == "__main__":
    main()
