#!/usr/bin/env python
"""Test MVGenMaster on our own view sets: our COLMAP cameras, real target photos to compare with.

Runs in MVGenMaster's environment from its checkout (``slurm/mvgen_gpu.sbatch``), on sets written
by ``scripts/export_nvs_samples.py``. Unlike MVGenMaster's demo (which estimates poses with DUSt3R
and renders an interpolated path), the target cameras here are the real photos' cameras.

MVGenMaster needs a depth map per input view, to warp the inputs into the targets ("3D priors").
As in its training on SfM data, we take monocular depth (Depth-Pro, with our focal length) and
scale it to the view's COLMAP points (median ratio).

For every set, domain label (0 = regular datasets, 1 = MegaScenes) and noise seed it writes the
generated targets, the warped-input prior, a contact sheet and the PSNR against the real targets.

This harness keeps the numerics of MVGenMaster's demo (fp16 weights, autocast around the whole
pipeline, the sample's own scene scale) with one exception: depth is handed to the pipeline as
float64, so that far depth is contracted once for each 3D prior, as in the authors' training, and
not twice for the warped pixels, as when the demo hands over float32 depth on the GPU. Results
written before the evening of 2026-11-19 (the first ones under ``data/experiments/mvgen/nvs_test``,
and the released-model baselines quoted from them) were made with the double contraction. The
fine-tuning code renders through ``timemachine.mvgen.probes.render_views`` instead, which has the
numerics of a training step.

    SCRIPT=$PWD/scripts/mvgen/nvs_test.py ARGS="--samples $PWD/data/experiments/mvgen/samples \\
        --out $PWD/data/experiments/mvgen/nvs_test" sbatch slurm/mvgen_gpu.sbatch
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from easydict import EasyDict
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from depth_pro.depth_pro import create_model_and_transforms
from my_diffusers.models import UNet2DConditionModel
from my_diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_multiview import StableDiffusionMultiViewPipeline
from src.modules.position_encoding import global_position_encoding_3d
from src.modules.schedulers import get_diffusion_scheduler


def load_pipeline(model_dir: str, device: str = "cuda"):
    """The released model, loaded as in MVGenMaster's ``run_mvgen.py`` (fp16 weights, DDIM)."""
    config = EasyDict(OmegaConf.load(f"{model_dir}/config.yaml"))
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae", local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet", rank=0,
                                                model_cfg=config.model_cfg, low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True, local_files_only=True)
    weights = torch.load(f"{model_dir}/ema_unet.pt", map_location="cpu")
    print("released UNet weights:", {str(v.dtype) for v in weights.values()},
          f"{sum(v.numel() for v in weights.values()) / 1e6:.0f}M parameters", flush=True)
    unet.load_state_dict(weights)
    vae.requires_grad_(False).to(device, dtype=torch.float16)
    unet.requires_grad_(False).eval().to(device, dtype=torch.float16)
    pipeline = StableDiffusionMultiViewPipeline.from_pretrained(
        config.pretrained_model_name_or_path, vae=vae, unet=unet, scheduler=get_diffusion_scheduler(config, name="DDIM"),
        safety_checker=None, torch_dtype=torch.float16, local_files_only=True).to(device)
    print("scheduler:", pipeline.scheduler.config.prediction_type, pipeline.scheduler.config.beta_schedule, flush=True)
    return config, pipeline


def load_sample(folder: Path) -> dict:
    meta = json.loads((folder / "sample.json").read_text())
    views = meta["views"]
    images = torch.stack([torch.from_numpy(np.asarray(Image.open(folder / f"view_{j}.png").convert("RGB"), dtype=np.float32)
                                           / 127.5 - 1).permute(2, 0, 1) for j in range(len(views))])
    c2w = torch.tensor([v["c2w"] for v in views], dtype=torch.float32)
    return {"images": images, "K": torch.tensor([v["K"] for v in views], dtype=torch.float32),
            "w2c": torch.linalg.inv(c2w), "is_input": [v["is_input"] for v in views],
            "years": [v["year"] for v in views],
            "sparse": [np.load(folder / f"sparse_{j}.npy") for j in range(len(views))]}


@torch.no_grad()
def reference_depth(depth_model, transform, image: torch.Tensor, K: torch.Tensor, sparse: np.ndarray):
    """Depth-Pro depth of one input view, scaled to its COLMAP points. Returns (depth (H, W), stats)."""
    rgb = ((image.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    mono = depth_model.infer(transform(rgb), f_px=torch.tensor(float(K[0, 0])))["depth"].float().cpu().numpy()
    u, v = sparse[:, 0].round().astype(int).clip(0, mono.shape[1] - 1), sparse[:, 1].round().astype(int).clip(0, mono.shape[0] - 1)
    ratio = sparse[:, 2] / np.clip(mono[v, u], 1e-6, None)
    scale = float(np.median(ratio))
    spread = float(np.median(np.abs(ratio / scale - 1)))        # relative disagreement after scaling
    return mono * scale, {"points": int(len(sparse)), "scale": scale, "median_rel_error": spread}


def normalise_scale(w2c: torch.Tensor, depth: torch.Tensor, longest: float):
    """As ``run_mvgen.load_dataset``: shrink the scene if the cameras span more than ``longest``."""
    centres = torch.linalg.inv(w2c)[:, :3, 3]
    size = float((centres.max(0).values - centres.min(0).values).max())
    rescale = longest / size if size > longest else 1.0
    w2c = w2c.clone()
    w2c[:, :3, 3] *= rescale
    return w2c, depth * rescale, size, rescale


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float((((a.astype(np.float32) - b.astype(np.float32)) / 255) ** 2).mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=Path, required=True, help="folder of exported sets")
    p.add_argument("--sets", nargs="*", help="set names (default: all)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-dir", default="check_points/pretrained_model")
    p.add_argument("--class-labels", type=int, nargs="+", default=[0, 1])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0],
                   help="extra factors on camera translations and depth, after the longest-side rule: MVGenMaster's "
                        "training scales a scene by the extent of ALL its cameras, its demo by the sample's")
    args = p.parse_args()

    config, pipeline = load_pipeline(args.model_dir)
    depth_model, transform = create_model_and_transforms(device=torch.device("cuda"))
    depth_model.eval()
    names = args.sets or sorted(d.name for d in args.samples.iterdir() if (d / "sample.json").exists())
    results, T = [], 128
    for name in names:
        s = load_sample(args.samples / name)
        n_in, n = sum(s["is_input"]), len(s["is_input"])
        H, W = s["images"].shape[-2:]
        depths, stats = zip(*(reference_depth(depth_model, transform, s["images"][j], s["K"][j], s["sparse"][j])
                              for j in range(n_in)))
        depth = torch.zeros(n, 1, H, W)
        depth[:n_in, 0] = torch.from_numpy(np.stack(depths))
        w2c, depth, size, rescale = normalise_scale(s["w2c"], depth, config.camera_longest_side)
        print(f"{name}: years {[round(y) for y in s['years']]}, camera extent {size:.2f} (rescale {rescale:.3f}), "
              f"depth alignment {[(d['points'], round(d['median_rel_error'], 3)) for d in stats]}", flush=True)
        cfg_run = copy.deepcopy(config)
        cfg_run.nframe = n
        images = torch.cat([s["images"][:n_in], torch.zeros(n - n_in, 3, H, W)]).cuda()      # targets' pixels are not given
        real = ((s["images"].permute(0, 2, 3, 1).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        rows = [("REAL (inputs | targets)", real)]
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for scale in args.scales:
            w2c_s, depth_s = w2c.clone(), depth * scale
            w2c_s[:, :3, 3] *= scale
            suffix = "" if scale == 1.0 else f"_scale{scale:g}"
            for label in args.class_labels:
                for seed in args.seeds:
                    with torch.no_grad(), torch.autocast("cuda"):
                        preds = pipeline(images=images, nframe=n, cond_num=n_in, key_rescale=None, height=H, width=W,
                                         intrinsics=s["K"].cuda(), extrinsics=w2c_s.cuda(),
                                         num_inference_steps=args.steps, guidance_scale=args.cfg, output_type="np",
                                         config=cfg_run, tag=["custom"] * n, class_label=label,
                                         depth=depth_s.double().cuda(),      # see the module docstring
                                         vae=pipeline.vae, generator=torch.Generator().manual_seed(seed)).images
                    preds = (preds * 255).astype(np.uint8)
                    score = float(np.mean([psnr(preds[j], real[j]) for j in range(n_in, n)]))
                    results.append({"set": name, "class_label": label, "seed": seed, "scale": scale, "psnr": score})
                    rows.append((f"scale {scale:g}, domain {label}, seed {seed}: {score:.1f} dB", preds))
                    for j in range(n_in, n):
                        Image.fromarray(preds[j]).save(out_dir / f"label{label}_seed{seed}_view{j}{suffix}.png")
                    print(f"   scale {scale:g} domain {label} seed {seed}: PSNR {score:.2f} dB", flush=True)
        with torch.no_grad():          # what the model is given: the inputs warped into each target camera by their depth
            warp = global_position_encoding_3d(cfg_run, depth.cuda(), s["K"].cuda(), w2c.cuda(), n_in, nframe=n, device="cuda",
                                               pe_scale=1 / 8, embed_dim=config.model_cfg.get("coord_dim", 192),
                                               colors=images)[0]
        Image.fromarray(np.asarray(warp).astype(np.uint8)).save(out_dir / "warped_inputs.png")
        sheet = Image.new("RGB", (230 + n * T, 16 + len(rows) * T), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 2), f"MVGenMaster, {name}: years {[round(y) for y in s['years']]} ({n_in} inputs, {n - n_in} targets)", fill="black")
        for r, (label, frames) in enumerate(rows):
            draw.text((4, 16 + r * T + T // 2 - 6), label, fill=(190, 0, 0) if r == 0 else "black")
            for j in range(n):
                if r == 0 or j >= n_in:
                    sheet.paste(Image.fromarray(frames[j]).resize((T, T)), (230 + j * T, 16 + r * T))
        sheet.save(out_dir / "sheet.jpg", quality=88)
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
    print("\nmean PSNR vs the real targets (dB):")
    for scale in args.scales:
        for label in args.class_labels:
            per_set = [np.mean([r["psnr"] for r in results
                                if r["set"] == name and r["class_label"] == label and r["scale"] == scale])
                       for name in names]
            print(f"   scale {scale:g} domain {label}: " + "  ".join(f"{name} {v:.2f}" for name, v in zip(names, per_set))
                  + f"  | mean {np.mean(per_set):.2f}", flush=True)

if __name__ == "__main__":
    main()
