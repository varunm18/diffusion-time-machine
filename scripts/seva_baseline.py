#!/usr/bin/env python
"""Unmodified-SEVA baseline on a single-scene dataset, plus the date-module identity check (GPU).

1. Identity check: the dated model with a fresh (zero) date module must give exactly
   SEVA's output, and must react to dates once its last layer is non-zero.
2. Baseline: for N samples (3 inputs + 5 targets), generate the targets with plain SEVA
   (no dates) and save contact sheets + PSNR. ``--mode random`` draws samples as in
   training (targets share a date, half of them pre-1970); ``--mode neighbors`` takes 8
   modern views with neighbouring cameras, the easy regime SEVA was trained on, as a
   sanity check of our camera conventions.

Needs SEVA (scripts/setup_seva.sh) and TIMEMACHINE_HF_HUB pointing at a hub cache with
the SEVA / SD2.1-VAE / OpenCLIP snapshots. See slurm/seva_gpu.sbatch.

    python scripts/seva_baseline.py --dataset data/datasets/25512_model0 --samples 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn

from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.sampling import SamplingConfig
from timemachine.model.inference import generate_targets
from timemachine.model.loading import load_components


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR (dB) between images in [-1, 1]."""
    mse = ((a - b) / 2).pow(2).mean().item()
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def to_pil(x: torch.Tensor, size: int = 224) -> Image.Image:
    arr = ((x.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size))


def contact_sheet(sample: dict, generated: torch.Tensor, scores: list[float], size: int = 224) -> Image.Image:
    """Top row: real views (inputs green, targets red). Bottom row: SEVA's output for the targets."""
    n = sample["images"].shape[0]
    sheet = Image.new("RGB", (size * n, 2 * size + 40), "white")
    draw = ImageDraw.Draw(sheet)
    for i in range(n):
        inp = bool(sample["is_input"][i])
        lo, hi = sample["date"][i].tolist()
        sheet.paste(to_pil(sample["images"][i], size), (i * size, 20))
        draw.text((i * size + 4, 4), f"{'IN' if inp else 'GT'} {int(lo)}" + (f"-{int(hi - 1e-6)}" if hi - lo > 1.01 else ""),
                  fill=(0, 140, 0) if inp else (200, 0, 0))
        if not inp:
            sheet.paste(to_pil(generated[i], size), (i * size, size + 40))
            draw.text((i * size + 4, size + 24), f"SEVA  {scores[i]:.1f} dB", fill=(0, 0, 0))
    return sheet


@torch.no_grad()
def identity_check(components, sample: dict) -> dict:
    """Plain vs dated forward pass of the real model on one noisy input."""
    model = components.model
    T = sample["images"].shape[0]
    x = torch.randn(T, 4, 72, 72, device="cuda", dtype=torch.bfloat16)
    t = torch.full((T,), 500, device="cuda")
    c = {"crossattn": torch.randn(T, 1, 1024, device="cuda", dtype=torch.bfloat16),
         "concat": torch.randn(T, 7, 72, 72, device="cuda", dtype=torch.bfloat16),
         "dense_vector": torch.randn(T, 6, 72, 72, device="cuda", dtype=torch.bfloat16)}
    dates = sample["date_mid"].cuda()
    with torch.autocast("cuda"):
        plain = model(x, t, dict(c), num_frames=T)
        dated = model(x, t, dict(c), date=dates, num_frames=T)
    # Perturb outside the autocast block: autocast caches low-precision weight copies
    # for the whole block, so an in-place change inside it would go unseen.
    last = model.date_module.date_embedding.mlp[-1]
    saved = last.weight.detach().clone()
    nn.init.normal_(last.weight, std=0.02)
    with torch.autocast("cuda"):
        perturbed = model(x, t, dict(c), date=dates, num_frames=T)
    last.weight.copy_(saved)
    return {"max_abs_diff_zero_init": float((plain - dated).abs().max()),
            "max_abs_diff_after_perturbing": float((plain - perturbed).abs().max())}


def neighbor_rows(dataset: DatedViewsDataset, rng: np.random.Generator, n_views: int = 8) -> np.ndarray:
    """A random modern view and its nearest modern neighbours (camera position + direction)."""
    c2w = dataset.c2ws
    modern = np.flatnonzero(dataset.dates >= 2000)
    centers = c2w[modern, :3, 3]
    scale = np.median(np.linalg.norm(centers - np.median(centers, 0), axis=1)) or 1.0
    feats = np.concatenate([centers / scale, c2w[modern, :3, 2]], axis=1)
    anchor = rng.integers(len(modern))
    order = np.argsort(np.linalg.norm(feats - feats[anchor], axis=1))[:n_views]
    return modern[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="folder with views.parquet")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--mode", choices=["random", "neighbors"], default="random")
    parser.add_argument("--out", type=Path, help="output folder (default: data/experiments/seva_baseline/<dataset>)")
    parser.add_argument("--version", type=float, default=1.1, help="SEVA weights version")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = args.out or Path("data/experiments/seva_baseline") / f"{args.dataset.name}_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    views = pd.read_parquet(args.dataset / "views.parquet")
    dataset = DatedViewsDataset(views, SamplingConfig(n_inputs=3, n_targets=5, old_target_fraction=0.5),
                                resolution=576, samples_per_epoch=args.samples, seed=args.seed)
    components = load_components(version=args.version, dated=True)

    report = {"identity_check": identity_check(components, dataset[0])}
    print("identity check:", report["identity_check"], flush=True)

    rows = []
    rng = np.random.default_rng(args.seed)
    for i in range(len(dataset)):
        sample = dataset[i] if args.mode == "random" else dataset.make_sample(neighbor_rows(dataset, rng), 3)
        generated = generate_targets(components, sample["images"].cuda(), sample["K"].cuda(),
                                     sample["c2w"].cuda(), sample["is_input"].cuda(),
                                     num_steps=args.steps, seed=args.seed + i).cpu()
        scores = [psnr(generated[j], sample["images"][j]) for j in range(len(generated))]
        targets = (~sample["is_input"]).nonzero().flatten().tolist()
        old_targets = bool(sample["date_mid"][targets].max() < 1970)
        for j in targets:
            rows.append({"sample": i, "view": j, "file_key": sample["file_keys"][j],
                         "target_year": float(sample["date_mid"][j]), "old_target_sample": old_targets,
                         "input_years": [round(float(y), 1) for y in sample["date_mid"][sample["is_input"]]],
                         "psnr": scores[j]})
        contact_sheet(sample, generated, scores).save(out / f"sample_{i}.jpg", quality=90)
        print(f"sample {i}: target years {[int(sample['date_mid'][j]) for j in targets]}, "
              f"PSNR {np.mean([scores[j] for j in targets]):.2f} dB", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out / "psnr.csv", index=False)
    report["psnr_mean"] = float(table["psnr"].mean())
    report["psnr_by_target_group"] = table.groupby("old_target_sample")["psnr"].mean().rename(
        {True: "old_targets", False: "modern_targets"}).to_dict()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
