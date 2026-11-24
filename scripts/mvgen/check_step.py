#!/usr/bin/env python
"""GPU check: does ``timemachine.mvgen.step`` still give the recorded loss of the released model?

On the exported view set ``easy0`` (3 references, 5 targets) this computes the fp32 loss of the
released model at five timesteps with the training code (``load_mvgen``, ``encode_images``,
``build_inputs``, ``v_prediction_loss``): depth from Depth-Pro scaled to the COLMAP points, the
sample's own scene-scale rule (factor 1 for this set), latents from the mode of the VAE's posterior,
noise ``torch.randn(latents.shape, generator=torch.Generator().manual_seed(t))``, no dropout, no
dates set, domain label 0. It exits with an error if a loss differs from the recorded one
(``EXPECTED``) by more than the tolerance (1%). That the step feeds the UNet what the authors'
``train.py`` feeds it is tested on the CPU (``tests/test_mvgen_step.py``); this script guards the
numbers on the real weights.

``--record`` prints the measured losses in the form of ``EXPECTED`` and exits without comparing. Use
it once after a deliberate change of the step, and paste the result below.

Runs in MVGenMaster's environment, from its checkout:

    SCRIPT=$PWD/scripts/mvgen/check_step.py sbatch slurm/mvgen_gpu.sbatch
    SCRIPT=$PWD/scripts/mvgen/check_step.py ARGS="--record" sbatch slurm/mvgen_gpu.sbatch
"""
import argparse
import sys
from pathlib import Path

import torch

from depth_pro.depth_pro import create_model_and_transforms
from timemachine.mvgen.model import assert_exact_matmul, load_mvgen
from timemachine.mvgen.step import LossConfig, build_inputs, encode_images, v_prediction_loss

sys.path.insert(0, str(Path(__file__).parent))
from nvs_test import load_sample, normalise_scale, reference_depth  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# fp32 losses on easy0, by timestep, recorded on 2026-11-19 (A6000) with far depth contracted once, as the
# authors' train.py does. Before that fix the warped pixels came from depth contracted twice and the losses were
# 0.30278, 0.17434, 0.20687, 0.37548, 0.42420: the difference grows with the noise level, up to 0.6%.
EXPECTED = {50: 0.30276, 300: 0.17434, 600: 0.20716, 900: 0.3777, 999: 0.42517}


def exported_sample(folder: Path, camera_longest_side: float, device: str) -> dict:
    """An exported view set as a training sample: the keys that ``build_inputs`` reads, references first."""
    s = load_sample(folder)
    n_ref, n_views = sum(s["is_input"]), len(s["is_input"])
    if s["is_input"] != [True] * n_ref + [False] * (n_views - n_ref):
        raise ValueError(f"{folder}: the input views must come first")
    depth_model, transform = create_model_and_transforms(device=torch.device(device))
    depth_model.eval()
    depth = torch.zeros(n_views, 1, *s["images"].shape[-2:])
    for j in range(n_ref):
        metric, stats = reference_depth(depth_model, transform, s["images"][j], s["K"][j], s["sparse"][j])
        depth[j, 0] = torch.from_numpy(metric)
        print(f"   reference {j}: depth scaled to {stats['points']} COLMAP points, "
              f"median disagreement {stats['median_rel_error']:.1%}", flush=True)
    del depth_model
    torch.cuda.empty_cache()
    w2c, depth, extent, rescale = normalise_scale(s["w2c"], depth, camera_longest_side)
    print(f"   camera extent {extent:.2f}, rescale {rescale:.3f}", flush=True)
    return {"images": s["images"], "K": s["K"], "w2c": w2c, "depth": depth, "n_ref": n_ref}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=Path, default=REPO / "data/experiments/mvgen/samples/easy0",
                   help="the exported view set the expected losses were recorded on")
    p.add_argument("--model-dir", default="check_points/pretrained_model")
    p.add_argument("--tolerance", type=float, default=0.01, help="largest accepted relative difference")
    p.add_argument("--record", action="store_true",
                   help="print the measured losses as the EXPECTED table and exit 0, without comparing")
    args = p.parse_args(argv)
    assert_exact_matmul()
    device = "cuda"

    model = load_mvgen(args.model_dir, device=device)
    print(f"sample {args.sample}", flush=True)
    sample = exported_sample(args.sample, model.config.camera_longest_side, device)
    latents = encode_images(model, sample["images"], sample_posterior=False)
    inputs = build_inputs(model, sample, latents)
    cfg = LossConfig(precision="fp32", class_label=0)

    measured = {}
    for timestep in EXPECTED:
        noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(timestep))
        with torch.no_grad():
            loss, _ = v_prediction_loss(model, latents, inputs, timestep, noise, None, None, cfg, sample["n_ref"])
        measured[timestep] = round(float(loss), 5)
    if args.record:
        print(f"RECORDED, not compared. Paste into scripts/mvgen/check_step.py:\nEXPECTED = {measured}")
        return

    failed = []
    for timestep, expected in EXPECTED.items():
        difference = measured[timestep] / expected - 1
        ok = abs(difference) <= args.tolerance
        failed += [] if ok else [timestep]
        print(f"t {timestep:4d}: loss {measured[timestep]:.5f} | expected {expected:.5f} | difference "
              f"{difference:+.2%} | {'ok' if ok else 'MISMATCH'}", flush=True)
    if failed:
        raise SystemExit(f"FAILED: the loss differs by more than {args.tolerance:.0%} at timesteps {failed}")
    print(f"PASSED: all {len(EXPECTED)} losses within {args.tolerance:.0%} of the recorded values")


if __name__ == "__main__":
    main()
