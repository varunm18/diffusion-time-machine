#!/usr/bin/env python
"""Render a smooth camera path through a scene at chosen dates, as one multi-view generation per date.

For demos and for reconstruction from generated views: the three references are real photos; the
targets are NEW cameras interpolated between keyframe cameras (by default the references themselves, as
MVGenMaster's own demo ``run_mvgen.py`` builds its novel frames: positions linear, rotations interpolated,
intrinsics the keyframes' mean). All targets are generated together in one pass of the sampling pipeline
(the authors generate up to 28 frames at once), so they are consistent with each other; every date
uses the same references, cameras, look and noise seed, so the only difference between dates is the date.

Writes under ``--out``:
  ``<date>/frame_XX.png``    the generated frames along the path, and ``<date>/ref_X.png`` the references;
  ``cameras.json``           per frame: intrinsics (pixels, at the render resolution), camera-to-world and
                             world-to-camera (OpenCV axes, COLMAP world in the scene's units, times
                             ``scale``), image size; the references' too; for reconstruction pipelines;
  ``strip_<date>.jpg``       the frames side by side, and ``compare.gif`` (dates alternating per frame) and
                             ``path_<date>.gif`` (a fly-through).

    python scripts/mvgen/render_path.py --checkpoint <ckpt> --dataset <folder> --refs <key> <key> <key> \\
        --dates 2016 2021 --frames 21 --out <folder>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from timemachine.mvgen.data import MVGenScene, ViewSamplingConfig
from timemachine.mvgen.model import assert_exact_matmul, load_finetuned
from timemachine.mvgen.multiscene import scene_edges
from timemachine.mvgen.probes import render_views, rows_of_file_keys, to_uint8
from timemachine.mvgen.step import LossConfig, cross_date_drops
from timemachine.mvgen.trainer import parse_scale
from timemachine.training.loss import shown_dates


def interpolate_path(c2w: np.ndarray, n: int) -> np.ndarray:
    """``n`` camera-to-world matrices strictly between consecutive keyframes ``c2w`` ``[k, 4, 4]`` (k >= 2),
    spread over the segments in proportion to their length; positions linear, rotations by slerp."""
    centres = c2w[:, :3, 3]
    lengths = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    counts = np.floor(n * lengths / lengths.sum()).astype(int)
    counts[np.argsort(-lengths)[: n - counts.sum()]] += 1         # the remainder to the longest segments
    out = []
    for j, count in enumerate(counts):
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(c2w[j:j + 2, :3, :3]))
        for i in range(count):
            t = (i + 1) / (count + 1)
            pose = np.eye(4)
            pose[:3, :3] = slerp([t]).as_matrix()[0]
            pose[:3, 3] = (1 - t) * centres[j] + t * centres[j + 1]
            out.append(pose)
    return np.stack(out)


def order_along_path(c2w: np.ndarray) -> np.ndarray:
    """The keyframes' order along their main direction of spread (first principal axis of the centres)."""
    centres = c2w[:, :3, 3] - c2w[:, :3, 3].mean(0)
    axis = np.linalg.svd(centres, full_matrices=False)[2][0]
    return np.argsort(centres @ axis)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-dir", default="check_points/pretrained_model")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--refs", nargs=3, required=True, help="the three reference views: file keys or view-table rows")
    p.add_argument("--keyframes", nargs="*", help="cameras the path passes (file keys or rows; default: the references)")
    p.add_argument("--frames", type=int, default=21, help="generated frames (references + frames <= 28)")
    p.add_argument("--dates", type=float, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--look", default="ref0", help="targets' look code: ref0 / ref1 / ref2 (that reference's), "
                                                  "a view-table row, or 'neutral'")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.frames <= 25:
        raise SystemExit("--frames: 1 to 25 (the authors generate at most 28 views in one pass)")
    assert_exact_matmul()
    model, ckpt = load_finetuned(args.checkpoint, args.model_dir)
    model.unet.eval()
    trained = ckpt.get("train_config", {})
    loss_cfg = LossConfig(**trained.get("loss", {}))
    edges = tuple(trained.get("sampling", {}).get("period_edges") or scene_edges(args.dataset))
    scene = MVGenScene(args.dataset, ViewSamplingConfig(n_ref=3, period_edges=edges),
                       resolution=trained.get("resolution", 512), camera_longest_side=model.config.camera_longest_side,
                       require_depth=False, scale=parse_scale(trained.get("scale", "scene")))
    keys = scene.views["file_key"].tolist()

    def as_rows(items):
        return [int(x) if str(x).isdigit() else rows_of_file_keys(keys, [x])[0] for x in items]

    refs = as_rows(args.refs)
    keyframes = as_rows(args.keyframes) if args.keyframes else refs
    sample = scene.make_sample(np.asarray(refs + keyframes), 3)
    c2w_keys = np.linalg.inv(sample["w2c"][3:].double().numpy())
    c2w_keys = c2w_keys[order_along_path(c2w_keys)]
    path = interpolate_path(c2w_keys, args.frames)
    n, res = len(path), sample["images"].shape[-1]
    K_path = sample["K"][3:].mean(0, keepdim=True).repeat(n, 1, 1)
    w2c_path = torch.from_numpy(np.linalg.inv(path)).float()
    path_sample = {
        "images": torch.cat([sample["images"][:3], torch.zeros(n, *sample["images"].shape[1:])]),
        "K": torch.cat([sample["K"][:3], K_path]), "w2c": torch.cat([sample["w2c"][:3], w2c_path]),
        "depth": torch.cat([sample["depth"][:3], torch.zeros(n, *sample["depth"].shape[1:])]),
        "n_ref": 3, "scale": sample["scale"]}

    look = known = None
    if getattr(model, "look_stats", None) is not None and scene.look is not None:
        codes = np.asarray(scene.look["codes"])
        if args.look == "neutral":
            target = np.full(codes.shape[1], np.nan)            # standardised to 0: the population mean
        else:
            target = codes[refs[int(args.look[3:])]] if args.look.startswith("ref") else codes[int(args.look)]
        raw = np.concatenate([codes[refs], np.repeat(target[None], n, 0)])
        look = torch.from_numpy(model.look_stats.standardise(raw))
        known = torch.ones(3 + n, dtype=torch.bool)

    pipeline = model.sampling_pipeline()
    pipeline.set_progress_bar_config(disable=True)
    args.out.mkdir(parents=True, exist_ok=True)
    real_refs = to_uint8(sample["images"][:3])
    is_ref = torch.arange(3 + n) < 3
    renders = {}
    for date in args.dates:
        year = torch.cat([sample["date_mid"][:3], torch.full((n,), float(date))])
        valid = shown_dates(torch.ones(3 + n, dtype=torch.bool), is_ref, loss_cfg.date_views)
        prior_views, pixel_views = cross_date_drops(year, 3, loss_cfg, edges)
        frames = render_views(model, pipeline, path_sample, year, valid, seed=args.seed, steps=args.steps,
                              guidance=args.guidance, class_label=loss_cfg.class_label, precision=loss_cfg.precision,
                              dropped_prior_views=prior_views, dropped_pixel_views=pixel_views, look=look,
                              look_known=known)[3:]
        renders[date] = frames
        folder = args.out / f"{date:g}"
        folder.mkdir(exist_ok=True)
        for j, frame in enumerate(frames):
            Image.fromarray(frame).save(folder / f"frame_{j:02d}.png")
        for j, ref in enumerate(real_refs):
            Image.fromarray(ref).save(folder / f"ref_{j}.png")
        Image.fromarray(np.concatenate(list(frames), axis=1)).save(args.out / f"strip_{date:g}.jpg", quality=90)
        gif = [Image.fromarray(f).resize((384, 384)) for f in list(frames) + list(frames[::-1])]
        gif[0].save(args.out / f"path_{date:g}.gif", save_all=True, append_images=gif[1:], duration=120, loop=0)
        print(f"date {date:g}: {n} frames written to {folder}", flush=True)
    if len(args.dates) > 1:             # the same frame at each date in turn, frame by frame
        both = [Image.fromarray(np.concatenate([renders[d][j] for d in args.dates], axis=1)).resize(
            (384 * len(args.dates), 384)) for j in range(n)]
        both[0].save(args.out / "compare.gif", save_all=True, append_images=both[1:] + both[::-1], duration=150, loop=0)

    def camera(K, w2c, kind, j, file_key=None):
        c2w = np.linalg.inv(w2c.double().numpy())
        entry = {"kind": kind, "index": j, "width": res, "height": res, "K": K.tolist(),
                 "c2w": c2w.tolist(), "w2c": w2c.double().numpy().tolist()}
        if file_key is not None:
            entry["file_key"] = file_key
        return entry

    cameras = ([camera(sample["K"][j], sample["w2c"][j], "reference", j, sample["file_keys"][j]) for j in range(3)]
               + [camera(K_path[j], w2c_path[j], "frame", j) for j in range(n)])
    (args.out / "cameras.json").write_text(json.dumps({
        "convention": "OpenCV camera axes (x right, y down, z forward); COLMAP world of the scene, translations "
                      "multiplied by 'scale'; K in pixels of the width x height frames",
        "scale": sample["scale"], "dataset": str(args.dataset), "checkpoint": str(args.checkpoint),
        "step": ckpt.get("step"), "dates": args.dates, "seed": args.seed, "look": args.look,
        "reference_dates": [float(d) for d in sample["date_mid"][:3]], "cameras": cameras}, indent=1))
    print(f"wrote cameras.json and {len(args.dates)} date folders to {args.out}")


if __name__ == "__main__":
    main()
