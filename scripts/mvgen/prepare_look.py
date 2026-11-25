#!/usr/bin/env python
"""Write a dataset folder's look codes: 24 colour and medium statistics per view (CPU; experiment look1).

Two files go into the dataset folder (their layout and the code itself are described in
``timemachine/mvgen/look.py``):

* ``class_maps.npz``: SegFormer-B2's ADE20K classes of each view's 512 crop, at 128 px. A cache: when
  it exists and was made for this view table with the same model, it is used as it is (about 12
  minutes of CPU otherwise). ``--recompute-class-maps`` makes it again.
* ``look_codes.npz``: every view's raw look code (from the 512 crop of ``MVGenScene.make_sample``, over
  the pixels outside the view's changed region, grown by 8 px, when ``changed_regions.npz`` has a
  non-empty mask for it), which classes it shows, its tone, the tone curves of its toned prints (arm D's
  medium conversion) and the statistics of the usable views. An existing file is not replaced unless
  ``--overwrite`` is given: a run that trains with the look stores a hash of it and is not resumed on
  another one.

Tones: ``--tone-labels`` names a file of labels to use where it has one (an ``.npz`` with ``rows``, rows
of the view table, and ``tone``, one of grey / bw, toned, colour per row); every other view gets the
residual-chroma rule (``look.tone_rule``). For Notre-Dame give the appearance study's eye-corrected
labels, ``data/appearance_study/design_minimal/codes.npz`` (the 1,115 usable views: 47 grey, 44 toned,
1,024 colour); the file's rows are checked to be the usable rows of the view table.

SegFormer is loaded from the local Hugging Face cache only (set ``HF_HOME``; ``HF_HUB_OFFLINE=1``), at
the snapshot the appearance study used, and given the crop with ImageNet normalisation, as the study
did. Runs in MVGenMaster's environment from its checkout (it has OpenCV, for the tone rule); give
absolute paths and keep to 4 threads on a login node:

    cd external/MVGenMaster && HF_HOME=/vulcanscratch/ltahboub/.hf-cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \\
        nice -n 10 .venv/bin/python $REPO/scripts/mvgen/prepare_look.py \\
        --tone-labels $REPO/data/appearance_study/design_minimal/codes.npz
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from timemachine.data.dataset import load_view
from timemachine.mvgen.data import REGIONS_FILE, read_region_weights
from timemachine.mvgen.look import (
    CLASS_MAP_FILE, CLASS_MAP_SIZE, IMAGENET_MEAN, IMAGENET_STD, LOG_SD_EPS, LOOK_CLASSES, LOOK_DIM, LOOK_FILE,
    MIN_CLASS_SHARE, REGION_GROW, SEGFORMER, SEGFORMER_REVISION, TONES, LookStats, class_groups, excluded_pixels,
    grown_cells, look_code, to_image, tone_rule, toned_curve, write_look_file,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / "data/datasets/25512_model0_v2"
RESOLUTION = 512                    # the crops the code is made from: what the runs train on
LABEL_NAMES = {"bw": "grey", "grey": "grey", "toned": "toned", "colour": "colour"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                   help="dataset folder; the files are written into it")
    p.add_argument("--tone-labels", type=Path,
                   help="labels to use instead of the tone rule where they have one (.npz with rows and tone); for "
                        "Notre-Dame: data/appearance_study/design_minimal/codes.npz")
    p.add_argument("--overwrite", action="store_true", help=f"replace an existing {LOOK_FILE}")
    p.add_argument("--recompute-class-maps", action="store_true",
                   help=f"run SegFormer again even if {CLASS_MAP_FILE} fits the view table")
    p.add_argument("--batch", type=int, default=4, help="crops per SegFormer forward pass")
    p.add_argument("--threads", type=int, default=4, help="CPU threads (at most 4 on a login node)")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def has_image(views: pd.DataFrame) -> np.ndarray:
    """``[N]`` bool: the views whose image file exists (the others get no code)."""
    return np.array([Path(path).is_file() for path in views["image_path"]])


def crop(views: pd.DataFrame, row: int) -> np.ndarray:
    """The 512 crop of view ``row`` exactly as ``MVGenScene.make_sample`` gives it: float32 ``[S, S, 3]`` in [0, 1]."""
    image, _ = load_view(views.iloc[row], RESOLUTION)
    return to_image(image)


def stored_class_maps(path: Path, file_keys: np.ndarray) -> np.ndarray | None:
    """The labels of an existing class-map file, if it was made for this view table with this model."""
    if not path.exists():
        return None
    with np.load(path) as stored:
        fits = (np.array_equal(stored["file_keys"], file_keys) and str(stored["model"]) == SEGFORMER
                and str(stored["revision"]) == SEGFORMER_REVISION)
        return stored["labels"] if fits else None


def segformer(device: str):
    """SegFormer-B2 (ADE20K) from the local cache only, at the appearance study's snapshot."""
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(SEGFORMER, revision=SEGFORMER_REVISION,
                                                            local_files_only=True).eval().to(device)


@torch.no_grad()
def class_labels(model, crops: list[np.ndarray], device: str) -> np.ndarray:
    """ADE20K label maps ``[n, 128, 128]`` uint8 of crops in [0, 1], ImageNet-normalised as the study did."""
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(device)
    mean, std = torch.tensor(IMAGENET_MEAN, device=device), torch.tensor(IMAGENET_STD, device=device)
    logits = model(pixel_values=(x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)).logits
    if logits.shape[-1] != CLASS_MAP_SIZE:
        raise ValueError(f"SegFormer gave label maps of {tuple(logits.shape[-2:])}, expected {CLASS_MAP_SIZE}")
    return logits.argmax(1).cpu().numpy().astype(np.uint8)


def read_tone_labels(path: Path, views: pd.DataFrame) -> dict[int, str]:
    """``{row: tone}`` from a labels file; the appearance study's must cover exactly the usable rows."""
    with np.load(path, allow_pickle=True) as stored:
        rows, tones = stored["rows"].astype(np.int64), stored["tone"].astype(str)
    if len(rows) != len(tones) or rows.min() < 0 or rows.max() >= len(views):
        raise ValueError(f"{path}: rows and tone must be of one length and name rows of the view table")
    if "design_minimal" in str(path) and not np.array_equal(rows, np.flatnonzero(views["usable"].to_numpy(bool))):
        raise ValueError(f"{path}: the appearance study's labels are for the usable rows of the view table, and "
                         "these rows are not: the table was rebuilt since")
    unknown = sorted(set(tones) - set(LABEL_NAMES))
    if unknown:
        raise ValueError(f"{path}: unknown tone labels {unknown}")
    return {int(row): LABEL_NAMES[tone] for row, tone in zip(rows, tones)}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    dataset = args.dataset
    out = dataset / LOOK_FILE
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists: pass --overwrite to replace it (a run that trains with the look is not "
                         "resumed on another file)")
    views = pd.read_parquet(dataset / "views.parquet").reset_index(drop=True)
    n, file_keys = len(views), views["file_key"].to_numpy().astype(str)
    computed = has_image(views)
    print(f"[look] {dataset}: {n} views, {int(computed.sum())} with an image", flush=True)

    regions = read_region_weights(dataset, file_keys, RESOLUTION)     # None without a usable regions file
    side = RESOLUTION // 8
    region = np.zeros((n, side, side), dtype=bool)
    if regions is not None:
        for row in range(n):
            if (regions[row] > 0).any():
                region[row] = grown_cells(regions[row] > 0, REGION_GROW)
    print(f"[look] changed regions: {'none' if regions is None else REGIONS_FILE}; "
          f"{int(region.any(axis=(1, 2)).sum())} views have a region left out of their code", flush=True)

    labels_given = {} if args.tone_labels is None else read_tone_labels(args.tone_labels, views)
    maps_path = dataset / CLASS_MAP_FILE
    labels = None if args.recompute_class_maps else stored_class_maps(maps_path, file_keys)
    model = None
    if labels is None:
        labels = np.full((n, CLASS_MAP_SIZE, CLASS_MAP_SIZE), 255, dtype=np.uint8)
        model = segformer(args.device)
        print(f"[look] {CLASS_MAP_FILE}: running {SEGFORMER} on the CPU", flush=True)
    else:
        print(f"[look] {CLASS_MAP_FILE}: using the stored class maps", flush=True)

    codes = np.full((n, LOOK_DIM), np.nan, dtype=np.float32)
    shown = np.zeros((n, len(LOOK_CLASSES)), dtype=bool)
    tone = np.full(n, "", dtype=object)
    tone_source = np.full(n, "", dtype=object)
    curves, curve_rows = [], []
    rows = np.flatnonzero(computed)
    started = time.time()
    for start in range(0, len(rows), args.batch):
        batch = rows[start:start + args.batch]
        crops = [crop(views, row) for row in batch]
        if model is not None:
            labels[batch] = class_labels(model, crops, args.device)
        for row, image in zip(batch, crops):
            given = labels_given.get(int(row))
            tone[row], tone_source[row] = (given, "labels") if given else (tone_rule(image), "rule")
            codes[row], shown[row] = look_code(image, class_groups(labels[row]),
                                               excluded_pixels(region[row], RESOLUTION), tone[row])
            if tone[row] == "toned":
                curves.append(toned_curve(image))
                curve_rows.append(int(row))
        done = start + len(batch)
        if done % (25 * args.batch) == 0 or done == len(rows):
            print(f"[look] {done} of {len(rows)} views ({time.time() - started:.0f} s)", flush=True)

    if model is not None:
        tmp = maps_path.with_name(maps_path.name + ".tmp")
        with open(tmp, "wb") as file:
            np.savez_compressed(file, labels=labels, file_keys=file_keys, model=SEGFORMER, revision=SEGFORMER_REVISION,
                                computed=computed, resolution=RESOLUTION)
        tmp.replace(maps_path)
        print(f"[look] wrote {maps_path}", flush=True)

    usable = views["usable"].to_numpy(bool) & computed
    stats = LookStats.from_codes(codes[usable])
    config = {"resolution": RESOLUTION, "class_map_size": CLASS_MAP_SIZE, "model": SEGFORMER,
              "revision": SEGFORMER_REVISION, "min_class_share": MIN_CLASS_SHARE, "log_sd_eps": LOG_SD_EPS,
              "region_grow_cells": REGION_GROW, "regions": None if regions is None else REGIONS_FILE,
              "tone_labels": None if args.tone_labels is None else str(args.tone_labels),
              "class_maps_sha256": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()}
    path = write_look_file(dataset, {
        "codes": codes, "shown": shown, "tone": tone.astype(str), "tone_source": tone_source.astype(str),
        "computed": computed, "region": region, "file_keys": file_keys,
        "toned_curves": np.asarray(curves, dtype=np.float32).reshape(-1, 256, 3),
        "toned_rows": np.asarray(curve_rows, dtype=np.int64),
        "stats_mean": np.asarray(stats.mean), "stats_sd": np.asarray(stats.sd)}, config)
    counts = {name: int(((tone == name) & usable).sum()) for name in TONES}
    sources = {name: int((tone_source == name).sum()) for name in ("labels", "rule")}
    print(f"[look] wrote {path}: usable views by tone {counts}; tones from {sources}; {len(curves)} tone curves; "
          f"classes shown among usable views {dict(zip(LOOK_CLASSES, shown[usable].mean(0).round(3).tolist()))}",
          flush=True)


if __name__ == "__main__":
    main()
