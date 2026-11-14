#!/usr/bin/env python
"""Dataset v1, step 5: audit what the loader actually produces.

Draws many samples from the training split and reports the target eras, the
date gap between inputs and targets, the share of each scene cluster and the
loading speed; checks that the splits share no file or image hash; and saves
contact sheets of a few samples (inputs on the left, targets on the right).

    python scripts/audit_dataset_v1.py
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from timemachine.analysis.census import OLD_ERAS
from timemachine.config import DataPaths
from timemachine.data.multiscene import MultiSceneDataset
from timemachine.data.splits import find_leaks


def sample_sheet(data: MultiSceneDataset, index: int, thumb: int = 200) -> Image.Image:
    """The views of sample ``index`` in a row, each labelled with its drawn year (inputs outlined)."""
    item = data[index]
    views = pd.read_parquet(data.models[data.draw(index)[0]].folder / "views.parquet").set_index("file_key")
    sheet = Image.new("RGB", (thumb * len(item["file_keys"]), thumb + 22), "white")
    draw = ImageDraw.Draw(sheet)
    for col, key in enumerate(item["file_keys"]):
        sheet.paste(Image.open(views.loc[key, "image_path"]).convert("RGB").resize((thumb, thumb)), (col * thumb, 22))
        role = "input" if item["is_input"][col] else "target"
        year = f"{item['year'][col]:.0f}" if item["year_valid"][col] else "unknown"
        draw.text((col * thumb + 4, 4), f"{role} {year}", fill=(200, 0, 0) if role == "input" else (0, 0, 160))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--sheets", type=int, default=6, help="contact sheets per kind (old / modern targets)")
    args = parser.parse_args()

    paths = DataPaths()
    train = MultiSceneDataset(paths, "train", samples_per_epoch=args.draws)
    test = MultiSceneDataset(paths, "test", samples_per_epoch=args.draws)
    for name, data in (("train", train), ("test", test)):
        views = sum(len(m.file_keys) for m in data.models)
        old = sum(m.weight_counts[0] for m in data.models)
        print(f"{name}: {len(data.models)} models, {len({m.scene_id for m in data.models})} scenes, "
              f"{views:,} views, {old:,} old anchors")

    eras, clusters, gaps, by_era = Counter(), Counter(), [], {}
    for index in range(args.draws):
        i, era, rows, _ = train.draw(index)
        m = train.models[i]
        eras[era] += 1
        clusters[m.cluster] += 1
        n_in = train.config.n_inputs
        gaps.append(np.abs(m.dates[rows[:n_in]] - m.dates[rows[n_in:]].mean()).max())
        by_era.setdefault(era, index)
    gaps = np.array(gaps)
    print(f"\n{args.draws:,} training draws")
    print("target era:      " + ", ".join(f"{e} {n / args.draws:.1%}" for e, n in sorted(eras.items())))
    print(f"largest date gap between an input and the targets: median {np.median(gaps):.0f} years, "
          f"{(gaps > 50).mean():.1%} over 50 years, {(gaps < 2).mean():.1%} under 2 years")
    print("largest clusters: " + ", ".join(f"{c} {n / args.draws:.1%}" for c, n in clusters.most_common(4)))

    keys = pd.concat([pd.DataFrame({"file_key": m.file_keys, "split": name})
                      for name, data in (("train", train), ("test", test)) for m in data.models])
    hashes = pd.concat([pd.read_parquet(m.folder / "views.parquet", columns=["dhash"]).assign(split=name)
                        for name, data in (("train", train), ("test", test)) for m in data.models])
    print(f"\nleaks between splits: {len(find_leaks(keys))} file names, {len(find_leaks(hashes, 'dhash'))} image hashes")

    loader = torch.utils.data.DataLoader(train, batch_size=4, num_workers=4, persistent_workers=True)
    batches = iter(loader)
    next(batches)                                               # workers started, memory maps opened
    start, n = time.time(), 0
    for _, batch in zip(range(100), batches):
        n += len(batch["latents"])
    print(f"loading speed: {n / (time.time() - start):.0f} samples/s with 4 workers")

    out = paths.root / "datasets" / "samples_v1"
    out.mkdir(parents=True, exist_ok=True)
    old_done = modern_done = 0
    for index in range(args.draws):
        era = train.draw(index)[1]
        is_old = era in OLD_ERAS
        if (is_old and old_done < args.sheets) or (not is_old and modern_done < args.sheets):
            kind = "old" if is_old else "modern"
            number = old_done if is_old else modern_done
            sample_sheet(train, index).save(out / f"{kind}_target_{number}.jpg", quality=90)
            old_done, modern_done = old_done + is_old, modern_done + (not is_old)
        if old_done >= args.sheets and modern_done >= args.sheets:
            break
    print(f"contact sheets: {out}")


if __name__ == "__main__":
    main()
