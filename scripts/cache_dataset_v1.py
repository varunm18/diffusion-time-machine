#!/usr/bin/env python
"""Dataset v1, step 4: cache VAE latents and CLIP embeddings of every kept view (GPU).

For each model's ``views.parquet`` (after scripts/clean_dataset_v1.py), drops the
views with a ``drop_reason``, rewrites the table and saves ``latents.npy``,
``clip.npy`` and ``cache_keys.json`` beside it (see ``timemachine.data.cache``).
Resumable: models whose cache matches their view table are skipped.

Ends with a sanity check: decodes a few cached latents and compares them with the
original crops. A healthy VAE gives roughly 25 dB or more on photos.

    python scripts/cache_dataset_v1.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from timemachine.config import DataPaths
from timemachine.data.cache import (
    LATENTS, Encoders, ViewImages, cache_is_current, encode_views, psnr, write_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6, help="image-loading processes")
    parser.add_argument("--check", type=int, default=16, help="latents to decode for the sanity check")
    args = parser.parse_args()

    paths = DataPaths()
    models = pd.read_parquet(paths.tables / "models_v1.parquet")
    models = models[models["role"] != "excluded"].reset_index(drop=True)
    encoders = Encoders()
    print(f"{len(models)} models; encoders on {torch.cuda.get_device_name(0)}")

    total, start_all, last = 0, time.time(), None
    for i, m in enumerate(models.itertuples(), 1):
        folder = paths.dataset_dir(int(m.scene_id), int(m.model))
        label = f"[{i}/{len(models)}] {m.scene} ({m.scene_id}) model {m.model}"
        views = pd.read_parquet(folder / "views.parquet")
        if "drop_reason" not in views:
            raise SystemExit(f"{folder}: run scripts/clean_dataset_v1.py first")
        kept = views[views["drop_reason"] == ""].reset_index(drop=True)
        keys = kept["file_key"].tolist()
        total += len(kept)
        last = (folder, kept)
        if len(kept) == len(views) and cache_is_current(folder, keys):
            print(f"{label}: {len(kept)} views (already cached)", flush=True)
            continue
        start = time.time()
        latents, clip = encode_views(kept, encoders.encode, args.batch_size, args.workers)
        if not np.isfinite(latents).all() or not np.isfinite(clip).all():
            raise SystemExit(f"{label}: non-finite values in the encodings")
        if len(kept) != len(views):
            tmp = folder / "views.parquet.part"
            kept.to_parquet(tmp, index=False)
            tmp.replace(folder / "views.parquet")
        write_cache(folder, keys, latents, clip)
        print(f"{label}: {len(kept)} views, {len(views) - len(kept)} dropped [{time.time() - start:.0f}s]", flush=True)
    print(f"\n{total:,} views cached in {len(models)} models [{(time.time() - start_all) / 60:.1f} min]")

    if args.check and last is not None:
        folder, kept = last
        n = min(args.check, len(kept))
        images = torch.stack([ViewImages(kept)[j] for j in range(n)])
        latents = torch.from_numpy(np.load(folder / LATENTS, mmap_mode="r")[:n].astype(np.float32))
        decoded = encoders.decode(latents).clamp(-1, 1).cpu()
        scores = [psnr(decoded[j], images[j]) for j in range(n)]
        print(f"sanity check on {n} views of {folder.name}: decode(latent) vs crop PSNR "
              f"mean {np.mean(scores):.1f} dB, min {np.min(scores):.1f} dB")


if __name__ == "__main__":
    main()
