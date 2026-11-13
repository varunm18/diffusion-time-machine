#!/usr/bin/env python
"""Dataset v1, step 2: download the images of every selected model.

Reads ``<data>/tables/models_v1.parquet`` (scripts/select_models.py). For each
model that is not excluded: builds its view table, keeps all pre-2000 views and
at most ``--max-modern`` 2000+ views, downloads the images into the shared image
cache and writes ``<data>/datasets/<scene_id>_model<m>/views.parquet``.

Resumable: a model whose views.parquet exists is skipped (pass --overwrite to
redo it), and images already in the cache are not downloaded again.

    python scripts/build_dataset_v1.py --dry-run          # counts only
    python scripts/build_dataset_v1.py --max-modern 200
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from timemachine.config import DataPaths
from timemachine.data.images import download_view_images
from timemachine.data.selection import cap_modern_views
from timemachine.data.views import build_view_table
from timemachine.megascenes.s3 import MegaScenesBucket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-modern", type=int, default=200, help="2000+ views kept per model")
    parser.add_argument("--max-side", type=int, default=1024, help="longest side of cached images")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads")
    parser.add_argument("--dry-run", action="store_true", help="print the counts, download nothing")
    parser.add_argument("--overwrite", action="store_true", help="rebuild models that are already done")
    args = parser.parse_args()

    paths, bucket = DataPaths(), MegaScenesBucket()
    models = pd.read_parquet(paths.tables / "models_v1.parquet")
    models = models[models["role"] != "excluded"].reset_index(drop=True)
    print(f"{len(models)} models, at most {args.max_modern} modern views each")

    total, failures = 0, []
    for i, m in enumerate(models.itertuples(), 1):
        sid, model = int(m.scene_id), int(m.model)
        out = paths.dataset_dir(sid, model) / "views.parquet"
        label = f"[{i}/{len(models)}] {m.scene} ({sid}) model {model}"
        if out.exists() and not args.overwrite:
            n = len(pd.read_parquet(out, columns=["file_key"]))
            total += n
            print(f"{label}: {n} views (already done)", flush=True)
            continue
        views = cap_modern_views(build_view_table(paths, sid, model), args.max_modern)
        if args.dry_run:
            total += len(views)
            print(f"{label}: {len(views)} views to fetch", flush=True)
            continue
        start = time.time()
        views = download_view_images(bucket, paths, views, args.max_side, args.workers)
        failed = views.attrs.get("failed", [])
        failures += [{"scene_id": sid, "model": model, **f} for f in failed]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        views.to_parquet(tmp, index=False)   # written last and renamed: a crash never leaves a half file
        tmp.replace(out)
        total += len(views)
        print(f"{label}: {len(views)} views, {len(failed)} failed [{time.time() - start:.0f}s]", flush=True)

    print(f"\n{total:,} views in {len(models)} models" + (" (dry run)" if args.dry_run else ""))
    if failures:
        log = paths.root / "datasets" / "download_failures_v1.csv"
        pd.DataFrame(failures).to_csv(log, mode="a", header=not log.exists(), index=False)
        print(f"{len(failures)} images failed to download or decode; listed in {log}")


if __name__ == "__main__":
    main()
