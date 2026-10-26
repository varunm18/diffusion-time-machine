#!/usr/bin/env python
"""Step 0: download the MegaScenes index files and build the scene table.

Downloads (~310 MB, once) into ``<data>/megascenes/metadata/``:
  images_index.parquet, categories.json (S3) and recon_metadata.json (web viewer repo),
then writes ``<data>/tables/scenes.parquet`` (one row per scene) and prints the
largest scenes.

    python scripts/fetch_index.py
"""
from __future__ import annotations

import argparse

import requests

from timemachine.config import RECON_METADATA_URL, USER_AGENT, DataPaths
from timemachine.megascenes.index import (
    build_scene_table, load_categories, load_image_index, load_recon_metadata,
)
from timemachine.megascenes.s3 import MegaScenesBucket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="re-download and rebuild everything")
    args = parser.parse_args()

    paths, bucket = DataPaths(), MegaScenesBucket()
    for name, dest in (("images_index.parquet", paths.images_index), ("categories.json", paths.categories)):
        print(f"fetching metadata/{name}")
        bucket.download(f"metadata/{name}", dest, overwrite=args.overwrite)
    if args.overwrite or not paths.recon_metadata.exists():
        print("fetching recon_metadata.json")
        response = requests.get(RECON_METADATA_URL, headers={"User-Agent": USER_AGENT}, timeout=120)
        response.raise_for_status()
        paths.recon_metadata.write_bytes(response.content)

    print("building scene table")
    index = load_image_index(paths.images_index, columns=["cat", "subcat"])
    table = build_scene_table(index, load_categories(paths.categories), load_recon_metadata(paths.recon_metadata))
    paths.scene_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(paths.scene_table, index=False)
    print(f"wrote {paths.scene_table} ({len(table):,} scenes)\n")
    print(table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
