"""Take a first look at a downloaded COLMAP model and check it against the index.

WHY THIS FILE EXISTS
--------------------
After downloading a scene's reconstruction, this is the first thing to run. It
answers: "does our COLMAP reader work on the real file, how big is the model,
and can we connect its image names back to MegaScenes' image index?" It was
how we learned that the Taj Mahal model has 5,334 registered images sharing
3,560 cameras.

HOW TO RUN (from the repo root)
-------------------------------
    python data_pipeline/inspect_reconstruction.py data/raw/taj/colmap/0 Taj_Mahal

The second argument is the scene's name in the index (the `cat` column).

WHAT IT PRINTS
--------------
 * number of cameras and registered images, and which camera models they use
 * the smallest and largest image width (a huge max means panoramas to filter)
 * the first few registered image names
 * how many of those names match rows in metadata/images_index.parquet

NOTE ON NAME MATCHING
---------------------
COLMAP names use underscores in filenames while the index uses spaces. Commons
treats the two as the same character, so we normalize both sides before
comparing. (Before we did, only 331 of 5,334 matched.)
"""

import sys
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.colmap_reader import read_model  # noqa: E402

# Where we saved the index with: aws s3 cp ... metadata/images_index.parquet
INDEX = "data/raw/metadata/images_index.parquet"


def main() -> None:
    # Command-line arguments, with the Taj Mahal as the default example.
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/taj/colmap/0"
    scene = sys.argv[2] if len(sys.argv) > 2 else "Taj_Mahal"

    cameras, images = read_model(model_dir)
    print(f"cameras: {len(cameras)}   registered images: {len(images)}")
    print("camera models:", dict(Counter(c.model for c in cameras.values())))
    widths = [c.width for c in cameras.values()]
    print(f"image widths: min {min(widths)}, max {max(widths)}")
    # If there are more images than cameras, several photos share one camera.
    print(f"cameras shared by several images: {len(images) > len(cameras)}")

    names = [im.name for im in images.values()]
    print("\nfirst 5 registered image names:")
    for n in names[:5]:
        print("  ", repr(n))

    # Load only this scene's rows from the (236 MB) index, using a filter so the
    # whole file never has to fit in memory.
    table = ds.dataset(INDEX).to_table(
        columns=["cat", "image", "image_name"],
        filter=pc.equal(ds.field("cat"), scene),
    ).to_pandas()
    print(f"\nindex rows for scene '{scene}': {len(table)}")

    # Commons treats "_" and " " as the same character, but COLMAP names use
    # underscores in filenames while the index uses spaces. Normalize both.
    def norm(x: str) -> str:
        return x.replace("_", " ")

    by_full = {norm(x) for x in table["image"]}          # full path in the index
    by_name = {norm(x) for x in table["image_name"]}     # just the filename
    basenames = [norm(n.rsplit("/", 1)[-1]) for n in names]
    hit_full = sum(norm(n) in by_full for n in names)
    hit_name = sum(b in by_name for b in basenames)
    print(f"matches on full path (underscores = spaces): {hit_full}/{len(names)}")
    print(f"matches on filename only:                    {hit_name}/{len(names)}")

    # Show a few failures so a mismatch can be diagnosed by eye.
    unmatched = [n for n, b in zip(names, basenames) if b not in by_name]
    if unmatched:
        print(f"\n{len(unmatched)} unmatched; first 5 (to diagnose):")
        for n in unmatched[:5]:
            print("  ", repr(n))


if __name__ == "__main__":
    main()
