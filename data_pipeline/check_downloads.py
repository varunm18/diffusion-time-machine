"""Check that downloaded images exist, open, and match their camera calibration.

WHY THIS FILE EXISTS
--------------------
A pose only makes sense for the exact pixels it was computed from. If a file is
missing, or was replaced on Commons after MegaScenes ran its reconstruction, the
camera's pose and focal length no longer match the image, and training on it
would quietly teach the model wrong geometry. This script finds those cases
before they get into the dataset.

HOW TO RUN (from the repo root)
-------------------------------
    # Quick look at a random sample of 300:
    python data_pipeline/check_downloads.py data/raw/taj/colmap/0 data/raw/taj/images

    # Check EVERY image (only reads image headers, so it's fast):
    python data_pipeline/check_downloads.py data/raw/taj/colmap/0 data/raw/taj/images --sample 6000

WHAT IT CHECKS
--------------
 1. Every registered image name points to a real file on disk. (On Windows, a
    handful of Commons filenames contain characters Windows forbids, such as a
    double quote, and can't be saved. We saw 5 of 5,334 in the Taj Mahal.)
 2. The file opens, and its pixel size agrees with the size its camera was
    calibrated at. The focal length and image center are measured in pixels at
    that exact size, so a different-sized file needs its intrinsics rescaled,
    or the image should be dropped.

WHAT TO DO WITH THE RESULT (rules for the dataset code)
-------------------------------------------------------
    match      keep as-is
    scaled     keep, but rescale the intrinsics by (image width / camera width)
    rotated    drop (EXIF rotation swapped width and height; needs care)
    mismatch   drop (the file was almost certainly replaced after the SfM run)
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.colmap_reader import read_model  # noqa: E402


def find_missing(names: Iterable[str], root: Path) -> List[str]:
    """Return the registered image names that have no file under `root`."""
    return [n for n in names if not (root / n).is_file()]


def compare_size(img_size: Tuple[int, int], cam_size: Tuple[int, int], tol: float = 0.01) -> str:
    """Classify an image's (width, height) against its camera's (width, height).

    'match'     identical size, so the intrinsics are usable as-is
    'scaled'    same shape but different size, so rescale K by w_img / w_cam
    'rotated'   width and height are swapped (EXIF orientation)
    'mismatch'  anything else (cropped, or replaced by a different upload)

    `tol` is how close two aspect ratios must be (1% by default) to count as
    "the same shape", which absorbs rounding from resizing.
    """
    iw, ih = img_size
    cw, ch = cam_size
    if (iw, ih) == (cw, ch):
        return "match"
    # Same aspect ratio => just a resized copy of the same picture.
    if abs(iw / ih - cw / ch) <= tol * (cw / ch):
        return "scaled"
    # Aspect ratio matches the camera's *flipped* shape => it's rotated 90 degrees.
    if abs(iw / ih - ch / cw) <= tol * (ch / cw):
        return "rotated"
    return "mismatch"


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify downloaded images against a COLMAP model.")
    ap.add_argument("model_dir", help="COLMAP model folder (contains cameras.bin, images.bin)")
    ap.add_argument("images_root", help="folder that contains the 'commons/' image tree")
    ap.add_argument("--sample", type=int, default=300,
                    help="how many images to open and measure (use a big number for all)")
    args = ap.parse_args()

    root = Path(args.images_root)
    cameras, images = read_model(args.model_dir)
    names = [im.name for im in images.values()]

    # Part 1: is every registered image on disk?
    missing = find_missing(names, root)
    print(f"registered images: {len(names)}   found on disk: {len(names) - len(missing)}   "
          f"missing: {len(missing)}")
    for n in missing[:8]:
        print("  missing:", n)

    # Part 2: for a random sample (fixed seed, so reruns look at the same files),
    # open each image and compare its real size to its camera's.
    present = [im for im in images.values() if (root / im.name).is_file()]
    rng = random.Random(0)
    sample = rng.sample(present, min(args.sample, len(present)))

    try:
        from PIL import Image
    except ImportError:
        print("\nPillow not installed; run: python -m pip install pillow")
        return

    # Some panoramas are enormous. We only read the file header (never decode
    # the pixels), so it's safe to turn off Pillow's decompression-bomb guard.
    Image.MAX_IMAGE_PIXELS = None

    kinds = Counter()
    unreadable, examples = [], []
    for im in sample:
        cam = cameras[im.camera_id]
        try:
            with Image.open(root / im.name) as img:
                size = img.size  # (width, height), read from the header only
        except Exception as e:  # noqa: BLE001 - any failure just means "unreadable"
            unreadable.append((im.name, repr(e)))
            continue
        kind = compare_size(size, (cam.width, cam.height))
        kinds[kind] += 1
        if kind != "match" and len(examples) < 6:
            examples.append((kind, size, (cam.width, cam.height), im.name))

    print(f"\nsampled {len(sample)} images: {dict(kinds)}; unreadable: {len(unreadable)}")
    for kind, size, cam_size, name in examples:
        print(f"  {kind}: image {size} vs camera {cam_size}  {name}")
    for name, err in unreadable[:5]:
        print("  unreadable:", name, err)
    if kinds.get("scaled") or kinds.get("rotated") or kinds.get("mismatch"):
        print("\nWARNING: some images differ from their camera's calibrated size; "
              "intrinsics must be rescaled (or the image dropped) in the dataset code.")


if __name__ == "__main__":
    main()
