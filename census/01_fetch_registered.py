"""Step 01: list the registered images, with camera poses, for each scene.

For every COLMAP model of every scene, downloads images.bin, keeps each
registered image's name and pose, then deletes the file (a large scene's
images.bin is ~1 GB, mostly 2D keypoints we don't need). Writes one CSV per
scene to data/registered/<scene_id>.csv.

Resumable: scenes that already have a CSV are skipped, and each CSV is written
only once its whole scene has finished, so rerunning after a crash is safe.
Scenes with no reconstruction get a header-only CSV so they're recorded too.

    python 01_fetch_registered.py --scenes pilot_scenes.txt
"""
import argparse
import csv

from census_lib import (BUCKET, MEGASCENES, REGISTERED, list_models,
                        load_categories, read_colmap_images, read_scene_list,
                        s5, scene_prefix)

FIELDS = ["scene", "scene_id", "model", "image", "image_id", "camera_id",
          "qw", "qx", "qy", "qz", "tx", "ty", "tz"]


def fetch_model(scene_id: int, model: str) -> list[dict]:
    key = f"{scene_prefix(scene_id)}/colmap/{model}/images.bin"
    local = MEGASCENES / key
    local.parent.mkdir(parents=True, exist_ok=True)
    s5("cp", f"{BUCKET}/{key}", str(local))
    try:
        return read_colmap_images(local)
    finally:
        local.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="pilot_scenes.txt",
                    help="file of scene names, one per line (default pilot_scenes.txt)")
    args = ap.parse_args()

    cats = load_categories()
    REGISTERED.mkdir(parents=True, exist_ok=True)

    for name in read_scene_list(args.scenes):
        if name not in cats:
            print(f"? {name}: not in categories.json, skipping")
            continue
        sid = cats[name]
        out = REGISTERED / f"{sid}.csv"
        if out.exists():
            print(f"skip {name} (done)")
            continue

        models = list_models(sid)
        print(f"{name} ({sid}): {len(models) if models else 'no'} models")
        rows = []
        for m in models:
            images = fetch_model(sid, m)
            rows += [{"scene": name, "scene_id": sid, "model": int(m), **img}
                     for img in images]
            print(f"  model {m}: {len(images)} images")

        tmp = out.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(out)  # appears only when complete


if __name__ == "__main__":
    main()
