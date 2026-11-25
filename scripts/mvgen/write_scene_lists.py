"""Write the training and held-out scene lists of the multi-scene run from the folders built so far.

A scene is listed once its folder has depth for every loadable view and look codes (prepare_depth.py, prepare_look.py).
Held-out scenes are those marked ``held_out`` in data/multiscene_plan/scene_pool/selection.csv; the
training list also gets the Notre-Dame folder used by the single-scene runs (its hand date fixes and
changed-region maps), which the mixture gives a fixed share.

    python scripts/mvgen/write_scene_lists.py            # prints counts, writes the two lists
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "data" / "multiscene_plan"
NOTRE_DAME = REPO / "data" / "datasets" / "25512_model0_v2"
ROOT = Path("/fs/nexus-projects/wildnvs/diffusion-time-machine/datasets")


def ready(folder: Path) -> bool:
    """Built, with look codes, and with a depth record for every view prepare_depth.py can load."""
    if not all((folder / name).exists() for name in ("build_summary.json", "depth_stats.parquet", "look_codes.npz")):
        return False
    views = pd.read_parquet(folder / "views.parquet", columns=["image_path", "k1", "k2"])
    loadable = views["image_path"].map(lambda path: Path(path).exists()).to_numpy(dtype=bool) & np.isfinite(
        views[["k1", "k2"]].to_numpy(dtype=np.float64)).all(axis=1)       # prepare_depth.loadable_views
    return len(pd.read_parquet(folder / "depth_stats.parquet", columns=["row"])) >= int(loadable.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--train-out", type=Path, default=PLAN / "scenes_train.txt")
    parser.add_argument("--heldout-out", type=Path, default=PLAN / "scenes_heldout.txt")
    args = parser.parse_args()
    selection = pd.read_csv(PLAN / "scene_pool" / "selection.csv")
    held = set(selection.loc[selection["held_out"].astype(bool), "scene_id"].astype(int))
    folders = sorted(p for p in args.root.iterdir() if p.is_dir())
    train, heldout, waiting = [NOTRE_DAME], [], []
    for folder in folders:
        if not ready(folder):
            waiting.append(folder.name)
            continue
        (heldout if int(folder.name.split("_")[0]) in held else train).append(folder)
    args.train_out.write_text("".join(f"{p}\n" for p in train))
    args.heldout_out.write_text("".join(f"{p}\n" for p in heldout))
    print(f"training scenes {len(train)} (with Notre-Dame), held-out scenes {len(heldout)}, "
          f"not ready {len(waiting)}: {' '.join(waiting)}")


if __name__ == "__main__":
    main()
