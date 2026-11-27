#!/usr/bin/env python
"""Exclude from newly built scene folders every photo that an existing run holds out (before training on them).

Commons files can belong to several MegaScenes scenes (Notre-Dame's photos also appear in the Louvre's or
the Sainte-Chapelle's scene). When scenes are added to a run, a photo that the run holds out for
evaluation (a held-out view of a training scene, as ``SceneMixture`` draws the hold-out, or any view of a
held-out scene) must not come back as a training view of a new scene. For each new folder this script
sets ``excluded = "heldout_leak"`` and ``usable = False`` on the views whose file is held out, in
``views.parquet`` (rows are not renumbered, so depth and look codes stay valid). Already-excluded views
keep their reason. Run it on the new folders only; it refuses a folder that is in the training or held-out list.

    python scripts/mvgen/mark_heldout_leaks.py --train-list data/multiscene_plan/scenes_train.txt \\
        --heldout-list data/multiscene_plan/scenes_heldout.txt --new <folder> [<folder> ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd

from timemachine.mvgen.data import ViewSamplingConfig
from timemachine.mvgen.multiscene import SceneMixture

REASON = "heldout_leak"


def nfc(key: str) -> str:
    return unicodedata.normalize("NFC", str(key))


def read_list(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def heldout_keys(train: list[Path], heldout: list[Path]) -> dict[str, str]:
    """File key -> where it is held out: the held-out views of the training scenes (the run's own hold-out rule,
    ``SceneMixture``) and every view of the held-out scenes."""
    keys: dict[str, str] = {}
    mixture = SceneMixture(train, ViewSamplingConfig())          # the run's scenes and hold-out, without looks
    for name, scene in zip(mixture.names, mixture.scenes):
        for key in scene.views["file_key"].to_numpy()[scene.heldout]:
            keys.setdefault(nfc(key), f"held-out view of {name}")
    if len(mixture.names) != len(train):
        raise SystemExit(f"{len(train) - len(mixture.names)} training scenes could not be opened: the hold-out would be incomplete")
    for folder in heldout:
        for key in pd.read_parquet(folder / "views.parquet", columns=["file_key"])["file_key"]:
            keys.setdefault(nfc(key), f"view of held-out scene {folder.name}")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--heldout-list", type=Path, required=True)
    parser.add_argument("--new", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, help="write what was excluded, per folder, as JSON")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train, heldout = read_list(args.train_list), read_list(args.heldout_list)
    listed = {p.resolve() for p in train + heldout}
    clash = [str(p) for p in args.new if p.resolve() in listed]
    if clash:
        raise SystemExit(f"these folders are already in a list, not new: {clash}")
    keys = heldout_keys(train, heldout)
    print(f"{len(keys)} held-out files ({len(train)} training scenes, {len(heldout)} held-out scenes)", flush=True)
    report = {}
    for folder in args.new:
        path = folder / "views.parquet"
        views = pd.read_parquet(path)
        hit = views["file_key"].map(lambda k: nfc(k) in keys).to_numpy()
        newly = hit & views["usable"].to_numpy(dtype=bool)
        report[folder.name] = {"views": len(views), "held_out_files": int(hit.sum()), "newly_excluded": int(newly.sum()),
                               "where": sorted({keys[nfc(k)] for k in views["file_key"][hit]})}
        print(f"{folder.name}: {int(hit.sum())} held-out files, {int(newly.sum())} usable views excluded "
              f"{report[folder.name]['where'][:3]}", flush=True)
        if newly.any() and not args.dry_run:
            views.loc[newly, "excluded"] = REASON
            views.loc[newly, "usable"] = False
            tmp = path.with_name(path.name + ".tmp")
            views.to_parquet(tmp, index=False)
            tmp.replace(path)
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
