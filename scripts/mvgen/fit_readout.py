#!/usr/bin/env python
"""Fit the colour-blind state read-out of experiment look1 on the training views of its hold-out (CPU).

The read-out (``timemachine.mvgen.readout``: frozen DINOv2-base on the grey, contrast-normalised crop
around the changed region, and a logistic probe) says whether a picture shows Notre-Dame before or after
the fire; look1's read-outs R2 and R4 apply it to renders of held-out views. So the probe must never
have seen those views: it is fitted here on the TRAINING views of the hold-out that the evaluation-sets
file names (``configs/look1_eval_sets.json``, ``scripts/mvgen/make_eval_sets.py``), and the file it writes
lists them (``StateReadout.check_unseen`` refuses to read any of them).

The photos are those of the appearance study (``data/appearance_study/design_learned/04a``): usable
views of the modern period or after the fire whose changed region covers at least 3% of the frame
(the ``share`` of ``changed_regions.npz``) and whose stored mask is not empty; label 1 after the fire.
Crops come from the 512 crop that ``MVGenScene.make_sample`` gives (rounded to 8 bits) and the stored
mask, exactly as the study made them.

Printed, and kept in the file's record:

* the study's check, on all of those photos: the area under the ROC curve with 5 folds grouped by
  capture day, 3 repeats (the study: 0.998), and how the out-of-fold logit moves with the season inside a
  period (spring minus the rest, leaf-on minus leaf-off; the study: about -0.9 after the fire, -0.5 to 0
  before it);
* the same on the training views alone;
* the fitted probe on the held-out photos, which it has never seen: AUC, accuracy and the logit gap
  between the periods.

Runs in MVGenMaster's environment (OpenCV, scikit-learn) from its checkout, with the Hugging Face cache
that holds ``facebook/dinov2-base`` (nothing is downloaded); a few minutes on 4 CPU threads:

    cd external/MVGenMaster && PYTHONPATH=$PWD HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 nice .venv/bin/python \\
        $REPO/scripts/mvgen/fit_readout.py

Refuses to overwrite its output without ``--overwrite``.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch

from timemachine.data.dataset import load_view
from timemachine.mvgen.data import REGIONS_FILE, MVGenScene, ViewSamplingConfig
from timemachine.mvgen.readout import (
    CROP_SIZE, DINOV2_REPO, MIN_REGION_SHARE, PROBE_C, DinoV2, dinov2_folder, encode, fit_probe, readout_crop,
    save_probe,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / "data/datasets/25512_model0_v2"
DEFAULT_SETS = REPO / "configs/look1_eval_sets.json"
READOUT_FILE = "state_readout.npz"          # in the dataset folder: where eval_dates.py looks for it
FOLDS, REPEATS = 5, 3
MONTH_PRECISION = 0.09                      # a date known to this many years (about a month) has a season
LEAF_ON = (105, 310)                        # days of the year with leaves, as the appearance study


def photo(scene: MVGenScene, row: int) -> np.ndarray:
    image, _ = load_view(scene.views.iloc[row], scene.resolution)
    return ((image.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)


def capture_days(scene: MVGenScene) -> np.ndarray:
    """An integer per view: its capture day (``floor(date_lo * 365.25)``), the study's fold groups."""
    return np.floor(scene.views["date_lo"].to_numpy() * 365.25 + 1e-6).astype(np.int64)


def seasons(scene: MVGenScene) -> tuple[np.ndarray, np.ndarray]:
    """Per view: spring (March to May) and leaf-on (days 105 to 310), or NaN where the date is not known to a month."""
    def moment(year: float) -> datetime:
        whole = int(np.floor(year))
        start, end = datetime(whole, 1, 1), datetime(whole + 1, 1, 1)
        return start + (end - start) * (year - whole)

    views = scene.views
    known = (views["date_hi"] - views["date_lo"]).to_numpy() <= MONTH_PRECISION
    moments = [moment(year) for year in views["date_mid"]]
    day = np.array([m.timetuple().tm_yday for m in moments])
    month = np.array([m.month for m in moments])
    spring = np.where(known, np.isin(month, (3, 4, 5)).astype(float), np.nan)
    leaf = np.where(known, ((day >= LEAF_ON[0]) & (day <= LEAF_ON[1])).astype(float), np.nan)
    return spring, leaf


def out_of_fold(features: np.ndarray, after: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, float]:
    """Out-of-fold logits ``[repeats, n]`` with folds grouped by ``groups`` and the study's AUC (taken inside each
    test fold, weighted by its size)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    logits = np.zeros((REPEATS, len(after)))
    total = weight = 0.0
    for repeat in range(REPEATS):
        for train, test in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=repeat).split(features, after, groups):
            logits[repeat, test] = fit_probe(features[train], after[train]).logit(features[test])
            if len(np.unique(after[test])) == 2:
                total += roc_auc_score(after[test], logits[repeat, test]) * len(test)
                weight += len(test)
    return logits, float(total / weight)


def within_period(logit: np.ndarray, after: np.ndarray, cue: np.ndarray) -> dict:
    """Per period: the mean logit where the cue holds minus where it does not, with its standard error."""
    out = {}
    for name, period in (("modern", False), ("after", True)):
        known = (after == period) & np.isfinite(cue)
        yes, no = logit[known & (cue == 1)], logit[known & (cue == 0)]
        if len(yes) < 5 or len(no) < 5:
            out[name] = None
            continue
        out[name] = {"difference": float(yes.mean() - no.mean()),
                     "se": float(np.sqrt(yes.var(ddof=1) / len(yes) + no.var(ddof=1) / len(no))),
                     "n": [int(len(yes)), int(len(no))]}
    return out


def check(features: np.ndarray, after: np.ndarray, groups: np.ndarray, spring: np.ndarray, leaf: np.ndarray) -> dict:
    """The study's check of the read-out on these photos (see the module docstring)."""
    logits, auc = out_of_fold(features, after, groups)
    logit = logits.mean(0)
    return {"photos": int(len(after)), "after_share": float(after.mean()), "auc_by_day": auc,
            "logit_gap": float(logit[after].mean() - logit[~after].mean()),
            "sd_within": float(np.sqrt((logit[after].var() + logit[~after].var()) / 2)),
            "spring_minus_other": within_period(logit, after, spring),
            "leaf_on_minus_off": within_period(logit, after, leaf)}


def describe(name: str, result: dict) -> str:
    def cue(of: dict) -> str:
        return "; ".join(f"{period} " + ("n/a" if v is None else f"{v['difference']:+.2f} +- {v['se']:.2f} "
                                                                  f"({v['n'][0]}/{v['n'][1]})")
                         for period, v in of.items())

    return (f"{name}: {result['photos']} photos ({result['after_share']:.0%} after the fire); AUC, folds by day "
            f"{result['auc_by_day']:.4f}; logit gap {result['logit_gap']:.2f}, sd inside a period "
            f"{result['sd_within']:.2f}\n   spring minus other: {cue(result['spring_minus_other'])}\n   leaf-on minus "
            f"leaf-off: {cue(result['leaf_on_minus_off'])}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-sets", type=Path, default=DEFAULT_SETS,
                   help="the evaluation-sets file whose hold-out the probe must not see (default: %(default)s)")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="dataset folder (default: %(default)s)")
    p.add_argument("--out", type=Path, help=f"the probe file (default: <dataset>/{READOUT_FILE})")
    p.add_argument("--overwrite", action="store_true", help="replace an existing probe file")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from sklearn.metrics import roc_auc_score

    args = parse_args(argv)
    out = args.out or args.dataset / READOUT_FILE
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists: give --overwrite to replace it")
    torch.set_num_threads(4)
    scene = MVGenScene(args.dataset, ViewSamplingConfig(), resolution=512)
    keys = scene.views["file_key"].tolist()
    plan = json.loads(args.eval_sets.read_text())
    held_rows = plan["holdout"]["rows"]
    if [keys[row] for row in held_rows] != plan["holdout"]["file_keys"]:
        raise SystemExit(f"{args.eval_sets} was made for another view table than {args.dataset}'s")
    held = np.zeros(len(keys), dtype=bool)
    held[held_rows] = True
    with np.load(args.dataset / REGIONS_FILE) as regions:
        if regions["file_keys"].tolist() != keys:
            raise SystemExit(f"{args.dataset / REGIONS_FILE} was made for another view table")
        share = regions["share"]
    masks_exist = (scene.regions > 0).reshape(len(keys), -1).any(1)
    rows = np.flatnonzero(scene.usable & (scene.period >= 1) & (share >= MIN_REGION_SHARE) & masks_exist)
    after = scene.period[rows] == 2
    print(f"{len(rows)} photos with a region of {MIN_REGION_SHARE:.0%} of the frame or more ({after.sum()} after the "
          f"fire); {held[rows].sum()} of them held out by {args.eval_sets.name} (seed {plan['holdout']['seed']})",
          flush=True)

    folder = dinov2_folder()
    encoder = DinoV2.from_folder(folder)
    crops = np.stack([readout_crop(photo(scene, row), np.repeat(np.repeat(scene.regions[row] > 0, 8, 0), 8, 1))
                      for row in rows])
    print(f"encoding {len(crops)} crops of {CROP_SIZE} px with {DINOV2_REPO} ({folder.name}) ...", flush=True)
    features = encode(encoder, crops)
    groups = capture_days(scene)[rows]
    spring, leaf = (cue[rows] for cue in seasons(scene))

    study = check(features, after, groups, spring, leaf)
    print(describe("all photos (the study's check)", study), flush=True)
    training = ~held[rows]
    on_training = check(features[training], after[training], groups[training], spring[training], leaf[training])
    print(describe("training photos of the hold-out", on_training), flush=True)

    probe = fit_probe(features[training], after[training])
    logit = probe.logit(features[~training])
    unseen = {"photos": int((~training).sum()), "after": int(after[~training].sum()),
              "auc": float(roc_auc_score(after[~training], logit)),
              "accuracy": float(((logit > 0) == after[~training]).mean()),
              "logit_after": float(logit[after[~training]].mean()), "logit_modern": float(logit[~after[~training]].mean())}
    print(f"fitted on {training.sum()} training photos; on the {unseen['photos']} held-out photos it never saw: AUC "
          f"{unseen['auc']:.4f}, accuracy {unseen['accuracy']:.3f}, mean logit after the fire {unseen['logit_after']:+.2f}, "
          f"before it {unseen['logit_modern']:+.2f}", flush=True)
    record = {"made_on": date.today().isoformat(), "dataset": str(args.dataset), "eval_sets": str(args.eval_sets),
              "holdout_seed": plan["holdout"]["seed"], "encoder": DINOV2_REPO, "encoder_snapshot": folder.name,
              "probe_c": PROBE_C, "min_region_share": MIN_REGION_SHARE,
              "selection": f"usable, modern or after the fire, region share >= {MIN_REGION_SHARE}, stored mask not "
                           "empty; label: after the fire",
              "fitted_on": [keys[row] for row in rows[training]], "fitted_rows": rows[training].tolist(),
              "check_all_photos": study, "check_training_photos": on_training, "heldout_photos": unseen}
    save_probe(out, probe, record)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
