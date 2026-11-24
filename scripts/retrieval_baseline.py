#!/usr/bin/env python
"""Model-free retrieval baselines on a single-scene dataset (CPU only, no SEVA needed).

For each sample (3 input views + 5 target views, drawn exactly as in
scripts/seva_baseline.py), "predict" every target by handing back a real photo:

    input_date      the input nearest in date (ties: nearest camera)
    input_pose      the input nearest in camera (date-blind control)
    pool_date_pose  from the whole scene: nearest dates, then nearest camera
                    (never a near-duplicate of the target; optimistic until the
                    repo has a train/validation split)

and score it with PSNR against the true target. See
src/timemachine/eval/retrieval.py for the reasoning and the caveats.

    python scripts/retrieval_baseline.py --dataset data/datasets/25512_model0 --samples 200

To compare with plain SEVA on the SAME targets, run seva_baseline.py first with the
same dataset and seed, then pass its psnr.csv:

    python scripts/retrieval_baseline.py --dataset data/datasets/25512_model0 \
        --samples 8 --compare data/experiments/seva_baseline/25512_model0_random/psnr.csv

The --samples and --seed must match the SEVA run, or the targets won't line up
(the comparison matches on sample, view and file, so a mismatch shows up as zero
matched targets instead of a wrong answer).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from timemachine.data.dataset import DatedViewsDataset, load_view
from timemachine.data.sampling import SamplingConfig
from timemachine.eval.retrieval import BASELINES, compare_with_seva, run_baselines, summarize
from timemachine.eval.splits import DATE_TOL, POSE_TOL, near_duplicate_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="folder with views.parquet")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=576,
                        help="image size for scoring; keep 576 to match seva_baseline.py")
    parser.add_argument("--k-date", type=int, default=20,
                        help="pool_date_pose: how many nearest-in-date photos to consider")
    parser.add_argument("--date-tol", type=float, default=DATE_TOL,
                        help="near-duplicates: max date difference in years (default %(default)s)")
    parser.add_argument("--pose-tol", type=float, default=POSE_TOL,
                        help="near-duplicates: max camera distance, see timemachine.eval.splits (default %(default)s)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="don't exclude near-duplicates of the target from the pool (shows how much they inflate scores)")
    parser.add_argument("--compare", type=Path, help="psnr.csv from scripts/seva_baseline.py")
    parser.add_argument("--out", type=Path, help="output folder (default: data/experiments/retrieval_baseline/<dataset>)")
    args = parser.parse_args()

    out = args.out or Path("data/experiments/retrieval_baseline") / args.dataset.name
    out.mkdir(parents=True, exist_ok=True)

    views = pd.read_parquet(args.dataset / "views.parquet")
    # Same sampling settings as scripts/seva_baseline.py, so the samples are identical.
    dataset = DatedViewsDataset(views, SamplingConfig(n_inputs=3, n_targets=5, old_target_fraction=0.5),
                                resolution=args.resolution, samples_per_epoch=args.samples, seed=args.seed)

    def load_image(row: pd.Series) -> np.ndarray:
        pixels, _intrinsics = load_view(row, dataset.resolution)   # (3, S, S) in [-1, 1], centre-cropped
        return pixels.numpy()

    # Group views that are near-duplicates (same moment, same spot) so the pool baseline
    # can't simply retrieve another copy of the target.
    groups = near_duplicate_groups(dataset.dates, dataset.c2ws, args.date_tol, args.pose_tol)
    sizes = np.bincount(groups)
    print(f"{len(views)} views in {len(sizes)} near-duplicate groups "
          f"({int((sizes > 1).sum())} groups of 2+, largest {sizes.max()})", flush=True)

    table = run_baselines(dataset, args.samples, load_image, tuple(BASELINES), k_date=args.k_date,
                          groups=None if args.no_dedup else groups)
    table.to_csv(out / "retrieval_baseline.csv", index=False)

    report = summarize(table)
    report["near_duplicate_groups"] = {"views": int(len(views)), "groups": int(len(sizes)),
                                       "groups_of_2_plus": int((sizes > 1).sum()), "largest": int(sizes.max())}
    if args.compare:
        report["vs_seva"] = compare_with_seva(table, pd.read_csv(args.compare))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out / 'retrieval_baseline.csv'} and {out / 'report.json'}")


if __name__ == "__main__":
    main()
