#!/usr/bin/env python
"""Write a dataset folder's changed-region weights: where the scene changed at a period edge, per view (CPU).

Builds the masks of ``timemachine.mvgen.regions.ChangedRegions`` (for Notre-Dame: the spire, the roofs
and the scaffolding at the crossing, in views of either period) for every view of the folder and
writes ONE file into it, ``changed_regions.npz``: each mask reduced to the latents' resolution (per
latent cell, how many of its 8 x 8 pixels are masked), with the date of the change, the crop size, the
thresholds the masks were made with and each view's masked share of the frame (the layout is in
``timemachine/mvgen/data.py``).

A mask that covers less than ``--min-share`` of the frame is stored EMPTY. In frontal views of the west
facade the mask is specks on unrelated stone, not the spire (``timemachine/mvgen/regions.py``), and
such a view must count as showing no region. The rule removes most of them, not all. With the default
of 2%, 46 of Notre-Dame's 484 usable frontal views keep a region (618 usable views have one). Of 20
of them looked at on 2026-11-20, 17 show the facade from close by (portals, statues, a tower from
below) and their region is still specks on that stone, 2% to 15% of the frame. A size threshold
cannot tell them from real regions, nor can the size of the largest connected area. With a region
weight of 10, frontal views are the whole region of 8% of the training samples that show one, and
hold 1.5% of the loss weight that lies inside regions.

What the file changes. A scene opened on the folder hands the weights to its samples
(``sample["region"]``). Training uses them only with ``--region-weight`` other than 1
(``scripts/mvgen/train.py``); a run without it trains exactly as before. The date-loss probes of every
run on the folder additionally report the loss gaps inside and outside the region; their other
numbers, their files and their baselines are unchanged.

An existing file is not replaced unless ``--overwrite`` is given: a run that trains with a region
weight stores a hash of the weights and is not resumed on other ones.

Runs in MVGenMaster's environment from its checkout, on one CPU core in a minute or two, so give
absolute paths:

    cd external/MVGenMaster && PYTHONPATH=$PWD .venv/bin/python $REPO/scripts/mvgen/prepare_regions.py
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from timemachine.mvgen.data import (
    LATENT_FACTOR, REGIONS_FILE, MVGenScene, ViewSamplingConfig, period_names, write_region_file,
)
from timemachine.mvgen.regions import FIRE, ChangedRegions, view_region_cells

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / "data/datasets/25512_model0_v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="dataset folder; the file is written into it")
    p.add_argument("--edge", type=float, default=FIRE, help="the date at which the scene changed (decimal year)")
    p.add_argument("--period-edges", type=float, nargs="*", default=list(ViewSamplingConfig.period_edges),
                   help="the scene's period edges; the two periods next to --edge are compared")
    p.add_argument("--resolution", type=int, default=512,
                   help="side of the crops the masks are made for: the resolution of the runs that will use them")
    p.add_argument("--min-share", type=float, default=0.02,
                   help="a mask that covers less than this share of the frame is stored empty")
    p.add_argument("--overwrite", action="store_true", help=f"replace an existing {REGIONS_FILE}")
    return p.parse_args(argv)


def region_table(scene: MVGenScene, share: np.ndarray, min_share: float) -> pd.DataFrame:
    """Per period, over the usable views: how many there are, how many have a region (a mask of at least
    ``min_share`` of the frame), how many a smaller mask (stored empty) and how many none; and the median and
    the largest share of the frame among the views with a region.

    ``share`` is each view's masked share of the frame (NaN: no readable image, counted as no mask).
    """
    names = period_names(scene.config.period_edges)
    share = np.nan_to_num(share, nan=0.0)
    rows = []
    for period in range(len(names)):
        of_period = share[scene.usable & (scene.period == period)]
        kept = of_period[of_period >= min_share]
        rows.append({"usable": len(of_period), "with_region": len(kept),
                     "too_small": int(((of_period > 0) & (of_period < min_share)).sum()),
                     "no_mask": int((of_period == 0).sum()),
                     "median_share": float(np.median(kept)) if len(kept) else float("nan"),
                     "largest_share": float(kept.max()) if len(kept) else float("nan")})
    return pd.DataFrame(rows, index=names)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path = args.dataset / REGIONS_FILE
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists: give --overwrite to replace it (runs that train with a region weight on "
                         "the old file are then not resumed)")
    scene = MVGenScene(args.dataset, ViewSamplingConfig(period_edges=tuple(args.period_edges)),
                       resolution=args.resolution, require_depth=False)
    regions = ChangedRegions(scene, args.edge)
    print(f"changed points at {args.edge:.4f}: " + "; ".join(
        f"{kind}: " + ", ".join(f"{count} {rule}" for rule, count in counts.items())
        for kind, counts in regions.points.counts.items()), flush=True)
    cells, share = view_region_cells(regions, args.resolution, args.min_share)
    unreadable = int(np.isnan(share).sum())
    if unreadable:
        print(f"note: {unreadable} of the {len(share)} views have no readable image and get no region "
              f"({int((np.isnan(share) & scene.usable).sum())} of them usable)", flush=True)

    written = write_region_file(args.dataset, cells, scene.views["file_key"].tolist(), share,
                                resolution=args.resolution, min_share=args.min_share, edge=args.edge,
                                config=asdict(regions.config))
    table = region_table(scene, share, args.min_share)
    print(f"usable views with a region (a mask of at least {args.min_share:.1%} of the frame), per period:\n"
          + table.to_string(float_format=lambda value: f"{value:.3f}"), flush=True)
    kept = cells.any(axis=(1, 2))
    covered = (f"; in those views it covers {cells[kept].mean() / LATENT_FACTOR ** 2:.1%} of the frame on average"
               if kept.any() else "")
    print(f"wrote {written} ({written.stat().st_size / 1e3:.0f} kB): {int(kept.sum())} of the {len(kept)} views of "
          f"the table have a region ({int((kept & scene.usable).sum())} of the {int(scene.usable.sum())} usable "
          f"ones){covered}")


if __name__ == "__main__":
    main()
