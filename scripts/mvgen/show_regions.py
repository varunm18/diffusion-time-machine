#!/usr/bin/env python
"""Overlay sheets of the changed-region masks: where the scene differs across a period edge (CPU).

For every view given, one row of a sheet: the view's crop as a training sample has it, and the same
crop with the mask of ``timemachine.mvgen.regions.ChangedRegions`` tinted red and outlined. Each row
is labelled with the view's table row, its year, whether it lies before or after the edge and the
share of the frame that is masked. A few views go on each sheet (``<out>/<name>_<i>.jpg``), and
``<out>/<name>.json`` lists the rows, years and shares together with the number of changed points
each rule kept.

The views are table rows (``--rows``), the views of a keys file (``--keys-file``: every reference
and target of its view sets, once), or ``--spread N``: N usable views of the two periods next to
the edge, half from each, spread evenly over the directions from which the cameras look at the
changed points.

Runs in MVGenMaster's environment from its checkout, on the CPU, so give absolute paths:

    cd external/MVGenMaster && PYTHONPATH=$PWD .venv/bin/python $REPO/scripts/mvgen/show_regions.py \\
        --keys-file $REPO/configs/mvgen_gate2.json --out $REPO/data/mvgen_regions/sheets --name gate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from timemachine.data.dataset import load_view
from timemachine.mvgen.data import MVGenScene, ViewSamplingConfig
from timemachine.mvgen.probes import read_keys_file, to_uint8
from timemachine.mvgen.regions import FIRE, ChangedRegions

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / "data/datasets/25512_model0_v2"
TINT = np.array([255, 40, 40], dtype=np.float32)
_LABEL = 16                     # height of a row's label, in pixels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="output folder")
    p.add_argument("--name", default="regions", help="file name stem of the sheets and the JSON")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="dataset folder")
    p.add_argument("--rows", type=int, nargs="*", default=[], help="view-table rows to show")
    p.add_argument("--keys-file", type=Path,
                   help='show the views of these view sets ({"refs": [file_key, ...], "targets": [...]} each)')
    p.add_argument("--n-ref", type=int, default=ViewSamplingConfig.n_ref,
                   help="references per view set of the keys file")
    p.add_argument("--spread", type=int, default=0,
                   help="also show this many usable views spread over viewing directions and both periods")
    p.add_argument("--edge", type=float, default=FIRE, help="the date at which the scene changed (decimal year)")
    p.add_argument("--period-edges", type=float, nargs="*", default=list(ViewSamplingConfig.period_edges),
                   help="the scene's period edges; the two periods next to --edge are compared")
    p.add_argument("--resolution", type=int, default=512, help="side of the crops the masks are made for")
    p.add_argument("--per-sheet", type=int, default=4, help="views per sheet")
    p.add_argument("--thumb", type=int, default=384, help="side of the sheet's pictures, in pixels")
    return p.parse_args(argv)


def spread_rows(regions: ChangedRegions, n: int) -> list[int]:
    """``n`` usable views, half from each side of the edge, spread over the directions from which they look
    at the changed points.

    A view's direction is the compass angle of its camera around the centre of the changed points, in
    the plane of the first two principal axes of the camera positions (the ground plane, for
    photographs taken from the ground). On each side the views that have the changed points in
    front of them (within 45 degrees of their viewing direction) are sorted by that angle and picked
    at even steps through the list; an odd ``n`` gives the side before the edge one view more.
    """
    scene = regions.scene
    centres = scene.c2ws[:, :3, 3]
    middle = np.median(regions.xyz, axis=0)
    forward = scene.c2ws[:, :3, 2]
    facing = ((middle - centres) * forward).sum(axis=1) > 0.7 * np.linalg.norm(middle - centres, axis=1)
    used = centres[regions.side >= 0]
    axes = np.linalg.svd(used - used.mean(axis=0), full_matrices=False)[2][:2]
    offset = (centres - middle) @ axes.T
    angle = np.arctan2(offset[:, 1], offset[:, 0])
    rows: list[int] = []
    for side in (0, 1):
        pool = np.flatnonzero((regions.side == side) & facing)
        pool = pool[np.argsort(angle[pool], kind="stable")]
        count = min(len(pool), n // 2 + (n % 2 if side == 0 else 0))
        rows.extend(int(row) for row in pool[np.arange(count) * len(pool) // max(count, 1)])
    return rows


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``image`` (uint8 ``[H, W, 3]``) with the masked pixels tinted red and the mask's outline drawn."""
    tinted = image.astype(np.float32)
    tinted[mask] = 0.55 * tinted[mask] + 0.45 * TINT
    outline = mask & ~ndimage.binary_erosion(mask, iterations=1)
    tinted[outline] = TINT
    return tinted.astype(np.uint8)


def sheet(tiles: list[tuple[str, np.ndarray, np.ndarray]], thumb: int) -> Image.Image:
    """One row per view: its label, then the crop and the crop with the mask (``tiles``: label, image, overlay)."""
    page = Image.new("RGB", (2 * thumb, len(tiles) * (thumb + _LABEL)), "white")
    draw = ImageDraw.Draw(page)
    for i, (label, image, marked) in enumerate(tiles):
        top = i * (thumb + _LABEL)
        draw.text((4, top + 2), label, fill="black")
        for j, picture in enumerate((image, marked)):
            page.paste(Image.fromarray(picture).resize((thumb, thumb), Image.Resampling.LANCZOS),
                       (j * thumb, top + _LABEL))
    return page


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    scene = MVGenScene(args.dataset, ViewSamplingConfig(period_edges=tuple(args.period_edges)),
                       resolution=args.resolution, require_depth=False)
    regions = ChangedRegions(scene, args.edge)
    print(f"changed points at {args.edge:.4f}: " + "; ".join(
        f"{kind}: " + ", ".join(f"{count} {rule}" for rule, count in counts.items())
        for kind, counts in regions.points.counts.items()), flush=True)

    rows = list(args.rows)
    if args.keys_file is not None:
        for view_set in read_keys_file(args.keys_file, scene.views["file_key"].tolist(), args.n_ref):
            rows.extend(view_set)
    rows.extend(spread_rows(regions, args.spread))
    rows = list(dict.fromkeys(rows))
    if not rows:
        raise SystemExit("no views to show: give --rows, --keys-file or --spread")

    args.out.mkdir(parents=True, exist_ok=True)
    records, tiles = [], []
    for row in rows:
        image = to_uint8(load_view(next(scene.views.iloc[[row]].itertuples()), args.resolution)[0][None])[0]
        mask = regions.mask(row, args.resolution)
        when = {0: "before", 1: "after", -1: "not used"}[int(regions.side[row])]
        records.append({"row": row, "file_key": scene.views.file_key[row], "year": round(float(scene.dates[row]), 4),
                        "side": when, "share": regions.share(row, args.resolution)})
        label = f"row {row}   {scene.dates[row]:.2f} ({when})   mask {records[-1]['share']:.1%} of the frame"
        tiles.append((label, image, overlay(image, mask)))
        print(f"   {label}", flush=True)
    for i in range(0, len(tiles), args.per_sheet):
        sheet(tiles[i:i + args.per_sheet], args.thumb).save(args.out / f"{args.name}_{i // args.per_sheet}.jpg",
                                                            quality=90)
    (args.out / f"{args.name}.json").write_text(json.dumps(
        {"dataset": str(args.dataset), "edge": args.edge, "resolution": args.resolution,
         "points": regions.points.counts, "views": records}, indent=1))
    print(f"wrote {-(-len(tiles) // args.per_sheet)} sheets and {args.name}.json to {args.out}")


if __name__ == "__main__":
    main()
