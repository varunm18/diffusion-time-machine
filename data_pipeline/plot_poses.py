"""Sanity-check camera poses against dates: do OLD photos sit where they should?

WHY THIS FILE EXISTS
--------------------
A correct date is useless if the camera pose attached to it is wrong. Old
photographs (1860s albumen prints, say) are the hardest images for
structure-from-motion to place, so their poses are the ones most likely to be
off. This script checks whether the old photos registered at believable
vantage points, meaning roughly where modern tourists stood.

HOW TO RUN (from the repo root)
-------------------------------
    python -m pip install matplotlib        # once
    python data_pipeline/plot_poses.py data/raw/taj/colmap/0 data/processed/image_dates.csv

It writes data/processed/pose_map.png (a top-down map of camera positions,
modern ones colored by year, pre-1950 ones as big red dots) and prints stats.

HOW THE CHECK WORKS
-------------------
 * "Scene center" = the median position of the modern (2000+) cameras.
 * Each camera's "radius" = its distance from that center.
 * We look at the range of radii that 90% of modern cameras fall in (5th to
   95th percentile). An old photo is "plausible" if its radius lands inside that
   band. For a perfectly typical photo there's a ~90% chance of landing inside,
   so a score near 0.9 is the baseline; a much lower score means trouble.
 * The check is TWO-SIDED on purpose. Too far out can mean a wild pose, and too
   close can mean the camera was placed inside the building. Our first version
   only checked "too far" and missed the second case.

IMPORTANT LIMITS
----------------
This is a coarse test. Passing it does not prove a pose is right. It only
catches poses that are clearly implausible. Anything printed as flagged below is
worth opening by hand, next to its Commons page.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.colmap_reader import read_images_binary  # noqa: E402


def plot_axes(centers: np.ndarray) -> np.ndarray:
    """Flatten 3D camera positions to 2D for plotting, using PCA.

    COLMAP's world frame has no guaranteed "up" direction, so we can't just drop
    one coordinate. Instead we project onto the two directions in which the
    cameras are most spread out, which gives a sensible top-down-ish map.
    """
    c = centers - np.median(centers, axis=0)           # center the cloud
    _, _, vt = np.linalg.svd(c, full_matrices=False)    # rows of vt = principal directions
    return c @ vt[:2].T                                 # keep the top two


def old_vs_modern_stats(centers: np.ndarray, years: np.ndarray,
                        old_cut: float = 1950, modern_cut: float = 2000) -> dict:
    """Compare where old cameras sit with where modern cameras sit.

    Returns counts, median radii, the modern 5th/95th percentile radii, and:
      within_modern_range  fraction of old cameras inside that band (baseline ~0.9)
      too_far / too_close  how many old cameras fell outside it, and on which side
    """
    old = years < old_cut
    modern = years >= modern_cut

    center = np.median(centers[modern], axis=0)
    radius = np.linalg.norm(centers - center, axis=1)

    lo = np.percentile(radius[modern], 5)
    hi = np.percentile(radius[modern], 95)

    old_r = radius[old]
    return {
        "n_old": int(old.sum()),
        "n_modern": int(modern.sum()),
        "median_radius_modern": float(np.median(radius[modern])),
        "median_radius_old": float(np.median(old_r)) if old.any() else float("nan"),
        "modern_p05_radius": float(lo),
        "modern_p95_radius": float(hi),
        "within_modern_range": float(((old_r >= lo) & (old_r <= hi)).mean()) if old.any() else float("nan"),
        "too_far": int((old_r > hi).sum()),
        "too_close": int((old_r < lo).sum()),
    }


def load_years(csv_path: str, min_conf: float, min_year: int) -> Dict[str, float]:
    """Read image_dates.csv into {image name: decimal year}, keeping only usable dates."""
    years: Dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["date_start"] == "" or float(r["date_conf"]) < min_conf:
                continue  # undated, or too shaky to trust
            y = (int(r["date_start"]) + int(r["date_end"]) + 1) / 2.0
            if y >= min_year:  # skip anything dated before photography existed
                years[r["colmap_name"]] = y
    return years


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot camera positions colored by year.")
    ap.add_argument("model_dir", help="COLMAP model folder containing images.bin")
    ap.add_argument("dates_csv", help="image_dates.csv from commons_dates.py")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--min-year", type=int, default=1826)
    ap.add_argument("--out", default="data/processed/pose_map.png")
    args = ap.parse_args()

    images = read_images_binary(Path(args.model_dir) / "images.bin")
    years_by_name = load_years(args.dates_csv, args.min_conf, args.min_year)

    # Pair each registered image's camera position with its year.
    centers, years, names = [], [], []
    for im in images.values():
        if im.name in years_by_name:
            centers.append(im.camera_center())
            years.append(years_by_name[im.name])
            names.append(im.name)
    centers, years = np.array(centers), np.array(years)
    print(f"{len(years)} registered images with a usable date")

    stats = old_vs_modern_stats(centers, years)
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # List the old cameras that fall outside the modern band, on either side.
    # These are the ones to open by hand.
    center = np.median(centers[years >= 2000], axis=0)
    radius = np.linalg.norm(centers - center, axis=1)
    flagged = [i for i in np.argsort(years)
               if years[i] < 1950 and not (stats["modern_p05_radius"] <= radius[i] <= stats["modern_p95_radius"])]
    if flagged:
        print("\nold photos OUTSIDE the modern distance band (inspect these by hand):")
        for i in flagged:
            side = "too far" if radius[i] > stats["modern_p95_radius"] else "too close"
            print(f"  {years[i]:.0f}  radius {radius[i]:.2f}  ({side})  {names[i].rsplit('/', 1)[-1]}")

    # Draw the map. We zoom to the central 98% of cameras so a few far-away
    # outliers can't squash everything else into a dot.
    xy = plot_axes(centers)
    lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)

    import matplotlib
    matplotlib.use("Agg")  # draw to a file; no window needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    modern = years >= 1950
    sc = ax.scatter(xy[modern, 0], xy[modern, 1], c=years[modern], s=4, alpha=0.4, cmap="viridis")
    ax.scatter(xy[~modern, 0], xy[~modern, 1], c="red", s=60, edgecolors="black",
               label=f"before 1950 (n={(~modern).sum()})")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title("Camera positions (top-down, principal axes), colored by year")
    ax.legend()
    fig.colorbar(sc, label="year (1950+ shown in color)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
