"""Audit how well-dated, and how date-diverse, a scene is.

WHY THIS FILE EXISTS
--------------------
Our whole project depends on having OLD dated photos next to MODERN ones of the
same place. Before building anything on top of a scene, we need to know: does
this scene actually have enough of both? This script answers that, and it's
the go/no-go check from our proposal (10+ images before 1970 and 10+ modern).

HOW TO RUN (from the repo root)
-------------------------------
    python data_pipeline/audit_date_coverage.py data/processed/image_dates.csv --min-conf 0.5

The input is the CSV written by commons_dates.py.

WHAT IT PRINTS
--------------
 * how many images have a date, how many don't, how many we dropped
 * the date sources (which kind of metadata the dates came from)
 * a text histogram of images per decade
 * counts before 1900 / 1950 / 1970 and from 2000 on
 * unique Commons titles (the same file can appear twice under different
   subcategories; duplicates must never straddle the train/validation split)
 * the 15 OLDEST images with their raw Commons date text

ALWAYS SPOT-CHECK THE OLDEST 15 BY HAND
---------------------------------------
A wrong "old" date is far more damaging than a missing date: a modern photo
mislabelled 1648 teaches the model nonsense. This is how we caught a broken
camera clock ("1648"), museum date strings that were being misread, and a
print-vs-negative mix-up. Open a few of those Commons pages and compare.
"""

import argparse
import csv
from collections import Counter
from typing import Dict, List

# The first photograph was made in 1826. A "photo" dated earlier is either a
# painting/engraving or an error, so by default we drop those from the count.
MIN_PHOTO_YEAR = 1826


def midpoint(row: Dict[str, str]) -> float:
    """Decimal year at the middle of a row's date interval (same as DateEstimate.midpoint)."""
    return (int(row["date_start"]) + int(row["date_end"]) + 1) / 2.0


def summarize(rows: List[Dict[str, str]], min_conf: float = 0.0,
              min_year: int = MIN_PHOTO_YEAR) -> dict:
    """Compute all the audit numbers from CSV rows.

    Kept separate from main() and free of printing, so it can be unit-tested
    with a handful of fake rows.
    """
    total = len(rows)

    # Narrow down in stages so we can report how many fell out at each one:
    #   dated      -> has any date estimate at all
    #   confident  -> dated AND confidence >= min_conf
    #   usable     -> confident AND not earlier than the first photograph
    dated = [r for r in rows if r["date_start"] != ""]
    confident = [r for r in dated if float(r["date_conf"]) >= min_conf]
    usable = [r for r in confident if midpoint(r) >= min_year]
    years = [midpoint(r) for r in usable]

    def count(pred):
        return sum(1 for y in years if pred(y))

    unique = len({r["title"] for r in rows})
    return {
        "total": total,
        "dated": len(dated),
        "undated": total - len(dated),
        "usable": len(usable),
        "low_conf_dropped": len(dated) - len(confident),
        "pre_photo_dropped": len(confident) - len(usable),
        "unique_titles": unique,
        "duplicate_rows": total - unique,
        "sources": Counter(r["date_source"] for r in dated),
        # Bucket each year into its decade (1863 -> 1860) for the histogram.
        "decades": Counter(int(y // 10 * 10) for y in years),
        "pre_1900": count(lambda y: y < 1900),
        "pre_1950": count(lambda y: y < 1950),
        "pre_1970": count(lambda y: y < 1970),
        "post_2000": count(lambda y: y >= 2000),
        "oldest": sorted(usable, key=midpoint)[:15],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit date coverage of a scene.")
    ap.add_argument("csv_path", help="the image_dates.csv written by commons_dates.py")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="ignore date estimates below this confidence (0.5 is a sensible start)")
    ap.add_argument("--min-year", type=int, default=MIN_PHOTO_YEAR,
                    help="ignore dates before this year (default 1826, the first photograph)")
    args = ap.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    s = summarize(rows, args.min_conf, args.min_year)

    print(f"images: {s['total']}")
    print(f"  with a date estimate: {s['dated']}   undated: {s['undated']}")
    if args.min_conf:
        print(f"  dropped below confidence {args.min_conf}: {s['low_conf_dropped']}")
    print(f"  dropped as pre-photography (< {args.min_year}): {s['pre_photo_dropped']}")
    print(f"  usable: {s['usable']}")
    print(f"unique Commons titles: {s['unique_titles']}  (duplicate rows: {s['duplicate_rows']})")
    print(f"date sources: {dict(s['sources'])}")

    # Text bar chart: the longest bar is scaled to 60 characters, and any
    # non-empty decade always gets at least one '#' so tiny counts stay visible.
    print("\nimages per decade:")
    biggest = max(s["decades"].values()) if s["decades"] else 1
    for d in sorted(s["decades"]):
        n = s["decades"][d]
        print(f"  {d}s  {n:>5}  {'#' * min(60, max(1, n * 60 // biggest))}")

    print(f"\nbefore 1900: {s['pre_1900']}   before 1950: {s['pre_1950']}   "
          f"before 1970: {s['pre_1970']}   2000 or later: {s['post_2000']}")
    ok = s["pre_1970"] >= 10 and s["post_2000"] >= 10
    print("proposal threshold (10+ pre-1970 and 10+ modern):", "MET" if ok else "NOT MET")

    print("\noldest 15 (spot-check these against the Commons pages!):")
    for r in s["oldest"]:
        print(f"  {r['date_start']}-{r['date_end']}  conf {r['date_conf']}  "
              f"raw={r['date_raw']!r}  {r['title']}")


if __name__ == "__main__":
    main()
