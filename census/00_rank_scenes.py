"""Step 00: rank MegaScenes scenes as candidates for the date census.

Uses only the local image index: no reconstructions, no API calls. Prints two
rankings:

  size        scenes with the most downloaded images
  historical  scenes with the most images filed in Wikimedia categories whose
              names suggest old photos ("... in the 1920s", "History of ...",
              "photochrom"), counted from the `subcat` column

Neither ranking says whether the old photos registered in a reconstruction;
steps 01-03 answer that. Use --out to write the historical top list as the
scene list for step 01.

    python 00_rank_scenes.py --top 50 --out candidates.txt
"""
import argparse

import pandas as pd

from census_lib import MEGASCENES

# Years 1800-1969 as standalone numbers (also catches "1920s"), plus keywords.
HISTORICAL = (r"(?<!\d)(?:18\d\d|19[0-6]\d)(?!\d)"
              r"|histor|old[ _]photograph|vintage|photochrom")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=50, help="rows to print (default 50)")
    ap.add_argument("--min-images", type=int, default=500,
                    help="ignore scenes with fewer downloaded images (default 500)")
    ap.add_argument("--out", help="write the top historical scene names here, one per line")
    args = ap.parse_args()

    path = MEGASCENES / "metadata" / "images_index.parquet"
    if not path.exists():
        raise SystemExit(f"Missing {path}. See README: 'Download the metadata'.")
    idx = pd.read_parquet(path, columns=["cat", "subcat", "image_name"])
    idx = idx[idx["cat"].str.len() > 1]  # drops parsing artifacts like the category "S"

    # A scene whose own name contains a year would count every image as
    # historical, so only subcategories count, not the scene category itself.
    is_hist = (idx["subcat"].str.contains(HISTORICAL, case=False, regex=True, na=False)
               & (idx["subcat"] != idx["cat"]))

    table = pd.DataFrame({
        "images": idx.groupby("cat")["image_name"].nunique(),
        "historical": idx[is_hist].groupby("cat")["image_name"].nunique(),
    }).fillna(0).astype(int)
    table = table[table["images"] >= args.min_images]
    table["hist_share"] = (table["historical"] / table["images"]).round(3)

    print(f"== Largest scenes (>= {args.min_images} images)")
    print(table.sort_values("images", ascending=False).head(args.top).to_string(), "\n")

    hist = table[table["historical"] > 0].sort_values("historical", ascending=False)
    print(f"== Most images in historical subcategories ({len(hist)} scenes have any)")
    print(hist.head(args.top).to_string())

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Top scenes by images in historical subcategories (00_rank_scenes.py)\n")
            f.write("\n".join(hist.head(args.top).index) + "\n")
        print(f"\nwrote {min(args.top, len(hist))} scene names to {args.out}")


if __name__ == "__main__":
    main()
