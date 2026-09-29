"""Step 03: count registered images per era, per COLMAP model.

Prints data-quality checks, then the census table: for every model with at
least 50 unique images, how many images fall in each era. Also lists the
"usable" models, where old and modern photos share one reconstruction (and so
share camera poses), which is what date-conditioned training needs. The full
table is saved to data/census_table.csv.

Hand-check a model's old images (are they photographs, is the year right?):

    python 03_census_table.py --inspect Dam_Square:0 --inspect Cathédrale_Notre-Dame_de_Paris:0
"""
import argparse

import numpy as np
import pandas as pd

from census_lib import (CENSUS_TABLE, DATES, MIN_MODEL_IMAGES, clean_date_text,
                        commons_url, load_registered, parse_date)

ERAS = ["pre-1970", "1970-1999", "2000+"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=30, help="table rows to print (default 30)")
    ap.add_argument("--min-old", type=int, default=20,
                    help="pre-1970 images for a model to count as usable (default 20)")
    ap.add_argument("--min-new", type=int, default=100,
                    help="2000+ images for a model to count as usable (default 100)")
    ap.add_argument("--inspect", action="append", default=[], metavar="SCENE:MODEL",
                    help="print a sample of a model's pre-1970 images with links")
    ap.add_argument("--n", type=int, default=15, help="images per --inspect (default 15)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = load_registered(min_images=0, dedupe=False)
    reg = load_registered()
    if not DATES.exists():
        raise SystemExit(f"Missing {DATES}. Run 02_fetch_dates.py first.")
    dates = pd.read_csv(DATES, dtype=str, keep_default_na=False).drop_duplicates("image_name")
    parsed = pd.DataFrame(dates["date_raw"].map(parse_date).tolist(),
                          columns=["year", "year_max", "date_flag"], index=dates.index)
    dates = dates.join(parsed)
    reg = reg.merge(dates[["image_name", "date_raw", "year", "year_max", "date_flag"]],
                    on="image_name", how="left")
    reg["year"] = pd.to_numeric(reg["year"], errors="coerce")
    reg["date_flag"] = reg["date_flag"].fillna("not looked up")

    # ---------------------------------------------------------------- quality
    within_dupes = len(raw) - len(raw.drop_duplicates(["scene_id", "model", "image_name"]))
    per_file = reg.groupby("image_name")["scene_id"].nunique()
    print("== Data quality")
    print(f"registered rows downloaded:            {len(raw)}")
    print(f"  same file twice in one model:        {within_dupes} (removed)")
    print(f"unique images in models >= {MIN_MODEL_IMAGES} images: {len(reg)} "
          f"in {reg.groupby(['scene_id', 'model']).ngroups} models, "
          f"{reg['scene_id'].nunique()} scenes")
    print(f"  files registered in >1 scene:        {(per_file > 1).sum()}")
    flags = reg["date_flag"].replace("", "ok").value_counts()
    print("date quality (share of images):")
    for flag, count in flags.items():
        print(f"  {flag:<15} {count:>7}  {count / len(reg):6.1%}")

    # ---------------------------------------------------------------- census
    y = reg["year"]
    reg["era"] = np.select([y < 1970, y < 2000, y.notna()], ERAS, default="undated")
    table = (pd.crosstab([reg["scene"], reg["model"]], reg["era"])
             .reindex(columns=ERAS + ["undated"], fill_value=0))
    table = table.sort_values(ERAS, ascending=False)
    CENSUS_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CENSUS_TABLE)

    print(f"\n== Images per era, per model (top {args.top}; full table in {CENSUS_TABLE})")
    print(table.head(args.top).to_string())

    usable = table[(table["pre-1970"] >= args.min_old) & (table["2000+"] >= args.min_new)]
    print(f"\n== Usable models: >= {args.min_old} pre-1970 and >= {args.min_new} "
          f"2000+ images in the same model")
    print(f"{len(usable)} models across "
          f"{usable.index.get_level_values('scene').nunique()} scenes")
    if len(usable):
        print(usable.to_string())

    # ---------------------------------------------------------------- inspect
    for spec in args.inspect:
        scene, _, model = spec.rpartition(":")
        old = reg[(reg["scene"] == scene) & (reg["model"] == int(model)) & (reg["year"] < 1970)]
        print(f"\n== {scene} model {model}: {len(old)} pre-1970 images")
        if old.empty:
            continue
        sample = old.sample(min(args.n, len(old)), random_state=args.seed).sort_values("year")
        for r in sample.itertuples():
            text = clean_date_text(r.date_raw)[:32]
            print(f"  {int(r.year)}  {r.date_flag or 'ok':<7} {text:<32}  {commons_url(r.image_name)}")


if __name__ == "__main__":
    main()
