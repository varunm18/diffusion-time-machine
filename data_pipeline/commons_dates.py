"""Look up capture dates for our images on Wikimedia Commons.

WHY THIS FILE EXISTS
--------------------
MegaScenes gives us images and camera poses, but NO dates. The dates live on
Wikimedia Commons, where every MegaScenes photo has a file page with a "Date"
field, categories and a description. This script fetches that information for
every registered image in a scene and turns it into a year estimate using
date_parser.py.

HOW TO RUN (from the repo root)
-------------------------------
    # Try 20 images first, to make sure everything works:
    python data_pipeline/commons_dates.py data/raw/taj/colmap/0 --contact you@example.com --limit 20

    # Then the whole scene (about 107 requests for the Taj Mahal):
    python data_pipeline/commons_dates.py data/raw/taj/colmap/0 --contact you@example.com

WHAT IT DOES, STEP BY STEP
--------------------------
 1. Reads the registered image names from the COLMAP model's images.bin.
 2. Converts each into a Commons title, e.g.
    "commons/Scene/0/pictures/Taj_Mahal_19.jpg"  ->  "File:Taj Mahal 19.jpg".
 3. Asks the Commons API for each file's date, categories and description,
    50 titles per request.
 4. Saves every raw answer to a cache file (JSON lines). If the script stops or
    you run it again, it only fetches what's still missing.
 5. Runs our date parser and writes data/processed/image_dates.csv, one row per
    image.

THINGS TO KNOW
--------------
* Wikimedia asks API users to identify themselves, so --contact (your email) is
  required and goes into the User-Agent header.
* The API's "DateTime" field is when the file was uploaded or last edited, NOT
  when the photo was taken. We save it for reference but never use it as a date.
* Underscores and spaces mean the same thing in Commons titles. COLMAP names use
  underscores, so we swap them for spaces.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Let this file be run directly (python data_pipeline/commons_dates.py ...) and
# still import our sibling modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.colmap_reader import read_images_binary  # noqa: E402
from data_pipeline.date_parser import estimate_image_date  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
BATCH = 50  # the API accepts at most 50 titles per request


# ----------------------------------------------------------------------------
# Pure helpers. None of these touch the network, so they're easy to unit-test.
# ----------------------------------------------------------------------------

def title_from_colmap_name(name: str) -> str:
    """'commons/Scene/0/pictures/Taj_Mahal_19.jpg' -> 'File:Taj Mahal 19.jpg'."""
    base = name.rsplit("/", 1)[-1]          # drop the folders, keep the filename
    return "File:" + base.replace("_", " ")  # Commons treats _ and space alike


def strip_html(text: Optional[str]) -> str:
    """Commons returns some fields as HTML; turn them into plain text.

    Example: '<time datetime="2016-11-28">28 November 2016</time>' becomes
    '28 November 2016'. We drop the tags, decode entities like &amp;, and
    collapse repeated whitespace.
    """
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def chunks(items: List[str], n: int) -> Iterable[List[str]]:
    """Yield successive slices of at most n items (to respect the 50/request cap)."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def parse_batch(response: dict, requested: List[str]) -> Dict[str, dict]:
    """Turn one API response into {requested_title: record}.

    The API doesn't always echo our titles back unchanged:
      * it "normalizes" them (e.g. capitalizes the first letter), and
      * it follows "redirects" (a renamed file points to its new name).
    So for each title we asked about, we follow those two alias lists to find
    the matching page. Files that don't exist get found=False.

    Each record has: found, date_raw (the Date field), upload_datetime (NOT the
    capture date), categories (a list), and description.
    """
    query = response.get("query", {})

    # Build a "this title is really that title" lookup from both alias lists.
    alias: Dict[str, str] = {}
    for item in query.get("normalized", []) + query.get("redirects", []):
        alias[item["from"]] = item["to"]

    pages = {p.get("title"): p for p in query.get("pages", [])}

    def resolve(title: str) -> str:
        # Follow aliases until we reach a title with no further alias. The
        # `seen` set guards against an (unlikely) alias loop.
        seen = set()
        while title in alias and title not in seen:
            seen.add(title)
            title = alias[title]
        return title

    out: Dict[str, dict] = {}
    for req in requested:
        page = pages.get(resolve(req))
        if page is None or page.get("missing"):
            out[req] = {"found": False, "date_raw": "", "upload_datetime": "",
                        "categories": [], "description": ""}
            continue

        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {}) or {}

        def val(key: str) -> str:
            return strip_html((meta.get(key) or {}).get("value", ""))

        # Categories arrive as one string separated by "|".
        cats = [c for c in val("Categories").split("|") if c]
        out[req] = {
            "found": True,
            "date_raw": val("DateTimeOriginal"),   # the page's "Date" field
            "upload_datetime": val("DateTime"),    # upload time: NOT a capture date
            "categories": cats,
            "description": val("ImageDescription"),
        }
    return out


# ----------------------------------------------------------------------------
# Network and command-line part.
# ----------------------------------------------------------------------------

def fetch_batch(titles: List[str], contact: str) -> dict:
    """Ask the Commons API about up to 50 files in a single request."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "imageinfo",
        "iiprop": "extmetadata",
        # Only ask for the four fields we use; the full metadata is large.
        "iiextmetadatafilter": "DateTimeOriginal|DateTime|ImageDescription|Categories",
        "titles": "|".join(titles),
        "redirects": "1",   # follow renamed files
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": f"diffusion-time-machine-capstone/0.1 (class project; {contact})"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_cache(path: Path) -> Dict[str, dict]:
    """Load previously fetched results (one JSON object per line), if any."""
    cache: Dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["title"]] = rec
    return cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir", help="COLMAP model folder containing images.bin")
    ap.add_argument("--contact", required=True, help="your email, sent in the User-Agent")
    ap.add_argument("--limit", type=int, default=0, help="only look up the first N images")
    ap.add_argument("--cache", default="data/raw/commons_cache.jsonl")
    ap.add_argument("--out", default="data/processed/image_dates.csv")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds to wait between requests (be polite to Wikimedia)")
    args = ap.parse_args()

    images = read_images_binary(Path(args.model_dir) / "images.bin")
    names = [im.name for im in images.values()]
    if args.limit:
        names = names[: args.limit]

    # The same file can be registered more than once, so ask for each title once.
    titles = sorted({title_from_colmap_name(n) for n in names})

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(cache_path)
    todo = [t for t in titles if t not in cache]
    print(f"{len(names)} images -> {len(titles)} unique titles; "
          f"{len(titles) - len(todo)} cached, {len(todo)} to fetch")

    # Fetch what's missing, appending to the cache after every batch so an
    # interrupted run loses almost nothing.
    with open(cache_path, "a", encoding="utf-8") as cf:
        for i, batch in enumerate(chunks(todo, BATCH), 1):
            resp = fetch_batch(batch, args.contact)
            for title, rec in parse_batch(resp, batch).items():
                rec["title"] = title
                cache[title] = rec
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cf.flush()
            print(f"  batch {i}/{(len(todo) + BATCH - 1) // BATCH}")
            time.sleep(args.sleep)

    # Run the date parser on everything and write the CSV.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_dated = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["colmap_name", "title", "found", "date_raw", "upload_datetime",
                    "date_start", "date_end", "date_conf", "date_source"])
        for name in names:
            title = title_from_colmap_name(name)
            # If a title somehow isn't cached, treat it as "not found".
            rec = cache.get(title, {"found": False, "date_raw": "", "upload_datetime": "",
                                    "categories": [], "description": ""})
            est = estimate_image_date(
                filename=name.rsplit("/", 1)[-1],
                categories=rec["categories"],
                description=rec["description"],
                date_field=rec["date_raw"],
            )
            n_dated += est is not None
            w.writerow([name, title, rec["found"], rec["date_raw"], rec["upload_datetime"],
                        est.start_year if est else "", est.end_year if est else "",
                        est.confidence if est else "", est.source if est else ""])
    print(f"\nwrote {out_path}: {len(names)} images, {n_dated} with a date estimate")


if __name__ == "__main__":
    main()
