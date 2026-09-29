"""Step 02: look up each registered image's capture date on Wikimedia Commons.

MegaScenes' metadata has no dates, but every image is a Wikimedia Commons
file, so we ask the Commons API for its metadata. Two fields are kept:

  date_raw    DateTimeOriginal: the "date" on the file page (capture date)
  upload_raw  DateTime: upload/modification time, kept for spot checks

Raw strings are stored as-is; 03_census_table.py parses the year. Only images
in models with at least 50 images are looked up. Resumable: results are
appended to data/dates.csv as each batch finishes, and files already there are
skipped.

Wikimedia asks automated clients to identify themselves. Pass a contact, which
is sent only in the User-Agent header:

    python 02_fetch_dates.py --contact you@umd.edu
    (or set WIKIMEDIA_CONTACT=you@umd.edu)
"""
import argparse
import csv
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from census_lib import DATES, load_registered

API = "https://commons.wikimedia.org/w/api.php"
BATCH = 50  # max titles per request for clients without a bot account
RETRYABLE = {429, 500, 502, 503, 504}


def ssl_context() -> ssl.SSLContext:
    """Use certifi's certificates when available (fixes python.org macOS builds)."""
    if os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context()
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def post(params: dict, headers: dict, ctx: ssl.SSLContext, tries: int = 5) -> dict:
    """POST to the API (50 long titles can exceed URL limits), retrying transient errors."""
    data = urllib.parse.urlencode(params).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(API, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE or attempt == tries - 1:
                raise
            try:
                wait = int(e.headers.get("Retry-After") or 0)
            except ValueError:
                wait = 0
            wait, reason = wait or 5 * 2 ** attempt, f"HTTP {e.code}"
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLError):
                raise SystemExit(
                    f"SSL error: {e.reason}\nRun `python -m pip install certifi` in this "
                    "environment, or on macOS run '/Applications/Python 3.x/Install "
                    "Certificates.command'.")
            if attempt == tries - 1:
                raise
            wait, reason = 5 * 2 ** attempt, str(e.reason)
        print(f"  {reason}; retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)


def parse_pages(query: dict) -> dict[str, tuple[str, str]]:
    """{file name: (DateTimeOriginal, DateTime)} from an API `query` block."""
    # The API rewrites titles (underscores -> spaces); map them back to what we sent.
    back = {n["to"]: n["from"] for n in query.get("normalized", [])}
    out = {}
    for page in query.get("pages", {}).values():
        title = page.get("title", "")
        title = back.get(title, title).removeprefix("File:")
        meta = (page.get("imageinfo") or [{}])[0].get("extmetadata", {})
        out[title] = (meta.get("DateTimeOriginal", {}).get("value", ""),
                      meta.get("DateTime", {}).get("value", ""))
    return out


def fetch_batch(names: list[str], headers: dict, ctx) -> dict[str, tuple[str, str]]:
    params = {"action": "query", "format": "json", "prop": "imageinfo",
              "iiprop": "extmetadata",
              "iiextmetadatafilter": "DateTimeOriginal|DateTime",
              "titles": "|".join("File:" + n for n in names)}
    out = {}
    while True:
        resp = post(params, headers, ctx)
        if "error" in resp:
            raise RuntimeError(f"API error: {resp['error']}")
        for name, value in parse_pages(resp.get("query", {})).items():
            if name not in out or any(value):  # don't let a continuation blank a result
                out[name] = value
        if "continue" not in resp:
            return out
        params = {**params, **resp["continue"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contact", default=os.environ.get("WIKIMEDIA_CONTACT"),
                    help="email for the User-Agent (or set WIKIMEDIA_CONTACT)")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between requests (default 1)")
    args = ap.parse_args()
    if not args.contact:
        ap.error("pass --contact you@umd.edu (or set WIKIMEDIA_CONTACT)")
    headers = {"User-Agent": "cmsc473-megascenes-census/1.0 "
                             f"(UMD CMSC473 student project; {args.contact})"}
    ctx = ssl_context()

    names = load_registered()["image_name"].unique()
    done = set()
    if DATES.exists():
        done = set(pd.read_csv(DATES, dtype=str, keep_default_na=False)["image_name"])
    todo = [n for n in names if n not in done]
    print(f"{len(names)} registered images in usable models; {len(todo)} to look up")

    write_header = not DATES.exists()
    DATES.parent.mkdir(parents=True, exist_ok=True)
    with open(DATES, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["image_name", "date_raw", "upload_raw"])
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            result = fetch_batch(batch, headers, ctx)
            for n in batch:
                w.writerow([n, *result.get(n, ("", ""))])
            f.flush()
            if (i // BATCH) % 20 == 0:
                print(f"{i}/{len(todo)}")
            time.sleep(args.delay)
    print(f"done; {DATES} is up to date")


if __name__ == "__main__":
    main()
