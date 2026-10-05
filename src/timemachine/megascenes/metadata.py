"""Per-image Wikimedia Commons metadata shipped by MegaScenes.

Every subcategory folder of a scene has a ``raw_metadata.json``: the raw output
of the Commons API (``prop=imageinfo`` with ``extmetadata`` and
``commonmetadata``) for each image, captured during the MegaScenes crawl
(June 2023 - June 2024). This module flattens that into compact
:class:`ImageMetadata` records and harvests them for whole scenes.

File structure::

    {"elements": [{"<pageid>": {"pageid", "title", "imageinfo": [{
        "timestamp",            # upload time of the current file version
        "width", "height", "url", "descriptionurl", ...,
        "extmetadata": {"DateTimeOriginal": {"value", "source"}, "Categories", ...},
        "commonmetadata": [{"name": "DateTimeOriginal", "value": "2014:07:07 11:15:00"}, ...]
    }]}}, ...]}
"""
from __future__ import annotations

import gzip
import html
import json
import re
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields

import pandas as pd

from timemachine.config import DataPaths, safe_filename
from timemachine.megascenes.index import file_key
from timemachine.megascenes.s3 import MegaScenesBucket, raw_metadata_key

_HIDDEN_DIV = re.compile(r"<div[^>]*display:\s*none[^>]*>.*?</div>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_QS = re.compile(r"date QS:([^<]*)")
_LIFESPAN = re.compile(r'title="[^"]*\((?:c\.\s*)?(\d{4})\s*[-–]\s*(?:c\.\s*)?(\d{4})\)"')
_EXIF_FIELDS = ("DateTimeOriginal", "DateTimeDigitized", "DateTime", "Make", "Model", "Software")


@dataclass
class ImageMetadata:
    """The fields of one Commons file that matter for dating and typing it.

    Text fields are plain text (HTML stripped). ``date_text`` is the visible
    date of the file description page and ``date_qs`` the hidden
    machine-readable date Commons renders for date templates (may be None).
    """

    pageid: int
    title: str                      # Commons title without the "File:" prefix
    file_key: str                   # see megascenes.index.file_key
    upload_time: str                # ISO timestamp of the current file version
    width: int
    height: int
    date_text: str = ""             # extmetadata DateTimeOriginal, visible text
    date_qs: str | None = None      # hidden "date QS:" string, if any
    date_source: str = ""           # "commons-desc-page", "file-metadata", ...
    categories: list[str] = field(default_factory=list)
    object_name: str = ""
    description: str = ""
    artist: str = ""
    artist_lifespan: tuple[int, int] | None = None  # from the Creator tooltip
    credit: str = ""
    license: str = ""
    exif: dict[str, str] = field(default_factory=dict)  # subset of _EXIF_FIELDS
    url: str = ""


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
def strip_html(value: object) -> str:
    """Plain text from an extmetadata value (drops hidden divs and tags).

    Multilingual values (``{"en": ..., "de": ..., "_type": "lang"}``) resolve
    to English when available, else the first language.
    """
    if isinstance(value, dict):
        langs = {k: v for k, v in value.items() if not k.startswith("_")}
        value = langs.get("en", next(iter(langs.values()), ""))
    text = _TAG.sub(" ", _HIDDEN_DIV.sub(" ", str(value)))
    return " ".join(html.unescape(text).split())


def iter_pages(raw: dict) -> Iterator[dict]:
    """Yield the per-file page dicts of a parsed ``raw_metadata.json``."""
    for element in raw.get("elements", []):
        yield from element.values()


def parse_page(page: dict) -> ImageMetadata:
    """Flatten one page dict into an :class:`ImageMetadata`."""
    info = page["imageinfo"][0]
    ext = info.get("extmetadata", {})

    def ext_text(key: str) -> str:
        return strip_html(ext.get(key, {}).get("value", ""))

    date_raw = str(ext.get("DateTimeOriginal", {}).get("value", ""))
    qs = _QS.search(html.unescape(date_raw))
    artist_raw = str(ext.get("Artist", {}).get("value", ""))
    lifespan = _LIFESPAN.search(artist_raw)
    exif = {
        item["name"]: str(item["value"])
        for item in info.get("commonmetadata", [])
        if item.get("name") in _EXIF_FIELDS and isinstance(item.get("value"), (str, int, float))
    }
    title = page["title"].removeprefix("File:")
    categories = ext.get("Categories", {}).get("value", "")
    return ImageMetadata(
        pageid=int(page["pageid"]),
        title=title,
        file_key=file_key(title),
        upload_time=info.get("timestamp", ""),
        width=int(info.get("width", 0)),
        height=int(info.get("height", 0)),
        date_text=strip_html(date_raw),
        date_qs=qs.group(1).strip() if qs else None,
        date_source=str(ext.get("DateTimeOriginal", {}).get("source", "")),
        categories=[c for c in str(categories).split("|") if c],
        object_name=ext_text("ObjectName"),
        description=ext_text("ImageDescription")[:2000],
        artist=ext_text("Artist"),
        artist_lifespan=(int(lifespan.group(1)), int(lifespan.group(2))) if lifespan else None,
        credit=ext_text("Credit")[:500],
        license=ext_text("LicenseShortName"),
        exif=exif,
        url=info.get("url", ""),
    )


def parse_raw_metadata(raw: dict) -> list[ImageMetadata]:
    """All records of one ``raw_metadata.json`` (pages without imageinfo are skipped)."""
    return [parse_page(p) for p in iter_pages(raw) if p.get("imageinfo")]


def records_to_frame(records: Iterable[ImageMetadata]) -> pd.DataFrame:
    """Records as a DataFrame (exif flattened to ``exif_<Field>`` columns)."""
    rows = []
    for rec in records:
        row = asdict(rec)
        exif = row.pop("exif")
        for name in _EXIF_FIELDS:
            row[f"exif_{name}"] = exif.get(name)
        lifespan = row.pop("artist_lifespan")
        row["artist_born"], row["artist_died"] = lifespan if lifespan else (None, None)
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("artist_born", "artist_died"):
        if col in df:
            df[col] = df[col].astype("Int64")
    return df


def frame_to_records(df: pd.DataFrame) -> list[ImageMetadata]:
    """Inverse of :func:`records_to_frame` (used when reading cached tables).

    Extra columns (e.g. ``scene_id``, ``subcats``) are ignored.
    """
    names = {f.name for f in fields(ImageMetadata)} - {"exif", "artist_lifespan"}
    records = []
    for row in df.to_dict("records"):
        exif = {n: str(row[f"exif_{n}"]) for n in _EXIF_FIELDS
                if f"exif_{n}" in row and pd.notna(row[f"exif_{n}"])}
        born, died = row.get("artist_born"), row.get("artist_died")
        lifespan = (int(born), int(died)) if pd.notna(born) and pd.notna(died) else None
        kwargs = {k: (None if v is pd.NA else v) for k, v in row.items() if k in names}
        cats = kwargs.get("categories")
        kwargs["categories"] = [] if cats is None else list(cats)
        if isinstance(kwargs.get("date_qs"), float):  # NaN from an all-null column
            kwargs["date_qs"] = None
        records.append(ImageMetadata(**kwargs, exif=exif, artist_lifespan=lifespan))
    return records


# ----------------------------------------------------------------------------
# Harvesting whole scenes
# ----------------------------------------------------------------------------
def fetch_raw_metadata(bucket: MegaScenesBucket, paths: DataPaths, scene_id: int, subcat: str) -> dict:
    """``raw_metadata.json`` of one subcategory, cached gzipped under the data root."""
    cache = paths.raw_metadata_dir(scene_id) / f"{safe_filename(subcat)}.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return json.load(f)
    data = bucket.get(raw_metadata_key(scene_id, subcat))
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".part")
    with gzip.open(tmp, "wb") as f:
        f.write(data)
    tmp.replace(cache)
    return json.loads(data)


def harvest_scene(
    bucket: MegaScenesBucket, paths: DataPaths, scene_id: int, subcats: Iterable[str],
    workers: int = 16, overwrite: bool = False,
) -> pd.DataFrame:
    """Metadata of every file in a scene, one row per unique Commons file.

    Downloads each subcategory's ``raw_metadata.json`` in parallel (cached), parses
    it, and deduplicates files that appear in several subcategories. The
    ``subcats`` column lists every subcategory a file was found in. Missing
    metadata files are skipped and reported in the ``attrs["missing_subcats"]``
    of the returned frame. Cached at ``paths.image_metadata_table(scene_id)``.
    """
    out = paths.image_metadata_table(scene_id)
    if out.exists() and not overwrite:
        return pd.read_parquet(out)

    subcats = sorted(set(subcats))

    def load(subcat: str) -> tuple[str, list[ImageMetadata] | None]:
        try:
            return subcat, parse_raw_metadata(fetch_raw_metadata(bucket, paths, scene_id, subcat))
        except FileNotFoundError:
            return subcat, None

    found: dict[str, ImageMetadata] = {}
    where: dict[str, list[str]] = {}
    missing = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for subcat, records in pool.map(load, subcats):
            if records is None:
                missing.append(subcat)
                continue
            for rec in records:
                found.setdefault(rec.file_key, rec)
                where.setdefault(rec.file_key, []).append(subcat)

    df = records_to_frame(found.values())
    if len(df):
        df.insert(0, "scene_id", scene_id)
        df["subcats"] = df["file_key"].map(where)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.attrs["missing_subcats"] = missing
    return df
