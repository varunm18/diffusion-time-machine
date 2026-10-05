"""Date and type every image of a scene: metadata table in, dates table out."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from timemachine.config import DataPaths
from timemachine.dating.interval import DateCandidate, Source
from timemachine.dating.medium import PHOTOGRAPHIC, Medium, MediumResult, classify_medium
from timemachine.dating.resolve import DateResult, resolve
from timemachine.dating.sources.categories import parse_categories
from timemachine.dating.sources.exif import parse_exif
from timemachine.dating.sources.qs import parse_qs
from timemachine.dating.sources.text import parse_date_text
from timemachine.dating.sources.title import FIRST_PHOTO_YEAR, parse_title
from timemachine.megascenes.metadata import ImageMetadata, frame_to_records

SUBCATEGORY_CONFIDENCE_SCALE = 0.8  # a parent folder is weaker evidence than a direct category
MIN_CREATOR_AGE = 10               # years; works made before this age are implausible


def gather_candidates(meta: ImageMetadata, subcats: Iterable[str] = ()) -> list[DateCandidate]:
    """Run every source parser on one image."""
    candidates: list[DateCandidate | None] = []

    # The page date: prefer its hidden QS form; the visible text is the same field.
    # When Commons filled DateTimeOriginal from EXIF (no template date), the EXIF
    # parser below already covers it.
    qs = parse_qs(meta.date_qs)
    candidates.append(qs)
    if qs is None and meta.date_source == "commons-desc-page":
        candidates.append(parse_date_text(meta.date_text))

    candidates.append(parse_exif(meta.exif))
    candidates.extend(parse_categories(meta.categories))

    direct = {c.replace("_", " ") for c in meta.categories}
    extra = [s for s in subcats if s.replace("_", " ") not in direct]
    candidates.extend(parse_categories(extra, Source.SUBCATEGORY, SUBCATEGORY_CONFIDENCE_SCALE))
    candidates.append(parse_title(meta.title))
    return [c for c in candidates if c is not None]


def date_image(meta: ImageMetadata, subcats: Iterable[str] = ()) -> tuple[DateResult, MediumResult]:
    """Best date and medium for one image.

    Hard limits passed to the resolver: photos cannot predate 1826; a work cannot
    be made before its creator was ~10 or after they died (lifespan from the
    Commons Creator template, when present).
    """
    medium = classify_medium(meta)
    artwork = medium.medium not in (*PHOTOGRAPHIC, Medium.UNKNOWN)
    floor = FIRST_PHOTO_YEAR if medium.medium in PHOTOGRAPHIC else None
    ceiling = None
    if meta.artist_lifespan:
        born, died = meta.artist_lifespan
        floor = max(floor or born + MIN_CREATOR_AGE, born + MIN_CREATOR_AGE)
        ceiling = died + 1.0
    result = resolve(gather_candidates(meta, subcats), meta.upload_time,
                     historical_hint=medium.historical_hint or artwork,
                     floor_year=floor, ceiling_year=ceiling, artwork=artwork)
    return result, medium


def _row(meta: ImageMetadata, result: DateResult, medium: MediumResult) -> dict:
    iv, dig = result.interval, result.digitized
    return {
        "file_key": meta.file_key,
        "date_lo": iv.lo if iv else None,
        "date_hi": iv.hi if iv else None,
        "date_mid": iv.mid if iv else None,
        "date_width": iv.width if iv else None,
        "date_confidence": result.confidence,
        "date_source": result.source,
        "date_precision": result.precision,
        "date_flags": result.flags,
        "digitized_year": dig.lo if dig else None,
        "n_candidates": len(result.candidates),
        "candidates": "; ".join(f"{c.source}:{c.interval}@{c.confidence}" for c in result.candidates),
        "medium": medium.medium,
        "medium_evidence": medium.evidence,
        "color_hint": medium.color,
        "historical_hint": medium.historical_hint,
    }


def date_metadata_table(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Dates table for a metadata table (one row per unique file)."""
    records = frame_to_records(meta_df)
    subcats = meta_df["subcats"] if "subcats" in meta_df else [()] * len(records)
    rows = [_row(m, *date_image(m, s)) for m, s in zip(records, subcats)]
    out = pd.DataFrame(rows)
    if "scene_id" in meta_df:
        out.insert(0, "scene_id", meta_df["scene_id"].to_numpy())
    return out


def date_scene(paths: DataPaths, scene_id: int, overwrite: bool = False) -> pd.DataFrame:
    """Date a harvested scene, cached at ``paths.dates_table(scene_id)``."""
    out = paths.dates_table(scene_id)
    if out.exists() and not overwrite:
        return pd.read_parquet(out)
    table = date_metadata_table(pd.read_parquet(paths.image_metadata_table(scene_id)))
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    return table
