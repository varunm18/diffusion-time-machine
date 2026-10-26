"""Combine date candidates into one best estimate per image.

Steps (each leaves a flag on the result so decisions are auditable):

1. **Upload bound.** Nothing can be made after it was uploaded: clip every
   candidate to the upload time, drop those entirely after it.
2. **Upload date typed in as the date.** A day-precise description-page date
   within 2 days of the upload, on an image with historical evidence, is the
   upload day, not the capture day -> drop it (``date_is_upload_date``).
3. **Scans.** An EXIF date from a scanner/reproduction device, or one more than
   5 years after strong historical evidence, is a digitization date -> move it
   to ``digitized`` (``exif_is_digitization``). Placeholder EXIF dates are dropped.
4. **Pick and refine.** Take the most confident candidate (ties: narrowest),
   then narrow it with overlapping candidates of at least medium-low confidence.
   Born-digital EXIF agreeing with the page date gets very high confidence.
5. **Conflicts.** Candidates that do not overlap the result are flagged
   (``conflict:`` / ``weak_conflict:``); a strong conflict lowers confidence and
   asks for review.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from timemachine.dating.interval import (
    Confidence, DateCandidate, DateInterval, Precision, Source, decimal_year,
)

UPLOAD_SLACK_DAYS = 2      # page date this close to the upload = probably the upload date
SCAN_GAP_YEARS = 5         # EXIF this much later than historical evidence = digitization
NARROW_MIN_CONFIDENCE = 0.45
CONFLICT_MIN_CONFIDENCE = 0.5
AGREEMENT_SLACK_YEARS = 0.1  # ~5 weeks: page date vs EXIF often differ by days (time zones, memory)
MODERN_YEAR = 1990         # evidence older than this counts as "historical"


@dataclass
class DateResult:
    """Best date estimate for one image, with the evidence behind it."""

    interval: DateInterval | None
    confidence: float = 0.0
    source: str | None = None
    precision: str | None = None
    flags: list[str] = field(default_factory=list)
    candidates: list[DateCandidate] = field(default_factory=list)  # after filtering
    digitized: DateInterval | None = None


def _upload_year(upload_time: str) -> float | None:
    try:
        when = dt.datetime.fromisoformat(upload_time.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return decimal_year(when.date())


def _agree(a: DateInterval, b: DateInterval, slack: float = AGREEMENT_SLACK_YEARS) -> bool:
    """Do two intervals overlap once each is widened by ``slack`` years on both sides?"""
    return a.lo - slack < b.hi + slack and b.lo - slack < a.hi + slack


def _is_independent_old(c: DateCandidate, min_confidence: float) -> bool:
    """An old date from a source other than EXIF / the page date, with real bounds."""
    return (c.interval.hi <= MODERN_YEAR and c.confidence >= min_confidence
            and c.precision != Precision.BOUND
            and c.source not in (Source.EXIF, Source.DATE_TEXT, Source.QS))


def _is_historical(candidates: list[DateCandidate], hint: bool) -> bool:
    """Is there evidence, independent of the page date, that the image is old?"""
    return hint or any(_is_independent_old(c, NARROW_MIN_CONFIDENCE) for c in candidates)


def resolve(
    candidates: list[DateCandidate],
    upload_time: str = "",
    historical_hint: bool = False,
    floor_year: float | None = None,
    ceiling_year: float | None = None,
    artwork: bool = False,
) -> DateResult:
    """Best single date from ``candidates``.

    ``historical_hint`` comes from metadata (archive credit, PD-old license, old
    medium, ...). ``floor_year`` / ``ceiling_year`` clip every candidate (e.g. 1826
    for photos, the creator's death year). For ``artwork`` (painting, print, ...)
    EXIF can only be a reproduction date, never the creation date.
    """
    flags: list[str] = []
    kept: list[DateCandidate] = []
    digitized = None

    # 1. upload bound (+ medium floor)
    upload = _upload_year(upload_time)
    ceilings = [x for x in (ceiling_year, None if upload is None else upload + 1 / 365) if x is not None]
    for c in candidates:
        interval = c.interval.clip(lo=floor_year, hi=min(ceilings) if ceilings else None)
        if interval is None:
            flags.append(f"impossible:{c.source}")
            continue
        kept.append(c if interval == c.interval else DateCandidate(
            interval, c.source, c.precision, c.confidence, c.evidence, c.flags + ("clipped",)))

    historical = _is_historical(kept, historical_hint)

    # 2. upload date typed in as the capture date (unless a real camera EXIF confirms it)
    camera_exif = [c for c in kept if c.source == Source.EXIF and not c.flags]
    if upload is not None and historical:
        slack = UPLOAD_SLACK_DAYS / 365
        page_dates = [c for c in kept if c.source in (Source.QS, Source.DATE_TEXT)
                      and c.precision == Precision.DAY and abs(c.interval.lo - upload) <= slack
                      and not any(_agree(e.interval, c.interval) for e in camera_exif)]
        if page_dates:
            flags.append("date_is_upload_date")
            # "Photographs taken on <date>" categories are generated from the same field.
            kept = [c for c in kept if c not in page_dates and not (
                c.source == Source.CATEGORY and c.precision == Precision.DAY
                and abs(c.interval.lo - upload) <= slack)]
    explicit_upload = [c for c in kept if "upload_date" in c.flags]
    if explicit_upload and len(explicit_upload) < len(kept):
        flags.append("dropped_upload_date_text")
        kept = [c for c in kept if c not in explicit_upload]

    # 3. scans / reproductions / placeholder EXIF
    # Evidence the image is older than its EXIF: confident dated evidence, or an
    # explicit "before X" on the description page itself.
    strong_old = [c.interval.hi for c in kept if c.source != Source.EXIF and (
        (c.confidence >= Confidence.MEDIUM and c.precision != Precision.BOUND)
        or (c.source in (Source.QS, Source.DATE_TEXT) and "open_start" in c.flags))]
    for c in [c for c in kept if c.source == Source.EXIF]:
        if "placeholder" in c.flags:
            kept.remove(c)
            flags.append("exif_placeholder")
        elif artwork or "scanner" in c.flags or (
                strong_old and min(strong_old) + SCAN_GAP_YEARS <= c.interval.lo):
            kept.remove(c)
            digitized = c.interval
            flags.append("exif_is_digitization")

    if not kept:
        return DateResult(None, flags=flags, digitized=digitized)

    # 4. pick the best candidate, then narrow with consistent evidence
    ranked = sorted(kept, key=lambda c: (-c.confidence, c.interval.width))
    best = ranked[0]
    interval, confidence = best.interval, best.confidence
    for other in ranked[1:]:
        # One-sided bounds are cheap to apply when consistent; other evidence must be decent.
        weak = other.confidence < NARROW_MIN_CONFIDENCE and other.precision != Precision.BOUND
        if weak or not other.interval.overlaps(interval):
            continue
        narrowed = interval.intersect(other.interval)
        if narrowed is not None and narrowed.width < interval.width:
            interval = narrowed
            if other.precision != Precision.BOUND:
                confidence = min(confidence, (confidence + other.confidence) / 2 + 0.1)
            flags.append(f"narrowed_by:{other.source}")

    exif_ok = [c for c in kept if c.source == Source.EXIF and not c.flags]
    page = [c for c in kept if c.source in (Source.QS, Source.DATE_TEXT)]
    if exif_ok and page and any(_agree(e.interval, p.interval) for e in exif_ok for p in page):
        confidence = max(confidence, Confidence.VERY_HIGH)
        flags.append("exif_agrees")

    # 5. conflicts
    for other in ranked[1:]:
        if _agree(other.interval, interval):
            continue
        if other.confidence >= CONFLICT_MIN_CONFIDENCE:
            flags.append(f"conflict:{other.source}")
            if other.confidence >= best.confidence - 0.15:
                confidence *= 0.6
                flags.append("needs_review")
        elif other.confidence >= Confidence.LOW:
            flags.append(f"weak_conflict:{other.source}")  # worth a look, no penalty

    return DateResult(interval, round(confidence, 3), best.source, best.precision,
                      sorted(set(flags)), kept, digitized)
