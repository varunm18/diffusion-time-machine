"""Turn messy Wikimedia Commons metadata into a capture-year estimate.

WHY THIS FILE EXISTS
--------------------
Our model is conditioned on *when* each photo was taken, but MegaScenes has no
clean date column. The only dates we have are whatever people typed into
Wikimedia Commons: "28 November 2016, 10:23:25", "1849", "circa 1860",
"between 1850 and 1900", "negative 1859; print 1862", and so on. This file
reads that text and answers: "what year was this photo taken, and how sure are
we?"

WHAT IT RETURNS
---------------
Not a single year, but a `DateEstimate`:
    start_year, end_year   an inclusive interval ("the 1920s" = 1920..1929)
    confidence             0..1, how much we trust it
    source                 where the text came from (date_field, filename, ...)

Keeping the interval and confidence (instead of rounding to one number) lets
training sample a year inside the interval and lets us throw away shaky dates
later. See docs/interface_decisions.md, Section 2.

THE BIG LESSONS (each one is covered by a test)
-----------------------------------------------
1. The structured "Date" field of the Commons page is the most reliable
   source. Free-text descriptions can contain *unrelated* years (an artist's
   lifespan "1825 - 1917", a museum accession number "2013.55.12"), so they
   must never override an explicit Date field.
2. EXIF and "upload date" are not capture dates. Old scanned photos carry the
   scan date, which is decades too late. These get low confidence.
3. A clock time next to an early year ("1648-01-03 17:00:23") is a broken
   camera clock, not a 17th-century photo.
4. Museum uploads append machine-readable junk ("date QS:P,+1500-00-00...")
   whose numbers look like years but are Wikidata codes. We cut it off first.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import Iterable, Optional

# Years outside this window are rejected outright. 1500 is generous (it lets
# paintings and engravings through); 2100 just guards against junk numbers.
MIN_YEAR = 1500
MAX_YEAR = 2100

# Regex for "a standalone 4-digit year". The lookbehind/lookahead `(?<!\d)` and
# `(?!\d)` mean the year can't be glued to other digits, so "IMG_20190704123"
# does NOT yield 2019. Range 1500-2099.
_YEAR = r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)"

# Month names/abbreviations for patterns like "6 May 1889". Short forms come
# first; the regex engine backtracks to the long form when needed.
_MONTHS = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    "january|february|march|april|june|july|august|september|october|november|december"
)

# How precise the *wording* was, as a base confidence. A full date is nearly
# certain; a bare number in a filename could be anything.
_PRECISION_CONF = {
    "full_date": 0.95,
    "month_year": 0.9,
    "year_labeled": 0.8,   # e.g. "taken in 1920", "negative 1859"
    "range": 0.7,
    "circa": 0.6,
    "decade": 0.6,
    "century": 0.4,
    "bare_year": 0.5,      # a lone number: might be a house number, an ID, ...
}

# How much we trust the *place* the text came from. Final confidence =
# precision * source weight. Anything not listed here gets 0.8.
_SOURCE_WEIGHT = {
    "date_field": 1.0,        # the "Date" row on the Commons file page: best
    "structured_data": 1.0,
    "category": 0.95,
    "filename": 0.85,
    "description": 0.8,       # free text, can contain unrelated years
    "exif": 0.7,              # often the scan or camera-clock date
}


@dataclass(frozen=True)
class DateEstimate:
    """An uncertain capture date: an inclusive year interval plus a confidence."""

    start_year: int
    end_year: int
    confidence: float
    source: str  # date_field | filename | category | description | exif | structured_data

    def __post_init__(self) -> None:
        # Fail loudly on nonsense instead of letting a bad value flow downstream.
        if self.start_year > self.end_year:
            raise ValueError("start_year must be <= end_year")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")

    @property
    def midpoint(self) -> float:
        """Decimal year at the center of the interval (1920..1929 -> 1925.0)."""
        # The +1 is because end_year is inclusive: the interval really covers
        # the *end* of the end year, so 1920..1929 spans 1920.0 up to 1930.0.
        return (self.start_year + self.end_year + 1) / 2.0

    @property
    def width(self) -> int:
        """How many years wide the interval is (0 means an exact year)."""
        return self.end_year - self.start_year

    def sample_year(self, rng: Optional[random.Random] = None) -> float:
        """Pick a random decimal year inside the interval (used at training time).

        Instead of always training on the midpoint of "the 1920s", we sample a
        year in 1920.0..1930.0 each time, so the model learns the date is fuzzy.
        """
        rng = rng or random
        return rng.uniform(self.start_year, self.end_year + 1)


def _make(start: int, end: int, precision: str, source: str) -> Optional[DateEstimate]:
    """Build a DateEstimate, or None if the years are out of range or reversed."""
    if start < MIN_YEAR or end > MAX_YEAR or start > end:
        return None
    conf = _PRECISION_CONF[precision] * _SOURCE_WEIGHT.get(source, 0.8)
    return DateEstimate(start, end, round(conf, 3), source)


def parse_date_text(text: str, source: str) -> Optional[DateEstimate]:
    """Parse free text (a filename, category, description, or date string).

    Tries patterns from MOST specific to LEAST specific and returns the first
    hit, so "6 May 1889" is read as an exact date instead of falling through to
    the weak "any number that looks like a year" rule at the bottom.
    Returns None if nothing date-like is found.
    """
    if not text:
        return None

    # Lesson 4: museum uploads tack on "date QS:P,+1500-00-00T00:00:00Z/6,..."
    # Those numbers are Wikidata precision codes, not dates. Cut everything
    # from "date QS:" onward so only the human-readable part remains.
    text = re.split(r"\s*date\s+QS:", text, maxsplit=1, flags=re.IGNORECASE)[0]

    # Lowercase for matching, and treat underscores as spaces (filenames use _).
    t = text.lower().replace("_", " ")

    # --- Exact dates ---------------------------------------------------------
    # ISO-style: 1889-05-06, 1889:05:06 or 1889/05/06 (EXIF uses the colons).
    m = re.search(_YEAR + r"[-:/](0?[1-9]|1[0-2])[-:/](0?[1-9]|[12]\d|3[01])(?!\d)", t)
    if m:
        y = int(m.group(1))
        return _make(y, y, "full_date", source)

    # Written out: "6 May 1889" or "May 6, 1889".
    m = re.search(r"\b\d{1,2}\s+(?:%s)\.?,?\s+%s" % (_MONTHS, _YEAR), t) or re.search(
        r"\b(?:%s)\.?\s+\d{1,2},?\s+%s" % (_MONTHS, _YEAR), t
    )
    if m:
        # The match includes the day and month; we only need the year part.
        y = int(re.search(_YEAR, m.group(0)).group(1))
        return _make(y, y, "full_date", source)

    # Month and year only: "May 1889" (also catches "January-March 1864").
    m = re.search(r"\b(?:%s)\.?\s+%s" % (_MONTHS, _YEAR), t)
    if m:
        y = int(re.search(_YEAR, m.group(0)).group(1))
        return _make(y, y, "month_year", source)

    # --- Ranges --------------------------------------------------------------
    # "between 1850 and 1900" (very common in museum uploads).
    m = re.search(r"between\s+" + _YEAR + r"\s+and\s+" + _YEAR, t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b:
            return _make(a, b, "range", source)

    # "1920-1925", "1920 – 1925", "1920 to 1925".
    m = re.search(_YEAR + r"\s*(?:-|–|—|to)\s*" + _YEAR, t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b:
            return _make(a, b, "range", source)

    # --- Approximate dates ---------------------------------------------------
    # "c. 1890", "ca. 1890", "circa 1890": we only know it's roughly then,
    # so allow +/- 5 years.
    m = re.search(r"\b(?:circa|ca|c)\.?\s*" + _YEAR, t)
    if m:
        y = int(m.group(1))
        return _make(y - 5, y + 5, "circa", source)

    # Decades: "1920s", "1920's", optionally "early / mid / late 1920s".
    # early = first ~3 years, mid = middle, late = last ~3.
    m = re.search(r"\b(early|mid|late)?[\s-]*" + r"(1[5-9]\d0|20[0-2]0)'?s\b", t)
    if m:
        qualifier, d = m.group(1), int(m.group(2))
        if qualifier == "early":
            return _make(d, d + 3, "decade", source)
        if qualifier == "mid":
            return _make(d + 3, d + 6, "decade", source)
        if qualifier == "late":
            return _make(d + 6, d + 9, "decade", source)
        return _make(d, d + 9, "decade", source)

    # Centuries: "19th century", "early 19th century". Very coarse, low trust.
    m = re.search(r"\b(early|mid|late)?[\s-]*(1[6-9]|20|21)(?:st|nd|rd|th)[\s-]+century", t)
    if m:
        qualifier, c = m.group(1), int(m.group(2))
        base = (c - 1) * 100  # the 19th century starts at 1800
        span = {"early": (0, 33), "mid": (33, 66), "late": (66, 99)}.get(qualifier, (0, 99))
        return _make(base + span[0], base + span[1], "century", source)

    # --- A single year, with and without a clue ------------------------------
    # A year right after a word like "taken", "photographed" or "negative".
    # "negative" matters for museum prints: "negative 1859; print 1862" means the
    # photo was captured in 1859, and the print was made later.
    m = re.search(r"\b(?:taken|photographed|photo|negative|built|in|dated|from)\s+(?:in\s+)?" + _YEAR, t)
    if m:
        y = int(m.group(1))
        return _make(y, y, "year_labeled", source)

    # Last resort: any standalone 4-digit year. Weak on purpose, because it
    # could be a street number or an ID.
    m = re.search(_YEAR, t)
    if m:
        y = int(m.group(1))
        return _make(y, y, "bare_year", source)

    return None


# Words that mean "this date is NOT when the photo was taken", such as
# "9 April 2008 (original upload date)" or "scanned 2005". The photo must be
# older than such a date, so we keep the year but trust it far less (Lesson 2).
_NOT_CAPTURE_CUES = re.compile(r"\b(upload\w*|scan\w*|digiti[sz]ed)\b", re.IGNORECASE)
_NOT_CAPTURE_PENALTY = 0.4

# Digital cameras (and so, EXIF clock times) barely exist before the mid 1990s.
_EARLIEST_DIGITAL_YEAR = 1995


def parse_date_field(text: str) -> Optional[DateEstimate]:
    """Parse the Commons file page's structured 'Date' field (our best source).

    Beyond plain text parsing, this applies two sanity rules specific to this
    field:
      * a field that is *only* a year ("1849") is an explicit statement by the
        uploader, so it gets high confidence (a bare year in a filename doesn't);
      * dates that are really upload/scan dates, or timestamps from a broken
        camera clock, get their confidence cut.
    """
    if not text:
        return None

    # Some fields arrive wrapped in stray quotes or whitespace.
    cleaned = text.strip().strip('"\'')

    # The whole field is just a year, e.g. "1849".
    if re.fullmatch(_YEAR, cleaned):
        y = int(cleaned)
        return _make(y, y, "month_year", "date_field")

    est = parse_date_text(cleaned, "date_field")
    if est is None:
        return None

    # Lesson 2: "(original upload date)" / "scanned ..." are not capture dates.
    if _NOT_CAPTURE_CUES.search(cleaned):
        est = replace(est, confidence=round(est.confidence * _NOT_CAPTURE_PENALTY, 3))

    # Lesson 3: a clock time (HH:MM) means a digital timestamp, and an early
    # year with one is almost certainly a broken camera clock, such as the real
    # case "1648-01-03 17:00:23" on a Taj Mahal photo. Cut confidence by 5x.
    if re.search(r"\d{1,2}:\d{2}", cleaned) and est.start_year < _EARLIEST_DIGITAL_YEAR:
        est = replace(est, confidence=round(est.confidence * 0.2, 3))

    return est


def parse_exif_datetime(value: str) -> Optional[DateEstimate]:
    """Parse an EXIF timestamp like '2019:07:04 12:33:10' (year only).

    EXIF is only a fallback (see estimate_image_date): for digitized old photos
    it records when the *scan* was made, not when the photo was taken.
    """
    if not value:
        return None
    m = re.match(r"\s*" + _YEAR + r"[:\-/](\d{1,2})", value)
    if not m:
        return None
    y = int(m.group(1))
    return _make(y, y, "full_date", "exif")


def best_estimate(estimates: Iterable[Optional[DateEstimate]]) -> Optional[DateEstimate]:
    """From several candidates, keep the most confident one.

    If two are equally confident, the narrower interval wins (an exact year
    beats a whole decade). Returns None if there are no candidates.
    """
    candidates = [e for e in estimates if e is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda e: (e.confidence, -e.width))


def estimate_image_date(
    filename: str = "",
    categories: Iterable[str] = (),
    description: str = "",
    exif_datetime: str = "",
    date_field: str = "",
) -> Optional[DateEstimate]:
    """Combine everything we know about one image into a single date estimate.

    Priority order (this order is the whole point):
      1. The structured Commons 'Date' field, if it parses. It wins even over a
         more "confident-looking" description, because descriptions can hold
         unrelated years (an artist's lifespan, an accession number).
      2. The best of the filename / category / description text.
      3. EXIF, only as a last resort. For scanned old photos it's the scan date,
         which is always later than the real capture date, so letting it compete
         with an explicit date would label 1920s photos as modern.
    """
    field_est = parse_date_field(date_field)
    if field_est is not None:
        return field_est

    estimates = [
        parse_date_text(filename, "filename"),
        parse_date_text(description, "description"),
    ]
    estimates.extend(parse_date_text(c, "category") for c in categories)
    best = best_estimate(estimates)
    if best is not None:
        return best

    return parse_exif_datetime(exif_datetime)
