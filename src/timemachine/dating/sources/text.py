"""Parse free-text dates, as found in the visible date field of a Commons page.

Used when there is no hidden QS date (plain ``|date=1878`` in ``{{Information}}``,
``{{Taken on|...}}``, typed text). Handles, in order of precedence::

    unknown / empty               -> nothing
    "1887/1888", "1890-1900",     -> range
      "between 1890 and 1900", "from 1890 until 1900"
    "before 1935" / "after 1890"  -> one-sided bound
    "2014-07-07 11:15", "7 July 2014", "July 2014", "2014-07"
                                  -> day / month
    "circa 1905", "ca. 1930", "um 1930", "vers 1930"
                                  -> year +- 5
    "1920s", "1920er"             -> decade
    "19th century"                -> century
    "1878"                        -> year
    several unrelated years       -> their span, lower confidence

Anything mentioning an upload ("original upload date") is flagged, since it is
not a capture date.
"""
from __future__ import annotations

import re

from timemachine.dating.interval import (
    MAX_YEAR, MIN_YEAR, Confidence, DateCandidate, DateInterval, Precision, Source,
)
from timemachine.dating.sources.qs import CIRCA_YEARS

_MONTHS = {
    # English, German, French, Spanish, Italian, Dutch (lowercase, accents kept)
    1: "january jan januar janvier enero gennaio januari",
    2: "february feb februar février fevrier febrero febbraio februari",
    3: "march mar märz maerz mars marzo maart",
    4: "april apr avril abril aprile",
    5: "may mai mayo maggio mei",
    6: "june jun juni juin junio giugno",
    7: "july jul juli juillet julio luglio",
    8: "august aug août aout agosto augustus",
    9: "september sep sept septembre septiembre settembre",
    10: "october oct oktober octobre octubre ottobre",
    11: "november nov novembre noviembre",
    12: "december dec dezember décembre decembre diciembre dicembre",
}
MONTH_NUMBER = {name: num for num, names in _MONTHS.items() for name in names.split()}
MONTH_PATTERN = "|".join(sorted(map(re.escape, MONTH_NUMBER), key=len, reverse=True))

_Y = r"(\d{4})"
_RANGE = re.compile(rf"(?:between|from|zwischen|entre)?\s*\b{_Y}\s*(?:-|–|—|/|and|to|until|till|bis|und|et|à)\s*{_Y}\b")
_ISO_DAY = re.compile(rf"\b{_Y}[-:.](\d{{1,2}})[-:.](\d{{1,2}})\b")
_ISO_MONTH = re.compile(rf"\b{_Y}-(\d{{1,2}})\b(?![-:.]\d)")
_DMY = re.compile(rf"\b(\d{{1,2}})\.?\s+({MONTH_PATTERN})\.?,?\s+{_Y}\b")
_MDY = re.compile(rf"\b({MONTH_PATTERN})\.?\s+(\d{{1,2}}),?\s+{_Y}\b")
_MY = re.compile(rf"\b({MONTH_PATTERN})\.?,?\s+{_Y}\b")
_BEFORE = re.compile(rf"\b(?:before|prior to|vor|avant|antes de|not after)\s+(?:[\w.]+\s+){{0,2}}{_Y}\b")
_AFTER = re.compile(rf"\b(?:after|nach|après|apres|después de|not before|since)\s+(?:[\w.]+\s+){{0,2}}{_Y}\b")
_CIRCA = re.compile(rf"(?:\b(?:circa|ca|c|around|about|approx|approximately|um|vers|env|hacia)\b\.?|~)\s*{_Y}\b(?!s)")
_DECADE = re.compile(r"\b(\d{3}0)(?:s|'s|er|er jahre)\b")
_CENTURY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th|\.)\s+(?:century|jahrhundert)\b")
_YEAR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_UPLOAD = re.compile(r"upload", re.I)


def _plausible(year: int) -> bool:
    return MIN_YEAR <= year <= MAX_YEAR


def _candidate(interval: DateInterval, precision: str, confidence: float, text: str,
               source: str, flags: list[str]) -> DateCandidate:
    return DateCandidate(interval, source, precision, confidence, text, tuple(flags))


def parse_date_text(
    text: str, source: str = Source.DATE_TEXT, confidence: float = Confidence.HIGH,
) -> DateCandidate | None:
    """Parse one free-text date; None if no usable date is found."""
    if not text:
        return None
    low = " ".join(text.lower().split())
    if not (_YEAR.search(low) or _CENTURY.search(low)):  # "Unknown date", empty, etc.
        return None
    flags = ["upload_date"] if _UPLOAD.search(low) else []
    if "circa" in low or re.search(r"\b(?:ca|c)\.", low):
        flags.append("circa")

    m = _RANGE.search(low)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if _plausible(a) and _plausible(b) and a <= b:
            return _candidate(DateInterval.from_years(a, b), Precision.RANGE, confidence, text, source, flags)

    m = _BEFORE.search(low)
    if m and _plausible(int(m.group(1))):
        interval = DateInterval(float(MIN_YEAR), int(m.group(1)) + 1.0)
        return _candidate(interval, Precision.BOUND, Confidence.LOW, text, source, flags + ["open_start"])
    m = _AFTER.search(low)
    if m and _plausible(int(m.group(1))):
        interval = DateInterval(float(m.group(1)), MAX_YEAR + 1.0)
        return _candidate(interval, Precision.BOUND, Confidence.LOW, text, source, flags + ["open_end"])

    for pattern, order in ((_ISO_DAY, "ymd"), (_DMY, "dmy"), (_MDY, "mdy")):
        m = pattern.search(low)
        if m:
            parts = dict(zip(order, m.groups()))
            month = parts["m"] if parts["m"].isdigit() else MONTH_NUMBER[parts["m"]]
            try:
                interval = DateInterval.from_date(int(parts["y"]), int(month), int(parts["d"]))
            except ValueError:  # e.g. day 31 in a 30-day month, or a non-date number triple
                continue
            if _plausible(int(parts["y"])):
                return _candidate(interval, Precision.DAY, confidence, text, source, flags)

    for pattern, order in ((_MY, "my"), (_ISO_MONTH, "ym")):
        m = pattern.search(low)
        if m:
            parts = dict(zip(order, m.groups()))
            month = int(parts["m"]) if parts["m"].isdigit() else MONTH_NUMBER[parts["m"]]
            if 1 <= month <= 12 and _plausible(int(parts["y"])):
                interval = DateInterval.from_date(int(parts["y"]), month)
                return _candidate(interval, Precision.MONTH, confidence, text, source, flags)

    m = _CIRCA.search(low)
    if m and _plausible(int(m.group(1))):
        y = int(m.group(1))
        interval = DateInterval.from_years(y - CIRCA_YEARS, y + CIRCA_YEARS)
        return _candidate(interval, Precision.YEAR, confidence, text, source,
                          flags if "circa" in flags else flags + ["circa"])

    m = _DECADE.search(low)
    if m and _plausible(int(m.group(1))):
        d = int(m.group(1))
        return _candidate(DateInterval.from_years(d, d + 9), Precision.DECADE, confidence, text, source, flags)

    m = _CENTURY.search(low)
    if m:
        c = int(m.group(1))
        if _plausible(100 * c):
            interval = DateInterval.from_years(100 * (c - 1) + 1, 100 * c)
            return _candidate(interval, Precision.CENTURY, confidence, text, source, flags)

    years = sorted({int(y) for y in _YEAR.findall(low) if _plausible(int(y))})
    if len(years) == 1:
        return _candidate(DateInterval.from_years(years[0], years[0]), Precision.YEAR,
                          confidence, text, source, flags)
    if len(years) > 1:
        return _candidate(DateInterval.from_years(years[0], years[-1]), Precision.RANGE,
                          min(confidence, Confidence.LOW), text, source, flags + ["multiple_years"])
    return None
