"""Parse the hidden machine-readable date ("date QS") on Commons description pages.

When an uploader uses a date template (``{{circa|1905}}``, ``{{other date|...}}``)
or the ``{{Artwork}}``/``{{Photograph}}`` templates, Commons renders a hidden
string in QuickStatements syntax next to the visible date, e.g.::

    P571,+1851-00-00T00:00:00Z/9,P1480,Q5727902          circa 1851
    P,+1950-00-00T00:00:00Z/7,P1319,+1900-00-00T00:00:00Z/9,P1326,+1930-00-00T00:00:00Z/9
                                                          between 1900 and 1930
    P,+1920-00-00T00:00:00Z/8                              1920s

It is a list of ``property, value`` pairs. The first pair is the main date
(``P571`` = inception, or an empty ``P``); later pairs are qualifiers. Commons
computes it with its own date modules, so it is the most reliable structured
date we have.
"""
from __future__ import annotations

import math
import re

from timemachine.dating.interval import (
    MAX_YEAR, MIN_YEAR, Confidence, DateCandidate, DateInterval, Precision, Source,
)

CIRCA_YEARS = 5  # "circa 1905" -> 1900..1910

_TIME = re.compile(r"^([+-])(\d{1,4})-(\d{2})-(\d{2})T[^/]*/(\d+)$")
_EARLIEST = ("P1319", "P580")   # earliest date / start time
_LATEST = ("P1326", "P582")     # latest date / end time
_SOURCING = "P1480"             # sourcing circumstances
_CIRCA = "Q5727902"
_UNCERTAIN = {"Q18122778", "Q56644435", "Q30230067"}  # presumably / probably / possibly


def parse_time(value: str) -> tuple[DateInterval, str] | None:
    """A Wikibase time value (``+1920-00-00T00:00:00Z/8``) -> (interval, precision).

    Wikibase precision codes: 11 day, 10 month, 9 year, 8 decade, 7 century,
    6 millennium. Digits below the precision are ignored. Centuries follow
    Wikibase: ``+1850/7`` and ``+1900/7`` are both the 19th century (1801-1900).
    """
    m = _TIME.match(value.strip())
    if not m or m.group(1) == "-":
        return None
    year, month, day, precision = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    if not MIN_YEAR <= year <= MAX_YEAR:
        return None
    try:
        if precision >= 11 and month and day:
            return DateInterval.from_date(year, month, day), Precision.DAY
        if precision >= 10 and month:
            return DateInterval.from_date(year, month), Precision.MONTH
    except ValueError:  # malformed month/day (e.g. "+1934-13-00"): fall back to the year
        return DateInterval.from_years(year, year), Precision.YEAR
    if precision >= 9:
        return DateInterval.from_years(year, year), Precision.YEAR
    if precision == 8:
        decade = year // 10 * 10
        return DateInterval.from_years(decade, decade + 9), Precision.DECADE
    if precision == 7:
        century = math.ceil(year / 100)
        return DateInterval.from_years(100 * (century - 1) + 1, 100 * century), Precision.CENTURY
    return None


def parse_qs(qs: str | None) -> DateCandidate | None:
    """Turn a ``date QS`` string into a candidate (None if unusable)."""
    if not qs:
        return None
    tokens = [t.strip() for t in qs.split(",") if t.strip()]
    pairs = list(zip(tokens[0::2], tokens[1::2]))
    if not pairs:
        return None

    main = parse_time(pairs[0][1])
    earliest = latest = None
    flags: list[str] = []
    for prop, value in pairs[1:]:
        if prop in _EARLIEST:
            earliest = parse_time(value)
        elif prop in _LATEST:
            latest = parse_time(value)
        elif prop == _SOURCING:
            flags.append("circa" if value == _CIRCA else "uncertain" if value in _UNCERTAIN else "sourcing")

    confidence = Confidence.HIGH if "uncertain" not in flags else Confidence.MEDIUM
    if earliest and latest:
        interval = DateInterval(earliest[0].lo, max(latest[0].hi, earliest[0].lo + 1))
        return DateCandidate(interval, Source.QS, Precision.RANGE, confidence, qs, tuple(flags))
    if latest:  # "before X": lower end unknown
        interval = DateInterval(float(MIN_YEAR), latest[0].hi)
        return DateCandidate(interval, Source.QS, Precision.BOUND, Confidence.LOW, qs,
                             tuple(flags + ["open_start"]))
    if earliest:  # "after X": upper end unknown
        interval = DateInterval(earliest[0].lo, float(MAX_YEAR + 1))
        return DateCandidate(interval, Source.QS, Precision.BOUND, Confidence.LOW, qs,
                             tuple(flags + ["open_end"]))
    if main is None:
        return None

    interval, precision = main
    if "circa" in flags:
        interval = DateInterval(interval.lo - CIRCA_YEARS, interval.hi + CIRCA_YEARS)
    return DateCandidate(interval, Source.QS, precision, confidence, qs, tuple(flags))
