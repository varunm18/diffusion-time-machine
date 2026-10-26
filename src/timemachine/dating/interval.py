"""Core types: date intervals and the date candidates parsers produce.

Dates on Commons are rarely exact for old images ("circa 1905", "1920s",
"between 1890 and 1900"), so every date is an interval. We use half-open
intervals in *decimal years*: the year 1900 is ``[1900.0, 1901.0)``, the 1920s
are ``[1920.0, 1930.0)`` and 7 July 2014 is ``[2014.512..., 2014.515...)``.
Decimal years are easy to compare, intersect and feed to a model, and still
keep month/day resolution for seasonality.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

# Earliest/latest years any parser will accept as a capture date.
MIN_YEAR = 1000
MAX_YEAR = dt.date.today().year


class Confidence:
    """Shared confidence levels, so sources are weighed consistently."""

    VERY_HIGH = 0.95  # e.g. born-digital EXIF agreeing with the description page
    HIGH = 0.85       # explicit date on the description page
    MEDIUM = 0.6      # dated category ("1930 photographs of Dresden")
    LOW = 0.35        # decade/era category, year in the subcategory name
    VERY_LOW = 0.2    # a year somewhere in the file title


class Precision:
    """How the interval was specified (independent of its width)."""

    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    DECADE = "decade"
    CENTURY = "century"
    RANGE = "range"    # explicit start and end ("between 1890 and 1900")
    BOUND = "bound"    # only one side known ("before 1935")


def decimal_year(day: dt.date) -> float:
    """``date(2014, 7, 2)`` -> 2014.4986... (fraction of the year elapsed)."""
    start = dt.date(day.year, 1, 1)
    days_in_year = (dt.date(day.year + 1, 1, 1) - start).days
    return day.year + (day - start).days / days_in_year


@dataclass(frozen=True)
class DateInterval:
    """Half-open interval ``[lo, hi)`` in decimal years."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not self.lo < self.hi:
            raise ValueError(f"Empty interval [{self.lo}, {self.hi})")

    # -- constructors ------------------------------------------------------------
    @classmethod
    def from_years(cls, first: int, last: int) -> DateInterval:
        """Whole years, both inclusive: ``from_years(1920, 1929)`` is the 1920s."""
        return cls(float(first), float(last + 1))

    @classmethod
    def from_date(cls, year: int, month: int | None = None, day: int | None = None) -> DateInterval:
        """A single year, month or day."""
        if month is None:
            return cls.from_years(year, year)
        if day is None:
            start = dt.date(year, month, 1)
            end = dt.date(year + (month == 12), month % 12 + 1, 1)
        else:
            start = dt.date(year, month, day)
            end = start + dt.timedelta(days=1)
        return cls(decimal_year(start), decimal_year(end))

    # -- properties --------------------------------------------------------------
    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2

    @property
    def width(self) -> float:
        return self.hi - self.lo

    # -- set operations ----------------------------------------------------------
    def overlaps(self, other: DateInterval) -> bool:
        return self.lo < other.hi and other.lo < self.hi

    def intersect(self, other: DateInterval) -> DateInterval | None:
        lo, hi = max(self.lo, other.lo), min(self.hi, other.hi)
        return DateInterval(lo, hi) if lo < hi else None

    def clip(self, lo: float | None = None, hi: float | None = None) -> DateInterval | None:
        """Restrict to ``[lo, hi)``; None if nothing is left."""
        new_lo = self.lo if lo is None else max(self.lo, lo)
        new_hi = self.hi if hi is None else min(self.hi, hi)
        return DateInterval(new_lo, new_hi) if new_lo < new_hi else None

    def __str__(self) -> str:
        first, last = int(self.lo), int(self.hi - 1e-9)
        return str(first) if first == last else f"{first}-{last}"


@dataclass(frozen=True)
class DateCandidate:
    """One piece of evidence about when an image was made.

    Each parser (description-page date, EXIF, categories, ...) turns what it
    finds into candidates; :mod:`timemachine.dating.resolve` weighs them.
    """

    interval: DateInterval
    source: str          # which parser produced it, see Source
    precision: str       # see Precision
    confidence: float    # see Confidence
    evidence: str        # the raw text it came from, for debugging/review
    flags: tuple[str, ...] = ()

    def with_flags(self, *flags: str) -> DateCandidate:
        return DateCandidate(self.interval, self.source, self.precision, self.confidence,
                             self.evidence, self.flags + tuple(f for f in flags if f not in self.flags))


class Source:
    """Names of the evidence sources (one parser module each)."""

    QS = "qs"                    # hidden machine-readable date on the description page
    DATE_TEXT = "date_text"      # visible date text on the description page
    EXIF = "exif"                # camera EXIF DateTimeOriginal
    CATEGORY = "category"        # current Commons categories of the file
    SUBCATEGORY = "subcategory"  # MegaScenes subcategory the file was found in
    TITLE = "title"              # the file title itself
