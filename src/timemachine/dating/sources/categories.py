"""Dates from Commons category names (also used for MegaScenes subcategory names).

Many categories carry a date: "1930 photographs of Dresden", "Dresden in 1898",
"Brandenburg Gate in July 1945", "Notre-Dame de Paris in the 1920s",
"Photographs taken on 2015-06-01", "Frauenkirche Dresden (before 1945)".

Caveats encoded below:
* Era categories ("X in the 1900s") describe the *depicted* era, which is
  usually but not always when the image was made -> lower confidence.
* Year-dated postcards/prints may be reprints ("2000 postcards of Dresden" for
  a c.1930 view) -> flagged ``publication``.
* Many years in categories are not capture dates at all: construction years,
  camera models, ship names, licenses... -> excluded.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from timemachine.dating.interval import (
    MAX_YEAR, MIN_YEAR, DateCandidate, DateInterval, Precision, Source,
)
from timemachine.dating.sources.text import MONTH_NUMBER, MONTH_PATTERN

# Categories whose years say nothing about when the image was made.
_EXCLUDE = re.compile(
    r"\b(?:built|constructed|completed|established|founded|opened|inaugurated|consecrated|"
    r"demolished|disestablished|closed|births|deaths|uploaded|taken with|created with|"
    r"files from|license|licen[cs]ed|cc-|pd-|gfdl|media needing|supported by|"
    r"ship, \d{4}|in use since)\b",
    re.I,
)
# Hidden/maintenance categories: about the file page, not the image content.
_MAINTENANCE = re.compile(
    r"^(?:pages|files|media|images|photos) (?:with|without|from|by|uploaded|needing|lacking)\b|"
    r"\b(?:uploaded|taken with|created with|self-published|supported by|wikidata|geocoded|"
    r"location|coordinates|exif|quality images|valued images|featured pictures|"
    r"license|licen[cs]ed|copyright|author died|cc-|pd-|gfdl|picture of the day|"
    r"media of the day|wiki loves|translation possible|user:|uploaded by)\b",
    re.I,
)


def is_maintenance_category(category: str) -> bool:
    """True for technical categories ("Pages with maps", "CC-BY-SA-4.0", ...)."""
    return bool(_MAINTENANCE.search(category.replace("_", " ")))


def content_categories(categories: Iterable[str]) -> list[str]:
    """Categories describing the image itself (maintenance categories removed)."""
    return [c for c in categories if not is_maintenance_category(c)]


# A year in parentheses is usually a disambiguator: "(1732 building)", "(ship, 1869)".
_PAREN_YEAR = re.compile(r"\((?![^)]*\b(?:before|after)\b)[^)]*\d{4}[^)]*\)")

_IMAGE_KINDS = r"(?:black and white |colou?r |aerial |stereoscopic )?(?:photographs|photos|images)"
_ART_KINDS = r"(?:paintings|drawings|engravings|etchings|lithographs|prints|watercolou?rs|postcards|maps)"


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern
    kind: str            # how to turn the match into an interval (see _interval)
    precision: str
    confidence: float
    flags: tuple[str, ...] = ()


def _r(regex: str) -> re.Pattern:
    return re.compile(regex, re.I)


# Ordered: the first matching rule wins for a given category.
_RULES: tuple[_Rule, ...] = (
    _Rule(_r(r"photographs taken on (\d{4})-(\d{2})-(\d{2})"), "ymd", Precision.DAY, 0.8),
    _Rule(_r(rf"^(\d{{4}}) {_IMAGE_KINDS}\b"), "year", Precision.YEAR, 0.75),
    _Rule(_r(rf"^(\d{{3}}0)s {_IMAGE_KINDS}\b"), "decade", Precision.DECADE, 0.55),
    _Rule(_r(r"^(\d{4}) postcards\b"), "year", Precision.YEAR, 0.55, ("publication",)),
    _Rule(_r(rf"^(\d{{4}}) {_ART_KINDS}\b"), "year", Precision.YEAR, 0.7, ("artwork",)),
    _Rule(_r(rf"^(\d{{3}}0)s {_ART_KINDS}\b"), "decade", Precision.DECADE, 0.5, ("artwork",)),
    _Rule(_r(rf"\bin ({MONTH_PATTERN}) (\d{{4}})$"), "month", Precision.MONTH, 0.65),
    _Rule(_r(r"\b(?:in|im|en) (\d{4})$"), "year", Precision.YEAR, 0.6),
    _Rule(_r(r"^(\d{4}) (?:in|im|en)\b"), "year", Precision.YEAR, 0.6),
    _Rule(_r(r"\bwiki loves monuments (\d{4})"), "year", Precision.YEAR, 0.5, ("contest",)),
    _Rule(_r(r"\bin the (\d{3}0)s\b"), "decade", Precision.DECADE, 0.45, ("depicted_era",)),
    _Rule(_r(r"^(\d{3}0)s (?:in|im|en)\b"), "decade", Precision.DECADE, 0.45, ("depicted_era",)),
    _Rule(_r(r"\bin the (\d{1,2})(?:st|nd|rd|th) century\b"), "century", Precision.CENTURY, 0.3,
          ("depicted_era",)),
    _Rule(_r(r"\bbefore (\d{4})\b"), "before", Precision.BOUND, 0.35, ("open_start",)),
    _Rule(_r(r"\bafter (\d{4})\b"), "after", Precision.BOUND, 0.35, ("open_end",)),
    _Rule(_r(r"\b(\d{4}) ?[-–] ?(\d{4})\b"), "span", Precision.RANGE, 0.35, ("span",)),
)


def _interval(kind: str, groups: tuple[str, ...]) -> DateInterval | None:
    try:
        if kind == "ymd":
            return DateInterval.from_date(int(groups[0]), int(groups[1]), int(groups[2]))
        if kind == "month":
            return DateInterval.from_date(int(groups[1]), MONTH_NUMBER[groups[0].lower()])
        year = int(groups[0])
        if kind == "year":
            return DateInterval.from_years(year, year)
        if kind == "decade":
            return DateInterval.from_years(year, year + 9)
        if kind == "century":
            return DateInterval.from_years(100 * (year - 1) + 1, 100 * year)
        if kind == "before":
            return DateInterval(float(MIN_YEAR), float(year))
        if kind == "after":
            return DateInterval(float(year), float(MAX_YEAR + 1))
        if kind == "span":
            last = int(groups[1])
            return DateInterval.from_years(year, last) if year <= last else None
    except ValueError:
        return None
    raise ValueError(f"Unknown rule kind {kind!r}")


def _plausible(interval: DateInterval) -> bool:
    return MIN_YEAR <= interval.lo and interval.hi <= MAX_YEAR + 1


def parse_category(category: str, source: str = Source.CATEGORY,
                   confidence_scale: float = 1.0) -> DateCandidate | None:
    """Candidate from one category name (underscores are treated as spaces)."""
    name = category.replace("_", " ").strip()
    if _EXCLUDE.search(name):
        return None
    name = _PAREN_YEAR.sub(" ", name).strip()
    for rule in _RULES:
        m = rule.pattern.search(name)
        if not m:
            continue
        interval = _interval(rule.kind, m.groups())
        if interval is None or not _plausible(interval):
            return None
        return DateCandidate(interval, source, rule.precision,
                             round(rule.confidence * confidence_scale, 3), category, rule.flags)
    return None


def parse_categories(categories: Iterable[str], source: str = Source.CATEGORY,
                     confidence_scale: float = 1.0) -> list[DateCandidate]:
    """Candidates from every dated category in ``categories``."""
    found = (parse_category(c, source, confidence_scale) for c in categories)
    return [c for c in found if c is not None]
