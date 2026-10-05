"""Weak date evidence from the file title (e.g. "Frauenkirche um 1897.jpg").

Titles are noisy: they contain Flickr ids, camera counters (IMG_2034),
catalogue numbers and years of *other* things. So we only accept three clear
patterns and give them low confidence:

* a full date ``2007-04-29`` or a camera-style ``20230306`` -> day
* "um/circa/ca./vers 1930"                                  -> circa year
* exactly one standalone year 1826..now                     -> year
"""
from __future__ import annotations

import re

from timemachine.dating.interval import (
    MAX_YEAR, Confidence, DateCandidate, DateInterval, Precision, Source,
)
from timemachine.dating.sources.qs import CIRCA_YEARS

FIRST_PHOTO_YEAR = 1826  # earliest surviving photograph; older "years" in titles are rarely dates

_EXTENSION = re.compile(r"\.[a-z0-9]{2,5}$", re.I)
_NOISE = re.compile(r"\(\d{6,}\)|\b(?:img|dsc|dscn|dscf|p|pict|photo|scan)[_ -]?\d+\b|\d{5,}", re.I)
_FULL_DATE = re.compile(r"(?<!\d)((?:18|19|20)\d\d)[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)")
_CIRCA = re.compile(r"\b(?:um|circa|ca\.?|c\.|vers|around|anno)\s*((?:18|19|20)\d\d)(?!\d)", re.I)
_YEAR = re.compile(r"(?<![\d.])((?:18|19|20)\d\d)(?![\d])")


def parse_title(title: str) -> DateCandidate | None:
    """Candidate from a Commons file title, or None if nothing clear is found."""
    stem = _EXTENSION.sub("", title).replace("_", " ")

    m = _FULL_DATE.search(stem)
    if m:
        try:
            interval = DateInterval.from_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            interval = None
        if interval and int(m.group(1)) <= MAX_YEAR:
            return DateCandidate(interval, Source.TITLE, Precision.DAY, Confidence.LOW, title)

    m = _CIRCA.search(stem)
    if m and FIRST_PHOTO_YEAR <= int(m.group(1)) <= MAX_YEAR:
        y = int(m.group(1))
        return DateCandidate(DateInterval.from_years(y - CIRCA_YEARS, y + CIRCA_YEARS), Source.TITLE,
                             Precision.YEAR, Confidence.LOW, title, ("circa",))

    cleaned = _NOISE.sub(" ", stem)
    years = {int(y) for y in _YEAR.findall(cleaned) if FIRST_PHOTO_YEAR <= int(y) <= MAX_YEAR}
    if len(years) == 1:
        (y,) = years
        return DateCandidate(DateInterval.from_years(y, y), Source.TITLE, Precision.YEAR,
                             Confidence.VERY_LOW, title)
    return None
