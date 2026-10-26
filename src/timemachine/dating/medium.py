"""Image type (photo, painting, print, ...) and color hints from Commons metadata.

This is a cheap first pass over categories, title and object name; pixel-based
color detection and a VLM can refine it later. The type matters for dating
(photos cannot predate 1826) and for training (paintings and engravings do not
have faithful perspective, so they cannot be posed like photos).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from timemachine.dating.sources.categories import content_categories
from timemachine.megascenes.metadata import ImageMetadata


class Medium:
    PHOTO = "photo"
    POSTCARD = "postcard"    # usually photographic, sometimes drawn/printed
    PAINTING = "painting"
    PRINT = "print"          # engraving, etching, lithograph, woodcut, ...
    DRAWING = "drawing"
    PLAN = "plan"            # maps, floor plans, elevations
    ARTWORK = "artwork"      # "in art" without a more specific type
    UNKNOWN = "unknown"


# Media whose geometry is a real camera projection (can be posed).
PHOTOGRAPHIC = frozenset({Medium.PHOTO, Medium.POSTCARD})


def _r(regex: str) -> re.Pattern:
    return re.compile(regex, re.I)


# Checked in order; the first match wins. Specific artwork types beat "photo"
# because photos *of* paintings still show a painting.
_MEDIUM_RULES: tuple[tuple[str, re.Pattern], ...] = (
    (Medium.PLAN, _r(r"\bmaps?\b|\bfloor ?plans?\b|\bplans? of\b|grundriss|\belevations?\b|"
                     r"cross[- ]sections?|blueprints?|architectural drawings?")),
    (Medium.PAINTING, _r(r"\bpaintings?\b|oil on (?:canvas|panel|wood)|gemälde|\bpeintures?\b|"
                         r"watercolou?rs?|aquarell|gouache|tempera|\bfrescos?\b")),
    (Medium.PRINT, _r(r"engravings?|etchings?|lithograph|woodcuts?|aquatints?|mezzotints?|"
                      r"stahlstich|kupferstich|holzstich|radierung|gravures?|chromolitho")),
    (Medium.DRAWING, _r(r"\bdrawings?\b|\bsketch(?:es)?\b|\bdessins?\b|zeichnung")),
    (Medium.POSTCARD, _r(r"postcards?|cartes? postales?|postkarten?|ansichtskarten?")),
    (Medium.PHOTO, _r(r"photograph|\bphotos?\b|albumen|glass plate|\bnegatives?\b|daguerreotyp|"
                      r"ambrotyp|tintyp|autochrom|photochrom|stereo|gelatin silver|lantern slides?|"
                      r"cyanotyp|fotografie|\bfotos?\b")),
    (Medium.ARTWORK, _r(r"\bin art\b|\bartworks?\b")),
)
_BW = _r(r"black[- ]and[- ]white|\bb&w\b|monochrome|gr[ae]yscale|\bsepia\b|schwarz-?wei|noir et blanc|"
         r"blanco y negro")
_COLOR = _r(r"colou?r photographs|hand-colou?red|colou?ri[sz]ed|autochrom|photochrom|kodachrome|"
            r"ektachrome|chromolitho")
_HISTORICAL = _r(r"historical (?:images|photographs|pictures|views)|history of|old (?:photographs|postcards)|"
                 r"\bvintage\b|before 1[89]\d\d|in the 1[89]\d0s|1[89]th century|bundesarchiv|"
                 r"nationaal archief|library of congress|fortepan|deutsche fotothek|rijksmuseum|gallica|"
                 r"europeana|author died more than|pd-old|pd-us-expired|published before 19")


@dataclass(frozen=True)
class MediumResult:
    medium: str             # see Medium
    evidence: str           # the text that decided it
    color: str              # "bw", "color" or "unknown"
    historical_hint: bool   # metadata suggests an old image (archive, PD-old, old era, ...)


def classify_medium(meta: ImageMetadata) -> MediumResult:
    """Classify one image from its metadata (no pixels)."""
    fields = [*content_categories(meta.categories), meta.title, meta.object_name]
    text = " | ".join(f for f in fields if f)

    medium, evidence = Medium.UNKNOWN, ""
    for label, pattern in _MEDIUM_RULES:
        m = pattern.search(text)
        if m:
            medium, evidence = label, m.group(0)
            break
    if medium == Medium.UNKNOWN and (meta.exif.get("Make") or meta.exif.get("Model")):
        medium, evidence = Medium.PHOTO, "exif camera"

    if _BW.search(text):
        color = "bw"
    elif _COLOR.search(text):
        color = "color"
    else:
        color = "unknown"

    # License/credit categories ("Author died more than 100 years ago ...") are
    # maintenance categories but still good evidence of age, so use all of them here.
    historical = bool(_HISTORICAL.search(" | ".join([*meta.categories, meta.title,
                                                     meta.license, meta.credit])))
    return MediumResult(medium, evidence, color, historical)
