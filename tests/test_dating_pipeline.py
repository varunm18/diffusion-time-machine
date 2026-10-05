"""End-to-end dating of real Commons pages (Dresden Frauenkirche fixtures).

Each case documents a situation we met in the data; see docs/findings.md.
"""
import json
from pathlib import Path

import pytest

from timemachine.dating.interval import DateInterval
from timemachine.dating.pipeline import date_image
from timemachine.megascenes.metadata import parse_page

FIXTURES = Path(__file__).parent / "fixtures"
Y = DateInterval.from_years


@pytest.fixture(scope="module")
def dated():
    out = {}
    for item in json.loads((FIXTURES / "commons_pages.json").read_text()):
        meta = parse_page(item["page"])
        out[meta.title] = date_image(meta, [item["subcat"]])
    return out


def get(dated, prefix):
    (key,) = [k for k in dated if k.startswith(prefix)]
    return dated[key]


@pytest.mark.parametrize("prefix, expected, medium", [
    # hidden QS "circa 1851" on an engraving after a painting
    ("A. H. Payne", Y(1846, 1856), "painting"),
    # QS range narrowed by the category "Dresden in the 1890s"; a color photochrom
    ("Altstadt, Dresden", Y(1890, 1899), "photo"),
    # archive photo with a plain year
    ("Bundesarchiv B 145", Y(1930, 1930), "photo"),
    # "1887/1888" typed in; EskoScan EXIF has no date
    ("Fotothek df hauptkatalog", Y(1887, 1888), "photo"),
    # QS circa 1913 narrowed by "1913 paintings"
    ("Gotthardt Kuehl Dresdner Töpfermarkt", Y(1913, 1913), "painting"),
])
def test_historical_images(dated, prefix, expected, medium):
    result, med = get(dated, prefix)
    assert result.interval == expected
    assert med.medium == medium


def test_museum_scan_exif_is_digitization(dated):
    # Library of Congress photochrom, EXIF from a Phase One P 65+ reproduction back in 2017
    result, med = get(dated, "Dresden. Frauenkirche mit Neumarkt LOC")
    assert result.interval == Y(1898, 1898)
    assert "exif_is_digitization" in result.flags and int(result.digitized.lo) == 2017
    assert med.color == "color"


def test_painting_bounded_by_creator_lifespan(dated):
    # "by 1915" + Photoshop EXIF from 2011: EXIF is impossible (after the painter's death)
    result, _ = get(dated, "Gotthardt Kuehl Neumarkt")
    assert result.interval.lo == 1860 and result.interval.hi == 1916


@pytest.mark.parametrize("prefix, year", [
    ("Anna selbdritt", 2010),       # modern photo of a sculpture from the old church
    ("DD-Lapidarium-29", 2018),     # modern photo filed under "(before 1945)"
    ("20121007150DR", 2012),
])
def test_modern_photos_trust_camera_exif(dated, prefix, year):
    result, med = get(dated, prefix)
    assert int(result.interval.lo) == year and result.confidence >= 0.95
    assert med.medium == "photo"


def test_reprint_postcard_is_flagged_for_review(dated):
    # A 2000 reprint of a c.1930 view: we keep the stated date but flag the conflict.
    result, med = get(dated, "32342-Dresden-2000")
    assert result.interval == Y(2000, 2000)
    assert "weak_conflict:title" in result.flags and med.medium == "postcard"
