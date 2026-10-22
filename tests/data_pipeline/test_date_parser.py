"""Tests for data_pipeline/date_parser.py.

HOW TO READ THIS FILE
---------------------
The first half uses synthetic strings to check each pattern the parser knows
(exact dates, ranges, decades, ...). The second half uses REAL strings we copied
from Wikimedia Commons while auditing the Taj Mahal scene. Those real ones are
the valuable tests: each one pins down a mistake the parser actually made once
(a broken camera clock, an artist's lifespan mistaken for a date, ...) so it
can never quietly come back.

Run with:   pytest tests/data_pipeline/test_date_parser.py
"""

import random

import pytest

from data_pipeline.date_parser import (
    DateEstimate,
    best_estimate,
    estimate_image_date,
    parse_date_field,
    parse_date_text,
    parse_exif_datetime,
)


def span(est):
    """Shorthand: the (start_year, end_year) of an estimate."""
    return (est.start_year, est.end_year)


# ----------------------------------------------------------------------------
# Each kind of date wording the parser understands
# ----------------------------------------------------------------------------

def test_iso_full_date():
    est = parse_date_text("Taj Mahal 1889-05-06.jpg", "filename")
    assert span(est) == (1889, 1889)
    assert est.source == "filename"  # the source we pass in is recorded on the result


def test_day_month_year():
    assert span(parse_date_text("Photographed on 6 May 1889", "description")) == (1889, 1889)


def test_month_year():
    assert span(parse_date_text("Taken in March 1932", "description")) == (1932, 1932)


def test_year_range():
    # A range keeps both ends instead of collapsing to one year.
    assert span(parse_date_text("Building works 1920-1925", "description")) == (1920, 1925)


def test_circa():
    # "c. 1890" means roughly 1890, so we allow plus or minus 5 years.
    assert span(parse_date_text("View of the gate, c. 1890", "description")) == (1885, 1895)


def test_decade():
    assert span(parse_date_text("Street scene 1920s", "category")) == (1920, 1929)
    assert span(parse_date_text("Street scene 1920's", "category")) == (1920, 1929)


def test_decade_qualifiers():
    # early / mid / late narrow the decade to roughly its first, middle, last years.
    assert span(parse_date_text("early 1920s", "category")) == (1920, 1923)
    assert span(parse_date_text("mid 1920s", "category")) == (1923, 1926)
    assert span(parse_date_text("late 1920s", "category")) == (1926, 1929)


def test_century():
    assert span(parse_date_text("19th century engraving", "description")) == (1800, 1899)
    assert span(parse_date_text("early 19th century engraving", "description")) == (1800, 1833)


def test_labeled_year():
    # A year right after a cue word like "taken" is more trustworthy than a bare number.
    assert span(parse_date_text("Photo taken in 1957", "description")) == (1957, 1957)


def test_bare_year_in_filename_is_low_confidence():
    # "Taj_Mahal_1957.jpg" might be a year or just a number, so it's trusted less
    # than a full date, even though we still extract the year.
    bare = parse_date_text("Taj_Mahal_1957.jpg", "filename")
    full = parse_date_text("Taj Mahal 1957-03-02.jpg", "filename")
    assert span(bare) == (1957, 1957)
    assert bare.confidence < full.confidence


# ----------------------------------------------------------------------------
# Things that must NOT be read as dates
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    ["", "Taj Mahal front view.jpg", "IMG_123456.jpg", "DSC00019.JPG", "Agra Fort gate"],
)
def test_no_date(text):
    # Ordinary filenames full of digits must not produce a year.
    assert parse_date_text(text, "filename") is None


def test_out_of_range_year_rejected():
    assert parse_date_text("Item 3099", "filename") is None


def test_year_not_taken_from_inside_longer_number():
    # "20190704123" contains "2019" but it's part of a longer number, not a year.
    assert parse_date_text("IMG_20190704123.jpg", "filename") is None


# ----------------------------------------------------------------------------
# EXIF timestamps
# ----------------------------------------------------------------------------

def test_exif_parses_year():
    est = parse_exif_datetime("2019:07:04 12:33:10")
    assert span(est) == (2019, 2019)
    assert est.source == "exif"


def test_exif_less_trusted_than_description():
    # EXIF can be a scan date or a wrong camera clock, so an explicit written
    # date should outrank it.
    exif = parse_exif_datetime("1999:01:01 00:00:00")
    desc = parse_date_text("Photographed on 1 January 1999", "description")
    assert exif.confidence < desc.confidence


def test_exif_garbage():
    assert parse_exif_datetime("not a date") is None
    assert parse_exif_datetime("") is None


# ----------------------------------------------------------------------------
# Combining several sources into one answer
# ----------------------------------------------------------------------------

def test_best_estimate_prefers_higher_confidence():
    good = DateEstimate(1920, 1920, 0.9, "description")
    weak = DateEstimate(1900, 1999, 0.3, "category")
    assert best_estimate([weak, good, None]) is good  # None entries are ignored


def test_best_estimate_empty():
    assert best_estimate([None, None]) is None


def test_text_date_beats_exif():
    # A scanned 1920s photo: EXIF holds the (modern) scan date and must lose to
    # the explicit "1920s" in the category.
    est = estimate_image_date(
        filename="IMG_0001.jpg",
        categories=["Taj Mahal in the 1920s"],
        exif_datetime="2015:03:02 10:00:00",
    )
    assert span(est) == (1920, 1929)
    assert est.source == "category"


def test_exif_used_when_nothing_else():
    # EXIF is the last resort, so it still gets used when there is nothing better.
    est = estimate_image_date(filename="IMG_0001.jpg", exif_datetime="2015:03:02 10:00:00")
    assert span(est) == (2015, 2015)
    assert est.source == "exif"


def test_estimate_image_date_none():
    assert estimate_image_date(filename="IMG_0001.jpg") is None


# ----------------------------------------------------------------------------
# The DateEstimate object itself
# ----------------------------------------------------------------------------

def test_midpoint_and_width():
    est = DateEstimate(1920, 1929, 0.6, "category")
    assert est.midpoint == 1925.0  # the middle of 1920.0 .. 1930.0
    assert est.width == 9


def test_sample_year_stays_in_interval():
    # Training samples a random year inside the interval; it must never leave it.
    est = DateEstimate(1920, 1929, 0.6, "category")
    rng = random.Random(0)
    for _ in range(200):
        assert 1920 <= est.sample_year(rng) <= 1930


def test_invalid_estimate_rejected():
    with pytest.raises(ValueError):
        DateEstimate(1930, 1920, 0.5, "filename")   # end before start
    with pytest.raises(ValueError):
        DateEstimate(1920, 1920, 1.5, "filename")   # confidence above 1


# ----------------------------------------------------------------------------
# REAL strings from Commons "Date" fields (Taj Mahal scene)
# ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field,year",
    [
        ("28 November 2016, 10:23:25", 2016),
        ("1849", 1849),
        ("23 September 2022, 08:05:42", 2022),
        ("19 February 2014, 21:04", 2014),
        ('Taken on 21 May 2009, 15:42"', 2009),   # note the stray trailing quote
        ("24 December 2013, 16:31:37", 2013),
        ("17 February 2025, 10:44:33", 2025),
        ("28 September 2016", 2016),
    ],
)
def test_real_commons_date_fields(field, year):
    est = parse_date_field(field)
    assert span(est) == (year, year)
    assert est.source == "date_field"


def test_bare_year_date_field_is_high_confidence():
    # If the whole Date field is just "1849", the uploader stated it outright.
    # That deserves more trust than a number found inside a filename.
    assert parse_date_field("1849").confidence > parse_date_text("Taj_1849.jpg", "filename").confidence


def test_description_lifespan_does_not_override_date_field():
    """The 1849 albumen print: its description has the photographer's lifespan
    (1825 - 1917) and a museum accession number (2013.55.12). Before we gave the
    structured Date field priority, the parser labeled this photo 1825-1917."""
    desc = (
        "Agra, Le Tadj, vue générale; Baron Alexis de La Grange (French, 1825 - 1917); "
        "negative 1849 - 1851; print 1851; Albumen silver print from a waxed paper negative; "
        "17.6 × 21.1 cm (6 15/16 × 8 5/16 in.); 2013.55.12; No Copyright"
    )
    est = estimate_image_date(
        filename="Agra, Le Tadj, vue générale.jpg", description=desc, date_field="1849"
    )
    assert span(est) == (1849, 1849)
    assert est.source == "date_field"


# More real Date fields from the first 20-image lookup.
@pytest.mark.parametrize(
    "field,year",
    [
        ("2016-07-10 16:48:37", 2016),
        ("Taken on 2 March 2010, 09:36:31", 2010),
        ("Taken on 12 March 2008", 2008),
        ("2018-08-22", 2018),
        ("1977-06-04", 1977),
        ("9 March 2023 (according to Exif data)", 2023),
    ],
)
def test_more_real_date_fields(field, year):
    assert span(parse_date_field(field)) == (year, year)


def test_upload_date_is_low_confidence():
    # Real case: "(original upload date)" is when it was posted to Panoramio,
    # not when it was taken. We keep the year but trust it much less.
    upload = parse_date_field("9 April 2008 (original upload date)")
    taken = parse_date_field("Taken on 12 March 2008")
    assert span(upload) == (2008, 2008)
    assert upload.confidence < 0.5 < taken.confidence


def test_scan_date_is_low_confidence():
    assert parse_date_field("scanned 12 May 2005").confidence < 0.5


def test_empty_date_field():
    assert parse_date_field("") is None
    assert parse_date_field("   ") is None


# ----------------------------------------------------------------------------
# REAL strings from the full audit's "oldest 15" spot check
# (each one is a bug we found by reading the real output)
# ----------------------------------------------------------------------------

# The machine-readable tail museums append. Its numbers (1500, 1850, 1900) are
# Wikidata codes. The parser must ignore them.
QS_TAIL = ",+1500-00-00T00:00:00Z/6,P1319,+1850-00-00T00:00:00Z/9,P1326,+1900-00-00T00:00:00Z/9"


def test_museum_between_range_with_machine_suffix():
    # Used to come out as just (1850, 1850).
    raw = "between 1850 and 1900 date QS:P" + QS_TAIL
    assert span(parse_date_field(raw)) == (1850, 1900)


def test_museum_between_decade():
    raw = "between 1860 and 1869 date QS:P,+1860-00-00T00:00:00Z/8,P1319,+1860-00-00T00:00:00Z/9"
    assert span(parse_date_field(raw)) == (1860, 1869)


def test_museum_circa_with_machine_suffix():
    raw = "circa 1860 date QS:P,+1860-00-00T00:00:00Z/9,P1480,Q5727902"
    assert span(parse_date_field(raw)) == (1855, 1865)


def test_negative_year_is_capture_not_print():
    # "negative 1859; print 1862": the photo was captured in 1859; the print
    # (made later) is not the capture date.
    assert span(parse_date_field("negative 1859; print 1862")) == (1859, 1859)


def test_month_range_year():
    assert span(parse_date_field("January\u2013March 1864")) == (1864, 1864)


def test_impossible_clock_timestamp_is_low_confidence():
    # A "1648" photo with a time of day: photography didn't exist, so this is a
    # broken camera clock. It must not look trustworthy.
    assert parse_date_field("1648-01-03 17:00:23").confidence < 0.5


def test_plausible_old_date_only_stays_confident():
    # The clock-time rule must not punish genuinely old dates that have no clock time.
    assert parse_date_field("1977-06-04").confidence > 0.9
    assert parse_date_field("1863").confidence > 0.8
