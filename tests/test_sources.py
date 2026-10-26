"""Unit tests for the individual date-evidence parsers."""
import pytest

from timemachine.dating.interval import DateInterval, Precision
from timemachine.dating.sources.categories import is_maintenance_category, parse_category
from timemachine.dating.sources.exif import parse_exif
from timemachine.dating.sources.qs import parse_qs
from timemachine.dating.sources.text import parse_date_text
from timemachine.dating.sources.title import parse_title

Y = DateInterval.from_years


# --- hidden QS date -------------------------------------------------------------
@pytest.mark.parametrize("qs, expected, precision, flag", [
    ("P571,+1851-00-00T00:00:00Z/9,P1480,Q5727902", Y(1846, 1856), Precision.YEAR, "circa"),
    ("P,+1950-00-00T00:00:00Z/7,P1319,+1900-00-00T00:00:00Z/9,P1326,+1930-00-00T00:00:00Z/9",
     Y(1900, 1930), Precision.RANGE, None),
    ("P,+1920-00-00T00:00:00Z/8", Y(1920, 1929), Precision.DECADE, None),
    ("P,+1850-00-00T00:00:00Z/7", Y(1801, 1900), Precision.CENTURY, None),
    ("P571,+1934-05-12T00:00:00Z/11", DateInterval.from_date(1934, 5, 12), Precision.DAY, None),
])
def test_qs(qs, expected, precision, flag):
    c = parse_qs(qs)
    assert c.interval == expected and c.precision == precision
    assert flag is None or flag in c.flags


def test_qs_before_is_open_bound():
    c = parse_qs("P,+1915-00-00T00:00:00Z/7,P1326,+1915-00-00T00:00:00Z/9")
    assert c.precision == Precision.BOUND and c.interval.hi == 1916 and "open_start" in c.flags


def test_qs_garbage():
    assert parse_qs(None) is None and parse_qs("") is None and parse_qs("P571,Q42") is None


# --- free-text dates -----------------------------------------------------------
@pytest.mark.parametrize("text, expected, precision", [
    ("1878", Y(1878, 1878), Precision.YEAR),
    ("1887/1888", Y(1887, 1888), Precision.RANGE),
    ("between 1890 and 1900", Y(1890, 1900), Precision.RANGE),
    ("2007-01-09 00:05", DateInterval.from_date(2007, 1, 9), Precision.DAY),
    ("Taken on 7 July 2014, 11:15", DateInterval.from_date(2014, 7, 7), Precision.DAY),
    ("June 2015", DateInterval.from_date(2015, 6), Precision.MONTH),
    ("circa 1905", Y(1900, 1910), Precision.YEAR),
    ("um 1930", Y(1925, 1935), Precision.YEAR),
    ("1920s", Y(1920, 1929), Precision.DECADE),
    ("19th century", Y(1801, 1900), Precision.CENTURY),
    ("before 19 June 2013", DateInterval(1000, 2014), Precision.BOUND),
])
def test_date_text(text, expected, precision):
    c = parse_date_text(text)
    assert c.interval == expected and c.precision == precision


def test_date_text_unknown_and_upload():
    assert parse_date_text("Unknown date") is None
    assert parse_date_text("") is None
    assert "upload_date" in parse_date_text("2005-12-24 (original upload date)").flags


# --- EXIF ----------------------------------------------------------------------
def test_exif_camera_is_very_confident():
    c = parse_exif({"DateTimeOriginal": "2022:11:15 17:34:28", "Make": "Sony", "Model": "SO-03K"})
    assert c.interval == DateInterval.from_date(2022, 11, 15) and c.confidence > 0.9 and not c.flags


@pytest.mark.parametrize("exif, flag", [
    ({"DateTimeOriginal": "2017:06:17 03:29:29", "Make": "Phase One", "Model": "P 65+"}, "scanner"),
    ({"DateTimeOriginal": "2012:01:01 10:00:00", "Model": "HP pst_p04h"}, "scanner"),
    ({"DateTimeOriginal": "2000:01:01 00:00:00"}, "placeholder"),
    ({"DateTimeDigitized": "2011:01:06 14:00:50", "Software": "Adobe Photoshop"}, "digitized"),
    ({"DateTimeOriginal": "1931:05:01 12:00:00"}, "pre_digital"),
])
def test_exif_flags(exif, flag):
    assert flag in parse_exif(exif).flags


def test_exif_invalid():
    assert parse_exif({"DateTimeOriginal": "0000:00:00 00:00:00"}) is None
    assert parse_exif({}) is None


# --- categories ----------------------------------------------------------------
@pytest.mark.parametrize("category, expected, precision", [
    ("1930 photographs of Dresden", Y(1930, 1930), Precision.YEAR),
    ("1898 in Dresden", Y(1898, 1898), Precision.YEAR),
    ("Notre-Dame de Paris in 2016", Y(2016, 2016), Precision.YEAR),
    ("Dresden photographs taken on 2012-10-07", DateInterval.from_date(2012, 10, 7), Precision.DAY),
    ("Brandenburg Gate in July 1945", DateInterval.from_date(1945, 7), Precision.MONTH),
    ("Dresden in the 1890s", Y(1890, 1899), Precision.DECADE),
    ("Frauenkirche_Dresden_(before_1945)", DateInterval(1000, 1945), Precision.BOUND),
])
def test_dated_categories(category, expected, precision):
    c = parse_category(category)
    assert c.interval == expected and c.precision == precision


@pytest.mark.parametrize("category", [
    "Cutty Sark (ship, 1869)", "Covent Garden Theatre (1732 building)", "Buildings completed in 1726",
    "Taken with Olympus C-900Z / D-400Z", "Uploaded in 2015", "Exterior of the Frauenkirche, Dresden",
])
def test_undated_or_misleading_categories(category):
    assert parse_category(category) is None


def test_postcard_years_are_flagged_as_publication():
    assert "publication" in parse_category("2000 postcards of Dresden").flags


def test_maintenance_categories():
    assert is_maintenance_category("Pages with maps")
    assert is_maintenance_category("CC-BY-SA-4.0")
    assert not is_maintenance_category("House of Commons of the United Kingdom")
    assert not is_maintenance_category("Paintings of Dresden")


# --- titles --------------------------------------------------------------------
@pytest.mark.parametrize("title, expected", [
    ("Frauenkirche um 1897.jpg", Y(1892, 1902)),
    ("2007-04-29 Dresden 05.jpg", DateInterval.from_date(2007, 4, 29)),
    ("20230306.Frauenkirche (Dresden).-014.jpg", DateInterval.from_date(2023, 3, 6)),
    ("Dresden Brühlsche Terrasse 1900.jpg", Y(1900, 1900)),
])
def test_title(title, expected):
    assert parse_title(title).interval == expected


@pytest.mark.parametrize("title", [
    "Caution, cheetah crossing, Northern Cape (6252718221).jpg",  # Flickr id
    "IMG_2034.JPG",
    "Paris 1900 and 1925 compared.jpg",                            # ambiguous
])
def test_title_ignores_noise(title):
    assert parse_title(title) is None


def test_qs_malformed_month_falls_back_to_year():
    c = parse_qs("P571,+1934-13-00T00:00:00Z/10")
    assert c.interval == Y(1934, 1934) and c.precision == Precision.YEAR
