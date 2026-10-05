"""Unit tests for the candidate resolver, on synthetic candidates."""
from timemachine.dating.interval import Confidence, DateCandidate, DateInterval, Precision, Source
from timemachine.dating.resolve import resolve

Y = DateInterval.from_years
D = DateInterval.from_date


def cand(interval, source, confidence, precision=Precision.YEAR, flags=()):
    return DateCandidate(interval, source, precision, confidence, "test", tuple(flags))


def test_page_date_and_exif_a_day_apart_agree():
    page = cand(D(2005, 8, 21), Source.DATE_TEXT, Confidence.HIGH, Precision.DAY)
    exif = cand(D(2005, 8, 20), Source.EXIF, Confidence.VERY_HIGH, Precision.DAY)
    result = resolve([page, exif], upload_time="2006-01-01T00:00:00Z")
    assert "exif_agrees" in result.flags and "needs_review" not in result.flags
    assert result.confidence >= Confidence.VERY_HIGH


def test_years_apart_is_a_conflict():
    page = cand(Y(2011, 2011), Source.DATE_TEXT, Confidence.HIGH)
    exif = cand(D(2004, 2, 22), Source.EXIF, Confidence.VERY_HIGH, Precision.DAY)
    result = resolve([page, exif], upload_time="2012-01-01T00:00:00Z")
    assert "needs_review" in result.flags and result.confidence < Confidence.HIGH


def test_upload_date_typed_as_date_is_dropped_for_historical_images():
    page = cand(D(2019, 4, 20), Source.DATE_TEXT, Confidence.HIGH, Precision.DAY)
    era = cand(Y(1910, 1919), Source.CATEGORY, 0.45, Precision.DECADE)
    result = resolve([page, era], upload_time="2019-04-20T23:44:00Z", historical_hint=True)
    assert result.interval == Y(1910, 1919) and "date_is_upload_date" in result.flags


def test_same_day_upload_of_a_modern_photo_is_kept():
    page = cand(D(2022, 11, 15), Source.DATE_TEXT, Confidence.HIGH, Precision.DAY)
    exif = cand(D(2022, 11, 15), Source.EXIF, Confidence.VERY_HIGH, Precision.DAY)
    result = resolve([page, exif], upload_time="2022-11-15T20:00:00Z")
    assert int(result.interval.lo) == 2022 and "date_is_upload_date" not in result.flags


def test_exif_long_after_historical_evidence_is_digitization():
    year = cand(Y(1900, 1900), Source.DATE_TEXT, Confidence.HIGH)
    exif = cand(D(1999, 5, 9), Source.EXIF, Confidence.VERY_HIGH, Precision.DAY)
    result = resolve([year, exif], upload_time="2008-01-01T00:00:00Z")
    assert result.interval == Y(1900, 1900) and int(result.digitized.lo) == 1999


def test_candidates_after_upload_are_impossible():
    late = cand(Y(2020, 2020), Source.TITLE, Confidence.VERY_LOW)
    ok = cand(Y(2010, 2010), Source.CATEGORY, Confidence.MEDIUM)
    result = resolve([late, ok], upload_time="2015-06-01T00:00:00Z")
    assert result.interval == Y(2010, 2010) and "impossible:title" in result.flags


def test_bounds_narrow_consistent_results():
    decade = cand(Y(1940, 1949), Source.CATEGORY, 0.45, Precision.DECADE)
    before = cand(DateInterval(1000, 1945), Source.CATEGORY, Confidence.LOW, Precision.BOUND, ("open_start",))
    result = resolve([decade, before], floor_year=1826)
    assert result.interval == DateInterval(1940, 1945)


def test_artwork_exif_is_never_the_creation_date():
    exif = cand(D(2018, 6, 20), Source.EXIF, Confidence.MEDIUM, Precision.DAY)
    era = cand(Y(1900, 1909), Source.CATEGORY, 0.45, Precision.DECADE)
    result = resolve([exif, era], artwork=True)
    assert result.interval == Y(1900, 1909) and "exif_is_digitization" in result.flags
