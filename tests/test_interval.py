import pytest

from timemachine.dating.interval import DateInterval, decimal_year
import datetime as dt


def test_years_are_inclusive_and_half_open():
    decade = DateInterval.from_years(1920, 1929)
    assert (decade.lo, decade.hi) == (1920.0, 1930.0)
    assert decade.width == 10
    assert str(decade) == "1920-1929"
    assert str(DateInterval.from_years(1900, 1900)) == "1900"


def test_day_and_month_resolution():
    day = DateInterval.from_date(2014, 7, 7)
    assert 2014.51 < day.lo < day.hi < 2014.52
    december = DateInterval.from_date(1999, 12)
    assert december.hi == pytest.approx(2000.0)


def test_decimal_year_handles_leap_years():
    assert decimal_year(dt.date(2000, 1, 1)) == 2000.0
    assert decimal_year(dt.date(2000, 12, 31)) == pytest.approx(2000 + 365 / 366)


def test_set_operations():
    a, b = DateInterval.from_years(1900, 1930), DateInterval.from_years(1920, 1940)
    assert a.overlaps(b)
    assert a.intersect(b) == DateInterval.from_years(1920, 1930)
    assert a.intersect(DateInterval.from_years(1950, 1960)) is None
    assert a.clip(lo=1910) == DateInterval(1910, 1931)
    assert a.clip(hi=1890) is None


def test_empty_interval_rejected():
    with pytest.raises(ValueError):
        DateInterval(1900, 1900)
