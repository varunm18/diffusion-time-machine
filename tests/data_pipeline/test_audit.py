"""Tests for the summary logic in data_pipeline/audit_date_coverage.py.

`summarize` is a pure function, so we test it with a few hand-made rows instead
of a real CSV. Run with:   pytest tests/data_pipeline/test_audit.py
"""

from data_pipeline.audit_date_coverage import summarize


def row(start, end, conf, source="date_field", raw="x", title="File:a.jpg"):
    """Make one fake CSV row. Pass "" for start/end/conf to mean 'undated'."""
    return {
        "date_start": str(start) if start != "" else "",
        "date_end": str(end) if end != "" else "",
        "date_conf": str(conf) if conf != "" else "",
        "date_source": source,
        "date_raw": raw,
        "title": title,
    }


# A small scene: 12 modern photos, three old ones (1849, 1949, 1960),
# one shaky-confidence 1977 photo, and one undated photo.
ROWS = (
    [row(2016, 2016, 0.95)] * 12
    + [row(1949, 1949, 0.95), row(1960, 1960, 0.95), row(1849, 1849, 0.95)]
    + [row(1977, 1977, 0.4)]            # low confidence
    + [row("", "", "", source="")]      # undated
)


def test_counts():
    s = summarize(ROWS)
    assert s["total"] == 17
    assert s["dated"] == 16
    assert s["undated"] == 1
    assert s["pre_1900"] == 1
    assert s["pre_1950"] == 2     # 1849 and 1949
    assert s["pre_1970"] == 3     # plus 1960
    assert s["post_2000"] == 12


def test_min_conf_filters():
    # With a 0.5 cutoff the 0.4-confidence 1977 photo is dropped.
    s = summarize(ROWS, min_conf=0.5)
    assert s["usable"] == 15
    assert s["low_conf_dropped"] == 1


def test_oldest_sorted():
    s = summarize(ROWS)
    years = [int(r["date_start"]) for r in s["oldest"]]
    assert years == sorted(years)   # oldest first, so a human can spot-check them
    assert years[0] == 1849


def test_decade_histogram():
    s = summarize(ROWS)
    assert s["decades"][2010] == 12
    assert s["decades"][1840] == 1


def test_pre_photography_dropped():
    # Real case: a "1648" date. Photography started in 1826, so it's excluded.
    rows = [row(1648, 1648, 0.95), row(1860, 1860, 0.95), row(2016, 2016, 0.95)]
    s = summarize(rows, min_conf=0.5)
    assert s["pre_photo_dropped"] == 1
    assert s["usable"] == 2
    assert s["oldest"][0]["date_start"] == "1860"


def test_duplicate_titles_counted():
    # The same Commons file can be registered under several subcategories.
    rows = [row(2016, 2016, 0.95, title="File:a.jpg"),
            row(2016, 2016, 0.95, title="File:a.jpg"),
            row(2017, 2017, 0.95, title="File:b.jpg")]
    s = summarize(rows)
    assert s["unique_titles"] == 2
    assert s["duplicate_rows"] == 1
