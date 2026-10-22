"""Tests for the pure helper functions in data_pipeline/commons_dates.py.

The network part of that script (talking to the Commons API) isn't tested here.
Instead we feed `parse_batch` a MOCKED API response shaped like the real one.
That mock was written from the API's documentation, and the live behavior was
then confirmed by running the script on 20 real images (it worked).

Run with:   pytest tests/test_commons_dates.py
"""

from data_pipeline.commons_dates import (
    chunks,
    parse_batch,
    strip_html,
    title_from_colmap_name,
)
from data_pipeline.date_parser import estimate_image_date


def test_title_from_colmap_name():
    name = "commons/Water_reflections_at_the_Taj_Mahal/0/pictures/Taj_Mahal,_Agra_19.jpg"
    # Folders are dropped and underscores become spaces, as Commons expects.
    assert title_from_colmap_name(name) == "File:Taj Mahal, Agra 19.jpg"


def test_title_keeps_folders_out():
    assert title_from_colmap_name("a/b/c/x_y.JPG") == "File:x y.JPG"


def test_strip_html():
    raw = '<time class="dtr" datetime="2016-11-28T10:23:25">28 November 2016, 10:23:25</time>'
    assert strip_html(raw) == "28 November 2016, 10:23:25"
    assert strip_html("Caf&eacute; &amp; bar") == "Café & bar"   # HTML entities are decoded
    assert strip_html(None) == ""


def test_chunks():
    # Requests are limited to 50 titles; this is the splitter (here, size 2).
    assert list(chunks(list("abcde"), 2)) == [["a", "b"], ["c", "d"], ["e"]]


# A fake API response. It includes the three situations parse_batch must handle:
#   * a title the API "normalized" (first letter capitalized),
#   * a title that is a redirect to a renamed file,
#   * a file that doesn't exist (marked "missing").
MOCK = {
    "query": {
        "normalized": [{"from": "File:taj one.jpg", "to": "File:Taj one.jpg"}],
        "redirects": [{"from": "File:Old name.jpg", "to": "File:New name.jpg"}],
        "pages": [
            {
                "title": "File:Taj one.jpg",
                "imageinfo": [{"extmetadata": {
                    "DateTimeOriginal": {"value": '<time datetime="1849">1849</time>'},
                    "DateTime": {"value": "2013-05-01 10:00:00"},
                    "Categories": {"value": "Taj Mahal|Albumen prints"},
                    "ImageDescription": {"value": "<p>Agra, Le Tadj (French, 1825 - 1917)</p>"},
                }}],
            },
            {
                "title": "File:New name.jpg",
                "imageinfo": [{"extmetadata": {
                    "DateTimeOriginal": {"value": "28 September 2016"},
                }}],
            },
            {"title": "File:Gone.jpg", "missing": True},
        ],
    }
}


def test_parse_batch_follows_normalization_and_redirects():
    req = ["File:taj one.jpg", "File:Old name.jpg", "File:Gone.jpg", "File:Never asked.jpg"]
    out = parse_batch(MOCK, req)

    # Normalized title: found, with HTML stripped and categories split on "|".
    a = out["File:taj one.jpg"]
    assert a["found"] and a["date_raw"] == "1849"
    assert a["categories"] == ["Taj Mahal", "Albumen prints"]
    assert a["description"] == "Agra, Le Tadj (French, 1825 - 1917)"
    assert a["upload_datetime"] == "2013-05-01 10:00:00"   # saved, but never used as a capture date

    # The redirect was followed to the new name's data.
    assert out["File:Old name.jpg"]["date_raw"] == "28 September 2016"

    # Missing file, and a title we asked about that the API never returned.
    assert out["File:Gone.jpg"]["found"] is False
    assert out["File:Never asked.jpg"]["found"] is False


def test_end_to_end_date_for_old_print_ignores_upload_time_and_lifespan():
    # Whole chain on the 1849 print: the Date field "1849" must beat both the
    # 2013 upload time and the "1825 - 1917" lifespan in the description.
    rec = parse_batch(MOCK, ["File:taj one.jpg"])["File:taj one.jpg"]
    est = estimate_image_date(
        filename="taj one.jpg",
        categories=rec["categories"],
        description=rec["description"],
        date_field=rec["date_raw"],
    )
    assert (est.start_year, est.end_year) == (1849, 1849)
    assert est.source == "date_field"
