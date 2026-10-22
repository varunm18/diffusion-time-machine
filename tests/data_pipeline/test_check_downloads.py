"""Tests for the helpers in data_pipeline/check_downloads.py.

Run with:   pytest tests/data_pipeline/test_check_downloads.py
"""

from data_pipeline.check_downloads import compare_size, find_missing


def test_compare_size_match():
    assert compare_size((1024, 768), (1024, 768)) == "match"


def test_compare_size_scaled():
    # Half the size, same shape: just a resized copy, so K must be scaled.
    assert compare_size((512, 384), (1024, 768)) == "scaled"


def test_compare_size_rotated():
    # Width and height swapped: the EXIF-rotation case.
    assert compare_size((768, 1024), (1024, 768)) == "rotated"


def test_compare_size_mismatch():
    # A different shape entirely: the file was probably replaced after the SfM run.
    assert compare_size((1000, 1000), (1024, 768)) == "mismatch"


def test_find_missing(tmp_path):
    # Build a real folder with one file and ask about two names.
    (tmp_path / "commons" / "S" / "0" / "pictures").mkdir(parents=True)
    (tmp_path / "commons" / "S" / "0" / "pictures" / "a b.jpg").write_bytes(b"x")
    names = ["commons/S/0/pictures/a b.jpg", "commons/S/0/pictures/nope.jpg"]
    assert find_missing(names, tmp_path) == ["commons/S/0/pictures/nope.jpg"]
