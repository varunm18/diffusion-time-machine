import numpy as np
import pandas as pd

from timemachine.data.selection import anchor_counts, assign_role, era_of, model_row


def test_era_of_boundaries():
    assert era_of(np.array([1899.9, 1900.0, 1944.9, 1945.0, 1969.9, 1970.0, 2000.0])).tolist() == [
        "pre-1900", "1900-1944", "1900-1944", "1945-1969", "1945-1969", "1970-1999", "2000+"]


def test_spread_out_old_photos_are_not_anchors():
    # 10 old photos, 11+ years apart: no 5 of them fall within +-5 years of each other.
    old = np.arange(1860.0, 1970.0, 11.0)
    modern = np.full(12, 2015.5)
    counts = anchor_counts(np.concatenate([old, modern]))
    assert sum(counts[e] for e in ("pre-1900", "1900-1944", "1945-1969")) == 0
    assert counts["2000+"] == 12


def test_clustered_old_photos_are_anchors():
    old = np.array([1930.0, 1931.0, 1932.0, 1933.0, 1934.0, 1880.0])  # five within +-5 years
    counts = anchor_counts(np.concatenate([old, np.full(10, 2010.2)]))
    assert counts["1900-1944"] == 5 and counts["pre-1900"] == 0


def test_modern_tolerance_is_one_year():
    # Five modern photos two years apart never share a date within +-1 year.
    counts = anchor_counts(np.array([2001.0, 2003.0, 2005.0, 2007.0, 2009.0, 2011.0, 2013.0, 2015.0]))
    assert counts["2000+"] == 0


def test_assign_role():
    assert assign_role(n_old=9, n_modern=500, old_anchors=3) == "excluded"
    assert assign_role(n_old=50, n_modern=9, old_anchors=20) == "excluded"
    assert assign_role(n_old=12, n_modern=40, old_anchors=0) == "old_input"
    assert assign_role(n_old=12, n_modern=40, old_anchors=1) == "old_target"


def test_model_row_wide_tolerance_finds_more_anchors():
    # Old photos 4 years apart: at most 3 within +-5 years, but 5 within +-10 years.
    old = np.arange(1900.0, 1940.0, 4.0)                        # 10 photos
    views = pd.DataFrame({"date_mid": np.concatenate([old, np.full(15, 2012.3)])})
    row = model_row(views)
    assert row["old_views"] == 10 and row["modern_views"] == 15
    assert row["old_anchors"] == 0 and row["old_anchors_wide"] > 0
    assert row["role"] == "old_input"


def test_model_row_empty_views():
    row = model_row(pd.DataFrame({"date_mid": pd.Series([], dtype=float)}))
    assert row["views"] == 0 and row["role"] == "excluded"
