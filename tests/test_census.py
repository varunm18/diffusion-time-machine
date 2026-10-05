import numpy as np
import pandas as pd

from timemachine.analysis.census import assign_era, era_counts, passes_gate


def test_assign_era_uses_midpoint_and_rejects_vague_dates():
    mid = pd.Series([1850.5, 1922.0, 1960.0, 2015.5, np.nan, 1885.0])
    width = pd.Series([1.0, 10.0, 22.0, 0.01, np.nan, 119.0])  # last: "before 1945"
    assert assign_era(mid, width).tolist() == [
        "pre-1900", "1900-1944", "1945-1969", "2000+", "undated", "undated"]


def test_era_counts_deduplicate_files():
    files = pd.DataFrame({"scene_id": [1, 1, 1], "file_key": ["a", "a", "b"],
                          "era": ["pre-1900", "pre-1900", "2000+"]})
    counts = era_counts(files, ["scene_id"]).loc[1]
    assert counts["pre-1900"] == 1 and counts["2000+"] == 1 and counts["pre-1970"] == 1


def test_gate_two_ways():
    summary = pd.DataFrame({
        "best_model": [0, None], "best_model_pre-1970": [12, 0], "best_model_2000+": [40, 0],
        "photo_pre-1970": [30, 15], "photo_2000+": [100, 50],
    })
    gated = passes_gate(summary)
    assert gated["gate_posed"].tolist() == [True, False]
    assert gated["gate_any"].tolist() == [True, True]
