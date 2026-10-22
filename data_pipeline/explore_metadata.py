"""Peek inside MegaScenes' metadata files without loading them fully.

WHY THIS FILE EXISTS
--------------------
Before writing any real pipeline code we needed to know what MegaScenes'
metadata actually contains. This was our first look. It showed that the image
index has NO date column (so dates must come from Wikimedia Commons instead)
and that categories.json maps each of the 431,035 scene names to an integer ID.

HOW TO RUN (from the repo root), after downloading the two files:
    aws s3 cp --no-sign-request s3://megascenes/metadata/images_index.parquet data/raw/metadata/
    aws s3 cp --no-sign-request s3://megascenes/metadata/categories.json data/raw/metadata/
    python data_pipeline/explore_metadata.py

WHAT IT PRINTS
 * the index's row count (about 8.7 million images) and its column names
 * the first 5 rows, so you can see what a row looks like
 * how categories.json is structured (a dict of scene name -> integer)
"""

import json

import pyarrow.parquet as pq

# Open the parquet file lazily. ParquetFile reads the footer (row count and
# schema) without pulling the 236 MB of data into memory.
pf = pq.ParquetFile("data/raw/metadata/images_index.parquet")
print("rows:", pf.metadata.num_rows)
print(pf.schema_arrow)

# Read just the first 5 rows (one small batch) and show them as a table.
print(next(pf.iter_batches(batch_size=5)).to_pandas().to_string())

# categories.json: scene name -> integer.
with open("data/raw/metadata/categories.json", encoding="utf-8") as f:
    cats = json.load(f)
print(type(cats), len(cats))
if isinstance(cats, dict):
    print(list(cats.items())[:5])
else:
    print(cats[:5])
