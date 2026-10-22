"""Find a scene (a Wikimedia Commons category) in MegaScenes by name.

WHY THIS FILE EXISTS
--------------------
MegaScenes has 431,035 scenes. To pick one to work on we need to look it up by
name and see how many images it has in the index. That is how we found the Taj
Mahal: scene name "Taj_Mahal", ID 113197, 8,634 images in the index.

HOW TO RUN (from the repo root)
    python data_pipeline/find_scene.py Taj_Mahal

Use underscores for spaces, as Commons does. The search is a case-insensitive
"contains", so a short word like "Taj" would also match unrelated names; prefer
a longer, specific name.

WHAT IT PRINTS
 * scene names containing the query, with their integer IDs
 * how many index images belong to each matching scene
 * a few example image paths

HOW THE ID LEADS TO THE FILES
 The reconstruction for ID 113197 lives at
     s3://megascenes/reconstruct/113/197/colmap/<model number>/
 i.e. the first three digits, then the last three. A scene can be split into
 several disconnected models (the Taj Mahal has 13); model 0 is usually the
 biggest. Poses from different models are in different coordinate frames, so
 never mix them in one training sample.
"""

import json
import sys

import pyarrow.compute as pc
import pyarrow.dataset as ds

query = sys.argv[1] if len(sys.argv) > 1 else "Taj_Mahal"

# 1) Look the name up in categories.json (a dict: scene name -> integer ID).
with open("data/raw/metadata/categories.json", encoding="utf-8") as f:
    cats = json.load(f)
hits = {k: v for k, v in cats.items() if query.lower() in k.lower()}
print(f"{len(hits)} scene names containing '{query}':")
for k, v in sorted(hits.items()):
    print(f"  {v:>8}  {k}")

# 2) Count that scene's images in the big index. The filter is applied while
# reading, so only the matching rows are loaded, not all 8.7 million.
dataset = ds.dataset("data/raw/metadata/images_index.parquet")
tbl = dataset.to_table(
    columns=["cat", "subcat", "image", "image_name"],
    filter=pc.match_substring(ds.field("cat"), query),
)
df = tbl.to_pandas()
print("\nimages per scene in the index:")
print(df.groupby("cat").size().sort_values(ascending=False).head(15))
print("\nsample image paths:")
print(df["image"].head(5).to_string())
