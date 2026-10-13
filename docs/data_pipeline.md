# Data Pipeline: Overview and Handoff

This document explains what the data pipeline does, how to run it, what we found
on the first scene (the Taj Mahal), and what is left to build. Read this first
if you are picking up where the first half left off.

## The goal in one paragraph

Our model is trained on photos of the same place taken in different years, so
every training image needs three things: the **pixels**, the **camera pose**
(where it was taken from), and the **year** it was taken. MegaScenes gives us
images and poses but **no dates**. The dates come from Wikimedia Commons, where
every MegaScenes photo has a file page with a "Date" field. The pipeline joins
these three sources and cleans the result.

## How the pieces fit together

```
 MegaScenes S3 bucket                                    Wikimedia Commons API
   metadata/images_index.parquet  (which images exist)     (the "Date" field, categories)
   metadata/categories.json       (scene name -> ID)                |
   reconstruct/<aaa>/<bbb>/colmap/<n>/  (camera poses)              |
   images/<aaa>/<bbb>/commons/...       (the photos)                |
            |                                                       |
            v                                                       v
   find_scene.py  ---> pick a scene          commons_dates.py  --> image_dates.csv
   colmap_reader.py ---> read poses                  ^
   inspect_reconstruction.py ---> sanity check       |
            |                                  date_parser.py (the date logic)
            v
   check_downloads.py ---> do the files exist, do sizes match the cameras?
   audit_date_coverage.py ---> enough old AND modern photos?
   plot_poses.py ---> are old photos at believable camera positions?
            |
            v
   [NOT BUILT YET]  sample maker + PyTorch Dataset  ---> SEVA training
```

## Running it end to end (Taj Mahal example)

All commands run from the repo root. `data/` is git-ignored.

```powershell
# 1. Get the metadata (about 255 MB) and look at it
aws s3 cp --no-sign-request s3://megascenes/metadata/images_index.parquet data\raw\metadata\
aws s3 cp --no-sign-request s3://megascenes/metadata/categories.json data\raw\metadata\
python data_pipeline/explore_metadata.py

# 2. Find the scene. Taj_Mahal has ID 113197 -> shard 113 / 197
python data_pipeline/find_scene.py Taj_Mahal

# 3. Download its biggest reconstruction (model 0, about 626 MB) and all its images (2.8 GB)
aws s3 cp --no-sign-request s3://megascenes/reconstruct/113/197/colmap/0/ data\raw\taj\colmap\0\ --recursive
aws s3 sync --no-sign-request s3://megascenes/images/113/197/commons/ data\raw\taj\images\commons\

# 4. Sanity-check the reconstruction against the index
python data_pipeline/inspect_reconstruction.py data/raw/taj/colmap/0 Taj_Mahal

# 5. Look up every image's date on Commons (try --limit 20 first)
python data_pipeline/commons_dates.py data/raw/taj/colmap/0 --contact you@example.com

# 6. Audit: enough old and modern photos?  (read the oldest 15 by hand!)
python data_pipeline/audit_date_coverage.py data/processed/image_dates.csv --min-conf 0.5

# 7. Verify the downloaded files (use a big --sample to check every image)
python data_pipeline/check_downloads.py data/raw/taj/colmap/0 data/raw/taj/images --sample 6000

# 8. Are the old photos at believable camera positions?
python data_pipeline/plot_poses.py data/raw/taj/colmap/0 data/processed/image_dates.csv
```

## How MegaScenes is organized (things we had to figure out)

* A **scene** is a Wikimedia Commons category (431,035 of them). `categories.json`
  maps each scene name to an integer ID.
* The ID gives the S3 location: ID `113197` lives at `reconstruct/113/197/` (first
  three digits, then last three). Not every ID has a reconstruction.
* A scene can be split into several **disconnected COLMAP models** (`colmap/0`,
  `colmap/1`, ...). The Taj Mahal has 13. Model 0 is usually the largest. Each
  model has its own coordinate frame, so never mix them in one sample.
* Images in a model are named like `commons/<subcategory>/0/pictures/<file>`. The
  folder is the *subcategory*, not the scene name.
* The index (`images_index.parquet`, 8.7 million rows) has **no dates**.

## What we found on the Taj Mahal (model 0)

| Fact | Value |
|---|---|
| Registered images / cameras | 5,334 / 3,560 (several images share a camera) |
| Images found on disk | 5,329 (5 have a `"` in the filename; Windows can't save them) |
| Images with a date estimate | 5,256 (78 undated) |
| Usable (confidence >= 0.5, year >= 1826) | 5,193 |
| Before 1900 / 1950 / 1970 | 31 / 40 / 55 |
| 2000 or later | 5,123 (about 96% of the scene) |
| Size mismatches vs. camera calibration | 2 of 300 sampled |
| Old photos near modern camera positions | 39 of 40 inside the modern distance range |

### What this means

* The proposal's threshold (10+ before 1970 and 10+ modern) is **met**, but the
  data is extremely lopsided. Almost everything is from the 2010s.
* The old photos come in **two small clusters**: about 31 from the 1850s to 1890s
  (mostly museum scans: Rijksmuseum, the Met) and about 22 from the 1960s and
  1970s. There are only about 9 between 1900 and 1950.
* In this scene, **"old" almost always means "albumen or black-and-white print"**.
  So the model could learn to associate date with *photo style* instead of how
  the place changed. That is exactly the confound our proposal wants to separate
  and the reason `photo_type` is stored in the schema.
* The Taj Mahal barely changed since the 1860s, so it is a good scene for building
  and debugging the pipeline but a weak test of whether the model learns change.
* Two old photos sit very close to the scene center while modern cameras form a
  tight ring around it (radius 0.59 and 0.83 vs. a modern median of 4.27). Worth
  opening by hand; they may have bad poses.

## Hard-won rules (each is covered by a test)

1. The structured Commons **Date field outranks free text.** Descriptions can
   contain unrelated years (an artist's lifespan, an accession number).
2. **EXIF and "upload date" are not capture dates.** EXIF is only a last-resort
   fallback; dates mentioning upload or scan get low confidence.
3. A **clock time next to an early year** (e.g. `1648-01-03 17:00:23`) is a broken
   camera clock.
4. Museum date strings end in machine-readable junk (`date QS:P,+1500-...`) whose
   numbers are Wikidata codes, not years. It is stripped before parsing.
5. In `negative 1859; print 1862`, the capture date is the negative's.
6. COLMAP names use underscores where the index uses spaces. Commons treats them
   as the same character, so normalize both.
7. The same Commons file can appear under several subcategories. **Deduplicate
   by title before splitting train and validation.**

## What is left to build (suggested order)

1. **Sample maker + PyTorch Dataset.** Follow `docs/interface_decisions.md`
   (Sections 1, 7, 8, 8b). Groups of about 5 input and 5 target views from one
   COLMAP model, with targets at *different dates* than inputs and a wide spread
   of date gaps. Apply the cleaning rules in Section 8b first.
2. **Pose convention conversion** (`colmap_to_seva`) with unit tests. Needs the
   SEVA pair to say what convention SEVA expects.
3. **Photo type labels** (`bw`, `painting_or_drawing`, ...): a cheap saturation
   check for black and white, Commons categories for paintings.
4. **Month extraction** for the seasonal stretch goal (the raw date text is
   already saved in `image_dates.csv`, so this can be re-parsed later).
5. **Scene screening.** Taj Mahal is a poor test of "change over time". Sample
   about 300 image dates for each of the biggest few hundred scenes and rank
   them by how many old images they have. Scenes with lots of museum or archive
   uploads (Rijksmuseum, the Met, Library of Congress) tend to have the old
   photos.
6. **Filter extreme image shapes.** Camera widths range from 274 to 38,288 pixels
   (panoramas). Plan to drop tiny images and extreme aspect ratios before
   padding to square.

## Known limitations

* The date parser reads **years only**. Month and day are discarded (the raw text
  is kept).
* The intrinsics matrix ignores lens distortion.
* `commons_dates.py` was validated on 20 live images and then the full scene, but
  the Commons API's exact response format could change.
* Windows: very long paths and a few illegal filename characters cause download
  problems. Short folder paths help.
