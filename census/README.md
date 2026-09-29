# MegaScenes date census

Which MegaScenes scenes have **old and modern photos registered in the same
COLMAP reconstruction**? Those photos share camera poses, which is what
date-conditioned SEVA needs for training and evaluation. This pipeline answers
that question per scene and per model, without downloading any photos.

```
scene name → scene ID → COLMAP models → registered images (+ poses)
           → Wikimedia capture dates → images per era, per model
```

| Step | Script | Reads | Writes |
|---|---|---|---|
| 00 | `00_rank_scenes.py` | image index | candidate scene list (optional) |
| 01 | `01_fetch_registered.py` | scene list, S3 reconstructions | `data/registered/<scene_id>.csv` |
| 02 | `02_fetch_dates.py` | registered CSVs, Wikimedia API | `data/dates.csv` |
| 03 | `03_census_table.py` | registered CSVs, dates | `data/census_table.csv` + report |

Every step is resumable: rerun it after a crash or when you add scenes, and it
skips work already done.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate            # fish: source .venv/bin/activate.fish
python -m pip install -r census/requirements.txt
brew install peak/tap/s5cmd          # S3 client (macOS)
```

### Download the metadata

Download files one by one. Don't copy `metadata/*` recursively: `subcat/`
holds a file per scene (hundreds of thousands), which hits the macOS
open-file limit and isn't needed.

```bash
cd census
mkdir -p data/MegaScenes/metadata
s5cmd --no-sign-request cp s3://megascenes/metadata/images_index.parquet data/MegaScenes/metadata/
s5cmd --no-sign-request cp s3://megascenes/metadata/categories.json data/MegaScenes/metadata/
```

## Run

From the `census/` folder:

```bash
python 00_rank_scenes.py --top 200 --out candidates.txt   # optional: pick scenes
python 01_fetch_registered.py --scenes pilot_scenes.txt   # or candidates.txt
python 02_fetch_dates.py --contact you@umd.edu
python 03_census_table.py
python 03_census_table.py --inspect Dam_Square:0          # hand-check old images
```

`02_fetch_dates.py` needs a contact email because Wikimedia asks automated
clients to identify themselves. It is sent only in the User-Agent header; you
can set `WIKIMEDIA_CONTACT` instead of passing `--contact`.

Set `CENSUS_DATA=/some/path` to keep data outside `census/data/`.

## Outputs

**`data/registered/<scene_id>.csv`**: one row per registered image in each
model: `scene, scene_id, model, image, image_id, camera_id, qw, qx, qy, qz,
tx, ty, tz`. Poses follow COLMAP's world-to-camera convention
(`x_cam = R(q) x_world + t`; camera center `= -R(q)ᵀ t`). Poses are only
comparable within one model. Scenes without a reconstruction get a
header-only file. (CSVs from the first pilot run have no pose columns; they
still work for the census.)

**`data/dates.csv`**: one row per file: `image_name, date_raw, upload_raw`.
`date_raw` is Wikimedia's DateTimeOriginal (capture date), `upload_raw` its
DateTime (upload/modification).

**`data/census_table.csv`**: images per era (`pre-1970`, `1970-1999`,
`2000+`, `undated`) for each model with at least 50 unique images.

`data/MegaScenes/` is git-ignored. The registered CSVs and `dates.csv` are a
few MB; committing them saves teammates the S3 downloads and API calls.

## Known data issues

- **Duplicate files.** MegaScenes downloads a file once per Wikimedia category
  it belongs to, and COLMAP registers each copy, so one photo can appear
  several times in a model under different category paths. The census counts
  unique file names. **Split train/test by `image_name`, never by path**, or
  copies of test images leak into training.
- **Separate models.** One scene can have dozens of reconstructions, each in
  its own coordinate frame. Models under 50 images are dropped as fragments.
- **Date quality.** The year is the earliest year in the date text. Step 03
  reports how many dates carry a flag: `upload` (an upload date, not a
  capture date), `bound` ("before 2013"), `range` ("between 1855 and 1900"),
  `circa`, `decade`, `unparsed`, `missing`.
- **Pre-1970 ≠ photograph.** Old-dated files can be paintings, engravings,
  postcards, or close-ups of details. Check with `--inspect` before trusting
  a scene.
- **No reconstruction.** Only ~100K of ~430K scenes were reconstructed. The
  Eiffel Tower is one without.

## Pilot results (20 scenes, September 2026)

- Old photos do register alongside modern ones in the main model of some
  scenes. After removing duplicates: **Dam Square model 0 has 656 unique
  pre-1970 images; Notre-Dame de Paris model 0 has 113.** Hand-checked
  samples are mostly archive photographs (Nationaal Archief, Rijksmuseum,
  Library of Congress, Fortepan) with curated dates.
- Six more scenes have a model with 20–55 pre-1970 images (Taj Mahal,
  Rothenburg, Alhambra, St. Peter's, Strasbourg, St. Paul's; pre-dedupe
  counts). Rerun step 03 for current numbers.
- Scenes that changed the most register the fewest old photos: Sagrada
  Família has 4 in its main model, Times Square none.
