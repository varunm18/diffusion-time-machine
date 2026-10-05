# Plan — Goal 1: Dated MegaScenes

**Goal:** know, per MegaScenes image, *when* it was made (as an interval with confidence), *what kind* of image it is, and *whether we have its camera pose*. From that, answer the go/no-go question (how many old images, how old, in which scenes, how many posed) and build a small dummy training dataset.

**Guiding principles:**
- Small single-purpose modules, so each stage can be rerun alone and from cached output.
- Raw downloads are cached; parsing is pure functions with unit tests.

---

## Steps

**1. Scene index (no image downloads)**
- Join `images_index.parquet` + `categories.json` + `recon_metadata.json` (per-reconstruction registered counts, from the MegaScenes web-viewer repo) + subcategory-name date hints.
- Output: one row per scene (image count, reconstruction sizes, rough "old-looking" count).
- Pick a candidate set, e.g. the ~4.8K scenes with ≥200 images (32% of all images). Start the pilot on ~5 scenes.

**2. Metadata harvest**
- For candidate scenes, stream every subcategory's `raw_metadata.json` from S3.
- Keep a compact per-image record: pageid, title, date field (+ hidden machine-readable date), categories, description, author, EXIF date/camera/software, upload time, license.
- Store raw JSON only for pilot scenes.
- No Commons API calls needed at this stage. The API is a later fallback for wikitext and Structured Data.

**3. Date + type parsing** (the core module)
- `DateInterval` = earliest, latest, precision, source, confidence.
- One parser per source:
  - hidden date string;
  - free-text date field ("circa 1905", "1887/1888", "between 1890 and 1900");
  - EXIF;
  - year/decade categories;
  - subcategory name;
  - title/filename.
- A resolver combines them and flags known traps: upload date typed as the date, scan dates in EXIF, reprints, modern photos inside "historical" categories.
- Photo type (photo / painting / drawing / print / postcard) from templates and categories; B&W vs color from pixels later.
- Unit tests built from the real examples we've already collected.
- **Accuracy check:** hand-label ~200–300 random images (needs a human; I'll make a quick labelling page).

**4. Pose join**
- For candidate scenes, download only `images.minibin` + `cameras.bin` per reconstruction. These hold poses without the 2D keypoints and are ~1000× smaller than `images.bin`.
- Mark each image registered/unregistered, with model id, pose and intrinsics.

**5. Analysis → `docs/findings.md`**
- Date histograms: all images vs posed images.
- Per-scene counts of pre-1900 / pre-1945 / pre-1970 / modern images.
- The proposal's gate (≥10 pre-1970 + ≥10 modern), counted **two ways**:
  1. posed in MegaScenes already;
  2. any dated image, which could get posed by our own MASt3R/VGGT step later.
- Photo-type and B&W breakdown. A ranked list of the best scenes, plus a hand-picked list of scenes with **known physical changes**: Frauenkirche Dresden, Notre-Dame (2019 fire), Brandenburg Gate (Wall), Sagrada Família (still under construction), Reichstag (1999 dome), Cologne Cathedral (completed 1880).

**6. Dummy dataset (3–5 scenes)**
- Download resized images for those scenes.
- A PyTorch `Dataset` yields SEVA-shaped samples: N input + M target views, each with image, intrinsics, camera pose (normalized the way SEVA does it), date interval, photo type, and input/target mask. Plücker maps come from a separate camera utility.
- v0 uses posed images only, so it will be mostly modern. Old unposed images are listed for the next goal (pose estimation).

---

## Proposed code layout
```
src/timemachine/
  config.py              # paths, S3 base URL, User-Agent; one place for settings
  megascenes/
    s3.py                # URL building, listing, cached/polite downloads
    index.py             # scene table from index/categories/recon metadata
    metadata.py          # raw_metadata.json -> compact per-image records
    colmap.py            # minimal COLMAP readers (cameras.bin, images.minibin/.bin)
  dating/
    interval.py          # DateInterval
    sources/             # one parser per date source
    resolve.py           # combine candidates, flag conflicts
    medium.py            # photo type / B&W
  data/
    cameras.py           # pose normalization, Plücker rays
    dataset.py           # torch Dataset for SEVA-style samples
scripts/                 # thin command-line entry points, one per step above
tests/                   # pytest, fixtures from real Commons examples
```
- The existing `generate.py` demo stays untouched.
- Data lives outside the repo, in a configurable data root.

## Out of scope for Goal 1 (next up)
- Posing old images (MASt3R-SfM / VGGT + Doppelgangers++), validated by angular error.
- Re-crawling Commons beyond MegaScenes' 4-level depth for our chosen scenes, to find more old images.
- SEVA training loop (no official training code exists).

---

## Status (2026-10-04)
- **Done (steps 1–5, pilot of 42 scenes):** scene table, metadata harvest from S3 (no API calls), date + type parsing with tests, pose join via `images.minibin`, census with the gate counted both ways. Results in `docs/findings.md`.
- **Changes agreed with the team since the plan:**
  - Dummy dataset → **a single scene with many dates first**; the model must overfit it before scaling.
  - Pose old images with **VGGT-Omega**.
  - Targets in one sample share one date.
  - A VLM joins the pipeline later (curation, spotting real scene change).
- **Next:**
  1. Run the census at scale (`slurm/census_cpu.sbatch`).
  2. Pick the single scene. Proposal: Notre-Dame, with real structural change (no spire before 1859, spire 1859–2019, fire 2019) and old + modern photos posed in one model.
  3. Download its images and build the PyTorch dataset with SEVA's camera conventions.
  4. Pose its unposed old photos with VGGT-Omega.
  5. Hand-label ~200 images to measure date accuracy.
