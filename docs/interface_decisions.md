# Interface Decisions (Data pair <-> SEVA pair)

**Status (updated 2026-10-27):** the SEVA-side questions below are now answered, using the SEVA study in `docs/findings.md`. The dataset and sampler are implemented in `src/timemachine/data/` (8 views per sample: 3 inputs and 5 targets, cameras normalized like SEVA, one date interval and image type per view). Sections 1, 7 and 8 describe the original plan; where the code differs, the code wins. Items still open are marked **OPEN**.

**Why this exists:** the data pair builds the dataset and the SEVA pair builds the model and training script, in parallel. These are the decisions both sides must agree on up front, because changing them later means redoing work on both sides.

---

## 1. What one training sample contains

One sample is a set of **input views** and a set of **target views** of the same scene (about 5 of each to start).

For every view (input or target):

| Field | Meaning |
|---|---|
| `image` | RGB image, preprocessed per Section 5 |
| `K` | Camera intrinsics (fx, fy, cx, cy), rescaled to match the preprocessed image |
| `w2c` | 4x4 world-to-camera extrinsics (convention: see Section 6) |
| `year` | Decimal year used by the model (see Section 2) |
| `year_valid` | 1 if the date is known, 0 if missing (see Section 2) |
| `photo_type` | Categorical (see Section 3). **Stored now, ignored by the model in v1** |
| `month` | 1-12 or 0 for unknown. **Stored now, ignored by the model in v1** |

Sample-level fields: `scene_id`, and the list of `image_id`s for inputs and targets (for debugging and reproducibility).

## 2. Timestamps

**Stored in the dataset (keeps the uncertainty):**
`date_start` (year), `date_end` (year), `date_conf` (0 to 1), `date_source` (one of `date_field`, `filename`, `category`, `description`, `exif`, `structured_data`, `none`).

**Given to the model (one number per view):**
- `year` = a decimal year, e.g. `1923.5`.
- At training time, if `date_start != date_end`, **sample `year` uniformly from the interval** instead of always taking the midpoint. This teaches the model that "1920s" is fuzzy.
- Images with no usable date get `year_valid = 0`. The model replaces the embedding with a learned "unknown date" token. Randomly dropping the date on known images (e.g. 10-15%) during training enables classifier-free guidance on the date later.
- **Embedding (proposal, SEVA pair free to adjust):** normalize `t = (year - 1900) / 100`, then apply sinusoidal/Fourier features at several frequencies, followed by a small MLP to the conditioning dimension. This is the "higher-dimensional representation" from our proposal.
- **Valid range:** 1500 to 2100. Pre-photography material (paintings, engravings) can exist in the data, but we expect almost all useful data to be 1850+.

**Source priority:** the Commons file page's structured `Date` field first, then filename/category/description text, then EXIF only as a last-resort fallback. Free-text descriptions can contain unrelated years (e.g. an artist's lifespan like "1825 - 1917" or a museum accession number), so they must never override an explicit `Date` field.

**Known caveat for the data pair:** for scanned old photos, EXIF and some Commons "date" fields are the **scan or upload date, not the capture date**. The parser must not treat these as equally trustworthy (see confidence rules in `data_pipeline/date_parser.py`).

## 3. Photo type (stored now, model ignores in v1)

Categorical, stored as a string:

`color_digital`, `color_film`, `bw`, `painting_or_drawing`, `unknown`

- `bw` can be auto-labeled with a cheap saturation check.
- `painting_or_drawing` comes from Commons category names.
- Everything else defaults to `unknown` until we have a better signal.

Why store it now: adding a column later means reprocessing the whole dataset. The model can start using it whenever the SEVA pair is ready, with no data changes.

## 4. Month / season (stored now, model ignores in v1)

`month` is 1-12 when EXIF or a Commons category gives it, otherwise 0. Same reasoning as photo type.

## 5. Image preprocessing

- **Resolved:** SEVA works at **576x576** (SEVA study, `docs/findings.md`).
- **Resolved for now: center-crop**, because SEVA itself center-crops, and `src/timemachine/data/dataset.py` follows that. Trade-off: old photos are portrait or odd-shaped more often, so cropping can cut off content. Padding to square is the fallback if that proves to matter. Either way, `K` must be updated to match the transformation.
- **Resolved:** pixel values are scaled to [-1, 1].

## 6. Cameras and scale

- MegaScenes reconstructions are in **COLMAP format**, which stores world-to-camera poses in COLMAP's convention (OpenCV-style camera axes).
- **Resolved:** SEVA takes **camera-to-world** poses in OpenCV axes, encoded as Plucker ray maps relative to the first input view. The conversion lives in `src/timemachine/data/cameras.py` (tests in `tests/test_posing.py`), so no one converts poses by hand.
- **Scene scale:** SfM scale is arbitrary per scene. **Resolved:** SEVA re-centers on the mean camera position and scales so that a reference camera sits at distance 2 (not the "median distance 1" we first proposed), and the dataset code does the same.

## 7. How samples are composed

- About **5 input views and 5 target views** per sample, all from one scene.
- **Targets must span different dates than the inputs**, with a wide spread of date gaps (including gaps of decades). If inputs and targets usually share a date, the model learns to ignore the date. Include a minority of same-date samples so it still learns standard view synthesis.
- Pick views with enough camera overlap that the target is plausibly predictable from the inputs.

**Splits (evaluation plan from our proposal):**
- Hold out **whole scenes** to test generalization to new places.
- Within training scenes, hold out some **oldest** images (test going backward in time) and some **most recent** images (test going forward).
- No image appears in both train and eval.

## 8. On-disk format

Per scene, a folder containing:

- `images/` (original images, or preprocessed copies)
- `metadata.parquet`, one row per image, with columns:
  `image_id, scene_id, path, width, height, fx, fy, cx, cy, w2c (16 floats), registered, date_start, date_end, date_conf, date_source, photo_type, month`

A top-level `samples.parquet` (or generated on the fly by the Dataset class) lists which `image_id`s form each sample.

The PyTorch `Dataset` returns the fields from Section 1 as tensors, and nothing else is required to train.

## 8b. Data cleaning rules (learned from the Taj Mahal scene)

**Where each rule stands in `src/timemachine/data`:** duplicates are handled (one row per unique file, `drop_duplicates("file_key")`); images that fail to download are skipped; dates are filtered by confidence and interval width. **OPEN:** images whose shape disagrees with their COLMAP camera are not dropped. `scripts/build_scene_dataset.py` only prints the worst-case aspect mismatch after writing the table. On the Taj Mahal, 2 of 300 sampled images were far off (for example 1473x814 against a 1800x1350 camera), so a filter is needed.

The sample maker must apply these before an image can appear in any sample:

1. **Drop images that are missing from disk.** (5 of 5,334 in the Taj Mahal; the filenames contain a double quote, which Windows can't store.)
2. **Check each image's pixel size against its camera's calibrated size** (`data_pipeline/check_downloads.py`):
   `match` keep; `scaled` keep and rescale the intrinsics by `image_width / camera_width`; `rotated` or `mismatch` drop. (About 2 of 300 sampled images mismatched, most likely files replaced on Commons after the reconstruction was made.)
3. **Deduplicate by Commons title** before splitting into train and validation. The same file can be registered under several subcategories, and a duplicate on both sides of the split would leak.
4. **Only use dates with confidence >= 0.5 and year >= 1826** as real dates. Everything else is treated as undated (`year_valid = 0`).
5. **Never mix images from different COLMAP models** (`colmap/0`, `colmap/1`, ...) in one sample. Each model has its own coordinate frame, so their poses are not comparable.

## 9. v1 scope

**In v1:** images, cameras, decimal-year conditioning with unknown-date handling.
**Stored but ignored in v1:** `photo_type`, `month`.
**Out of scope for now:** painting-to-photo generation, future extrapolation, history probing.

## 10. Open questions checklist

- [x] SEVA pair: resolution and normalization (576x576, [-1, 1]); see Section 5
- [x] SEVA pair: pose convention (camera-to-world, OpenCV axes, Plucker rays); see Section 6
- [x] SEVA pair: scene-scale expectation (reference camera at distance 2); see Section 6
- [x] SEVA pair: training code? **No**, only inference code and weights are public, so we write our own training loop (`docs/findings.md`).
- [x] Joint: pad vs. center-crop (center-crop for now); see Section 5
- [x] Joint: date embedding and unknown-date handling are implemented in `src/timemachine/model/` (sine/cosine features plus a small network, with a learned "unknown date" input, added to each view's timestep embedding)

## Sign-off

- [ ] OPEN: filter images whose shape disagrees with their camera (Section 8b)
