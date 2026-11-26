# Findings Log

Running notes on what we learn as the project moves forward. Newest sections go at the bottom of each topic. Numbers come from our own scripts unless a source is cited.

---

## MegaScenes: what's actually in the bucket (2026-10-04)

**Layout**
- The data is on a public S3 bucket (`s3://megascenes`, region `us-west-2`). No account needed, so plain HTTP works.
- Top-level folders: `images/`, `reconstruct/` (COLMAP 3D models), `reconstruct_aux/`, `databases/` (COLMAP feature databases), `metadata/`, `nvs_checkpoints/`.
- Every scene has a numeric id. Scene `110925` lives at `images/110/925/`, `reconstruct/110/925/`, etc.
- `metadata/categories.json` maps scene name to scene id (431K scenes). Example: `Frauenkirche_(Dresden)` → `110925`.
- `metadata/images_index.parquet` (247 MB) lists **8.7M images** in 425K scenes: scene, subcategory, file path, width/height, license.
- `metadata/subcat/XXX/YYY/subcats.json` holds each scene's Commons subcategory tree, as it looked when MegaScenes was crawled.
- Heavy files to avoid unless needed: `databases/main/.../database.db` (~1 GB per big scene) and `descriptors.db.gz` (~2 GB per big scene).

**No dates anywhere**
- The index has **no date field** and no description or EXIF. Capture dates have to come from Wikimedia Commons itself (API or dumps).

**Scene sizes**
- Most scenes are tiny: median 6 images, 75th percentile 15.
- Only **533 scenes have ≥1000 images**, 1,319 have ≥500, and 4,818 have ≥200.
- Biggest: Rothenburg ob der Tauber (18K), Eiffel Tower (14K), Notre-Dame de Paris (12.9K, filed as `Cathédrale_Notre-Dame_de_Paris`), Central Park, Alhambra, Taj Mahal.

**Subcategory names already contain dates, for free**
- Many subcategories carry a date in the name: `Brandenburg_Gate_in_the_1950s`, `Notre-Dame_de_Paris_in_2016`, `Frauenkirche_Dresden_(before_1945)`, `Historical_images_of_...`.
- Counting only these names (strict patterns, no API calls), about **72K images look old** (pre-1970 or in a "historical / black and white / old postcards / in art" subcategory).
- Scenes with ≥50 such old images: **~340**. With ≥100: **~150**.
- This is a **lower bound**. Most old photos sit in generic subcategories like `Exterior_of_X`, and only their own Commons metadata will date them.
- The first, looser keyword pass gave inflated numbers: "histor" matched `Natural_History_Museum`, and ship names like `(ship, 1869)` looked like dates. Patterns must be strict.

**Whether old photos get poses varies a lot by scene** *(corrected 2026-10-04)*
- Bad case: Dresden Frauenkirche (destroyed 1945, rebuilt 2005). 2,850 images; 1,900 (67%) registered in 11 COLMAP models.
  - `Frauenkirche_Dresden_(before_1945)`: **0 of 103** registered. `Ruin_of_the_Frauenkirche`: **4 of 117**. `Vorgängerbau` (the earlier building): **0 of 37**.
  - Modern year subcategories (e.g. `..._in_2012`, `..._in_2011`) are **~95–100%** registered.
- Good cases, from Kevin's 20-scene pilot (`kevin-calderon` branch, `census/`):
  - **Dam Square model 0 has 656 unique pre-1970 registered images**; Notre-Dame model 0 has 113 (vs 3,133 from 2000+).
  - Taj Mahal, Alhambra, St. Peter's, St. Paul's, Rothenburg, Tower Bridge and Strasbourg each have a model with ~17–47 pre-1970 images. Some of those counts are before removing duplicates.
- Pattern: **scenes that changed least register old photos fine; scenes that changed most register almost none.** Sagrada Família has 4 in its main model, Times Square 0, Frauenkirche 0. Those changed scenes are exactly the interesting ones for us.
- So we still need our own pose estimation (**VGGT-Omega**, already installed at `/vulcanscratch/ltahboub/vggt-omega`), mainly to bring in the old photos of *changed* scenes.

**Duplicate files are common** *(verified 2026-10-04)*
- MegaScenes stores a file once per subcategory it appears in, so the same Commons file shows up under several paths.
- Overall, 661K of 8.7M index rows (7.6%) are extra copies. In big scenes it's much worse: Notre-Dame 32%, Rothenburg 35%, Alhambra 34%, Sagrada Família 35%, Mong Kok 51%, Agha Bozorg Mosque 80%.
- **COLMAP registers every copy.** Frauenkirche model 1 has 1,297 registered entries but only 1,002 unique files (295 copies). That's also why some images showed up in two models.
- 303K files (3.8%) appear in more than one scene.
- After removing duplicates, scenes with ≥500 images drop from 1,319 to 1,208.
- **Rules:** count, sample and split train/test by unique file, never by path. Never let a sample's input and target be the same file (trivial copying). Near-duplicates (same photo re-uploaded under another name, crops) won't be caught by name and need image hashing later.

**Technical gotchas**
- COLMAP image names (and S3 keys) replace spaces with underscores. Index: `2007-04-29 Dresden 05.jpg`; COLMAP: `2007-04-29_Dresden_05.jpg`. Normalize before joining (also Unicode NFC, as Kevin found).

**Commons has changed since the MegaScenes crawl**
- Several categories were reorganized after MegaScenes was built. `Brandenburg_Gate_in_the_1950s` now holds only per-year subcategories, and `Historical_photographs_of_Notre-Dame_de_Paris` is now empty.
- So: treat MegaScenes' `subcats.json` as a frozen snapshot, and look up each file's metadata on Commons **by filename**. Expect some files to have been renamed or deleted since.

**Compute available**
- Slurm accounts: `nexus`, `scavenger`, `vulcan`, and `vulcan-jbhuang` (which includes H200 partitions).

---

## SEVA (Stable Virtual Camera): what matters for us (2026-10-04)

**Training code**
- **SEVA's training code is not public.** Stability says company policy prevents releasing it. Only inference code and weights are out. **We'll write our own training loop.**
- Partial help: an unmerged community PR (#51 on the SEVA repo) has a full training pipeline with known bugs. VMem (ICCV'25) published a SEVA LoRA recipe but no code: rank 256, 8 frames, lr 3e-6, 8× A40, which have the same memory as our A6000s.
- SEVA has no gradient checkpointing. We'll need to add it (a one-file patch exists in a fork) to fine-tune on 48 GB cards.

**Model basics**
- Built on Stable Diffusion 2.1-base. 1.26B params: 866M from SD plus 398M of new multi-view attention.
- Works at 576×576 and handles 21 views per forward pass (input and target views mixed). For training on A6000s, 8 views is the realistic setting.
- Inputs per view: the image latent, an input/target mask, and a Plücker camera ray map. A CLIP image embedding is **averaged over all input views**.
- Camera poses are re-centered and scaled so one "source" camera sits at distance 2. That suits MegaScenes, whose COLMAP scale is arbitrary.

**Licensing and downloads**
- Non-commercial license (research and fine-tunes are OK). Weights are gated on Hugging Face (auto-approve) and are ~5 GB each.
- **Heads-up:** the hard-coded SD 2.1-base VAE download now fails (HF returns 401; the repo seems removed). We'll need a mirror or a local copy.

**What SEVA was trained on**
- Training data is **undisclosed**. Nothing suggests MegaScenes or internet photo collections. It was most likely trained on clean video-like captures where every view has the same lighting and look.
- So SEVA has a strong habit of **copying the input views' appearance** into the targets. Our date conditioning has to fight exactly that.

**Where a per-view date can go** (each view gets its own date: inputs and targets)
- (A) Add a date embedding to each view's diffusion-timestep embedding. About 15 lines; reaches all 22 ResBlocks.
- (B) Append date features to the per-view camera-ray map that already modulates every ResBlock. This is the closest thing SEVA has to per-view adaptive normalization.
- Recommended start: A + B, zero-initialized so the model starts out exactly like SEVA, plus some "date dropout" so the model also learns "date unknown".
- Avoid putting the date in CLIP cross-attention. SEVA averages CLIP over views and several blocks only look at the first view's tokens, so per-view info would silently get lost.
- WildCAT3D handled per-view *appearance* by concatenating a learned 8-number code per view at the input. That's simpler, but it only acts at the very first layer.

**Risks flagged**
- Averaging CLIP over inputs leaks the inputs' era into every target view.
- Date is tangled up with photo medium: old ≈ black-and-white, grainy, odd aspect ratio.
- Real building changes over decades break SEVA's "static scene" assumption.

---

## Where capture dates come from (2026-10-04)

**MegaScenes already ships the Commons metadata, so most images need no API calls**
- Every subcategory folder has a `raw_metadata.json` (e.g. `images/110/925/commons/Frauenkirche_Dresden_(before_1945)/raw_metadata.json`).
- It holds each image's Commons `imageinfo`: the description-page date, categories (including hidden ones), description, author, license, full EXIF, and the upload timestamp. We verified this on the Frauenkirche scene.
- It's a snapshot from the MegaScenes crawl (June 2023 – June 2024). It lacks the raw page wikitext and Structured Data, which we can fetch from the Commons API if needed.
- Cost: about 8 KB per image. Scenes with ≥200 images have 2.8M images in 76K metadata files, roughly 22 GB if we downloaded them all. We can stream and parse instead of storing.

**The best date signal**
- When uploaders used a date template, the `DateTimeOriginal` field carries a hidden machine-readable string. Example: `circa 1851 … date QS:P571,+1851…`, or `between 1890 and 1900` with exact earliest/latest bounds.
- Commons computes this itself, so we don't have to re-implement its date logic.
- Other sources, roughly in order of trust: wikitext date templates → museum/archive fields (e.g. "Decade: 1900/1909") → EXIF (only for born-digital photos) → year categories ("1930 photographs of Dresden", "1898 in Dresden") → decade categories → free text in title/description.

**Traps we've actually seen**
- **EXIF on scans = scan date.** A 1900 photo had EXIF from a 1999 Olympus camera; museum scans showed 2011 Hasselblad and 2012 HP scanner dates.
- **Date field = upload day.** Some historical uploads have the upload date typed in as the "date" (2 of 10 in one sample).
- **Reprints.** A 2000 reprint postcard of a "circa 1930" photo is dated 2000 ("2000 postcards of Dresden").
- **Modern photos inside "historical" subcategories.** E.g. a 2018 photo of stones from the old church sits under `Frauenkirche_Dresden_(before_1945)`. **Per-file metadata beats subcategory names.**
- **Structured Data's "inception" (P571) is often just a bot copy of the wikitext date**, mistakes included. Use it only as corroboration.
- **Decade categories ("X in the 1900s") describe the depicted era**, not necessarily when the image was made.
- Nobody has published an evaluated parser for photo dates on Commons. **We should hand-label a small sample** (a few hundred images) to measure our parser's accuracy.

**Photo type (photo / painting / postcard / print) and color vs black-and-white**
- Type: cheap signals in templates (`{{Photograph}}`, `{{Artwork|object type=}}`) and category keywords (postcard, painting, engraving, lithograph, albumen, glass plate…).
- B&W vs color: most reliable from the pixels themselves (mean color saturation), with categories as a backup.

---

## MegaScenes paper facts and related work (2026-10-04)

**How MegaScenes was built**
- A scene = a Commons category linked to a Wikidata item from 24 classes (churches, monuments, museums, bridges, …).
- Images were crawled up to **4 subcategory levels deep**, keeping subcategories whose names contain the scene name. So "Historical images of X" and "X in art" got in, and **old photos and artworks are in the dataset**.
- Crawled June 2023 – June 2024. Images were downsized to at most 1800 px wide.

**How many scenes have poses**
- Only **~83K of 430K scenes have any 3D reconstruction** (106K reconstructions, 2.0M registered images).
- Counting each scene's largest reconstruction: **4,884 scenes have ≥50 posed images; 2,007 have ≥100; 761 have ≥200; 67 have ≥1000.**
- Per-reconstruction registered counts are in `recon_metadata.json` in the MegaScenes web-viewer GitHub repo. That lets us rank scenes without downloading anything.
- Coverage by scene size: scenes with ≥200 images (4,818) hold 32% of all images; scenes with ≥1000 (533) hold 14%.

**How MegaScenes' own model picked training pairs**
- Pairs of images taken **within 3 hours of each other** (by metadata), to avoid lighting changes. The opposite of what we want, but proof that they used the timestamps.

**WildCAT3D (NeurIPS 2025): the closest relative to our model**
- Learns an 8-number "appearance" code per view and feeds it as extra input channels to a CAT3D-style diffusion model trained on MegaScenes. Built on their own CAT3D re-implementation, **not SEVA**.
- **No code or weights released.**
- Trick worth copying: keep the per-view codes in the "unconditional" branch of classifier-free guidance; dropping them caused oversaturated images.
- It only mentions seasons and holiday decorations, nothing about historical photos or dates.
- Cost: 32× A100 for about a week.

**Others using MegaScenes**
- About 15 papers train on it (matching, extreme-rotation pose, feed-forward splatting, video NVS). **None condition on dates.** VGGT, MASt3R, CUT3R and SEVA did *not* train on it, so MegaScenes is new data for SEVA.
- Closest hint: LoMa (2026) built a "HardMatch" benchmark of 1,000 image pairs spanning **capture years 1900–2020s**, but doesn't say how the dates were obtained.
- **No Commons-based dataset (WikiScenes, Doppelgangers, MegaScenes, HaLo-NeRF) has released per-image dates.** Ours would be the first.

---

## Literature takeaways worth remembering (2026-10-04)
*Full details in `docs/literature_review.md`.*

**Novelty**
- **Still holds, but needs narrower wording.**
  - DiffusionSat (ICLR 2024) already conditions diffusion on year/month/day, but only for single satellite images.
  - Siglidis et al. (ECCV 2024) and Faces Through Time condition on decade, but only for single object and face images.
  - Nobody does multi-view, cross-scene, ground-level, calendar-date conditioning.

**Proposal text to fix**
- 4DiM's time is seconds-scale relative time, not "seconds to seasons".
- Neural Scene Chronology's 4 landmarks are Flickr photos from 2009–2013, not historical.

**Very recent, closest competitor:** CrossTimeEdit (arXiv, 29 Sep 2026) edits a street panorama to ~10 years earlier. Single view, no date input. Keep watching.

**Matching old photos to new ones is a known hard problem**
- Classic SIFT gets ~0 correct matches on then/now pairs. Learned matchers (SuperPoint+LightGlue, LoFTR, RoMa) do much better.
- "Registered" doesn't mean "correct": on symmetric buildings many old photos get flipped poses. **Measure angular error.**
- HistReNeRF (2026): only 25% of 230 archival photos of 3 landmarks registered with SIFT.
- No one has tested MASt3R/VGGT across decades. A small benchmark would itself be a contribution.

**The "grayscale = old" shortcut is real and documented**
- Date classifiers lean heavily on film color and grayscale cues (Palermo 2012, Salem 2016, "Seeing Time" 2026).
- Our model will too unless we:
  - give photo type as a separate input;
  - "age" modern photos as augmentation while keeping their modern date. Luo et al. 2021 gives an era-by-era film simulation recipe.

**Evaluating "does this look like 1920"**
- No off-the-shelf date estimator covers 1850s–2020s outdoor scenes. DEW only covers 1930–1999.
- We'd train our own on held-out scenes, with grayscale controls.

**Baselines**
- Neural Scene Chronology is only realistically runnable on 5Pointz (2009–2013), so it isn't a decades test.
- WildCAT3D has no code.
- Main baselines: plain SEVA + nearest-dated real image.

---

## Engineering notes (2026-10-04)
- **Poses without the huge files.** `reconstruct_aux/.../images.minibin` is COLMAP's image list (id, rotation, translation, camera id, name) **without the 2D keypoints**. 210 KB vs 211 MB for the same Frauenkirche model. With `cameras.bin`, it's all we need for poses.
- **Per-scene registration counts without downloads.** The MegaScenes web-viewer repo ships `recon_metadata.json`: scene id → `[name, n_reconstructions, (n_images, n_points) per reconstruction…]`.

---

## SEVA training: what the authors say (2026-10-04)
- In GitHub issue #27, SEVA's authors say they can't release the training script. But **SEVA was adapted from Stability's open `generative-models` (sgm) repo.** They suggest building training from its SD2.1 example config (`configs/example_training/txt2img-clipl.yaml`) and its training guide.
- Other pointers they gave: `generative-models` issues #239 and #249, and **SVD_Xtend** (a community training script for Stable Video Diffusion with a similar structure).
- Takeaway: training is "assemble from known parts" (sgm trainer + SEVA's model and conditioning + our dataloader), not research from scratch. Still real engineering work.

---

## Pilot census: 42 scenes (2026-10-04)
*Pipeline: `scripts/process_scenes.py` + `scripts/census.py` (see README). Scenes: `configs/scenes_pilot.txt` = Kevin's 20 + 10 scenes with big physical changes + 12 well-photographed landmarks. Counts are unique Commons files, not paths.*

**How much old material is there?**
- 162K unique files. 97% photos or unknown type; the rest are paintings, prints, drawings, plans.
- Photos (incl. postcards) by era:

  | Era | Photos | Posed in MegaScenes |
  |---|---|---|
  | pre-1900 | 4,125 | 370 |
  | 1900–1944 | 3,846 | 339 |
  | 1945–1969 | 2,912 | 868 |
  | 1970–1999 | 4,196 | 1,090 |
  | 2000+ | 140,730 | 62,393 |
  | undated / too vague | 2,365 | 441 |

- So about **10.9K pre-1970 photos in 42 scenes, ~1.6K of them already posed (15%)**, vs ~44% of modern photos posed.
- 98–100% of files get *some* date. "Too vague" means the interval is wider than 30 years (e.g. "before 1945").

**Go/no-go gate** (≥10 pre-1970 and ≥10 2000+ photos)
- **Posed today in one COLMAP model: 22 of 42 scenes pass.**
- **With our own posing of old photos (VGGT-Omega): all 42 pass.**
- Best scenes with posed old photos (pre-1970 / 2000+ in the best model):

  | Scene | pre-1970 | 2000+ |
  |---|---|---|
  | Dam Square | **649** | 87 |
  | Notre-Dame | 107 | 3,202 |
  | Arc de Triomphe | 88 | 1,112 |
  | Zwinger | 74 | 1,022 |
  | Semperoper | 68 | 544 |
  | Sydney Harbour Bridge | 67 | 415 |
  | Taj Mahal | 43 | 4,616 |
  | Florence Cathedral | 36 | 1,545 |
  | Alhambra | 30 | 919 |

- These agree with Kevin's independent pilot (Dam Square 656, Notre-Dame 113).

**The scenes that changed most are the ones without posed old photos**
- 1–2 posed pre-1970 photos each: Frauenkirche, Reichstag, Alexanderplatz, Cologne Cathedral, Sagrada Família. Times Square has 0.
- Yet they have plenty of old photos: Frauenkirche 176, Reichstag 253, Alexanderplatz 337, Cologne 296.
- **Posing old photos ourselves is what unlocks the interesting scenes.**

**What the posed old photos look like** (spot-check of ~40)
- Dates look right. They come mostly from archives with curated dates: Nationaal Archief, Rijksmuseum, Fortepan, Bundesarchiv, Library of Congress, State Library NSW, Brück & Sohn postcards.
- Many are **events with the landmark in the background** (ceremonies on Dam Square, ships under the Sydney Harbour Bridge), so lots of people, vehicles and other transient content. Valid views of the scene at that time, but harder training data.
- Dam Square's best model is unusual: mostly old photos (649 pre-1970 vs 87 modern).

**Date-parsing lessons from running at scale**
- Page date and EXIF often differ by a day or a few weeks (time zones, typed from memory). We now treat dates within ~5 weeks as agreeing; before that, ~140 photos in 2 scenes were wrongly flagged as conflicts.
- **Unset camera clocks**: EXIF dates of 1 January 2000/2001 are common. Any 1 January EXIF date is now ignored.
- Hidden "Pages with maps" category made photos look like maps. Maintenance categories are now ignored for image type.
- Some subcategory names contain `:` or `"`, which this filesystem rejects in file names, so cache names are now encoded.

**Engineering notes**
- Speed: ~2 MB of metadata per subcategory file, ~30 files/s at best; the 42 scenes took ~25 min. It's network-bound.
- This machine is a **submission node** (asks for no long or heavy jobs). For a full-scale census (hundreds or thousands of scenes) use `slurm/census_cpu.sbatch` (CPU job on `vulcan-cpu`).

---

## Single-scene dataset: Notre-Dame (2026-10-04)
*`scripts/build_scene_dataset.py --scene Cathédrale_Notre-Dame_de_Paris --max-modern 800` → `data/datasets/25512_model0/` (view table + preview contact sheets).*

**Why Notre-Dame:** old and modern photos are posed in one COLMAP model (model 0), and the building really changed:
- 3,332 posed, dated photos in model 0, grouped by the spire's history:

  | Period | Photos |
  |---|---|
  | Before 1859 (no spire yet; Baldus, Collard, 1840s–50s) | **16** |
  | 1859–1944 (Viollet-le-Duc's spire) | 75 |
  | 1945–1989 | 27 |
  | 1990–2009 | 579 |
  | 2010 to the April 2019 fire | 2,151 |
  | After the fire (spire gone; 2019–2023) | **484** |

- The MegaScenes crawl ended mid-2024, so it has no photos of the rebuilt spire (Dec 2024).

**What a sample looks like**
- 8 views = 3 inputs + 5 targets, all in one coordinate frame.
- Targets share one date: within 1 year for modern photos, 5 years for pre-1990. Inputs come from any date.
- Cameras are normalized exactly like SEVA (centre on the mean camera position, reference camera at distance 2). Intrinsics are normalized. Plücker rays are ported from SEVA and tested.
- Each view carries its date interval, image type and a monochrome hint.
- Optional knobs: a share of samples with old targets, and a minimum date gap between inputs and targets.
- The first build used a cap of 800 modern photos to keep the download small: 934 views, 911 can anchor a sample. The preview sheets look right, e.g. 1960–64 targets with 2011/2017/1890s inputs.

**Black-and-white / sepia detection from pixels is weak**
- Plain saturation misses sepia: brown albumen prints are strongly tinted.
- Better cue: toned prints vary along a single color axis, so the Lab chroma spread along the *minor* axis is near 0.
  - It flags 90% of 1900–1944 photos, but only 33% of pre-1900 prints (card mounts, stereo cards and hand-colored photochroms break it).
  - It also flags 7% of modern photos (grey stone, overcast skies).
- Conclusion: keep it as a weak hint (`is_monochrome`, plus continuous `chroma_minor`). **Real B&W/sepia labels should come from the VLM or a small CLIP classifier.**

**Other observations**
- Some old prints are scanned **with their cardboard mount** (borders around the photo). COLMAP registered them anyway, but the border is junk for training. We'll want border detection/cropping (VLM or simple heuristics) later.
- Old photos are portrait/odd aspect more often. The centre crop to a square cuts some content; fine for now.
- Cached images match their COLMAP cameras' aspect ratio within 1.2%, so intrinsics scale cleanly.

---

## Posing old photos with VGGT-Omega: setup (2026-10-04)
**Method** (`src/timemachine/posing/`, `scripts/pose_with_vggt.py`, `slurm/pose_vggt.sbatch`)
1. Run VGGT-Omega on batches of 32 old "query" photos plus 32 modern "anchor" photos. The anchors already have COLMAP poses and are spread around the scene by camera position and direction.
2. Fit a robust similarity transform (scale + rotation + translation, RANSAC) from VGGT's anchor cameras to their COLMAP cameras, and use it to move the queries into the COLMAP frame.
3. Pose each query in 3 different batches. The spread of its poses across batches serves as a confidence score.
- **Evaluation mode:** re-pose the old photos COLMAP *did* register and compare with their COLMAP poses (rotation error in degrees, position error relative to scene size). This gives the cross-decade accuracy number the literature lacks.
- Tested on CPU with a fake model: it recovers poses exactly even with 10% wildly wrong anchors.

**Notre-Dame targets:** 567 dated old photos are *not* registered in model 0 (400 pre-1900, 133 from 1900–1944, 34 from 1945–1969). Some are close-ups (the roof angel, gargoyles) that may not register at all.

**Practicalities**
- VGGT-Omega (1B, 512 px; checkpoint already in the HF cache) runs from your checkout via `VGGT_OMEGA_ROOT`, read-only. Its only extra dependency is `torchvision`.
- Memory: ~11 GB for 64 frames, so one A6000 is plenty.
- Gotcha: importing code from another folder makes Python write `__pycache__` files into it. The wrapper now disables that.

---

## Census at scale: the 800 scenes with the largest reconstructions (2026-10-04)
*Slurm job on `vulcan-cpu` (~35 min), plus the 42 pilot scenes = 802 scenes. Tables are committed in `docs/census/` (per scene and per COLMAP model), so nobody needs to rerun this.*

**Totals** (unique files)
- 892K files: 764K photos, 101K of unknown type, the rest paintings, artworks, postcards, plans, prints, drawings.
- **~47.5K dated pre-1970 photos, of which ~7.1K (15%) already have a pose.**

| Era | Photos | Posed |
|---|---|---|
| pre-1900 | 15.8K | 2.3K |
| 1900–1944 | 19.3K | 2.1K |
| 1945–1969 | 12.4K | 2.7K |
| 1970–1999 | 17.0K | 4.4K |
| 2000+ | 788K | 364K |

**How old?** Photos per decade (posed in brackets):

| 1840s | 1850s | 1860s | 1870s | 1880s | 1890s | 1900s | 1910s | 1920s | 1930s | 1940s | 1950s | 1960s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8K (18) | 1.7K (197) | 1.9K (335) | 2.6K (568) | 2.4K (308) | 4.3K (816) | 5.7K (641) | 4.2K (389) | 3.9K (358) | 4.2K (511) | 3.2K (598) | 4.8K (945) | 5.8K (1.3K) |

- Real photo coverage starts in the 1840s and is steady from the 1890s on.
- ~1K files dated before 1840 are artworks our type detector missed ("unknown" type).

**Go/no-go gate** (≥10 pre-1970 and ≥10 2000+ photos)
- **Posed today in one COLMAP model: 148 of 802 scenes.**
- **With our own posing: 482 of 802.**
- (149 and 491 before re-dating with the rule that undated-type images from before 1839 are artworks, not photos.)

| Scenes with at least… | ≥10 | ≥50 | ≥100 | ≥200 |
|---|---|---|---|---|
| pre-1970 photos | 492 | 214 | 125 | 64 |
| …posed in one model | 151 | 29 | 6 | 4 |
| pre-1900 photos | 298 | 77 | 34 | 12 |

- **Answer to "how much old data is there really":** a lot of old photos (125 scenes with 100+), but few are posed (6 scenes with 100+ in one model). **Posing old photos ourselves is the main lever.**

**Biases to keep in mind**
- **Amsterdam dominates** the posed old photos: 10 of the top 50 scenes are Dutch (Dam Square, Nationaal Monument, Royal Palace, Munttoren, Westerkerk, Nieuwe Kerk, …), thanks to the Nationaal Archief's huge, well-dated photo archive on Commons. A model trained mostly on these will learn "old Amsterdam".
- Other strong scenes: Notre-Dame, Lincoln Memorial, Arc de Triomphe, Great Sphinx, Potala Palace, Zwinger, Semperoper, Sydney Harbour Bridge, Altes Rathaus Leipzig.
- Dendera has 304 posed old photos but 0 posed modern ones in that model (an old-only reconstruction).
- This census covers only scenes with big reconstructions. Scenes with *no* reconstruction can still have many old photos (Eiffel Tower: 10.8K files, no COLMAP model at all), but we'd have to reconstruct them from scratch.

**Robustness fixes found at scale:** models listed in the web viewer's index but missing on S3, a malformed month in a date string, control characters and one truly malformed metadata file on S3. All are now handled (skipped and reported).


## VGGT-Omega evaluation, first try (2026-10-05)
*Notre-Dame model 0; the 107 old photos COLMAP did register are re-posed and compared with their COLMAP poses. One A6000, ~4 min.*

**Settings:** 32 queries + 32 anchors per batch, anchors spread over the scene, similarity transform fitted on camera positions only (5% threshold).

**Result: weak.**
- Only 3 of 12 batches aligned (median 6 consistent anchors), so only **42% of photos got placed**.
- Among placed photos: **rotation error median 5.4°**, but **position error median 0.41 scene-widths** (too far off to use).
- Overall, 37% are within 10°.

**Diagnosis**
- VGGT's *rotations* are decent even in rejected batches (anchor rotation errors mostly 3–6°). Its *positions* are not.
- Spreading anchors over the whole scene means few of them overlap with each other or with the queries, the hardest case for a feed-forward model.
- Batches with fewer queries (11 instead of 32) aligned much better.

**Next try** (now running): 8 queries + 24 anchors per batch, with anchors picked by visual similarity (DINOv2 embeddings), and alignment that fits rotation first from the anchors' orientations, then scale/position with a looser 15% threshold.

## VGGT-Omega evaluation, second try: it works for rotation (2026-10-05)
*Same 107 old Notre-Dame photos with known COLMAP poses.*

| Setting | Batches aligned | Placed | Rotation error (median) | Within 10° | Position error (median, scene-widths) |
|---|---|---|---|---|---|
| First try (32+32, spread anchors, fit on positions) | 3/12 | 42% | 5.4° | 37% | 0.41 |
| 8+24, spread anchors, rotation-first | 42/42 | 100% | 3.1° | 84% | 0.35 |
| **8+24, look-alike anchors (DINOv2), rotation-first** | **42/42** | **100%** | **2.4°** | **94%** | **0.21** |

- **Rotation is solved well enough:**
  - 94% of old photos are within 10° and 99% within 20°.
  - Pre-1900 photos: 3.6° median; 1900–1944: 1.7°.
- **Position is the weak part.** Median error ≈ 17% of the camera's distance to the building. The 1945–69 group is worse (0.89 scene-widths; few photos, many taken from far away).
- **The confidence score works:** keeping photos whose position varies little across batches (spread ≤ 0.1) keeps 49% with median position error 0.11 (vs 0.32 for the rest). Spread ≤ 0.05 keeps 27% at 0.06.
- **Takeaway:** picking anchors that *look like* the old photo is what makes feed-forward posing work across decades.
- **Possible upgrade for positions:** match each old photo to its retrieved anchors with RoMa (weights already cached) and solve for position against COLMAP's 3D points (PnP), keeping VGGT's good rotation as the starting point.
- This is, as far as we know, the first measurement of a feed-forward 3D model (VGGT-Omega) registering 19th-century photos against a modern reconstruction.

## VGGT-Omega registration of the old photos COLMAP missed (Notre-Dame, 2026-10-05)
- 558 old photos (9 failed to download). All were placed and 189/210 batches aligned, but these photos are **much harder** than the evaluation set:
  - Median rotation spread across batches: 12.8° (vs ~1° on the evaluation set).
  - Median position spread: 0.46 (vs 0.10).
- Passing the confidence filter: **80** at spread ≤ 0.1 (52 pre-1900, 22 from 1900–44, 6 from 1945–69); 114 at ≤ 0.2; 139 at ≤ 0.3.
- **Visual check of 8 "confident" ones** (`docs/figures/vggt_register_check_notredame.jpg`, old photo next to the modern anchor nearest its estimated pose):
  - 4 clearly right (apse with fountain 1847, river view 1892, portals with a 1930s car 1936, facade tower 1918).
  - 2 plausible (statue close-up, flying buttresses).
  - **2 wrong:** a 1910 *interior* (model 0 is exterior only) and an 1857 *book page with an engraving* that was typed as a photo.
- **Lesson: the confidence filter alone isn't enough.** Images that don't show the scene's exterior get placed *consistently wrong*. Repeats share their best-matching anchors, so their spread underestimates the error.
- **Needed before these poses join the training data:**
  1. A **content filter** (VLM or CLIP): exterior view vs interior / document / close-up detail. Needed for training views in general.
  2. **Geometric verification:** feature matches (RoMa/LightGlue) between the old photo and its nearest anchors.
- So far the dataset uses only COLMAP-posed views; VGGT poses are not merged yet. `build_scene_dataset.py --vggt-tag retrieval` can merge them once verified.

---

## Review of Aadhitya's data pipeline on `main` (2026-10-25)
*`data_pipeline/` + `docs/data_pipeline.md` + `docs/interface_decisions.md`, tested on the Taj Mahal.*

**What it does:** reads a scene's COLMAP model, looks up each registered image's Commons metadata through the API, parses a year interval + confidence, checks downloaded files against the camera sizes, audits old vs modern coverage, and plots old-photo camera positions.

**Verdict: solid.** His 81 tests pass, and the docs are clear and careful.
- **Same results as our parser:** on the Taj Mahal's model 0 (4,711 unique registered photos), the two parsers were compared where both give a usable date (4,587 photos):
  - **99.2% agree within 2 years, 100% within 10 years.**
  - The era split is identical except for one photo (1952 vs 1965).
  - Ours dates 90 more photos; his dates 7 that ours doesn't.
- **His Taj Mahal numbers match our census:** ~40–55 old photos, almost all of them albumen or black-and-white prints.

**Gaps (rare on the Taj Mahal, matter elsewhere)**
1. An *upload day typed into the date field* isn't detected (e.g. a 1910 postcard dated "2019-04-20" gets 2019 at 0.95). The upload time is fetched but never compared.
2. "built" / "in YYYY" count as capture-date cues, so categories like "Buildings completed in 1726" can date a photo when the date field is missing.
3. For old photos with no date text, the EXIF *scan* date is used and passes the 0.5 cutoff.
4. Years only (no month); no photo-type labels yet.

**Overlap:** `main`'s `data_pipeline/`, Kevin's `census/` and our `src/timemachine/` now do largely the same job three ways. Worth consolidating when we merge.

**His `interface_decisions.md` leaves open questions for the SEVA side.** Our SEVA study answers most of them:
- **Resolution / pixels:** 576×576, pixels in [-1, 1].
- **Crop vs pad:** SEVA centre-crops.
- **Poses:** camera-to-world, OpenCV axes, encoded as Plücker rays relative to the first input view.
- **Scale:** centre on the mean camera position, reference camera at distance 2. Not "median distance 1".
- **Training code:** none released (we write our own).

---

## Date-conditioned SEVA: setup, identity check, baseline (2026-10-25)
**Setup**
- SEVA is installed from GitHub at the commit we studied, via `scripts/setup_seva.sh`. It's cloned into the git-ignored `external/` and installed editable, because SEVA's own packaging is broken (`seva.modules` is missing from a normal install).
- Weights are read only from the shared cache `/fs/nexus-projects/wildnvs/liger/weights/hf/hub`: SEVA v1.1, the SD 2.1-base VAE mirror (`Manojb/...`; the original repo is gone) and OpenCLIP ViT-H.
- That cache also has **Qwen3-VL-32B-Instruct**, a candidate for the VLM labelling step.

**The date module** (`src/timemachine/model/`)
- Each view's year is encoded with sine/cosine features plus a small network, with a learned "unknown date" input.
- It is **added to SEVA's per-view timestep embedding**, which reaches every ResBlock. Implemented by swapping one submodule (`time_embed`), with no edits to SEVA's code.
- **Identity check on the real weights:** with the new module freshly initialized, outputs are exactly SEVA's (max difference 0.0). With a small random perturbation of its last layer, outputs change (0.014), so the date really reaches the network.
- Gotcha: `torch.autocast` caches low-precision weight copies for a whole block. An in-place weight change inside the same block is ignored, which made the first check falsely report "no effect".

**Baseline: plain SEVA on Notre-Dame** (3 inputs → 5 targets, 576², 50 steps, guidance 2.0; ~4.5 min for 8 samples on one A6000)

| Sampling | Mean PSNR |
|---|---|
| Random samples as in training (targets share a date, half pre-1970) | **8.1 dB** (modern targets 8.7, old 7.5) |
| Sanity check: 8 modern views with neighbouring cameras | **9.0 dB** |

- **Camera conventions are correct:** in the sanity check SEVA reproduces the target viewpoints, framing and layout. PSNR is low because of appearance (lighting, sky, season, SEVA's oversaturated look). Pixel PSNR is a poor metric for in-the-wild photos, so we should add LPIPS / feature metrics.
- **SEVA copies the inputs' appearance.** In a random sample, a 2017 color target came out as a sepia albumen print copied from an 1870s input (`docs/figures/seva_baseline_random_appearance_copy.jpg`).
- **The motivating failure, captured.** Given inputs from 2004–2017, SEVA draws Notre-Dame's spire into 2019 targets that actually show scaffolding and no spire after the fire (`docs/figures/seva_baseline_neighbors_spire.jpg`). This is what date conditioning should fix.
- Random samples with far-apart, sparse inputs give hallucinated generic Gothic architecture. Training samples should favour inputs that overlap with the targets.

---

## Training date-conditioned SEVA: setup (2026-11-02)
*Code: `src/timemachine/training/`, `src/timemachine/model/{lora,grad_checkpoint,finetune}.py`, `scripts/train_dated_seva.py`.*

**What trains**
- SEVA stays frozen. Two things are trained, both starting at zero so step 0 is exactly SEVA:
  - the date module (0.7M parameters);
  - LoRA adapters, rank 32, on every attention and feed-forward layer of the multi-view transformer blocks (30M parameters).
- **The loss matches SEVA's sampler, line for line:**
  - same conditioning tensors as `seva.eval.do_sample`;
  - input views are swapped for their clean latents;
  - same input scaling, 1000 discrete noise levels, ε-prediction;
  - the loss is computed only on target views.
- Classifier-free guidance keeps working because 10% of steps train the "no input images" branch, as SEVA's sampler expects.
- Each view's year is drawn inside its date interval, so "circa 1890" is not always exactly 1890.5.
- 10% of samples are marked "date unknown", so the model learns that case too.

**Samples are spatially coherent now**
- New `max_neighbors` option (24): targets and inputs come from the cameras closest to the anchor view.
- Before, a sample's views could come from opposite sides of the cathedral, so the targets couldn't be inferred from the inputs.
- Inputs still come from any date, so most samples mix eras (e.g. 2015 inputs, 1880 targets).
- Half the samples get pre-1970 targets.

**Cost on one A6000** (8 views at 576², bf16, gradient checkpointing): **2.3 s/step, 14 GB**.
- This leaves room for more views per sample, or for turning checkpointing off (~30% faster).
- Checkpoints hold only the trained weights: 123 MB, plus optimizer state in `latest.pt`.

**How we measure whether dates are used** (`training/probes.py`)
- *Date-swap loss gap:* 16 fixed samples with fixed noise. We compare the loss with the true dates against the loss with the targets' dates moved to another era (old→2015, modern→1890). If the model uses dates, the ratio rises above 1. At step 0 it is exactly 1.
- *Probe renders:* fixed input views and target cameras, rendered at several years with the same random seed, so any difference between rows comes from the date alone. Notre-Dame probes (`configs/probes_notre_dame.json`):
  - **fire:** inputs are 2012–2016 photos with the spire; targets are post-fire photos (scaffolding, cranes). Rendered at 1850 (before Viollet-le-Duc's spire), 1900, 1960, 2015, 2021 and "unknown".
  - **old_from_modern:** modern photos of the portals as inputs; 1860s–1890s albumen prints of the same portals as targets.
  - **modern_from_old:** 1905–1936 black-and-white inputs; 2010–2016 colour targets.
- Sanity check passed: at step 0 every row of every probe is identical. Plain SEVA draws the spire into all the post-fire targets.

**A loss-weighting question (being tested)**
- 89% of SEVA's noise levels are high-noise (σ > 1). The overall look (colour era, spire or no spire) is decided there.
- With the plain ε-loss, those levels give tiny gradients (~1e-4). The few near-clean levels give spikes (norm ~80). Adam's running scale then sits near the spikes, which shrinks the high-noise updates.
- Two runs compare plain ε (what SD-style models train with) against EDM weighting (σ²+1), which makes every noise level contribute a similar loss.

## First Notre-Dame overfit run: the LoRA learning rate destroyed SEVA (2026-11-03)
*Run `data/experiments/train/nd_overfit_eps` (plain ε-loss, LoRA and date module both at lr 1e-4). Tools: `scripts/inspect_checkpoint.py`, `scripts/ablate_checkpoint.py`.*

**What happened**
- Steps 0–500: fine. The probe loss stayed at SEVA's level (0.024).
- Steps 600–1,000: gradient norms climbed from ~1.5 to ~20, a sign of instability. The probe loss rose 10× (0.25).
- From step 1,400: complete collapse. The model outputs pure noise for every date, its loss sits at ~0.9 (predicting nothing), and its gradients have nearly vanished, so it cannot recover.
- No date effect was ever learned: swapping the dates changed the loss by under 1%.

**Cause: the LoRA, not the date embedding.** We evaluated the step-2,500 checkpoint with parts switched off:

| | probe loss |
|---|---|
| plain SEVA (both off) | 0.0239 |
| date module only | 0.0238 |
| LoRA only | 0.90 |
| both (as trained) | 0.90 |

- The LoRA had rewritten some layers. Its update reached 58% of the original weight in a temporal feed-forward layer and 45% in cross-attention (which reads the CLIP embedding of the input images).
- The date embedding stayed modest: RMS 0.03–0.13, against 0.29 for the timestep embedding it is added to. On its own it left SEVA intact.

**Fix and safeguards**
- LoRA learning rate is now 1e-5 by default (date module unchanged at 1e-4). For reference, VMem fine-tuned SEVA at an even lower rate.
- Every date-gap probe (each 500 steps) now also logs the size of the date embedding and of the LoRA updates.
- Training stops by itself if the probe loss exceeds 3× its starting value, instead of burning GPU hours on a dead model.
- An optional bound on the date embedding exists (`--date-max-abs`). It is off, since the date embedding was not the problem.

## Second run (LoRA lr 1e-5): stable, but the LoRA learns an era-blind look (2026-11-03)
*Run `data/experiments/train/nd_overfit_eps_lr1e-5`; same as the first run but LoRA lr 1e-5.*

**Stable.** By step 2,500:
- the probe loss has fallen 6% below SEVA's (0.0239 → 0.0225);
- the largest LoRA update is 6% of its weight;
- nothing has collapsed.

**But the renders went wrong in a way the loss does not show.** At step 2,500 every render has a strong sepia-orange cast and less detail, at every date. The date does change them (old dates pinker, 2015/2021 golden), but no date gives natural colour back. The spire is still drawn into the post-fire targets.

**Same switch-off test, now also rendering the fire probe** (`ablate_checkpoint.py --render fire`):

| | probe loss | date-swap gap | renders (1880 / 2015) |
|---|---|---|---|
| plain SEVA | 0.0239 | 0 | natural colour, identical |
| date module only | 0.0234 | **+1.8%** (true date fits better) | clean; 1880 harsher and less saturated (sat 0.29), 2015 normal colour (0.41) |
| LoRA only | 0.0223 | 0 | same sepia-orange cast at both dates (sat 0.47) |
| both | 0.0225 | −3.2% | cast at both dates (0.48 / 0.54) |
| both, no guidance | | | even stronger cast (0.57 / 0.62): guidance is not the cause |

**What this means**
- The **date module alone learns the right thing**: older dates look older, the images stay clean, and the true date fits better than a wrong one.
- The **LoRA cannot see the date**, so it learns the average look of the training targets. Half of them are pre-1970 prints, so that average is half-sepia, applied to every date.
- In the combined model, the LoRA's era-blind shift is learned faster than the date pathway learns to steer it. The two together are worse than the date module alone.
- **The probe loss missed this.** It improved while the renders got worse, because a global colour shift is decided at high noise levels, where the ε-loss is tiny. From now on, judge runs by the renders, not the loss.

**It got worse, not date-aware.** By step 5,000 every render is saturated red-orange noise, at every date. The probe loss barely moved (0.0241, SEVA 0.0239). Run stopped at 5,000.

**Why, most likely: the plain ε-loss is blind to high noise levels in SEVA's schedule.**
- SEVA's noise levels go up to σ = 85, and 89% are above σ = 1.
- At σ = 85, a noise-prediction error δ becomes an image error of 85·δ. So a colour shift big enough to ruin a render costs the ε-loss almost nothing.
- Those levels therefore give almost no training signal. Meanwhile the LoRA, steered by the low-noise levels, slowly breaks what happens at high noise.
- EDM weighting (σ²+1) counts exactly that image-space error, so it should prevent this. The queued `nd_overfit_edm` run tests it.

**A loss number that does see it.** `x0_mse_high_noise` is the denoised-image error (σ² × noise error) at the probe's high-noise samples (σ > 10). It is now logged with every date-gap probe:

| step 2,500 / 5,000 | plain SEVA | date only | LoRA only | both |
|---|---|---|---|---|
| denoising loss | 0.0239 / 0.0239 | 0.0234 / 0.0257 | 0.0223 / 0.0234 | 0.0225 / 0.0241 |
| `x0_mse_high_noise` | 0.53 / 0.53 | 0.53 / 0.55 | 0.80 / **1.45** | 0.86 / **1.41** |
| date-swap gap | 0 | +1.8% / **+4.7%** | 0 | −3.2% / −1.9% |

- The high-noise error tracks the renders: ×1.6 when the sepia cast appeared, ×2.7 when they were destroyed.
- **Training now also stops if it exceeds 1.5× its starting value** (`--abort-x0-ratio`). That would have stopped this run around step 2,500.
- The date module trained alongside the LoRA is still healthy and increasingly date-aware (+4.7% at step 5,000) when evaluated without the LoRA.

- If the LoRA must stay and EDM weighting doesn't fix it, two options: train the date module first and add the LoRA afterwards, or give the adapted layers direct access to the date.

## Third run (date module only, plain ε-loss): the date embedding grew too large (2026-11-03)
*Run `data/experiments/train/nd_overfit_dateonly`; no LoRA, date lr 1e-4, plain ε-loss. Stopped at step ~2,600.*

**The date vector kept growing.** Its size (RMS) for 2015 went 0.14 → 0.30 → 0.41 → 0.48 → 0.55 over steps 500–2,500. SEVA's own noise-level vector, which the date vector is added to, has RMS 0.29. With the LoRA present (second run) the date vector levelled off at ~0.30.

**Renders at step 2,500**
- Modern dates (2015, 2021) come out blotchy and broken in all three probes.
- Old dates and "unknown" look like plain SEVA: **not older**, no sepia or B&W.
- `old_from_modern` (already hard for plain SEVA) collapsed into black-and-white blocks at every date.

**The probe loss got worse, not better** (0.0246 vs SEVA's 0.0239, on training samples). The +3.5% date-swap gap only means a wrong date hurts more than the right one; the right date doesn't help.

**Shrinking the date vector at test time** (`ablate_checkpoint.py --date-scales 0.5 0.25`, fire probe):

| date vector | probe loss | swap gap | `x0_mse_high_noise` | 2015 render |
|---|---|---|---|---|
| off (plain SEVA) | 0.0239 | 0 | 0.53 | clean |
| as trained (RMS 0.55) | 0.0246 | +3.5% | 0.58 | blotchy, broken |
| × 0.5 | 0.0238 | +0.4% | 0.55 | clean |
| × 0.25 | 0.0236 | −0.2% | 0.54 | clean |

1880 looks like plain SEVA at every scale.

**What this means**
- **Size, not content, broke the renders.** At twice the size of the noise-level signal it is added to, the date vector makes the model misjudge how noisy its input is, which leaves blotches. Half the size is harmless.
- **Nothing about eras was learned** in 2,500 steps.
- Likely reason it grew: with plain ε-loss the near-clean noise levels dominate the gradients (~10⁵×). There the date has nothing to explain, so the date module drifts on noise instead of learning, which also fits the probe loss getting worse.
- **The 1.5× high-noise guard did not catch this** (×1.08): the damage is at lower noise levels. The date vector's size (the `[weights]` lines) is the warning sign here.

## Fourth run (date module only, EDM weighting, date vector capped at 0.3): safe, but date-blind so far (2026-11-03)
*Caveat (code audit, 2026-11-10): runs 1–7 trained in fp16 without a gradient scaler, and a gradient clip at 1.0 rescaled almost every step, which largely cancelled the EDM weighting. Read the ε-vs-EDM comparisons below with that in mind; see "Code audit" at the end.*
*Run `data/experiments/train/nd_overfit_edm_dateonly`; `--weighting edm --lora-rank 0 --date-max-abs 0.3`. Started 11:07, 10,000 steps.*

| step | 0 | 500 | 1,000 | 1,500 | 2,000 | 2,500 |
|---|---|---|---|---|---|---|
| probe loss (SEVA 0.0239) | 0.0239 | 0.0225 | 0.0223 | 0.0224 | 0.0224 | 0.0226 |
| date-swap gap | 0 | −0.2% | +0.4% | +0.1% | 0.0% | +0.3% |
| `x0_mse_high_noise` | 0.53 | 0.54 | 0.54 | 0.54 | 0.55 | 0.54 |
| date vector RMS, 2015 / unknown | 0 | 0.10 / 0.06 | 0.18 / 0.11 | 0.21 / 0.14 | 0.23 / 0.16 | 0.25 / 0.19 |
| how much it differs between dates ("spread") | 0 | 0.05 | 0.07 | 0.08 | 0.08 | 0.08 |

**Step 2,500 renders**
- **Clean** in all three probes, no blotches. `old_from_modern` is back to plain SEVA's quality.
- **Date-blind:** every date gives nearly the same image (pixel difference between dates 0.02–0.07; the eps date-only run had 0.07–0.21). 1850 does not look older than 2021.

**What this means**
- EDM weighting plus the cap fixed the damage.
- The 6% lower probe loss comes from a **date-independent offset**: "unknown" fits as well as the true date. It looks like a Notre-Dame-specific correction, not dates.
- **Concern:** that offset already uses most of the cap (RMS 0.25 of 0.3). The date-dependent part ("spread") has stopped growing at 0.08, so the cap may leave it no room.

**Step 5,000: still date-blind.**
- Swap gap +0.5% (+0.3% at 2,500), spread flat at 0.09, date vector RMS 0.27 (cap 0.3). Probe loss 0.0225, `x0_mse_high_noise` 0.54: still safe.
- Renders: still clean; dates still barely change them. In `old_from_modern`, 1880 is slightly *more* saturated than 2021 (0.28 vs 0.23), not sepia or B&W.
- **Conclusion:** a date vector added to the noise-level signal is either unsafe (too large, run 3) or too weak to change the era look (capped, this run). **Proposal:** give the date its own zero-initialised scale-and-shift in every block, next to the one SEVA uses for the camera rays (`h * (1 + scale) + shift`, `seva/modules/layers.py`). It leaves the noise-level signal untouched, and scaling is the natural way to drain colour.

Stopped at step ~6,000 (gap +0.6%, spread 0.09) to free the GPU for date FiLM.

## Date FiLM: the date gets its own scale-and-shift in every block (2026-11-03)
*`src/timemachine/model/date_film.py`; now the default (`--date-injection film`). The old path is `--date-injection timestep`.*

**Why.** With only a date vector added to the timestep vector, the network can use the date only through SEVA's frozen layers that read that vector, which were trained to react to noise level. That is either unsafe (run 3) or too weak (run 4). FiLM adds new, trainable "listeners" instead.

**How**
- In each of SEVA's ResBlocks, right after its own camera scale-and-shift: `h = h · (1 + s_date) + b_date`, with one value per channel and view.
- year → Fourier features → shared encoder (17 → 256) → one zero-initialised linear layer per block → `(s_date, b_date)`. 15.4M parameters (the LoRA has 30.7M).
- Step 0 is exactly SEVA: probe loss 0.0239 and `x0_mse_high_noise` 0.5321 on the real model.
- **Trap avoided:** gradient checkpointing re-runs every block during backward, after the dates context has closed. So the dates are encoded once per forward pass at the top of the UNet (outside the checkpointed blocks) and kept until the next one. A test checks that gradients with and without checkpointing are identical.
- Checkpoints store the encoder and per-block layers (`trainable["film"]`). Older checkpoints still load as `timestep`. `ablate_checkpoint.py --date-scales` also shrinks the FiLM.
- New `[weights]` numbers: `film scale / shift` (RMS over channels, max over blocks; a scale of 1 doubles a channel) and `spread` (how much they differ between years).

**Now running:** `nd_overfit_film`: FiLM only (no LoRA), EDM weighting, FiLM lr 1e-5 (the LoRA rate that changed the renders visibly by 2,500 steps without breaking them). Renders every 1,250 steps. Started 13:47; the allocation ends 17:41 (~6,500 steps).

**Step 1,250 (first look)**

| | run 4 (timestep path, capped) at 1,000 | FiLM at 1,000 |
|---|---|---|
| probe loss (SEVA 0.0239) | 0.0223 | 0.0220 |
| date-swap gap, all / old targets | +0.4% / +0.4% | **+1.2% / +1.7%** |
| `x0_mse_high_noise` (SEVA 0.53) | 0.54 | 0.54 |
| pixel difference between dates in renders | 0.02–0.07 | **0.08–0.18** |

- The date now visibly changes the renders, and the gap grows ~3× faster than in run 4. FiLM is still small (scale RMS 0.016), and the high-noise error is unchanged.
- **Not the right change yet:** no sepia or B&W for old dates. At 2015 and 2021 a dark smear appears over the roof in some fire-probe views (in the 2015 row too, so it isn't a clean pre/post-fire split), and `old_from_modern` is blurrier than plain SEVA at every date.

**Step 2,500: the date part stalled.** Swap gap +2.1% at 1,500, then back to +0.2% at 2,500. The part of the FiLM that differs between years stayed at 0.008 from step 1,500, while the part common to all dates grew to 0.03. Renders: no sepia, spire at every date, some smearing; still safe (`x0_mse_high_noise` 0.533). Stopped at ~2,700.

**Why the date may not be needed: most training samples let the model copy the look from an input.** Measured on 2,000 drawn samples (`--min-input-gap` = inputs must be more than N years from the target date):

| | gap 0 (runs 1–5) | gap 3 | gap 30 |
|---|---|---|---|
| samples with an input within 5 years of the targets | 49% | 30% | **0%** |
| old targets: all inputs modern / mixed / all old | 32 / 59 / 9% | 38 / 54 / 8% | 61 / 37 / 1% |
| modern targets: all inputs modern / mixed / all old | 87 / 13 / 0% | 80 / 19 / 1% | 1 / 30 / 69% |

- With gap 0, only old targets with all-modern inputs strictly need the date: ~18% of samples. In the rest, the look can be copied from a same-era input.
- Gap 30 makes every sample need the date. It also trains the fire: inputs from before 1990 all show the spire, so for 2019+ targets the date alone has to remove it.
- Cost: modern targets are almost never paired with modern inputs, so the fire probe (modern inputs → post-fire targets) becomes a generalisation test. Later, a per-sample mix of gaps would cover both.

## Sixth run: date FiLM with inputs >30 years from the targets (2026-11-03)
*`nd_overfit_film_gap30`: as `nd_overfit_film` plus `--min-input-gap 30`. Started 15:15; renders every 1,250 steps; the allocation ends 17:41 (~4,000 steps).* Step-0 probe loss is 0.0264 (not 0.0239): the probe now draws only era-mismatched samples, which are harder.

**Step 1,250**
- Probe: swap gap +2.0% at step 1,000 (old targets +3.5%); loss 0.0264 → 0.0240; `x0_mse_high_noise` 0.527, below its start (0.538).
- **The date now changes what gets drawn, but it draws a blur instead of the right look.** In `modern_from_old` (B&W inputs), old dates and "unknown" give crisp B&W copies of the inputs. 2015 and 2021 give blurred, hazy versions: at 2021 the spire is faint, but still grey, with no colour. `old_from_modern`: no sepia; modern dates blurrier.
- Reading: the model has learned that a modern target should *not* look like old inputs, but not yet what it should look like, so it averages. Checking whether that sharpens into colour by 2,500–3,750.

**Step 2,500: the strongest date use so far, but only partial era looks**

| step | 1,000 | 1,500 | 2,000 | 2,500 |
|---|---|---|---|---|
| swap gap, all | +2.0% | +2.2% | +2.9% | +2.0% |
| swap gap, old targets | +3.5% | +4.4% | **+5.2%** | +3.6% |
| `x0_mse_high_noise` (start 0.538) | 0.527 | 0.530 | 0.534 | 0.531 |
| FiLM scale / shift / spread | 0.015 / 0.018 / 0.006 | 0.022 / 0.029 / 0.007 | 0.027 / 0.034 / 0.008 | 0.031 / 0.036 / 0.008 |

- **Fire probe** (modern inputs): renders now differ by era. 1960 looks like a faded colour print; 1850/1900 look harsh and bleached (blown-out whites), *not* B&W or sepia; 2015/2021 look normal. **The spire is still there at 2021.**
- **`modern_from_old`** (B&W inputs): old dates give grainy B&W copies of the inputs. 2015/2021 give a smoother, cleaner "modern photo" look, but **still grey**, and the framing shifts.
- **`old_from_modern`**: no sepia; blurry and smeared at every date, worse than plain SEVA.
- The date-dependent part of the FiLM is still tiny (spread 0.008), yet it changes the renders a lot (pixel difference 0.13–0.24 between dates).

**What this means so far:** with every sample needing the date, the FiLM picks up the photographic *style* of an era (grain, contrast, fading). It does not yet do the big changes: colour ↔ B&W, and the spire. Those need the model to override what attention copies from the inputs, which a per-channel scale-and-shift may not be able to do alone.

**Step 3,750 (end of the allocation).** Swap gap fell to +0.9% (3,000) and +0.2% (3,500): noisy. In `modern_from_old`, 2015/2021 renders picked up a faint cool tint (saturation 0.10–0.11 vs 0.04 at 1880), the first hint of colour from B&W inputs. Spire still present; fire-probe renders harsher at every date. Checkpoint `step_003750.pt`.

## Seventh run: date FiLM + LoRA, EDM, no input gap: the LoRA broke the geometry (2026-11-03/04)
*`nd_overfit_edm` (job 8135029, queued before FiLM existed, so it picked up the new default): FiLM 15.4M + LoRA 30.1M, lr 1e-5 both, EDM, `min_input_gap` 0. 10,000 steps, 2.3 s/step, 14 GB.*

- **The numbers looked healthy throughout.** Probe loss 0.0239 → 0.0220–0.0235; `x0_mse_high_noise` 0.51–0.52, *below* plain SEVA's 0.53; swap gap +1–1.6%; largest LoRA update 7% of its weight. Neither automatic stop fired.
- **The renders were broken from step 2,500 on.** At 2,500: heavy blur, a sepia cast and broken geometry. At 5,000: sharper but faded. At 7,500–10,000: sharp, but the cathedral is *invented*: plausible Gothic shapes that don't match the cameras. In `modern_from_old`, the rise in saturation (0.04 → 0.35–0.42) is a brown cast over melted structures, not colour.
- **What this means:** the LoRA memorises a generic "Notre-Dame look" and stops following the inputs and cameras. That's independent of the date, so it would very likely recur with the 30-year gap. With LoRA in the model, the probe loss and the high-noise error don't see this kind of damage; only the renders do.


## Code audit (2026-11-10)
*Me plus two independent reviewers, over all code. Fixes in the same commit; tests in `tests/test_audit_fixes.py`.*

**Camera intrinsics after cropping, and Plücker rays: correct.**
- Step for step, our crop/resize intrinsics and Plücker rays match SEVA's (`transform_img_and_K`, `get_plucker_coordinates`); training builds rays with SEVA's own `get_value_dict`. All 934 cached images have the same orientation and aspect as their COLMAP cameras.
- End to end: COLMAP's 3D points, projected into our 576×576 crops, land on the building. After SEVA's camera normalisation each projected point lies within ≤ 0.6 px of its pixel's Plücker ray.

**Bugs fixed**

| | Problem | Effect on past runs | Fix |
|---|---|---|---|
| 1 | Training autocast was **fp16** (torch's CUDA default) with no gradient scaler; the comments said bf16 | Small gradients could underflow, likely worst in the plain-ε runs | Training and the date-gap probe now use bf16 (`TRAIN_AUTOCAST_DTYPE`) |
| 2 | **Gradient clip 1.0** fired in 100% of EDM logging windows and 66–86% of ε windows (one sample per step) | The noise-level weighting barely acted; "EDM fixed it" is confounded | Clipping is now a spike guard: above 10× the median of the last 200 norms (`--grad-clip-factor`; absolute `--grad-clip` optional) |
| 3 | **Lens distortion dropped** (COLMAP SIMPLE_RADIAL `k`) | At crop corners: median 4.5 px (modern) / 8.3 px (old) off; >16 px (2 latent cells) for 11% of views | `k1`/`k2` kept in pose and view tables; `load_view` undistorts while cropping |
| 4 | **Stereo cards** (two side-by-side photos) registered as one camera | 9 of Notre-Dame's 91 old views | `likely_stereo_cards` (pre-1930 and aspect 1.9–2.1, or "stéréo" in the title); dropped by `build_scene_dataset.py` unless `--keep-stereo` |
| 5 | VGGT focal: width/height passed where VGGT-Omega uses height/width | Focal off by ~the aspect ratio for every VGGT pose (none in training yet; rotation results unaffected) | `posing/vggt.py` |
| 6 | VGGT candidates skipped the date-confidence filter | 11 of 80 confident registrations had title-only dates | `unposed_candidates(min_date_confidence=0.3)` |
| 7 | Render probe "unknown" marked only the targets unknown; training marks all views | That contact-sheet row was out of distribution | All views unknown |
| 8 | `min_input_gap` measured from the anchor only | Inputs could be gap − tolerance from a target | Measured from every target; eligibility uses gap + tolerance |
| 9 | Dating: upload-day ceiling rounding plus no time-zone slack; "not before X" parsed as "before X" | 2 of 8,870 Notre-Dame dates changed after re-dating | Ceiling = upload day + 2 days; negation-aware patterns |

Smaller fixes: COLMAP FOV model parameters; probes use the dataset's `camera_scale`; `inspect_checkpoint.py` respects the date cap and reports FiLM sizes; logs show every group's learning rate and the fraction of clipped steps; the VGGT eval uses only COLMAP poses as ground truth; the dataset checks it holds one scene; duplicate render years are ignored.

**Not fixed yet (low impact):** "before X" includes X in the text parser but not in the category parser; every 1 January EXIF is treated as an unset clock; multilingual date strings skip HTML cleanup; some aggressive "painting" medium rules; VGGT intrinsics may refer to the padded canvas when a batch mixes image shapes.

**Rebuilt** `data/datasets/25512_model0`: 925 views (stereo cards out; 2 re-dated views in, 2 failed downloads); 82 before 1970. The old table is kept as `views_before_audit.parquet`.

**Now:** restarting the FiLM-only, 30-year-gap run from scratch with the fixed code (`nd_film_gap30_v2`). Results from before the audit aren't directly comparable.


## Restart after the audit (`nd_film_gap30_v2`): dated renders break (2026-11-17)

Same as the sixth run (FiLM only, inputs >30 years from targets, EDM, 20k steps), but with the audit fixes: bf16, spike-guard clipping only, the rebuilt dataset.

- **Renders with a real date fall apart from step 2,500**: targets become collages of input fragments. By 20k they also turn yellow. Renders with the date set to unknown stay intact (only a slight warm drift).
- **The loss again missed it**: probe loss stayed within ~2% of plain SEVA throughout.
- **Not precision**: the same checkpoints rendered in bf16 (training's precision) instead of fp16 (SEVA's sampler) look identical. Peak activations ~450, no overflow.
- **Not the size of the change**: at step 2,500 the FiLM scale/shift is about half the old run's at step 3,750, which rendered fine.
- **Not the clipping change either**: a 2,500-step rerun with the old clip (`--grad-clip 1.0`, `nd_film_gap30_clip1`) is clean at 1,250 but broken at 2,500 in 3 of 5 target views. Its probe loss is a little better (0.0258 vs 0.0267).
- **Not bf16, not the rebuilt dataset**: two more 2,500-step tests with the old clip, one change each, both break.
  - fp16 on the new data (`--autocast fp16`, `nd_film_gap30_clip1_fp16`): worse. Blotches by step 1,250, even with the date unknown.
  - bf16 on the pre-audit table, no undistortion (`data/datasets/25512_model0_preaudit`, `nd_film_gap30_clip1_olddata`): clean at 1,250, broken at 2,500. Its probe looked healthy (loss −7%, true dates beat unknown ones), so the loss misses this again.
- Undistortion is a small change anyway: only 8 of 925 views have >5% of the crop outside the photo, and the typical largest shift is 4–7 px.
- That run used the old run's exact views, anchors and training noise. So either the one remaining code difference (inputs must be >30 years from *every* target) matters, or the old run's clean renders were partly luck.
- **It was luck.** Rerunning the old code itself (`git worktree` of `c0f9263`, old data, fp16, clip 1.0; `nd_film_gap30_oldcode_repro`) matches the old run's step-0 probe exactly (0.0264). Yet it is already broken at step 1,250, where the original was clean. The trajectories drift apart through GPU nondeterminism. The audit introduced no regression; this setup breaks easily, and the old run happened not to (yet).

**The cause: the inputs' own dates.** I rendered the broken `olddata` checkpoint at step 2,500 with the dates assigned in different ways (fire probe: inputs 2012–17, river-side targets).

| Dates at render time | Result |
|---|---|
| inputs real, targets 2021 (the probe) | broken |
| same, no guidance (cfg 1) | worse: guidance is not the culprit |
| inputs **unknown**, targets 2021 | clean |
| **all** views 2021 | clean |
| inputs real, targets **unknown** | broken |
| inputs real, targets 1850 | broken |
| **all** views 1850 | clean |

- Real modern dates on the inputs break the render, whatever the target date. Unknown input dates, or every view sharing one date, render fine.
- Likely shortcut: with a 30-year gap, a modern-dated input *always* came with old targets in training. Each view's FiLM sees only its own date, so the model learned "modern-dated input ⇒ old target ⇒ discount this input". That strips the input geometry and the targets get hallucinated.
- The old "clean" run (step 2,500) shows the same pattern, only milder: real input dates with unknown or 1850 targets already degrade it. So the problem is the setup, not an audit regression.
- **Not the size of the modulation.** I scaled the date FiLM by α at render time and measured each block's relative activation change ||h′−h||/||h||, separately for input and target views.
  - Step 2,500 (`nd_film_gap30_clip1`): α ≤ 0.25 is clean, but its date effect is barely visible. α = 0.5 starts breaking; α = 1 is broken.
  - **α = 1 with the input views left unmodulated is clean**, for both 1850 and 2021 targets.
  - Per block, inputs change by 2–5% (up to 12–17% in the last two decoder blocks). Giving every view 2021 changes the inputs *more* (4.1% on average) than their real dates do (3.7%), yet renders clean. It is what the modulation does to modern-dated inputs, not how much.
  - Step 20,000 (`nd_film_gap30_v2`): breaks from α = 0.25. With inputs unmodulated, 1850 targets are mostly fine, but 2021 targets turn yellow and partly break. After long training the target pathway misfires too.
- **Root cause, both pathways:** with a 30-year gap every training sample spans two eras. Each view's FiLM (input or target) learns an era *translation* that assumes the other views are from the other era. On a same-era scene (the fire probe) both misfire. Hiding input dates removes the input shortcut, but targets can still learn "my inputs are from another era". Mixing in same-era samples addresses both.
- Fix tests (5,000 steps, same settings as `nd_film_gap30_clip1`):
  - `nd_film_fix_mixgap` (`--input-gap-fraction 0.5`): half the samples take inputs from any date, so an input's date no longer predicts the target era.
  - `nd_film_fix_targetdates` (`--date-views targets`): only targets see their date. The inputs' era must come from how they look.
- New options this round, all off by default: `--autocast`, `--film-freqs`, `--input-gap-fraction`, `--date-views`.

**Fix results so far**
- **Target-only dates (`nd_film_fix_targetdates`): clean at 1,250 and 2,500**, every date row. Every earlier setup was broken by 2,500. The date effect is still mild (rows differ in sky and lighting, no clear era look yet).
  - At 3,750–5,000 the target date starts *changing content*: 2021 gets black, jagged, crane-like shapes over the roof and spire, plus teal tints. That is where the real 2019–24 training photos of these views show cranes, scaffolding and tarps. It is crude, and at 3,750 it also leaked onto 2015 (date aliasing, see below). 1850 and unknown stay clean; one tree-filled view memorises its training photo.
  - Modern targets from old inputs (`modern_from_old`, a training-type case) stay black-and-white, and 2015 fragments. FiLM can only scale/shift channels per view, so it cannot take structure from the inputs while dropping their look. Pushing away from the inputs scrambles the geometry too. **FiLM alone looks capacity-limited for real changes (colour, new structures).**
- **Target-only dates + mixed gap (`nd_film_fix_both`): clean at 1,250, 2,500, 3,750 and 5,000 in every row and view.**
  - At 5,000 on `modern_from_old` (black-and-white inputs from 1905–36): 2015 and 2021 targets turn bluish and the structure stays intact. 1880, 1930 and unknown stay black-and-white. The first clean, date-driven step toward colour.
  - Probe caveat: unknown dates still score slightly lower loss than true ones (−3%), as in every run so far. That includes 2015/2021 (broken with the mixed gap alone) and the tree view (broken with target-only alone). No crane scrambles and no aliasing. The most stable setup so far, but its date effect stays mild (tones, contrast). With same-era samples, 2021 targets often have modern inputs to copy from.
- **Target-only with 4 octaves (`nd_film_fix_targetdates_f4`, no mixed gap): fewer octaves alone don't fix it.** Clean at 2,500. At 5,000, 1960 (an era with few photos) breaks, and 2015/2021 turn strongly golden. True dates are nearest neutral vs unknown here (−0.5%), but high-noise error drifted most (+8.9%). Stopped at ~5,000. Without same-era samples the target pathway still miscalibrates.
- **Combined + LoRA (`nd_film_fix_both_lora`, rank 32, lr 1e-5)**, the only difference being LoRA:
  - Clean on the fire probe at 1,250 and 2,500.
  - Best probe so far at 2,500: loss −7.5%, and high-noise error *below* its start (0.535 vs 0.541). Every other run drifted up.
  - But on `modern_from_old` its 2015/2021 targets are washed out and partly fragmented, with no colour. FiLM alone kept the structure and added a blue tint. LoRA does not (yet) give the hoped-for "structure without appearance". Running to 5,000.
- **Combined resumed from 5,000 toward 12,500 (same run dir)**:
  - Still structurally clean at 6,250, but 2021 picks up a purple-pink cast and 1850/1900 show small dark blotches. Colour drift may be starting (cf. v2's yellow at 20k).
  - At 7,500 the purple has faded and 1850–2015 are clean. One 2021 view has a white-blue patch over the roof (tarps in the real 2021 photos, or a defect?). Swapping old targets' dates now raises loss by 2.1%.
  - At 8,750 degradation creeps in: two 1900 views partly fragment, 2021 is purple again with one dark blob, and contrast turns harsh. **The fix delays and softens the failure but does not remove it in long training.** The remaining per-date memorisation is the likely cause (a single scene makes memorising easy). For now, about 5,000 steps is the safe range.
- **LoRA run, final (5,000): clean on the fire probe at every checkpoint, best probe numbers.** High-noise error keeps falling (0.541 → 0.531) while every FiLM-only run drifts up.
  - At 2021 it adds faint whitish vertical shapes on the left of three views, roughly where the real 2024 photos have cranes/scaffolding, without disturbing the cathedral. Cleaner than FiLM-only's crude black shapes, but faint.
  - Still no colour from old inputs: on `modern_from_old`, 2015 is warm sepia and 2021 hazy; 1880, 1930 and unknown stay black-and-white.
  - Verdict: LoRA is safe with the fixed date setup (it broke geometry before, with the input shortcut present). It doesn't yet give much stronger date effects in 5,000 steps.
  - Resumed to 10,000:
    - 6,250: the faint shapes grow into silvery vertical structures on the left of most views, but for 2015 as well as 2021.
    - 7,500: a large silvery-dark structure on the left of four views for 2015 *and* 2021; 1850–1960 and unknown stay clean. The probe turned: high-noise error 0.531 → 0.549 (now above its start).
    - So LoRA also drifts after ~5,000 steps. Its date-specific content lands on the wrong years (cranes went up in 2018–19, not 2015): the 8-octave date aliasing, made more visible by the extra capacity.
    - 8,750–10,000: the left-side object becomes a large translucent pillar or arch in four views, for 2015 and 2021. The spire stays at 2021 although the real 2021 photos have none.
    - On `modern_from_old` at 8,750, the 2015/2021 targets are washed-out, fragmented sepia smears; the saturation jump to 0.30 is smear, not colour.
    - Probe at 10,000: high-noise error 0.548 (best was 0.531 at 5,000); unknown beats true dates by 2.7%.
    - It seems to learn what modern photos from these viewpoints tend to contain, not what changed over time.
- **Combined + 4 octaves (`nd_film_fix_both_f4`, no LoRA):**
  - 2,500: clean, with a smooth warm trend toward modern dates. True dates nearly neutral vs unknown (−0.2%), high-noise drift +1%.
  - 5,000: mostly clean. One 1850 tile is the memorised tree photo, and 2021 has odd objects on the river.
  - 7,500: still clean in all old-date rows and in 4 of 5 views at 2015/2021 (the tree view goes dark). **No wrong-year objects**: the crane/pillar shapes of the 8-octave runs never appear. 1930 turns sepia on `modern_from_old`. But the probe drifts like the others: unknown beats true by 4.4%, high-noise +4.2%.
  - 10,000: the cathedral is intact in most views, no wrong-year objects, less harsh than the 8-octave run at 10k. But memorisation spreads: the tree view is dark at every date, and its branches intrude into two 1850 views. Probe: unknown beats true by 4.6%, swapping dates no longer hurts, high-noise +4.9%.
  - The smooth encoding fixes the aliasing (date effects follow the calendar) but not the slow single-scene drift.
- **Where this leaves us.**
  - Best setup so far: target-only dates + half mixed gap + 4 octaves, about 5–7.5k steps.
  - The remaining limit is memorising a single scene's photos and viewpoints: every run's probe drifts after ~5,000 steps, LoRA included.
  - Next: train on several dated scenes, and evaluate on held-out views so memorising and learning can be told apart.
  - **Overall: every configuration drifts or adds wrong-year content after ~5,000 steps on this single scene.** The fixed date setup removes the early collapse; the 8-octave aliasing and single-scene memorisation remain.
  - At 10,000 it is clearly degraded: harsh contrast, fragmented 1900 views, dark purple partial breaks on 2021. Probe: unknown dates beat true ones by 4% (the widest gap yet), swapping dates no longer costs anything, high-noise error +6.7%. Stopped at 10,000.
- Next:
  - The combined setup with the smooth 4-octave encoding (`nd_film_fix_both_f4`, 10k steps), to see whether it stays clean past 5,000.
  - The LoRA run resumed from 5,000 to 10,000, the only setup whose probe keeps improving.
- **Mixed gap (`nd_film_fix_mixgap`): only partly fixed.** Clean at 1,250. At 2,500, 1850–1960 are mostly clean but 2015 and 2021 break, although modern inputs with modern targets are common in its training. True dates now *raise* probe loss vs unknown (−3.4%). Stopped at ~3,000 to free a slot.
- **Second problem: per-date memorisation.** With 8 Fourier octaves the shortest period is 1.6 years. Along 2000–2025, the last block's modulation wiggles within 5 years as much as it changes across 5-year averages. In `nd_film_gap30_v2` (20k), 2017 is *further* from 2015 than 2021 is (0.87 vs 0.83). Era style cannot do that; it is memorising individual photos' looks (lighting, exposure) by date. Next: target-only dates with `--film-freqs 4` (periods ≥ 25 years), 10k steps (`nd_film_fix_targetdates_f4`). Also queued: target-only + mixed gap (`nd_film_fix_both`).

---

## Toward multi-scene: second seed, held-out views, multi-scene code (2026-11-18)
**Why:** every single-scene run drifts after ~5,000 steps. Before scaling up, check that the fix isn't luck (a second seed), and measure memorising directly (views the model never trains on).

**Held-out views** (`--holdout-fraction 0.1 --holdout-probe-targets`; `src/timemachine/data/holdout.py`)
- 10% of each era's views, plus the render probes' target views, never appear in a training sample. On Notre-Dame that is 107 views (14 old), leaving 74 of the 89 old anchors.
- So the render-probe sheets of these runs show cameras and photos the model never trained on.
- New **held-out probe** (`HeldoutProbe`, every 1,000 steps, ~1 min). It takes 48 held-out views (12 old), each paired with a trained view of the same era, and compares their denoising loss at the same noise.
  - It reports each group's loss change since step 0, and the difference between the two (`memorisation_gap`).
  - Memorising looks like this: the trained views' loss falls while the held-out views' doesn't.

**Runs (10k steps each)**
- `nd_film_fix_both_f4_seed1`: the best setup again, with seed 1.
- `nd_film_f4_holdout`: the best setup with the hold-out.
- `nd_lora_f4_holdout`: the same plus LoRA rank 32. Does the extra capacity memorise more?
- The date-gap probe now uses the same samples whatever the training seed. For seed-0 runs these are the same samples as before.

**Multi-scene code** (written and tested; not yet run on a GPU)
- `--dataset` takes several dataset folders. Each scene/COLMAP model is its own coordinate frame, and a sample never mixes them.
- Each step picks a scene with probability proportional to its training views^0.5 (`--group-weight-power`), so one big scene can't dominate.
- `--holdout-scenes ID ...` keeps whole scenes for evaluation only. The held-out probe then also reports unseen-scene views (`heldout_change_scene`).
- Render probes need a `scene_id` when training on several scenes.

**Multi-scene data: 13 usable scenes** (`data/datasets/<id>_model0/views_clean.parquet`; full notes in `data/datasets/MULTISCENE_NOTES.md`)
- 15 scenes built with COLMAP poses only, model 0, at most 600 modern views each. Their cached tables predated the audit, so they were re-posed and re-dated first.
- After cleaning, 312 views were dropped: interiors, book pages, multi-picture postcards, misdated photos, missed stereo cards.
- **Training (12 + Notre-Dame):** Royal Palace (Amsterdam), Munttoren (Amsterdam), Arc de Triomphe, Great Sphinx, Sydney Harbour Bridge, Semperoper, Zwinger, Helsinki Cathedral, Vilnius Cathedral, Saint Peter's Square, Taj Mahal, Piazza San Marco. 7,883 views (8,278 with Garnier).
  - The two Amsterdam scenes hold 45% of the old anchors. A scene is sampled in proportion to √views, not by its old photos, so each scene gets 5–9% of samples.
- **Held out: Opéra Garnier.** 395 views, old photos from 1865–1966, including construction.
- **Dropped:**
  - Potala: 87 of 95 old views are one 1938 expedition set, many registered through the archive's caption strip.
  - Lincoln Memorial: model 0 is mostly the statue chamber, and 63 old views are scrapbook clippings all dated 1922.
- Real changes worth probing: the Sphinx half-buried in sand before 1925–36, the Sydney bridge under construction in 1924–32, and Garnier under construction in 1860–75.

**Dating problems found (shared code)**
- **A fake date, 30 Nov 1899** (fixed in `parse_date_text`). Commons shows an all-zero EXIF date this way. It affects 520 files in 126 of the 802 census scenes, almost all digital photos.
  - All of Faisal Mosque's 108 and Cochabamba's 44 "pre-1900" photos were this.
  - Notre-Dame has none. **The census tables in `docs/census/` still include them** and should be re-dated.
- Old photos dated by their scan or upload date: the Commons date field (confidence 0.85) beats an older category year (0.6), and the resolver only records the conflict. About 40 cases in these scenes (e.g. a 1954 royal visit dated 2014).
- Internet Archive book scans are dated by the book's publication year (Notre-Dame has 2).
- The stereo-card filter misses cards outside aspect 1.9–2.1.
- COLMAP sometimes registers archive photos through their printed caption strip. You can spot this from where the matched keypoints sit. Worth filtering before VGGT poses are merged.
- Multi-picture postcards (137 in the two Dresden scenes) register through one small sub-picture. Only a content filter (VLM/CLIP) will catch these in general.

**Multi-scene runs** (25k steps; 10% of each scene's views, the probe targets and all of Garnier held out; `configs/probes_multiscene.json`)
- `ms13_film_f4_holdout`: the best single-scene setup (target-only dates, half mixed gap, 4 octaves), FiLM only.
- `ms13_lora_f4_holdout`: the same plus LoRA rank 32.

## Broken renders, second cause: the dated unconditional CFG branch (2026-11-19)
**Observation.** The held-out runs' render probes, at step 2500, fragment at 1960–2021 on views the model never trained on:
- `nd_film_f4_holdout`, fire probe;
- `ms13_film_f4_holdout`, Notre-Dame fire.

The same checkpoint rendered clean on the nearest *trained* views, which first looked like memorisation.

**But it depends on luck.** Rendering 4 sampler noise seeds per setting:
- Held-out run, 2021: held-out targets broke in 2 of 4 seeds; trained targets in 0 of 4.
- Seed 1, with every view trained: broke in several seeds at 1850 and 2021. Its own probe render (noise seed 1) happened to look clean.
- So **the earlier "clean to ~5,000 steps" verdicts rest on one noise seed and are weaker than stated.**

**Cause.** SEVA's *conditional* prediction alone (guidance 1) gives fragmented renders even for **plain SEVA** with the date FiLM switched off. SEVA needs the unconditional branch.
- The dated model gives the unconditional branch the dates too, but trains it on only 10% of steps. That weakens the guidance, and the output slides toward the fragmented guidance-1 look.

| Inference variant (held-out run, step 2500, 4 seeds) | 1850 | 2021 |
|---|---|---|
| guidance 2, both branches dated (all probes so far) | clean | 2/4 broken |
| guidance 1 (conditional only) | all broken | all broken |
| guidance 2, unconditional branch dates **unknown** | clean | **clean** |
| guidance 2, unconditional branch FiLM off | clean | **clean** |

- Seed 1 is also much better with the undated unconditional branch; some river artefacts remain at 1850.
- So the "held-out views break" signal was mostly this guidance artefact, not memorisation. This fits the held-out probe, whose denoising loss shows no memorisation gap.

**Fix** (`generate_targets(..., uncond_dates="unknown")`; `--render-uncond-dates unknown` for render probes; `--uncond-dates unknown` trains the unconditional branch undated to match)
- The unconditional half of the guidance batch sees every date as unknown. The date then acts like any guided condition: uncond (undated) + 2·(cond (dated) − uncond).
- It works on existing checkpoints without retraining. Defaults stay "same" so the running jobs are unchanged.
- **Next:** re-render the running runs' checkpoints this way, and train with `--uncond-dates unknown` as well.

**Held-out probe so far** (denoising loss vs plain SEVA, held-out minus trained = memorisation gap)
- Notre-Dame, FiLM-only, step 2000: +10.0% vs +10.2%, gap −0.2%.
- Notre-Dame, LoRA, step 1000: +3.6% vs +5.9%, gap −2.3%.
- Multi-scene, FiLM-only, step 5000: +3.6% vs +7.2%, gap −3.7%.
- No memorisation yet in any run. True dates still give a *higher* loss than unknown dates (−4% to −9%), on held-out and trained views alike: the date pathway isn't yet helping the denoiser.

**Update, multi-scene step 5000: the undated unconditional branch isn't enough.**
- `ms13_film_f4_holdout` at step 5000 breaks at 1960, 2015 and 2021 on the Notre-Dame fire probe even with the unconditional branch undated. 2015 turns into green blur; 1850, 1900 and unknown stay clean. The *conditional* prediction itself is now damaged for modern dates.
- **Plain SEVA (step 0) already fragments on the harder multi-scene probes:** Garnier old-from-modern and the buried Sphinx, at guidance 2 with 25 steps.
  - So fragmentation is SEVA's generic failure on hard view sets: large baselines, inputs that disagree, or weak guidance.
  - The date FiLM pushes renders into it more often. **Render probes must be judged against their step-0 render, with several noise seeds.**
- The modulation is small (per-channel scale/shift ≤ 0.02), but about 1.5–2× larger for modern dates than for 1850 or unknown.
- **What the evidence says now:** the date pathway learns modulations that *hurt* SEVA's predictions instead of learning era looks. True dates give a higher denoising loss than unknown dates, even on trained views.

**Single-change runs** (Notre-Dame with the hold-out, 10k steps; all with the unconditional branch trained and rendered undated)

| Run | Change vs A1 | Tests |
|---|---|---|
| `nd_abl_uncond` (A1) | none (vs `nd_film_f4_holdout`: undated uncond branch in training) | training the uncond branch undated |
| `nd_abl_noclip` | no fixed gradient clip (spike guard only) | the clip at 1.0 equalises every update and cancels the noise-level weighting (code audit). The "best setup" inherited it from the diagnosis runs |
| `nd_abl_gapold` | input gap only for old targets (`--gap-old-targets-only`) | half the modern targets get 30+-year-older inputs, so "modern date" can mean "don't trust the inputs" |
| `nd_abl_accum4` | 4 samples per step (3k steps = 12k samples) | is the date signal too noisy with one sample per step? |
| `nd_abl_cosine` | cosine learning-rate decay to 5% (`--lr-schedule cosine`) | does a decaying rate keep the early date effect from washing out? |

**Second seed, step 10000 (`nd_film_fix_both_f4_seed1` vs seed 0).** The date-gap probe uses the same samples for both seeds.

| Step | Unknown vs true (seed 0 / 1) | Swapped vs true (seed 0 / 1) |
|---|---|---|
| 2500 | −0.2% / −3.2% | −0.2% / +0.2% |
| 5000 | −2.7% / −2.0% | −1.1% / −0.4% |
| 10000 | −4.6% / −2.7% | −0.9% / +0.1% |

- In both seeds **the model never learns to use the dates.** Swapping a target's date to the other era costs nothing, and unknown dates beat true ones.
- So target-only dates removed the collapse, not the problem. The clean old-date renders are clean because the date does almost nothing there.
- Seed 1's renders also break in some noise seeds (see above).
- **The single-change runs are judged on this:** swapping dates must start to cost loss while renders stay clean.
- **The date effect is learned early, then washed out.** In the held-out runs, swapping targets' dates to the other era costs +4% to +6% of loss at steps 1000–2000, about 0 by step 5000, and stays there. Meanwhile unknown dates increasingly beat true ones.
  - `nd_film_f4_holdout`: +6.3% (1k) → +4.9% (2k) → +2.7% (3k) → 0% (5k).
  - `nd_abl_uncond`: +4.3% → +4.2% → +1.7%.
  - One explanation: with one sample per step and a constant learning rate, the per-step noise random-walks the FiLM weights and erodes the coarse era effect learned at the start. Larger batches (`nd_abl_accum4`) or learning-rate decay would counter it.
- **The modern dates are the harmful part (suggestive; the date-gap probe has only ~8 samples per era).** Effect of swapping targets' dates to the other era:
  - Modern targets: swapping their dates to 1890 *lowers* the loss in `ms13_film_f4_holdout` at every checkpoint (−3% to −4%), and in `nd_film_f4_holdout` at 10k (−6%).
  - Old targets: swapping their dates to 2015 *raises* it (+3% to +4% at 10k).
  - The no-hold-out seeds are mixed around 0.
  - This fits the renders, which break at modern dates, and the `--gap-old-targets-only` hypothesis. The probe needs more samples (32+) for this split.
- `nd_film_f4_holdout` at 10k, held-out probe (plain average): held-out views +7.8%, trained +4.9% vs plain SEVA. True dates hurt by 12–14%. Training made the denoiser worse, not better.
- **FiLM-only vs FiLM + LoRA on Notre-Dame with the hold-out, 10k steps** (held-out probe, plain average, vs plain SEVA)

  | | Trained views | Held-out views | True vs unknown dates | High-noise error |
  |---|---|---|---|---|
  | FiLM only (`nd_film_f4_holdout`) | +4.9% (worse) | +7.8% | −12% to −14% | +11% |
  | FiLM + LoRA (`nd_lora_f4_holdout`) | −3.4% (better) | −0.3% | −3% to −6% | ±0% |

  - LoRA is the healthier model. It improves the denoiser, its renders were clean on held-out targets, and the date FiLM does less harm beside it.
  - Its held-out gain fades (−2.1% at 2k → −0.3% at 10k) while trained views keep improving. That is the first **memorisation** signal, about 3 points by 10k.
  - Neither model uses the dates yet: swapping them costs about 0 in the date-gap probe.
- **`ms13_film_f4_holdout` stopped at 15k.** It got steadily worse (held-out / trained vs plain SEVA): +3.6% / +7.2% at 5k, +5.1% / +9.2% at 10k, +8.3% / +11.4% at 15k. True dates hurt by 9–10%. It can be resumed from `latest.pt`.
- Its slot went to **`ms13_film_noclip_uncond`**: multi-scene FiLM-only, no fixed clip, unconditional branch trained and rendered undated, date-gap probe with 32 samples.
  - Why: at 5k, `nd_abl_noclip` is the healthiest Notre-Dame run. Its date-gap probe shows unknown −0.2% and swapped +1.5%, and the high-noise error is flat.

## Separating scene adaptation from temporal learning (2026-11-19, after advisor review)
**The question:** multi-scene LoRA improves held-out views (−6.3% vs −7.1% on trained views, at 4k). That could be ordinary adaptation to these scenes and photo styles rather than anything temporal. Learning a real date dependence is unsolved: in every run so far, swapping dates costs ~0.

**Milestone:** the true date must reliably beat a wrong-era date on unseen, temporally informative photos, without losing the LoRA gains.

**Runs** (multi-scene, identical except as noted)

| Run | What it adds |
|---|---|
| `ms13_lora_f4_holdout` | date FiLM + LoRA (the date-conditioned baseline) |
| `ms13_lora_nodate_holdout` | **no date conditioning** (`--date-injection none`), LoRA only: the control. If the dated run matches it, the gains aren't temporal |
| `ms13_lora_rank_events` | the baseline + a date-discrimination ranking loss + change-event sampling |
| `ms13_lora_rank` / `ms13_lora_events` | each of the two alone, to tell which matters |

**New measurements**
- The held-out probe now also scores the targets with a **wrong-era date** (`*_rel_swapped`, > 0 when the true date wins), split by era and by held-out view vs held-out scene. A no-date model scores 0.
- `scripts/eval_heldout.py` scores checkpoints from several runs and plain SEVA on *identical* held-out samples. It rebuilds the run's hold-out and checks it against `heldout.json`.

**New training options** (off by default)
- `--rank-weight`: on half the conditional steps with σ ≥ 1, the same noised targets are also scored with a wrong-era date. The term is relu(margin − log(loss_wrong / loss_true)) per target view, margin 0.05. It is scale-free across noise levels and costs one extra forward pass.
- `--event SCENE:YEAR --event-prob 0.25`: samples across a known physical change, with targets on one side and inputs on the other, preferring inputs within 15 years of the change.
  - Events: Notre-Dame fire (2019.29), Sphinx excavation (~1930), Sydney bridge completion (1932.2).
  - Example draws: Notre-Dame 2011–12 targets with 2019–22 inputs; Sydney 1930–31 (under construction) with 1932–41; Sphinx 1850 (buried) with 1935–41.
- The date encoding is already coarse in all current runs: 4 octaves, periods ≥ 25 years. The 1.6-year features were the 8-octave setting, dropped on 2026-11-18. Era bins would be the next step if this isn't coarse enough.
- Per the review, no further FiLM-only stability runs. The FiLM ablations already running are left to finish.
- **Dated multi-scene LoRA does not use the dates** (`scripts/eval_heldout.py`, 64 held-out pairs, vs plain SEVA on identical samples)

  | `ms13_lora_f4_holdout` | Held-out views | Trained views | Held-out scene (Garnier) | True vs unknown | **True vs wrong-era** |
  |---|---|---|---|---|---|
  | step 2500 | −5.5% | −6.5% | −0.6% | −3.2% | **+0.5%** |
  | step 4500 | −3.9% | −5.6% | +1.7% | −3.9% | **−0.6%** |

  - A wrong-era date is as good as the true one (±1%), on old and modern views and on the unseen scene alike. **The LoRA gains are scene/photo adaptation, not temporal learning**, as the review suspected. The no-date control should match them.
  - The gains also fade with training (held-out −5.5% → −3.9%; Garnier −0.6% → +1.7%).
- **`nd_abl_noclip` degrades by 10k too:** about +10% vs plain SEVA on held-out and trained views, true dates −9%. No FiLM-only variant tested here stays healthy.
- **FiLM-only single-change runs at 10k** (Notre-Dame with the hold-out; held-out probe, plain average, vs plain SEVA)

  | Run | Held-out | Trained | True vs unknown | Date-gap swap | High-noise error |
  |---|---|---|---|---|---|
  | `nd_film_f4_holdout` (reference) | +7.8% | +4.9% | −12% / −14% | −0.3% | 0.620 |
  | `nd_abl_uncond` (undated uncond branch) | +2.1% | +0.8% | −7% / −8% | −1.8% | 0.631 |
  | `nd_abl_noclip` (+ no clip) | +2.7% | +2.2% | −9% / −10% | +0.4% | 0.578 |
  | **`nd_abl_cosine`** (+ cosine decay) | **−4.2%** | **−5.4%** | −3% | −1.3% | 0.594 |

  - **A decaying learning rate stops the drift:** it is the only FiLM-only run that ends better than plain SEVA. This fits the explanation that per-step noise at a constant rate washes things out. Use cosine decay by default from now on.
  - **But none of them uses the dates:** the swap costs ~0 everywhere. Stability isn't temporal learning, which is the review's point.
- **First sign of date discrimination:** `ms13_lora_rank_events` at step 700. The training-sample log ratio (wrong-era / true loss) is +0.014 and rising. The date-gap probe swap is +0.8% at step 500. Early and on training samples only; held-out checks come at 2.5k.
- `nd_abl_gapold` stopped at ~5k: at 5k it was +9% / +8% (held-out / trained) vs plain SEVA, with true dates −8%. No better than the others, and it was preempted repeatedly. `nd_abl_accum4` stays queued.

### Step 2500: no-date control vs dated LoRA vs ranking + events (`scripts/eval_heldout.py`, identical held-out samples, vs plain SEVA)

| Run | Held-out | Trained | Garnier | True vs unknown | **True vs wrong-era** | old | modern | Garnier |
|---|---|---|---|---|---|---|---|---|
| `ms13_lora_f4_holdout` (dated LoRA) | −5.5% | −6.5% | −0.6% | −3.2% | +0.5% | +0.6% | +0.5% | +0.6% |
| `ms13_lora_nodate_holdout` (no dates) | **−7.4%** | **−8.0%** | **−2.8%** | 0 | 0 | 0 | 0 | 0 |
| `ms13_lora_rank_events` | −2.5% | −3.3% | +6.9% | −5.9% | **+2.4%** | **+4.9%** | +0.7% | **+4.1%** |

- **The no-date control beats the dated LoRA** on every denoising number. The LoRA gains are scene/photo adaptation, and the date FiLM costs about 2 points. This is the review's suspicion, confirmed.
- **Ranking + events gives the first temporal signal that carries over to unseen data.** The true date beats a wrong-era date on held-out views (+4.9% for old ones) and on the unseen scene (+4.1%), as much as on trained views.
- **But it got there the wrong way.** Unknown still beats the true date (−5.9%), the denoising gains shrink, Garnier ends 6.9% worse than plain SEVA, and **renders at modern dates break badly** (purple Sphinx, red Sydney, fragmented Notre-Dame).
  - The hinge can be satisfied by making wrong-date predictions *worse*. That is harmful for generation, where any date must give a plausible image.
- **Fix: `--rank-detach`.** No gradient through the reference branch, so only the true-date prediction can improve. In effect it puts extra denoising weight where the date isn't helping yet. It also runs the reference without a backward pass, which is cheaper.
- `ms13_lora_rank_events` and `ms13_lora_rank` stopped at ~2.6k. Replaced by:
  - `ms13_lora_events_rankdet`: events + detached rank vs a **wrong-era** date.
  - `ms13_lora_events_rankunk`: events + detached rank vs **unknown**. The true date must add information over no date.
  - `ms13_lora_events` (events only, no ranking) continues.
- **Correction: `ms13_lora_rank_events` (and `ms13_lora_rank`) were trained with a gradient bug.** Their numbers above don't test the ranking idea.
  - The date-FiLM layers read the dates of the *last* forward pass when gradient checkpointing recomputes a block in the backward pass.
  - The ranking loss ran a second pass (the wrong-era reference) after the true-date pass and before the backward. So the true-date loss's gradients were computed with the wrong-era dates' features.
  - It failed silently while both passes had gradients. It surfaced as a checkpoint error once the reference was detached.
  - This may explain the broken renders and the lost gains.
  - **Fixed:** the reference pass now runs first and without gradients, and the true-date pass runs last. Ranking without a detached reference is refused. A new test checks that checkpointed gradients are unaffected by an earlier gradient-free pass with other dates.
  - The ranking runs were restarted: `ms13_lora_events_rankdet` (vs wrong-era) and `ms13_lora_events_rankunk` (vs unknown).
  - The rest of the step-2500 table stands: the no-date control beats the dated LoRA.

---

## Full pipeline audit: every run so far trained on numerical noise (2026-11-19)
*Full report with tables: `docs/audit_2026-11-19.md`. Raw evidence: `data/audit_2026-11-19/`.*

**Root cause: half-precision training.**
- In bf16 (all runs since 11-10) and fp16 (all runs before), the gradient reaching the adapters is rounding noise. Its norm is ~1,000 against a true (fp32) norm of ~1, and its direction is unrelated to the true one (cosine ≈ 0).
- Why: SEVA predicts the noise, and at medium and high noise its error is ~1–2% of its output, the same size as bf16's rounding. The loss is built from that small difference.
- The logs' gradient norms of 700–1,200 ("clipped 100%") were this noise, not the loss weighting.
- **Proof:** on one fixed sample with fixed noise, the loss stalls at −6% to −11% in bf16 and falls −34% to −40% in 60 steps in fp32. Two same-seed bf16 runs move their weights in unrelated directions.
- **Fixed:** training and loss probes now run in fp32 (`--autocast fp32`, the default). Cost: 16 GB instead of 14 GB, ~5 s per step instead of 2.8 s.

**Second cause: we rendered with 8 frames.**
- SEVA always runs 21 frames (padding by repeating the last one); fewer gives "abnormal color overflows" (its paper).
- Plain SEVA scores 9.6 dB at 8 frames and 11.6 dB at 21 on easy samples. At 21 frames it is coherent even with guidance 1.
- The "fragmented renders" were largely this. **Fixed:** renders pad to 21 frames by default.

**Third cause: ill-posed training samples.**
- 24–74% of training targets (lenient–strict count) show something none of the 3 inputs shows; 42–97% for modern targets with old inputs. An easy same-date baseline: 9–25%.
- On training-style samples plain SEVA barely uses the cameras: scrambling the target cameras costs 4–6% (12–18% on easy samples).
- Two of the three Notre-Dame probes (`old_from_modern`, `modern_from_old`) are ill-posed or nearly so. **Not fixed yet.**

**Evaluation artefacts**
- Seed 0 (every training probe) has an extreme noise draw on the 4th target, which renders dark in 9 of 9 plain-SEVA sheets. Use several seeds.
- Guidance 2.0 and 25 steps are at the weak end of SEVA's recommendations (2–5, 50 steps).

**Checked and correct:** cameras (0.36 px median reprojection error vs COLMAP), conventions, the conditioning tensors (equal to SEVA's own sampler's), the loss scaling, weights/VAE/CLIP/schedule, gradient plumbing, save/load/resume.

**Smaller fixes made:** LoRA no longer targets the cross-attention query/key (zero gradient, 11% of its parameters); resume refuses a precision mismatch; FiLM arithmetic in fp32; stale comments.

**Open data issues:** 7% of Notre-Dame views are sideways (EXIF rotation ignored), 18 duplicate photo pairs, camera scale outside SEVA's training range in >90% of samples, the fixed gradient clip at 1.0.

**What this means for everything above this section**
- No conclusion about *learning* from any earlier run stands (date-pathway designs, weightings, learning rates, FiLM vs LoRA, the no-date control, the ranking loss): they compared random walks.
- Findings about plain SEVA and the data stand (census, dating, camera chain).

**Proof the fixed pipeline learns** (two pinned samples, real trainer, 600 steps): 
- Loss on the two training samples: −32% to −51% in fp32 (LoRA + FiLM, lr 1e-4); the same run in bf16 was +3% to +25%.
- The fp32 run's renders reproduce its training photos (night-lit facade, blossom tree, rose window); the bf16 run renders white mush.
- **First real date effect:** with FiLM only, the correct date beats the wrong one by 4–8 points for the pre-fire sample at every noise level.

---

## New direction: MVGenMaster and Photo Tourism (2026-11-19)
*Plan: `docs/mvgenmaster_time_plan.md`. Setup: `external/MVGenMaster/` (not in git), `slurm/mvgen_gpu.sbatch`.*

**Why MVGenMaster**
- It is built on SD 2.1 with v-prediction, was trained on MegaScenes among 12 sources, and its training code is released (Apache-2.0).
- Besides camera rays it conditions on the input views warped into each target by depth.
- It has a "domain switcher" (one embedding per data source, added to the timestep embedding) made for MegaScenes' varying appearance: a natural slot for a date.

**Setup notes**
- `stabilityai/stable-diffusion-2-1` is gone from Hugging Face (401). The mirror `sd2-community/stable-diffusion-2-1` works; MVGenMaster needs its VAE, UNet config and scheduler.
- Its own environment (Python 3.10, torch 2.5.1) lives in `external/MVGenMaster/.venv`, separate from ours.
- The authors' demo (3 photos → 25 views) runs in 41 s on an A6000.

**Tested on our Notre-Dame view sets** (our COLMAP cameras, real target photos; `scripts/mvgen/nvs_test.py`)

| View set | SEVA, 8 frames | SEVA, 21 frames | MVGenMaster |
|---|---|---|---|
| Easy same-year sets (mean of 4) | 9.6 dB | 11.6 dB | **12.6 dB** |
| Fire probe | 9.9 dB | 10.8 dB | **11.8 dB** |

- Renders are sharp and stable across noise seeds, with no fragmentation.
- It needs a depth map per input view. Depth-Pro scaled to the view's COLMAP points agrees with them to 1–14% (median) on easy sets.
- Its domain label "others" scores ~1 dB above "megascenes" on same-date sets.

**Its gradient is sound in bf16** (`scripts/mvgen/grad_precision.py`): cosine 0.998–1.000 with the fp32 gradient at every noise level, where SEVA's was ~0. fp16 without a gradient scaler is not (0.77–0.82).

**Photo Tourism: which scene** (analysis in `data/phototourism/`; `analysis/scene_comparison.csv` compares 35 scenes)
- **Usable today:**
  - **MegaDepth** (196 Flickr landmarks): already on the cluster at `/fs/vulcan-datasets/MegaDepth`, original files with EXIF dates for 75% of registered images.
  - **IMC-PhotoTourism** (the NeRF-W scenes): dates only through YFCC100M lookups; range 2004–2014.
  - **NeuSC 5Pointz**: change-focused, 6,502 dated images, 2006–2014.
  - The other change-focused sets (Scene Chronology, Times Square) were never released.
- **Nothing in Photo Tourism spans more than about a decade.** MegaDepth ends in early 2017, IMC in 2014. For decades of range, MegaScenes stays the only source.
- **Recommended: Milan Cathedral (MegaDepth 0063).** View table built: `data/phototourism/views/megadepth_0063_milan_cathedral/views.parquet` (loads in `DatedViewsDataset`).
  - 1,696 posed views, 1,045 with an exact EXIF date (2002–2017; 90% in 2007–2016). The other 651 only have an upper bound from the upload time.
  - **A real change tied to the date:** the facade is fully wrapped in scaffolding in late 2005, half covered by hoardings in 2006, hoardings at the base in 2007–08, clean from January 2009 (checked on the date-sorted contact sheet).
  - Good overlap: 88% of dated images share 30+ COLMAP points with 5 or more images in each of three eras. Poses: 0.54 px median reprojection error.
  - Limits: the change is local (the lower 15–50% of the facade) and over by 2009; only 33 dated images are from 2005–06. Expect a before/after step, not smooth date control.
- **Runner-up: 5Pointz** (murals repainted continuously): far denser change, but photos years apart share almost no feature points, so it is a harder first overfit.

**Scene decision: Notre-Dame from MegaScenes** (2026-11-19; measured on one rubric with an independent verifier per scene, `data/scene_choice_2026-11-19/`; every headline number reproduced)

| | MegaScenes Notre-Dame | Milan (MegaDepth 0063) | Photo Tourism's Notre-Dame (MegaDepth 0004) |
|---|---|---|---|
| Views dated to within a year | 3,262 | 1,045 | 4,186 |
| Dates (5th–95th percentile) | 2004.9–2019.6 (min 1840) | 2007.1–2016.1 | 2004.8–2009.6 |
| Views in the changed state | 460 after the fire | 41 wrapped, 338 with base hoardings | 105 show a scaffold on one tower |
| Frame share that changes | 4–10% in side and apse views; <1% frontal | 7–26%, clearly visible in ~100 views | small |
| Changed-state views with 5+ co-visible views from another state | 87% | 91–95% | 96–97% |
| Sideways photos | 7.2% | 0.8% | 3.4% |

- **Why Notre-Dame:** the most photos of a changed state, the only range beyond a decade, and everything already built targets it. Photo Tourism's own Notre-Dame ends in 2009, ten years before the fire.
- **The fire is invisible from the front:** under 1% of the frame in west-facade views. About 37% of modern views show the spire or roof at all; about 70% of after-fire views show the damage. Probes and samples should use south, south-east and apse views.
- **"Before the spire" is not a usable state:** 5 exactly dated views, one misdated, two the same photograph.
- **Only 58 exactly dated views predate 1970.** Old photos' co-visibility with other periods is split: 20 of 53 (west facade) see 30+ views of other states, 14 (river side) see none.
- **Date fixes** (`configs/notre_dame_date_fixes.json`): the fire boundary is 2019.2849, not 2019.29 (68 fire-day photos were on the wrong side); 6 "after-fire" photos show the intact spire (3 confirmed, 3 suspected) and one "1851" photo shows the spire.
- Milan is kept as a clean modern control: an orderly restoration sequence, almost no sideways photos, excellent overlap.

**Census re-dated without the fake 30 Nov 1899 date** (2026-11-19; all 802 scenes re-dated from cached metadata, no downloads; old tables in `data/census_before_redate_2026-11-19/`)
- The fake date is gone: 520 files before, 0 now. 380 are now dated 2000+ from their categories, 137 are undated, 3 fall in 1900–1969.
- Pre-1900 photos: 13,653 → 13,179 (posed 2,242 → 1,956). Pre-1970: 45.4K → 44.9K (posed 7.0K → 6.7K).
- Scenes passing the gate: posed in one model 148 → 140; with our own posing 482 → 471. Every lost scene is a fake-date scene (Faisal Mosque 108 → 0, the six Cochabamba churches, Hiran Minar). 42 scenes lose all their "pre-1900" photos.
- **Side effect of the audit's upload-date change** (`resolve.py`, a capture date may end up to 2 days after the upload day, was 1): 407 photos went from undated to 2000+ (good), but 36 files moved from an older era to 2000+, and about 23 of those look like real old images now dated by their upload day (judged from file names only: Brooklyn Bridge 1896, Moscow 1986, St Paul's 1951). No gate changed. Worth revisiting.
- Other suspicious exact days are still counted and not yet checked: 1900-01-01 (111 files), 1865-01-01 (57), 1980-01-01 (56).
- The totals table higher up in this file (15.8K / 2.3K pre-1900) came from an earlier run; the committed tables had 13,653 / 2,242 before this fix.
- `scripts/census.py` crashed on a backup file in the dates folder (`25512_before_audit.parquet`); it now skips names that are not scene ids.


**MVGenMaster date conditioning: code written, checked against the authors' training code** (2026-11-19, evening; code in `src/timemachine/mvgen/`, nothing trained yet)
- **What exists:** date module, model wrapper with LoRA, training step, dataset loader with a co-visibility sampler, trainer, probes, train and render scripts. Tests run in MVGenMaster's own environment.
- **First GPU check passed:** our training step gives the same loss as a hand-built copy of the authors' step at five noise levels (within 0.3%). The comparison is against our copy of `train.py`, not their script run end to end.
- **A gradient hazard closed before training.** With gradient checkpointing, a second forward pass before `backward()` would silently corrupt the date gradients (34% error on a toy model). Each forward pass now carries its own date embedding, attached to its timestep-embedding tensor, so this cannot happen. This is the same kind of bug as the SEVA ranking loss.
- **Dataset rebuilt:** `data/datasets/25512_model0_v2`, 1,314 views. Usable before the second round of date fixes: 420 after the fire, 632 modern, 88 before 1970. Excluded: 100 sideways, 44 taken on the day of the fire, 23 duplicates, 7 misdated.
- **More wrong dates found by review:** five "2020" photos are film scans showing the intact spire (upload dates), one photo of the burnt building is dated before the fire, two are before/after montages. After-fire photos now need a precise date unless checked by eye. The few pre-1859 views are dropped (two show the spire, two are one photo).

**How our code compares with the authors' training code** (125 details compared, each difference re-checked by a second reader; full list in `data/mvgen_fidelity/audit_results.json`)
- **The authors released** `train.py`, two stage configs and seven dataset classes. The MegaScenes dataset class is not released, so how they chose a scene's 8 views is unknown.
- **Matches (73 details):** architecture and weight loading, noise schedule, loss on target views only, clean reference views, camera embedding, where an extra condition is added, dropout rates.
- **Scene scale differs.** They scale a scene once, by the extent of all its cameras (Notre-Dame: ×0.087). We used their demo's per-sample rule (×1.0 on our test sets, and a different factor from draw to draw on about 40% of sampled sets). Fix: one constant factor per scene; value to be chosen on GPU.
- **Domain label differs.** They trained MegaScenes with label 1; we used 0 because it scored about 1 dB higher on same-date sets. That score rewards copying the input's look. To be decided from the loss on cross-period sets.
- **Far-depth compression differs.** Their training compresses far depth once; their demo, which we followed, does it twice for the warped pixels (sky shifts about 2 px on our gate sets, 43 px on one close-up set). Fix: once.
- **Earlier MVGenMaster numbers** (12.6 dB and others) were measured the demo's way: priors computed in half precision, double compression, per-sample scale. They are a sanity check, not a reference.
- **Half-precision warps are fragile.** The code that warps the input photos is not protected from autocast. Under bf16 the warped pixels covered 24% of the targets instead of 70% in a CPU test. Our training and render paths compute it in fp32.
- **Only the date encoder trains by default,** against frozen layers that read it. An arm that also adapts those layers is added.
- **The released weights are probably not a smooth average.** Their EMA was accumulated in half precision, which loses small updates. This is an inference from the code, not measured.
- **Authors' recipe, for reference:** learning rate 5e-5 then 2.5e-5, constant after 500 warm-up steps, clip 1.0, 8 views, 3 reference views in stage 1 and 1–3 in stage 2, fp16 through DeepSpeed.

**Why appearance has to be modelled** (2026-11-19; a design discussion, not yet built)
- Fitting six numbers per image (gain and offset per colour channel) to MVGenMaster's renders removes 56% of their squared error against the real photos (12.4 → 15.7 dB; an oracle fit on 5 view sets, so an upper bound). The fire changes only 4–10% of the frame.
- The date is confounded with the look: 65% of after-fire photos were taken in spring (26% of modern ones), and ten days hold 56% of the after-fire photos. 61% of pre-1970 photos are monochrome.
- Two published designs cover this: WildCAT3D (an appearance encoder for multi-view diffusion on in-the-wild photos) and Neural Scene Chronology (per-image lighting code plus time as learned step functions; a Fourier time code overfitted per-image lighting there).
- Proposal: learn appearance first (no dates needed, so any posed photos can be used), then time as step functions with appearance given. The main risk is that an appearance code computed from the target leaks the building's state.

**Fixes after review, and the two open choices settled on the GPU** (2026-11-19, night; `scripts/mvgen/baseline_choices.py`, `scripts/mvgen/nvs_test.py --scales`)
- **All review and audit fixes are in** (323 tests pass; two independent re-checks found no major defect). Far depth is compressed once, the scene scale is one constant per scene, a resumed run refuses changed settings, and the size of the date embedding is logged at every step.
- **Dataset after the second round of date fixes:** 1,314 views, **1,115 usable: 77 before 1970, 630 modern, 408 after the fire.** Excluded: 100 sideways, 44 fire-day, 26 duplicates, 15 misdated, 7 before the spire, 5 after-fire photos with an uncertain date, 2 montages.
- **The sampler now redraws references until every target is seen by one** (before: 5% of targets were seen by none). Side effect: old (pre-1970) targets almost never get references only from other periods (23% → 3% of old-target samples). The fire periods are barely affected (about 50% stay cross-period).
- **Scene scale does not matter to the released model.** Loss on 11 fixed view sets changes by at most 0.2% between factors 0.087, 0.2, 0.5 and 1; render PSNR by at most 0.05 dB between 0.087 and 3. We use the authors' training rule (×0.087 for Notre-Dame). Caveat: measured with the 3D priors present; without them the camera rays carry all the geometry and the scale may matter.

| Released model, 5 view sets, 2 seeds | Label 0 (regular) | Label 1 (MegaScenes) |
|---|---|---|
| Render PSNR, scale 1 | 12.42 dB | 11.64 dB |
| Render PSNR, scale 0.087 | 12.37 dB | 11.61 dB |
| Training loss, 11 view sets (lower is better) | 0.3508 | 0.3497 (−0.4% ± 0.4%) |

- **Domain label: 0 stays the default.** The two labels give the same training loss, but label 1 renders 0.8 dB lower. Reading: with label 1 the model draws a look of its own instead of copying the references' look, which costs PSNR against the real photo but not loss. That may make label 1 the better start for date control; it is compared at gate 2.
- **The step check was re-recorded** after the depth fix: losses moved by up to 0.6%, most at high noise.
- **Smoke test on the GPU:** at step 0 the adapted model equals the released one (true, swapped and unknown dates give the same loss); a step takes 2.6 s and 8.5 GB with 8 views, rank-32 LoRA and bf16.
- **The date embedding grows fast at learning rate 1e-4:** 15–19% of the timestep embedding's size after 6 steps. Gate 1 is run at 1e-4 and at 1e-5 to see whether this hurts.
- **GPU jobs now run from a task list** (`slurm/mvgen_gpu_queue.sbatch`): one job holds a GPU and runs the next task appended to a file, so short runs do not each wait hours in the queue.

**Gate 1 passed: the trainer overfits one fixed sample** (2026-11-20; `data/experiments/mvgen/train/gate1*`; date encoder + rank-32 LoRA, 400 steps of 1 s, bf16, same 3 pre-fire reference photos in both runs)

| Run | Targets | Loss on the sample (fixed noise) | Render PSNR, step 0 → 400 |
|---|---|---|---|
| gate1a_post | 5 photos after the fire | −48% | 12.5 → **19.4 dB** |
| gate1a_post, date learning rate 1e-5 | same | −48% | 12.5 → 19.4 dB |
| gate1b_pre | 5 photos before the fire | −39% | 13.3 → **19.3 dB** |

- **The structure changes as it should (1a).** The released model draws the intact spire and roof it sees in the references. After 400 steps the renders show no spire, the bare vaults and the scaffold (still blurred), from references that all predate the fire. Pictures: `gate1a_post/gate1a_summary.jpg`.
- **Most of the gain is the photos' look, not the building (1b).** With targets from the references' own period the structure was already right at step 0. The 6 dB gained are sky, light and the colour of the water (one target shows the Seine in flood). This is the appearance problem in one picture: `gate1b_pre/gate1b_summary.jpg`.
- **The loss is still falling at step 400** (about −10% per 100 steps), so "more than half" needs a few hundred more steps; the renders already reproduce the targets.
- **Dates play no part yet, as expected for one sample:** true and swapped dates give the same render (19.4 vs 19.4 dB) and the same loss (+0.6%). One sample can be memorised without reading the date. Gate 2 is the test of that.
- **The date learning rate did not matter here.** At 1e-4 the date embedding grew to 0.9–1.1 times the timestep embedding without harming the renders; at 1e-5 it stayed at 0.13 with the same result.
- **Depth for the whole dataset is done:** 1,205 of 1,314 views have reliable depth (1,019 of the 1,115 usable ones); median disagreement with COLMAP's points 2–3%.
- The step check varies by up to 0.8% from run to run because it re-estimates depth each time (Depth-Pro is not deterministic on the GPU). Training reads stored depth maps.

**What the 3D priors are worth to the released model** (2026-11-20; `baseline_choices.py --cross-date-pixels`, 11 fixed view sets, fp32, fixed noise; `data/experiments/mvgen/baseline_cross_date/`)

| Loss change against "all priors kept" | Same-period sets (5) | Cross-period sets (4) |
|---|---|---|
| Warped pixels dropped for cross-date targets, 3D coordinates kept | +10.7% ± 3.4% | +6.5% ± 1.3% |
| Both priors dropped for cross-date targets | +10.7% ± 3.4% | +6.6% ± 1.3% |
| No priors at all (references included) | +20.8% ± 4.8% | +7.0% ± 1.5% |

- **The 3D coordinates alone are worth nothing to the released model.** Keeping them while dropping the warped pixels gives the same loss as dropping both. So option B of the plan (keep coordinates, drop pixels) has no advantage over dropping both at the start of training.
- **The priors help half as much when the references are from another period** (7% against 21%). The warped pixels then show the wrong state and the wrong light.
- "Same-period" sets also lose here because the default rule marks a target as cross-date when every reference is more than a year away.
- Without priors the scene scale matters slightly (factor 1.0 is about 1% better than 0.087); with priors it does not.

**Gate 2 not passed as designed: with two samples the date learns the sky, not the building** (2026-11-20; `data/experiments/mvgen/train/gate2_*`; two fixed samples with the same 3 pre-fire reference photos, one with 5 after-fire targets and one with 5 pre-fire targets)

| Run (800 steps unless noted) | Loss on the two samples | Wrong date costs | Render PSNR, true date vs wrong date |
|---|---|---|---|
| Date encoder only (400 steps) | −4.6% | +3.5% | after-fire set 12.3 vs 12.0 dB; pre-fire set 14.0 vs 13.3 dB |
| Date + LoRA, incl. the layers that read the date | −56% | **+18%** | after-fire set 20.1 vs 19.2 dB; pre-fire set 20.6 vs 20.3 dB |

- **The date is used, and more so the longer it trains:** the loss with a wrong date rises steadily (+0.5% at step 50, +7% at 400, +18% at 800).
- **But it changes the look, not the structure.** Rendering the after-fire cameras with a 2013 date turns the grey sky blue and leaves the spire missing; rendering the pre-fire cameras with a 2019 date greys the light and leaves the spire standing. Pictures: `gate2_lora32_temb/gate2_set0_summary.jpg`, `gate2_set1_summary.jpg`.
- **Why:** the two samples have different cameras, so the model can tell them apart, and memorise each building state, without the date. Nothing in a two-sample test forces the state onto the date. What is left for the date is what the cameras do not explain: the photos' light. This is the appearance confound showing up in the smallest possible test.
- **The date encoder alone has little leverage** on the frozen model (−4.6% loss), and its embedding grew to 3 times the timestep embedding. Adapting the layers that read the date (LoRA on `time_emb_proj`) triples the date's effect at equal steps (wrong-date cost +3.4% vs +1.1% at step 250).
- **Gate 3 is the real test of state.** Over the whole scene the cameras of the two periods overlap, so only the date (or looks that go with a date, like spring leaves) can say whether the spire stands. `gate3_keep` runs now: 998 training views, 140 held out, date + rank-32 LoRA, 4,000 steps.

**Gate 3, first look (step 1,000 of 4,000): on held-out views the date starts to move the spire** (2026-11-20; `data/experiments/mvgen/train/gate3_keep`; whole scene, 998 training views, 140 held out, date + rank-32 LoRA incl. the layers that read the date, 2 samples per step)
- **Pictures:** rendering held-out after-fire targets from pre-fire references, the true date fades the spire or removes it; a 2013 date draws it intact. The roof is still there in both. `gate3_keep/step1000_set0_summary.jpg`.
- **This is the first time a date changes the building, not the look,** and it is on photos the run never trained on.
- **Whole-image numbers do not show it yet:** PSNR 12.6 dB (true date) vs 12.5 dB (wrong date); a wrong date costs 0.15% in held-out loss. The spire is a few percent of the frame. A score restricted to the spire and roof region is needed to track this.
- Training and held-out loss fall together (both −3.0% at step 1,000), so the run is not memorising.
- Second arm running: `gate3_dropall` (both priors dropped for targets from another period than the references). At step 0 this costs the released model 0.4 dB on the after-fire set (12.1 vs 12.5 dB).

**Gate 3 at step 2,000: the model draws both building states, but the date does not choose between them** (2026-11-20; `gate3_keep`, renders of the two gate-2 view sets, whose targets are held out; correction to the entry above)
- **What the renders show now:** for the after-fire cameras the spire is gone and a dark scaffold-like mass stands in its place, at the true date and at a 2013 date alike, for all 4 noise seeds. For the pre-fire cameras one seed draws the spire and another replaces it by the same dark mass, again at both dates. Pictures: `gate3_keep/step2000_set0_summary.jpg`, `step2000_set1_summary.jpg`, `step2000_seed1_check.jpg`.
- **Reading:** the adapter has learnt that this scene has two states. Which one it draws follows the cameras and the noise seed. The date effect seen at step 1,000 (spire faded at the true date, intact at 2013) did not grow into control; it was overtaken.
- **Date guidance at render time does not rescue it:** dropping the date in the unconditional half of guidance (scale 2) gives almost the same renders at both dates (`data/experiments/mvgen/render/gate3_keep_step2000_uncond_unknown`).
- **Cameras partly give the period away.** A nearest-neighbour vote on camera position and viewing direction predicts before/after the fire for 76% of the 1,038 modern and after-fire views (always guessing "before" gives 61%). After-fire photos were taken from where the public could still stand.
- **Why the date loses:** it is a perfect cue for the state but a weak one in the loss. The state shows in a few percent of the frame, in about half of the samples (those whose references are from the other period), and a wrong date costs only 0.24% of held-out loss (0.37% on training sets), rising slowly.
- **Scores:** held-out loss −4.0% and training loss −3.9% at step 2,000 (no memorising). PSNR on the pre-fire set fell from 13.2 to 12.4 dB because of the wrongly removed spires.
- **Next:** score the changed region directly (in progress), then strengthen the date's signal there: a higher loss weight inside the changed region, or the planned term that makes the true date beat a wrong date. The run continues to 4,000 steps.
- **Check with 4 noise seeds per date (step 2,000):** the true date does not score higher than the wrong one on either view set, with or without date guidance (after-fire set 12.6 dB true vs 12.8 dB wrong; pre-fire set 11.7 vs 12.0 dB).
- **PSNR against one real photo falls as the model learns the scene's looks.** On the pre-fire set the released model scores 13.2 dB and the step-2,000 adapter 11.7 dB, although its held-out loss is 4% lower. The released model copies the references' light; the adapter has learnt that a target's light differs from the references' and draws one at random. Without an appearance input, PSNR punishes that. This is a second reason to model appearance: evaluation needs it too.

**Changed-region masks: where the scene differs before and after the fire** (2026-11-20; `timemachine.mvgen.regions`, overlays in `data/mvgen_regions/sheets/`)
- Built from which dated views observe which COLMAP points; nothing is hand-drawn. A point counts as changed when one period observes it on 3+ days and the other period never does although 3+ of its views should, and nearly all its neighbours are flagged too.
- **3,822 changed points form one group:** the spire, the three roofs and the scaffold at the crossing (2,610 gone, 1,212 new).
- Counting observers alone is useless: 23–31% of testable points have no observer on the other side by chance, because after-fire photos cluster on a few days.
- **Mask size:** median 0.14% of the frame in frontal views; 5–13% in south-west, south, south-east and apse views. The gate views: 6–10% for the pre-fire targets, 8–23% for the after-fire targets.
- Not handled: trees in front of the building are masked with it; the nave roof is only partly covered in close south-west views; views with a bad COLMAP pose get a misplaced mask (e.g. row 119, a night view).

**Gate 3 at step 4,000 (priors kept): the date begins to flip the state, in some samples** (2026-11-20; `gate3_keep`, picture `gate3_keep/step4000_check.jpg`)

| Step | Held-out loss | Wrong date costs (held-out / training sets) | PSNR after-fire set, true vs wrong date | PSNR pre-fire set, true vs wrong date |
|---|---|---|---|---|
| 0 | 0.3721 | 0% / 0% | 12.5 / 12.5 dB | 13.2 / 13.2 dB |
| 1,000 | −3.0% | +0.15% / +0.19% | 12.6 / 12.5 | 12.9 / 13.1 |
| 2,000 | −4.0% | +0.24% / +0.37% | 13.0 / 13.1 | 12.4 / 12.6 |
| 4,000 | −5.0% | +0.49% / +0.67% | 13.3 / 13.2 | 11.9 / 12.1 |

- **Pre-fire cameras, one of two noise seeds:** the true date draws the spire, the 2019 date replaces it by the scaffold. That is the date choosing the state. With the other seed the spire is missing at both dates.
- **After-fire cameras:** no spire at either date, with both seeds. The cameras decide.
- **The date's weight grows steadily** (the cost of a wrong date doubled between steps 2,000 and 4,000) but is still small. The run is being continued to 8,000 steps.
- **Gate 3 is not passed:** on held-out views the true date does not yet beat the wrong one in PSNR on both sets, and the pre-fire set is 1.3 dB below the released model.
- The second arm (`gate3_dropall`, both priors dropped for cross-period targets) shows no stronger date effect at step 3,000 (+0.2% in loss) and a lower loss gain (−3.6% held-out).

**Scores inside and outside the changed region, gate 3 at step 4,000** (2026-11-20; `scripts/mvgen/eval_dates.py`, 10 held-out view sets: 9 with after-fire targets from pre-fire references, 1 with pre-fire targets; `data/experiments/mvgen/eval_dates/gate3_keep_4000`)

| Targets | Area | Wrong date costs (loss) | PSNR, true − wrong date |
|---|---|---|---|
| After the fire (9 sets) | inside the region | +0.66% ± 0.21% | +0.17 ± 0.13 dB |
| | outside | +0.39% ± 0.06% | +0.08 ± 0.03 dB |
| Before the fire (1 set) | inside the region | +2.84% | +0.35 dB |
| | outside | +0.02% | −0.31 dB |

- **The date's effect on the building is real but small,** and it sits where it should: inside the region, and at high noise levels, where structure is decided (+1.1–1.3% at timesteps 700 and 900 against +0.3% at 100–500).
- **It is four times stronger for pre-fire targets** than for after-fire targets. For after-fire cameras the model draws the burnt state almost whatever the date says.
- **Outside the region the date changes the look:** for the pre-fire set a 2019 date gives a higher PSNR outside the region than the true date. The date also carries light and season.
- The standard errors are too small: the 9 after-fire sets share 18 distinct targets.
- The evaluation and the masks were checked independently (alignment within 0.002 px of the sample's cameras; 97% of the changed points a view observes fall inside its mask, 2.4% of the unchanged ones). Known limits: frontal views have no usable region; only one pre-fire side-view set exists in the hold-out.

**Inside the changed region, over training: after-fire targets gain what pre-fire targets lose** (2026-11-20; same 10 held-out view sets, `data/experiments/mvgen/eval_dates/gate3_keep_{released,1000,2000,4000}`; loss with fixed noise at the true date, PSNR of renders at the true date)

| Checkpoint | After-fire targets, inside: loss / PSNR | Pre-fire targets, inside: loss / PSNR | Wrong date costs, inside (after / pre) | Renders differ between dates, inside / outside |
|---|---|---|---|---|
| Released model | 0.463 / 10.2 dB | 0.308 / 13.2 dB | 0% / 0% | 0.012 / 0.011 |
| Step 1,000 | 0.413 / 10.2 | 0.337 / 12.7 | +0.22% / +0.45% | 0.033 / 0.029 |
| Step 2,000 | 0.391 / 11.3 | 0.360 / 11.5 | +0.43% / +0.90% | 0.049 / 0.040 |
| Step 4,000 | 0.379 / 12.1 | 0.371 / 10.5 | +0.66% / +2.84% | 0.059 / 0.041 |

- **The adapter learnt the burnt state:** inside the region, after-fire targets improve by 18% in loss and 2.0 dB (outside the region: 4% and 0.3 dB).
- **But it applies it to pre-fire targets too:** inside the region they get worse by 21% in loss and 2.7 dB. One state is drawn for this side of the building at any date; the date corrects only a small part of it.
- **This is the precise sense in which gate 3 fails:** the building's state is learnt but not tied to the date. A date-conditional model would improve the after-fire region without hurting the pre-fire one.
- The date's effect grows at every checkpoint, inside the region faster than outside, and the renders at two dates differ more inside the region than outside (the released model's floor is 0.012).
- The second arm (`gate3_dropall`) ended at step 4,000 with a held-out loss gain of −4.3% (priors kept: −5.0%) and a wrong-date cost of +0.33% (priors kept: +0.49%). Dropping the priors for cross-period targets did not help the date.
- The run with priors kept continues beyond 4,000 steps (at step 4,750: wrong-date cost +0.62% on held-out sets).

**Gate 3 continued to 8,000 steps: the date's weight keeps growing, the pictures barely change** (2026-11-20, read on 2026-11-24; `gate3_keep`, picture `gate3_keep/step8000_check.jpg`)

| Step | Held-out loss | Wrong date costs (held-out) | PSNR after-fire set, true / wrong date | PSNR pre-fire set, true / wrong date |
|---|---|---|---|---|
| 4,000 | −5.0% | +0.49% | 13.3 / 13.2 dB | 11.9 / 12.1 dB |
| 6,000 | −5.8% | +0.94% | 13.5 / 13.2 | 11.7 / 11.8 |
| 8,000 | −6.2% | +1.18% | 13.5 / 13.3 | 11.3 / 11.4 |

- The cost of a wrong date more than doubled again between steps 4,000 and 8,000, and the true date now wins by 0.26 dB on the after-fire set.
- The renders look as at step 4,000: after-fire cameras show the burnt state at either date; pre-fire cameras show the date choosing the state with one noise seed and no spire at either date with the other.
- Plain training is moving in the right direction, slowly.

**Three new training options, to make the date decide the state** (2026-11-20/24; `tests/test_mvgen_signal.py`; all off by default, old runs resume unchanged: checked bit for bit on 10,000 real training draws)
- **Loss weight inside the changed region** (`--region-weight`, needs `changed_regions.npz` from `scripts/mvgen/prepare_regions.py`): 618 of the 1,115 usable views show the region. A weight of 10 raises the region's share of the loss from 7.9% to 32.7% (after-fire targets 51%, modern 20%), about 4 times, because each sample is divided by its own sum of weights.
- **Camera-balanced targets** (`--balance-cameras`, the largest weight an anchor may get): anchors whose neighbours in camera space are of mixed periods are drawn more often. It removes only part of the camera's hint: a vote of the 15 nearest cameras predicts the period of drawn targets 74% of the time without it and 63% with a cap of 5 (guessing: 48%), because only the anchor is weighted, not the other four targets.
- **Start from another run** (`--init-from`): the new objective starts from a trained adapter's raw weights at step 0.
- The trainer's probe now also prints the cost of a wrong date inside and outside the changed region (when the dataset has the regions file).
- Known limits: 46 frontal views keep a "region" that is specks on the facade (1.5% of the weight inside regions); the probe's region line sees almost only after-fire targets, so pre-fire targets are scored with `eval_dates.py` and `configs/mvgen_gate3_prefire_heldout.json` (4 more held-out pre-fire view sets).
- **Runs started 2026-11-24**, all from `gate3_keep` at step 8,000, 2,000 steps each: `gate3_region10` (weight 10), `gate3_region10_bal5` (weight 10, balance 5), and `gate3_keep` continued to 10,000 as the control.

**Correction: at step 8,000 the date does choose the building's state on most held-out view sets** (2026-11-24; `eval_dates.py` on 14 held-out view sets: 9 with after-fire targets from pre-fire references, 5 with pre-fire targets; `data/experiments/mvgen/eval_dates2/gate3_keep_8000`, sheets in its `sheets/` folder)

| Held-out targets | Area | Wrong date costs (loss) | PSNR, true − wrong date | Renders differ between the two dates |
|---|---|---|---|---|
| Before the fire (5 sets) | changed region | **+11.7% ± 1.4%** | **+1.49 ± 0.44 dB** | 0.148 |
| | rest of the frame | +0.6% ± 0.1% | −0.07 ± 0.08 dB | 0.066 |
| After the fire (9 sets) | changed region | +1.9% ± 0.6% | +0.47 ± 0.19 dB | 0.084 |
| | rest of the frame | +0.9% ± 0.2% | +0.11 ± 0.07 dB | 0.057 |

- **The earlier reading was too pessimistic.** It rested on the two gate-2 view sets and two noise seeds. Over 14 held-out sets the wrong date costs more inside the region in all 14, and the true date gives the higher PSNR there in 12.
- **Pictures:** pre-fire cameras rendered at their true dates show spire and roof; at 2019.4 the spire and roof are gone and the scaffold and the white cover over the crossing appear (`sheets/prefire_0.jpg`, `prefire_2.jpg`). After-fire cameras rendered at 2013.4 get the spire and roof back in one set (`sheets/heldout_0.jpg`); in others they stay burnt.
- **The effect is concentrated where it should be:** inside the changed region and at high noise (after-fire sets, inside: +4.9% at timestep 700 against +0.4% at 100).
- **It is still uneven:** pre-fire sets +8% to +15% inside the region; after-fire sets 0% to +5%. The two gate-2 sets are among the weakest.
- **Between steps 4,000 and 8,000 the effect inside the region roughly tripled** on the after-fire sets (+0.66% → +1.9%). For pre-fire targets the step-4,000 figure (+2.8%) is from one set; the same 5 sets are being evaluated at step 4,000.
- **Open half of the gate:** quality at the true date against the released model on these 14 sets (evaluation queued). On the earlier 10 sets the released model scored 13.2 dB on the one pre-fire set, and this checkpoint scores 11.9 dB on the five.
- **First sign from the new runs:** 250 steps with region weight 10 and camera balance raise the cost of a wrong date inside the region from +1.8% to +3.3% on the trainer's held-out probe sets; the control is at +1.8% after 500 more plain steps.

**The three new options passed their second independent check** (2026-11-24; sampling, start-from-another-run, resume of old runs; 453 tests pass)
- No code defect found. The option-off path draws the committed code's samples bit for bit (checked on the samples the running control draws), and all running runs resume from their own checkpoints.
- **The control is not like-for-like:** continuing `gate3_keep` keeps its optimiser state and its long weight average and draws another seed, while the new runs start fresh from the same weights. A cleaner control, `gate3_init_only` (the region-weight run without the weight), is queued.
- **Camera balance removes only part of the shortcut:** among the targets the balanced run draws, the camera still separates before from after the fire with AUC 0.71 (0.83 without). After-fire cameras mostly stand where no pre-fire camera stood; no weighting can fix that.
- Early probes of the new runs are noisy because their weight average is short; they are read at step 2,000 with `eval_dates.py`.

**Region weight and camera balance both strengthen the date; with both, the true date wins in 13 of 14 held-out view sets** (2026-11-24; `eval_dates.py` on the same 14 held-out view sets; `data/experiments/mvgen/eval_dates2/`; the two new runs train 2,000 steps from `gate3_keep` at step 8,000)

Inside the changed region:

| Model | Pre-fire targets (5 sets): wrong date costs / PSNR at true date / true − wrong | After-fire targets (9 sets): wrong date costs / PSNR at true date / true − wrong |
|---|---|---|
| Released model | 0% / 12.2 dB / 0 | 0% / 10.2 dB / 0 |
| Plain, step 4,000 | +3.7% / 10.9 / +0.8 dB | +0.7% / 12.1 / +0.2 dB |
| Plain, step 8,000 | +11.7% / 11.7 / +1.5 | +1.9% / 12.7 / +0.5 |
| Plain, step 10,000 (control) | +15.1% / 12.0 / +1.8 | +2.5% / 12.8 / +0.6 |
| 8,000 + 2,000 with region weight 10 | +19.5% / 12.4 / +1.9 | +3.5% / 13.2 / +0.8 |
| 8,000 + 2,000 with weight 10 and camera balance | **+25.0% / 12.5 / +2.1** | **+7.8% / 12.9 / +1.4** |

- **Both options help, and they add up.** The region weight helps both kinds of target; the camera balance triples the date's effect on after-fire targets (+2.5% → +7.8%), which is where the camera shortcut was strongest.
- **The damage to pre-fire targets is repaired.** Inside the region they are back above the released model (12.5 against 12.2 dB) while after-fire targets stay 2.8 dB above it. At step 4,000 the pre-fire targets were 1.3 dB below.
- **With both options the true date gives the higher PSNR inside the region in 13 of 14 sets** and the lower loss in all 14. Pre-fire cameras rendered at 2019.4 lose spire and roof in every view of the gate-2 set that failed before (`eval_dates2/gate3_region10_bal5_2000/sheets/mvgen_gate2_1.jpg`).
- **Weak case left:** after-fire cameras asked for a pre-fire date. In the gate-2 after-fire set the 2013 date brings back roof and sky but not the spire (`sheets/mvgen_gate2_0.jpg`); in other sets it brings back both. Many after-fire cameras stand where no pre-fire camera stood, so there the model has never seen the intact state.
- **Whole-frame PSNR at the true date:** after-fire targets 12.6–12.8 dB (released 11.7); pre-fire targets 11.8–11.9 dB (released 12.2). The 0.3–0.4 dB shortfall on pre-fire targets is outside the region: the adapter draws its own light. The loss at the true date is below the released model's everywhere.
- **Caveat:** the plain control kept its optimiser state, weight average and seed; a like-for-like control (`gate3_init_only`) is running.

**Matched control: the camera balance is what strengthens the date; the region weight adds little** (2026-11-24; `gate3_init_only` = the region-weight run without the weight: same start from step 8,000, same fresh optimiser, same seed, 2,000 steps; same 14 held-out view sets, compared set by set; correction to the entry above)

| Comparison (set by set, 14 sets) | Wrong-date cost inside the region | PSNR, true − wrong date, inside | PSNR at the true date, whole frame |
|---|---|---|---|
| Region weight 10 vs matched control | +2.5 ± 0.7 points (12 of 14 up) | −0.03 ± 0.07 dB | −0.07 ± 0.04 dB |
| Adding camera balance to the weight | **+4.8 ± 1.1 points (14 of 14 up)** | **+0.48 ± 0.22 dB** | −0.20 ± 0.06 dB |
| Weight + balance vs matched control | +7.3 ± 1.4 points (14 of 14 up) | +0.44 ± 0.19 dB | −0.28 ± 0.07 dB |
| Matched control vs plain continuation to 10,000 | −0.3 ± 0.3 points | +0.17 ± 0.11 dB | +0.39 ± 0.09 dB |

- **Matched control, absolute:** a wrong date costs +13.9% (pre-fire targets) and +2.7% (after-fire) inside the region; the true date gives the higher PSNR inside the region in **14 of 14** sets. Gate 3's date criterion is met by plain training alone at 10,000 steps.
- **The region weight alone** raises the loss cost of a wrong date a little, mostly for pre-fire targets (+5.6 points), and does not change how much better the true-date render is. The earlier table overstated it, because the plain control was not like-for-like.
- **The camera balance** is the ingredient that helps: every one of the 14 sets moves up, most for after-fire targets (+4.4 points, +0.6 dB), where the camera's hint about the period is strongest.
- **Whole-frame PSNR at the true date is noisy between checkpoints.** Two runs of the same recipe differ by 0.4 dB on average and 0.8 dB on pre-fire targets (matched control 12.1 dB, plain continuation 11.3 dB, step 8,000 11.9 dB; released model 12.2 dB). So the 0.2–0.3 dB cost seen with the balance, and the earlier "0.3–0.4 dB below the released model", are inside that noise. With the matched control the pre-fire shortfall against the released model is 0.1 dB.
- Not tested: camera balance without the region weight.

## Appearance study (2026-11-24)
*Decision by the project owner: sepia and black-and-white count as appearance, not time. Study: two literature reviews, three measurements on Notre-Dame and our code, three competing designs, two judges. Notes, scripts and tables: `data/appearance_study/` (not in git). Nothing here ran on a GPU.*

**Which appearance signal carries the look without giving the period away** (1,115 usable views, folds grouped by capture day; "leak" = how well the signal tells after-fire from modern photos, 0.5 = not at all; camera pose alone: 0.82)

| Signal computed from one photo | Leak (AUC) | Reads the building itself? | Error it removes from a render if given |
|---|---|---|---|
| Frozen DINOv2 or CLIP embedding (also cut to 4 numbers) | 0.96–0.98 | yes (0.996 where the region shows) | not measured |
| 8×8 colour thumbnail | 0.77 | it is the target's layout | 3 dB (1 dB less when borrowed from a same-day photo) |
| Mean and spread of colour per class (sky, building, vegetation, ground, water), outside the changed region: 30 numbers | 0.70 | no | 2.0 dB |
| Six global colour numbers | 0.67 | no | 1.0 dB |
| Learned 8-number encoder on colour cells without positions (CPU stand-in) | 0.67 | no | 1.5 dB |
| Season, time of day and pixel flags from metadata | 0.91 | no | under 2% of the look |

- **Colour statistics are the best trade.** They leak less than the camera does, do not read the building (same leak whether the changed region is visible or not), and a learned encoder restricted to the same input carries less at the same leak.
- **Pretrained image encoders are ruled out as appearance encoders**: they read the fire almost perfectly, even from 4 numbers.
- **Metadata labels (season, hour) give the period away far more than pixels do** and say little about the look. After-fire photos are 63% spring (modern 27%); ten days hold 58% of them; five photographers hold 58%.
- **A global code carries a minority of the look**: 1–2 dB. The earlier "six numbers remove 56% of the error" was an oracle fit that also lowered contrast; the honest figure is about 20%. Most of the difference between a render and the real photo is local (clouds, shadows, leaves), which is the noise seed's job.
- **The date has learnt the period's look.** Outside the changed region a date swap moves the render in the direction of the real difference between after-fire and modern photos (0.48 of a 0.58 standard-deviation gap, growing since step 1,000). Only 20–23% of that is global colour.
- **The look belongs to the day.** A capture day shares 42–57% of a photo's global colour statistics; smooth functions of the date predict none of it on unseen days. A date code that can tell days apart can memorise the look. The trained date embedding has no step at the fire and moves more across 2018 than across the fire.
- **Medium:** colour statistics tell monochrome from colour at AUC 0.98, so a colour code carries it. 71 of the 77 pre-1970 photos are monochrome, so "the past in colour" is extrapolation on this scene.
- **Published work.** WildCAT3D, the one multi-view diffusion model with a learned appearance code, uses 8 numbers from a small CNN, relies on size alone against leaks, reports no leak test, and never uses a target's own code at inference. Neural Scene Chronology lets time explain each photo first and applies the light code last; its Fourier time code overfitted per-image light. Encoders that see the scored photo inflate scores (GS-W: 2.5–4.5 dB). No paper treats sepia or black-and-white as its own factor.
- **A better instrument for the building's state:** a read-out on a frozen encoder applied to the grey, contrast-normalised crop of the changed region tells the two states apart at AUC 0.998 on real photos of unseen days. On `gate3_region10_bal5` a date swap flips it for 28 of 28 held-out targets.
- **Corrections to earlier entries:** the "13.2 → 11.9 dB" drop on pre-fire targets compared one view set with five; on the same five sets the released model leads by 0.2–0.4 dB, within noise. 88% of the held-out after-fire views share a capture day and photographer with a training view, so the next fresh run should hold out whole days.
- **Both judges ranked the designs the same way:** fixed colour statistics with recoloured targets first, the learned encoder second (its own numbers do not yet justify learning), no appearance input last (it needs a mask and a change date per scene).
- **Related work in the team** (branches `aadhitya/date-shortcut-probe`, `aadhitya/appearance-conditioning`, not merged): across four scenes 14 appearance statistics tell pre-1945 from modern photos at AUC 0.89 and toning a modern photo makes it "old"; removing 7 of the 14 dimensions makes the era unreadable while sunlit-versus-overcast survives; a per-block scale-and-shift pathway for SEVA is written but not wired into training.

## look1: an untrained per-view look code (launched 2026-11-25, 01:42)
*Spec: four arms on Notre-Dame, fresh from the released model, 10,000 steps each, identical except one factor. Code: commit f10f862. Plan of record and evaluation: `data/look1_build/`, `configs/look1_eval_sets.json`.*

| Arm | Look code | Data | Seed |
|---|---|---|---|
| A | no | as is | 0 |
| A2 | no | as is | 1 (seed replicate: the measured null) |
| B | yes | as is | 0 |
| D | yes | colour views turned grey (10%) or sepia (5%) at unchanged dates | 0 |

- **Look code:** 24 numbers per view (colour mean and spread, mean colour of sky, building, vegetation, ground, water, and grey/toned/colour), computed outside the changed region, for references and targets; added to each view's timestep embedding beside the date through a zero-initialised MLP. No loss or sampling tricks (no region weight, no camera balance).
- **Hold-out:** whole capture days and photographers (132 views in 62 units); every scored evaluation target is held out under the arms' settings. Evaluation: a 2×2 of input period × target period, with a colour-blind state read-out; pre-registered pass lines.
- **Independent reviews (training and evaluation sides): no defect in the running code.** Notes for reading the results:
  - B and D start from bit-identical weights and draw the same samples as A; GPU arithmetic differs between jobs by about 1e-4, so A will not match B bit for bit.
  - The standard errors are too small with so few hold-out units (after-fire targets sit in 8 units): a "2 SE" test rejects 8-13% of the time instead of 5%.
  - The grey-flag check in R1 cannot pass as written (no decoded render is as grey as a real grey photo); it is not part of the gate.
  - 12 of the 53 scored targets are by photographers who have training photos in the other period.
- **Early observation (step 250-350, B):** the look embedding grows faster than the date's (look rms 0.097 against date 0.071 at step 350; together 0.91 of the timestep embedding).

**look1 gate at step 2,000, arm B: the look code steers the render** (2026-11-25; raw weights; 25 held-out view sets; `data/experiments/mvgen/eval_look1/B_step2000_raw/`, picture `gate_sheet.jpg` there)
- **Steering passes clearly:** giving a target another photo's code moves the render's colour toward it with slopes 0.79 (red), 0.77 (green), 0.60 (blue); gate line 0.4. By eye: another photo's code changes sky, light and white balance; the grey code gives an almost grey render (some residual tint, so the strict "as grey as a real grey photo" line fails, as expected).
- **The date has not taken the look so far:** a date swap across the fire shifts the look along the real after-fire-vs-modern difference by −0.04 ± 0.02 sd (date-only runs: about +0.35 sd at step 2,000). Arm A's step-2,000 weights are copied for the matched comparison when A gets a GPU.
- **The date does not yet choose the building** (as in the date-only runs at step 2,000): after-fire input photos still give a burnt cathedral for pre-fire targets.
- The within-period swap null is +0.04 ± 0.01 sd (not zero but small).
- **Arm D at step 2,000 (grey/sepia conversion of colour photos at unchanged dates):** steering passes as in B (slopes 0.79 / 0.79 / 0.65), and **the grey code now gives a truly grey render** (chroma outside the region 0.62 against 5.28 in B; real grey photos about 0). In B grey photos exist only among old photos, so B learned "grey" only half; D's conversions teach it at every date. The date's look shift across the fire is −0.01 ± 0.01 sd.

**look1, date-loss probes: the look code takes the within-period appearance off the date** (2026-11-25 evening; `probes/heldout_losses.jsonl` of each arm; loss change when the targets' dates are changed, held-out view sets)
- **Date only (A, A2):** swapping a date *inside* its period hurts held-out views by +1.0 to +1.8% (A, steps 750-2,250) and +0.7% (A2). The date is then used for the day's look (light, colour), which is not the place's state. Across periods, A's held-out swap effect is around zero or negative at step 2,250 (−0.22%), A2's +0.20%.
- **With the look code (B, D):** the within-period effect stays at +0.1 to +0.5%, and the across-period effect grows steadily: B +0.32% (6k), +0.54% (8k), +0.66% (10k); D +0.26%, +0.44%, +0.64%. The look code itself moves the loss by +8 to +10% (other photo's code).
- **Decision for the multi-scene run:** date + look code with the grey/sepia conversion (arm D's recipe). B and D are on par; D also makes "grey" a clean control.

## Multi-scene run: data and code (2026-11-25)
*Plan: `docs/presentation_plan.md`. Code: `scripts/mvgen/build_scene.py`, `src/timemachine/mvgen/multiscene.py`, `scripts/mvgen/train_multi.py`, `slurm/mvgen_multi.sbatch`.*
- **129 scenes are built in our folder on `/fs/nexus-projects/wildnvs/diffusion-time-machine/`** (about 220 MB per scene; images separately, 6 GB so far). 116 builds first failed because their cached pose tables predated the lens-distortion columns; re-fetched (cached COLMAP files, seconds per scene) and rebuilt.
- **Depth and look codes** take about 10 minutes per scene on one A6000; tasks now sit in one shared pool (`scripts/mvgen/pool_worker.py`) that any of our GPU jobs drains.
- **Depth fix:** Depth-Pro's fitted map can be zero or negative on panoramas and drawings (first seen: a 1909 panorama of the Opéra). Such a view now gets no depth map and is marked not reliable, instead of stopping the scene.
- **Data-parallel trainer tested** (two ranks on one GPU through gloo: the weights stay identical across ranks, losses and probes are normal, checkpoints save and resume, also with a different number of GPUs). A multi-scene checkpoint renders any scene with `scripts/mvgen/render.py`; `scripts/mvgen/scene_sets.py` draws view sets with targets of one period and references of another.
- **Old photographs rarely anchor a sample** outside Notre-Dame: 1-20% of a scene's anchors are before 1970 (Notre-Dame 3%). Old targets are therefore rare. `--old-anchor-share s` (off by default) draws old anchors with probability s in scenes that have at least 10 of them; whether to use it in the main run is the owner's call (it is a sampling choice).

**ms1, the main multi-scene run, is training** (job 8577831 from 2026-11-25 23:28; 4×A6000; `data/experiments/mvgen/multi/ms1/`)
- 118 training scenes (Notre-Dame at 15% of samples, the others 0.4–1% each by size), 11 held-out scenes; 29,205 training views. Arm D's recipe (date + look code, grey/sepia conversions).
- Speed: 1.05 s per step of 4 samples (one per GPU), about 330,000 samples a day.
- Dataset in numbers (report: `scripts/mvgen/dataset_stats.py`): 35,758 usable photos, 2,639 (7%) before 1970, oldest 1845; the 2010s alone are 61%. 29 training scenes have no old photo, 34 have 20 or more. 1,743 of the 3,130 black-and-white training photos are from after 1970.
- Only 572 of 16,183 anchors are old: about 3% of samples are built around a pre-1970 photo. From the second job (8581426, chained after 36 h) `--old-anchor-share 0.2` is on (owner's decision, 2026-11-26), which roughly doubles that to 6% (14 scenes have at least 10 old anchors).
- Date sweeps at steps 10,000 and 30,000 run on their own GPU (queue `data/queue/prep/d.txt`): Great Sphinx, Sydney Harbour Bridge, Opéra Garnier, Vilnius Cathedral, Yosemite, Forth Bridge (held out), Arc de Triomphe and the Royal Palace of Amsterdam (training scenes); targets of one period, references of the other, rendered at their true dates and at 1880–2015.
- **ms1 at step 10,000 (first sweeps, 2026-11-26 03:40; `data/experiments/mvgen/multi/ms1/sweeps/step010000/`):** geometry and views are right on held-out places (Opéra Garnier drawn from three modern photos at the camera of 1886–1900 photos), but the year does not change the picture yet: 1880 to 2015 give nearly the same render. Notre-Dame has had about 6,000 samples by then (look1 arms: 20,000 at their step 10,000), and the Notre-Dame date probe is small (wrong date +0.05%, unknown +0.4% at step 10,000).
- **The renders take the references' look** (the inference default): grey 1950s references give grey 2008 renders of the Royal Palace of Amsterdam. That is the look code doing its job, but a sweep then mixes the look of the inputs with the date. The step-30,000 sweeps use `--look neutral` (the population mean), so that only the date changes between rows.
- **ms1 on Notre-Dame at step 10,000, same evaluation sets as look1 (`eval_look1/ms1_step10000/`), against look1 D at its step 10,000:**
  - **The look code works as well as in D:** steering slope 0.91 (D 0.87), the grey code gives a nearly grey render (chroma 0.51; D's step-2,000 gate 0.62), and the date takes no look (R3 cross-fire +0.01 sd; D +0.09).
  - **The date does not yet choose the fire state:** the state read-out flip is +0.08 ± 0.06 logits (D +1.81 ± 0.42), and the wrong date costs +0.14% more inside the region than outside (D +2.1%).
  - PSNR 11.6 dB at the references' look and 13.2 dB with the target's own code (D 12.2 and 14.0).
  - This is expected at this stage: Notre-Dame has had about 6,000 samples in ms1 against 20,000 in D at step 10,000; ms1 reaches D's count near step 33,000. The step-30,000 evaluation is queued.

**look1 final result at step 10,000: the look code passes both pre-registered lines** (2026-11-26; `eval_look1/*_step10000/`, read with `eval_dates.py --from-results D_step10000/results.json --compare A=... A2=... B=... released=...`)
- **The date no longer takes the photo's look:** a date swap across the fire moves the render's look along the real after-fire-vs-modern difference by +0.09 sd in D and B, against +0.63 (A) and +0.67 (A2) without a look code. Pass line (at most half of A, and at least 0.15 below it): PASS.
- **The date still chooses the building as well as without a look code:** the fire-state read-out flips by +1.81 logits in D (B +1.96), against A +2.13 and A2 +3.80 (the two date-only seeds differ a lot). Non-inferiority: PASS.
- The look code steers the render (slope 0.87), and D's grey code gives greyer renders than B's (chroma −0.14).
- Against the released model, D is +0.6 dB at the references' look and +1.3 dB with the target's own code (under the 0.8 dB line where noise dominates for the first).
- **ms1 sweeps at step 30,000 with the neutral look** (`sweeps/step030000_neutral/`): the building is still the same at every year; what the year changes is colour and sky (later years bluer and more colourful, earlier years greyer), also with the look code held fixed. A "neutral" code is not neutral in tone (it averages grey, toned and colour), so renders of one set can still differ in colour from target to target.
- **ms1 on Notre-Dame at step 30,000 (`eval_look1/ms1_step30000/`): the date starts to choose the fire state.** The state read-out flip is +0.70 ± 0.21 logits (step 10,000: +0.08), about where look1 D was at its step 6,000 (+0.63 ± 0.16); the wrong date costs +0.95% more inside the region than outside (step 10,000: +0.14%). The date still takes no look (R3 +0.02 sd), the look code steers (slope 0.89), and PSNR with the target's own code is 13.8 dB (D at step 10,000: 14.0). Notre-Dame has had about 18,000 samples by then.
