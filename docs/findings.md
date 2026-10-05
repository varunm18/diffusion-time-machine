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
