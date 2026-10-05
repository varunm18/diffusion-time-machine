# Literature Review

*Compiled 2026-10-04. Each entry covers what matters for our project, not a full summary. Details come from reading the papers and repos; anything marked (unverified) we haven't confirmed ourselves.*

---

## 1. Data

### MegaScenes — Tung, Chou et al., ECCV 2024 ([arXiv 2406.11819](https://arxiv.org/abs/2406.11819), [site](https://megascenes.github.io/), [dataset repo](https://github.com/MegaScenes/dataset))
- A **scene = a Wikimedia Commons category** tied to a Wikidata item from 24 classes (churches, monuments, museums, bridges, …). About 430K scenes and 9M images.
- Images were crawled **up to 4 subcategory levels deep**, keeping only subcategories whose names contain the scene name. That's why "Historical images of X" and "X in art" are included, along with old photos, paintings and postcards.
- COLMAP SfM (SIFT + vocab-tree matching) gives ~83K scenes with ≥1 reconstruction, 106K reconstructions, and 2.0M registered images.
  - Counting each scene's largest reconstruction, only **~4.9K scenes have ≥50 posed images** and ~2K have ≥100.
- Each subcategory folder ships a `raw_metadata.json` holding the **Commons metadata of each image** (date field, categories, description, EXIF, license, upload time). This is our main date source. The crawl ran June 2023 – June 2024.
- Their NVS model (a ZeroNVS/Zero-1-to-3-style fine-tune) trained on **image pairs taken within 3 hours of each other**. They deliberately avoided cross-time pairs, which is the opposite of what we need.
- Doppelgangers filtering was applied only to manually flagged scenes. A later paper (Long-Tail, 2026) found ~25% of large scenes have unreliable reconstructions (people, statues, look-alike facades).

### Other Commons-based datasets
- **WikiScenes** (Wu, Averbuch-Elor, Snavely, ICCV 2021, [2108.05863](https://arxiv.org/abs/2108.05863)): 63K images of 99 cathedrals with captions, categories and COLMAP models. Same scraper lineage as MegaScenes. **No dates.**
- **Doppelgangers / Doppelgangers++** (Cai et al., ICCV 2023 / CVPR 2025, [2412.05826](https://arxiv.org/abs/2412.05826)): classifiers that reject false matches between look-alike structures. Doppelgangers++ plugs into COLMAP and MASt3R-SfM. Useful when we add old images to reconstructions.
- **HaLo-NeRF** ([2404.16845](https://arxiv.org/abs/2404.16845)): semantic labels from WikiScenes. No dates.
- **Faces Through Time** (Chen et al., Eurographics 2023, [2210.06642](https://arxiv.org/abs/2210.06642)): Commons portraits 1880s–2010s, **dated from Commons categories + biographical metadata**. About 6% were removed by hand for wrong dates. The closest precedent for dating Commons images at scale.
- **YearGuessr** ([2512.21337](https://arxiv.org/abs/2512.21337)) and **TIME10k** ([2510.19559](https://arxiv.org/abs/2510.19559)): Commons-based, but they date **when the building/object was made**, not when the photo was taken.
- **No released per-image capture-date annotations exist for any of these.** A dated MegaScenes subset would be new.

---

## 2. Base model

### Stable Virtual Camera (SEVA) — Zhou et al., ICCV 2025 ([arXiv 2503.14489](https://arxiv.org/abs/2503.14489), [repo](https://github.com/Stability-AI/stable-virtual-camera))
- SD 2.1-base UNet (866M) + 398M of new multi-view attention = 1.26B params. 576×576 images, **21 views per forward pass** (any mix of inputs and targets).
- **Per-view inputs:**
  - the image latent, which is clean for inputs and noisy for targets;
  - an input/target mask;
  - a 6-channel **Plücker ray map** for the camera. It enters at the first layer *and* modulates every ResBlock through a per-pixel scale/shift.
- **Shared input:** a CLIP image embedding **averaged over all input views**. Several attention blocks only read the first view's CLIP token.
- Timestep embedding is per view and added as a bias in all 22 ResBlocks. This is the natural place to add a per-view date embedding.
- Training data **undisclosed** (likely clean video/multi-view captures, not internet photo collections). MegaScenes would be new data for it.
- **Training code not released** ("company policy"). Only inference code and weights are public.
  - Non-commercial license; gated HF weights (~5 GB).
  - The hard-coded SD2.1-base VAE download currently fails, so we need a mirror.
- Inference: up to ~32 input views; two-pass "procedural" sampling for long trajectories; ~18–22 GB VRAM at 576².

### Fine-tuning SEVA: precedents
- **VMem** (ICCV 2025, [2506.18903](https://arxiv.org/abs/2506.18903)): the only published SEVA LoRA recipe. Rank 256, 8 frames, RealEstate10K, 8× A40 (48 GB), lr 3e-6. Weights released, training code not.
- **InstructMix2Mix** ([2511.14899](https://arxiv.org/abs/2511.14899)): full fine-tunes SEVA per scene (~40 min on one H200). Found LoRA worse for its task. A fork adds gradient checkpointing.
- **Community PR #51** on the SEVA repo: unmerged training pipeline (DL3DV loader, latent precompute) with known bugs. A useful reference, not a dependency.

---

## 3. Multi-view diffusion on in-the-wild photos (appearance, no time)

- **WildCAT3D** (Alper, Novotny, Kokkinos, Averbuch-Elor, Monnier; NeurIPS 2025; [2506.13030](https://arxiv.org/abs/2506.13030)). **Closest relative to our model.**
  - CAT3D-style model (their own re-implementation, **not SEVA**) fine-tuned on MegaScenes + CO3D.
  - Learns an **8-number appearance code per view** (lighting, day/night, indoor) from the VAE latent. The code is concatenated as input channels and given for inputs *and* targets during training.
  - **Keeps the appearance codes in the unconditional branch of classifier-free guidance**; dropping them caused oversaturation.
  - Uses "full MegaScenes without aggressive filtering". 32× A100 for ~1 week. **No code/weights released.** No notion of time beyond seasons.
- **KFC-W** ([2411.13549](https://arxiv.org/abs/2411.13549)): multiview-inpainting video model trained on MegaScenes without 3D labels.
- **MVGenMaster** ([2411.16157](https://arxiv.org/abs/2411.16157)): uses a "domain switcher" token for MegaScenes to absorb its lighting inconsistency.
- **Feed-forward in-the-wild splatting:** GenWildSplat ([2604.28193](https://arxiv.org/abs/2604.28193)), WildSplatter ([2604.21182](https://arxiv.org/abs/2604.21182)), WildSplat ([2607.05347](https://arxiv.org/abs/2607.05347)). Appearance codes, no time.

---

## 4. Time

### The chronology line (all per-scene optimization, none generative)
- **4D Cities** — Schindler, Dellaert, Kang, CVPR 2007; Schindler & Dellaert, CVPR 2010.
  - Infers the temporal **order** of city photos (1897–2006) from which buildings are visible.
  - Key data point: of 490 Atlanta photos (1930s–2000s), **SIFT SfM registered only 102, and no reconstruction spanned the decades**.
  - Of 337 archive photos, **<11% had an exact date**: 47% "circa", 29% year only. Dates should be stored as intervals.
- **Scene Chronology** — Matzen & Snavely, ECCV 2014.
  - Estimates when each 3D planar patch existed, from Flickr photos with timestamps (Times Square, 5Pointz, ~2003–2014).
  - Can date a photo to within ~2.5–5.5 months from which patches it shows.
- **Time-lapse Mining / 3D Time-lapse** — Martin-Brualla, Gallup, Seitz, SIGGRAPH 2015 / ICCV 2015.
  - Builds time-lapses over years from 86M geotagged photos: fixed viewpoint, then small camera motion.
  - Wrong timestamps cause visible artifacts.
- **Neural Scene Chronology (NSC)** — Lin et al., CVPR 2023 ([2306.07970](https://arxiv.org/abs/2306.07970)).
  - NeRF with a per-image lighting code plus a time input encoded as **learned step functions**. Plain time inputs either blur changes or overfit lighting and flicker.
  - **Its 4 scenes are Flickr photos from 2009–2013** (Times Square, Akihabara, 5Pointz, The Met), not historical. Geometry is static, and black-and-white photos are excluded from test sets.
  - Code exists but is hard to run, and only the 5Pointz data is public. **As a baseline it's only realistic on 5Pointz, which tests years, not decades.**

### Generative models with time
- **4DiM** — Watson et al., ICLR 2025 ([2407.07860](https://arxiv.org/abs/2407.07860)).
  - Pose + time conditioned diffusion. Time is a **relative timestamp** (seconds scale, from video and consecutive Street View panoramas). It uses the same sinusoidal encoding as the noise level and goes through "masked FiLM" layers that pass through when time is unknown.
  - Has a separate guidance weight for time. **Nothing about seasons or years.**
- **DiffusionSat** — Khanna et al., ICLR 2024 ([2312.03606](https://arxiv.org/abs/2312.03606)).
  - SD 2.1 conditioned on numeric **year, month, day** (sinusoidal + MLP), plus a 3D ControlNet that generates a satellite image at a past or future date.
  - **Absolute-date conditioning of diffusion already exists**, but only for single overhead images.
- **Diffusion Models as Data Mining Tools** — Siglidis et al., ECCV 2024 ([2408.02752](https://arxiv.org/abs/2408.02752)): SD fine-tuned with the decade in the prompt (cars, faces). Single images.
- **CrossTimeEdit** — Lu et al., [2609.36616](https://arxiv.org/abs/2609.36616), posted 29 Sep 2026.
  - FLUX-based editor turning a recent street panorama into its look ~10 years earlier, guided by satellite-image differences.
  - Single panorama, no explicit date input. Very recent, so watch this space.
- **CityRAG** ([2604.19741](https://arxiv.org/abs/2604.19741)): video model grounded in Street View captured on different dates, but treats date differences as nuisance.
- **Short-timescale 4D:** CAT4D ([2411.18613](https://arxiv.org/abs/2411.18613)), SpaceTimePilot ([2512.25075](https://arxiv.org/abs/2512.25075)), TimeNeRF ([2507.13929](https://arxiv.org/abs/2507.13929); time of day).
- **Per-scene time-aware 3DGS:** LTGS ([2510.09881](https://arxiv.org/abs/2510.09881)), ChronoGS ([2511.18794](https://arxiv.org/abs/2511.18794)), Cross-Temporal 3DGS ([2512.00534](https://arxiv.org/abs/2512.00534)), Gaussian Time Machine ([2405.13694](https://arxiv.org/abs/2405.13694)), SeasonScapes ([2605.09039](https://arxiv.org/abs/2605.09039)).
- **Synthetic History** (ICLR 2026, [2505.17064](https://arxiv.org/abs/2505.17064)): text-to-image models stereotype past eras and produce anachronisms. Good motivation and a warning for evaluation.

---

## 5. Historical photos specifically

### Getting camera poses for old photos
- **HistImage2021** (Maiwald et al., IJGI 2021): learned matchers localized ~160/190 historical photos, but on a symmetric building most "registered" photos were flipped (65–177° error). **Registration counts overstate success; measure angular error.**
- **Photogrammetry now and then** (Morelli et al., 2022): on then/now pairs, ORB and RootSIFT got **0 correct matches**. LoFTR and SuperGlue were best but still weak.
- **Deep-Image-Matching** (Morelli et al., 2024): Semperoper Dresden photos 1880–2020. SuperPoint+LightGlue registered 161/165 vs RootSIFT 147. **Within one era SIFT is fine; bridging eras is the hard part.**
- **HistReNeRF** (Hughes & James, [2608.15420](https://arxiv.org/abs/2608.15420), Aug 2026).
  - Benchmark *HistScene*: Arc de Triomphe, Brandenburg Gate and Piazza del Duomo, with 230 archival photos from Wikimedia and Europeana. **Only 25% registered with SIFT.**
  - Feature-space adaptation helps; pixel-space style transfer (CycleGAN) hurts.
- **Coloring the Past** (GCPR 2024, [2311.17810](https://arxiv.org/abs/2311.17810)): 229 photos of one theater (1875–1965, >90% grayscale), posed with learned matchers. A per-scene NeuS.
- **Computational Re-Photography** (Bae et al., TOG 2010): old view cameras often have **off-center principal points**. Worth allowing in the camera model.
- **Gap:** no published evaluation of DUSt3R/MASt3R/VGGT/RoMa on ground-level registration across decades. A small benchmark would itself be a contribution.

### Estimating when a photo was taken
- **Palermo, Hays, Efros** (ECCV 2012): decade classification of color photos (1930s–70s) works mostly from **film/color characteristics**, not scene content.
- **Date Estimation in the Wild (DEW)** (Müller et al., ECIR 2017): 1M Flickr photos 1930–1999; CNN mean error 7.3 years (humans 10.9). Dataset and model available. **Only covers 1930–1999.**
- **A Century of Portraits** (Ginosar et al., 2015) and **Salem et al.** (WACV 2016): yearbook dating.
  - Background-only patches still date photos well in color but much worse in grayscale.
  - This directly measures the **"photo medium" shortcut**.
- **Blind Dates** (Barancová et al., 2023, [2310.06633](https://arxiv.org/abs/2310.06633)): zero-shot CLIP dates grayscale photos too early, and colorizing flips the bias. A linear probe on CLIP features gets a mean error of 6.65 years.
- **Seeing Time** ([2606.05702](https://arxiv.org/abs/2606.05702)): VLMs lean on grayscale-vs-color cues.
- **Gap:** no off-the-shelf dater covers outdoor scenes from the 1850s–2020s. We'd train our own for evaluation.

### Separating "how the photo looks" from "what the scene looked like"
- **Time-Travel Rephotography** (Luo et al., SIGGRAPH Asia 2021, [2012.12261](https://arxiv.org/abs/2012.12261)): simulates antique film by era.
  - Blue-sensitive film, then orthochromatic (from 1873), then panchromatic (from 1906), plus blur and tone curves.
  - **A ready recipe for "aging" modern photos as augmentation.**
- **Bringing Old Photos Back to Life** (Wan et al., CVPR 2020, [2004.09484](https://arxiv.org/abs/2004.09484)): a synthetic degradation recipe (grain, scratches, blur, JPEG, paper texture).
- **Colorization/restoration models** (DDColor, DeOldify, DiffBIR) invent content. Risky as preprocessing when date is the signal we're learning.
- **Gap:** no generative work explicitly separates photographic medium from real scene change across decades.

---

## 6. Is the project novel?

**Yes, with narrower wording.**

| Closest work | What it has | What it lacks |
|---|---|---|
| DiffusionSat | absolute dates | multi-view; satellite only |
| 4DiM | multi-view + time | calendar time (only relative seconds) |
| WildCAT3D | multi-view, in the wild, MegaScenes | any time variable |
| CrossTimeEdit / CityRAG | real places across years | date input; multi-view 3D |
| Siglidis / Faces Through Time | decade conditioning | places, 3D |
| NSC and the chronology line | dated photo collections | cross-scene, generative; per-scene only |

**Suggested rewording for the proposal:**

> "Absolute-date conditioning has been explored for satellite imagery [DiffusionSat] and for single images binned by decade [Siglidis et al.; Faces Through Time], and time-conditioned multi-view diffusion exists only for short relative timestamps [4DiM, CAT4D]. To our knowledge, we present the first feed-forward, cross-scene, 3D-consistent generative model conditioned on per-image calendar dates, spanning the 1850s–2020s at ground level."

**Factual fixes the proposal needs:**
- 4DiM's time axis is seconds-scale relative time, not "seconds to seasons".
- NSC's four landmarks are 2009–2013 Flickr scenes, not historical.

---

## 7. Takeaways for our method

1. **Dates as intervals.** Store [earliest, latest] + precision + source. Encode years sinusoidally (DiffusionSat). Learn an explicit **"unknown date"** embedding with pass-through (4DiM), and give date its own guidance weight.
2. **Where to put the date in SEVA:** add it to the per-view timestep embedding, and/or to the per-view Plücker modulation path, zero-initialized. Give dates for both inputs and targets (as WildCAT3D does for appearance). Don't use CLIP cross-attention: it's averaged and partly first-view-only.
3. **Fight the "grayscale = old" shortcut from day one:**
   - a separate photo-type/medium input;
   - age modern photos synthetically (Luo's film model + Wan's degradations) while keeping their *modern* date;
   - grayscale/tone-normalized evaluation.
4. **Poses for old photos are core work.** Match within historical subcategories with learned matchers, then bridge to the modern model with MASt3R-SfM/RoMa + Doppelgangers++. Validate with angular error, not registration counts. Expect roughly 25–85% of old photos to register.
5. **Evaluation:**
   - PSNR/LPIPS vs held-out archival photos will be dominated by film look, so also report appearance-normalized metrics.
   - Train a dater on held-out scenes and check it with grayscale controls.
   - Add an **anachronism check** from Wikidata construction/demolition dates (e.g. no rebuilt Frauenkirche in a 1960 view).
6. **Baselines:** SEVA without dates; nearest-dated real image retrieval; NSC only on 5Pointz. WildCAT3D has no code, so it can't be a baseline.
