# First multi-scene experiment (100+ scenes): plan and data, 2026-11-24

For the project owner and the team. This plan merges five studies made today and one critique of the
draft. Their notes and tables are under `data/multiscene_plan/` (not in git): `inventory/`, `scene_pool/`,
`beyond/`, `pipeline/`, `design/`.

> **Update, 2026-11-24 22:15.** The deletion this plan asks about was approved by the owner: the SEVA-era
> checkpoints and the Photo Tourism downloads were removed on their go-ahead, keeping logs, configs and
> small tables. 65 GB free (of the 400 GB quota) is therefore confirmed; the 40 GB sizing below is
> conservative. Decision 1 is answered. Nothing here ran on a GPU or through Slurm, and
nothing in the repository's source, tests, configs or docs was changed.

How to read the numbers:
- **[measured]** means counted from our tables or logs, or from a public API, by a script in the named folder.
- **[rechecked]** means I checked it again while revising this plan (2026-11-24, 21:30-22:10).
- **[inferred]** means arithmetic or judgement. Every count of views, hours or GB for the future is inferred.

All 21 points of the critique were checked. Most were right and are applied below. Section 9 lists the
ones I did not take, or took in a changed form, with the reason.

---

## 0. Short version

1. **Two things must be settled before anything else.**
   - **The deletion (answered).** The SEVA-era checkpoints and the Photo Tourism downloads were removed
     at 21:18 on the owner's go-ahead (see the update above). Free space is 65 GB (it was 40 GB).
   - **look1's held-out views do not match its evaluation sets.** The arms in
     `data/look1_build/arms.txt` hold out with `--holdout-seed 5`. The evaluation sets in
     `configs/look1_eval_sets.json` were built with seed 7. With seed 5, 65 of the file's 88 distinct
     target views would be training views [rechecked, with `timemachine.data.holdout` at the arms'
     settings]. No look1 run has started yet (no run folder, no log), so this can be fixed first.
2. **The aim.** Find out what date conditioning does across about 138 MegaScenes scenes (about 122 for
   training, about 16 held out). Four things are tested. Does Notre-Dame's per-scene state control
   survive when scenes are mixed? Does the date transfer to unseen scenes (by following dated references,
   and as era-wide content)? Does it leave unchanged scenes alone? Does the look code keep light,
   season and medium out of the date?
3. **Scenes are not the shortage. Old photos linked to modern ones are.** Only 39% of the old photos
   COLMAP posed share points with modern photos (2,204 of 5,588) [measured, scene_pool]. The selection
   has about 1,200 such linked old views today. Posing old photos ourselves could raise that to about
   3,800 [inferred]. Posing is the main lever for "as much good data as possible".
4. **Storage.** The 100+ scene set is about 50K views (an upper bound). Stored as Notre-Dame is, that
   is 41 GB. Stored compactly (512 crop, 256-px depth, sparse points), it is 8-9 GB. A middle layout is
   18 GB [inferred from measured bytes per view]. The compact layout needs a new loader path, because
   training today reads the 1024-px JPEG and undistorts it on the fly.
5. **GPUs are not the limit for building the data** (about 45 A6000-hours [inferred]). **They are the
   limit for training:** about 72K samples per A6000-day [measured]. At 200K samples, each change scene
   gets about 5-6K samples. That is between Notre-Dame's single-scene steps 2,000 and 4,000, where the
   in-region effect was still small. So the 100-scene run will probably not show per-scene learning in
   change scenes. It is built for the transfer questions.
6. **First a pilot of 14 training scenes and 6 held out.** Notre-Dame gets 30% of samples, so it reaches
   8K and 16K samples early and can be compared with look1's arms at matched samples. Each arm is 70K
   samples (about 29 h), which fits vulcan-high's 1.5-day limit. It has explicit kill points (5.3).
7. **The location signal is left for a later experiment, but recorded now.** Country, coordinates,
   continent and region are in tables for 791-802 of the 802 census scenes. A country one-hot is a weak
   design: 13 of the 38 training countries have one scene, so for those it would name the scene
   [measured].
8. **Corrections to today's studies** (draft and critique):
   - Vilnius Cathedral is not a change scene. It has 0 posed views from 1945-69 and 1 from 1970-99, the
     years without the roof statues [measured, SEVA-15 table].
   - The look code needs changed-region masks in change scenes **during training**, not only for
     evaluation.
   - "Date jitter clipped at event dates" is not implemented. Dates are drawn uniformly inside each
     view's interval (`training/loss.py: draw_dates`).
   - Most "colour" pre-1900 views are sepia prints on coloured card mounts, not colour photos.
   - Qwen3-VL-4B is not cached (an empty stub). The RoMa matcher (`romatch`) is not installed.
   - Several held-out scenes share files with training scenes (2.5).

---

## 1. Goal, questions and pass criteria

### 1.1 Goal

Train one date-conditioned MVGenMaster adapter on 100+ scenes and measure the four things below.
1. Scene-specific learning survives mixing.
2. The date transfers to unseen scenes in the two ways a date can transfer:
   - following the dates of the references;
   - era-wide content (vehicles, clothing, street furniture).
3. The date does nothing where nothing changed.
4. The look code keeps the photos' look out of the date across scenes.

A model cannot know that an unseen building burnt in 2019. So the unseen-scene tests are about
reference following and era content, not about the scene's own history [inferred, design part].

### 1.2 Questions, and what counts as a pass

Thresholds marked *proposed* are my choice and need the owner's agreement.

| # | Question | Where | Pass criterion | Instrument (status) |
|---|---|---|---|---|
| Q0 | Is quality kept? | pilot, 100-scene run | On held-out days, the median over training scenes of (loss at the true date) is at most the released model's. *Proposed tolerance:* at most 1 scene in 10 is worse than the released model by more than 2 standard errors of its own view sets. On held-out scenes, whole-frame PSNR is within run-to-run noise (0.4 dB [measured, findings]). | `eval_dates.py` (exists) |
| Q1 | Does per-scene event learning survive mixing, and at what cost per scene? | **pilot** (main purpose) | Notre-Dame's in-region wrong-date cost, at matched Notre-Dame samples (8K and 16K), is at least half of look1's at the same samples, on the same evaluation sets. Reference curves: look1_A/A2 for the arm without the look code, look1_B for the arm with it. Other change scenes are reported, not judged, in the pilot (too few views; 5.3). | look1 arms and `configs/look1_eval_sets.json` (after the seed fix) |
| Q2a | Reference following: on an unseen scene with references from two states, does the asked date choose the state? | 100-scene run (first look in pilot) | On each held-out change scene, the true date gives the lower in-region loss in most of its view sets. This must hold on 5 of 5 scenes (one-sided sign test p = 0.031), or in a pooled test over view sets with scenes as clusters. The in-region loss at the true date must also be below the released model's and below the same model with the date "unknown". Control for old-era scenes: mixed references + old date against modern-only references + the same old date, with the target's look code fixed; report date × look as a 2×2. | per-scene state read-out (exists for Notre-Dame: AUC 0.998, flips 28 of 28 [measured]); mixed-reference view sets (to build) |
| Q2b | Era prior: on an unseen scene with modern references, does an old date bring era-typical content without breaking the building? | 100-scene run | With the look code fixed to a modern colour look, the era read-out separates renders at the old date from renders at 2015 at AUC 0.7 or more. The old date is between 1900 and 1925 and at least 5 years after the structure's inception (Wikidata P571). This must hold on 8 of 9 eligible held-out scenes (p = 0.020). | era read-out (**to build**; validate at AUC 0.8+ on real photos of held-out scenes, and against degraded modern photos, 4.3) |
| Q3 | What does the date carry: era-wide or scene-specific? | 100-scene run | A measurement, not a pass/fail: the date-swap effect on training scenes (held-out days) minus the effect on held-out scenes, inside and outside the building mask. | building masks from SegFormer (cached) |
| Q4 | Null: on scenes where nothing changed, does the date leave the building alone? | pilot, 100-scene run | With seed and look code fixed, the building-region change between renders at 2014 and 2022 is *(proposed)* at most 1.5 times the change for a same-period swap (2014 vs 2016) on the same scene, and at most half the in-region change of a swap across the event on change scenes. | date sweep (partly exists) |
| Q5 | Does the look code stay out of the building's state, across scenes? | pilot, 100-scene run | Swapping look codes between the two states flips the state read-out in under 10% of targets. *Proposed:* the colour statistics of a render follow the requested code (correlation at least 0.8 on the 6 global numbers). | look code (look1, uncommitted) |
| Q6 | How do the answers scale with the number of scenes? | pilot vs 100-scene run | Q2a and Q4 scores on the same held-out scenes (the pilot's 6 are a subset of the 100-scene hold-out) are at least as good at 100+ scenes as at 14. | fixed hold-out (frozen before the pilot) |

Why the criteria changed from the draft:
- **Q1.** The draft compared with gate 3 (steps 4,000 and 8,000). Gate 3 held out a random 10% of views
  (88% of its held-out after-fire views share a day and photographer with training), had no look code
  and no camera balance [measured, findings]. look1's arms A, A2, B and D run 10,000 steps from scratch
  with the component hold-out and balance 0 [rechecked, arms.txt]. They are the right reference.
- **Q2a.** 4 of 5 scenes gives p = 0.19, so it proves nothing. With at most 5 change hold-outs, 5 of 5 or
  a pooled test is needed. The released model has no date input, so it is compared on loss at the true
  date only. The "unknown date" baseline exists: training drops dates 10% of the time, and
  `eval_dates.py` reports `gap_unknown` [rechecked].
- **Q2b.** The Vehicle Assembly Building (1966), Sydney Harbour Bridge (1932) and CN Tower (1976) cannot
  be rendered "at 1900". 8 of 10 gives p = 0.055.
- **Q4.** The draft's "twice the released model's floor (0.012)" cannot be met. That 0.012 is how much
  the released model's renders differ between two dates when it ignores dates. At step 8,000 the trained
  date already moves the rest of Notre-Dame's frame by 0.057-0.066 [rechecked, findings lines 1309 and
  1345-1348]. The Library of Birmingham (built 2010-13) is not a null for a 2008→2020 swap, so it is
  dropped (2.5).

**Unit of evidence.** For transfer the unit is the held-out scene. Within a scene, use view sets with
disjoint targets and 2+ noise seeds. Count the independent held-out units on each side of every change
before the hold-out is frozen. look1's hold-out gives Notre-Dame only 7-14 independent units per cell
[rechecked, `data/look1_build/eval/make_eval_sets.log`]; other scenes will have fewer. *Proposed:* a
change scene counts for Q1 or Q2a only with 3+ photographers and 6+ independent held-out units per state.

### 1.3 Choices carried over from Notre-Dame, and what must change for many scenes

- **One global calendar axis** for all scenes, as now: t = (year − 1900) / 100.
- **Per-view dates on references and targets.** Each view's date is drawn uniformly inside its date
  interval. **Nothing clips at events** [rechecked, `training/loss.py: draw_dates`]. Notre-Dame's builder
  drops fire-day views and low-confidence after-fire views by date midpoint
  (`scripts/mvgen/build_dataset.py: exclusion_reasons`). It does not check whether an interval straddles
  the event. The generic builder must split or exclude views whose interval straddles an event. 271 of
  Notre-Dame's 1,115 usable views are within 3 months of the fire [measured, design part].
- **The per-view look code**, for reference and target views, as in look1 (24 colour and medium numbers).
  At inference a target takes the references' code by default, or a code the user chooses.
  - **In change scenes the code needs masks before training.** `look.py` computes the code outside the
    changed region only when the folder has a mask; otherwise it uses the whole frame [rechecked]. The
    building colour (3 numbers) and the global statistics would then carry the state (cover, scaffold,
    sand). So every training change scene needs masks before its codes are computed, or its building
    entries must be set to "unspecified".
  - In scenes without masks (H and N), use the whole frame, as look1 does elsewhere. Facade cleaning is
    a state change the building colour can carry; Q5 measures whether that matters.
- **Camera balance is off in the pilot.** Against a matched control it raised the in-region wrong-date
  cost in 14 of 14 sets (+4.8 ± 1.1 points) [measured, findings], but only together with the region
  weight. look1's arms run with balance 0. Adding it to the pilot would confound Q1 (decision 9).
- **No region weight in training.** Its effect was small against a matched control, and the owner
  prefers not to use it. look1's arms pass `--region-weight 1`, which is the plain loss [rechecked,
  `step.py`].
- **Hold-out by whole capture day and photographer inside scenes, and by leakage group across scenes.**
  For old photos, day and photographer barely exist: 6.5% of pre-1900 old files are dated to a day, and
  16.8% of old photos are unattributed [rechecked, `scene_pool/work/old_files_sources.parquet`]. Use
  archive series, caption-and-date sessions and same-scan groups as units for them (2.3).
- **A sample never mixes scenes.** Each scene keeps its own scale factor (`scene.json`), its own period
  edges and its own hold-out.

### 1.4 What decides the pilot's arms: the look1 run

The question is whether the look code leaves Notre-Dame's in-region wrong-date cost intact (or raises
it): look1_B against look1_A/A2.
- If yes: both pilot arms run, one with the look code (compared with look1_B) and one without (compared
  with look1_A/A2). Together they test Q5 across scenes.
- If no: the pilot runs without the look code, and Q5 waits [design part].
- Fix the seed mismatch (0.1) before look1's results are read.

**Arm D (colour views turned grey or toned) does not get free examples from other scenes.** The draft
said 27% of pre-1970 views across 15 scenes are in colour, so many scenes show "the past in colour". The
27% (305 of 1,148) includes 146 of 328 pre-1900 views labelled colour [rechecked,
`appearance_study/judge_practice/02_log.txt`]. The contact sheet `sheet_pre1900_colour.jpg` shows mostly
sepia albumen prints on coloured card mounts, a few photochroms and a colour-checker strip. On
Notre-Dame the residual-chroma rule gives toned prints 11.9 ± 1.5 and colour photos 13.4 ± 0.8 outside
the region, so the two overlap [rechecked, make_eval_sets.log]. Validate the tone label by VLM or by eye
on about 300 old views across scenes before arm D or the multi-scene code relies on it.

### 1.5 Deliberately left for later

| Left out | Why | What to do now |
|---|---|---|
| **Location signal** (the owner's country idea) | The references already say which scene it is. Location can only help on unseen scenes, and only where enough training scenes share a region. Among old-rich scenes that is mainly Germany, France, Italy and the Netherlands [measured, design part]. | **Record now:** QID, present-day country (P17) and ISO code, continent, 8-region group, latitude/longitude, 10-km city cluster, leakage group, inception and demolition, events. Per view: photographer, upload time, source archive. Sources: `scene_pool/location_table.csv` (791 of 802 with coordinates, all with a country), `design/scene_location.csv`, `beyond/scene_wikidata.csv`. MegaScenes ships every scene's Wikidata record on S3. |
| How to test location later | | Arms with matched data and budget: no location; continent + country embedding, each with about 30% dropout to "unknown"; optionally low-frequency latitude/longitude. Evaluate on held-out long-history scenes in seen countries (3+ training scenes) and unseen ones, with the look code fixed, using the era read-out. Leak check: if the country can be predicted from the look code, its help is the archive's scan look. P17 is the present country, not the polity at the photo's date (Dresden 1960 = GDR). |
| A learned appearance encoder | Pretrained encoders read the building's state (leak AUC 0.96-0.98) [measured, appearance study]. | none |
| Sharpness, grain and vignetting in the look code | look1 is already built. These cues are not in the code, so the date can learn them (4.3). | Validate the era read-out against degraded modern photos now; consider these statistics for a later look version. |
| Date-code variants (integrated Fourier, shared learned steps) | Budget | a third pilot arm only if a slot is free |
| Data-parallel training | The trainer is single-GPU. | Write it only if a run needs more than 7 days. |
| Sources outside MegaScenes, the census beyond 802, rebuilding scenes without a model | Cost, licences, installs (section 3) | Smapshot as a posing benchmark, if time allows |
| The 1970-99 gap | 2,341 posed views in the 150 candidate models [measured] | Keep all posed 1970-99 views |

---

## 2. The data

### 2.1 The pool and its tiers (802 census scenes) [measured, scene_pool]

"Linked" means an old (pre-1970) photo shares at least 30 COLMAP points with at least 5 modern (2000+)
photos in the same model. It was computed from the true point tracks (`points3D.bin` of 150 models,
streamed, nothing stored). The check against Notre-Dame's stored matrix gave Spearman 0.95. 61% of posed
old photos link to no modern photo.

| Tier (10 linked old photos, 50+ modern posed) | Scenes | Linked old photos (best model) | Clean old photos not yet posed | What it gives |
|---|---|---|---|---|
| T1: 10+ linked old photos in one model | 64 | 1,447 | 9,043 | usable now with COLMAP poses |
| T2: old photos exist, fewer than 10 linked | 374 | 390 | 23,378 | needs our own posing |
| T3: modern only | 356 | 0 | 771 | modern looks; null scenes |
| none: no usable model (e.g. Eiffel Tower) | 8 | 50 | 697 | rebuild only (3.2) |

Other cuts of the same pool:
- 142 scenes have 10+ *posed* old photos in one model, 65 have 20+, 28 have 50+ [measured, design part].
- 265 scenes have 20+ dated old photos and a model with 200+ modern photos (24,863 unposed and 3,935
  posed old photos, 48 countries) [measured, beyond].
- With the Notre-Dame posing yield (10/15/25%), 139/170/221 scenes would reach 10 linked old photos
  [inferred]. That yield is measured on Notre-Dame only.

**Scenes with a dated, visible change** are the scarcest kind. About 15-20 candidates exist [inferred,
design part]. Several were named from memory and need an eye check (2.4).

### 2.2 Scene types used by the experiment

| Type | Definition | Role |
|---|---|---|
| C, change | a dated, visible change with posed views on both sides (masks possible) | Q1 in training, Q2a held out |
| H, long history | old photos, posed by COLMAP or by us | the bulk of the era signal; Q2b |
| N, modern only | no documented change, dense 2004-2023 | Q4 null; modern looks |

Sample shares in the 100-scene run: C 35%, H 45%, N 20%, then √(capped views) within a type, with no
scene above 3% [inferred, design part]. The pilot has 2 N training scenes and 2 N held-out scenes, so it
gives a first reading of Q4 (5.3).

### 2.3 Selection rule

Built by `scene_pool/scripts/assemble.py` (131 scenes). C scenes are added by hand (2.4).
1. **Gate.** At least 50 modern posed photos in the best model. "Core" also needs 10+ linked old photos.
   "Extension" needs clean old photos for our posing. "Modern-only" needs neither.
2. **Exclusions** [measured]: interior scenes (14); scenes whose old photos are mostly book plates,
   crops or portraits (12); single-batch scenes, where one archive and one year hold 60%+ of the old
   photos (18); scenes nested in another (29).
3. **Diversity caps:** 3 scenes per 10-km city cluster, 12 per country, 4 per dominant archive. 55
   scenes are in reserve, left out only by a cap. **The C additions break the city cap** (2.4); the
   exceptions are owner decisions.
4. **Shared files and duplicates.** 3.7% of photographic files sit in more than one scene. Hold out by
   leakage group. Drop files shared between held-out and training scenes, from the held-out side when
   the training scene is Notre-Dame (so its folder stays as it is). Before splitting, remove rescans
   across files and near-duplicates within each leakage group and city cluster.
5. **Hold-out units inside scenes.** Modern views: capture day and photographer. Old views: archive
   series, caption-and-date sessions (29.7% of old photos share a caption and date with another) and
   same-scan groups [measured, scene_pool].
6. **In the sampler:**
   - At most about 15% of old-target samples from any one archive. The Nationaal Archief gives 17% of
     core linked old photos, and Dutch scenes 23% [measured].
   - Inside a scene, keep all old and 1970-99 views and all views of an event's minority state.
   - Draw the modern views for camera coverage, capped at about 400 per scene, with at most 3 views per
     capture day and 5% per photographer.
   - **When the rules conflict, keeping the minority state wins.** The day and photographer caps then
     apply only to the majority state. Prague's 2017-18 photos are 98% one photographer, and
     Kiyomizu-dera's 2010 photos are 89% one photographer and 83% one day [measured, design part]. Such
     scenes are flagged in the shortcut report and count for Q1 or Q2a only under the unit rule of 1.2.

### 2.4 The selection [measured, scene_pool/selection.csv; C additions inferred]

| Role | Train scenes | Held-out scenes | Modern views (cap 400, upper bound) | Linked old views now | Old views expected after our posing |
|---|---|---|---|---|---|
| core (T1) | 44 | 5 | 15,975 | 1,080 | about 2,050 |
| extension (T2) | 46 | 6 (Hoover Dam moved here) | 15,681 | 134 | about 1,700 |
| modern-only (T3) | 27 | 2, plus 2-3 new nulls | about 11,300 | 0 | 0 |
| C additions, after an eye check: Sagrada Família, Prague Old Town Hall + Astronomical Clock (one group), Frauenkirche Dresden, Space Needle; held out: Kiyomizu-dera | 5 scenes (4 groups) | 1 | about 2,350 | few | few |
| **Total** | **about 122** | **about 16** | **about 45,000** | **about 1,200** | **about 3,800** |

- The modern-view counts use the largest model's modern photos, capped at 400, before cleaning and
  before the day and photographer caps. They are an upper bound. 29 of the 52 extension scenes and all 30
  modern-only scenes have `m_mod_posed` = 0 because their models were not streamed. Their co-visibility
  and cross-era links are unknown until built [rechecked].
- The Library of Birmingham is dropped (2.5).
- The training scenes span 38 countries, 13 with one scene. Germany 12, US 10, France 9, Italy 9, the
  Netherlands 7 [checked in the draft, rechecked].
- Notre-Dame stays in training and keeps its full `25512_model0_v2` folder: no 400-view cap, same
  layout. Otherwise the Q1 comparison with look1 breaks.

**C additions: status and problems** [rechecked, candidate_pool.csv and scene_file_overlap.csv].

| Scene | Status in the pool | Change | Problems |
|---|---|---|---|
| Sagrada Família | no role (T2) | gradual construction since 1882 | needs one chosen edge after the eye check (for example the Tower of the Virgin Mary, finished in December 2021; to confirm in posed views) |
| Prague Old Town Hall + Astronomical Clock | no role (T2); one leakage group (804 shared files) | 2017-18 clock renovation | 98% of 2017-18 photos are one city photographer. Shares 243 files with Old Town Square and 163 with Týn church (both training, same leakage group). Prague already has 3 selected scenes, so this is a 4th: cap exception, or swap Týn out. |
| Frauenkirche Dresden | reserve (Dresden city cap) | 1994-2005 rebuild | Would be the 4th Dresden scene (Zwinger, Semperoper, Theaterplatz are in): cap exception, or swap Theaterplatz out. Shares 68 files with Semperoper and 42 with Zwinger (training): de-duplicate. |
| Space Needle | no role (T2) | to confirm by eye | Shares 2 files with CN Tower (held out): drop them. |
| Kiyomizu-dera (held out) | no role (T2), not built | roof renovation 2008-20 | 2010 photos: 89% one photographer, 83% one day |
| Hoover Dam (held out, moved from training) | extension, 6 linked old photos | bypass bridge, about 2005-10 | 90% of its old photos are NARA, 42% one 1932 batch. Named from memory; eye check. |
| Milan Duomo | extension, 1 linked old photo | **none defined** | **Dropped as a C scene.** MegaDepth 0063 is another source (Flickr licences, EXIF dates); do not mix it in. |

### 2.5 Held-out scenes (frozen before the pilot) and known changes for evaluation

| Held-out scene | Type | Posed before / after | Notes | Leakage found [rechecked] |
|---|---|---|---|---|
| Opéra Garnier | C, construction 1860-75 | 12 / 401 | continuity with the 13-scene runs; only about 2 disjoint target sets before 1875 | Paris cluster with Notre-Dame and the Arc de Triomphe; shares 1 file with Notre-Dame (drop it from Garnier), 7 with the Eiffel Tower and 6 with Montmartre (neither in training) |
| Sydney Harbour Bridge | C, construction 1924-32 | 27 / 469 | southern hemisphere; not in Q2b (inception 1932) | **shares 209 files with Sydney Opera House** (same leakage group): bar the Opera House from training |
| Great Sphinx | C, sand before the 1925-36 excavation | 62 / 502 | | none |
| Kiyomizu-dera | C, recent | not built | after the eye check | none |
| Hoover Dam | C, recent | not built | after the eye check; not in Q2b (inception 1936) | none |
| Vilnius Cathedral | **H** (not C) | statues missing 1950-97: 0 posed from 1945-69, 1 from 1970-99 | the draft's 51 / 297 compares pre-1950 with post-1997, and both have statues | none |
| Yosemite | H, natural | | decision 6 | shares 81 files (28%) with Glacier Point (reserve; bar it) and 1 with Yellowstone (training; drop it) |
| Forth Bridge, Vehicle Assembly Building, Prambanan, Dendera, Jasna Góra | H, extension | after our posing | VAB not in Q2b (inception 1966) | to check after building |
| CN Tower, Geghard | N | | null scenes | CN Tower shares 6 files with Taipei 101 and 5 with Tokyo Skytree (both training) and 2 with Space Needle: drop them from the training side. Also 9 with Willis Tower, 8 with Burj Khalifa, 5 with Empire State (not selected). |
| 2-3 new nulls from T3 | N | | candidates with no documented change, no shared files, alone in their leakage group and city cluster: Mosteiro da Batalha, Badshahi Mosque, Azadi Tower, Badami Cave Temples [rechecked, candidate_pool.csv]; check by eye | none by construction |

- **Q2a has 5 change scenes at most**, and Garnier is thin. If an eye check fails, Q2a falls to 3-4
  scenes and only the pooled test remains.
- **Q2b's eligible scenes** (old date between 1900 and 1925, at least 5 years after inception): Garnier,
  Sphinx, Vilnius, Yosemite, Forth Bridge, Prambanan, Dendera, Jasna Góra, Kiyomizu-dera. That is 9.
- **Q4's nulls:** CN Tower, Geghard and the 2-3 new ones held out, plus the training N scenes (the date
  should not move their buildings either).
- **The Library of Birmingham is dropped.** It was built 2010-13 with no posed before-state, and
  Wikidata also gives it a 2016 "dissolved/demolished" date that I could not explain.
- **The city-cap decision (decision 8) must re-run this leakage check.** Lifting the Paris cap brings in
  Montmartre and Grand Palais, which share files with Garnier.
- **Known changes inside training scenes** are scored on held-out days (`scene_pool/known_change_eval.csv`):
  Notre-Dame fire (2,809 / 460 posed), Semperoper ruin 1945-85 (45 / 658), San Marco campanile 1902-12
  (23 / 704), Zwinger rebuild 1952-63 (12 / 1,222), Oslo City Hall, Völkerschlachtdenkmal, Palace of
  Westminster. Recent ones: Sagrada Família, Prague, Frauenkirche, Space Needle. Score only views whose
  date interval lies wholly on one side of the event.

### 2.6 What is gathered per scene and per view

| Item | What | Bytes per view | Source / code |
|---|---|---|---|
| Photo | 512×512 undistorted centre crop, as `MVGenScene.make_sample` reads it, JPEG q90-95 or WebP, with its intrinsics stored. Mounts, borders and scan targets cropped away first (4.3). No 1024-px source kept: S3 stays the archive. | 69-116 kB | S3 + `data/images.py` (to adapt) |
| Date | interval, confidence, source, rule flags, manual fixes; intervals split or flagged at events | < 1 kB | `src/timemachine/dating/`, census tables |
| Metadata | photographer (`artist`), upload time, categories, title, licence, source archive, series/session id | < 1 kB | cached Commons snapshot (S3) |
| Pose | COLMAP pose, intrinsics, k1/k2, pose source (COLMAP / ours), PnP inliers; sideways photos rotated rather than dropped | < 1 kB | `cameras.bin` + `images.minibin` |
| Depth | Depth-Pro scaled to the view's COLMAP points, `ok` flag. 16-bit log-depth PNG at 512 (lossless to 7e-5) or 256 (68 kB; median 0.1% change after upsampling). | 68-226 kB | `prepare_depth.py` |
| Co-visibility | sparse pairs with 30+ shared points, from the `points3D.bin` tracks | about 0.9 kB | builder (sparse format to add) |
| Sparse points | **kept for every view** (masks and PnP need them), float32/int32 | 18 kB | builder |
| Look-code inputs | SegFormer-B2 class map (128 px) and the 24-number look code; tone from the rule only after validation | about 0.7 kB | `prepare_look.py` (look1, uncommitted) |
| QC flags | rule hits, CLIP scores, VLM answers, exclusion reasons | < 1 kB | to build (4.3) |
| Changed-region masks | change scenes only; **used by the look code in training** and by evaluation | < 1 kB | `mvgen/regions.py` |
| Per scene | `scene.json` (scale), events-and-fixes file, shortcut report, location row, leakage group, hold-out lists | a few kB | 4.1 |

### 2.7 How much data in total [inferred from measured bytes per view, `inventory/storage_formats.csv`]

About 45K modern (upper bound) + 1.2-3.8K old + about 1-2K 1970-99 views gives **about 50,000 views**
in about 138 scenes.

| Layout | kB per view | 50K views | Notes |
|---|---|---|---|
| A: as Notre-Dame is stored today | 824 | 41 GB | Notre-Dame keeps it (0.74 GB, already on disk) |
| B: 512 crop JPEG q95 + 512 log-depth PNG + points | 361 | 18 GB | the same crop and depth as now |
| C+points: 512 crop JPEG/WebP q90 + 256 log-depth PNG + points | 157-170 | 8-9 GB | depth changes slightly; a short training check first |

- **Recommended:** C+points if the 40 GB budget holds, B if the 65 GB is confirmed. The draft's layout C
  dropped the sparse points that masks and PnP need.
- **The trainer must read both layouts.** Training today reads the 1024-px JPEG through `load_view`
  and undistorts it on the fly [rechecked, `mvgen/data.py: make_sample`]. The compact layout needs a new
  loader path with stored intrinsics. The bit-for-bit test (4.6) holds only on the old layout, which is
  why Notre-Dame stays in layout A.

Transient data, never kept:
- Images streamed from S3: about 50K × 0.3-0.5 MB = 15-25 GB of traffic, plus about 7 GB of old query
  photos for posing.
- `points3D.bin`: 2.1 GB in total for 150 models.
- `images.bin`: 20.2 GB in total, streamed one model at a time only where its keypoints are needed. The
  largest single file is 0.9-1.7 GB [measured, inventory and pipeline].

---

## 3. Sources

### 3.1 The census pool: main source

- **The 802 census scenes** are the source for this experiment. Their dates are parsed from
  MegaScenes' cached 2023-24 Commons snapshot, re-dated on 2026-11-19, and their poses are in tables.
- **Licences** are per file (mostly CC BY-SA and public domain). Keep licence and author per view, and
  do not redistribute images.
- **The snapshot ends mid-2024.** There are no photos of Notre-Dame's rebuilt spire.

### 3.2 MegaScenes beyond the top 800 [measured on a 50-scene stratified sample, extrapolated; beyond]

| What | Size | Cost | Use now? |
|---|---|---|---|
| Scenes passing "10 old + 10 modern" with our own posing | about 1,960 ± 500 more | census of strata A-C: 1.24M paths, about 17 GB streamed, about 1 h at 16 threads (no CPU slots, so inside a GPU job, where vulcan-default gives 4 CPUs) | **No.** Mostly Swedish (KMB) and Dutch (RCE) church inventories with tiny models. Later, to find more change scenes. |
| Scenes passing with today's poses | about 110 ± 56 | same | later |
| Big scenes with no or fragmented model (175 with 1,000+ files; Eiffel Tower: 523 old and 9,851 modern photos, no model) | 175 | COLMAP feature databases on S3 (Eiffel 6.5 GB); COLMAP/GLOMAP not installed | later; needs an install (decision 5) |

### 3.3 External sources [checked by API today unless cited; beyond/sources.csv]

| Source | What it has | Licence | Recommendation |
|---|---|---|---|
| **Smapshot** (HEIG-VD) | 101,032 ground-level images with a full geographic 6-DoF pose; 76,079 shot by 1945. Around the Grossmünster: 1,557, 92% dated within a year. | per image | **Independent ground truth for our old-photo posing**; later, posed old views for Swiss scenes |
| Commons, live | new states, e.g. Notre-Dame in 2024/2025/2026: 123 / 59 / 24 files | as Commons | later; needs an identifying user agent (decision 12) |
| PastVu | 811 photos around Notre-Dame (648 pre-1970) with year and direction | non-commercial; watermarks | not now (decision 12) |
| Ajapaik | Estonia and Finland; azimuth for 83% of a sample | varies | not now |
| LoC, Nationaal Archief, Bundesarchiv, Fortepan, Deutsche Fotothek | archives | open | mostly in MegaScenes already |
| ETH-Bibliothek, Gallica | about 560K online (ETH); press photos (Gallica) | ETH varies; Gallica non-commercial | barely in MegaScenes; later |
| MegaDepth (cluster, read-only) | 196 modern landmarks, EXIF dates 2004-2017 | research | optional N scenes, read in place; not mixed with MegaScenes scenes |

Evaluation-only sets: HistReNeRF/HistScene (230 archival photos, 3 landmarks; repository returned 404
today), Coloring the Past, Faroe rephotography pairs, LoMa's HardMatch (not released). No published
dataset gives posed photos of the same place across decades at scale [beyond].

---

## 4. Pipeline and quality control at scale

### 4.1 Steps per scene

| # | Step | Exists (ours / teammates') | Must be built |
|---|---|---|---|
| 1 | Scene list, tiers, leakage groups, hold-out | `scene_pool/` scripts and tables; Kevin's `splits.py` | **freeze** one split file (scenes, groups, held-out units) after the day-1 eye check |
| 2 | Dates | dating code; 802 scenes re-dated | strict category-year and IA book flags; suspicious exact days (1900-01-01: 111 files, 1865-01-01: 57, 1980-01-01: 56) as year-only; **publisher series** widened (4.3); **intervals that straddle an event** split or excluded |
| 3 | Events and fixes per scene | Notre-Dame's `DateFixes` (its schema names the fire) | a generic file: events (date, uncertainty, kind), per-file overrides, exclusion windows. Proposed from Wikidata (P793, P571, P576) and categories, confirmed by eye. |
| 4 | View selection | `select_views` (Notre-Dame-specific); Kevin's stable hash-based modern cap | the rule of 2.3; append-only rows, because depth is stored by row number |
| 5 | COLMAP | the builder parses `points3D.bin`; `scene_pool/scripts/true_covis.py` streams it | co-visibility from point tracks; stream `images.bin` only for the caption-strip rule and PnP anchors |
| 6 | Images | `data/images.py` (1024 px) | write the 512 crop and its intrinsics; crop mounts and borders first; rotate sideways photos (7.2% of Notre-Dame's rows) |
| 7 | QC flags | `registration_support.py`, `flag_views`; Kevin's `cleaning.py`; `pipeline/scripts/` (CLIP and rules, validated) | wire into the builder (4.3) |
| 8 | Our posing of old photos | VGGT-Omega with look-alike anchors (Notre-Dame only) | content filter → VGGT → matching + PnP (4.2) |
| 9 | Depth | `prepare_depth.py` (1.69 s per view on an A6000 [measured]) | compact storage; loader path for both layouts |
| 10 | Look code | `look.py`, `prepare_look.py` (look1, **uncommitted**) | SegFormer on the GPU inside the build job; masks first in C scenes; tone only after validation |
| 11 | Changed-region masks | `mvgen/regions.py` (edge = the fire) | one edge per event from the events file; C scenes only, before the look code |
| 12 | Shortcut report per scene | `design/scene_stats.csv`; Aadhitya's 14 appearance statistics and probes | one table per admitted scene; held-out units per side of each change |
| 13 | Evaluation sets and read-outs | `make_eval_sets.py`, `eval_dates.py`, `readout.py` (look1, uncommitted) | per-scene held-out view sets, mixed-reference sets, a state read-out per change scene, the era read-out |
| 14 | Contact sheets | findings tools | about 5 sheets per scene: old views, event windows, flagged views, views we posed, masks |
| 15 | **Regression test** | `25512_model0_v2` | rebuild Notre-Dame with the generic builder and diff it against the stored folder (views, exclusions, poses, co-visibility, depth within quantisation) |

CPU-heavy steps run inside GPU jobs through `slurm/mvgen_gpu_queue.sbatch`, because the CPU QOS is taken
by the owner's other jobs. A vulcan-default job gets only 4 CPUs and 32 GB.

**Caches must stay inside the repo** [rechecked, environment]. `TORCH_HOME` is
`/fs/nexus-scratch/ltahboub/torch_home`, `HF_HOME` and `UV_CACHE_DIR` are on `/vulcanscratch`,
`PIP_CACHE_DIR` and `TMPDIR` are under `/fs/nexus-scratch/ltahboub/`. All are outside the repo. Any first
download (romatch, Kornia fallbacks) would write there. Build jobs set `HF_HUB_OFFLINE=1`, point
`TORCH_HOME`, `HF_HOME`, `PIP_CACHE_DIR`, `UV_CACHE_DIR` and `TMPDIR` into the repo, and load the 32B VLM
from its snapshot path.

### 4.2 Posing old photos: the largest gain and the largest unknown

What we know from Notre-Dame [measured]:
- On the 107 old photos COLMAP registered, VGGT-Omega's rotation error has a median of 2.4°, its
  position error a median of 0.21 scene widths.
- Of 558 unregistered old photos, 80 pass the confidence filter. Of 8 checked by eye, 2 were wrong.
- With VGGT poses the depth-scale error is 13-18% (median), against 1.1% with the COLMAP pose. Only
  28-39% of the neighbours chosen for VGGT-posed views are co-visible.

So VGGT alone does not give references.

Route:
1. **Content filter first** (CLIP and VLM, including the new "artwork" class; 4.3).
2. **VGGT-Omega** with DINOv2 look-alike anchors, for rotation and focal length.
3. **Match each old photo to its 5-10 anchors** and solve its pose against COLMAP's points (PnP-RANSAC).
   Accept with 50+ inliers from 2+ anchors. The inliers act as observations, so the view gets
   co-visibility and a depth scale like a COLMAP view.
   - LoFTR and DISK+LightGlue (Kornia; installed, weights cached [checked in the draft]) give matches
     that do not fall on COLMAP keypoints. Lift them to 3D with the anchors' scaled depth maps (no
     `images.bin` needed), or snap them to keypoints within about 2 px.
   - RoMa: weights cached, `romatch` not installed [rechecked]. The installed `roma` is a rotation library.
4. **Fallback** (passes VGGT, fails PnP): target only, with co-visibility from reprojection (Jaccard
   overlap, optical axes within 60°: AUC 0.92 against COLMAP co-visibility, measured with COLMAP poses).

**Go/no-go, measured before the extension scenes are built:**
- Run the route on Notre-Dame's 107 registered old photos (truth known). These are the easy cases, and
  their own observations helped place the 3D points, so the result is optimistic.
- Then on the 558 unregistered ones and on 2-3 T2 scenes. **Check 30 random accepted poses by eye in
  each.**
- Go if at least 10% of content-clean queries pass PnP, accepted poses on the 107 are within 5° and 0.05
  scene widths, and at least 27 of each 30 checked by eye are right (*proposed*).
- If no-go: the extension scenes keep their modern views and few old views. The old-photo data stays
  at about 1,200 linked views.

### 4.3 Automated checks in place of the hand fixes [measured on 9,787 labelled views of Notre-Dame + 15 scenes; pipeline]

| Hand problem | Automated check | Measured | Then |
|---|---|---|---|
| Old photo with a modern (upload or scan) date | strict older-year rule **or** CLIP "old photograph" score > 0.8 | 51 of 60 found (42 of 42 outside Notre-Dame); 7 of 1,670 clean modern views flagged, 4 of them real old photos the hand check missed | auto-drop or re-date |
| Modern photo in the wrong year across an event | low date confidence inside an event window → "uncertain"; state read-out; VLM | upload date + no EXIF alone: precision 0.05 | change scenes only; a human confirms |
| Photos taken during the event; intervals straddling it | date or interval inside the event window | exact | split or exclude |
| **Postcard and publisher series** (Brück & Sohn, Photochrom, Detroit Publishing) | source and title patterns | 238 Brück & Sohn and 214 Photochrom/Detroit old files in the selected scenes [rechecked] | widen the interval to the series' printing years |
| **Artworks** (engravings, lithographs, paintings) | add an "artwork" class to CLIP and VLM; word rules | medium is unknown for 41% of old files in the selected scenes (9,568 of 23,422); artwork words match 69 to 918 files depending on the fields searched [rechecked] | drop before our posing |
| Multi-picture postcards | CLIP zero-shot | AUC 0.996-0.998 | auto-drop |
| Internet Archive book plates | title pattern **and** category | 69 of 69, nothing else | auto-drop |
| Other book pages, albums, interiors, caption-strip registrations | CLIP pre-filter (AUC 0.90-0.975), then VLM | CLIP alone has about 0.1 precision at the real base rate | VLM or contact sheet |
| **Mounts, borders and scan targets** | detect the print area; crop to it and shift the principal point, or mask outside it | 16% of linked old photos are scans of prints, cards or mounts by category, 40% of linked pre-1900 ones (lower bounds) [rechecked]; the contact sheet shows mounts and a colour-checker strip | crop; validate on Notre-Dame's old views. Otherwise these pixels get depth, are warped into targets as scene content, and act as an era cue the look code does not cover. |
| Stereo cards | aspect 1.8-2.3 before 1940, stereo words, CLIP | words found 3 of 4 missed cards; CLIP AUC 1.0 on 4 | auto |
| Sideways photos | roll from the pose > 45° | exact for posed views (CLIP AUC 0.38) | rotate |
| Duplicates and rescans | COLMAP views: 2 of 3 signs (built). Others: CLIP cosine ≥ 0.95 with overlapping dates, then a homography check | CLIP 34 of 39 known pairs, 540 other candidates | auto after the geometric check; before the split |
| Tone (grey / toned / colour) | residual-chroma rule | overlaps on Notre-Dame (1.4) | validate on about 300 old views across scenes by VLM or eye |
| Scene-level traps | top-year, top-day, top-photographer shares; interior share; modern count | exact statistics | scene gate |

**Process cues can fake Q2b.** Grain, sharpness, contrast curve, vignetting and the narrower old field
of view are not in the look code, so the date will learn them. The field of view reaches the model
through the camera rays and cannot be put in the look code. So the era read-out must be validated
against modern photos degraded to match (grey or toned, grain, blur, vignetting, contrast curve).
*Proposed:* it must separate real old photos from degraded modern ones at AUC 0.8+, and degraded modern
from plain modern at AUC 0.6 or less.

**The vision-language model (VLM).** Qwen3-VL-4B is not on disk: its cache folder holds only `refs/main`.
- Qwen3-VL-32B (63 GB) is complete under `/fs/nexus-projects/wildnvs/liger/weights/hf/hub`, read-only
  for us. It needs an H200 (held by the owner's other jobs), or 2-4 A6000s. vulcan-high (4 GPUs, 128 GB
  RAM, 1.5 days) suits it better than vulcan-medium (64 GB RAM for 63 GB of weights).
- It must run in the repo's `.venv` (transformers 5.18). MVGenMaster's venv has transformers 4.30.2,
  which cannot load Qwen3-VL [rechecked].
- vllm is not installed. Budget 1-2 s per image, not 0.2-0.5 s, and send it only old photos pre-filtered
  by CLIP (about 5-10K images: 1.5-5.5 GPU-hours [inferred]).
- Validate the prompts (`pipeline/vlm_prompts.json`) on our about 3K labelled views first. Keep a rule
  only where its precision is 0.9 or more.

### 4.4 What still needs a person [inferred]

| Task | Time | When |
|---|---|---|
| Eye check of the recent-change candidates: Sagrada Família, Prague, Kiyomizu-dera, Hoover Dam, Frauenkirche, Space Needle (is the change visible in posed views; are the states co-visible) | 2-3 h | **day 1**: it blocks the hold-out and the pilot build |
| Eye check of the 2-3 new null scenes | 0.5 h | day 1 |
| 30 random accepted poses in each posing test | 1-2 h | days 4-6 |
| Tone labels on about 300 old views (if no VLM) | 1-2 h | days 4-7 |
| Contact sheets: about 5 per scene at 1-2 min each, about 138 scenes | 11-23 h | days 9-11 |
| Event lists of the other change scenes | 2-4 h | days 9-11 |
| Masks of change scenes (overlay sheets) | 1-2 h | before their look codes |
| VLM validation spot checks | 1-2 h | days 4-7 |

### 4.5 Teammates' code and how to coordinate

- **Kevin (`origin/kevin/dataset-v1`, not merged).** Reuse `data/cleaning.py`, `data/splits.py` and the
  stable hash-based modern cap. `cache.py`, `multiscene.py` and `selection.py` are SEVA-only. No outputs
  of v1 were found; ask whether it ran.
- **Aadhitya (`origin/aadhitya/date-shortcut-probe`, `appearance-conditioning`).** The 14 appearance
  statistics and the probes become columns of the shortcut report. The FiLM pathway is SEVA-only.
- **Coordination:**
  - One builder (the folder layout of `timemachine/mvgen/data.py`), one split file, one scene list.
  - Commit the look1 work before the multi-scene code is built on it. At 22:00 the working tree had
    [rechecked, git status]:
    - new: `look.py`, `readout.py`, `prepare_look.py`, `make_eval_sets.py`, `fit_readout.py`,
      `configs/look1_eval_sets.json`, `tests/test_mvgen_look.py`, `tests/test_mvgen_eval_look.py`;
    - edited: `train.py`, `eval_dates.py`, `render.py`, and the data, date, model, step, probes,
      evaluation, trainer and hold-out modules.
  - Who owns which piece is the owner's call (decision 13).

### 4.6 The multi-scene trainer: what to write [pipeline, design]

1. A mixture of `MVGenScene` folders, reading both layouts (2.7). Each scene keeps its own scale,
   period edges and tolerances. Today's `ViewSamplingConfig` defaults are Notre-Dame's 1970 and 2019.2877.
2. Type-based scene shares with caps (2.2), and the archive cap (2.3).
3. Three kinds of draw: same-state; cross-state; **mixed**, with at least one reference per state. Q2a
   needs mixed draws, and the current sampler has none. In the pilot, Notre-Dame's draws stay exactly
   look1's (no mixed draws), so Q1 is not confounded; the other two-state scenes get them. *Proposed:*
   about a quarter of samples in two-state scenes (depends on decision 10).
4. Hold-out by scene group, and by day, photographer, series and session inside scenes.
5. A fingerprint and a scale factor per scene in each checkpoint.
6. The look code read per folder. Standardisation per scene or global is still open; I suggest global,
   so the code keeps each scene's average look. Today `LookStats` is per folder.
7. Probes across scenes with fixed seeds; per-scene losses and sampler shares in the log.
8. **Test:** a one-scene "mixture" of Notre-Dame (old layout) must reproduce the single-scene run's
   draws bit for bit, followed by an independent review as for gate 3.

The GPU side is unchanged: 8 views, 2.28-2.4 s per step of 2 samples, 8.6 GB [measured].

---

## 5. Storage, compute and timeline

### 5.1 Storage now [rechecked, 22:00]

| Place | State |
|---|---|
| `/fs/nexus-scratch` (400 GB quota, shared with the owner's other projects) | 336 GB used, 65 GB free. At midday: 361 used, 40 free. |
| repo | `data/` 16 GB (was 41 GB). `data/experiments` 3.8 GB (35 SEVA checkpoint folders emptied at 21:18:08). `data/phototourism` 37 MB (the Nov-19 view tables remain). `external/` 17 GB, `.venv` 6 GB. |
| `/vulcanscratch/ltahboub` (HF and uv caches) | 497 of 500 GB: no downloads there |
| Still freeable in the repo, if the owner agrees | `pretrained_model.zip` 1.4 GB (unpacked copy beside it), `data/audit_2026-11-19/gpu` 3.7 GB, literature PDFs 1.2 GB, the SEVA-era 1024-px images (`data/images`, 2.6 GB) once the pilot folders exist: about 9 GB. **The DUSt3R checkpoint (2.2 GB) stays:** MVGenMaster's own inference on unposed photos uses it (`run_mvgen.py`, README). |

**Budget while the deletion is unconfirmed (40 GB assumed free):**
- Layout C+points (8-9 GB for 50K views), run outputs 5-10 GB, transient files 2 GB at a time: at most
  about 21 GB. That leaves about 19 GB, above the floor.
- **Every build job stops at 15 GB free** (`df` at start and per scene).
- If the 65 GB is confirmed: layout B (18 GB) becomes possible, and 256-px depth needs no test first.
- For 200-400 scenes later, the remaining 9 GB must be freed or more quota requested (decision 2).
- Transient COLMAP and image files go to memory or a temporary folder inside the repo, deleted per scene.
  Node-local `/scratch0` is faster but outside the repo (decision 3).

### 5.2 Compute

**Per-job limits** [measured, inventory/compute.csv]:

| QOS | Per user | Per job | Use here |
|---|---|---|---|
| vulcan-default | 2 jobs | 1 GPU, 4 CPUs, 32 GB, **7 days** | the 100-scene run |
| vulcan-medium | 2 jobs | 2 GPUs, 8 CPUs, 64 GB, 3 days | builds |
| vulcan-high | 2 jobs | 4 GPUs, 16 CPUs, 128 GB, **1.5 days** | pilot arms (29 h), VLM |
| vulcan-*-h200 | 2 jobs each | 1-4 H200 | held by the owner's other jobs |

In practice that is about 4 single-GPU jobs at once. look1's arms take slots on days 1-2. A fresh job
waited about 2 hours [earlier notes]. No Slurm queries were made for this plan.

| Work | Rate [measured] | Amount | A6000-hours [inferred] |
|---|---|---|---|
| Depth, pilot | 1.69 s per view | about 10K views | 5 |
| Depth, rest of the 100+ set | 1.69 s per view | about 40K views | 19 |
| VGGT posing | 2.16 s per query | about 17K old photos | 10 |
| Matching + PnP | about 1.5 s per query (not measured) | about 17K | 7 |
| SegFormer + CLIP (GPU) | not measured on the GPU (CPU: 0.3-0.55 s per image) | about 50K | 1-2 |
| VLM, 32B | 1-2 s per image (estimate; never run) | 5-10K CLIP-filtered old photos | 2-6 |
| **Build total** | | | **about 45-50** |
| Released-model baselines on the pilot and held-out sets | 17-23 min per checkpoint on 14 view sets | about 20 scenes | 5-8 |
| Pilot: smoke run + 2 arms × 70K samples | 2.4 s per step; about 3.0 s per step with probes and renders (2,000 steps in 100 min) | 2,000 + 2 × 35K steps | about 60 |
| 100+ scene run: 200K samples | same | 100K steps | 67-83 |
| Evaluation | 17-23 min per checkpoint | about 20 scenes, several checkpoints | 10-20 |

**The training budget, in plain terms.**
- One A6000 gives about 72K samples a day (58K with today's probe and render overhead).
- 200K samples over about 122 training scenes is about 1,600 samples per scene. Notre-Dame needed about
  20K to pass gate 3.
- With C scenes at 35% over about 12 scenes, each gets 5-6K samples. That is Notre-Dame's single-scene
  steps 2,000-4,000, where its in-region wrong-date cost was +0.4% to +0.7% on after-fire targets and
  +0.9% to +2.8% on pre-fire targets (one set at 4,000) [measured, findings 2026-11-20].
- **So Q1 at 100 scenes, and the in-scene part of Q2a's training, will probably come out null because of
  budget, not because of the idea.** The 100-scene run is built for the transfer questions (Q2, Q4, Q5),
  which pool over scenes. Q1 is answered in the pilot.
- 200K samples take 67-83 h. That is more than vulcan-medium's 3 days, so it runs on vulcan-default
  (7 days) or resubmits with `--resume auto` (resume from `latest.pt` exists).
- The alternative is 400K samples (decision 11): 133-167 h, so 5.6-6.9 days, right at vulcan-default's
  limit. It needs lighter probing or a resubmit.

### 5.3 The pilot: defined

| | Scenes | Images on disk? |
|---|---|---|
| Training C (3 groups) | Notre-Dame; Prague Old Town Hall + Astronomical Clock (one group); Sagrada Família (one edge, after the eye check) | Notre-Dame only |
| Training H with old-era changes (3) | Semperoper (45 posed before the change), Piazza San Marco (23), Zwinger (12) | yes |
| Training H (6) | Royal Palace Amsterdam, Arc de Triomphe de l'Étoile, Saint Peter's Square, Helsinki Cathedral, Taj Mahal, Wartburg Castle (29 linked old photos; replaces Munttoren) | 5 of 6 |
| Training N (2) | Gateway of India, Matsumoto Castle (no documented change; confirm on the contact sheet) | no |
| Held out (6) | Opéra Garnier, Sydney Harbour Bridge, Great Sphinx (C, old-era); Vilnius Cathedral (H); CN Tower, Geghard (N). All are in the frozen hold-out of 2.5. Kiyomizu-dera joins if its eye check passes and time allows. | 4 of 6 |

- **Munttoren is out:** a reserve T2 scene with 1 linked old photo and 68 posed modern ones, so it gives
  no cross-era samples, although 184 of its 267 SEVA views are pre-1970 [rechecked]. **Milan is out:**
  no defined change (2.4).
- **Old-era change scenes in training are reported, not judged.** With 12-45 posed views before each
  change, they cannot carry a pass/fail.
- **Size:** about 10-11K views: 7,086 clean SEVA-era views in the 12 on-disk scenes (fewer after the
  400-view modern cap), Notre-Dame's 1,314 rows, and about 3,000-3,500 images to download (about 1.2 GB
  transient). Depth about 5 A6000-hours.
- **Sample shares:** Notre-Dame 30%, Prague + Sagrada 15%, H 40%, N 15%. At 30%, Notre-Dame reaches 8K
  samples at step 13,300 (about 11 h), 16K at step 26,700 (about 22 h) and 21K at the end, matching
  look1's 20K [inferred]. At the draft's 15% it would reach 8K only after about 22 h and never 16K.
- **Arms:** 70K samples each (35K steps, about 29 h with probes), so they fit vulcan-high. Both use
  balance 0 and no region weight, like look1. Arm 1: date + look code (if look1 is positive) + mixed
  draws outside Notre-Dame. Arm 2: the same without the look code. A third arm only if a slot is free.

**Kill points.**

| When | Check | If it fails |
|---|---|---|
| 2,000-step smoke run (about 1.7 h) | per-scene losses finite and falling; sampler shares within 2 points of the targets; no NaNs; probes run | fix and rerun. **Arm 2 starts only after arm 1 passes this.** |
| Step 13,300 (Notre-Dame 8K samples, about 27K total) | Q0 per scene on held-out days; Q1 against look1_A or look1_B at step 4,000 | Q0 fails: stop. Q1 far below: let it run, but note it. |
| Step 26,700 (Notre-Dame 16K) | Q1 against look1 at step 8,000; Q4 on the 4 nulls | as below |
| End (35K steps) | Q0, Q1, Q4, Q5; first look at Q2a on Garnier, Sydney, Sphinx | as below |

**Hard stops and what only reshapes the 100-scene run.**
- **Hard stops** (fix before the 100-scene run): Q0 fails; Q4 fails (the date moves buildings on null
  scenes beyond the control); the pipeline needed hand fixes beyond the contact sheets.
- **Reshapes the run:** Q1 fails (per-scene learning collapses when mixed). The 100-scene run then keeps
  its transfer questions, C scenes get a higher floor share, and per-scene learning becomes its own
  follow-up. Q5 fails: the 100-scene run drops the look code.

### 5.4 Schedule (day 1 = Wednesday 25 November; day 2 is Thanksgiving)

Human steps on days 2-5 (Thursday to Sunday) will slip. GPU jobs run unattended.

**Morning of day 1:**
1. (Done: the deletion was approved.)
2. Owner decisions 2-6 (section 7).
3. Eye-check Prague, Sagrada Família, Kiyomizu-dera, Hoover Dam, Frauenkirche, Space Needle, and the
   new null candidates.
4. Fix the look1 seed question (arms seed 5, evaluation sets seed 7).
5. Start the generic builder against the Notre-Dame regression test.

| Days | Work | GPU | Output, decision |
|---|---|---|---|
| 1 | Morning list. Freeze the split file (after the eye check). Commit the look1 code. | look1 | frozen hold-out |
| 1-3 | Generic builder: scene arguments, events-and-fixes schema, interval splitting at events, selection rule, compact layout and loader path for both layouts, mount cropping, sideways rotation, sparse co-visibility from `points3D.bin`. Port the 15 scenes' 312 flags. Events files for the pilot's C scenes. Notre-Dame regression test. | look1 | builder with tests |
| 2-4 | Build the 20 pilot folders; depth (about 5 h); masks for the pilot's C scenes, then look codes; released-model baselines on the pilot sets | 1 slot | pilot folders, noise floors |
| 2-5 | Write the matching + PnP code (LoFTR or DISK+LightGlue, lifted with anchor depth) | CPU | posing code |
| 3-6 | Multi-scene dataset and trainer; bit-for-bit test on Notre-Dame; independent review | CPU | trainer |
| 4-6 | Posing test: Notre-Dame 107, then 558, then 2 T2 scenes; 30 accepted poses checked by eye in each | 1 slot, about 3 h | **go/no-go for the extension scenes** |
| 4-7 | Era read-out (decade classifier on grey, masked non-building crops), validated on real held-out photos and degraded modern photos; state read-outs for the pilot's C scenes; mixed-reference view sets; tone validation; VLM validation on the 3K labelled views (if allowed) | 1 slot | instruments for Q2 and Q5 |
| 6-8 | Pilot: smoke run, then arm 1, then arm 2 (about 29 h each) | 2 slots | Q0, Q1, Q4, Q5 at the kill points |
| 5-11 | 100+ scene build: core and modern first, extension after a "go"; depth about 19 h, posing about 17 h, VLM | 1-2 slots | about 138 folders, contact sheets |
| 9-11 | Human pass (about 15-30 h, 4.4); pilot evaluation; **go/no-go for the 100-scene run** | eval | final fixes files |
| 11-18 | First 100+ scene run: 200K samples on vulcan-default, read at 100K and 200K | 1 slot (+1 if the pilot calls for a second arm) | first multi-scene result around day 15-18 (9-12 December) |

---

## 6. Risks, and how each is caught early

| Risk | Early sign | Check, and when |
|---|---|---|
| Space runs out (the quota is shared with the owner's other projects) | `df` at each job start | stop at 15 GB free |
| look1's evaluation is contaminated | 65 of 88 eval targets are training views at seed 5 | fix the seed before reading look1 (day 1) |
| Our own posing gives few or wrong poses | the Notre-Dame test, then eye checks on unregistered and T2 photos | day 4-6 go/no-go; Smapshot later |
| Date errors at scale (upload dates, book years, publisher series, artworks, a wrong year across an event, intervals straddling events) | rule and CLIP hits per scene; date histograms | automated checks (4.3) before depth; intervals split at events; human pass on change scenes |
| Mounts, borders and scan targets become scene content and an era cue | crop rate per scene; contact sheets of old views | crop rule validated on Notre-Dame (days 1-3) |
| Process cues (grain, blur, field of view) fake the era prior | era read-out scores on degraded modern photos | read-out validation (days 4-7); no Q2b claim if it fails |
| Shortcuts across scenes: camera (median AUC 0.75), sessions (10 busiest days hold 38% of modern photos), photographers (1.2% appear in both 2004-11 and 2015-23), narrower old field of view (AUC 0.60), the September contest effect (11% → 21% after 2015) [measured, design part] | the per-scene shortcut report | caps per day and photographer, hold-out by day, photographer, series and session; flagged scenes looked at before admission |
| Archive looks become "the era" (Nationaal Archief 17% of linked old photos) | archive share of old targets in the sampler log | 15% archive cap; Q5 swap test |
| Per-scene learning too slow when mixed | the pilot's Q1 at Notre-Dame 8K and 16K samples | pilot kill points |
| The 100-scene run is null on Q1 because of budget | known now (5.2) | state it; decision 11 |
| Held-out scene leaks into training | `scene_file_overlap.csv`; city clusters | the list in 2.5; re-run after any cap change |
| Too few held-out units in change scenes | units per side of each change | count before freezing; unit rule of 1.2 |
| The released model has seen these scenes (its split is not released) | — | claim date transfer, not new geometry; the released model is always the baseline |
| Masks fail for old-era changes or single-photographer periods | mask overlays per C scene | before look codes; fallback: state read-out only, or a hand-drawn 3D box |
| Trainer bug in the mixture | the bit-for-bit test | days 3-6; independent review |
| Jobs exceed QOS time limits | arm length vs limit | pilot arms on vulcan-high (29 h < 36 h); 100-scene run on vulcan-default or resubmits |
| Writes outside the repo through caches | env variables in the job | build jobs set all caches and `TMPDIR` into the repo, `HF_HUB_OFFLINE=1` |
| Holidays | day 2 is Thanksgiving | human steps moved off days 2-5 where possible |
| Wikimedia rate limits | — | nothing at volume depends on live APIs |
| Licences | per-view licence column | Commons only for now |

---

## 7. Decisions only the owner can make

> **Owner's answers, 2026-11-25:**
> - Compute: an H200, several A6000s or several H200s may be used for the 100-scene run (so the 400K-sample budget, or data-parallel training, is open).
> - VLM: use the existing Qwen3-VL-32B (read-only from the wildnvs folder).
> - Held-out set: fine as proposed (Garnier, Sydney Harbour Bridge, the Sphinx, Kiyomizu-dera, Hoover Dam held out).
> - City cap: 3 scenes per city.
> - Storage: besides the repo, our own subfolders of `/fs/nexus-projects/wildnvs` (169 GB free) and `/fs/vulcan-projects/worldtour` (143 GB free) may be used for data.
> - Kevin's dataset v1: unknown whether it ran.
> - The look1 training experiments start overnight.

1. **The deletion.** Answered: approved by the owner; 65 GB free.
2. **Storage.** May this experiment use up to about 21 GB (40 GB case) or 30 GB (65 GB case)? Compact
   layout (512 crop, 256-px depth, points, no 1024-px sources)? May the remaining about 9 GB be freed
   (zip copy, audit GPU outputs, PDFs, SEVA-era 1024-px images), or should more quota be requested? The
   DUSt3R checkpoint stays.
3. **Transient files:** in memory or a temporary folder inside the repo (default), or node-local
   `/scratch0`, which is faster but outside the repo?
4. **VLM:** Qwen3-VL-32B read-only from the wildnvs folder on vulcan-high (or an H200)? Download
   Qwen3-VL-4B (about 9 GB) into the repo? Or skip it and accept more hand checking?
5. **Installs inside the repo:** `romatch` (optional; LoFTR and DISK+LightGlue already work), and
   COLMAP/GLOMAP later for scenes without a model, with all caches inside the repo.
6. **The held-out set:**
   - Give up Garnier, Sydney Harbour Bridge and the Sphinx (old-era changes) and both Kiyomizu-dera and
     Hoover Dam (recent changes)? Hoover Dam moves out of training.
   - Natural or park scenes (Yosemite, Jasna Góra) in the hold-out at all?
   - Garnier shares Paris with Notre-Dame and the Arc de Triomphe and 1 file with Notre-Dame: keep it
     for continuity?
7. **Cap exceptions for the C additions:** Prague as a 4th Prague group (or swap Týn out) and Frauenkirche
   as a 4th Dresden scene (or swap Theaterplatz out)?
8. **Caps and exclusions:** 3 scenes per city or 4-5 (adds 11 T1 scenes, more Western-European and Dutch
   weight; the leakage check must be re-run)? The 18 single-batch scenes: out, or in with a sample cap?
9. **Training choices:** camera balance off in the pilot to match look1, and tested on its own later?
   Region weight kept out of training, as you said?
10. **What the product is:** "render an unseen place at any date from dated references"? That sets the
    share of mixed-reference draws.
11. **Budget for the 100-scene run:** 200K samples (about 3-3.5 days; Q1 at scale probably null) or 400K
    (about 6-7 days, at the vulcan-default limit)?
12. **Compute and external data:** the `nexus` account (tron partition) and multi-GPU jobs? An
    identifying user agent for a live Commons crawl (post-2024 photos)? Non-commercial sources (PastVu,
    Gallica) for internal training? Smapshot as a posing benchmark?
13. **Team:** who does the day-1 eye checks and the later human pass (about 15-30 h)? Fold Kevin's and
    Aadhitya's pieces into one builder? Did Kevin's dataset v1 ever run?
14. **Look code scope:** is colour-film look appearance (in the look code), like sepia and black-and-white?
    Add sharpness and grain statistics in a later look version?
15. **Location experiment:** confirm it waits until after the first 100+ scene run, with the fields of
    1.5 recorded now.

---

## 8. Not checked, and where the evidence is

**Not checked:**
- Today's queue, and whether look1 has been submitted since 22:00.
- LoFTR/DISK + PnP on old photos (the largest unknown).
- The VLM's speed and precision.
- SegFormer and matcher speed on the GPU.
- The training effect of q90 crops and 256-px depth.
- MVGenMaster's speed on A5000/A4000 or H200.
- Whether the change candidates show their change in posed views, and whether the new nulls are really
  unchanged.
- Whether masks work with few observing days.
- Kevin's v1 outputs; image overlap between external sources and Commons.
- The artwork count: 69 to 918 old files depending on which metadata fields are searched; the critique
  counted 747.

**The posing yield (10-25%) and all GPU-hour totals extrapolate from Notre-Dame.**

**Contradictions between the studies, resolved here:**
- Disk: 40 GB free at midday, 65 GB tonight, after an unconfirmed deletion. data/ was 34-41 GB, and is
  16 GB tonight.
- The 4B VLM: listed as cached, but it is an empty stub.
- RoMa: listed as available, but only its weights are.
- Garnier: described as "alone in its city cluster", but it is in the Paris cluster with two training
  scenes and shares 1 file with Notre-Dame.
- Vilnius: listed as a change scene, but the statue-less years have 0-1 posed views.
- Old-photo scene counts: "posed" (142 scenes with 10+) vs "linked" (64 with 10+). Linked is what the
  sampler needs.
- Country one-hot: called both "a scene id" and "not a scene id". Most scenes sit in countries with 10+
  scenes, but most countries hold 1-3 scenes, and for those it names the scene.

**Evidence:**
- `data/multiscene_plan/inventory/constraints.md`, `storage_formats.csv`, `compute.csv`
- `scene_pool/FILES.txt`, `selection.csv`, `candidate_pool.csv`, `tier_summary.csv`,
  `known_change_eval.csv`, `scene_file_overlap.csv`, `location_table.csv`, `work/old_files_sources.parquet`
- `beyond/notes.md`, `sources.csv`
- `pipeline/README.md`, `vlm_prompts.json`, the `*_measured.txt` files
- `design/notes.md`, `scene_stats.csv`
- `data/look1_build/arms.txt`, `data/look1_build/eval/make_eval_sets.log`, `configs/look1_eval_sets.json`
- `data/appearance_study/judge_practice/02_log.txt`, `sheet_pre1900_colour.jpg`
- `src/timemachine/training/loss.py` (`draw_dates`), `src/timemachine/mvgen/look.py`,
  `src/timemachine/mvgen/data.py` (`make_sample`), `scripts/mvgen/build_dataset.py` (`exclusion_reasons`)
- `docs/findings.md` (gate 3, region weight and appearance study entries of 2026-11-20 to 2026-11-24)

---

## 9. Critic's points not taken

Every point was checked. These were not taken, or were taken in a changed form.

| Point | What the critique said | What I did, and why |
|---|---|---|
| 1 (detail) | `data/phototourism/views` was emptied at 21:18:28-30. | Changed. `views/` still holds its 19 November view tables. It was `data/phototourism` itself and its `analysis/` folder that changed at 21:18:28-31, and the folder shrank from 7.0 GB to 37 MB. The point itself (state the times, ask the owner, do not count the 65 GB as settled) is taken. |
| 3 | Raise Notre-Dame's share to about 30%, with arms of 100K samples. | Taken, but the arms are shortened to 70K samples. At 30%, Notre-Dame then gets 21K samples, about look1's 20K, and each arm takes about 29 h, which fits vulcan-high's 36 h. 100K-sample arms would take about 42 h and need vulcan-medium. |
| 5 | Use 6 of 6 scenes or a pooled test for Q2a. | Changed. With Vilnius gone there are at most 5 change hold-outs, so 6 of 6 is impossible. The plan uses 5 of 5 (p = 0.031) or a pooled test. |
| 6 (detail) | Frauenkirche shares 265 files with Neumarkt. | True, but Neumarkt is not a selected scene, so that is not a leak. What matters is the Dresden cap and the 68 and 42 files shared with Semperoper and Zwinger, which are training scenes, so they only need de-duplication. |
| 9 (number) | 16% of linked old photos are print or mount scans, 19% before 1900. | The 16% is confirmed. Before 1900 I measured 40% of linked old photos (233 of 575; 315 of 1,934 overall), not 19%. The 19% looks close to the share among all pre-1900 old files (21%). The plan uses 40%. |
| 11 (number) | 747 old files match engraving, lithograph or painting words. | Could not reproduce exactly. I get 69 (titles and subcategories, narrow word list) to 918 (also artist and credit fields). The plan gives the range. The recommendation (an "artwork" class before our posing) is taken. |
| 15 (detail) | The 32B VLM on 2 A6000s needs vulcan-medium, whose 64 GB of RAM is tight. | Changed. vulcan-high also allows multi-GPU jobs (4 GPUs, 128 GB RAM, 1.5 days), which suits a short VLM job better. |
| 16 (detail) | Eye-check Milan on day 1. | Not taken. Milan is dropped as a change scene (no defined change; MegaDepth 0063 is another source), so there is nothing to check. |
| 20 | Budget about 400K samples, since vulcan-default allows 7 days. | Left to the owner (decision 11), with the null-by-budget warning stated plainly. 400K samples take 5.6-6.9 days at the measured step times, right at the 7-day limit, so they need lighter probing or a resubmit. The default stays 200K. |
| 10 (part) | Consider adding sharpness and noise statistics to the look code. | Taken as a later option, not for look1, which is already built and about to run. The validation of the era read-out against degraded modern photos is taken now. |
