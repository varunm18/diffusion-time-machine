# Plan to a working multi-scene checkpoint by the presentation (Monday 1 December 2026)

Written Wednesday 25 November, 16:30. Scope is cut to what can be trained and shown in time; the full
plan is `docs/multiscene_plan.md` and stays the reference for later.

## What we will show on 1 December
1. Notre-Dame, single scene (done or finishing now): the date flips the building's state on held-out
   views (gate 3), and the untrained look code steers light, colour and black-and-white (look1).
2. **One multi-scene checkpoint** (about 120 training scenes, about 12 held-out scenes), trained with date
   + look code, and what it does:
   - on training scenes, held-out days: true date vs wrong date (Notre-Dame fire; Semperoper ruin,
     San Marco campanile, Zwinger rebuild where views allow);
   - on held-out scenes: date sweeps (Garnier, Sydney Harbour Bridge, Great Sphinx: their old states),
     look-code control, and the released model as the baseline.
3. Honest limits (see the end).

## Cuts made to fit the time
- **No posing of old photos by us** (VGGT + PnP is the largest unknown): only views COLMAP posed.
- **No vision-language checks**: rule-based checks only (sideways, duplicates, stereo cards, bad lens,
  date width and confidence). Hand-made date fixes exist only for Notre-Dame.
- **Storage layout as Notre-Dame** (1024-px images + 512 depth), so the trainer's loader is unchanged;
  data goes to our own folder on `/fs/nexus-projects/wildnvs`.
- **Changed-region masks only for Notre-Dame**; other scenes use whole-frame look codes.
- **No region weight, no camera balance** (as in look1).
- **Location signal:** recorded, not used.

## Recipe for the run
- Start from the released MVGenMaster (fresh adapters), date + look code as look1 arm D (look code
  for every view; colour views turned grey 10% / sepia 5% at unchanged dates). Look-code statistics
  standardised over all scenes together.
- The look-code decision is checked tonight: arm A (date only) reaches step 6,000 around 21:30; if A's
  date controls the building clearly better than B's at the same step, the run drops the look code
  (or runs both arms if GPUs allow).
- Sampling: a scene is drawn with weight proportional to sqrt(its usable views), Notre-Dame fixed at
  15%; inside a scene, the existing co-visibility sampler. Hold-out per scene by day-and-photographer
  units (10%), and 12 whole scenes held out.
- **Data-parallel training on 4 GPUs** (each GPU draws its own samples; gradients of the adapters are
  averaged before every step). Target: about 2.5 days of training, about 600-800K samples on A6000s,
  more on H200s.

## Schedule
| When | Work | Done when |
|---|---|---|
| Wed evening (now) | Generic scene builder (from `scripts/mvgen/build_dataset.py`): any scene id, no Notre-Dame date fixes, all old and 1970-99 views + up to 300 modern views, generic flags. Smoke test on 2 scenes. | 2 scene folders pass `verify` |
| Wed night | Build all ~130 scene folders in GPU jobs (downloads, COLMAP points, co-visibility, scale) | ~130 folders, a summary table |
| Wed night - Thu | Depth (Depth-Pro) and look codes for all views, on 3-4 GPUs | every folder has depth and look codes |
| Thu | Multi-scene dataset + trainer (scene mixture, per-scene hold-out, global look statistics, probes on a few scenes) and data-parallel training; CPU tests; 30-minute smoke run on 2 GPUs | smoke run's losses fall, checkpoints resume |
| **Fri morning** | **Launch the main run on 4 GPUs** (H200s preferred). This is when the owner may need to free GPUs. | run going |
| Fri - Sun | Training. Evaluation every ~12 h on Notre-Dame sets and a few held-out-scene date sweeps; pick demo views | renders and numbers |
| Sun night | Final checkpoint, final evaluation, figures for the slides | slide material |

## Risks and fallbacks
- Builder or data bugs: smoke test on 2 scenes first; summary table per scene before depth.
- Data-parallel code not ready by Friday: start one single-GPU run per arm instead (fewer samples, still a
  checkpoint), add the data-parallel run when ready.
- GPU queue waits (2+ hours seen): submit early; ask the owner to free H200s on Friday morning.
- Misdated photos in other scenes (no hand fixes): accepted; date effects will be noisier outside
  Notre-Dame.
- A per-scene effect may not appear in 2.5 days (Notre-Dame alone needed ~8K steps): the presentation then
  shows the transfer and look-code results and says so.
