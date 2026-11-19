# Plan: teaching MVGenMaster time (2026-11-19)

**Goal.** Give MVGenMaster a per-view capture date, so that from a few photos of a place it renders new views *as the place looked at a chosen date*. First milestone: overfit one Photo Tourism scene as well as possible.

**Status today.** MVGenMaster is installed and tested on our data; the two measurements that decide whether it is a sound base are done (section 2). No training code is written yet.

---

## 1. Why MVGenMaster, and what it is

| | SEVA (until now) | MVGenMaster |
|---|---|---|
| Base | SD 2.1-base, predicts the noise, noise levels up to σ = 85 | SD 2.1, **v-prediction**, zero-terminal-SNR schedule |
| Trained on in-the-wild photos | no evidence | **yes: MegaScenes** (about 62k scenes, per its paper), plus 11 other sources |
| Training code | not released | **released** (`train.py`, configs), Apache-2.0 |
| Geometry conditioning | camera rays only | camera rays **+ input views warped into each target by depth** ("3D priors") |
| Views per pass | 21 (padded) | 8 in training, up to 28+ at inference |
| Appearance differences between photos | not handled | a **domain switcher**: one learned embedding per data source, added to the timestep embedding, zero-initialised |
| Our result on Notre-Dame, plain model | 11.6 dB at its best | **12.6 dB** |

Facts that shape the plan (from its code, `external/MVGenMaster`):
- **Inputs to the network** for each of F views:
  - the noisy latent (clean for reference views);
  - a mask (0 = reference, 1 = target);
  - 6 Plücker ray channels;
  - the 3D priors: a Fourier-encoded map of warped 3D coordinates plus the warped reference pixels.
- **No text or CLIP branch.** Attention is full 3D self-attention over all views.
- **Depth is needed for every reference view**, in the same scale as the cameras. The authors used monocular depth scaled to SfM points for training, and DUSt3R or Depth-Pro at inference.
- **The domain switcher** is `class_embedding` (3 entries: others, megascenes, objaverse). It is per sample; the embedding is then repeated for every view and enters each ResBlock.
- **Training recipe of the authors:** full UNet (819M parameters), AdamW lr 5e-5 then 2.5e-5, EMA 0.9995, fp16 autocast with the loss in fp32, gradient clip 1.0, v-prediction MSE on the target views only, 10% dropout of the camera rays and 15% of the priors (for guidance).
- **Released weights** are an EMA UNet stored in fp16.

---

## 2. What we measured today

### 2.1 It runs, and it is better than SEVA on our data
*`scripts/mvgen/nvs_test.py` on view sets exported by `scripts/export_nvs_samples.py`: our COLMAP cameras, real target photos, 2 noise seeds, guidance 2, 50 steps. Depth: Depth-Pro scaled to each view's COLMAP points.*

| View set (Notre-Dame, 3 inputs → 5 targets) | SEVA, 8 frames | SEVA, 21 frames | MVGenMaster (domain "others") | MVGenMaster (domain "megascenes") |
|---|---|---|---|---|
| Easy same-year sets (mean of 4) | 9.6 dB | 11.6 dB | **12.6 dB** | 11.7 dB |
| Fire probe (pre-fire inputs, post-fire targets) | 9.9 dB | 10.8 dB | **11.8 dB** | 11.3 dB |

- Renders are sharp, geometrically right and stable across seeds; no fragmentation.
- On the fire probe it draws the intact cathedral, as a model without dates must. That is the gap to close.
- Depth-Pro scaled by one number per view agrees with COLMAP's points to 1–14% (median error) on the easy sets and 7–19% on the fire probe's inputs.
- The authors' demo (3 photos → 25 new views) takes 41 s on an A6000.

### 2.2 Its gradient is sound in bf16 (the test SEVA failed)
*`scripts/mvgen/grad_precision.py`: one real training step, gradient compared with fp32.*

| Precision | Cosine with the fp32 gradient (mean over 5 noise levels) | Gradient norm vs fp32 |
|---|---|---|
| bf16 autocast | **0.998 – 1.000** | equal |
| fp16 autocast without a gradient scaler | 0.77 – 0.82 | 0.8 – 1.5× |
| (SEVA, bf16, for comparison) | about 0 | about 1,000× |

- So MVGenMaster can be trained in **bf16** (fast) or fp32. Never fp16 without a scaler.
- Reason: with v-prediction the loss is 0.17–0.42 on our sample, not a tiny difference of two nearly equal numbers.

---

## 3. Design

### 3.1 Where the date goes
A **per-view date embedding added to the timestep embedding of each view**: the same place as the domain switcher, but one value per view instead of one per sample.

- Encoding: year → `t = (year − 1900) / 100` → low-frequency Fourier features (periods of 25 years and longer, plus `t` itself) → 2-layer MLP → 1,280 dimensions (the timestep-embedding width). The last layer starts at zero, so at step 0 the model is exactly MVGenMaster.
- A learned "unknown date" vector, used when a view has no date and for date dropout (10% of samples), so guidance on the date is possible.
- Implementation without editing the vendored code: a forward pre-hook on every `ResnetBlock2D` adds the per-view vector to the `temb` argument. This equals adding it to the timestep embedding after it is repeated per view.
- If a single added vector proves too weak (it was for SEVA, but that result came from noise-trained runs), the fallback is a per-block scale and shift (FiLM) through the same hooks.

### 3.2 The 3D priors work against date changes; plan for it
The warped-pixel prior hands the model the reference photos' pixels *as they looked on the references' dates*. For a target at another date the right output differs. Three options, to be compared in the two-sample test (section 5, gate 2):

| Option | What changes | Risk |
|---|---|---|
| A. Keep both priors, let fine-tuning learn to override them | nothing | the model keeps copying the references' look |
| B. Drop the warped pixels when a reference's date differs from the target's by more than a threshold; keep the warped 3D coordinates | the 4 pixel channels are zeroed per target (the model already saw zeroed priors 15% of the time) | less texture detail on cross-date targets |
| C. Keep both and also give each view the *date difference* to its references | a second small embedding | more machinery |

Start with A and B side by side.

### 3.3 What to train

| Stage | Trainable | Why |
|---|---|---|
| 1 | date encoder only (~2M parameters) | can the date slot alone change the output? Cheap, and safe for the base model |
| 2 | + LoRA on the attention and feed-forward layers (rank 32–64) | lets the model change structure (scaffolding, missing spire), not just tone |
| 3 (only if needed) | full UNet at lr 1e-5 with EMA, as the authors did | maximum capacity; memory not measured yet, and it needs care on a single scene |

- Precision: bf16 autocast, loss computed in fp32 (measured sound, 2.2). Keep `--precision fp32` as an option and re-run the gradient check once adapters exist.
- Optimiser: AdamW, cosine decay, no fixed gradient clip (a spike guard only), EMA of the trainable weights.
- Views per step: 8 (3 references + 5 targets), as the model was trained. Gradient accumulation of 4–8 samples per step, because one sample per step was too noisy before.

### 3.4 Loss and dropout
- The authors' loss: v-prediction MSE on the target views, uniform timesteps.
- Dropout for guidance: camera rays 10%, priors 15%, **date 10%** (whole sample set to "unknown").
- Dates are drawn inside each view's date interval (as now).
- Later, only if the date is ignored: the date-discrimination term (true date must beat a wrong date), with the reference pass run first and without gradient.

### 3.5 Data: one Photo Tourism scene
**Scene: Milan Cathedral (MegaDepth 0063)**, chosen by the Photo Tourism analysis (docs/findings.md, 2026-11-19).
- 1,696 posed views on the cluster, 1,045 with exact EXIF dates (2002–2017). View table: `data/phototourism/views/megadepth_0063_milan_cathedral/views.parquet`.
- The change to learn: facade scaffolding and hoardings from 2005 to 2008, clean from January 2009. It is local and early, so the target is a clear before/after, checked at the facade's lower half.
- Gate 2's two samples: frontal views from 2006–08 (hoardings) against frontal views from 2010 onward (clean).

What the pipeline needs:
- A view table like ours: image path, COLMAP pose and intrinsics, date interval.
- **A depth map for every view**: the dataset's own depth maps where they exist, otherwise Depth-Pro scaled to the view's COLMAP points (2.1). Views whose scaled depth disagrees with their points by more than ~15% are not used as references.
- Upright images only (apply the EXIF rotation to image and camera), no duplicates.
- **Sampling that keeps samples well-posed** (the audit found 24–74% of our SEVA targets were not visible from their inputs):
  - choose references by *shared COLMAP points* with the targets, not by camera distance;
  - require each target to share at least ~50 points with some reference;
  - take account of field of view;
  - then prefer references from other dates.
- Hold out 10% of views per period for evaluation, plus the probe targets.

### 3.6 Evaluation
- **Baseline:** the untouched MVGenMaster on the same view sets and seeds.
- **Numbers:** PSNR and LPIPS against the real held-out photo, rendered at its true date and at a wrong date. Success means the true date wins clearly. The same comparison at the loss level (as our held-out probe does).
- **Pictures:** one fixed camera rendered across the years (a time-lapse strip), 3 noise seeds each.
- Every probe is read against its step-0 render.

---

## 4. Implementation steps

New code goes in `src/timemachine/mvgen/` and `scripts/mvgen/`, runs in MVGenMaster's environment (`external/MVGenMaster/.venv`, our package installed into it in editable mode), and never edits the vendored checkout.

| # | Step | Files | Done when |
|---|---|---|---|
| 1 | Model wrapper: load the released model (as `nvs_test.py` does), date encoder, ResBlock hooks, save/load of the trainable part | `mvgen/model.py`, `mvgen/date.py` | with the zero-init date module the output equals the released model's, bit for bit; a test covers the hook with 2 samples × 8 views |
| 2 | Conditioning + loss: one function that builds the batch (latents, mask, rays, priors) and returns the v-prediction loss, mirroring `train.py` | `mvgen/step.py` | its loss equals the one computed in `grad_precision.py` on the same sample |
| 3 | Data: view table → sample with images, cameras, depth, dates; co-visibility sampler; depth preparation script | `mvgen/data.py`, `scripts/mvgen/prepare_depth.py` | exported samples pass the visibility check; depth agreement is reported per view |
| 4 | Trainer: single GPU, bf16, EMA, cosine schedule, accumulation, resumable, probes | `mvgen/trainer.py`, `scripts/mvgen/train.py` | gate 1 below |
| 5 | Probes: loss with true / wrong / unknown date on held-out views; renders across years with several seeds | `mvgen/probes.py` | step-0 probes reproduce section 2.1's numbers |
| 6 | LoRA: reuse `timemachine.model.lora` on the UNet's attention and feed-forward linears | `mvgen/model.py` | zero-init equals the base model; gradient check repeated with adapters |

---

## 5. Gates (each must pass before the next)

| Gate | Test | Pass |
|---|---|---|
| 0 | Plain model on our data; gradient precision | **passed today** (section 2) |
| 1 | Overfit one fixed sample (LoRA + date): loss on that sample | falls by well over half within a few hundred steps, and the render reproduces the target photos |
| 2 | Two samples with the same cameras' neighbourhood but different dates (the Photo Tourism scene's clearest change) | rendering the same cameras at the two dates gives the two different appearances; compares options A and B of 3.2 |
| 3 | The whole scene, with held-out views | on held-out views the true date beats a wrong date in loss and in PSNR, without losing the baseline's quality at the references' own date |
| 4 | Several scenes | the date effect transfers to a held-out scene |

---

## 6. Risks and open questions
- **Date range of Photo Tourism.** Confirmed: nothing in it spans more than about a decade (MegaDepth to early 2017, IMC to 2014). Milan gives one clear change inside that decade. For control over decades, the dated MegaScenes scenes (Notre-Dame: 1850–2023) remain the only source, and MVGenMaster was trained on MegaScenes. Plan: Milan for gates 1–3, then Notre-Dame as the long-range test. 5Pointz is the option for dense change.
- **The priors may dominate** (3.2). This is the main technical risk, and gate 2 is designed to expose it early.
- **Depth for old or unusual photos** (prints, night shots) may align poorly with COLMAP; such views can still be targets but not references.
- **Single-scene training of a 819M-parameter model** will memorise unless most of it stays frozen; hence stages 1–2 first, with held-out views from the start.
- **Domain label.** "Others" scored higher than "megascenes" on same-date sets. For targets at another date the "megascenes" label (trained on photos that differ in look) may be the better starting point; compare both at gate 2.
- **Camera scale.** MVGenMaster shrinks scenes whose cameras span more than 5 units and leaves smaller ones alone; our Notre-Dame sets span 0.1–1.3 COLMAP units. Check on the new scene that results do not depend on the reconstruction's arbitrary unit.

---

## 7. What exists already

| What | Where |
|---|---|
| MVGenMaster checkout, weights, own environment | `external/MVGenMaster/` (not in git), `slurm/mvgen_gpu.sbatch` |
| SD 2.1 files it needs | `external/MVGenMaster/check_points/stable-diffusion-2-1/`, from the mirror `sd2-community/stable-diffusion-2-1` (the original repo is gone) |
| Export of view sets with cameras and COLMAP points | `scripts/export_nvs_samples.py` |
| Test on our view sets (cameras, depth alignment, PSNR, sheets) | `scripts/mvgen/nvs_test.py`, results in `data/experiments/mvgen/nvs_test/` |
| Gradient-precision gate | `scripts/mvgen/grad_precision.py`, results in `data/experiments/mvgen/grad_precision.json` |
