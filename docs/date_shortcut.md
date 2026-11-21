# Is our date signal real, or just old film?

Every date-conditioned experiment in this repo rests on one assumption: that an
image's date tells you something about **what the scene looked like**. But in our
data, old photos are also grey, sepia, flat, soft and grainy, while modern ones
are colour and sharp. A model can score well on dates having learned only
"brown means old" — and `docs/findings.md` already shows this happening: in the
second training run, the LoRA learned a single sepia cast and applied it at
*every* date.

`scripts/date_shortcut_probe.py` measures how big that shortcut is in our own data.

## Run it

```bash
python scripts/date_shortcut_probe.py --dataset data/datasets/25512_model0
python scripts/date_shortcut_probe.py --dataset data/datasets/*_model0 --max-views 4000
```

CPU only, no SEVA, no torch. Minutes, not hours. Output lands in
`data/experiments/date_shortcut/`: `report.md` (paste into `findings.md`),
`treatments.csv`, `feature_groups.csv`, `split.csv`.

## What it does

A deliberately weak **linear** probe predicts the capture year from
appearance-only statistics — colour, tone, grain, framing, never scene content.
The question is not how well a model *could* date photos, but how much era
information is lying around for free.

**1. Treatments.** The same photos, three ways:

| treatment | removes |
|---|---|
| `original` | nothing |
| `grayscale` | colour |
| `tone_matched` | colour, and the tone curve (histogram equalized) |

**2. Feature groups.** The probe is also fitted on one cue at a time (`colour`,
`tone`, `detail`, `framing`), which says *which* cue carries the date rather than
only that one does.

**3. A causal check.** Modern photos are sepia-toned and fed back to the probes
fitted on untreated images. The scene is untouched, so any change is caused by
the medium alone. Correlational ablations can be argued with; this one cannot.

The headline is the **flip rate**: the share of modern photos the old/modern
classifier moves across its decision boundary once toned. The year-regression
shift is reported too, but read it loosely — across several scenes that probe is
pulled towards the middle of a bimodal year distribution, so its absolute shifts
understate the effect (on one scene it reported -104 years and 94% crossing 1945;
on four it reported -40 years and 5%, while the classifier flip rate stayed high).

Splits reuse the near-duplicate grouping in `timemachine.eval.splits`, per scene,
so the same photograph can never sit on both sides.

## Reading the result

**AUC** is the headline: the chance that a randomly chosen old photo scores as
older than a randomly chosen modern one. 0.5 is coin-flipping, 1.0 is perfect.

- **High AUC on `original`** → the date is trivially readable from the medium.
  Any model trained on this data can reach a good date loss without learning
  anything about how places change, and PSNR against old targets is partly
  measuring film look.
- **A large negative `median_shift`** in the sepia check → the same, demonstrated
  causally.

**Why AUC and not year error.** Our view tables are bimodal and about
three-quarters modern (Notre-Dame: 400 of 534 views are 2000+, median year 2009
but mean 1989). A least-squares fit is pulled towards the mean while the
constant-guess baseline sits at the median, roughly 20 years apart, so the `skill`
column in `treatments.csv` can go *negative* for a probe that separates old from
modern perfectly well. Plain accuracy is no better: with 17% old photos, always
answering "modern" scores 83%. AUC and balanced accuracy are immune to both.
`mae` and `skill` are still written to the CSV as secondary diagnostics.

Two caveats the report repeats, because both are easy to get wrong:

- **`tone_matched` is an upper bound, not a measurement.** It removes colour and
  tone but *not* grain, softness or scan resolution, which are medium cues too
  and cannot be removed without destroying the scene. On synthetic photos with no
  content at all, it still leaves the probe about a third of its skill.
- **Grayscale alone is not a defence.** An old print is flat and mid-grey as well
  as brown, and that tone survives desaturation — in tests the grayscale row
  sometimes scores *higher* than the original. The literature review suggests
  grayscale evaluation as a control; on its own it is not enough.

## Why this matters for the model

If the shortcut is large, it is a plausible explanation for the open blocker in
`docs/findings.md` — that swapping a target's date costs the model ~0. The era
look is already fully predictable from the input views' appearance, which SEVA
copies, so the date pathway has no residual left to explain. That is the argument
for giving the model an explicit per-view **appearance** representation
(WildCAT3D-style) so that lighting, weather and medium are absorbed somewhere
else, leaving real scene change as the thing the date has to account for.

This probe does not prove that. It measures the premise.
