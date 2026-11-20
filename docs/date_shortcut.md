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

**3. A causal check.** Modern photos are sepia-toned and fed back to the probe
fitted on untreated images. The scene is untouched, so any shift in the predicted
year is caused by the medium alone. Correlational ablations can be argued with;
this one cannot.

Splits reuse the near-duplicate grouping in `timemachine.eval.splits`, per scene,
so the same photograph can never sit on both sides.

## Reading the result

`skill` is the share of the constant-guess error the probe removes: 0 is useless,
1 is perfect.

- **High skill on `original`** → the date is trivially readable from the medium.
  Any model trained on this data can reach a good date loss without learning
  anything about how places change, and PSNR against old targets is partly
  measuring film look.
- **A large negative `median_shift`** in the sepia check → the same, demonstrated
  causally.

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
