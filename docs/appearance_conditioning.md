# Appearance conditioning: giving the medium its own pathway

## The problem this addresses

We measured that our capture dates are largely readable from the photographic
medium alone (`docs/findings.md`, `scripts/date_shortcut_probe.py`): across four
scenes a linear probe on colour, tone and grain separates pre-1945 from modern
photos at **AUC 0.89**, and sepia-toning a modern photo makes that probe call it
old. In our training data, "1890" and "brown, flat, grainy" are nearly the same
signal.

That is a plausible mechanism for the open blocker. SEVA copies the input views'
appearance; if era is mostly appearance, and appearance is already being copied
from the inputs, the date pathway has no residual left to explain - which is
exactly what "swapping a target's date costs ~0" has been telling us across
every run. It also fits what the LoRA did: it learned one sepia cast and applied
it at every date.

## The idea

Give each view an explicit **appearance** vector, conditioned on exactly the way
the date is, so lighting, tone and medium have somewhere to live other than the
date. The date pathway is then left with what actually changed about the place.

This is WildCAT3D's per-view appearance code applied to our problem. What is new
here is *where the vector comes from*: the same statistics we measured to be the
confound. A learned code (from the VAE latent, as WildCAT3D does) could absorb
anything, including era itself, which would defeat the purpose. Hand-picked
medium statistics cannot - they describe the photograph, not the scene, so
whatever the date learns afterwards has to be something else.

## What is in this change

| File | What |
|---|---|
| `src/timemachine/data/appearance.py` | 14 appearance statistics per view, from the cropped pixels the model sees |
| `src/timemachine/data/dataset.py` | `DatedViewsDataset(..., with_appearance=True)` adds an `appearance` (V, F) field |
| `src/timemachine/model/appearance_film.py` | `AppearanceEncoder` + per-block FiLM, mirroring `date_film.py` |

Design points worth knowing:

- **Zero-init.** Every per-block layer starts at zero, so a fresh model is
  exactly the model it wrapped. Step 0 is unchanged SEVA, as with date FiLM.
- **It nests.** `add_appearance_film` wraps whatever is in the ResBlock slot -
  SEVA's raw conv, or an existing `DateFiLMConv`. Nothing in `date_film.py` or
  in SEVA is modified, so this cannot break the date path.
- **Shared state, not a submodule.** Features are computed once per forward pass
  and read by every block, so recomputation under gradient checkpointing sees
  the same tensor. Same contract as `DateState`.
- **Statistics come from the model's pixels**, after resize and centre crop, and
  go through the same code as the measurement, so conditioning and measurement
  cannot drift apart (there is a test for exactly this).
- **Standardization travels with the checkpoint.** The raw statistics are on very
  different scales; `AppearanceEncoder.fit_normalization` sets buffers from the
  training set.
- **Given for inputs and targets**, as WildCAT3D does. At training time a
  target's own look is known; at sampling time it becomes a control: "render
  1890, as a modern colour photo" versus "as an albumen print".

## What is NOT in this change

Wiring into the training loop. That lives on the `laith` branch
(`src/timemachine/training/`, `scripts/train_dated_seva.py`), which is actively
being developed, so it should go in as a reviewed change rather than a
unilateral merge. What it needs:

1. `--appearance-injection film|none` and `--lr-appearance`, alongside the
   existing date flags.
2. Build the dataset with `with_appearance=True`; call
   `dataset_appearance_stats()` once and `fit_normalization()` on the encoder.
3. Encode once per forward pass in `TimeEmbedWithDate` (or beside it) and store
   into the `AppearanceState`, mirroring how date features are stored.
4. `--appearance-dropout` (10%, like `--date-dropout`) so the model learns to run
   without it, and **keep appearance in the unconditional CFG branch** - WildCAT3D
   found that dropping it there caused oversaturation.
5. Save/load via `appearance_film_state_dict` in the checkpoint.

## The experiment it is for

Two multi-scene runs, identical except for the flag:

| Run | What |
|---|---|
| `ms13_lora_f4_holdout` | the current date-conditioned baseline |
| the same + `--appearance-injection film` | appearance has its own pathway |

Scored on **identical** held-out samples with `scripts/eval_heldout.py`. The
question is whether the date-swap gap (`*_rel_swapped`) moves off ~0 once the
medium is accounted for elsewhere, without losing the LoRA's held-out gains.

A null result is still informative: it would mean the date-blindness is not
caused by the appearance confound, and the next suspect is the data (too few
genuinely informative old/modern pairs) rather than the architecture.
