# Phase C: does giving the medium its own pathway make the date matter?

## The hypothesis

Across four scenes, a linear probe on appearance statistics alone - colour, tone,
grain, **no scene content** - separates pre-1945 from modern photos at AUC 0.89,
and sepia-toning a modern photo makes the classifier call it old
(`docs/findings.md`). So in our training data "1890" and "brown and flat" are
nearly the same signal.

SEVA copies the input views' appearance. If era is mostly appearance, and
appearance is already arriving via the inputs, the date pathway has no residual
left to explain - which is what every run has measured: swapping a target's date
costs ~0.

**Hypothesis:** give the model an explicit appearance pathway, and the date
pathway is left with what actually changed about the place.

**Prediction:** the date-swap gap (`*_rel_swapped` on held-out views) moves off
~0, without losing the LoRA's held-out gains.

**A null result is still a result.** If appearance conditioning changes nothing,
the confound is not what is blocking date learning, and the next suspect is the
data - too few genuinely informative old/modern pairs of the same place - rather
than the architecture.

## Running it

Two runs, identical except for one flag:

```bash
# A: the current date-conditioned baseline
SCRIPT=scripts/train_dated_seva.py ARGS="--dataset data/datasets/*_model0 \
  --run-name ms_lora_f4_holdout --holdout-fraction 0.1 --holdout-probe-targets \
  --weighting edm --probes configs/probes_notre_dame.json" sbatch slurm/seva_gpu.sbatch

# B: the same, plus an appearance pathway
SCRIPT=scripts/train_dated_seva.py ARGS="--dataset data/datasets/*_model0 \
  --run-name ms_lora_f4_appearance --holdout-fraction 0.1 --holdout-probe-targets \
  --weighting edm --probes configs/probes_notre_dame.json \
  --appearance-injection film" sbatch slurm/seva_gpu.sbatch
```

Then score both on **identical** held-out samples:

```bash
python scripts/eval_heldout.py --runs data/experiments/train/ms_lora_f4_holdout \
                                      data/experiments/train/ms_lora_f4_appearance
```

Watch, in order of importance:

1. `*_rel_swapped` on held-out views - does the true date now beat a wrong-era one?
2. the renders, not the loss (`docs/findings.md`: the probe loss improved while
   the renders got worse);
3. `x0_mse_high_noise` - the guard that caught the earlier collapses;
4. the held-out gains, which must not regress.

## New flags

| Flag | Default | What |
|---|---|---|
| `--appearance-injection film\|none` | `none` | per-block scale-and-shift from the view's appearance |
| `--appearance-dim` | 256 | width of the shared appearance features |
| `--appearance-dropout` | 0.1 | share of samples with appearance marked unknown |
| `--lr-appearance` | 1e-5 | same order as `--lr-film`; the per-block layers act like LoRA |

Off by default, so every existing command behaves exactly as before.

## What changed, and why it is safe

| File | Change |
|---|---|
| `model/appearance_film.py` | new: encoder + per-block FiLM, mirroring `date_film.py` |
| `data/appearance.py` | new: per-view statistics from the pixels the model sees |
| `data/dataset.py` | `with_appearance=True` adds an `appearance` (V, F) field to each sample |
| `model/dated_seva.py` | appearance stored, encoded per forward, cleared like dates |
| `model/finetune.py` | `appearance_injection`, an `"appearance"` parameter group, checkpointing |
| `training/loss.py` | appearance threaded through `predict_noise`; whole-sample dropout |
| `training/trainer.py` | `lr_appearance`, dataset built `with_appearance`, encoder standardized |

- **Zero-init**, so step 0 is exactly SEVA whatever the flags say. There is a test.
- **It nests.** Appearance FiLM wraps the existing `DateFiLMConv` rather than
  replacing it; `date_film.py` and SEVA are untouched.
- **Date-only checkpoints still load** (the `appearance` section is optional).

### Two things worth knowing

**A bug this nearly shipped with.** The first version cleared
`appearance_state.features` when the context manager exited. That is wrong, and
silently so: under gradient checkpointing the blocks are recomputed during
**backward**, after the context has exited, so the conditioning would have
vanished from the backward pass - no crash, just a model that mysteriously would
not learn. `date_film.py` already documents the right contract ("the features
stay stored until the next forward pass"), and appearance now follows it. There
is a test, and it was verified separately with checkpointing enabled.

**An error message worth copying back into `date_film.py`.** Calling
`add_appearance_film` on a model that already has it raises "Appearance FiLM is
already installed on all N ResBlocks". `add_date_film` in the same situation
reports "No SEVA ResBlock found", which sends you looking in the wrong place.

## Testing

Four files, 35 tests with SEVA present and 20 without:

- `test_appearance_data.py`, `test_appearance_film.py`, `test_dataset_appearance.py`
  need no SEVA and run anywhere.
- `test_appearance_training.py` needs SEVA's real `ResBlock` and skips without
  it, as elsewhere in this suite. Only modules that do not pull in SEVA are
  imported above its `importorskip`, so it skips cleanly rather than erroring.

The SEVA-gated tests were developed and verified against a minimal stand-in
`seva` package reproducing the shapes and class names the model code touches,
so the wiring is exercised end to end on CPU.
