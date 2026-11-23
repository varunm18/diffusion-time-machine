# Can appearance be conditioned on without handing over the date?

## Why this exists

`timemachine.model.appearance_film` conditions the model on a per-view appearance
vector so lighting, weather and medium have somewhere to live other than the date.
But those 14 statistics were chosen **because** they predict era - AUC 0.89 across
four scenes (`docs/findings.md`). That creates a hazard the original design
glossed over:

> Handing the model a near-perfect era signal through the appearance channel may
> teach it to ignore the date *more*, not less.

At training time the target's own appearance is given, so the model can render the
right look from appearance alone and never consult the date. That would make
date-blindness worse. This measures whether the risk is real before GPU hours go
into Phase C.

## The measurement

Era directions are removed from appearance space one at a time, **refitting a
fresh classifier after each removal**, until the date can no longer be read.
Refitting matters: a second, weaker era direction routinely survives the first
removal, and assuming otherwise would report success that is not there.

The headline is **how many of the 14 dimensions must be destroyed**, and how much
of appearance is left:

- **few dimensions** - an era-free appearance vector exists; condition on the
  residual and the factorization is clean;
- **many** - appearance and era are nearly the same thing in these statistics, and
  the pathway cannot be made era-free this way.

Then, with independent lighting labels, the other half: does the residual still
carry something we want the model to see?

| | era gone | era survives |
|---|---|---|
| **lighting kept** | Separable - condition on the residual | Removal failed; era is spread everywhere |
| **lighting lost** | Entangled - the look *is* the era here | Removal failed |

## Running it

```bash
# 1. Make a labelling sheet (thumbnails embedded; needs no network, works on the cluster)
python scripts/appearance_entanglement.py --dataset data/datasets/*_model0 \
    --make-labels --n-labels 150

# 2. Label day vs night in a browser, press "Download lighting_labels.csv"

# 3. Analyse
python scripts/appearance_entanglement.py --dataset data/datasets/*_model0 \
    --labels ~/Downloads/lighting_labels.csv
```

CPU only. Without `--labels` it still reports whether era can be removed, but not
whether anything useful survives - which is the half that decides the design.

## Two things done deliberately

**No circularity.** The lighting labels come from a human, never from the
statistics. Defining "night" as "low `lum_mean`" and then reporting that `lum_mean`
predicts night would measure nothing. The sheet shows the image and nothing else -
no dates, no statistics - because a labeller who could see the date would produce
labels contaminated by the very thing being tested against.

**The sheet is era-stratified.** Our view tables are ~75% modern, so a plain random
draw would be almost all recent photos - and if nearly every night photo on it were
also modern, "night" and "modern" would become the same label and the result would
be meaningless. Spreading across eras breaks that confound in the draw itself,
before any labelling happens.

## Sample size for the lighting half

Night photographs of famous landmarks are rare: a 150-image sheet across four
scenes yielded roughly a dozen. Two consequences, both learned the hard way:

- **The lighting AUC is cross-validated over the labelled rows**, not measured on a
  global hold-out. A single 25% split left *one* night photo in test, and an AUC
  over one positive is that photo's rank, not a measurement.
- **The report refuses to conclude** when either class is under
  `MIN_LABELS_PER_CLASS` (12). It says so instead of printing a confident number.

If night photos are still too few, enrich the sheet with candidates found from
**EXIF capture times**, which the data pipeline already extracts and which are
independent of these pixel statistics - so the sampling is enriched without the
measurement becoming circular. Picking candidates by brightness would *not* be
safe: brightness is one of the 14 features being tested.

## Reading the verdict

The report ends with one of four statements, because there are four cases and an
earlier draft of this analysis collapsed them into two and reported "lighting does
not survive" when lighting had survived perfectly and it was *era* that remained:

- **Separable** - condition Phase C on the residual.
- **Entangled** - lighting was readable before and is not after; the appearance
  pathway carries the date unavoidably.
- **Era could not be removed** - era is spread across the whole space.
- **Inconclusive** - lighting was not readable even before removal, so its loss
  says nothing; the labels or the features are at fault.

## What to do with the answer

- **Separable:** add a residualization buffer to `AppearanceEncoder` and run Phase C
  on era-free features. The factorization becomes real rather than hoped for.
- **Entangled or not removable:** Phase C still runs, but its result must be read
  knowing the appearance pathway carries the date. A null result would then be
  evidence about the *data* - too few genuinely informative old/modern pairs of the
  same place - rather than about the architecture.

Either way this is worth knowing before a multi-day training run, not after.
