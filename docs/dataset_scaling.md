# How many scenes can we afford, and what would they give us?

## The short answer

**All 140 scenes that pass the posed gate fit in 8.3 GB**, capping modern photos at
300 per scene, and yield **5,787 posed pre-1970 photographs**. Storage is not the
constraint anyone assumed it was: the whole usable MegaScenes subset is smaller
than SEVA's weights.

For comparison, from `docs/census/census_scenes_top800.csv`:

| scenes | old posed photos | approx. GB |
|---|---|---|
| 4 (current single-person set) | 1,533 | 0.5 |
| 13 (current training set) | 2,458 | 1.1 |
| 30 | 3,489 | 2.3 |
| 140 (everything gated) | 5,787 | 8.3 |

## Why this matters

Old, posed photographs are the scarce resource in everything we measure. The
shortcut probe and the appearance-entanglement analysis both ran with **58 old
photos in the test split**, which is why their confidence intervals are as wide as
they are. Scene count is the only lever on that, and it turns out to be cheap.

## Running it

```bash
python scripts/plan_dataset.py --budget-gb 50 --have data/datasets/*_model0
python scripts/plan_dataset.py --n-scenes 30
```

Writes `plan.md` (the table below), `plan.csv`, and `scenes.txt`, which feeds the
existing pipeline directly:

```bash
python scripts/process_scenes.py --scene-file data/experiments/dataset_plan/scenes.txt
for s in $(cat data/experiments/dataset_plan/scenes.txt); do
    python scripts/build_scene_dataset.py --scene "$s" --max-modern 300
done
```

`--have` does two things: excludes what is already built, and **measures the real
bytes-per-image from it** instead of guessing. A wrong size estimate is how a quota
fills up halfway through a download.

## The uncomfortable part

Old photographs are not spread evenly over the world. Across the 140 gated scenes:

- the top 20 scenes hold **51%** of all old posed photos;
- **eight Amsterdam scenes alone hold 33%** (Dam Square, Nationaal Monument, Royal
  Palace, Munttoren, Westerkerk, Nieuwe Kerk, Montelbaanstoren, Sint-Nicolaaskerk).

So in anything built from this census, **"old" and "Amsterdam" are correlated**.

That is the same shape of problem as the one we already measured for appearance:
a signal that looks like it means one thing and partly means another. It bears
directly on the proposed **location (lat/long) conditioning** - on a set of a few
dozen scenes a coordinate is close to a scene identifier, and the model could
learn "Amsterdam means old" rather than anything geographic.

`plan_dataset.py` reports the concentration (top-k shares and a Gini coefficient)
of every plan, so this can be judged before the download rather than after the
training run. It does not argue against location conditioning; it argues for
measuring it the way we measured appearance, with held-out *scenes* rather than
held-out views as the test.

## Choosing a plan

Ranking by old-photo yield maximises the scarce resource but concentrates it:
30 scenes gives Gini 0.39 with the top 5 holding 47%. A flatter plan - capping how
many scenes come from one city, or ranking by a blend of yield and diversity -
would trade total old photos for a less confounded dataset. Which trade is right
depends on whether the next experiment is about *era* or about *place*, so the
planner reports the numbers and leaves the choice explicit.
