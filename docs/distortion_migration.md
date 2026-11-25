# Migrating view tables to carry lens distortion

## The problem

`DatedViewsDataset` requires `k1`/`k2` per view. `load_view` undistorts the crop
with them and refuses any row where they are not finite, so a view table written
before those columns existed is rejected outright:

```
ValueError: View table is missing columns: ['k1', 'k2']
```

Every dataset built before the distortion columns landed is in this state. The
images are fine; only the small tables are stale. The preflight reports it as
`missing columns: k1, k2`.

## The fix

```bash
python scripts/migrate_distortion.py --dataset data/datasets/* --dry-run   # look first
python scripts/migrate_distortion.py --dataset data/datasets/*             # then write
```

Nothing is re-downloaded. The coefficients have been in each reconstruction's
`cameras.bin` all along, and the current `megascenes.poses.load_model` already
reads them out via `cam.radial_distortion()`. The script re-derives each scene's
poses table from the cached reconstruction, then joins `k1`/`k2` onto the view
table by `file_key`.

Both the poses tables and the view tables are rewritten, and both originals are
kept beside them as `*.pre_distortion.parquet`. The poses tables matter because
`build_view_table` reads them, so leaving them stale would mean every future
`build_scene_dataset.py` run produced another table needing this migration.
`--no-rewrite-poses` skips that if you want only the view tables touched.

## Why zeros are not an acceptable shortcut

Writing `k1 = k2 = 0` everywhere would be a one-line fix and it is the wrong one.

MegaScenes reconstructions are overwhelmingly `SIMPLE_RADIAL`, which means a real
non-zero `k1`. Zeros there do not fail loudly: they skip undistortion and leave a
radial error that is near zero at the frame centre and grows with r², worst at the
edges.

That error is also **correlated with era**. Older lenses distort differently from
modern ones, so an uncorrected radial error varies systematically with the date of
the photograph. This project conditions on camera geometry through Plücker rays and
is trying to establish whether a model reads the date from the *scene* or from the
*photograph*. Measurements so far say the photographic medium alone predicts era at
AUC 0.868, and `--appearance-remove-era` exists to strip that shortcut out. Baking
in a geometric error that tracks era would hand the model a fresh one through a
channel nothing is measuring. It would undermine the experiment this migration is
meant to unblock.

So zeros are written in exactly two cases, where they are the true value rather
than a guess:

| Case | Why zero is exact |
| --- | --- |
| `camera_model` in `PINHOLE`, `SIMPLE_PINHOLE` | these COLMAP models carry no distortion parameters at all |
| `pose_source == "vggt_omega"` | VGGT-Omega predicts pinhole cameras by construction |

Every other unmatched row is left **non-finite**, not zeroed, so `load_view`'s own
`np.isfinite` guard skips it. Unknown and zero are different claims, and the table
should not conflate them.

## Rejecting coefficients that cannot be inverted

Having the real coefficient is not the same as having a usable one. Across the 34
scenes measured here, `|k1|` looks like this:

| percentile | `\|k1\|` |
| --- | --- |
| p50 | 0.075 |
| p90 | 0.289 |
| p99 | 0.769 |
| max | 0.988 |

The median is believable for a consumer lens. The tail is not: 9.4% of views
exceed 0.3 and 3.8% exceed 0.5. These are cameras COLMAP saw too few times to
constrain, so it fitted whatever minimised reprojection error. **Undistorting with
a fitted-but-wrong coefficient is worse than leaving the image alone** - it warps a
good photograph on the strength of a number nothing pinned down.

Rather than pick a cutoff on `|k1|`, each row is judged against its own camera. The
forward map is

```
r_d = r (1 + k1 r² + k2 r⁴)
```

which is one-to-one only while its derivative `1 + 3 k1 r² + 5 k2 r⁴` stays
positive. Past that point the map folds: several scene rays arrive at the same
pixel and no undistortion can separate them. `r` here is in *normalized* camera
coordinates, so the radius the frame corner reaches depends on the focal length -
`max_radius_squared` works it out per row from `fx`, `fy`, `cx`, `cy` and the frame
size, taking the furthest corner since the principal point is rarely centred.

This matters because a single global threshold would be wrong in both directions. A
`k1` of -0.9 is perfectly invertible on a telephoto that only reaches `r² = 0.1`,
and ruinous on a wide angle reaching `r² = 0.8`. There is a test for exactly that
asymmetry.

Rejected rows are left **non-finite**, so `load_view`'s own `isfinite` guard skips
those views. They are reported but do not count as a failed migration - they have
been handled correctly. `--keep-uninvertible` writes them anyway.

## Partial migrations are refused

A table where some rows matched and others did not is the worst thing to discover
late: training runs, the loss curve looks normal, and an unknown share of views is
being silently skipped. So a partial result is reported and **not written**:

```
  FAIL  209_model0   140/180 matched, 40 rows unmatched; pass --allow-partial ...
```

`--allow-partial` writes it anyway, with those views marked unknown. Worth
understanding why rows go unmatched before reaching for it — usually the view table
contains files that the re-derived model no longer registers.

## Reading the report

```
    ok  113197_model0   354/357 matched, 354 distorted (median |k1| 0.070, max 0.561), 3 not invertible, left unknown
    ok  888_model0      30/30 matched, 30 pinhole, 30 predicted
    ok  999_model0      50/50 matched, already has finite k1/k2

3 dataset(s), 1,403 views, 1,389 with distortion coefficients (1,389 genuinely distorted)
14 view(s) had coefficients that cannot be inverted over their own frame and were left unknown; load_view skips those
rewrote 3 view table(s); originals kept as *.pre_distortion.parquet
```

`distorted` counts rows whose `k1` is non-zero - the ones where this migration
changed the geometry rather than confirming it. If that number is near zero across
the board, something is wrong: these reconstructions should be mostly radial.

The median is the number to watch, not the maximum. A maximum alone cannot
distinguish three bad cameras from half the data, which is why both are printed.

## Rolling back

```bash
cd data/datasets/113197_model0
mv views.pre_distortion.parquet views.parquet
```

A second migration never overwrites an existing backup, so the file beside a table
is always the pre-migration original rather than an intermediate state.
