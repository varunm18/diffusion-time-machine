# Migrating view tables to carry lens distortion

## The problem

`DatedViewsDataset` requires `k1`/`k2` per view. `load_view` undistorts the crop
with them and **raises** on any row where they are not finite, so a view table
written before those columns existed is rejected outright:

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

Every other row without a usable coefficient is **dropped from the table** - not
zeroed, and not written as non-finite. Unknown and zero are different claims, and
the table should not conflate them; and a non-finite row is not "unknown" to
`load_view` but an error (see the correction below).

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

Rejected rows are **dropped from the table** at migration time. They are reported,
but do not count as a failed migration - they have been handled correctly - and they
stay recoverable from `views.pre_distortion.parquet`. `--keep-uninvertible` writes
them, coefficients and all, instead.

## Correction: rejected rows are dropped, not left non-finite

The first version of this migration (commit `dc10d60`) wrote rejected rows as NaN,
and this page said `load_view` would skip them. **It does not.** `load_view`
(`src/timemachine/data/dataset.py`) raises on a non-finite `k1`/`k2`:

```
ValueError: <file_key>: unsupported lens model (no radial coefficients)
```

So the 49 rows that migration wrote, across 21 of the 34 tables, were 49 chances to
kill a training run at whichever step first drew one - hours in, on a 3-day queue.
The preflight did not catch it: it checked the columns existed, not their values,
and reported all 34 tables healthy. Nothing caught it until a test read the tables.

The rest of the repo already filtered these rows: `bad_distortion` in
`scripts/mvgen/build_dataset.py`, `loadable_views` in `scripts/mvgen/prepare_depth.py`.
Now this pipeline does too, everywhere a row can enter or be read from a table:

| Where | What happens to a row without usable coefficients |
| --- | --- |
| `migrate_distortion.py` (`attach_distortion`) | dropped; counted in the report |
| `build_scene_dataset.py` (`build_view_table`) | dropped before download, listed in the output - **the second gap** |
| `prune_unloadable_views.py` | removes the ones the first migration left behind |
| `DatedViewsDataset(...)` | refuses the table at construction, naming the views |
| `preflight.py` | fails that dataset, with the prune command as its fix |
| `tests/test_view_tables.py` | asserts no real table holds one |

The scene build was a real gap, not a theoretical one. The migration rewrote the
**poses** tables with the raw `cameras.bin` coefficients - deliberately, so future
builds would have the columns - and `build_view_table` applied no distortion filter
at all. A rebuild would have brought all 49 views back with their finite but
uninvertible `k1`, and `load_view` raises only on non-finite values, so it would
have undistorted each one through a map that folds: silently, with no error. The
build now applies the same `usable_distortion` rule as the migration, so a rebuilt
table and a migrated one contain the same views.

### Cleaning tables written by the first migration

```bash
python scripts/prune_unloadable_views.py --dataset data/datasets/* --dry-run --list
python scripts/prune_unloadable_views.py --dataset data/datasets/*
```

Local only; reads and rewrites the view tables and nothing else. Each original is
kept as `views.pre_prune.parquet` - a separate name, because the migration's
`views.pre_distortion.parquet` already exists and is never overwritten. Re-running it
on a clean table does nothing.

### What the 49 views were, and why zero is not the answer for them either

Looking at them settles whether dropping is right, independent of the crash:

* All 49 have **negative** `k1` (barrel), median `|k1|` 0.23, up to 0.77.
* Their frames reach a median `r² = 1.86` in normalized coordinates - an enormous
  field of view, far beyond any lens a two-term radial model describes. At the
  corner the map's derivative is negative for every one (median -0.34).
* Aspect ratios run up to 8.3:1 (median 1.5, against 1.33 for all views).
* By name they are panoramas (`Venice_pano1`, `Dam Square 360° Panorama`,
  `Panorama of Piazza San Pietro`), fisheye shots (`рибље око` - "fish-eye"),
  multi-image postcards (`mehrteilig`, `9-teilig`, `Mehrbild`) and photo collages.

None of these is one photograph from one camera. COLMAP registered each as a single
radial camera and fitted whatever coefficient minimised the error; no coefficient
makes such an image a pinhole view. That is why **zeros are not an acceptable
substitute**:

* `k1 = 0` is a claim - "this camera has no distortion" - and here it is known to be
  false. `load_view` would accept it and feed the model a stitched or fisheye frame
  as if it were a rectilinear photo, with intrinsics that describe neither.
* The error would not be noise. Collages and multi-view postcards are a genre with
  an era, and a geometric error that tracks era is exactly the shortcut this project
  measures and removes elsewhere (the section above). It would arrive through the
  camera pathway, which nothing is measuring.
* It would also defeat every check above: a zero is finite, so the dataset, the
  preflight and the test would all pass it.

The cost of dropping is small: 49 of 12,223 views (0.4%), and 6 of the 1,751 views
before 1945 (0.3%), so the old-photo pool the experiment depends on is untouched in
any meaningful sense.

## Partial migrations are refused

A table where some rows matched and others did not is worth stopping for: forcing
it drops every unmatched view, and how many is easy to miss once the table is
written. So a partial result is reported and **not written**:

```
  FAIL  209_model0   140/180 matched, 40 rows unmatched; pass --allow-partial ...
```

`--allow-partial` writes it anyway, **dropping** the unmatched views (it never
writes them as non-finite, and never writes an empty table). Worth understanding why
rows go unmatched before reaching for it — usually the view table contains files
that the re-derived model no longer registers.

## Reading the report

```
    ok  113197_model0   354/357 matched, 354 distorted (median |k1| 0.070, max 0.561), 3 not invertible, dropped
    ok  888_model0      30/30 matched, 30 pinhole, 30 predicted
    ok  999_model0      50/50 matched, already has finite k1/k2

3 dataset(s), 1,403 views, 1,389 with distortion coefficients (1,389 genuinely distorted)
14 view(s) had coefficients that cannot be inverted over their own frame and were dropped from the table
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

To undo only the pruning, keeping the migration:

```bash
mv views.pre_prune.parquet views.parquet
```

Note that restoring either backup brings back rows `load_view` cannot load; the
preflight and `DatedViewsDataset` will refuse the table until it is pruned again.
