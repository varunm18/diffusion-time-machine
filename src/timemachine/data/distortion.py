"""Backfill lens-distortion columns into view tables built before they existed.

``DatedViewsDataset`` needs ``k1``/``k2`` per view: ``load_view`` undistorts the
crop with them, and **raises** on a row where they are not finite. View tables
written before those columns were added have neither, so every such dataset is
rejected outright - the images are fine, only the table is stale.

A row whose coefficients cannot be used - no match in the poses table, or a
radial map that folds over the camera's own frame - is **dropped from the table**,
never written as non-finite. ``load_view`` does not skip such a row; it raises,
which would kill a training run at whichever step first draws that view. This
matches the rest of the repo, which filters them too (``bad_distortion`` in
``scripts/mvgen/build_dataset.py``, ``loadable_views`` in
``scripts/mvgen/prepare_depth.py``). :func:`usable_distortion` is the one
definition of a usable row, shared with the scene build
(:func:`timemachine.data.views.build_view_table`), so a rebuild cannot bring the
rows back either.

Nothing needs re-downloading. The coefficients were always in the COLMAP
``cameras.bin`` of each reconstruction, which is cached locally, and the current
:func:`timemachine.megascenes.poses.load_model` already reads them out. So this is
a recompute and a join, not a rebuild.

Why not simply write zeros: MegaScenes reconstructions are overwhelmingly
``SIMPLE_RADIAL``, which means a real non-zero ``k1``. Zeros there do not fail
loudly - they skip undistortion and leave a radial error that grows with r^2, worst
at the frame edges. That error is also *correlated with era*, because older lenses
distort differently from modern ones, so it would quietly hand a date-conditioned
model one more way to read the year off the photograph rather than the scene. That
is the exact confound the shortcut and entanglement analyses exist to remove, so
guessing zeros here would undermine the experiment it is meant to unblock.
``PINHOLE`` and ``SIMPLE_PINHOLE`` cameras are the one exception: they carry no
distortion parameters at all, so zero is their true value rather than a guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Camera models with no distortion parameters: zero is exact, not an approximation.
PINHOLE_MODELS = frozenset({"PINHOLE", "SIMPLE_PINHOLE"})

# Poses this project generated itself, rather than reading from a reconstruction.
# VGGT-Omega predicts pinhole cameras, so these are genuinely distortion-free.
PREDICTED_POSE_SOURCES = frozenset({"vggt_omega"})

DISTORTION_COLUMNS = ("k1", "k2")

# Columns needed to work out how far from the optical axis a camera actually sees,
# which is what decides whether its distortion can be inverted at all.
GEOMETRY_COLUMNS = ("fx", "fy", "cx", "cy", "camera_width", "camera_height")

# Written beside a migrated table so the original is recoverable.
BACKUP_SUFFIX = ".pre_distortion.parquet"

# Written beside a table by prune_folder. A separate name, because the migration's
# backup already exists for every migrated table and is never overwritten.
PRUNE_BACKUP_SUFFIX = ".pre_prune.parquet"


def max_radius_squared(views: pd.DataFrame) -> np.ndarray:
    """Per row, the squared normalized radius of the furthest corner the camera sees.

    ``undistorted_crop`` applies the radial polynomial in normalized camera
    coordinates, so the relevant radius is not in pixels: it is how far the frame
    corner sits from the optical axis once divided by the focal length. A long lens
    sees a narrow cone and reaches a small radius; a wide one reaches much further,
    which is why the same ``k1`` is harmless on one camera and ruinous on another.
    """
    fx, fy = views["fx"].to_numpy(dtype=np.float64), views["fy"].to_numpy(dtype=np.float64)
    cx, cy = views["cx"].to_numpy(dtype=np.float64), views["cy"].to_numpy(dtype=np.float64)
    width = views["camera_width"].to_numpy(dtype=np.float64)
    height = views["camera_height"].to_numpy(dtype=np.float64)
    # The principal point is rarely centred, so take whichever corner is furthest.
    dx = np.maximum(np.abs(cx), np.abs(width - cx)) / np.where(fx > 0, fx, np.nan)
    dy = np.maximum(np.abs(cy), np.abs(height - cy)) / np.where(fy > 0, fy, np.nan)
    return dx ** 2 + dy ** 2


def is_invertible(k1: np.ndarray, k2: np.ndarray, r2_max: np.ndarray) -> np.ndarray:
    """Whether the radial map stays monotonic out to the frame corner.

    The forward map is ``r_d = r (1 + k1 r^2 + k2 r^4)``, so it is one-to-one only
    while its derivative ``1 + 3 k1 r^2 + 5 k2 r^4`` stays positive. Past the point
    where that turns negative the map folds: several scene rays arrive at the same
    pixel, and no undistortion can separate them again.

    This matters because COLMAP will happily fit a wild coefficient to a camera with
    few observations, and MegaScenes is full of such cameras. Undistorting with a
    fitted-but-wrong ``k1`` is worse than leaving the image alone - it warps a good
    photograph on the strength of a number nothing constrained. Judging each camera
    against its own reach is better than one global threshold, because the bound
    genuinely differs: a 0.3 that is fine on a telephoto is not fine on a wide angle.
    """
    derivative = 1.0 + 3.0 * k1 * r2_max + 5.0 * k2 * r2_max ** 2
    return np.isfinite(derivative) & (derivative > 0)


def finite_distortion(views: pd.DataFrame) -> np.ndarray:
    """``[N]`` bool: ``k1`` and ``k2`` present and finite - the rows ``load_view`` accepts.

    ``load_view`` raises on anything else, so a view table must contain only these.
    A table missing either column has no loadable row at all.
    """
    if any(c not in views.columns for c in DISTORTION_COLUMNS):
        return np.zeros(len(views), dtype=bool)
    return np.isfinite(views[list(DISTORTION_COLUMNS)].to_numpy(dtype=np.float64)).all(axis=1)


def usable_distortion(views: pd.DataFrame) -> np.ndarray:
    """``[N]`` bool: finite coefficients whose radial map is invertible over the camera's frame.

    Stricter than :func:`finite_distortion`. A finite but uninvertible coefficient
    does not make ``load_view`` raise - it makes it warp a good photograph on the
    strength of a number COLMAP fitted to a weakly-observed camera. Needs the
    :data:`GEOMETRY_COLUMNS`, since invertibility depends on how far the frame reaches.
    """
    missing = [c for c in GEOMETRY_COLUMNS if c not in views.columns]
    if missing:
        raise KeyError(f"cannot judge invertibility without {missing}")
    ok = finite_distortion(views)
    k = views[list(DISTORTION_COLUMNS)].to_numpy(dtype=np.float64)
    return ok & is_invertible(k[:, 0], k[:, 1], max_radius_squared(views))


@dataclass
class Migration:
    """What happened to one dataset's view table."""

    folder: Path
    rows: int = 0
    matched: int = 0
    pinhole: int = 0
    predicted: int = 0
    nonzero_k1: int = 0
    implausible: int = 0
    dropped: int = 0               # rows removed from the written table (uninvertible + unmatched)
    k1_median: float = 0.0
    k1_max: float = 0.0
    written: bool = False
    dry_run: bool = False
    note: str = ""

    @property
    def unmatched(self) -> int:
        return self.rows - self.matched

    @property
    def ok(self) -> bool:
        """Every row ended up with a usable answer, rejections included.

        A row dropped for having an uninvertible coefficient has been handled
        correctly - it is removed from the table - so it does not make the
        migration a failure. Only a row the poses table never covered does.
        """
        return self.rows > 0 and self.unmatched == self.implausible

    def line(self) -> str:
        if self.note and not self.rows:
            return f"  skip  {self.folder.name:<22} {self.note}"
        mark = "  ok" if self.ok else "FAIL"
        bits = [f"{self.matched}/{self.rows} matched"]
        if self.nonzero_k1:
            bits.append(f"{self.nonzero_k1} distorted "
                        f"(median |k1| {self.k1_median:.3f}, max {self.k1_max:.3f})")
        if self.implausible:
            bits.append(f"{self.implausible} not invertible, dropped")
        if self.dropped > self.implausible:
            bits.append(f"{self.dropped - self.implausible} unmatched, dropped")
        if self.pinhole:
            bits.append(f"{self.pinhole} pinhole")
        if self.predicted:
            bits.append(f"{self.predicted} predicted")
        if self.note:
            bits.append(self.note)
        line = f"  {mark}  {self.folder.name:<22} {', '.join(bits)}"
        return line + " (dry run)" if self.dry_run else line


def parse_dataset_folder(folder: Path) -> tuple[int, int] | None:
    """``.../113197_model0`` -> ``(113197, 0)``; None for anything else."""
    match = re.fullmatch(r"(\d+)_model(\d+)", folder.name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def needs_migration(views: pd.DataFrame) -> bool:
    """A table is stale if it lacks either column, or carries them all non-finite."""
    if any(c not in views.columns for c in DISTORTION_COLUMNS):
        return True
    finite = np.isfinite(views[list(DISTORTION_COLUMNS)].to_numpy(dtype=np.float64))
    return not finite.all(axis=1).any()


def attach_distortion(views: pd.DataFrame, poses: pd.DataFrame, *,
                      drop_uninvertible: bool = True) -> tuple[pd.DataFrame, Migration]:
    """Join ``k1``/``k2`` onto a view table by ``file_key``.

    Rows that end up without usable coefficients - the poses table does not cover
    them, or (``drop_uninvertible``) their radial map folds over the frame - are
    **dropped** from the returned table, and counted in the result. They are never
    zeroed (that trains on a silently wrong crop) and never written as non-finite
    (``load_view`` raises on those). So the returned table is always loadable.
    The counts keep a partial match visible: ``unmatched`` and ``implausible``
    still describe the input rows.
    """
    result = Migration(folder=Path("."), rows=len(views))
    if "file_key" not in views.columns:
        result.note = "view table has no file_key to join on"
        return views, result

    wanted = ["file_key", *DISTORTION_COLUMNS]
    have = [c for c in wanted if c in poses.columns]
    if len(have) < len(wanted):
        result.note = f"poses table lacks {set(wanted) - set(have)}; re-derive it first"
        return views, result

    # One row per file_key: a file registered twice in a model has the same camera.
    lookup = poses[wanted].drop_duplicates("file_key").set_index("file_key")
    out = views.drop(columns=[c for c in DISTORTION_COLUMNS if c in views.columns])
    joined = out.join(lookup, on="file_key")

    # Rows whose camera genuinely has no distortion parameters: fill them exactly
    # rather than leaving them to be skipped as unknown.
    exact = pd.Series(False, index=joined.index)
    if "camera_model" in joined.columns:
        exact |= joined["camera_model"].isin(PINHOLE_MODELS)
    if "pose_source" in joined.columns:
        exact |= joined["pose_source"].isin(PREDICTED_POSE_SOURCES)
    for column in DISTORTION_COLUMNS:
        joined.loc[exact & joined[column].isna(), column] = 0.0

    # copy=True matters: a frame read back from parquet can hand out a read-only
    # array, and the rejection step below writes into it.
    values = joined[list(DISTORTION_COLUMNS)].to_numpy(dtype=np.float64, copy=True)

    # Reject coefficients that cannot be inverted over the frame this camera sees.
    # Writing them would warp the view; the rows are dropped below.
    if drop_uninvertible and all(c in joined.columns for c in GEOMETRY_COLUMNS):
        r2_max = max_radius_squared(joined)
        known = np.isfinite(values).all(axis=1)
        bad = known & ~is_invertible(values[:, 0], values[:, 1], r2_max)
        result.implausible = int(bad.sum())
        if result.implausible:
            values[bad] = np.nan
            for i, column in enumerate(DISTORTION_COLUMNS):
                joined[column] = values[:, i]

    finite = np.isfinite(values).all(axis=1)
    result.matched = int(finite.sum())
    predicted = (joined["pose_source"].isin(PREDICTED_POSE_SOURCES)
                 if "pose_source" in joined.columns else pd.Series(False, index=joined.index))
    result.predicted = int(predicted.sum())
    # Pinhole rows filled from their camera model, excluding ones already counted as
    # predicted, so the two numbers in the report sum to the rows they describe.
    result.pinhole = int((exact & ~predicted & ~joined["file_key"].isin(lookup.index)).sum())
    if result.matched:
        k1 = np.abs(values[finite, 0])
        result.nonzero_k1 = int((k1 != 0).sum())
        result.k1_median = float(np.median(k1))
        result.k1_max = float(k1.max())
    # Last, after every count above has seen the full table: keep only rows load_view can load.
    result.dropped = int((~finite).sum())
    return _order_columns(joined.loc[finite].reset_index(drop=True)), result


def _order_columns(views: pd.DataFrame) -> pd.DataFrame:
    """Put ``k1``/``k2`` back where the canonical view schema has them: after ``cy``.

    Purely cosmetic - nothing reads by position - but it keeps a migrated table
    diff-comparable against a freshly built one.
    """
    columns = [c for c in views.columns if c not in DISTORTION_COLUMNS]
    if "cy" not in columns:
        return views[columns + list(DISTORTION_COLUMNS)]
    at = columns.index("cy") + 1
    return views[columns[:at] + list(DISTORTION_COLUMNS) + columns[at:]]


def view_table(folder: Path) -> Path | None:
    """The view table inside a dataset folder, preferring the cleaned one."""
    return next((folder / name for name in ("views_clean.parquet", "views.parquet")
                 if (folder / name).is_file()), None)


def migrate_folder(folder: Path, poses: pd.DataFrame, *, dry_run: bool = False,
                   backup: bool = True, allow_partial: bool = False,
                   drop_uninvertible: bool = True) -> Migration:
    """Migrate one dataset folder in place, keeping the original beside it.

    Refuses a partial write by default. A table where some rows matched and others
    did not is worth stopping for: forcing it (``allow_partial``) drops the
    unmatched views, and how many is easy to miss once the table is written.
    """
    table = view_table(folder)
    if table is None:
        return Migration(folder=folder, note="no views parquet")
    views = pd.read_parquet(table)
    if not needs_migration(views):
        return Migration(folder=folder, rows=len(views), matched=len(views),
                         note="already has finite k1/k2")

    migrated, result = attach_distortion(views, poses,
                                        drop_uninvertible=drop_uninvertible)
    result.folder = folder
    if result.note and not result.matched:
        return result
    # Rows rejected as uninvertible are an intended outcome, not a half-finished
    # migration, so they do not need --allow-partial to be written.
    deliberate = bool(result.implausible) and result.unmatched == result.implausible
    if dry_run:
        result.dry_run = True
        return result
    if result.rows - result.dropped == 0:
        result.note = "no row has usable coefficients; not writing an empty table"
        return result
    if deliberate:
        return _write(folder, table, views, migrated, result, backup)
    if not result.ok and not allow_partial:
        result.note = (f"{result.unmatched} rows unmatched; pass --allow-partial to drop "
                       "those views and write the rest")
        return result
    return _write(folder, table, views, migrated, result, backup)


def _write(folder: Path, table: Path, original: pd.DataFrame, migrated: pd.DataFrame,
           result: Migration, backup: bool) -> Migration:
    if backup:
        spare = table.with_name(table.stem + BACKUP_SUFFIX)
        if not spare.exists():                    # never clobber an earlier backup
            original.to_parquet(spare, index=False)
    migrated.to_parquet(table, index=False)
    result.written = True
    return result


def render_report(results: list[Migration]) -> str:
    """One line per dataset, then the totals and what to do next."""
    lines = [r.line() for r in results]
    live = [r for r in results if r.rows]
    rows = sum(r.rows for r in live)
    matched = sum(r.matched for r in live)
    distorted = sum(r.nonzero_k1 for r in live)
    dropped = sum(r.dropped for r in live)
    written = [r for r in results if r.written]
    failed = [r for r in live if not r.ok and not r.written]

    lines += ["", f"{len(live)} dataset(s), {rows:,} views, {matched:,} with distortion "
                  f"coefficients ({distorted:,} genuinely distorted)"]
    implausible = sum(r.implausible for r in live)
    if implausible:
        lines.append(f"{implausible:,} view(s) had coefficients that cannot be inverted over "
                     "their own frame and were dropped from the table")
    if dropped > implausible:
        lines.append(f"{dropped - implausible:,} unmatched view(s) were dropped from the table")
    if written:
        lines.append(f"rewrote {len(written)} view table(s); originals kept as "
                     f"*{BACKUP_SUFFIX}")
    if failed:
        lines += ["", f"{len(failed)} dataset(s) not migrated:"]
        lines += [f"  - {r.folder.name}: {r.note or 'unmatched rows'}" for r in failed]
    return "\n".join(lines)


# --------------------------------------------------------------------------- pruning

@dataclass
class Pruning:
    """What happened to one view table when its unloadable rows were removed."""

    table: Path
    rows: int = 0
    dropped_keys: list[str] = field(default_factory=list)
    written: bool = False
    dry_run: bool = False
    note: str = ""

    @property
    def dropped(self) -> int:
        return len(self.dropped_keys)

    @property
    def ok(self) -> bool:
        return not self.note

    def line(self) -> str:
        name = f"{self.table.parent.name}/{self.table.name}"
        if self.note:
            return f"FAIL  {name:<36} {self.note}"
        if not self.dropped:
            return f"  ok  {name:<36} {self.rows} rows, nothing to drop"
        line = f"  ok  {name:<36} dropped {self.dropped} of {self.rows} rows"
        return line + " (dry run)" if self.dry_run else line


def prune_table(table: Path, *, dry_run: bool = False, backup: bool = True) -> Pruning:
    """Remove the rows of one view table whose ``k1``/``k2`` are not finite.

    These are left over from the first distortion migration, which wrote rejected
    coefficients as NaN on the mistaken belief that ``load_view`` skips them; it
    raises. The original is kept beside the table as ``*.pre_prune.parquet`` (an
    existing backup is never overwritten), so this is reversible.

    Refuses a table in which *no* row is finite, or that lacks the columns: that is a
    stale table that needs ``scripts/migrate_distortion.py``, and pruning it would
    leave nothing.
    """
    views = pd.read_parquet(table)
    result = Pruning(table=table, rows=len(views))
    if any(c not in views.columns for c in DISTORTION_COLUMNS):
        result.note = "no k1/k2 columns; run scripts/migrate_distortion.py first"
        return result
    keep = finite_distortion(views)
    if not keep.any():
        result.note = "no row has finite k1/k2; this table needs migrating, not pruning"
        return result
    result.dropped_keys = views.loc[~keep, "file_key"].astype(str).tolist() \
        if "file_key" in views.columns else [str(i) for i in np.flatnonzero(~keep)]
    if not result.dropped or dry_run:
        result.dry_run = dry_run
        return result
    if backup:
        spare = table.with_name(table.stem + PRUNE_BACKUP_SUFFIX)
        if not spare.exists():                    # never clobber an earlier backup
            views.to_parquet(spare, index=False)
    views.loc[keep].reset_index(drop=True).to_parquet(table, index=False)
    result.written = True
    return result


def prune_folder(folder: Path, *, dry_run: bool = False, backup: bool = True) -> list[Pruning]:
    """:func:`prune_table` on every view table in a dataset folder.

    Both names, not only the one :func:`view_table` prefers: the trainer reads
    ``views.parquet`` while the migration and preflight prefer ``views_clean.parquet``,
    so a folder holding both must have both cleaned.
    """
    tables = [folder / name for name in ("views_clean.parquet", "views.parquet")
              if (folder / name).is_file()]
    if not tables:
        return [Pruning(table=folder / "views.parquet", note="no views parquet")]
    return [prune_table(t, dry_run=dry_run, backup=backup) for t in tables]


def render_prune_report(results: list[Pruning]) -> str:
    lines = [r.line() for r in results]
    rows = sum(r.rows for r in results)
    dropped = sum(r.dropped for r in results)
    written = [r for r in results if r.written]
    failed = [r for r in results if not r.ok]
    lines += ["", f"{len(results)} table(s), {rows:,} rows, {dropped:,} with non-finite k1/k2"]
    if written:
        lines.append(f"rewrote {len(written)} table(s), {rows - dropped:,} rows remain; originals "
                     f"kept as *{PRUNE_BACKUP_SUFFIX}")
    if failed:
        lines.append(f"{len(failed)} table(s) not pruned (see FAIL lines)")
    return "\n".join(lines)
