#!/usr/bin/env python
"""Remove view-table rows whose lens coefficients are not finite.

    python scripts/prune_unloadable_views.py --dataset data/datasets/* --dry-run
    python scripts/prune_unloadable_views.py --dataset data/datasets/*

The first distortion migration (scripts/migrate_distortion.py) wrote coefficients
it rejected as uninvertible as NaN, believing ``load_view`` skips such rows. It does
not: it raises, so a training run would die at whichever step first drew one of
them. The migration now drops those rows itself; this cleans up the tables it
wrote before that fix. 49 of 12,223 views across the 34 datasets at the time.

Local only: reads and rewrites the view tables, nothing else. Each original is kept
beside its table as ``views.pre_prune.parquet`` (an existing backup is never
overwritten), so this is reversible. Safe to re-run: a clean table is left alone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timemachine.data.distortion import prune_folder, render_prune_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True, metavar="DIR",
                        help="dataset folders to clean (shell globs are fine)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be dropped without writing anything")
    parser.add_argument("--no-backup", action="store_true",
                        help="do not keep the original tables beside the new ones")
    parser.add_argument("--list", action="store_true",
                        help="also print the file_key of every dropped view")
    args = parser.parse_args()

    results = [r for folder in args.dataset if folder.is_dir()
               for r in prune_folder(folder, dry_run=args.dry_run, backup=not args.no_backup)]
    if not results:
        print("No dataset folders given.")
        return 1
    print(render_prune_report(results))
    if args.list:
        for r in results:
            for key in r.dropped_keys:
                print(f"  {r.table.parent.name}\t{key}")
    if args.dry_run:
        print("\nNothing was written (--dry-run). Re-run without it to prune.")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
