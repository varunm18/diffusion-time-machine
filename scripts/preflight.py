#!/usr/bin/env python
"""Check a training setup before queueing it.

    python scripts/preflight.py --dataset data/datasets/*_model0

Exits non-zero if anything would stop the run, so an sbatch can gate on it:

    python scripts/preflight.py --dataset "$DATA"/*_model0 --no-smoke || exit 1

By default the slow check - one real training step - is skipped, because the point
of running this on a login node is to catch the cheap failures before you ever
reach a GPU. Add --smoke on the GPU itself, inside the job or an srun session.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timemachine.training.preflight import MIN_OLD_VIEWS, run_checks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True, metavar="DIR",
                        help="dataset folders to check (shell globs are fine)")
    parser.add_argument("--hub", type=Path, default=None,
                        help="hub cache holding the weights (default: TIMEMACHINE_HF_HUB)")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="where the images live now, if the tables were built elsewhere")
    parser.add_argument("--min-old", type=int, default=MIN_OLD_VIEWS, metavar="N",
                        help=f"views before 1945 needed for the run to be worth it "
                             f"(default {MIN_OLD_VIEWS})")
    # Both spellings exist so a job script can state --no-smoke explicitly rather
    # than relying on the default staying off.
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False,
                        help="also run one real training step (needs a GPU; takes minutes)")
    parser.add_argument("--train-script", type=Path,
                        default=Path("scripts/train_dated_seva.py"),
                        help="training entrypoint for --smoke")
    parser.add_argument("--train-arg", action="append", default=[], metavar="FLAG",
                        help="extra flag to pass to the smoke run; repeatable. Flags the "
                             "script does not advertise are dropped rather than failing it")
    parser.add_argument("--no-appearance", action="store_true",
                        help="build samples without appearance features")
    parser.add_argument("--timeout", type=int, default=1800, metavar="SECONDS",
                        help="give up on the smoke step after this long (default 1800)")
    args = parser.parse_args()

    print(f"Preflight: {len(args.dataset)} dataset folder(s)\n")
    report = run_checks(
        args.dataset, hub=args.hub, image_root=args.image_root, min_old=args.min_old,
        train_script=args.train_script if args.smoke else None,
        extra=args.train_arg, appearance=not args.no_appearance,
        timeout=args.timeout, progress=lambda msg: print(f"  ..   {msg}\n"),
    )
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
