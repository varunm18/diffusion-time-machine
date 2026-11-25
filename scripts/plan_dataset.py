#!/usr/bin/env python
"""Pick the next scenes to build, within a storage budget (CPU only, no downloads).

Old, posed photographs are the scarce resource in everything we measure: the
shortcut probe and the appearance-entanglement analysis both ran on 58 old photos
in the test split. `docs/census/census_scenes_top800.csv` already knows how many
each of 802 scenes has, so this turns that into a build order and says what the
result would cost and contain - before anything is downloaded.

    # what would 50 GB buy us, given what is already built?
    python scripts/plan_dataset.py --budget-gb 50 --have data/datasets/*_model0

    # or a fixed number of scenes
    python scripts/plan_dataset.py --n-scenes 25

It writes a markdown plan and a scene list that feeds straight into the pipeline:

    python scripts/process_scenes.py --scene-file <out>/scenes.txt
    for s in $(cat <out>/scenes.txt); do
        python scripts/build_scene_dataset.py --scene "$s" --max-modern 300
    done

Storage is measured from the datasets passed to --have rather than guessed, since
a wrong estimate is how a quota fills up halfway through a download.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from timemachine.analysis import planning

DEFAULT_CENSUS = Path("docs/census/census_scenes_top800.csv")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    p.add_argument("--budget-gb", type=float, help="stop once the plan would exceed this")
    p.add_argument("--n-scenes", type=int, help="stop after this many scenes")
    p.add_argument("--have", type=Path, nargs="*", default=[],
                   help="dataset folders already built: excluded from the plan, and used to "
                        "measure the real bytes-per-image")
    p.add_argument("--max-modern", type=int, default=300,
                   help="cap on modern photos per scene, as build_scene_dataset.py uses")
    p.add_argument("--gate", choices=sorted(planning.GATES), default="posed",
                   help="'posed' keeps scenes that already pass the proposal's gate in MegaScenes' "
                        "own reconstructions; 'any' also allows scenes whose old photos would need "
                        "posing ourselves")
    p.add_argument("--show", type=int, default=20, help="rows to print in the table")
    p.add_argument("--out", type=Path, help="output folder (default: data/experiments/dataset_plan)")
    args = p.parse_args()

    if args.budget_gb is None and args.n_scenes is None:
        raise SystemExit("Give --budget-gb or --n-scenes (or both).")
    if not args.census.exists():
        raise SystemExit(f"{args.census} not found. It ships with the repo; pass --census if moved.")

    out = args.out or Path("data/experiments/dataset_plan")
    out.mkdir(parents=True, exist_ok=True)

    census = planning.load_census(args.census)
    have = [Path(h).name.split("_model")[0] for h in args.have]
    bytes_per_image, measured = planning.calibrate_bytes_per_image(args.have)
    if args.have and not measured:
        print(f"warning: no cached .jpg found under {', '.join(str(h) for h in args.have)}; "
              "falling back to the default size estimate")

    plan = planning.plan_scenes(census, n_scenes=args.n_scenes, budget_gb=args.budget_gb,
                                max_modern=args.max_modern, bytes_per_image=bytes_per_image,
                                gate=args.gate, exclude=have)
    text = planning.render_plan(plan, bytes_per_image, measured, have, args.max_modern, args.show)
    (out / "plan.md").write_text(text, encoding="utf-8")
    plan.to_csv(out / "plan.csv", index=False)
    (out / "scenes.txt").write_text("\n".join(plan["scene"].astype(str)) + "\n", encoding="utf-8")

    print(text)
    print(f"wrote {out}/plan.md, plan.csv, scenes.txt ({len(plan)} scenes)")


if __name__ == "__main__":
    main()
