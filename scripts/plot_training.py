#!/usr/bin/env python
"""Plot a training run's curves from its log.jsonl (needs the ``plots`` extra).

Panels: training loss (smoothed), the date-swap loss gap probe, and per render
probe the mean colour saturation of the generated targets for each render year.

    python scripts/plot_training.py data/experiments/train/notre_dame_overfit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_log(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--smooth", type=int, default=20, help="rolling window (log records) for the loss")
    args = parser.parse_args()

    records = load_log(args.run_dir)
    train = pd.DataFrame([r for r in records if "probe" not in r])
    gap = pd.DataFrame([r for r in records if r.get("probe") == "date_gap"])
    renders = [r for r in records if str(r.get("probe", "")).startswith("render:")]
    probe_names = sorted({r["probe"] for r in renders})

    n_panels = 2 + len(probe_names)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    ax = axes[0]
    if len(train):
        ax.plot(train["step"], train["loss"], alpha=0.25, color="C0")
        ax.plot(train["step"], train["loss"].rolling(args.smooth, min_periods=1).mean(), color="C0", label="loss")
        if train["loss_cond"].notna().any():
            ax.plot(train["step"], train["loss_cond"].rolling(args.smooth, min_periods=1).mean(),
                    color="C1", label="loss (conditional steps)")
        ax.set_yscale("log")
    ax.set(title="training loss", xlabel="step")
    ax.legend()

    ax = axes[1]
    if len(gap):
        for col in ("rel_gap_swapped", "rel_gap_unknown", "rel_gap_swapped_old_targets",
                    "rel_gap_swapped_modern_targets", "rel_gap_swapped_high_noise", "rel_gap_swapped_low_noise"):
            if col in gap:
                ax.plot(gap["step"], 100 * gap[col], marker="o", label=col.replace("rel_gap_", ""))
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(title="loss(wrong date) / loss(true date) - 1  [%]", xlabel="step")
    ax.legend(fontsize=8)

    for ax, name in zip(axes[2:], probe_names):
        rows = [(r["step"], m["year"], m["saturation"]) for r in renders if r["probe"] == name for m in r["per_year"]]
        df = pd.DataFrame(rows, columns=["step", "year", "saturation"])
        for year, group in df.groupby(df["year"].fillna(-1)):
            ax.plot(group["step"], group["saturation"], marker="o", label="unknown" if year == -1 else f"{year:g}")
        ax.set(title=f"{name.removeprefix('render:')}: saturation of renders", xlabel="step")
        ax.legend(fontsize=8, title="render year")

    fig.tight_layout()
    out = args.run_dir / "curves.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
