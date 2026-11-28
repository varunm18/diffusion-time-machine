#!/usr/bin/env python
"""Slide figures for the appearance findings.

Three charts for the final presentation, drawn from the numbers committed in
``docs/findings.md`` under "Both appearance findings, re-measured on 28 scenes
(2026-11-25)". The values are transcribed here rather than recomputed so a figure
can never disagree with the written record: the analyses read ~10k images and take
minutes, and a chart that silently drifts from the report it illustrates is worse
than one that takes a moment to regenerate. Re-run the two scripts named in
``SOURCE`` if the dataset changes, then update these constants in one place.

    python scripts/make_figures.py                  # writes docs/figures/*.png

Palette and mark choices follow one rule worth keeping: every bar carries its own
value label, so the chart is readable without matching colours to a legend, in
print, and for a colour-blind reader.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

SOURCE = ("28 scenes, 12,323 images · scripts/date_shortcut_probe.py, "
          "scripts/appearance_entanglement.py · docs/findings.md (2026-11-25)")

BLUE, ORANGE = "#257de0", "#f47230"
INK, MUTED, GRID = "#191919", "#666666", "#d9d9d9"
CHANCE = 0.5


def _style(ax, ylabel: str, ymax: float = 1.0, chance: bool = True) -> None:
    """Recessive axes, a horizontal grid only, and the line that gives AUC meaning."""
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=11, color=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", labelsize=11, colors=MUTED, length=0)
    for label in ax.get_xticklabels():
        label.set_color(INK)
    if chance:
        # Without this line an AUC bar is unreadable: 0.55 and 0.87 look alike.
        ax.axhline(CHANCE, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.text(ax.get_xlim()[1], CHANCE, "  chance", va="center", ha="left",
                fontsize=9, color=MUTED)


def _label(ax, bars, values, fmt="{:.3f}", dy=0.012) -> None:
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + dy,
                fmt.format(value), ha="center", va="bottom",
                fontsize=11, color=INK, fontweight="medium")


def _footer(fig) -> None:
    fig.text(0.01, 0.015, SOURCE, fontsize=7.5, color=MUTED, ha="left")


def figure_shortcut(out: Path) -> Path:
    """How much of the date signal is the photographic medium rather than the scene."""
    treatments = ["as captured", "grayscale", "tone-matched"]
    auc = [0.868, 0.779, 0.673]
    lo = [0.85, 0.75, 0.64]
    hi = [0.89, 0.81, 0.71]
    cues = ["colour", "tone", "detail", "framing"]
    cue_auc = [0.792, 0.751, 0.548, 0.516]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200,
                                      gridspec_kw={"width_ratios": [1, 1.1]})
    fig.patch.set_facecolor("white")

    bars = left.bar(treatments, auc, width=0.55, color=BLUE, zorder=2)
    left.errorbar(range(len(auc)), auc,
                  yerr=[[a - l for a, l in zip(auc, lo)], [h - a for a, h in zip(auc, hi)]],
                  fmt="none", ecolor=INK, elinewidth=1.4, capsize=5, zorder=3)
    _label(left, bars, auc, dy=0.035)
    _style(left, "era AUC  (pre-1945 vs modern)")
    left.set_title("Removing colour does not remove the shortcut",
                   fontsize=12.5, color=INK, pad=12, loc="left")

    bars = right.bar(cues, cue_auc, width=0.55, color=BLUE, zorder=2)
    _label(right, bars, cue_auc)
    _style(right, "era AUC")
    right.set_title("Colour and tone carry it; framing carries nothing",
                    fontsize=12.5, color=INK, pad=12, loc="left")

    fig.suptitle("The date is readable from the photograph, not the place",
                 fontsize=15, color=INK, x=0.01, ha="left", y=0.99, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    _footer(fig)
    path = out / "shortcut_era_auc.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def figure_separability(out: Path) -> Path:
    """Era can be projected out; the lighting the conditioning is for survives."""
    groups = ["era\n(pre-1945 vs modern)", "lighting\n(sunlit vs overcast)"]
    before = [0.858, 0.856]
    after = [0.552, 0.878]
    x = range(len(groups))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    b1 = ax.bar([i - width / 2 - 0.01 for i in x], before, width, label="appearance as-is",
                color=BLUE, zorder=2)
    b2 = ax.bar([i + width / 2 + 0.01 for i in x], after, width,
                label="6 of 14 era directions removed", color=ORANGE, zorder=2)
    _label(ax, b1, before)
    _label(ax, b2, after)

    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    _style(ax, "AUC")
    ax.legend(frameon=False, fontsize=10.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=MUTED)
    # A subtitle rather than a callout: an arrow into the bar collided with its
    # own value label, and the number is about the whole removal, not one bar.
    ax.set_title("Removing six of fourteen directions keeps 73% of the appearance variance",
                 fontsize=11, color=MUTED, pad=12, loc="left")
    fig.suptitle("Era is removable. Lighting is not collateral damage.",
                 fontsize=15, color=INK, x=0.012, ha="left", y=0.985, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    _footer(fig)
    path = out / "era_removal_separability.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def figure_causal(out: Path) -> Path:
    """The causal check: change only the tone and the classifier changes its mind."""
    labels = ["modern photographs,\nuntouched", "same photographs,\nsepia-toned"]
    share = [31, 100]

    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=200)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, share, width=0.5, color=[BLUE, ORANGE], zorder=2)
    _label(ax, bars, share, fmt="{:.0f}%", dy=1.5)
    _style(ax, "called pre-1945 by the era classifier  (%)", ymax=115, chance=False)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Tone the modern photographs and they read as old",
                 fontsize=15, color=INK, pad=14, loc="left", fontweight="semibold")
    ax.text(0, -0.28, "1,326 modern photographs. The scene is untouched - only the "
            "colour and tone change.", transform=ax.transAxes, fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    _footer(fig)
    path = out / "causal_sepia_flip.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for build in (figure_shortcut, figure_separability, figure_causal):
        print(f"  wrote {build(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
