#!/usr/bin/env python
"""
Plots the Average atom ablation (scripts/run_average_ablation.sh on the fast datasets,
scripts/run_average_noisy_ablation.sh on the noisy ones): per dataset, the paired difference
average_prob 0.5 minus 0 (same split and seed).

    python scripts/plot_average_ablation.py [--fast_dir results/ablation/average_d2]
        [--noisy_dir results/ablation/average_d2_noisy] [--out_dir results/figures]

Writes average_ablation_paired as PDF and PNG, three panels sharing the dataset axis:
validation balanced accuracy at the 1-SE lambda, at the best-CV lambda, and fit balanced
accuracy at the 1-SE lambda. Grey dots are runs, coloured dots the mean over runs with its
standard error, and the dashed segment of each group is the mean over its datasets. Each axis
is cut at 2.5 times the largest |mean| + s.e. of its panel; runs beyond it are drawn as
triangles at the edge.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_ablation import CHANCE, GRID, INK, INK_2, style_axis  # also sets the paper's rcParams

ARMS = (0.0, 0.5)
# group -> (label, color)
GROUP_STYLE = {"noisy": ("Noisy datasets", "#eb6834"), "fast": ("Fast datasets", "#2a78d6")}
PANELS = [
    ("val_balanced_accuracy", "Validation, 1-SE $\\lambda$"),
    ("val_balanced_accuracy_cvmax", "Validation, best-CV $\\lambda$"),
    ("fit_balanced_accuracy", "Fit, 1-SE $\\lambda$"),
]


def load_diffs(res_dir: Path) -> pd.DataFrame:
    """(dataset, run) x metric: value at average_prob 0.5 minus value at 0."""
    df = pd.concat([pd.read_csv(f) for f in sorted(res_dir.glob("*.csv"))], ignore_index=True)
    df = df[(df.status == "ok") & ~df.dataset.isin(CHANCE)]
    wide = df.pivot_table(index=["dataset", "run"], columns="average_prob", values=[c for c, _ in PANELS])
    return pd.DataFrame({c: wide[c][ARMS[1]] - wide[c][ARMS[0]] for c, _ in PANELS}).dropna()


def plot_paired(diffs: dict[str, pd.DataFrame]):
    # Rows top to bottom: noisy group then fast group, each sorted by its 1-SE mean, with a gap.
    rows, y, pos = [], {}, 0.0
    for g in GROUP_STYLE:
        order = diffs[g][PANELS[0][0]].groupby("dataset").mean().sort_values(ascending=False).index
        for ds in order:
            rows.append((g, ds))
            y[(g, ds)] = pos
            pos += 1
        pos += 0.8
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(9.6, 0.24 * len(rows) + 1.5), sharey=True,
                             constrained_layout=True)
    for ax, (col, title) in zip(axes, PANELS):
        stats_all = pd.concat([diffs[g][col].groupby("dataset").agg(["mean", "sem"]) for g in GROUP_STYLE])
        lim = 2.5 * float((stats_all["mean"].abs() + stats_all["sem"]).max())
        ax.set_xlim(-lim, lim)
        ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
        for g, (label, color) in GROUP_STYLE.items():
            d = diffs[g][col]
            stats = d.groupby("dataset").agg(["mean", "sem"])
            ys = np.array([y[(g, ds)] for ds in stats.index])
            for ds, yy in zip(stats.index, ys):
                runs = d.loc[ds].to_numpy()
                jitter = yy + rng.uniform(-0.18, 0.18, len(runs))
                inside = np.abs(runs) <= lim
                ax.scatter(runs[inside], jitter[inside], s=7, color="#bdbdbd", zorder=2, linewidths=0)
                for side, marker in ((-1, "<"), (1, ">")):
                    out = runs * side > lim
                    ax.scatter(np.full(out.sum(), side * lim * 0.97), jitter[out], s=12, marker=marker,
                               color="#bdbdbd", zorder=2, linewidths=0)
            ax.errorbar(stats["mean"], ys, xerr=stats["sem"], fmt="o", color=color, ecolor=color,
                        elinewidth=1.0, capsize=2, zorder=3, label=label)
            m = stats["mean"].mean()
            ax.plot([m, m], [ys.min() - 0.4, ys.max() + 0.4], color=color, linewidth=1.0, ls="--", zorder=2)
            ax.text(0.98, ys.max() + 0.55, f"mean ${m:+.3f}$", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8, color=INK_2)
        ax.set_title(title)
        ax.set_xlabel("Difference in balanced accuracy\n(with $-$ without Average)")
        style_axis(ax)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([y[r] for r in rows], [ds for _, ds in rows])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast_dir", default="results/ablation/average_d2")
    ap.add_argument("--noisy_dir", default="results/ablation/average_d2_noisy")
    ap.add_argument("--out_dir", default="results/figures")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_paired({"noisy": load_diffs(Path(args.noisy_dir)), "fast": load_diffs(Path(args.fast_dir))})
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"average_ablation_paired.{ext}", dpi=200, facecolor="white")
    print(f"wrote {out_dir / 'average_ablation_paired'}.{{pdf,png}}")


if __name__ == "__main__":
    main()
