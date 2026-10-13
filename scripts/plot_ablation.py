#!/usr/bin/env python
"""
Plots the validation ablation of scripts/run_ablation.py (depth x number of formulae, and the
cut point read off the saved lambda paths).

    python scripts/plot_ablation.py [--res_dir results/ablation/full] [--out_dir results/figures]

Datasets at chance (CHANCE) are dropped. Every figure averages runs within a dataset first and
datasets second. Writes, as PDF and PNG:
  ablation_depth_budget     (a) validation balanced accuracy and (b) number of selected formulae
                            against the number of sampled formulae M, one line per depth, at the
                            1-SE lambda. Bands: standard error over runs.
  ablation_cut_point        validation balanced accuracy against the number of selected formulae,
                            one point per cut point and one curve per depth, at M = --n_formulas
  ablation_depth_paired     per dataset, the paired difference in validation balanced accuracy
                            between consecutive depths at M = --n_formulas (same split and seed),
                            with its standard error over runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

CHANCE = ["AtrialFibrillation", "StandWalkJump", "SelfRegulationSCP2"]
CUT_POINTS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
# depth -> (color, marker, linestyle); marker and linestyle repeat the identity for print
DEPTH_STYLE = {1: ("#2a78d6", "o", "-"), 2: ("#eb6834", "s", "--"), 3: ("#1baf7a", "^", ":")}
INK, INK_2, GRID = "#000000", "#444444", "#d9d9d9"

# Computer Modern (bundled with matplotlib), so the figures match the paper's font
# without needing a LaTeX installation
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 10, "axes.titlesize": 10, "legend.fontsize": 9,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 4.5, "legend.frameon": True, "legend.fancybox": False,
    "legend.edgecolor": INK, "legend.framealpha": 1.0, "patch.linewidth": 0.6, "pdf.fonttype": 42,
})


def load_rows(res_dir: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted(res_dir.glob("*.csv"))], ignore_index=True)
    df = df[(df.status == "ok") & ~df.dataset.isin(CHANCE)].copy()
    return df.astype({"depth_max": int, "n_formulas": int})


def load_cut_points(res_dir: Path) -> pd.DataFrame:
    """One row per (fit, cut point): validation balanced accuracy and model size at the lambda
    that cut point selects, i.e. the first one within cut_point standard errors of the best CV score."""
    rows = []
    for f in sorted(res_dir.glob("*.paths.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["dataset"] in CHANCE:
                continue
            cv, se = np.array(r["cv_mean_score"]), np.array(r["cv_standard_error"])
            best = r["index_cvmax"]
            for c in CUT_POINTS:
                i = int(np.argmax(cv >= cv[best] - c * se[best]))
                rows.append({"dataset": r["dataset"], "run": r["run"], "depth_max": int(r["depth_max"]),
                             "n_formulas": int(r["n_formulas"]), "cut_point": c,
                             "val_balanced_accuracy": r["val_balanced_accuracy_path"][i],
                             "n_selected": r["n_selected_path"][i]})
    return pd.DataFrame(rows)


def mean_and_se(df: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    """Mean over datasets of the per-dataset mean of col, and its standard error over runs."""
    g = df.groupby(by + ["dataset"])[col].agg(["mean", "sem"]).reset_index()
    out = g.groupby(by).agg(mean=("mean", "mean"), var=("sem", lambda s: (s.fillna(0) ** 2).sum()),
                            n=("mean", "size"))
    out["se"] = np.sqrt(out["var"]) / out["n"]
    return out[["mean", "se"]].reset_index()


def style_axis(ax):
    ax.grid(color=GRID, linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)


def plot_depth_budget(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    panels = [("val_balanced_accuracy", "Validation balanced accuracy"), ("n_selected", "Selected formulae")]
    for ax, (col, label) in zip(axes, panels):
        t = mean_and_se(df, ["depth_max", "n_formulas"], col)
        for depth, (color, marker, ls) in DEPTH_STYLE.items():
            s = t[t.depth_max == depth]
            ax.fill_between(s.n_formulas, s["mean"] - s.se, s["mean"] + s.se, color=color, alpha=0.15, linewidth=0)
            ax.plot(s.n_formulas, s["mean"], color=color, marker=marker, linestyle=ls,
                    label=f"depth {depth}")
        ax.set_xscale("log")
        ax.set_xticks(sorted(df.n_formulas.unique()), [f"{m:,}" for m in sorted(df.n_formulas.unique())])
        ax.minorticks_off()
        ax.set_xlabel("Sampled formulae $M$")
        ax.set_ylabel(label)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=INK_2, alpha=0.2, linewidth=0))
    labels.append(r"$\pm 1$ s.e. over runs")
    axes[0].legend(handles, labels, loc="lower right")
    return fig


def plot_cut_point(cuts: pd.DataFrame, n_formulas: int):
    s = cuts[cuts.n_formulas == n_formulas]
    acc = s.groupby(["depth_max", "cut_point", "dataset"]).val_balanced_accuracy.mean().groupby(level=[0, 1]).mean()
    size = s.groupby(["depth_max", "cut_point", "dataset"]).n_selected.mean().groupby(level=[0, 1]).mean()
    fig, ax = plt.subplots(figsize=(4.6, 3.3), constrained_layout=True)
    for depth, (color, marker, ls) in DEPTH_STYLE.items():
        if depth not in acc.index.get_level_values(0):
            continue
        x, y = size[depth], acc[depth]
        ax.plot(x, y, color=color, marker=marker, linestyle=ls, label=f"depth {depth}")
        ax.plot(x[1], y[1], marker=marker, markersize=7, color=color, markeredgecolor=INK,
                markeredgewidth=1.0, linestyle="none")  # the 1-SE rule
    for c in CUT_POINTS:  # cut point values, once, on the depth-2 curve
        ax.annotate(f"{c:g}", (size[2][c], acc[2][c]), xytext=(5, -9), textcoords="offset points",
                    ha="left", fontsize=8, color=INK_2)
    ax.plot([], [], marker="o", markersize=7, color="white", markeredgecolor=INK,
            markeredgewidth=1.0, linestyle="none", label="cut point 1 (1-SE rule)")
    ax.set_xscale("log")
    ax.set_xticks([75, 100, 150, 200, 300, 400, 600], ["75", "100", "150", "200", "300", "400", "600"])
    ax.minorticks_off()
    ax.set_xlabel("Selected formulae")
    ax.set_ylabel("Validation balanced accuracy")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=2)
    style_axis(ax)
    return fig


def plot_depth_paired(df: pd.DataFrame, n_formulas: int):
    s = df[df.n_formulas == n_formulas]
    wide = s.pivot_table(index=["dataset", "run"], columns="depth_max", values="val_balanced_accuracy")
    pairs = [(1, 2), (2, 3)]
    stats = {p: (wide[p[1]] - wide[p[0]]).dropna().groupby("dataset").agg(["mean", "sem"]) for p in pairs}
    order = stats[pairs[0]]["mean"].sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.2 * len(order) + 1.0), sharey=True, sharex=True,
                             constrained_layout=True)
    for ax, (a, b) in zip(axes, pairs):
        t = stats[(a, b)].reindex(order)
        color = DEPTH_STYLE[b][0]
        y = np.arange(len(order))
        ax.axvline(0, color=INK, linewidth=0.6)
        ax.errorbar(t["mean"], y, xerr=t["sem"], fmt="o", color=color, ecolor=color, elinewidth=1.0,
                    capsize=2)
        m = t["mean"]
        ax.set_title(f"Depth {b} $-$ depth {a}\nmean ${m.mean():+.3f}$, {(m > 0).sum()} up, {(m < 0).sum()} down")
        ax.set_yticks(y, order)
        ax.set_xlabel("Difference in validation balanced accuracy\n(mean $\\pm 1$ s.e. over runs)")
        style_axis(ax)
        ax.tick_params(axis="y", length=0)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default="results/ablation/full")
    ap.add_argument("--out_dir", default="results/figures")
    ap.add_argument("--n_formulas", type=int, default=10000, help="M of the cut-point and paired figures")
    args = ap.parse_args()
    res_dir, out_dir = Path(args.res_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, cuts = load_rows(res_dir), load_cut_points(res_dir)
    figures = {
        "ablation_depth_budget": plot_depth_budget(df),
        "ablation_cut_point": plot_cut_point(cuts, args.n_formulas),
        "ablation_depth_paired": plot_depth_paired(df, args.n_formulas),
    }
    for name, fig in figures.items():
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{name}.{ext}", dpi=200, facecolor="white")
        print(f"wrote {out_dir / name}.{{pdf,png}}")


if __name__ == "__main__":
    main()
