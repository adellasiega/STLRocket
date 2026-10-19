#!/usr/bin/env python
"""
Plots the comparison of scripts/run_comparison.py.

    python scripts/plot_comparison.py [--res_dir results/comparison_balanced] [--out_dir results/figures]

Every figure uses the test balanced accuracy averaged over the runs of a dataset, on the
datasets where every plotted method has all its runs (the others are listed on stdout).
Methods with fewer than --min_datasets complete datasets (mrseql, so far) are left out.
Writes, as PDF and PNG:
  comparison_cd          critical difference diagram (aeon: average ranks, cliques from
                         pairwise one-sided Wilcoxon tests with Holm correction, alpha 0.1)
  comparison_pairwise    stlrocket against each other method, one point per dataset; points
                         above the diagonal are datasets where stlrocket is better
  comparison_time        mean test balanced accuracy against the median time per dataset of
                         the whole pipeline (features, fit and prediction, train and test), one
                         point per method
and comparison_table.tex, the per-dataset mean and standard deviation over runs, best in bold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REFERENCE = "stlrocket"
# method -> (label, color, marker); the marker repeats the identity for print
METHOD_STYLE = {
    "stlrocket": ("STLRocket", "#2a78d6", "o"),
    "stl_tree": ("STL tree", "#eb6834", "s"),
    "stl_ridge": ("STL + ridge", "#8a5cc2", "X"),
    "minirocket": ("MiniRocket", "#1baf7a", "^"),
    "multirockethydra": ("MultiRocket-Hydra", "#eda100", "D"),
    "minirocket_lasso": ("MiniRocket + lasso", "#e87ba4", "v"),
    "mrseql": ("MrSEQL", "#008300", "P"),
}
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
    df = pd.DataFrame([json.loads(f.read_text()) for f in sorted(res_dir.glob("*/run*/*.json"))])
    # Total time of the pipeline, train and test: the STL and MiniRocket features of both splits
    # are computed together (features_s), while aeon's classifiers transform the test set in predict
    df["total_s"] = df.get("features_s", 0).fillna(0) + df.fit_s + df.predict_s.fillna(0)
    return df


def complete(df: pd.DataFrame, min_datasets: int) -> tuple[pd.DataFrame, list[str]]:
    """Rows of the methods with at least min_datasets complete datasets, restricted to the
    datasets every such method has completed; a dataset is complete when a method has as
    many runs on it as the method with most runs."""
    n_runs = df.groupby(["method", "dataset"]).run.nunique().unstack()
    done = n_runs.eq(n_runs.max(axis=0), axis=1) & n_runs.notna()
    methods = [m for m in METHOD_STYLE if m in done.index and done.loc[m].sum() >= min_datasets]
    datasets = sorted(done.columns[done.loc[methods].all()], key=str.lower)
    for m in METHOD_STYLE:
        if m in done.index:
            missing = sorted(done.columns[~done.loc[m]], key=str.lower)
            print(f"{m:18s} {done.loc[m].sum():2d} complete datasets"
                  + ("" if m in methods else "  (left out)")
                  + (f"; incomplete: {', '.join(missing)}" if missing and len(missing) < 6 else ""))
    print(f"plotting {len(methods)} methods on {len(datasets)} datasets")
    return df[df.method.isin(methods) & df.dataset.isin(datasets)], methods


def style_axis(ax):
    ax.grid(color=GRID, linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)


def plot_cd(acc: pd.DataFrame, methods: list[str]):
    from aeon.visualisation import plot_critical_difference
    fig, _ = plot_critical_difference(acc[methods].to_numpy(), [METHOD_STYLE[m][0] for m in methods],
                                      test="wilcoxon", correction="holm", alpha=0.1, width=6, textspace=1.6)
    return fig


def plot_pairwise(acc: pd.DataFrame, methods: list[str]):
    others = [m for m in methods if m != REFERENCE]
    ncols = min(len(others), 4)
    nrows = int(np.ceil(len(others) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.95 * ncols, 2.15 * nrows), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    ref_label, ref_color, ref_marker = METHOD_STYLE[REFERENCE]
    for ax, m in zip(axes.flat, others):
        label, color, marker = METHOD_STYLE[m]
        x, y = acc[m], acc[REFERENCE]
        ax.plot([0, 1], [0, 1], color=INK_2, linewidth=0.6)
        ax.scatter(x, y, s=24, color=color, marker=marker, edgecolors="white", linewidths=0.5, zorder=3)
        d = y - x
        win, tie, loss = (d > 0.005).sum(), (d.abs() <= 0.005).sum(), (d < -0.005).sum()
        ax.text(0.04, 0.96, f"{ref_label} better", transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color=INK_2, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
        ax.text(0.96, 0.04, f"{label} better", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=INK_2, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
        ax.set_title(f"vs {label}\n{win} wins, {tie} ties, {loss} losses", fontsize=9)
        ax.set_xlabel(label)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        style_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(ref_label)
    for ax in axes.flat[len(others):]:
        ax.set_visible(False)
    return fig


def plot_time(df: pd.DataFrame, acc: pd.DataFrame, methods: list[str]):
    time = df.groupby(["method", "dataset"]).total_s.mean().groupby("method").median()
    fig, ax = plt.subplots(figsize=(4.6, 3.1), constrained_layout=True)
    for m in methods:
        label, color, marker = METHOD_STYLE[m]
        ax.plot(time[m], acc[m].mean(), marker=marker, markersize=8, color=color, linestyle="none",
                markeredgecolor="white", markeredgewidth=0.8, label=label)
        ax.annotate(label, (time[m], acc[m].mean()), xytext=(7, -3), textcoords="offset points",
                    fontsize=8, color=INK_2, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    ax.set_xscale("log")
    ax.set_xlabel("Median time per dataset, train and test (s)")
    ax.set_ylabel("Mean test balanced accuracy")
    ax.margins(x=0.35, y=0.12)
    style_axis(ax)
    return fig


def write_table(df: pd.DataFrame, methods: list[str], path: Path):
    g = df.groupby(["dataset", "method"]).bacc_test.agg(["mean", "std"]).unstack()
    lines = [r"\begin{tabular}{l" + "c" * len(methods) + "}", r"\toprule",
             "Dataset & " + " & ".join(METHOD_STYLE[m][0] for m in methods) + r" \\", r"\midrule"]
    for d in sorted(g.index, key=str.lower):
        best = g.loc[d, "mean"][methods].max()
        cells = []
        for m in methods:
            mu, sd = g.loc[d, ("mean", m)], g.loc[d, ("std", m)]
            s = f"{mu:.3f} $\\pm$ {sd:.3f}"
            cells.append(f"\\textbf{{{s}}}" if round(mu, 3) == round(best, 3) else s)
        lines.append(f"{d} & " + " & ".join(cells) + r" \\")
    means = g["mean"][methods].mean()
    ranks = g["mean"][methods].rank(axis=1, ascending=False).mean()
    lines += [r"\midrule", "Mean & " + " & ".join(f"{means[m]:.3f}" for m in methods) + r" \\",
              "Mean rank & " + " & ".join(f"{ranks[m]:.2f}" for m in methods) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    print(pd.DataFrame({"mean bacc": means, "mean rank": ranks}).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default="results/comparison_balanced")
    ap.add_argument("--out_dir", default="results/figures")
    ap.add_argument("--min_datasets", type=int, default=20,
                    help="leave out the methods with fewer complete datasets")
    args = ap.parse_args()
    res_dir, out_dir = Path(args.res_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, methods = complete(load_rows(res_dir), args.min_datasets)
    acc = df.groupby(["dataset", "method"]).bacc_test.mean().unstack()[methods]
    figures = {
        "comparison_cd": plot_cd(acc, methods),
        "comparison_pairwise": plot_pairwise(acc, methods),
        "comparison_time": plot_time(df, acc, methods),
    }
    for name, fig in figures.items():
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{name}.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
        print(f"wrote {out_dir / name}.{{pdf,png}}")
    write_table(df, methods, out_dir / "comparison_table.tex")


if __name__ == "__main__":
    main()
