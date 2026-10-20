#!/usr/bin/env python
"""
Plots the comparison of scripts/run_comparison.py.

    python scripts/plot_comparison.py [--res_dir results/comparison_balanced] [--out_dir results/figures]

Every figure uses the test balanced accuracy averaged over the runs of a dataset, on the
datasets where every plotted method has all its runs (the others are listed on stdout).
Methods with fewer than --min_datasets complete datasets (mrseql, so far) are left out.
Writes, as PDF and PNG:
  comparison_cd          critical difference diagram (aeon: average ranks, cliques from
                         pairwise one-sided Wilcoxon tests at ALPHA / (number of methods - 1))
  comparison_pairwise    stlrocket against each other method, one point per dataset; points
                         above the diagonal are datasets where stlrocket is better
  comparison_time        mean test balanced accuracy against the median time per dataset of
                         the whole pipeline (features, fit and prediction, train and test), one
                         point per method
and two LaTeX tables:
  comparison_table.tex   per dataset (rows) and method (columns), the mean and standard
                         deviation over runs of the test balanced accuracy, best in bold, with
                         the mean and the mean rank over datasets
  comparison_tests.tex   the Friedman test over all methods and, for every pair, the mean
                         difference of balanced accuracy, wins/losses over datasets and the
                         p-value of the two-sided Wilcoxon signed-rank test, Holm-adjusted over
                         all pairs (Demsar 2006; Benavoli et al. 2016)
  comparison_time.tex    time of each pipeline (features of both splits, training with model
                         selection, prediction): mean over the runs of a dataset, then median,
                         mean and maximum over datasets, the median ratio to MiniRocket and the
                         median share of feature extraction
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
# Order of the table columns: the STL feature methods, then the MiniRocket feature ones
METHOD_STYLE = {
    "stlrocket": ("STLRocket", "#2a78d6", "o"),
    "stl_ridge": ("STL + ridge", "#8a5cc2", "X"),
    "stl_tree": ("STL tree", "#eb6834", "s"),
    "minirocket_lasso": ("MiniRocket + LASSO", "#e87ba4", "v"),
    "minirocket": ("MiniRocket", "#1baf7a", "^"),
    "multirockethydra": ("MultiRocket-Hydra", "#eda100", "D"),
    "mrseql": ("MrSEQL", "#008300", "P"),
}
ALPHA = 0.05   # significance level of the tests (comparison_tests.tex) and of the CD diagram
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
                                      test="wilcoxon", correction="holm", alpha=ALPHA, width=6, textspace=1.6)
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
    n_runs = df.groupby(["dataset", "method"]).run.nunique().max()
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Test balanced accuracy of each method on each dataset: mean $\pm$ standard deviation over "
             f"{n_runs} runs, best in bold. Last rows: mean over the {len(g)} datasets and mean rank (1 is best).}}",
             r"\label{tab:accuracy}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "c" * len(methods) + "}", r"\toprule",
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
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    print(pd.DataFrame({"mean bacc": means, "mean rank": ranks}).round(3).to_string())


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(running, 1.0)
    return adj


def write_tests(acc: pd.DataFrame, methods: list[str], path: Path):
    """Friedman test over all methods, then every pair: two-sided Wilcoxon signed-rank test
    over datasets (datasets with equal accuracy dropped), Holm-adjusted over all pairs."""
    from itertools import combinations
    from scipy.stats import friedmanchisquare, wilcoxon
    ranks = acc[methods].rank(axis=1, ascending=False).mean()
    order = sorted(methods, key=lambda m: ranks[m])
    stat, p_friedman = friedmanchisquare(*(acc[m] for m in order))
    pairs = list(combinations(order, 2))
    rows = []
    for a, b in pairs:
        d = acc[a] - acc[b]
        rows.append({"a": a, "b": b, "diff": d.mean(), "se": d.std() / np.sqrt(len(d)),
                     "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": wilcoxon(acc[a], acc[b]).pvalue})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.to_numpy())
    label = {m: METHOD_STYLE[m][0] for m in methods}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Pairwise comparison of the methods over the " f"{len(acc)}" r" datasets (test balanced "
             r"accuracy, mean over runs). Friedman test over all methods: $\chi^2_{" f"{len(methods) - 1}" r"} = "
             f"{stat:.1f}" r"$, $p = " f"{p_friedman:.1e}".replace("e-0", r" \cdot 10^{-").replace("e-", r" \cdot 10^{-")
             + (r"}" if "e-" in f"{p_friedman:.1e}" else "") + r"$. For each pair, mean difference of balanced "
             r"accuracy (A $-$ B) with its standard error over datasets, datasets where A is better and worse, "
             r"and $p$-value of the two-sided Wilcoxon signed-rank test, Holm-adjusted over the "
             f"{len(pairs)} pairs; in bold, $p < {ALPHA}$. Methods ordered by mean rank.}}",
             r"\label{tab:accuracy_tests}", r"\small",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"A & B & A $-$ B & Wins / losses & $p$ & $p$ (Holm) \\", r"\midrule"]
    for r in t.itertuples():
        p_holm = f"{r.p_holm:.3f}" if r.p_holm >= 0.001 else "$<0.001$"
        p_raw = f"{r.p:.3f}" if r.p >= 0.001 else "$<0.001$"
        lines.append(f"{label[r.a]} & {label[r.b]} & ${r.diff:+.3f} \\pm {r.se:.3f}$ & {r.wins} / {r.losses} & "
                     f"{p_raw} & " + (f"\\textbf{{{p_holm}}}" if r.p_holm < ALPHA else p_holm) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    print(f"Friedman chi2({len(methods) - 1}) = {stat:.2f}, p = {p_friedman:.2e}")
    print(t.assign(a=t.a.map(label), b=t.b.map(label)).round(4).to_string(index=False))


def write_time_table(df: pd.DataFrame, methods: list[str], path: Path):
    t = df.groupby(["dataset", "method"]).total_s.mean().unstack()[methods]
    feat = (df.groupby(["dataset", "method"]).features_s.mean() /
            df.groupby(["dataset", "method"]).total_s.mean()).unstack().reindex(columns=methods)
    ratio = t.div(t["minirocket"], axis=0) if "minirocket" in methods else None

    def fmt(x):
        return "--" if pd.isna(x) else (f"{x:.2g}" if x < 10 else f"{x:.0f}")

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Time in seconds of each pipeline, from the series to the predictions on the test set "
             r"(feature extraction for both splits, training with model selection, and prediction), with $4$ CPU "
             f"cores: mean over the runs of a dataset, then median, mean and maximum over the {len(t)} datasets, "
             r"median over the datasets of the ratio to the time of MiniRocket, and median share of the time "
             r"spent extracting the features (for the methods that extract them separately).}",
             r"\label{tab:time}", r"\small", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Method & Median & Mean & Maximum & Ratio to MiniRocket & Features \\", r"\midrule"]
    for m in sorted(methods, key=lambda m: t[m].median()):
        share = feat[m].median() if m in feat and feat[m].notna().any() else np.nan
        lines.append(f"{METHOD_STYLE[m][0]} & {fmt(t[m].median())} & {fmt(t[m].mean())} & {fmt(t[m].max())} & "
                     f"{fmt(ratio[m].median()) if ratio is not None else '--'} & "
                     + ("--" if pd.isna(share) else f"{100 * share:.0f}\\%") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")
    print(pd.DataFrame({"median": t.median(), "mean": t.mean(), "max": t.max(),
                        "ratio": ratio.median() if ratio is not None else np.nan,
                        "features": feat.median()}).round(2).to_string())


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
    write_tests(acc, methods, out_dir / "comparison_tests.tex")
    write_time_table(df, methods, out_dir / "comparison_time.tex")


if __name__ == "__main__":
    main()
