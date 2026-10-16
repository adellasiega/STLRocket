#!/usr/bin/env python
"""
Plots the comparison of scripts/run_anchors_comparison.py (STLRocket instance explanations
against Anchors on the same classifier and features).

    python scripts/plot_anchors_comparison.py [--res_dir results/anchors] [--out_dir results/figures]

Every number is the mean over the runs of a dataset of the per-run mean over the explained
test series, on the test set and w.r.t. the model's predictions. Writes:
  anchors_precision_coverage  (PDF, PNG) test precision against test coverage of the instance
                              explanations, one point per dataset and method, joined per dataset
  anchors_table.tex           per dataset and method: precision, coverage and length of the
                              instance explanations, F1 of the global explanations and time per
                              series; Mean row with Wilcoxon signed-rank p-values over datasets
  anchors_summary.tex         the paper's table: one row per metric, the mean over datasets of
                              each method, the Wilcoxon p-value and the number of datasets where
                              STLRocket is higher. Precision is pooled over the explanations (as in
                              the explanation ablation)
and prints how often Anchors' precision guarantee (estimated precision >= threshold) holds on
test, and as a diagnostic how often test precision reaches our Wilson bound (an objective, not a claim).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_comparison import INK, INK_2, style_axis  # also sets the paper's rcParams

# method -> (label, color, marker)
METHOD_STYLE = {"stlrocket": ("STLRocket", "#2a78d6", "o"), "anchors": ("Anchors", "#eb6834", "s")}
COLUMNS = [("mean_precision_test", "Precision", "{:.3f}"), ("mean_coverage_test", "Coverage", "{:.3f}"),
           ("mean_length", "Length", "{:.2f}"), ("global_f1_test", "Global $F_1$", "{:.3f}"),
           ("time_s", "Time (s)", "{:.2g}")]


def load(res_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    done = [json.loads(f.read_text()) for f in sorted(res_dir.glob("*.done.json"))]
    empty = [(d["dataset"], d["run"]) for d in done if d["status"] != "ok"]
    if empty:
        print(f"runs with an empty model, left out: {empty}")
    summary = pd.concat([pd.read_csv(f) for f in sorted(res_dir.glob("*.summary.csv"))], ignore_index=True)
    inst = pd.concat([pd.read_csv(f).assign(dataset=f.name.split("_run")[0],
                                            run=int(f.name.split("_run")[1].split(".")[0]))
                      for f in sorted(res_dir.glob("*.instances.csv.gz"))], ignore_index=True)
    return summary, inst


def plot_precision_coverage(per_dataset: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    p = per_dataset.unstack("method")
    for d in p.index:
        ax.plot(p.loc[d, ("mean_coverage_test", list(METHOD_STYLE))], p.loc[d, ("mean_precision_test", list(METHOD_STYLE))],
                color=INK_2, linewidth=0.5, zorder=1)
    for m, (label, color, marker) in METHOD_STYLE.items():
        ax.scatter(p[("mean_coverage_test", m)], p[("mean_precision_test", m)], s=30, color=color, marker=marker,
                   edgecolors="white", linewidths=0.6, zorder=3, label=label)
    ax.set_xlabel("Test coverage of the instance explanations")
    ax.set_ylabel("Test precision of the instance explanations")
    ax.legend(loc="lower left")
    style_axis(ax)
    return fig


def write_table(per_dataset: pd.DataFrame, path: Path):
    p = per_dataset.unstack("method")
    methods = list(METHOD_STYLE)
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{label}}}" for _, label, _ in COLUMNS)
    sub = " & ".join(" & ".join(METHOD_STYLE[m][0] for m in methods) for _ in COLUMNS)
    lines = [r"\begin{tabular}{l" + "cc" * len(COLUMNS) + "}", r"\toprule", f" & {head} \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(COLUMNS))),
             f"Dataset & {sub} \\\\", r"\midrule"]
    for d in p.index:
        lines.append(d + " & " + " & ".join(fmt.format(p.loc[d, (c, m)]) for c, _, fmt in COLUMNS for m in methods)
                     + r" \\")
    lines.append(r"\midrule")
    lines.append("Mean & " + " & ".join(fmt.format(p[(c, m)].mean()) for c, _, fmt in COLUMNS for m in methods)
                 + r" \\")
    pvals = []
    for c, _, _ in COLUMNS:
        diff = (p[(c, "stlrocket")] - p[(c, "anchors")]).dropna()
        pv = wilcoxon(diff).pvalue if len(diff) > 1 and diff.abs().sum() > 0 else np.nan
        pvals.append(f"\\multicolumn{{2}}{{c}}{{$p = {pv:.2g}$}}" if np.isfinite(pv) else r"\multicolumn{2}{c}{--}")
    lines += ["Wilcoxon & " + " & ".join(pvals) + r" \\", r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


SUMMARY_ROWS = [("pooled_precision_test", "Precision", "{:.3f}"), ("mean_coverage_test", "Coverage", "{:.3f}"),
                ("mean_length", "Length", "{:.2f}"),
                ("global_f1_test", "Global $F_1$", "{:.3f}"), ("time_s", "Time (s)", "{}")]


def precision_gap(inst: pd.DataFrame) -> pd.Series:
    """Per (dataset, run, method): mean over the explained test series of the precision a
    method reports (Anchors: its estimate; ours: the Wilson bound on train, an objective rather
    than a claim) minus the test precision (rules no test series satisfies are left out)."""
    d = inst[(inst.split == "test") & inst.found].copy()
    d["claimed"] = np.where(d.method == "anchors", d.est_precision, d.lcb_train)
    d["gap"] = d.claimed - d.precision_test
    return d.dropna(subset=["gap"]).groupby(["dataset", "run", "method"]).gap.mean().rename("precision_gap")


def write_summary_table(summary: pd.DataFrame, path: Path):
    p = summary.groupby(["dataset", "method"])[[c for c, _, _ in SUMMARY_ROWS]].mean().unstack("method")
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Metric & STLRocket & Anchors & $p$ & STLRocket higher \\", r"\midrule"]
    for c, label, fmt in SUMMARY_ROWS:
        ours, theirs = p[(c, "stlrocket")], p[(c, "anchors")]
        diff = (ours - theirs).dropna()
        pv = wilcoxon(diff).pvalue
        if c == "time_s":
            cells = [f"{ours.mean():.3f}", f"{theirs.mean():.0f}"]
        else:
            cells = [fmt.format(ours.mean()), fmt.format(theirs.mean())]
        lines.append(f"{label} & {cells[0]} & {cells[1]} & {pv:.2g} & {(diff > 0).sum()}/{len(diff)} \\\\")
        print(f"{label:15s} {cells[0]:>8s} {cells[1]:>8s}  p={pv:.2g}  higher {(diff > 0).sum()}/{len(diff)}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def guarantees(inst: pd.DataFrame, threshold: float):
    """Share of explained test series whose method-specific precision claim holds on test."""
    d = inst[(inst.split == "test") & inst.found].copy()
    a = d[d.method == "anchors"]
    claimed = a[a.est_precision >= threshold]
    print(f"Anchors: estimated precision >= {threshold} for {len(claimed) / max(len(a), 1):.1%} of the "
          f"explanations; test precision >= {threshold} for {(claimed.precision_test >= threshold).mean():.1%} of those")
    s = d[d.method == "stlrocket"]
    print(f"STLRocket (diagnostic only, the bound is the search objective): test precision >= Wilson bound on train for "
          f"{(s.precision_test >= s.lcb_train).mean():.1%} of the explanations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default="results/anchors")
    ap.add_argument("--out_dir", default="results/figures")
    ap.add_argument("--threshold", type=float, default=0.95, help="Anchors precision threshold of the runs")
    args = ap.parse_args()
    res_dir, out_dir = Path(args.res_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, inst = load(res_dir)
    print(summary.groupby(["dataset", "method"]).run.nunique().unstack().to_string())
    cols = [c for c, _, _ in COLUMNS]
    per_dataset = summary.groupby(["dataset", "method"])[cols].mean()
    print(per_dataset.unstack("method").round(3).to_string())

    fig = plot_precision_coverage(per_dataset)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"anchors_precision_coverage.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
    print(f"wrote {out_dir / 'anchors_precision_coverage'}.{{pdf,png}}")
    write_table(per_dataset, out_dir / "anchors_table.tex")
    write_summary_table(summary, out_dir / "anchors_summary.tex")
    gap = precision_gap(inst).groupby(["dataset", "method"]).mean().unstack("method")["anchors"]
    print(f"Anchors: estimated minus test precision {gap.mean():+.3f} (mean over datasets)")
    guarantees(inst, args.threshold)


if __name__ == "__main__":
    main()
