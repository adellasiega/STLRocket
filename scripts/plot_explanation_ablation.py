#!/usr/bin/env python
"""
Plots the validation ablation of the explanation method (scripts/run_explanation_ablation.py).

    python scripts/plot_explanation_ablation.py [--res_dir results/ablation/explanations_v2] [--out_dir results/figures]

Metrics are those of the instance rules of the validation series (evaluated on validation) and
of the global explanations on validation, all w.r.t. the model's predictions. Runs whose model
selected no formula are dropped. Every figure averages runs within a dataset first and datasets
second. Writes, as PDF and PNG:
  expl_ablation_variants        one panel per metric: mean over datasets of each construction,
                                one point per confidence level (candidates gxi). Error bars:
                                standard error over datasets
  expl_ablation_paired          per dataset, ordered_scan - beam (gxi, --confidence) on precision,
                                coverage, attribution share and global F1, mean +- s.e. over runs
  expl_ablation_tradeoff        per dataset, precision against attribution share of the four
                                constructions (gxi, --confidence), joined per dataset
  expl_ablation_global_rules    rules per class of the global explanation and its F1 against the
                                number of fit series per class (gxi, --confidence)
  expl_ablation_simplification  (a) STL formula size before and after simplify_rule by rule length,
                                (b) per dataset, share of rules that are smaller and share whose
                                satisfied validation series change
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_ablation import INK, INK_2, style_axis   # also applies the paper's rcParams

# construction -> (label, color, marker, linestyle): categorical slots 1-4 in fixed order,
# marker and linestyle repeat the identity for print
CONSTRUCTIONS = {
    "beam": ("beam", "#2a78d6", "o", "-"),
    "ordered_scan": ("ordered scan", "#eb6834", "s", "--"),
    "ordered_prefix": ("ordered prefix", "#1baf7a", "^", ":"),
    "ordered_nocut": ("ordered, no cut", "#eda100", "D", "-."),
}
CONFIDENCES = [0.9, 0.95, 0.99]
CONF_ALPHA = {0.9: 0.45, 0.95: 0.7, 0.99: 1.0}
METRICS = [   # column, label, lower is better
    ("val_pooled_prec_val", "Rule precision", False),
    ("val_coverage_val", "Rule coverage", False),
    ("val_length", "Rule length (literals)", True),
    ("val_attr_share", "Attribution share", False),
    ("val_flip", "Prediction flips without $R$", False),
    ("global_f1_val", "Global explanation F1", False),
    ("global_rules_per_class", "Global rules per class", True),
    ("time", "Time per run (s)", True),
]


def load_variants(res_dir: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted(res_dir.glob("*.variants.csv"))], ignore_index=True)
    df = df[df.n_selected > 0].copy()
    df["time"] = df.time_fit_s + df.time_val_s
    df["fit_per_class"] = df.n_fit / df.n_classes
    return df


def load_simplification(res_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(res_dir.glob("*.simplification.csv"))]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def per_dataset(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    return df.groupby(["dataset"] + by)[cols].mean().reset_index()


def plot_variants(df: pd.DataFrame):
    d = per_dataset(df[df.candidates == "gxi"], ["construction", "confidence"], [c for c, _, _ in METRICS])
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.6), constrained_layout=True)
    x0 = np.arange(len(CONSTRUCTIONS))
    offsets = dict(zip(CONFIDENCES, (-0.2, 0.0, 0.2)))
    for ax, (col, label, lower) in zip(axes.flat, METRICS):
        t = d.groupby(["construction", "confidence"])[col].agg(["mean", "sem"])
        for i, (c, (_, color, marker, _)) in enumerate(CONSTRUCTIONS.items()):
            for conf in CONFIDENCES:
                m, s = t.loc[(c, conf)]
                ax.errorbar(i + offsets[conf], m, yerr=s, fmt=marker, color=color, alpha=CONF_ALPHA[conf],
                            elinewidth=1.0, capsize=1.5, markersize=5)
        ax.set_xticks(x0, [v[0].replace(", ", ",\n").replace(" ", "\n", 1) for v in CONSTRUCTIONS.values()],
                      fontsize=8)
        ax.set_title(label + (" $\\downarrow$" if lower else " $\\uparrow$"))
        style_axis(ax)
    # confidence legend: one marker per level, same shape, increasing opacity
    for conf in CONFIDENCES:
        axes.flat[0].plot([], [], "o", color=INK_2, alpha=CONF_ALPHA[conf], label=f"confidence {conf:g}")
    axes.flat[0].legend(loc="lower left", fontsize=7)
    return fig


def plot_paired(df: pd.DataFrame, confidence: float):
    s = df[(df.candidates == "gxi") & (df.confidence == confidence)]
    panels = [("val_pooled_prec_val", "Rule precision"), ("val_coverage_val", "Rule coverage"),
              ("val_attr_share", "Attribution share"), ("global_f1_val", "Global F1")]
    wide = s.pivot_table(index=["dataset", "run"], columns="construction", values=[c for c, _ in panels])
    stats = {c: (wide[c]["ordered_scan"] - wide[c]["beam"]).dropna().groupby("dataset").agg(["mean", "sem"])
             for c, _ in panels}
    order = stats["val_pooled_prec_val"]["mean"].sort_values().index
    fig, axes = plt.subplots(1, len(panels), figsize=(9.6, 0.2 * len(order) + 1.3), sharey=True,
                             constrained_layout=True)
    color = CONSTRUCTIONS["ordered_scan"][1]
    y = np.arange(len(order))
    for ax, (col, label) in zip(axes, panels):
        t = stats[col].reindex(order)
        ax.axvline(0, color=INK, linewidth=0.6)
        ax.errorbar(t["mean"], y, xerr=t["sem"], fmt="s", color=color, ecolor=color, elinewidth=1.0, capsize=2,
                    markersize=4)
        m = t["mean"]
        ax.set_title(f"{label}\nmean ${m.mean():+.3f}$, {(m > 0).sum()} up, {(m < 0).sum()} down")
        ax.set_yticks(y, order)
        ax.tick_params(axis="y", length=0)
        style_axis(ax)
    fig.supxlabel(f"ordered scan $-$ beam (gxi, confidence {confidence:g}; mean $\\pm 1$ s.e. over runs)",
                  fontsize=9)
    return fig


def plot_tradeoff(df: pd.DataFrame, confidence: float):
    s = per_dataset(df[(df.candidates == "gxi") & (df.confidence == confidence)], ["construction"],
                    ["val_pooled_prec_val", "val_attr_share"])
    fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
    for c, (label, color, marker, _) in CONSTRUCTIONS.items():
        g = s[s.construction == c]
        ax.scatter(g.val_attr_share, g.val_pooled_prec_val, color=color, marker=marker, s=18, alpha=0.45,
                   edgecolors="white", linewidths=0.5, zorder=1)
        # mean over datasets, large and outlined
        ax.scatter(g.val_attr_share.mean(), g.val_pooled_prec_val.mean(), color=color, marker=marker, s=90,
                   edgecolors=INK, linewidths=1.0, label=label, zorder=3)
    ax.text(0.02, 0.02, "small: one dataset; large: mean over datasets", transform=ax.transAxes, fontsize=7,
            color=INK_2)
    ax.set_xlabel("Attribution share (faithfulness to the model)")
    ax.set_ylabel("Rule precision on validation")
    ax.legend(loc="lower right", fontsize=8)
    style_axis(ax)
    return fig


def plot_global_rules(df: pd.DataFrame, confidence: float):
    s = per_dataset(df[(df.candidates == "gxi") & (df.confidence == confidence)], ["construction"],
                    ["fit_per_class", "global_rules_per_class", "global_f1_val"])
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True, sharex=True)
    for ax, (col, label) in zip(axes, [("global_rules_per_class", "Global rules per class"),
                                       ("global_f1_val", "Global explanation F1 (validation)")]):
        for c, (name, color, marker, _) in CONSTRUCTIONS.items():
            g = s[s.construction == c]
            ax.scatter(g.fit_per_class, g[col], color=color, marker=marker, s=22, label=name,
                       edgecolors="white", linewidths=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Fit series per class")
        ax.set_ylabel(label)
        style_axis(ax)
    big = s.loc[s.fit_per_class.idxmax(), "dataset"]
    g = s[(s.dataset == big) & (s.construction == "beam")].iloc[0]
    axes[1].annotate(big, (g.fit_per_class, g.global_f1_val), xytext=(-8, 4), textcoords="offset points",
                     ha="right", fontsize=8, color=INK_2)
    axes[0].legend(loc="upper left", fontsize=8)
    return fig


def plot_simplification(simp: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 0.17 * simp.dataset.nunique() + 1.2),
                             gridspec_kw={"width_ratios": [1, 1.4]}, constrained_layout=True)
    ax = axes[0]
    t = simp.groupby("n_literals")[["size_raw", "size_pruned"]].agg(["mean", "sem"])
    x = t.index.to_numpy()
    for col, label, color, dx in (("size_raw", "before", "#2a78d6", -0.15), ("size_pruned", "after", "#eb6834", 0.15)):
        ax.bar(x + dx, t[col]["mean"], width=0.28, yerr=t[col]["sem"], color=color, label=label, capsize=2,
               edgecolor="white", linewidth=1.0)
    for xi, (a, b) in zip(x, zip(t["size_raw"]["mean"], t["size_pruned"]["mean"])):
        ax.annotate(f"$-{100 * (1 - b / a):.0f}\\%$", (xi, a), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlabel("Literals in the rule")
    ax.set_ylabel("STL formula size (nodes)")
    ax.set_title(f"(a) size before and after simplification\n{len(simp):,} rules, all idempotent")
    ax.legend(loc="upper left")
    style_axis(ax)

    ax = axes[1]
    g = simp.assign(smaller=simp.size_pruned < simp.size_raw, changed=simp.raw_vs_pruned_val > 0)
    g = g.groupby("dataset")[["smaller", "changed"]].mean().sort_values("smaller")
    y = np.arange(len(g))
    ax.barh(y + 0.2, g.smaller, height=0.38, color="#2a78d6", label="smaller after simplification",
            edgecolor="white", linewidth=0.8)
    ax.barh(y - 0.2, g.changed, height=0.38, color="#eb6834",
            label="satisfied validation series change", edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, g.index)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of rules")
    ax.set_title(f"(b) per dataset; on fit series no rule changes ({int(simp.raw_vs_pruned_fit.sum())} series)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncols=2, fontsize=8)
    style_axis(ax)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default="results/ablation/explanations_v2")
    ap.add_argument("--out_dir", default="results/figures")
    ap.add_argument("--confidence", type=float, default=0.95, help="confidence of the per-dataset figures")
    args = ap.parse_args()
    res_dir, out_dir = Path(args.res_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, simp = load_variants(res_dir), load_simplification(res_dir)
    figures = {
        "expl_ablation_variants": plot_variants(df),
        "expl_ablation_paired": plot_paired(df, args.confidence),
        "expl_ablation_tradeoff": plot_tradeoff(df, args.confidence),
        "expl_ablation_global_rules": plot_global_rules(df, args.confidence),
        "expl_ablation_simplification": plot_simplification(simp),
    }
    for name, fig in figures.items():
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{name}.{ext}", dpi=200, facecolor="white")
        print(f"wrote {out_dir / name}.{{pdf,png}}")


if __name__ == "__main__":
    main()
