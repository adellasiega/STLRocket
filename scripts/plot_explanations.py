#!/usr/bin/env python
"""
Plots the explanation metrics of scripts/run_explanations.py per dataset, against the
model's balanced accuracy on test. Means over each dataset's complete runs.

    python scripts/plot_explanations.py [--res_dir results/explanations_balanced] [--out_dir results/figures]

Writes, as PDF and PNG:
  explanations_per_dataset_{test,train}
                             one row per dataset (both sorted by model accuracy on test):
                             (a) precision and (b) recall (= coverage, w.r.t. the model's
                             predictions) on the split of the instance rules
                             of its series and of the global explanations; the model's balanced
                             accuracy on the split and chance (1 / n_classes) are printed next
                             to the dataset names;
                             (c) mean precision of the instance rules vs mean F1 (over classes)
                             of the global explanations on the split;
                             (d) size (nodes of the simplified STL formula) of an instance rule
                             and of a class's global explanation (all its rules), averaged over
                             series and classes respectively
  precision_vs_accuracy      instance-rule test precision against model accuracy, one point
                             per dataset, with the y = x diagonal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_explanations import flatten

MODEL, RULES, GLOBAL = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ABBREV = {
    "ArticularyWordRecognition": "AWR", "AtrialFibrillation": "AF", "BasicMotions": "BM", "Cricket": "CR",
    "DuckDuckGeese": "DDG", "EigenWorms": "EW", "Epilepsy": "EP", "EthanolConcentration": "EC", "ERing": "ER",
    "FaceDetection": "FD", "FingerMovements": "FM", "HandMovementDirection": "HMD", "Handwriting": "HW",
    "Heartbeat": "HB", "Libras": "LIB", "LSST": "LSST", "MotorImagery": "MI", "NATOPS": "NATO",
    "PEMS-SF": "PEMS", "PenDigits": "PD", "PhonemeSpectra": "PS", "RacketSports": "RS",
    "SelfRegulationSCP1": "SCP1", "SelfRegulationSCP2": "SCP2", "StandWalkJump": "SWJ",
    "UWaveGestureLibrary": "UW",
}

plt.rcParams.update({
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "pdf.fonttype": 42,
})


def load(res_dir: str) -> pd.DataFrame:
    """Per-dataset means over complete runs. For each split s: model balanced accuracy, and the
    precision, coverage and size of the rules of the series of s (rules_s) evaluated on s, and of
    the global explanations evaluated on s. Rows sorted by model accuracy on test."""
    runs = []
    for p in Path(res_dir).glob("*/run*/summary.json"):
        s = json.loads(p.read_text())
        r = flatten(s)
        gm = pd.read_csv(p.parent / "global_metrics.csv")
        r["g_size"] = gm[(gm.version == "simplified") & (gm.evaluated_on == "train")]["size"].mean()
        tr = s["rules_train"]
        r.update(bacc_train=s["balanced_accuracy"]["train"], prec_train=tr["pooled_prec_train"],
                 cov_train=tr["mean_coverage_train"], size_train=tr["mean_size"],
                 g_prec_train=s["global"]["train"]["macro_precision"],
                 g_cov_train=s["global"]["train"]["macro_coverage"], size_test=r["size"])
        for sp in ("train", "test"):
            # mean (not pooled) precision of the instance rules of the series of sp, on sp
            rules = pd.read_csv(p.parent / f"rules_{sp}.csv")
            r[f"lprec_{sp}"] = rules[rules.n_literals > 0][f"prec_{sp}"].mean()
            # F1 of each class's global explanation on sp (0 when nothing satisfies it), mean over classes
            g = gm[(gm.version == "simplified") & (gm.evaluated_on == sp)]
            P, R = g.precision.fillna(0), g.coverage
            r[f"g_f1_{sp}"] = (2 * P * R / (P + R)).fillna(0).mean()
        runs.append(r)
    df = pd.DataFrame(runs)
    df = df[df.n_selected > 0]  # an empty model has no rules to score
    g = df.groupby("dataset")
    return g.mean(numeric_only=True).sort_values("bacc_test")


def dot_panel(ax, t: pd.DataFrame, series: list, title: str, xmax: float = 1.0) -> None:
    y = np.arange(len(t))
    for i in y:  # thin guide joining the two explanations of each row
        vals = [t[c].iloc[i] for c, _, _ in series]
        ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    for col, color, label in series:
        ax.scatter(t[col], y, s=42, color=color, edgecolor="white", lw=1, zorder=3, label=label)
    if xmax == 1.0:
        ax.set_xlim(0, 1.02)
        ax.set_xticks(np.linspace(0, 1, 6))
    else:  # formula size: log scale, plain tick labels
        ax.set_xscale("log")
        ticks = [t for t in (2, 5, 10, 20, 50, 100, 200) if t <= xmax * 1.5]
        ax.set_xlim(ticks[0] * 0.9, xmax)
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["left", "bottom"]].set_visible(True)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)


def per_dataset(t: pd.DataFrame, split: str, out: Path) -> None:
    fig, (a, b, d, c) = plt.subplots(1, 4, figsize=(17, 0.28 * len(t) + 1.6), sharey=True,
                                     gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    R, G = "Instance rules", "Global explanations"
    dot_panel(a, t, [(f"prec_{split}", RULES, R), (f"g_prec_{split}", GLOBAL, G)],
              f"Average precision on {split}")
    dot_panel(b, t, [(f"cov_{split}", RULES, R), (f"g_cov_{split}", GLOBAL, G)],
              f"Average recall on {split}")
    dot_panel(d, t, [(f"lprec_{split}", RULES, R), (f"g_f1_{split}", GLOBAL, G)],
              f"Local precision vs global F1 on {split}")
    dot_panel(c, t, [(f"size_{split}", RULES, R), ("g_size", GLOBAL, G)],
              "Average formula size (nodes)", xmax=1.2 * t[["size_train", "size_test", "g_size"]].max().max())
    # Model accuracy and chance as a small table between the dataset names and the first panel
    a.set_yticks(np.arange(len(t)), t.index)
    a.tick_params(axis="y", pad=92)
    cols = [(f"Bal. acc.\n({split})", t[f"bacc_{split}"], -50), ("Chance", 1 / t.n_classes, -10)]
    for head, vals, dx in cols:
        a.annotate(head, (0, 1), xycoords="axes fraction", xytext=(dx, 4), textcoords="offset points",
                   ha="right", va="bottom", fontsize=8, color=INK_2)
        for i, v in enumerate(vals):
            a.annotate(f"{v:.2f}", (0, i), xycoords=("axes fraction", "data"), xytext=(dx, 0),
                       textcoords="offset points", ha="right", va="center", color=INK_2)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3)
    save(fig, out / f"explanations_per_dataset_{split}")


def scatter(t: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.12, 0.06, "y = x", color=INK_2, fontsize=8)
    ax.scatter(t.bacc_test, t.prec_test, s=46, color=RULES, edgecolor="white", lw=1, zorder=3)
    # Abbreviated labels, pushed apart vertically where neighbouring points would overlap
    placed = []
    for d, r in t.sort_values("prec_test").iterrows():
        y = r.prec_test
        while any(abs(y - py) < 0.022 and abs(r.bacc_test - px) < 0.17 for px, py in placed):
            y += 0.022
        placed.append((r.bacc_test, y))
        ax.annotate(ABBREV.get(d, d), (r.bacc_test, r.prec_test), xytext=(r.bacc_test + 0.015, y - 0.008),
                    fontsize=7, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8) if y != r.prec_test else None)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model balanced accuracy (test)")
    ax.set_ylabel("Instance-rule precision (test)")
    ax.spines["left"].set_visible(True)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=INK_2)
    rho = t[["bacc_test", "prec_test"]].corr(method="spearman").iloc[0, 1]
    ax.set_title(f"Rule precision vs model accuracy (Spearman {rho:.2f})", loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, out / "precision_vs_accuracy")


def save(fig, stem: Path) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res_dir", default="results/explanations_balanced")
    p.add_argument("--out_dir", default="results/figures")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = load(args.res_dir)
    for split in ("test", "train"):
        per_dataset(t, split, out)
    scatter(t, out)


if __name__ == "__main__":
    main()
