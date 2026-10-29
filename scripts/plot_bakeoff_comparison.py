#!/usr/bin/env python
"""
Comparison with the classifiers of the multivariate bake off (Ruiz et al., 2021) on the 26
equal-length UEA datasets.

The bake off reports test accuracy (not balanced accuracy), published on timeseriesclassification.com
and loaded with aeon (load_classification_bake_off_2021_results). Its first resample is the
official train/test split, the one of results/comparison_balanced, so the comparison uses it:
one run per bake-off classifier, the mean over our runs for ours. The mean over the bake off's 30
resamples is written alongside for reference only, since those are other splits.

Writes to --out_dir:
  bakeoff_accuracy.csv     test accuracy, datasets x classifiers (ours and the bake off's)
  bakeoff_summary.tex      mean accuracy, mean rank, wins/ties/losses and Holm-corrected Wilcoxon
                           p-value of each classifier against STLRocket
  bakeoff_cd.{pdf,png}     critical difference diagram (Wilcoxon signed-rank with Holm, alpha 0.05)
  bakeoff_stlrocket_summary.tex, bakeoff_stlrocket_cd.{pdf,png}
                           the same for STLRocket and the bake-off classifiers alone (the comparison
                           of the paper); the table has no p-values, significance is read on the
                           CD diagram

    python scripts/plot_bakeoff_comparison.py [--res_dir results/comparison_balanced] [--out_dir results/figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_comparison import ALPHA, METHOD_STYLE, REFERENCE, holm, load_rows

BAKEOFF_LABELS = {"HIVE-COTEv1": "HIVE-COTE 1.0", "DTW_D": r"DTW$_D$", "DTW_I": r"DTW$_I$"}


def bakeoff(num_resamples: int, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache, index_col=0)
    from aeon.benchmarking.published_results import load_classification_bake_off_2021_results
    res, datasets, classifiers = load_classification_bake_off_2021_results(num_resamples=num_resamples,
                                                                           as_array=True)
    df = pd.DataFrame(res, index=datasets, columns=classifiers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df


def summary(acc: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    ranks = acc.rank(axis=1, ascending=False).mean()
    rows = []
    for m in acc.columns:
        d = acc[m] - acc[REFERENCE]
        p = wilcoxon(acc[m], acc[REFERENCE]).pvalue if m != REFERENCE and np.any(d != 0) else np.nan
        rows.append({"method": m, "label": labels[m], "accuracy": acc[m].mean(), "rank": ranks[m],
                     "wins": int((d > 1e-9).sum()), "ties": int((d.abs() <= 1e-9).sum()),
                     "losses": int((d < -1e-9).sum()), "p": p})
    t = pd.DataFrame(rows).set_index("method").sort_values("rank")
    others = t.index != REFERENCE
    t.loc[others, "p_holm"] = holm(t.loc[others, "p"].to_numpy())
    return t


def write_summary(t: pd.DataFrame, n: int, path: Path, p_values: bool = True) -> None:
    cols, head = ("lccccc", r" & $p$ (Holm)") if p_values else ("lccc", "")
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             rf"Classifier & Accuracy & Rank & W / T / L vs STLRocket{head} \\", r"\midrule"]
    for m, r in t.iterrows():
        if m == REFERENCE:
            cmp_, p = "--", "--"
        else:
            # wins/ties/losses of the classifier against STLRocket
            cmp_ = f"{r.wins} / {r.ties} / {r.losses}"
            p = f"{r.p_holm:.3f}" if r.p_holm >= 0.001 else "$<0.001$"
            p = rf"\textbf{{{p}}}" if r.p_holm < ALPHA else p
        name = rf"\textbf{{{r.label}}}" if m == REFERENCE else r.label
        lines.append(f"{name} & {r.accuracy:.3f} & {r['rank']:.2f} & {cmp_}"
                     + (f" & {p}" if p_values else "") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"{path} ({n} datasets)")


def plot_cd(acc: pd.DataFrame, t: pd.DataFrame, path: Path) -> None:
    from aeon.visualisation import plot_critical_difference
    order = list(t.index)
    fig, _ = plot_critical_difference(acc[order].to_numpy(), list(t.label),
                                      test="wilcoxon", correction="holm", alpha=ALPHA, width=7, textspace=1.8)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor="white", bbox_inches="tight")
    print(path.with_suffix(".pdf"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res_dir", default="results/comparison_balanced")
    p.add_argument("--out_dir", default="results/figures")
    p.add_argument("--cache_dir", default="results/bakeoff2021", help="downloaded bake-off results")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ours = load_rows(Path(args.res_dir)).groupby(["dataset", "method"]).acc_test.mean().unstack()
    ours = ours[[m for m in METHOD_STYLE if m in ours.columns]]
    theirs = bakeoff(1, Path(args.cache_dir) / "accuracy_resample0.csv")
    theirs30 = bakeoff(30, Path(args.cache_dir) / "accuracy_mean30.csv")
    missing = sorted(set(theirs.index) ^ set(ours.index))
    if missing:
        print("datasets not in both:", missing)
    datasets = sorted(set(theirs.index) & set(ours.index), key=str.lower)
    acc = pd.concat([ours.loc[datasets], theirs.loc[datasets]], axis=1).dropna(axis=1)
    labels = {m: METHOD_STYLE[m][0] if m in METHOD_STYLE else BAKEOFF_LABELS.get(m, m) for m in acc.columns}
    acc.to_csv(out / "bakeoff_accuracy.csv")

    t = summary(acc, labels)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())
    print("\nfor reference, mean accuracy of the bake off over its 30 resamples:")
    print(theirs30.loc[datasets].mean().sort_values(ascending=False).round(3).to_string())
    write_summary(t, len(datasets), out / "bakeoff_summary.tex")
    plot_cd(acc, t, out / "bakeoff_cd")

    # STLRocket and the bake-off classifiers alone, without our other baselines
    sub = acc[[REFERENCE] + [m for m in theirs.columns if m in acc.columns]]
    print(f"\nSTLRocket and the bake off: Friedman p = {friedmanchisquare(*sub.T.to_numpy()).pvalue:.2e}")
    t = summary(sub, labels)
    print(t.round(3).to_string())
    write_summary(t, len(datasets), out / "bakeoff_stlrocket_summary.tex", p_values=False)
    plot_cd(sub, t, out / "bakeoff_stlrocket_cd")


if __name__ == "__main__":
    main()
