#!/usr/bin/env python
"""
Cut point 0 (lambda with the best CV score) against cut point 1 (the sparsest lambda within
one standard error), from two runs of scripts/run_explanation_ablation.py that differ only
in --cut_point (same datasets, runs and seeds, hence the same formula bank, splits and CV
folds; one variant each, e.g. ordered_scan, gxi, 0.95):

    python scripts/compare_cut_point.py --cut1 results/ablation/cut1 --cut0 results/ablation/cut0

Runs are paired by (dataset, run). Per metric: the mean over runs of each dataset for both
cut points, then over datasets the mean of each, the mean difference (cut 0 - cut 1), the
wins and losses of cut 0, and the p-value of the two-sided Wilcoxon signed-rank test over
datasets. Classifier metrics use every run; explanation metrics only the runs where both
models are non-empty (an empty model has nothing to explain). Prints the table and writes it
as LaTeX to --out (default results/figures/cut_point_table.tex), with the per-dataset means
in a CSV next to it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# (column of .variants.csv, label, explanation metric)
METRICS = [
    ("bacc_val", "Validation balanced accuracy", False),
    ("n_selected", "Selected formulae", False),
    ("empty", "Empty models", False),
    ("val_pooled_prec_val", "Instance precision", True),
    ("val_coverage_val", "Instance coverage", True),
    ("val_length", "Instance length", True),
    ("val_attr_share", "Instance share", True),
    ("global_f1_val", "Global $F_1$", True),
    ("global_rules_per_class", "Global disjuncts per class", True),
    ("time", "Explanation time (s)", True),
]


def load(res_dir: Path) -> pd.DataFrame:
    files = sorted(res_dir.glob("*.variants.csv"))
    if not files:
        raise SystemExit(f"no .variants.csv in {res_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    variants = df[["construction", "candidates", "confidence"]].drop_duplicates()
    if len(variants) != 1:
        raise SystemExit(f"{res_dir}: expected one variant, found\n{variants.to_string(index=False)}")
    df["empty"] = (df.n_selected == 0).astype(float)
    df["time"] = df.time_fit_s + df.time_val_s
    return df.set_index(["dataset", "run"])


def compare(d1: pd.DataFrame, d0: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = d1.index.intersection(d0.index)
    d1, d0 = d1.loc[keys], d0.loc[keys]
    both = (d1.n_selected > 0) & (d0.n_selected > 0)
    rows, per_ds = [], {}
    for col, label, expl in METRICS:
        m = both if expl else pd.Series(True, index=keys)
        a = d1.loc[m, col].groupby("dataset").mean()
        b = d0.loc[m, col].groupby("dataset").mean()
        ok = a.notna() & b.notna()
        a, b = a[ok], b[ok]
        diff = b - a
        p = wilcoxon(b, a).pvalue if (diff != 0).any() else np.nan
        rows.append({"metric": label, "datasets": len(a), "cut1": a.mean(), "cut0": b.mean(),
                     "diff": diff.mean(), "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "p": p})
        per_ds[(col, "cut1")], per_ds[(col, "cut0")] = a, b
    return pd.DataFrame(rows), pd.DataFrame(per_ds)


def to_latex(t: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Metric & Datasets & Cut point 1 & Cut point 0 & Difference & W/L & $p$ \\", r"\midrule"]
    for r in t.itertuples():
        p = "--" if np.isnan(r.p) else (f"{r.p:.3f}" if r.p >= 0.001 else "$<0.001$")
        fmt = ".1f" if abs(r.cut1) >= 10 else ".3f"
        lines.append(f"{r.metric} & {r.datasets} & {r.cut1:{fmt}} & {r.cut0:{fmt}} & {r.diff:+{fmt}} & "
                     f"{r.wins}/{r.losses} & {p} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cut1", type=Path, default=Path("results/ablation/cut1"))
    ap.add_argument("--cut0", type=Path, default=Path("results/ablation/cut0"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/cut_point_table.tex"))
    args = ap.parse_args()
    d1, d0 = load(args.cut1), load(args.cut0)
    missing = d1.index.symmetric_difference(d0.index)
    if len(missing):
        print(f"{len(missing)} (dataset, run) pairs present in only one directory, left out")
    t, per_ds = compare(d1, d0)
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(t.to_string(index=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(to_latex(t))
    per_ds.to_csv(args.out.with_suffix(".csv"))
    print(f"written {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
