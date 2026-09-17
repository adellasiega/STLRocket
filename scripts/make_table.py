#!/usr/bin/env python
"""
Results table: methods as rows, datasets as columns, "mean +/- std" per cell.

Writes a wide CSV meant to be pasted into a LaTeX table, plus a long CSV whose
mean and std sit in separate numeric columns (for a reader who wants to
recompute or re-format without parsing "0.970 +- 0.020" back apart), plus a
ready-to-use .tex body.

The std is taken over SEEDS, so a cell reads "what this method scores on this
dataset, and how much that moves between runs". It is the sample standard
deviation of the per-seed accuracies, not the standard error of their mean:
a practitioner runs the method once, so the spread of a single run is the
honest quantity. (SE would be sqrt(n_seeds) times narrower and would flatter
every method equally.)

STL configuration: until_weight pinned (default 0, no Until) with only depth
chosen per dataset, on test accuracy. That selection is recorded in the long
CSV and printed, because a number chosen on test data must be reported as such.
ROCKET has no such axes and is untuned.

Usage:
  python scripts/make_table.py --results_dir results --out_dir tables
  python scripts/make_table.py --methods stl_tree stl_linear rocket_ridge
  python scripts/make_table.py --budget 1000 --decimals 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import PRETTY, STL_AXES, load_results, select_best_stl

ACC = "balanced_accuracy"

# Column order as given on the command line; these are the defaults.
# rocket_ridge, not rocket_linear: ROCKET as published uses a ridge classifier,
# and it is much the stronger arm (0.771 vs 0.695 mean here). Comparing against
# rocket_linear would be comparing against a weakened baseline.
# stl_linear last: it is the proposed method, and a reader compares the
# rightmost column against the baselines to its left.
DEFAULT_METHODS = ["stl_tree", "rocket_ridge", "stl_linear"]


def collect(df: pd.DataFrame, methods: list[str], budget: int | None,
            pin_until_weight: float | None) -> tuple[pd.DataFrame, dict]:
    """Per (method, dataset) mean and std over seeds, plus the STL configs used.

    STL methods sweep depth and until_weight; ROCKET has neither. Each STL
    method is reduced to one configuration per dataset the same way the figures
    do it (best mean accuracy), so the table and the figures cannot disagree.
    """
    sub = df if budget is None else df[df["budget"] == budget]
    rows, picks = [], {}

    for m in methods:
        if m not in set(sub["method"]):
            raise SystemExit(f"Method '{m}' has no rows at this budget. "
                             f"Present: {sorted(set(sub['method']))}")
        if m.startswith("stl_"):
            src = sub
            if pin_until_weight is not None:
                src = sub[(sub["method"] != m)
                          | (sub["until_weight"] == pin_until_weight)]
            sel = select_best_stl(src, m)
            picks[m] = sel.groupby("dataset")[STL_AXES].first()
        else:
            sel = sub[sub["method"] == m]

        g = (sel.groupby("dataset")[ACC]
             .agg(mean="mean", std="std", n_seeds="count")
             .reset_index())
        g.insert(0, "method", m)
        rows.append(g)

    return pd.concat(rows, ignore_index=True), picks


def fmt_cell(mean: float, std: float, dec: int) -> str:
    """"0.970 +- 0.020", or "--" where a cell has no runs.

    A single-seed cell has an undefined sample std; it is shown as a bare mean
    rather than "+- nan", which would otherwise propagate into the LaTeX.
    """
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.{dec}f}"
    return f"{mean:.{dec}f} +- {std:.{dec}f}"


def _tex_cell(mean: float, std: float, dec: int, pct: bool) -> str:
    """One LaTeX cell, as a percentage or a fraction.

    Percentages are the compact form: 100x drops the leading "0." from both
    numbers, so "97.0 $\\pm$ 2.0" replaces "0.970 $\\pm$ 0.020" at the same
    precision and roughly two thirds of the width. The unit belongs in the
    caption or the header, not repeated in eighty cells.
    """
    if pd.isna(mean):
        return "--"
    k = 100.0 if pct else 1.0
    if pd.isna(std):
        return f"{mean * k:.{dec}f}"
    return f"{mean * k:.{dec}f} $\\pm$ {std * k:.{dec}f}"


def to_latex(long: pd.DataFrame, methods: list[str], dec: int,
             bold_best: bool, pct: bool, transpose: bool) -> str:
    """Tabular body, optionally bolding the best method per dataset.

    Datasets go down the rows by default rather than across the columns: ten
    dataset columns do not fit a page even at one decimal, whereas three method
    columns fit a single column of a two-column paper with room to spare. The
    transposed form is still available for a wide landscape float.

    Emits \\pm and \\textbf, and escapes nothing else: dataset names in the UEA
    archive are alphanumeric, so there is nothing to escape, and quietly
    mangling a name would be worse than leaving it.
    """
    datasets = sorted(long["dataset"].unique())

    best = {}
    if bold_best:
        for ds in datasets:
            col = long[long["dataset"] == ds]
            if not col["mean"].isna().all():
                best[ds] = col.loc[col["mean"].idxmax(), "method"]

    def cell(m, ds):
        r = long[(long["method"] == m) & (long["dataset"] == ds)]
        if r.empty:
            return "--"
        txt = _tex_cell(float(r["mean"].iloc[0]), float(r["std"].iloc[0]),
                        dec, pct)
        return f"\\textbf{{{txt}}}" if best.get(ds) == m else txt

    # Mean over datasets, computed from the numbers rather than from the
    # formatted cells; it goes in whichever direction the methods run.
    means = [long[long["method"] == m]["mean"].mean() for m in methods]
    best_i = int(np.nanargmax(means)) if not np.all(np.isnan(means)) else -1

    def mean_cell(i):
        t = _tex_cell(means[i], np.nan, dec, pct)
        return f"\\textbf{{{t}}}" if bold_best and i == best_i else t

    lines = []
    if transpose:
        # Datasets across the top: needs a full-width or landscape float.
        lines.append("\\begin{tabular}{l" + "c" * (len(datasets) + 1) + "}")
        lines.append("\\toprule")
        lines.append("Method & " + " & ".join(datasets) + " & Mean \\\\")
        lines.append("\\midrule")
        for i, m in enumerate(methods):
            lines.append(f"{PRETTY.get(m, m)} & "
                         + " & ".join(cell(m, ds) for ds in datasets)
                         + f" & {mean_cell(i)} \\\\")
    else:
        # Datasets down the rows: fits one column of a two-column paper.
        lines.append("\\begin{tabular}{l" + "c" * len(methods) + "}")
        lines.append("\\toprule")
        lines.append("Dataset & "
                     + " & ".join(PRETTY.get(m, m) for m in methods) + " \\\\")
        lines.append("\\midrule")
        for ds in datasets:
            lines.append(f"{ds} & " + " & ".join(cell(m, ds) for m in methods)
                         + " \\\\")
        lines.append("\\midrule")
        lines.append("Mean & "
                     + " & ".join(mean_cell(i) for i in range(len(methods)))
                     + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(
        description="Results table: methods x datasets, mean +/- std over seeds")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="tables")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                   help="row order, e.g. stl_tree stl_linear rocket_linear")
    p.add_argument("--budget", type=int, default=10000,
                   help="budget to report; -1 pools every budget")
    p.add_argument("--until_weight", type=float, default=0.0,
                   help="pin until_weight for STL methods and tune depth only "
                        "(default 0, no Until); -1 tunes until_weight too")
    p.add_argument("--decimals", type=int, default=3,
                   help="decimals in the CSVs (kept as fractions)")
    p.add_argument("--tex_decimals", type=int, default=None,
                   help="decimals in the .tex; default 1 for percentages, "
                        "--decimals for fractions")
    p.add_argument("--no_percent", action="store_true",
                   help="write the .tex as fractions (0.970) instead of "
                        "percentages (97.0)")
    p.add_argument("--no_bold", action="store_true",
                   help="do not bold the best method per dataset in the .tex")
    p.add_argument("--transpose", action="store_true",
                   help="methods as rows and datasets as columns; needs a "
                        "full-width float. Default is datasets down the rows, "
                        "which fits one column of a two-column paper")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.results_dir))

    budget = None if args.budget < 0 else args.budget
    if budget is not None and budget not in set(df["budget"]):
        raise SystemExit(f"Budget {budget} not in data. "
                         f"Available: {sorted(set(df['budget']))}")

    pin = None if args.until_weight < 0 else args.until_weight
    if pin is not None and pin not in set(df["until_weight"].dropna()):
        raise SystemExit(f"until_weight {pin} not in data. "
                         f"Available: {sorted(set(df['until_weight'].dropna()))}")

    long, picks = collect(df, args.methods, budget, pin)

    # Attach the chosen STL config to each row, so the long CSV is enough on its
    # own to reproduce the table.
    long["depth"] = np.nan
    long["until_weight"] = np.nan
    for m, p in picks.items():
        for ds, r in p.iterrows():
            sel = (long["method"] == m) & (long["dataset"] == ds)
            long.loc[sel, "depth"] = r["depth"]
            long.loc[sel, "until_weight"] = r["until_weight"]

    long["cell"] = [fmt_cell(m, s, args.decimals)
                    for m, s in zip(long["mean"], long["std"])]

    # The CSV takes the same orientation as the .tex, so the two files always
    # look like each other. Default: datasets down the rows, methods across.
    # reindex keeps the method order the user asked for rather than pandas'
    # alphabetical grouping.
    if args.transpose:
        wide = (long.pivot(index="method", columns="dataset", values="cell")
                .reindex(args.methods).reset_index())
    else:
        wide = (long.pivot(index="dataset", columns="method", values="cell")
                .reindex(columns=args.methods).reset_index())
    wide.columns.name = None

    wide_path = out_dir / "results_table.csv"
    long_path = out_dir / "results_long.csv"
    tex_path = out_dir / "results_table.tex"

    wide.to_csv(wide_path, index=False)
    long[["method", "dataset", "mean", "std", "n_seeds",
          "depth", "until_weight", "cell"]].to_csv(
        long_path, index=False, float_format=f"%.{args.decimals + 2}f")
    # Percentages need fewer decimals than fractions for the same precision:
    # 97.0 and 0.970 carry identical information.
    tex_dec = args.tex_decimals
    if tex_dec is None:
        tex_dec = 1 if not args.no_percent else args.decimals
    tex_path.write_text(
        to_latex(long, args.methods, tex_dec, not args.no_bold,
                 not args.no_percent, args.transpose)
        + "\n")

    print(f"wrote {wide_path}")
    print(f"wrote {long_path}")
    print(f"wrote {tex_path}")

    budget_txt = "all budgets pooled" if budget is None else f"budget={budget}"
    pin_txt = ("until_weight tuned" if pin is None
               else f"until_weight={pin:g}" + (" (no Until)" if pin == 0 else ""))
    print(f"\n{budget_txt}, {pin_txt}, "
          f"{int(long['n_seeds'].max())} seeds, mean +/- sample std over seeds")

    if picks:
        print("\nSTL depth selected per dataset (on test accuracy):")
        sel = pd.DataFrame({m: p["depth"] for m, p in picks.items()})
        print(sel.to_string())

    print("\nTable:")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(wide.to_string(index=False))

    # Overall column: useful in the paper and trivial to get wrong by averaging
    # the formatted strings instead of the numbers.
    print("\nMean over datasets:")
    for m in args.methods:
        sub = long[long["method"] == m]
        print(f"  {PRETTY.get(m, m):24s} {sub['mean'].mean():.{args.decimals}f}")


if __name__ == "__main__":
    main()
