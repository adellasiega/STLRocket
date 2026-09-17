#!/usr/bin/env python
"""
LaTeX table of explanation quality, one row per dataset.

Reads ``results/explanations/summary.csv`` (written by ``run_explanations.py``)
and emits a booktabs tabular plus the matching CSV. Conventions follow
``make_table.py``: percentages in the compact ``97.0`` form, datasets down the
rows, a Mean row, and no escaping (UEA dataset names are alphanumeric).

Three table shapes:

    --style grouped  (default) local block (precision / coverage / depth) beside
                     a global block (F1 / depth), under spanning headers
    --style global   global explanation quality only: F1 / precision / coverage
                     on train and test
    --style full     every column, flat: accuracy, global train/test, local

Usage
-----
    python scripts/make_explanation_table.py
    python scripts/make_explanation_table.py --style full --out_dir tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# (csv column, header, percent-scaled, decimals)
#
# The default table is grouped: local-explanation quality on the left, global on
# the right, under a \cmidrule-separated pair of spanning headers. Local metrics
# are measured on the train set (that is what a local explanation is scored
# against); global F1 is the held-out test number.
LOCAL_GROUP = [
    ("mean_local_precision", "Precision", True, 1),
    ("mean_local_depth", "Depth", False, 1),
]
GLOBAL_GROUP = [
    ("global_macro_f1_test", "F1", True, 1),
    ("mean_global_depth", "Depth", False, 1),
]

# --style full keeps the older flat layout, with every column it had.
GLOBAL_COLS = [
    ("global_macro_f1_train", r"F1$_{\text{train}}$", True, 1),
    ("global_macro_f1_test", r"F1$_{\text{test}}$", True, 1),
    ("global_macro_precision_test", r"Prec$_{\text{test}}$", True, 1),
    ("global_macro_coverage_test", r"Cov$_{\text{test}}$", True, 1),
]
LOCAL_COLS = [
    ("mean_local_precision", r"Prec$_{\text{loc}}$", True, 1),
    ("mean_n_picks", r"$|\varphi|$", False, 2),
]
ACC_COL = ("balanced_accuracy", "Acc", True, 1)


def fmt(v: float, pct: bool, dec: int) -> str:
    if pd.isna(v):
        return "--"
    return f"{v * (100.0 if pct else 1.0):.{dec}f}"


def build_columns(style: str) -> list[tuple]:
    if style == "grouped":
        return LOCAL_GROUP + GLOBAL_GROUP
    if style == "global":
        return list(GLOBAL_COLS)
    return [ACC_COL] + GLOBAL_COLS + LOCAL_COLS


def to_latex_grouped(df: pd.DataFrame, bold_best: bool) -> str:
    """Local and global blocks side by side under spanning headers.

    Needs \\usepackage{booktabs}. The two \\cmidrule(lr) spans are what make the
    grouping readable: without them the reader cannot tell which "Depth" belongs
    to which half.
    """
    df = df.sort_values("dataset")
    cols = LOCAL_GROUP + GLOBAL_GROUP
    n_loc, n_glob = len(LOCAL_GROUP), len(GLOBAL_GROUP)

    best_ds = None
    if bold_best and not df["global_macro_f1_test"].isna().all():
        best_ds = df.loc[df["global_macro_f1_test"].idxmax(), "dataset"]

    lines = [
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        f"& \\multicolumn{{{n_loc}}}{{c}}{{Local explanations}}"
        f" & \\multicolumn{{{n_glob}}}{{c}}{{Global explanations}} \\\\",
        f"\\cmidrule(lr){{2-{1 + n_loc}}} \\cmidrule(lr){{{2 + n_loc}-{1 + n_loc + n_glob}}}",
        "Dataset & " + " & ".join(h for _, h, _, _ in cols) + " \\\\",
        "\\midrule",
    ]
    for r in df.itertuples():
        cells = []
        for key, _, pct, dec in cols:
            txt = fmt(getattr(r, key), pct, dec)
            if r.dataset == best_ds and key == "global_macro_f1_test":
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(f"{r.dataset} & " + " & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    lines.append("Mean & " + " & ".join(
        fmt(df[key].mean(), pct, dec) for key, _, pct, dec in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def to_latex(df: pd.DataFrame, cols: list[tuple], bold_best: bool) -> str:
    """Tabular body. Bolds the best test-F1 row, the headline number."""
    df = df.sort_values("dataset")

    best_ds = None
    if bold_best and not df["global_macro_f1_test"].isna().all():
        best_ds = df.loc[df["global_macro_f1_test"].idxmax(), "dataset"]

    lines = [
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        "Dataset & " + " & ".join(h for _, h, _, _ in cols) + " \\\\",
        "\\midrule",
    ]
    for r in df.itertuples():
        cells = []
        for key, _, pct, dec in cols:
            txt = fmt(getattr(r, key), pct, dec)
            # Bold the whole row of the best dataset would be noisy; bold only
            # the test-F1 cell that earns it.
            if r.dataset == best_ds and key == "global_macro_f1_test":
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(f"{r.dataset} & " + " & ".join(cells) + " \\\\")

    # Mean over datasets, from the numbers rather than the formatted cells.
    lines.append("\\midrule")
    means = [fmt(df[key].mean(), pct, dec) for key, _, pct, dec in cols]
    lines.append("Mean & " + " & ".join(means) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", type=Path,
                   default=Path("results/explanations/summary.csv"))
    p.add_argument("--out_dir", type=Path, default=Path("tables"))
    p.add_argument("--style", choices=("grouped", "global", "full"), default="grouped")
    p.add_argument("--no_bold", action="store_true",
                   help="do not bold the best test-F1 cell")
    p.add_argument("--fname", default=None,
                   help="basename for the .tex/.csv (default: explanations_<style>)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary.exists():
        raise SystemExit(f"{args.summary} not found -- run scripts/run_explanations.py first.")

    df = pd.read_csv(args.summary)
    cols = build_columns(args.style)

    missing = [k for k, _, _, _ in cols if k not in df.columns]
    if missing:
        raise SystemExit(f"{args.summary} is missing columns: {', '.join(missing)}")

    if args.style == "grouped":
        tex = to_latex_grouped(df, bold_best=not args.no_bold)
    else:
        tex = to_latex(df, cols, bold_best=not args.no_bold)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.fname or f"explanations_{args.style}"
    tex_path = args.out_dir / f"{stem}.tex"
    csv_path = args.out_dir / f"{stem}.csv"

    tex_path.write_text(tex + "\n")

    # Companion CSV: the same cells, so the numbers in the paper and the numbers
    # you can sort/inspect never drift apart.
    out = df.sort_values("dataset")[["dataset"] + [k for k, _, _, _ in cols]].copy()
    for key, _, pct, dec in cols:
        out[key] = out[key].map(lambda v: fmt(v, pct, dec))
    out.to_csv(csv_path, index=False)

    print(tex)
    print(f"\nWrote {tex_path}\nWrote {csv_path}")


if __name__ == "__main__":
    main()
