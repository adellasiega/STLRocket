#!/usr/bin/env python
"""Plot accuracy and timing (mean ± std) for a run_comparison.py result directory.

A comparison run writes one directory per dataset:

    <output_dir>/<dataset>_<timestamp>_<uid>/
        config.json
        results.csv      <- per-seed rows (the source of truth used here)
        summary.json

This script reads ``results.csv``, aggregates the per-seed runs into mean ± std
for both ``balanced_accuracy`` and ``time_total_s``, and saves two figures per
run directory:

    accuracy_<dataset>.png   balanced accuracy vs budget
    timing_<dataset>.png     wall-clock time vs budget (log y-axis)

Each figure draws one line per method.  ROCKET has no STL depth, so it is shown
as a single series; the STL heads (stl_linear, stl_tree) are split into one line
per ``depth``.  The shaded band / error bars show ± one standard deviation over
seeds.  Only rows with ``status == "ok"`` are aggregated.

Usage
-----
    # one run directory
    python scripts/plot_comparison.py results/comparison/EigenWorms_20260618_160028_d1241ee6

    # every run directory under a parent (e.g. the whole sweep)
    python scripts/plot_comparison.py results/comparison --glob

    # show instead of (or in addition to) saving
    python scripts/plot_comparison.py <run_dir> --show
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="tab10")

# Methods, in legend order, with the column that distinguishes their series.
# ROCKET has no depth; the STL heads are split by depth.
METHODS = ["rocket", "stl_linear", "stl_tree"]


def load_results(run_dir: Path) -> tuple[pd.DataFrame, str]:
    """Read results.csv from a run directory and return (ok_rows, dataset)."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no results.csv in {run_dir}")

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()

    # dataset name: prefer config.json, fall back to the column.
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        dataset = json.loads(cfg_path.read_text()).get("dataset", "")
    else:
        dataset = df["dataset"].iloc[0] if len(df) else run_dir.name

    return df, dataset


def aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean / std of ``value_col`` over seeds, per (method, depth, budget).

    Returns a tidy frame with columns: method, depth, budget, mean, std, n.
    ``depth`` is NaN for methods without one (rocket).
    """
    g = (
        df.groupby(["method", "depth", "budget"], dropna=False)[value_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    # std is NaN when a group has a single seed; treat it as 0 for plotting.
    g["std"] = g["std"].fillna(0.0)
    return g.sort_values(["method", "depth", "budget"])


def _series_label(method: str, depth) -> str:
    if method == "rocket" or pd.isna(depth):
        return method
    return f"{method} (d={int(depth)})"


def plot_metric(
    df: pd.DataFrame,
    value_col: str,
    dataset: str,
    ylabel: str,
    title: str,
    logy: bool,
    out_path: Path | None,
    show: bool,
) -> None:
    """One figure: ``value_col`` mean ± std vs budget, one line per series."""
    agg = aggregate(df, value_col)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Stable color per series across both figures.
    series_keys = []
    for method in METHODS:
        sub = agg[agg["method"] == method]
        for depth in sorted(sub["depth"].unique(), key=lambda d: (pd.isna(d), d)):
            series_keys.append((method, depth))
    palette = sns.color_palette("tab10", n_colors=max(len(series_keys), 1))

    for color, (method, depth) in zip(palette, series_keys):
        if pd.isna(depth):
            s = agg[(agg["method"] == method) & (agg["depth"].isna())]
        else:
            s = agg[(agg["method"] == method) & (agg["depth"] == depth)]
        s = s.sort_values("budget")
        if s.empty:
            continue

        x = s["budget"].to_numpy()
        mean = s["mean"].to_numpy()
        std = s["std"].to_numpy()

        ax.plot(x, mean, marker="o", color=color, label=_series_label(method, depth))
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("budget (kernels / formulae)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset} — {title}")
    ax.legend(title="method", fontsize=8)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"  saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def process_run_dir(run_dir: Path, show: bool) -> None:
    df, dataset = load_results(run_dir)
    if df.empty:
        print(f"  skip {run_dir.name}: no completed runs")
        return

    print(f"{run_dir.name} (dataset={dataset}, {len(df)} ok rows)")
    plot_metric(
        df, "balanced_accuracy", dataset,
        ylabel="balanced accuracy (mean ± std)",
        title="accuracy vs budget", logy=False,
        out_path=run_dir / f"accuracy_{dataset}.png", show=show,
    )
    plot_metric(
        df, "time_total_s", dataset,
        ylabel="total time [s] (mean ± std)",
        title="timing vs budget", logy=True,
        out_path=run_dir / f"timing_{dataset}.png", show=show,
    )


def find_run_dirs(root: Path, use_glob: bool) -> list[Path]:
    if not use_glob:
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "results.csv").exists())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path,
                   help="a run directory, or a parent dir with --glob")
    p.add_argument("--glob", action="store_true",
                   help="treat PATH as a parent and plot every run subdirectory")
    p.add_argument("--show", action="store_true", help="display figures interactively")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = find_run_dirs(args.path, args.glob)
    if not run_dirs:
        raise SystemExit(f"no run directories with results.csv under {args.path}")
    for run_dir in run_dirs:
        process_run_dir(run_dir, args.show)


if __name__ == "__main__":
    main()
