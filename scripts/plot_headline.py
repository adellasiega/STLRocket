#!/usr/bin/env python
"""
Headline comparison figure: STLRocket (STL features + glmnet) vs ROCKET.

The sweep spans three axes -- 10 datasets x 4 budgets x 6 STL (depth,
until_weight) configs -- which no single panel can show. This script collapses
them into one scatter that answers "does STLRocket beat ROCKET?" at a glance,
plus companions that carry the caveats.

Why the collapse has to be done carefully: at b=10000 STL accuracy spans a
median of 0.061 across its 6 configs while the STL-vs-ROCKET gap is ~0.010, so
how STL is summarised decides the headline. We take the best (depth,
until_weight) PER DATASET (not per budget -- the per-budget argmax is noise) and
label the result as tuned, showing the untuned d=1/uw=0 arm alongside.

Seed variability is shown rather than hidden: each point is a mean over 10
seeds whose standard error (~0.009) is the same size as the effect being
claimed, so every point carries +/-1 SE bars on both axes.

Figures written to --out_dir:
  1. headline_scatter.png       the main figure: 1 point per dataset, SE bars
  2. headline_scatter_aeon.png  same comparison in aeon's standard style
  3. cd_diagram.png             average-rank diagram over all four methods
  4. scaling_curve.png          accuracy vs budget, one line per method
  5. headline_summary.csv       means, stds, seed counts, Wilcoxon

Usage:
  python scripts/plot_headline.py --results_dir results --out_dir figures_headline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless on the cluster; must precede pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# Reuse the loader and the config selection rather than reimplementing them:
# load_results already drops non-ok rows, skips empty CSVs from killed tasks and
# de-duplicates re-run configs, all of which this results directory needs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import PRETTY, STL_AXES, load_results, select_best_stl

sns.set_theme(style="whitegrid", palette="tab10")

# Order matters only for the CD diagram's tie-breaking; all four are required
# there because aeon's Friedman test rejects fewer than three methods.
ALL_METHODS = ["stl_linear", "stl_tree", "rocket_ridge", "rocket_linear"]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def per_dataset_stats(df: pd.DataFrame, budget: int | None) -> pd.DataFrame:
    """Mean, std and seed count per (method, dataset), at one budget.

    The std must be taken over the raw per-seed rows. Averaging seeds away first
    and then taking a std would measure the wrong thing entirely (spread across
    configs, not across seeds) and collapse every error bar to a hairline.
    """
    sub = df if budget is None else df[df["budget"] == budget]
    return (sub.groupby(["method", "dataset"])["balanced_accuracy"]
               .agg(mean="mean", std="std", n_seeds="count")
               .reset_index())


def stl_arms(df: pd.DataFrame, budget: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The two STL arms plus the record of which config was chosen.

    Returns (tuned, untuned, picks). The tuned arm selects the best (depth,
    until_weight) per dataset over the whole sweep; the untuned arm pins the
    d=1, uw=0 default so the reader sees both bounds.
    """
    best = select_best_stl(df, "stl_linear")
    picks = best.groupby("dataset")[STL_AXES].first()

    tuned = per_dataset_stats(best, budget)
    tuned = tuned[tuned["method"] == "stl_linear"].set_index("dataset")

    default = df[(df["method"] == "stl_linear")
                 & (df["depth"] == 1) & (df["until_weight"] == 0.0)]
    untuned = per_dataset_stats(default, budget)
    untuned = untuned[untuned["method"] == "stl_linear"].set_index("dataset")

    return tuned, untuned, picks


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Paired signed-rank p-value, guarding the degenerate cases scipy rejects."""
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 5 or not np.any(d != 0):
        return float("nan")
    try:
        return float(wilcoxon(d, zero_method="zsplit").pvalue)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_headline_scatter(tuned, untuned, rocket, out_dir: Path,
                          budget_label: str, baseline: str) -> dict:
    """One point per dataset, with +/-1 SE bars on both axes.

    Drawn directly rather than via aeon's plot_pairwise_scatter because that
    helper takes bare score vectors with no way to pass error bars, and the SE
    here is the same magnitude as the effect -- omitting it would assert a
    precision the data does not support.
    """
    common = sorted(set(tuned.index) & set(rocket.index))
    if not common:
        raise SystemExit("No datasets have both STL and ROCKET rows.")

    x = tuned.loc[common, "mean"].to_numpy()
    y = rocket.loc[common, "mean"].to_numpy()
    # Standard error of a mean over n seeds; std alone would overstate the
    # uncertainty of the quantity actually plotted by roughly sqrt(10).
    xe = (tuned.loc[common, "std"] / np.sqrt(tuned.loc[common, "n_seeds"])).to_numpy()
    ye = (rocket.loc[common, "std"] / np.sqrt(rocket.loc[common, "n_seeds"])).to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    lo = float(min(x.min(), y.min())) - 0.06
    hi = float(max(x.max(), y.max())) + 0.06
    lo, hi = max(0.0, lo), min(1.0, hi)
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=1.2, zorder=1)

    # Untuned arm first so the tuned points and their bars draw on top.
    if untuned is not None and not untuned.empty:
        u_common = [d for d in common if d in untuned.index]
        if u_common:
            ax.scatter(untuned.loc[u_common, "mean"], rocket.loc[u_common, "mean"],
                       s=38, facecolor="none", edgecolor="tab:orange", lw=1.4,
                       zorder=2, label="STLRocket (default d=1, uw=0)")

    ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=7, color="tab:blue",
                ecolor="tab:blue", elinewidth=1.2, capsize=3, alpha=0.9,
                zorder=3, label="STLRocket (tuned)")

    # Offset labels away from the diagonal (up-left above it, down-right below)
    # so neighbouring points in a tight cluster do not overprint each other.
    for ds, xi, yi in zip(common, x, y):
        above = yi >= xi
        ax.annotate(ds, (xi, yi), fontsize=7.5, alpha=0.8,
                    xytext=(-6, 7) if above else (7, -10),
                    textcoords="offset points",
                    ha="right" if above else "left")

    wins = int((x > y).sum())
    losses = int((x < y).sum())
    ties = len(common) - wins - losses
    p = _wilcoxon(x, y)

    ax.text(0.03, 0.97,
            f"STLRocket better\n{wins}W / {ties}T / {losses}L\nWilcoxon p={p:.3g}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="#eef4ff", ec="tab:blue", alpha=0.9))
    # Bottom-right is where the legend goes, so this corner label sits low-left
    # of it, below the diagonal where the baseline-wins region actually is.
    ax.text(0.97, 0.30, f"{PRETTY.get(baseline, baseline)} better",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", fc="#fff3e8", ec="tab:orange", alpha=0.9))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("STLRocket (STL + glmnet) balanced accuracy")
    ax.set_ylabel(f"{PRETTY.get(baseline, baseline)} balanced accuracy")
    ax.set_title(f"STLRocket vs {PRETTY.get(baseline, baseline)} at {budget_label}\n"
                 f"one point per dataset, error bars +/-1 SE over seeds",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / "headline_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'headline_scatter.png'}")

    return {"datasets": common, "x": x, "y": y, "xe": xe, "ye": ye,
            "wins": wins, "ties": ties, "losses": losses, "p": p}


def plot_aeon_scatter(res: dict, out_dir: Path, baseline: str) -> None:
    """The same comparison in aeon's standard TSC style, for familiarity."""
    try:
        from aeon.visualisation import plot_pairwise_scatter
    except ImportError:
        print("  skipping headline_scatter_aeon.png (aeon.visualisation unavailable)")
        return
    try:
        out = plot_pairwise_scatter(
            res["x"], res["y"], "STLRocket (tuned)", PRETTY.get(baseline, baseline),
            metric="balanced accuracy",
        )
    except Exception as exc:  # aeon raises on degenerate inputs (all ties, n<5)
        print(f"  skipping headline_scatter_aeon.png ({type(exc).__name__}: {exc})")
        return
    fig = out[0] if isinstance(out, tuple) else out
    fig.savefig(out_dir / "headline_scatter_aeon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'headline_scatter_aeon.png'}")


def plot_cd_diagram(df: pd.DataFrame, budget: int | None, out_dir: Path) -> None:
    """Average-rank diagram over every method present.

    aeon's Friedman test needs at least three methods and will raise on two, so
    this is skipped rather than attempted when the sweep has fewer.
    """
    try:
        from aeon.visualisation import plot_critical_difference
    except ImportError:
        print("  skipping cd_diagram.png (aeon.visualisation unavailable)")
        return

    stats = per_dataset_stats(df, budget)
    series = {}
    for m in ALL_METHODS:
        if m not in set(stats["method"]):
            continue
        # The STL methods still span 6 configs here; use each one's own best
        # per dataset so every method is represented at its strongest.
        src = select_best_stl(df, m) if m.startswith("stl_") else df[df["method"] == m]
        s = per_dataset_stats(src, budget)
        s = s[s["method"] == m].set_index("dataset")["mean"]
        if not s.empty:
            series[m] = s

    if len(series) < 3:
        print(f"  skipping cd_diagram.png (needs >=3 methods, have {len(series)})")
        return

    common = sorted(set.intersection(*[set(s.index) for s in series.values()]))
    if len(common) < 3:
        print(f"  skipping cd_diagram.png (needs >=3 shared datasets, have {len(common)})")
        return

    names = list(series)
    scores = np.column_stack([series[m].loc[common].to_numpy() for m in names])
    try:
        fig, _ = plot_critical_difference(scores, [PRETTY.get(m, m) for m in names])
    except Exception as exc:
        print(f"  skipping cd_diagram.png ({type(exc).__name__}: {exc})")
        return
    fig.savefig(out_dir / "cd_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'cd_diagram.png'} ({len(common)} datasets, {len(names)} methods)")


def plot_scaling_curve(df: pd.DataFrame, out_dir: Path) -> None:
    """Accuracy vs budget, averaged over datasets, one line per method."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for m in ALL_METHODS:
        if m not in set(df["method"]):
            continue
        src = select_best_stl(df, m) if m.startswith("stl_") else df[df["method"] == m]
        # Average within a dataset first so datasets with more surviving rows do
        # not dominate the across-dataset mean.
        per_ds = src.groupby(["budget", "dataset"])["balanced_accuracy"].mean()
        g = per_ds.groupby("budget").agg(["mean", "std", "count"])
        if g.empty:
            continue
        se = g["std"] / np.sqrt(g["count"].clip(lower=1))
        ax.plot(g.index, g["mean"], marker="o", ms=5, lw=1.6, label=PRETTY.get(m, m))
        ax.fill_between(g.index, g["mean"] - se, g["mean"] + se, alpha=0.18)

    ax.set_xscale("log")
    ax.set_xlabel("budget (STL formulas / ROCKET kernels)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Scaling with budget (mean over datasets, band +/-1 SE)", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "scaling_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'scaling_curve.png'}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(tuned, untuned, rocket, picks, res, out_dir: Path,
                  baseline: str) -> pd.DataFrame:
    rows = []
    for ds in res["datasets"]:
        t, r = tuned.loc[ds], rocket.loc[ds]
        u = untuned.loc[ds] if ds in untuned.index else None
        rows.append({
            "dataset": ds,
            "best_depth": picks.loc[ds, "depth"] if ds in picks.index else np.nan,
            "best_until_weight": picks.loc[ds, "until_weight"] if ds in picks.index else np.nan,
            "stl_tuned_mean": t["mean"], "stl_tuned_std": t["std"],
            "stl_tuned_n_seeds": t["n_seeds"],
            "stl_default_mean": u["mean"] if u is not None else np.nan,
            "stl_default_std": u["std"] if u is not None else np.nan,
            f"{baseline}_mean": r["mean"], f"{baseline}_std": r["std"],
            f"{baseline}_n_seeds": r["n_seeds"],
            "delta_tuned": t["mean"] - r["mean"],
            "delta_default": (u["mean"] - r["mean"]) if u is not None else np.nan,
        })
    summary = pd.DataFrame(rows).sort_values("delta_tuned", ascending=False)

    u_idx = [d for d in res["datasets"] if d in untuned.index]
    p_default = (_wilcoxon(untuned.loc[u_idx, "mean"].to_numpy(),
                           rocket.loc[u_idx, "mean"].to_numpy())
                 if u_idx else float("nan"))

    summary.loc[len(summary)] = {
        "dataset": "ALL", "best_depth": np.nan, "best_until_weight": np.nan,
        "stl_tuned_mean": res["x"].mean(), "stl_tuned_std": np.nan,
        "stl_tuned_n_seeds": np.nan,
        "stl_default_mean": untuned.loc[u_idx, "mean"].mean() if u_idx else np.nan,
        "stl_default_std": np.nan,
        f"{baseline}_mean": res["y"].mean(), f"{baseline}_std": np.nan,
        f"{baseline}_n_seeds": np.nan,
        "delta_tuned": res["x"].mean() - res["y"].mean(),
        "delta_default": ((untuned.loc[u_idx, "mean"].mean() - res["y"].mean())
                          if u_idx else np.nan),
    }

    path = out_dir / "headline_summary.csv"
    summary.to_csv(path, index=False, float_format="%.5f")
    print(f"  wrote {path}")
    return summary, p_default


def parse_args():
    p = argparse.ArgumentParser(description="Headline STLRocket vs ROCKET figure")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="figures_headline")
    p.add_argument("--baseline", default="rocket_ridge",
                   help="ROCKET arm to compare against (rocket_ridge is "
                        "ROCKET as published)")
    p.add_argument("--budget", type=int, default=10000,
                   help="budget for the headline scatter and CD diagram; "
                        "use -1 to pool every budget")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.results_dir))
    present = sorted(df["method"].unique())
    print(f"Methods present: {present}")
    for m in ("stl_linear", args.baseline):
        if m not in present:
            raise SystemExit(f"Method '{m}' has no rows. Present: {present}")

    budget = None if args.budget < 0 else args.budget
    if budget is not None and budget not in set(df["budget"]):
        raise SystemExit(f"Budget {budget} not in data. "
                         f"Available: {sorted(set(df['budget']))}")
    budget_label = "all budgets pooled" if budget is None else f"budget={budget}"

    tuned, untuned, picks = stl_arms(df, budget)
    rocket = per_dataset_stats(df[df["method"] == args.baseline], budget)
    rocket = rocket[rocket["method"] == args.baseline].set_index("dataset")

    print(f"\nBest (depth, until_weight) per dataset for stl_linear, "
          f"selected on mean accuracy:")
    print(picks.to_string())
    print(f"\nHeadline at {budget_label}: {len(set(tuned.index) & set(rocket.index))} "
          f"datasets with both arms\n")

    print("Figures:")
    res = plot_headline_scatter(tuned, untuned, rocket, out_dir,
                                budget_label, args.baseline)
    plot_aeon_scatter(res, out_dir, args.baseline)
    plot_cd_diagram(df, budget, out_dir)
    plot_scaling_curve(df, out_dir)
    summary, p_default = write_summary(tuned, untuned, rocket, picks, res,
                                       out_dir, args.baseline)

    print(f"\nPer-dataset summary ({budget_label}):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    overall = summary.iloc[-1]
    base_name = PRETTY.get(args.baseline, args.baseline)
    winner = "STLRocket (tuned)" if overall["delta_tuned"] > 0 else base_name
    print(f"\nTuned:   {winner} ahead by {abs(overall['delta_tuned']):.4f} "
          f"({res['wins']}W/{res['ties']}T/{res['losses']}L across "
          f"{len(res['datasets'])} datasets, Wilcoxon p={res['p']:.3g})")
    if not np.isnan(overall["delta_default"]):
        d_win = "STLRocket (default)" if overall["delta_default"] > 0 else base_name
        print(f"Default: {d_win} ahead by {abs(overall['delta_default']):.4f} "
              f"(d=1, uw=0; Wilcoxon p={p_default:.3g})")

    med_se = float(np.median(res["xe"]))
    print(f"\nTypical seed SE on an STL point: {med_se:.4f} "
          f"vs a claimed gap of {abs(overall['delta_tuned']):.4f} -- "
          f"read the error bars before claiming a winner.")
    print("NOTE: the STL side used its best (depth, until_weight) per dataset, "
          "selected on accuracy -- report that margin as tuned.")


if __name__ == "__main__":
    main()
