#!/usr/bin/env python
"""
Plot a paired comparison between any two methods in results.csv.

Default: stl_linear vs stl_tree (the classifier comparison on shared STL
features). Pass --method_a/--method_b for any other pair, e.g.
  --method_a stl_linear --method_b rocket_ridge   (STLRocket vs ROCKET as published)

Results are PAIRED and every figure respects that: differences are taken within
a pair rather than between marginal means, which is the correct analysis and a
much tighter one. What a pair shares depends on the comparison:
  * within-STL   -- a shared feature matrix, so pairs also match on depth and
                    until_weight; only the classifier differs.
  * cross-family -- only (dataset, budget, seed) are shared, since ROCKET has no
                    depth/until_weight. The STL side is first reduced to its
                    best (depth, until_weight) per dataset, so the margin is a
                    TUNED one and must be reported as such.

Figures written to --out_dir:
  1. accuracy_vs_budget.png   per-dataset accuracy against formula budget
  2. paired_delta.png         per-dataset paired delta, with a CI on the mean
  3. win_matrix.png           who wins each (dataset, budget) cell, and by how much
  4. accuracy_vs_time.png     accuracy against wall-clock -- the cost of the win
  5. depth_until_effect.png   does the ranking survive every STL feature config
                              (within-STL comparisons only)
  6. summary.csv              the numbers behind the figures, incl. Wilcoxon

Usage:
  python scripts/plot_results.py --results_dir results --out_dir figures
  python scripts/plot_results.py --method_a stl_linear --method_b rocket_ridge \
         --out_dir figures_vs_rocket
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless on the cluster; must precede pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

sns.set_theme(style="whitegrid", palette="tab10")

PRETTY = {
    "stl_linear": "STL + glmnet",
    "stl_tree": "STL + decision tree",
    "rocket_linear": "ROCKET + glmnet",
    "rocket_ridge": "ROCKET + ridge",
}
# The STL-only axes. Both STL heads carry them; ROCKET rows leave them empty.
STL_AXES = ["depth", "until_weight"]
# Every comparison pairs on these; the STL axes are added back only when both
# sides of the pair are STL methods and therefore share a feature matrix.
BASE_PAIR_KEYS = ["dataset", "budget", "seed"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(results_dir / "*" / "results.csv")))
    if not files:
        raise SystemExit(f"No results.csv found under {results_dir}/*/")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue  # a task killed before its header flushed
        if not df.empty:
            frames.append(df)
    if not frames:
        raise SystemExit(f"All results.csv under {results_dir} were empty")

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(frames)} run directories")

    # Drop rows a config-budget timeout wrote as placeholders; they carry NaN
    # accuracy and would otherwise silently deflate every mean.
    n_skipped = int((df["status"] != "ok").sum())
    df = df[df["status"] == "ok"].copy()
    if n_skipped:
        print(f"  dropped {n_skipped} non-ok rows (budget-skipped or failed)")

    # Re-running a config appends a new directory rather than replacing the old
    # one, so identical rows can appear twice. De-duplicate on the full identity
    # of a run (including the STL axes, which are NaN for ROCKET), keeping the
    # last -- otherwise a re-submitted config is counted twice in every mean.
    before = len(df)
    df = df.drop_duplicates(subset=["method"] + BASE_PAIR_KEYS + STL_AXES, keep="last")
    if before != len(df):
        print(f"  dropped {before - len(df)} duplicate rows from repeated runs")

    return df


def select_best_stl(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Reduce an STL method's 6 (depth, until_weight) configs to one per dataset.

    STL sweeps every (depth_max, until_weight) combination, but ROCKET has no
    such axes, so a cross-family pairing needs a single STL representative. We
    take the (depth, until_weight) with the highest mean accuracy per dataset --
    STLRocket at its best against ROCKET as published. The choice is made per
    dataset rather than per (dataset, budget) so the reported configuration is
    one a user could actually pick, not a different one at every budget.

    This tunes on test accuracy, so the winning margin is optimistic; the chosen
    config is recorded in summary.csv and printed, and must be stated as tuned.
    """
    sub = df[df["method"] == method]
    if sub.empty or sub[STL_AXES].isna().all().all():
        return sub  # not an STL method, or the sweep pinned both axes

    picks = []
    for ds, g in sub.groupby("dataset"):
        means = g.groupby(STL_AXES, dropna=False)["balanced_accuracy"].mean()
        if means.empty:
            continue
        best_depth, best_uw = means.idxmax()
        picks.append(g[(g["depth"] == best_depth) & (g["until_weight"] == best_uw)])
    return pd.concat(picks, ignore_index=True) if picks else sub.iloc[:0]


def paired_frame(df: pd.DataFrame, method_a: str, method_b: str) -> pd.DataFrame:
    """One row per comparable run, with both methods side by side.

    The inner join is what enforces pairing: a seed where only one method wrote
    a row (mid-seed kill, OOM) is dropped rather than compared against nothing.

    Within-STL pairs (stl_linear vs stl_tree) share a feature matrix, so they
    pair on the STL axes too -- a much tighter comparison. Cross-family pairs
    (stl_linear vs rocket_ridge) share only dataset/budget/seed, and the STL
    side is first reduced to its best configuration per dataset.
    """
    both_stl = method_a.startswith("stl_") and method_b.startswith("stl_")

    if both_stl:
        sub = df[df["method"].isin([method_a, method_b])]
        pair_keys = BASE_PAIR_KEYS + STL_AXES
    else:
        frames = [select_best_stl(df, m) if m.startswith("stl_") else df[df["method"] == m]
                  for m in (method_a, method_b)]
        sub = pd.concat(frames, ignore_index=True)
        pair_keys = BASE_PAIR_KEYS

    if sub.empty:
        return sub

    wide = sub.pivot_table(
        index=pair_keys, columns="method", values="balanced_accuracy", aggfunc="last"
    )
    missing = [m for m in (method_a, method_b) if m not in wide.columns]
    if missing:
        return wide.iloc[:0]
    wide = wide.dropna(subset=[method_a, method_b])

    times = sub.pivot_table(
        index=pair_keys, columns="method", values="time_total_s", aggfunc="last"
    )
    wide = wide.join(times, rsuffix="_time").reset_index()
    wide["delta"] = wide[method_a] - wide[method_b]
    return wide


def _grid(n, ncols=5, w=3.4, h=2.9):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, min(ncols, n), figsize=(w * min(ncols, n), h * nrows),
                             squeeze=False)
    return fig, axes.flatten()


def _finish(fig, axes_flat, n_used, out_path):
    for ax in axes_flat[n_used:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_accuracy_vs_budget(df: pd.DataFrame, out_dir: Path) -> None:
    """Accuracy against budget, one panel per dataset, all methods overlaid."""
    datasets = sorted(df["dataset"].unique())
    fig, axes = _grid(len(datasets))

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for method in [m for m in PRETTY if m in set(sub["method"])]:
            g = (sub[sub["method"] == method]
                 .groupby("budget")["balanced_accuracy"]
                 .agg(["mean", "std", "count"]))
            if g.empty:
                continue
            # Standard error across seeds; std alone overstates the uncertainty
            # of the mean we are actually plotting.
            se = g["std"] / np.sqrt(g["count"].clip(lower=1))
            ax.errorbar(g.index, g["mean"], yerr=se, marker="o", ms=4,
                        capsize=2, lw=1.5, label=PRETTY[method])
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("budget (formulas / kernels)")
        ax.set_ylabel("balanced accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02))
    _finish(fig, axes, len(datasets), out_dir / "accuracy_vs_budget.png")


def plot_paired_delta(pairs: pd.DataFrame, out_dir: Path, label: str) -> None:
    """Per-dataset paired difference against budget.

    Plotting the paired difference rather than two separate curves is the point:
    what the two arms share cancels, so the band reflects only the effect under
    study and not the (much larger) variation between datasets and seeds.
    """
    datasets = sorted(pairs["dataset"].unique())
    fig, axes = _grid(len(datasets))

    for ax, ds in zip(axes, datasets):
        sub = pairs[pairs["dataset"] == ds]
        g = sub.groupby("budget")["delta"].agg(["mean", "std", "count"])
        se = g["std"] / np.sqrt(g["count"].clip(lower=1))
        ax.axhline(0, color="black", lw=1, ls="--", zorder=1)
        ax.plot(g.index, g["mean"], marker="o", ms=4, lw=1.5, color="tab:blue", zorder=3)
        ax.fill_between(g.index, g["mean"] - 1.96 * se, g["mean"] + 1.96 * se,
                        alpha=0.25, color="tab:blue", zorder=2)
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("budget")
        ax.set_ylabel(label)

    fig.suptitle(f"Paired accuracy difference ({label}, >0 favours the first), 95% CI",
                 y=1.01, fontsize=12)
    _finish(fig, axes, len(datasets), out_dir / "paired_delta.png")


def plot_win_matrix(pairs: pd.DataFrame, out_dir: Path, label: str) -> None:
    """Heatmap of mean paired delta per (dataset, budget)."""
    piv = pairs.pivot_table(index="dataset", columns="budget",
                            values="delta", aggfunc="mean")
    if piv.empty:
        return
    # Symmetric limits so the diverging colormap's neutral point sits at zero;
    # otherwise a mostly-positive matrix reads as if the tree never loses badly.
    lim = float(np.nanmax(np.abs(piv.values))) or 1e-9

    fig, ax = plt.subplots(figsize=(1.3 * len(piv.columns) + 4, 0.45 * len(piv) + 2.5))
    # RdBu (not _r) so positive -> blue: the annotation and the title agree.
    sns.heatmap(piv, annot=True, fmt=".3f", center=0, vmin=-lim, vmax=lim,
                cmap="RdBu", cbar_kws={"label": label}, ax=ax)
    ax.set_title(f"Mean paired accuracy difference ({label}; blue favours the first)")
    ax.set_xlabel("budget")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "win_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'win_matrix.png'}")


def plot_accuracy_vs_time(df: pd.DataFrame, out_dir: Path) -> None:
    """Accuracy against wall-clock: an accuracy win that costs 10x is a tradeoff."""
    agg = (df.groupby(["method", "dataset", "budget"])
             .agg(acc=("balanced_accuracy", "mean"),
                  secs=("time_total_s", "mean"))
             .reset_index())
    datasets = sorted(agg["dataset"].unique())
    fig, axes = _grid(len(datasets))

    for ax, ds in zip(axes, datasets):
        sub = agg[agg["dataset"] == ds]
        for method in [m for m in PRETTY if m in set(sub["method"])]:
            g = sub[sub["method"] == method].sort_values("secs")
            ax.plot(g["secs"], g["acc"], marker="o", ms=4, lw=1.4, label=PRETTY[method])
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("mean wall-clock per seed (s)")
        ax.set_ylabel("balanced accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02))
    _finish(fig, axes, len(datasets), out_dir / "accuracy_vs_time.png")


def plot_depth_until_effect(pairs: pd.DataFrame, out_dir: Path, label: str) -> None:
    """Does the classifier ranking hold across every STL feature configuration?

    Skipped when the sweep pinned depth and until_weight to one value each --
    there is no axis to show.
    """
    # Cross-family pairs do not pair on the STL axes, so those columns are absent
    # entirely; a single-valued sweep leaves them present but constant.
    if not all(c in pairs.columns for c in STL_AXES):
        print("  skipping depth_until_effect.png (STL axes not part of this pairing)")
        return
    depths = sorted(pairs["depth"].dropna().unique())
    uws = sorted(pairs["until_weight"].dropna().unique())
    if len(depths) <= 1 and len(uws) <= 1:
        print("  skipping depth_until_effect.png (single depth and until_weight)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), squeeze=False)
    ax = axes[0][0]
    sns.boxplot(data=pairs, x="depth", y="delta", ax=ax, color="tab:blue", width=0.5)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_title("Paired delta by formula depth")
    ax.set_xlabel("depth_max")
    ax.set_ylabel(label)

    ax = axes[0][1]
    sns.boxplot(data=pairs, x="until_weight", y="delta", ax=ax, color="tab:orange", width=0.5)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_title("Paired delta by until_weight")
    ax.set_xlabel("until_weight")
    ax.set_ylabel(label)

    fig.tight_layout()
    fig.savefig(out_dir / "depth_until_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'depth_until_effect.png'}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def write_summary(pairs: pd.DataFrame, out_dir: Path,
                  method_a: str, method_b: str) -> pd.DataFrame:
    """Per-dataset paired summary with a Wilcoxon signed-rank test.

    Signed-rank rather than a t-test because per-seed accuracy differences on
    small test sets are discrete and far from normal; it is the standard choice
    for this comparison in the TSC literature.
    """
    rows = []
    for ds, sub in pairs.groupby("dataset"):
        d = sub["delta"].to_numpy()
        p = np.nan
        # zero_method="zsplit" keeps all-tied cases from raising; Wilcoxon is
        # undefined on an all-zero vector, so guard that explicitly.
        if len(d) >= 5 and np.any(d != 0):
            try:
                p = float(wilcoxon(d, zero_method="zsplit").pvalue)
            except ValueError:
                p = np.nan
        rows.append({
            "dataset": ds,
            "n_pairs": len(d),
            f"{method_a}_mean": sub[method_a].mean(),
            f"{method_b}_mean": sub[method_b].mean(),
            "mean_delta": d.mean(),
            "median_delta": float(np.median(d)),
            "win_rate_a": float((d > 0).mean()),
            "wilcoxon_p": p,
        })

    summary = pd.DataFrame(rows).sort_values("mean_delta", ascending=False)

    d_all = pairs["delta"].to_numpy()
    p_all = np.nan
    if len(d_all) >= 5 and np.any(d_all != 0):
        try:
            p_all = float(wilcoxon(d_all, zero_method="zsplit").pvalue)
        except ValueError:
            pass
    summary.loc[len(summary)] = {
        "dataset": "ALL", "n_pairs": len(d_all),
        f"{method_a}_mean": pairs[method_a].mean(),
        f"{method_b}_mean": pairs[method_b].mean(),
        "mean_delta": d_all.mean(),
        "median_delta": float(np.median(d_all)),
        "win_rate_a": float((d_all > 0).mean()),
        "wilcoxon_p": p_all,
    }

    path = out_dir / "summary.csv"
    summary.to_csv(path, index=False, float_format="%.5f")
    print(f"  wrote {path}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Plot a paired method comparison")
    p.add_argument("--results_dir", default="results",
                   help="directory holding the per-run <name>/results.csv subdirs")
    p.add_argument("--out_dir", default="figures")
    p.add_argument("--method_a", default="stl_linear",
                   help="first method; deltas are method_a - method_b")
    p.add_argument("--method_b", default="stl_tree",
                   help="second method (the baseline)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a, b = args.method_a, args.method_b
    label = f"{PRETTY.get(a, a)} - {PRETTY.get(b, b)}"

    df = load_results(Path(args.results_dir))
    present = sorted(df["method"].unique())
    print(f"Methods present: {present}")
    for m in (a, b):
        if m not in present:
            raise SystemExit(f"Method '{m}' has no rows. Present: {present}")

    cross_family = not (a.startswith("stl_") and b.startswith("stl_"))
    if cross_family:
        for m in (a, b):
            if m.startswith("stl_"):
                picks = (select_best_stl(df, m)
                         .groupby("dataset")[STL_AXES].first())
                print(f"\nBest (depth, until_weight) per dataset for {m}, "
                      f"selected on mean accuracy:")
                print(picks.to_string())

    pairs = paired_frame(df, a, b)
    if pairs.empty:
        raise SystemExit(
            f"No paired {a}/{b} rows found -- nothing to compare. "
            f"Methods in the data: {present}"
        )
    print(f"\nPaired runs ({label}): {len(pairs)} "
          f"({pairs['dataset'].nunique()} datasets, {pairs['budget'].nunique()} budgets)\n")

    print("Figures:")
    plot_accuracy_vs_budget(df, out_dir)
    plot_paired_delta(pairs, out_dir, label)
    plot_win_matrix(pairs, out_dir, label)
    plot_accuracy_vs_time(df, out_dir)
    plot_depth_until_effect(pairs, out_dir, label)
    summary = write_summary(pairs, out_dir, a, b)

    print(f"\nPer-dataset paired summary ({label}):")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    overall = summary.iloc[-1]
    winner = PRETTY.get(a, a) if overall["mean_delta"] > 0 else PRETTY.get(b, b)
    print(f"\nOverall: {winner} ahead by {abs(overall['mean_delta']):.4f} "
          f"balanced accuracy across {int(overall['n_pairs'])} paired runs "
          f"(win rate for {PRETTY.get(a, a)} {overall['win_rate_a']:.1%}, "
          f"Wilcoxon p={overall['wilcoxon_p']:.2e})")
    if cross_family:
        print("NOTE: the STL side used its best (depth, until_weight) per dataset, "
              "selected on accuracy -- report this margin as tuned.")


if __name__ == "__main__":
    main()
