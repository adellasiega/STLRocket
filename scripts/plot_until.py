#!/usr/bin/env python
"""
Does the Until operator earn its place in the STL feature language?

until_weight is the sampling weight of Until among the six operator classes in
formula_sampler._sample_operator_node, so uw=0 removes Until from the language
entirely and uw=1 samples it as often as And/Or/G/F. Everything else about the
two arms is identical, which makes this a clean ablation: pair uw=1 against uw=0
on (dataset, budget, depth, seed) and the difference is attributable to Until
alone.

The pooled answer is small (+0.0065 for stl_linear) and would be easy to
misreport in either direction, so every figure here is built to show WHERE the
effect lives rather than to defend a single number:

  * it is real for stl_linear and absent for stl_tree, so the head is a factor;
  * it is concentrated at depth_max=1 (+0.0147, p=4e-6) and vanishes at depth
    2-3, which is the opposite of the obvious expectation and the finding worth
    explaining;
  * it is carried by four segmented-motion datasets and mildly negative on the
    rest, so the pooled mean understates both sides.

On depth=1 being where Until wins: depth_max=1 still permits one operator node
at depth 0 (formula_sampler._sample_formula stops at current_depth >= depth_max,
having already sampled the root), so Until(atom, atom) is reachable. At depth 1
Until is the only operator that can express an ordered two-event pattern; by
depth 2-3 nested G/F/And combinations reach the same behaviour, and Until also
becomes structurally more expensive because it SUMS its children's time budgets
where And/Or share them. That is the mechanism the depth panel is there to show.

Figures written to --out_dir:
  1. until_by_depth.png      the headline: effect vs depth, per head
  2. until_by_dataset.png    per-dataset deltas, sorted, with seed CIs
  3. until_delta_dist.png    distribution of paired deltas, by depth
  4. until_by_budget.png     does more budget buy Until more room
  5. until_summary.csv       every number behind the figures, incl. Wilcoxon

Usage:
  python scripts/plot_until.py --results_dir results --out_dir figures_until
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

# Reuse the loader: it drops non-ok rows, skips empty CSVs from killed tasks and
# de-duplicates re-run configs, all of which this results directory needs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import PRETTY, load_results

sns.set_theme(style="whitegrid", palette="tab10")

# Until is an STL-only axis, so only the STL heads have a uw=0/uw=1 pair.
STL_METHODS = ["stl_linear", "stl_tree"]
# What a pair holds fixed. Both arms share the dataset, the formula budget, the
# depth cap and the seed; only the presence of Until in the operator set differs.
PAIR_KEYS = ["dataset", "budget", "depth", "seed"]
ACC = "balanced_accuracy"

UNTIL_COLOR = "tab:purple"
NO_UNTIL_COLOR = "0.55"


# ---------------------------------------------------------------------------
# Pairing and statistics
# ---------------------------------------------------------------------------

def until_pairs(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Join the uw=1 and uw=0 arms of one head into one row per comparable run.

    The inner join is what enforces pairing: a (dataset, budget, depth, seed)
    cell where only one arm produced a row is dropped rather than compared
    against nothing. That matters here because the skipped cells are not random
    -- they are all uw=1 at budget=10000 on the two slowest datasets, since
    sampling Until costs more time per formula. Dropping them unpaired would
    compare Until's surviving easy cells against no-Until's full set.
    """
    sub = df[df["method"] == method]
    a = sub[sub["until_weight"] == 1.0]
    b = sub[sub["until_weight"] == 0.0]
    if a.empty or b.empty:
        return sub.iloc[:0]

    pairs = a.merge(b, on=PAIR_KEYS, suffixes=("_until", "_no_until"))
    pairs["delta"] = pairs[f"{ACC}_until"] - pairs[f"{ACC}_no_until"]
    return pairs


def _wilcoxon(delta: np.ndarray) -> float:
    """Signed-rank p-value on paired differences, guarding scipy's edge cases.

    zero_method='zsplit' keeps exact ties in the ranking rather than discarding
    them. Discarding matters here: StandWalkJump ties on 53% of its pairs (it is
    a 12-instance dataset), and dropping those would shrink n enough to turn a
    null result into a spurious one.
    """
    d = np.asarray(delta)
    d = d[~np.isnan(d)]
    if len(d) < 5 or not np.any(d != 0):
        return float("nan")
    try:
        return float(wilcoxon(d, zero_method="zsplit").pvalue)
    except ValueError:
        return float("nan")


def summarise(pairs: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Paired stats, optionally grouped. `by=None` collapses to a single row."""
    groups = [("ALL", pairs)] if by is None else list(pairs.groupby(by))

    rows = []
    for key, g in groups:
        d = g["delta"].to_numpy()
        n = len(d)
        row = {
            "n_pairs": n,
            "until_mean": g[f"{ACC}_until"].mean(),
            "no_until_mean": g[f"{ACC}_no_until"].mean(),
            "mean_delta": d.mean(),
            "median_delta": np.median(d),
            # Ties are frequent and meaningful (identical feature draws can
            # produce identical accuracy), so report them rather than folding
            # them into the loss rate and understating Until.
            "win_rate": float((d > 0).mean()),
            "tie_rate": float((d == 0).mean()),
            # SE of the mean paired difference -- the correct error bar for a
            # paired design, and much tighter than the SE of either arm alone.
            "se_delta": d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
            "wilcoxon_p": _wilcoxon(d),
        }
        if by is None:
            row["group"] = "ALL"
        else:
            for name, val in zip(by, key if isinstance(key, tuple) else (key,)):
                row[name] = val
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_by_depth(all_pairs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """The headline: Until's effect against the depth cap, one line per head.

    This is the figure that carries the paper's claim. A zero line and +/-1.96
    SE bars are drawn so the reader can see that only the depth=1 stl_linear
    point clears zero, rather than taking a p-value on trust.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    for method, pairs in all_pairs.items():
        if pairs.empty:
            continue
        s = summarise(pairs, ["depth"]).sort_values("depth")
        ci = 1.96 * s["se_delta"]
        ax.errorbar(s["depth"], s["mean_delta"], yerr=ci, marker="o", ms=7,
                    lw=1.8, capsize=4, label=PRETTY.get(method, method))

        # Annotate significance where it exists; an unmarked point is a null
        # result, which is the honest default for most of this figure.
        for _, r in s.iterrows():
            if r["wilcoxon_p"] < 0.05:
                ax.annotate(f"p={r['wilcoxon_p']:.1g}",
                            (r["depth"], r["mean_delta"]),
                            xytext=(8, 8), textcoords="offset points",
                            fontsize=8, alpha=0.85)

    ax.axhline(0, color="0.3", lw=1.2, ls="--", zorder=1)
    depths = sorted({int(d) for p in all_pairs.values() if not p.empty
                     for d in p["depth"].unique()})
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("depth cap (depth_max)")
    ax.set_ylabel("balanced accuracy:  with Until  -  without Until")
    ax.set_title("Until helps only at the shallowest depth\n"
                 "paired over (dataset, budget, seed), bars +/-1.96 SE",
                 fontsize=11)
    ax.legend(fontsize=9)

    # State the mechanism on the figure: depth_max=1 still allows one operator
    # node, so Until(atom, atom) is reachable and is the only way to express an
    # ordered two-event pattern at that depth.
    ax.text(0.98, 0.04,
            "at depth 1 Until(atom, atom) is the only\n"
            "ordered two-event pattern available;\n"
            "deeper, nested G/F/And reach the same",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
            style="italic", alpha=0.8,
            bbox=dict(boxstyle="round", fc="#f6f0fb", ec=UNTIL_COLOR, alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_dir / "until_by_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'until_by_depth.png'}")


def plot_by_dataset(all_pairs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Per-dataset deltas, sorted -- shows the effect is concentrated, not broad.

    Sorted by the stl_linear delta so the ordering is stable between the two
    panels and the reader can follow a dataset across them.
    """
    methods = [m for m, p in all_pairs.items() if not p.empty]
    if not methods:
        return

    stats = {m: summarise(all_pairs[m], ["dataset"]).set_index("dataset")
             for m in methods}
    order = stats[methods[0]].sort_values("mean_delta", ascending=True).index.tolist()

    fig, axes = plt.subplots(1, len(methods), figsize=(6.2 * len(methods), 5.2),
                             sharex=True, squeeze=False)

    # sharex only links the axes; the limits still come from whichever panel
    # drew last. Set them explicitly from every panel's bars AND whiskers, so
    # the two heads are read on one scale and a bar cannot look bigger than an
    # equal bar next to it.
    span = max(float((s["mean_delta"].abs() + 1.96 * s["se_delta"].fillna(0)).max())
               for s in stats.values()) * 1.15

    for ax, method in zip(axes.flatten(), methods):
        s = stats[method].loc[order]
        # Colour by direction, not by dataset: the sign is the message, and a
        # categorical palette here would imply a grouping that does not exist.
        colors = [UNTIL_COLOR if v > 0 else "tab:orange" for v in s["mean_delta"]]
        ax.barh(range(len(s)), s["mean_delta"],
                xerr=1.96 * s["se_delta"], color=colors, alpha=0.85,
                error_kw=dict(ecolor="0.3", lw=1.1, capsize=3))

        ax.axvline(0, color="0.3", lw=1.2, ls="--")
        ax.set_xlim(-span, span)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=9)
        ax.set_xlabel("mean paired delta (with Until - without)")
        ax.set_title(PRETTY.get(method, method), fontsize=11)

        # Mark significance next to the bar rather than recolouring it, so the
        # sign stays readable independently of the p-value.
        for i, (_, r) in enumerate(s.iterrows()):
            if r["wilcoxon_p"] < 0.05:
                off = 6 if r["mean_delta"] > 0 else -6
                ax.annotate("*", (r["mean_delta"], i), xytext=(off, -4),
                            textcoords="offset points", fontsize=13,
                            ha="center", alpha=0.8)

    fig.suptitle("Until is concentrated in segmented-motion datasets, "
                 "not a broad gain   (* Wilcoxon p<0.05)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "until_by_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'until_by_dataset.png'}")


def plot_delta_distribution(all_pairs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Distribution of paired deltas by depth -- the spread behind the means.

    The means in until_by_depth are small relative to the per-pair spread, and a
    reader is entitled to see that. Violins make the point that the depth=1
    stl_linear result is a shifted centre of a wide distribution, not a
    uniformly positive effect.
    """
    frames = []
    for method, pairs in all_pairs.items():
        if pairs.empty:
            continue
        f = pairs[["depth", "delta"]].copy()
        f["method"] = PRETTY.get(method, method)
        frames.append(f)
    if not frames:
        return
    long = pd.concat(frames, ignore_index=True)
    # Integer labels: depth is a cap, and "1.0" invites reading it as continuous.
    long["depth"] = long["depth"].astype(int)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.violinplot(data=long, x="depth", y="delta", hue="method",
                   split=len(frames) == 2, inner="quartile", cut=0,
                   linewidth=1.0, ax=ax)
    ax.axhline(0, color="0.3", lw=1.2, ls="--", zorder=1)

    # A handful of pairs swing past +/-0.3 (small test sets flipping a few
    # labels) and would otherwise squash the bulk of the distribution into a
    # flat band. Clip to a symmetric 1st-99th percentile window so the shape
    # being discussed is legible, and say so rather than silently cropping.
    q = float(np.nanpercentile(np.abs(long["delta"]), 99))
    n_out = int((np.abs(long["delta"]) > q).sum())
    ax.set_ylim(-q * 1.1, q * 1.1)
    if n_out:
        ax.text(0.99, 0.02, f"y-axis clipped at +/-{q:.2f}; "
                            f"{n_out} of {len(long)} pairs fall outside",
                transform=ax.transAxes, va="bottom", ha="right",
                fontsize=7.5, style="italic", alpha=0.75)

    ax.set_xlabel("depth cap (depth_max)")
    ax.set_ylabel("paired delta (with Until - without)")
    ax.set_title("Per-pair spread dwarfs the mean effect\n"
                 "quartiles marked; the claim is a shifted centre, not a uniform win",
                 fontsize=11)
    ax.legend(title=None, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "until_delta_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'until_delta_dist.png'}")


def plot_by_budget(all_pairs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Does a bigger formula budget give Until more room to pay off?

    Worth its own panel because the two plausible stories differ: if Until only
    needs to be sampled often enough, the effect should grow with budget; if it
    is expressiveness the linear head can exploit once, it should not.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for method, pairs in all_pairs.items():
        if pairs.empty:
            continue
        s = summarise(pairs, ["budget"]).sort_values("budget")
        ax.errorbar(s["budget"], s["mean_delta"], yerr=1.96 * s["se_delta"],
                    marker="o", ms=7, lw=1.8, capsize=4,
                    label=PRETTY.get(method, method))

    ax.axhline(0, color="0.3", lw=1.2, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("budget (number of sampled STL formulas)")
    ax.set_ylabel("mean paired delta")
    ax.set_title("Until's effect against formula budget\n"
                 "bars +/-1.96 SE over paired runs", fontsize=11)
    ax.legend(fontsize=9)

    # The missing cells are not random, and a reader comparing budgets needs to
    # know that before reading the rightmost point.
    ax.text(0.02, 0.03,
            "note: the 28 skipped runs are all uw=1 at budget=10000\n"
            "(Cricket, StandWalkJump) -- pairing drops their uw=0 twins too",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5,
            style="italic", alpha=0.75)

    fig.tight_layout()
    fig.savefig(out_dir / "until_by_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'until_by_budget.png'}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(all_pairs: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    """Every grouping behind the figures, in one long-format table."""
    frames = []
    for method, pairs in all_pairs.items():
        if pairs.empty:
            continue
        for by, label in [(None, "overall"), (["depth"], "by_depth"),
                          (["budget"], "by_budget"), (["dataset"], "by_dataset"),
                          (["dataset", "depth"], "by_dataset_depth")]:
            s = summarise(pairs, by)
            s.insert(0, "method", method)
            s.insert(1, "grouping", label)
            frames.append(s)

    summary = pd.concat(frames, ignore_index=True)
    cols = ["method", "grouping", "dataset", "depth", "budget", "n_pairs",
            "until_mean", "no_until_mean", "mean_delta", "median_delta",
            "se_delta", "win_rate", "tie_rate", "wilcoxon_p"]
    summary = summary.reindex(columns=[c for c in cols if c in summary.columns])

    path = out_dir / "until_summary.csv"
    summary.to_csv(path, index=False, float_format="%.5f")
    print(f"  wrote {path}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(
        description="Ablate the Until operator (until_weight=1 vs 0)")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="figures_until")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.results_dir))

    uw = sorted(df["until_weight"].dropna().unique())
    if not {0.0, 1.0}.issubset(set(uw)):
        raise SystemExit(f"Need until_weight 0.0 and 1.0 in the sweep; found {uw}")

    all_pairs = {}
    for m in STL_METHODS:
        pairs = until_pairs(df, m)
        if pairs.empty:
            print(f"  no Until pairs for {m}, skipping")
            continue
        all_pairs[m] = pairs
        print(f"  {m}: {len(pairs)} paired runs")
    if not all_pairs:
        raise SystemExit("No method had both until_weight arms.")

    print("\nFigures:")
    plot_by_depth(all_pairs, out_dir)
    plot_by_dataset(all_pairs, out_dir)
    plot_delta_distribution(all_pairs, out_dir)
    plot_by_budget(all_pairs, out_dir)
    summary = write_summary(all_pairs, out_dir)

    print("\nOverall (paired, uw=1 minus uw=0):")
    overall = summary[summary["grouping"] == "overall"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nBy depth:")
        print(summary[summary["grouping"] == "by_depth"]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # The one-line takeaway, computed rather than asserted, so it stays honest
    # if the results directory changes under it.
    if "stl_linear" in all_pairs:
        d1 = summarise(all_pairs["stl_linear"], ["depth"])
        d1 = d1[d1["depth"] == d1["depth"].min()].iloc[0]
        tot = summarise(all_pairs["stl_linear"]).iloc[0]
        print(f"\nstl_linear: Until is worth {tot['mean_delta']:+.4f} pooled "
              f"(p={tot['wilcoxon_p']:.3g}), but {d1['mean_delta']:+.4f} at "
              f"depth={int(d1['depth'])} (p={d1['wilcoxon_p']:.3g}).")
        print("Report the depth breakdown, not the pooled mean -- the pooled "
              "number averages a real shallow effect against three null ones.")


if __name__ == "__main__":
    main()
