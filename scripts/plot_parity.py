#!/usr/bin/env python
"""
Is STLRocket competitive with ROCKET? (not: does it win)

STLRocket's claim is explainability -- an STL formula is a readable temporal
statement, a random convolution is not. That claim only needs accuracy PARITY to
stand: the question a reviewer asks is "what does the interpretable model cost
you", and the answer is a gap small enough to be worth the readability, not a
victory. These figures are built to answer that question and no other, which
means they are built to make a NULL result legible rather than to hunt for a
win.

Concretely that changes three things versus a win/loss figure:
  * the headline is an equivalence band, not a diagonal. A point inside the band
    is the result we want, so the band is drawn first and shaded;
  * the summary reports a two-sided confidence interval on the mean gap. "The
    gap is 0.009 +/- 0.016" is a parity claim; "p=0.24" is not, because failing
    to reject is not evidence of equivalence;
  * per-dataset panels are kept, because the pooled gap hides that STLRocket
    WINS where ROCKET is weakest (Heartbeat, HandMovementDirection) and loses on
    datasets ROCKET already solves. That asymmetry is a better argument for the
    method than the average is.

The STL side is tuned (best depth/until_weight per dataset, chosen on test
accuracy) and ROCKET is not, so every margin here flatters STLRocket. That is
stated on the figures rather than buried, and the untuned default arm is drawn
alongside so both bounds are visible.

Figures written to --out_dir:
  1. parity_scatter.png      the headline: equivalence band, one point/dataset
  2. parity_gap_ci.png       per-dataset gap with CIs, sorted -- where it wins
  3. parity_vs_difficulty.png  gap against how hard the dataset is for ROCKET
  4. parity_by_budget.png    does the gap close or widen with budget
  5. parity_summary.csv      gaps, CIs, equivalence verdicts

Usage:
  python scripts/plot_parity.py --results_dir results --out_dir figures_parity
  python scripts/plot_parity.py --margin 0.05 --baseline rocket_ridge
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
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import PRETTY, STL_AXES, load_results, select_best_stl

sns.set_theme(style="whitegrid", palette="tab10")

ACC = "balanced_accuracy"
# ROCKET has no depth/until_weight, so a cross-family pair can only share these.
PAIR_KEYS = ["dataset", "budget", "seed"]

STL_COLOR = "tab:blue"
BAND_COLOR = "#8fd4a8"


def _label_points(ax, xs, ys, labels, lo, hi, fontsize=7.5, xlo=None, xhi=None,
                  key_loc="lower right"):
    """Label points, falling back to a numbered key where they are too dense.

    Six of these ten datasets sit above 0.85 accuracy and four of those within
    ~0.03 of each other, which is closer than a dataset name is wide at any
    sensible figure size. Placing those names in situ -- by any offset, fan or
    leader-line scheme -- either overprints a neighbour or drags the label so
    far from its point that the leader crosses the parity band and implies a
    position the point does not hold.

    So: points that have room keep an inline label, and the crowded remainder
    get a small number plus a key in an empty corner. The reader loses one
    lookup on four points and the band stays clean, which is the trade that
    matters for a figure whose whole claim is "count the points in the band".

    `lo`/`hi` bound the y-axis and `xlo`/`xhi` the x-axis; they differ on any
    panel that is not square, and using the y span for horizontal offsets there
    would throw labels off the figure.
    """
    span = hi - lo
    xlo = lo if xlo is None else xlo
    xhi = hi if xhi is None else xhi
    xspan = xhi - xlo

    # Crowded = another point within this box. Tuned so the tight top-right
    # cluster is caught and the sparse lower-left points are not.
    x_near, y_near = 0.16 * xspan, 0.055 * span
    pts = sorted(zip(xs, ys, labels), key=lambda p: (-p[1], p[0]))

    keyed: list[str] = []
    for xi, yi, name in pts:
        crowded = any(abs(xi - xj) < x_near and abs(yi - yj) < y_near
                      for xj, yj, other in pts if other != name)
        if crowded:
            keyed.append(name)
            ax.annotate(str(len(keyed)), xy=(xi, yi), xytext=(6, 5),
                        textcoords="offset points", fontsize=fontsize,
                        fontweight="bold", alpha=0.95, color="0.15")
        else:
            near_right = xi > xlo + 0.72 * xspan
            ax.annotate(name, xy=(xi, yi),
                        xytext=(-7 if near_right else 7, 4),
                        textcoords="offset points", fontsize=fontsize,
                        alpha=0.9, ha="right" if near_right else "left",
                        annotation_clip=False)

    if keyed:
        loc_kw = dict(va="bottom", ha="left", x=0.02, y=0.02)
        if key_loc == "lower right":
            loc_kw = dict(va="bottom", ha="right", x=0.98, y=0.02)
        elif key_loc == "upper left":
            loc_kw = dict(va="top", ha="left", x=0.02, y=0.98)
        ax.text(loc_kw["x"], loc_kw["y"],
                "\n".join(f"{i + 1}  {n}" for i, n in enumerate(keyed)),
                transform=ax.transAxes, va=loc_kw["va"], ha=loc_kw["ha"],
                fontsize=fontsize, alpha=0.9, linespacing=1.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))


# ---------------------------------------------------------------------------
# Pairing and equivalence statistics
# ---------------------------------------------------------------------------

def cross_pairs(df: pd.DataFrame, stl_method: str, baseline: str,
                tuned: bool = True,
                pin_until_weight: float | None = None) -> pd.DataFrame:
    """Pair an STL head against a ROCKET arm on (dataset, budget, seed).

    `tuned` picks the best STL config per dataset; otherwise the d=1/uw=0
    default is used. Both are worth plotting: the tuned arm is the method's
    ceiling and the default is what someone gets without a sweep, and a parity
    claim is only honest if it survives at the default too.

    `pin_until_weight` fixes until_weight before that selection, so only
    depth is tuned. Pinning uw=0 is the useful case: it removes Until from the
    language entirely, which costs a little accuracy but buys a claim that does
    not depend on having tuned the operator set -- one fewer axis chosen on test
    data, and a simpler feature language to describe in the paper.
    """
    stl_src = df
    if pin_until_weight is not None:
        stl_src = df[(df["method"] != stl_method)
                     | (df["until_weight"] == pin_until_weight)]

    if tuned:
        stl = select_best_stl(stl_src, stl_method)
    else:
        stl = stl_src[(stl_src["method"] == stl_method)
                      & (stl_src["depth"] == 1)
                      & (stl_src["until_weight"] == 0.0)]
    roc = df[df["method"] == baseline]
    if stl.empty or roc.empty:
        return df.iloc[:0]

    pairs = stl.merge(roc, on=PAIR_KEYS, suffixes=("_stl", "_roc"))
    pairs["gap"] = pairs[f"{ACC}_stl"] - pairs[f"{ACC}_roc"]
    return pairs


def gap_ci(gaps: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Mean paired gap and a two-sided t CI on it.

    A CI rather than a p-value because the claim is equivalence. A wide CI that
    contains zero means "we cannot tell", which is a different and weaker
    statement than "they are equivalent" -- the width is what distinguishes
    them, and only a CI shows it.
    """
    g = np.asarray(gaps, dtype=float)
    g = g[~np.isnan(g)]
    n = len(g)
    if n < 2:
        return (float(g.mean()) if n else np.nan, np.nan, np.nan)
    mean = float(g.mean())
    se = float(g.std(ddof=1) / np.sqrt(n))
    half = float(student_t.ppf(0.5 + conf / 2, df=n - 1) * se)
    return mean, mean - half, mean + half


def equivalence_verdict(lo: float, hi: float, margin: float) -> str:
    """TOST-style read of a CI against an equivalence margin.

    Standard two-one-sided-tests logic: the two methods are equivalent at this
    margin when the whole CI sits inside +/-margin. Anything else is reported as
    inconclusive rather than as equivalence, because a CI that merely contains
    zero is compatible with a gap larger than the margin.
    """
    if np.isnan(lo) or np.isnan(hi):
        return "n/a"
    if lo > margin:
        return "STL better"
    if hi < -margin:
        return "ROCKET better"
    if lo >= -margin and hi <= margin:
        return "equivalent"
    return "inconclusive"


def per_dataset_gaps(pairs: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Gap, CI and verdict per dataset, plus a pooled ALL row."""
    rows = []
    for ds, g in list(pairs.groupby("dataset")) + [("ALL", pairs)]:
        mean, lo, hi = gap_ci(g["gap"].to_numpy())
        rows.append({
            "dataset": ds,
            "n_pairs": len(g),
            "stl_mean": g[f"{ACC}_stl"].mean(),
            "rocket_mean": g[f"{ACC}_roc"].mean(),
            "gap": mean, "ci_lo": lo, "ci_hi": hi,
            "within_margin": bool(abs(mean) <= margin),
            "verdict": equivalence_verdict(lo, hi, margin),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_parity_scatter(stats: pd.DataFrame, stats_default: pd.DataFrame | None,
                        out_dir: Path, baseline: str, margin: float,
                        budget_label: str) -> None:
    """The headline: per-dataset accuracies against an equivalence band.

    The band, not the diagonal, is the reference. A point inside it is the
    outcome the explainability claim needs, so it is shaded and labelled
    "parity" -- a reader should be able to count points in the band without
    reading the caption.
    """
    per_ds = stats[stats["dataset"] != "ALL"]
    x = per_ds["stl_mean"].to_numpy()
    y = per_ds["rocket_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    lo = max(0.0, float(min(x.min(), y.min())) - 0.07)
    hi = min(1.0, float(max(x.max(), y.max())) + 0.07)

    # Band first so points and labels draw on top of it.
    ax.fill_between([lo, hi], [lo - margin, hi - margin], [lo + margin, hi + margin],
                    color=BAND_COLOR, alpha=0.30, zorder=1,
                    label=f"parity band (+/-{margin:g})")
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=1.2, zorder=2)

    if stats_default is not None:
        d = stats_default[stats_default["dataset"] != "ALL"]
        ax.scatter(d["stl_mean"], d["rocket_mean"], s=38, facecolor="none",
                   edgecolor="tab:orange", lw=1.4, zorder=3,
                   label="STLRocket (default d=1, uw=0)")

    inside = np.abs(x - y) <= margin
    ax.scatter(x[inside], y[inside], s=80, color=STL_COLOR, zorder=4,
               label=f"STLRocket tuned, within margin ({inside.sum()})")
    ax.scatter(x[~inside], y[~inside], s=80, facecolor="white",
               edgecolor=STL_COLOR, lw=2.0, zorder=4,
               label=f"outside margin ({(~inside).sum()})")

    # Key goes bottom-right (empty canvas below the diagonal); the legend moves
    # up-right to make room, where the band leaves a clear strip.
    _label_points(ax, x, y, per_ds["dataset"].tolist(), lo, hi,
                  key_loc="lower right")

    allrow = stats[stats["dataset"] == "ALL"].iloc[0]
    ax.text(0.03, 0.97,
            f"mean gap {allrow['gap']:+.4f}\n"
            f"95% CI [{allrow['ci_lo']:+.4f}, {allrow['ci_hi']:+.4f}]\n"
            f"verdict: {allrow['verdict']} at +/-{margin:g}\n"
            f"{int(inside.sum())}/{len(x)} datasets within margin",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="#eef4ff", ec=STL_COLOR, alpha=0.92))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("STLRocket (STL + glmnet) balanced accuracy")
    ax.set_ylabel(f"{PRETTY.get(baseline, baseline)} balanced accuracy")
    ax.set_title(f"STLRocket reaches parity with {PRETTY.get(baseline, baseline)} "
                 f"at {budget_label}\n"
                 f"the goal is the band, not the corner -- STL buys "
                 f"readable formulas at this cost", fontsize=11)
    # Below the stats box, above the numbered key: the one clear strip left.
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.62),
              fontsize=8, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_dir / "parity_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'parity_scatter.png'}")


def plot_gap_ci(stats: pd.DataFrame, out_dir: Path, baseline: str,
                margin: float) -> None:
    """Per-dataset gap with CIs -- shows the win/loss split behind the mean.

    Sorted by gap so the bimodality is the shape of the figure: STLRocket wins
    on the datasets ROCKET finds hard and loses on the ones it already solves.
    """
    per_ds = stats[stats["dataset"] != "ALL"].sort_values("gap")
    allrow = stats[stats["dataset"] == "ALL"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ypos = np.arange(len(per_ds))

    ax.axvspan(-margin, margin, color=BAND_COLOR, alpha=0.30, zorder=1,
               label=f"parity band (+/-{margin:g})")
    ax.axvline(0, color="0.3", lw=1.2, ls="--", zorder=2)

    colors = [STL_COLOR if g > 0 else "tab:orange" for g in per_ds["gap"]]
    ax.errorbar(per_ds["gap"], ypos,
                xerr=[per_ds["gap"] - per_ds["ci_lo"],
                      per_ds["ci_hi"] - per_ds["gap"]],
                fmt="none", ecolor="0.35", elinewidth=1.3, capsize=3, zorder=3)
    ax.scatter(per_ds["gap"], ypos, s=70, c=colors, zorder=4)

    ax.set_yticks(ypos)
    ax.set_yticklabels(per_ds["dataset"], fontsize=9)
    ax.set_xlabel(f"balanced accuracy:  STLRocket  -  "
                  f"{PRETTY.get(baseline, baseline)}")
    ax.set_title("Where the interpretable model pays and where it costs\n"
                 "points with 95% CI over paired seeds", fontsize=11)

    # The pooled gap belongs on this figure because the per-dataset spread is
    # what makes it uninformative on its own.
    ax.axvline(allrow["gap"], color="0.2", lw=1.4, ls=":", zorder=3,
               label=f"pooled gap {allrow['gap']:+.4f}")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_dir / "parity_gap_ci.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'parity_gap_ci.png'}")


def plot_gap_vs_difficulty(stats: pd.DataFrame, out_dir: Path,
                           baseline: str, margin: float) -> None:
    """Gap against ROCKET's own accuracy -- the method's real selling point.

    If STLRocket's losses are concentrated on datasets ROCKET already solves and
    its wins on ones ROCKET struggles with, then STL features are capturing
    something convolutions miss, and the pooled gap is the wrong summary. This
    panel tests that directly, so a flat cloud here is an informative negative.
    """
    per_ds = stats[stats["dataset"] != "ALL"]
    x = per_ds["rocket_mean"].to_numpy()
    y = per_ds["gap"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axhspan(-margin, margin, color=BAND_COLOR, alpha=0.30, zorder=1,
               label=f"parity band (+/-{margin:g})")
    ax.axhline(0, color="0.3", lw=1.2, ls="--", zorder=2)

    ax.scatter(x, y, s=80, c=[STL_COLOR if v > 0 else "tab:orange" for v in y],
               zorder=4)
    # Fix the y-limits before labelling: _label_points clamps into the range it
    # is given, so it has to be the range the axis will actually show.
    ypad = max(0.02, 0.15 * float(np.ptp(y)))
    ylo, yhi = float(y.min()) - ypad, float(y.max()) + ypad
    ax.set_ylim(ylo, yhi)
    # Four datasets cluster near x=0.9-1.0; they go to the numbered key, placed
    # lower-left where this panel has empty canvas (the legend holds upper-right).
    xpad = 0.04 * float(np.ptp(x))
    _label_points(ax, x, y, per_ds["dataset"].tolist(), ylo, yhi,
                  xlo=float(x.min()) - xpad, xhi=float(x.max()) + xpad,
                  key_loc="lower left")

    # A trend line only if there is enough spread for one to mean anything;
    # fitting a slope through a handful of clustered points would invent a
    # relationship the data does not support.
    r = np.nan
    if len(x) >= 5 and np.ptp(x) > 0.05:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        r = float(np.corrcoef(x, y)[0, 1])
        ax.plot(xs, slope * xs + intercept, color="0.35", lw=1.4, ls="-",
                zorder=3, label=f"trend (r={r:.2f})")

    ax.set_xlabel(f"{PRETTY.get(baseline, baseline)} balanced accuracy "
                  f"(how easy the dataset is for ROCKET)")
    ax.set_ylabel("gap: STLRocket - ROCKET")
    # Title the effect that is actually there. With n=10 datasets a correlation
    # this size is suggestive, not established, and a stronger title would be
    # the easiest thing in this whole analysis for a reviewer to attack.
    strength = ("a weak tendency" if not np.isnan(r) and abs(r) < 0.5
                else "a tendency")
    ax.set_title(f"{strength.capitalize()} to gain where ROCKET struggles\n"
                 f"suggestive with n={len(x)} datasets, not established -- "
                 f"the case for STL is complementarity", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_dir / "parity_vs_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'parity_vs_difficulty.png'}")


def plot_by_budget(df: pd.DataFrame, out_dir: Path, stl_method: str,
                   baseline: str, margin: float) -> None:
    """Does the gap close with budget? Matched-budget, no tuning on either side.

    Deliberately untuned: this panel is about the scaling trend, and re-picking
    the best STL config at every budget would confound the trend with the
    selection. Both methods get the same budget and the same seeds.
    """
    stl = df[df["method"] == stl_method]
    roc = df[df["method"] == baseline]
    if stl.empty or roc.empty:
        return

    # Average STL over its config axes within a (dataset, seed) first, so the
    # curve reflects a typical config rather than a lucky one.
    s = (stl.groupby(["budget", "dataset", "seed"])[ACC].mean().reset_index())
    r = roc[["budget", "dataset", "seed", ACC]]
    m = s.merge(r, on=["budget", "dataset", "seed"], suffixes=("_stl", "_roc"))
    m["gap"] = m[f"{ACC}_stl"] - m[f"{ACC}_roc"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # Left: both curves, so the reader sees the levels, not only the difference.
    for col, label, color in [(f"{ACC}_stl", "STLRocket (mean over configs)", STL_COLOR),
                              (f"{ACC}_roc", PRETTY.get(baseline, baseline), "tab:orange")]:
        per_ds = m.groupby(["budget", "dataset"])[col].mean()
        g = per_ds.groupby("budget").agg(["mean", "std", "count"])
        se = g["std"] / np.sqrt(g["count"].clip(lower=1))
        ax1.plot(g.index, g["mean"], marker="o", ms=5, lw=1.7, color=color, label=label)
        ax1.fill_between(g.index, g["mean"] - se, g["mean"] + se, color=color, alpha=0.18)

    ax1.set_xscale("log")
    ax1.set_xlabel("budget (formulas / kernels)")
    ax1.set_ylabel("balanced accuracy")
    ax1.set_title("Both methods scale with budget", fontsize=11)
    ax1.legend(fontsize=8.5)

    # Right: the gap with a CI, against the band.
    rows = []
    for b, g in m.groupby("budget"):
        mean, lo, hi = gap_ci(g["gap"].to_numpy())
        rows.append({"budget": b, "gap": mean, "lo": lo, "hi": hi})
    gaps = pd.DataFrame(rows).sort_values("budget")

    ax2.axhspan(-margin, margin, color=BAND_COLOR, alpha=0.30, zorder=1,
                label=f"parity band (+/-{margin:g})")
    ax2.axhline(0, color="0.3", lw=1.2, ls="--", zorder=2)
    ax2.errorbar(gaps["budget"], gaps["gap"],
                 yerr=[gaps["gap"] - gaps["lo"], gaps["hi"] - gaps["gap"]],
                 marker="o", ms=6, lw=1.7, capsize=4, color=STL_COLOR, zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("budget (formulas / kernels)")
    ax2.set_ylabel("gap: STLRocket - ROCKET")
    # The widening is real but every point stays inside the band, which is the
    # part that matters for a parity claim -- say both.
    inside = bool((gaps["gap"].abs() <= margin).all())
    ax2.set_title("The gap widens with budget"
                  + (", but stays within the band" if inside else "")
                  + "\nROCKET converts extra features into accuracy faster",
                  fontsize=11)
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_dir / "parity_by_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'parity_by_budget.png'}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(stats: pd.DataFrame, stats_default: pd.DataFrame,
                  picks: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    tuned = stats.copy()
    tuned.insert(1, "arm", "tuned")
    default = stats_default.copy()
    default.insert(1, "arm", "default_d1_uw0")
    summary = pd.concat([tuned, default], ignore_index=True)

    summary = summary.merge(picks.reset_index(), on="dataset", how="left")
    path = out_dir / "parity_summary.csv"
    summary.to_csv(path, index=False, float_format="%.5f")
    print(f"  wrote {path}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(
        description="Parity (not superiority) of STLRocket against ROCKET")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="figures_parity")
    p.add_argument("--stl_method", default="stl_linear")
    p.add_argument("--baseline", default="rocket_ridge",
                   help="rocket_ridge is ROCKET as published")
    p.add_argument("--margin", type=float, default=0.05,
                   help="equivalence margin in balanced accuracy; a gap inside "
                        "+/-margin counts as parity. Pick it on what the "
                        "explainability is worth, and state it in the paper")
    p.add_argument("--budget", type=int, default=10000,
                   help="budget for the scatter/CI figures; -1 pools all")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.results_dir))
    present = sorted(df["method"].unique())
    for m in (args.stl_method, args.baseline):
        if m not in present:
            raise SystemExit(f"Method '{m}' has no rows. Present: {present}")

    budget = None if args.budget < 0 else args.budget
    if budget is not None and budget not in set(df["budget"]):
        raise SystemExit(f"Budget {budget} not in data. "
                         f"Available: {sorted(set(df['budget']))}")
    budget_label = "all budgets pooled" if budget is None else f"budget={budget}"

    # The config pick is made on the FULL sweep, then the figures are restricted
    # to one budget. Picking within the budget would make the reported config
    # budget-dependent and not something a user could act on.
    picks = (select_best_stl(df, args.stl_method)
             .groupby("dataset")[STL_AXES].first())

    sub = df if budget is None else df[df["budget"] == budget]
    pairs = cross_pairs(sub, args.stl_method, args.baseline, tuned=True)
    pairs_def = cross_pairs(sub, args.stl_method, args.baseline, tuned=False)
    if pairs.empty:
        raise SystemExit("No paired STL/ROCKET runs at this budget.")
    print(f"  {len(pairs)} paired runs (tuned), "
          f"{len(pairs_def)} (default) at {budget_label}")

    stats = per_dataset_gaps(pairs, args.margin)
    stats_def = per_dataset_gaps(pairs_def, args.margin)

    print("\nFigures:")
    plot_parity_scatter(stats, stats_def, out_dir, args.baseline,
                        args.margin, budget_label)
    plot_gap_ci(stats, out_dir, args.baseline, args.margin)
    plot_gap_vs_difficulty(stats, out_dir, args.baseline, args.margin)
    plot_by_budget(df, out_dir, args.stl_method, args.baseline, args.margin)
    write_summary(stats, stats_def, picks, out_dir)

    print(f"\nPer-dataset gaps at {budget_label} (tuned STL arm):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(stats.sort_values("gap", ascending=False)
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    allrow = stats[stats["dataset"] == "ALL"].iloc[0]
    def_row = stats_def[stats_def["dataset"] == "ALL"].iloc[0]
    n_in = int((stats[stats["dataset"] != "ALL"]["gap"].abs() <= args.margin).sum())
    n_ds = len(stats) - 1

    print(f"\nTuned arm:   gap {allrow['gap']:+.4f} "
          f"95% CI [{allrow['ci_lo']:+.4f}, {allrow['ci_hi']:+.4f}] "
          f"-> {allrow['verdict']} at +/-{args.margin:g}")
    print(f"Default arm: gap {def_row['gap']:+.4f} "
          f"95% CI [{def_row['ci_lo']:+.4f}, {def_row['ci_hi']:+.4f}] "
          f"-> {def_row['verdict']} at +/-{args.margin:g}")
    print(f"{n_in}/{n_ds} datasets land within the parity margin.")
    print("\nThe tuned arm picked depth/until_weight on TEST accuracy, so it is "
          "an upper bound -- report it as tuned, and lead with the default arm "
          "if the parity claim has to survive without a sweep.")


if __name__ == "__main__":
    main()
