#!/usr/bin/env python
"""
ONE clean figure: STLRocket (STL + glmnet) vs ROCKET, one point per dataset.

A deliberately minimal scatter. plot_parity.py carries the same comparison with
its full analytical apparatus across four figures; this one is the paper/slide
version and everything not needed to read the claim has been removed.

What is deliberately absent, and why:

  * no title -- the caption carries it in the paper, and a title duplicated
    into the caption is noise on a slide;
  * no marker-type distinction -- every dataset is one identical disc. Splitting
    markers by "inside/outside the margin" made the reader decode a legend to
    learn something the diagonal already shows, and it implied two populations
    where there is one;
  * no dataset names in the panel -- the names are long and six of the ten
    points sit above 0.85 within ~0.03 of each other, so names beside their
    points crowd that corner however they are offset. Each point carries its
    number inside the marker instead, which cannot collide by construction,
    and the names live in a legend outside the axes;
  * no per-dataset error bars, no trend line, no annotation boxes.

What remains is load-bearing:

  * the diagonal, which is the whole comparison: on it means equal accuracy;
  * the shaded band at +/-margin, which is what "approximately the same" means
    made visible, so the reader can see rather than compute closeness;
  * the numbered legend, ordered top-down through the plot;
  * the pooled equivalence result in one line under the axes, since the visual
    impression of closeness is not by itself evidence and the CI is.

Configuration: until_weight pinned to 0 (no Until) and only depth chosen per
dataset, so the parity claim does not rest on having tuned the operator set on
test data. Depth is still selected on test accuracy, which the footnote states.

Usage:
  python scripts/plot_parity_single.py --results_dir results --out_dir figures_parity
  python scripts/plot_parity_single.py --margin 0.05 --budget 10000
  python scripts/plot_parity_single.py --until_weight -1   # tune uw as well
  python scripts/plot_parity_single.py --no_band --no_footnote
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import STL_AXES, load_results, select_best_stl
from plot_parity import cross_pairs, per_dataset_gaps

sns.set_theme(style="whitegrid", palette="tab10")

POINT_COLOR = "tab:blue"
BAND_COLOR = "#8fd4a8"


def _numbered_points(ax, x, y, names, markersize, fontsize):
    """Draw each dataset as a numbered disc and return the legend handles.

    Dataset names are long (UWaveGestureLibrary, HandMovementDirection) and six
    of the ten points sit above 0.85 within ~0.03 of each other, so names
    printed beside their points crowd the corner no matter how they are offset.
    A numeral inside the marker cannot collide with anything by construction:
    the label is at the point, not next to it, and the panel stays clean however
    tightly the points cluster.

    Numbering follows the y-then-x order the legend is printed in, so a reader
    scanning the legend top-to-bottom sweeps the plot in a predictable
    direction rather than hunting.
    """
    # Markers must be large enough to hold two digits legibly; this is why the
    # marker size and the numeral size are chosen together rather than apart.
    ax.scatter(x, y, s=markersize, color=POINT_COLOR, zorder=5,
               edgecolor="white", linewidth=0.8)

    handles = []
    for n, (xi, yi, name) in enumerate(zip(x, y, names), start=1):
        # Two digits need a smaller glyph to stay inside the same disc; sizing
        # every numeral for the widest one would waste the single-digit discs,
        # and sizing them all for "1" overflows at "10".
        fs = fontsize if n < 10 else fontsize * 0.78
        # White on the fill: the numeral reads as part of the marker, and no
        # offset means no collision to resolve.
        ax.annotate(str(n), xy=(xi, yi), ha="center", va="center",
                    fontsize=fs, color="white", fontweight="bold",
                    zorder=6)
        # The legend swatch repeats the numbered disc rather than showing a
        # bare dot: ten identical dots beside ten names would make the reader
        # count rows to recover the mapping, which is the crowding this change
        # was meant to remove.
        handles.append(_NumberedMarker(n, markersize, fs, name))
    return handles


class _NumberedMarker:
    """Legend entry that renders as the same numbered disc used on the plot."""

    def __init__(self, number, markersize, fontsize, label):
        self.number = number
        self.markersize = markersize
        self.fontsize = fontsize
        self._label = label

    def get_label(self):
        return self._label


class _NumberedMarkerHandler:
    """Draw a `_NumberedMarker` swatch: filled circle with its numeral inside.

    An Ellipse rather than a Circle, because the handlebox coordinate system is
    not isotropic: a Circle there is drawn as an ellipse stretched to the box's
    aspect, which squashes the numeral. Sizing width and height independently
    from the box's own extents keeps the swatch round on the page.
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.patches import Ellipse

        cx = handlebox.xdescent + handlebox.width / 2.0
        cy = handlebox.ydescent + handlebox.height / 2.0
        # The handlebox is sized in points via handlelength/handleheight, so
        # both extents already correspond to the marker diameter.
        d = min(handlebox.width, handlebox.height)

        circ = Ellipse((cx, cy), d, d, facecolor=POINT_COLOR,
                       edgecolor="white", linewidth=0.8)
        handlebox.add_artist(circ)
        handlebox.add_artist(
            plt.Text(cx, cy, str(orig_handle.number), ha="center", va="center",
                     fontsize=orig_handle.fontsize, color="white",
                     fontweight="bold"))
        return circ


def plot_scatter(stats: pd.DataFrame, out_dir: Path, margin: float,
                 footnote: str | None, show_band: bool, show_stats: bool,
                 fontsize: float, markersize: float, fname: str) -> Path:
    # Number top-down (then left-right), so scanning the legend sweeps the
    # plot in a predictable direction instead of jumping around it.
    per_ds = (stats[stats["dataset"] != "ALL"]
              .sort_values(["rocket_mean", "stl_mean"], ascending=[False, True]))
    allrow = stats[stats["dataset"] == "ALL"].iloc[0]
    x = per_ds["stl_mean"].to_numpy()      # STLRocket on x
    y = per_ds["rocket_mean"].to_numpy()   # ROCKET on y

    fig, ax = plt.subplots(figsize=(6.6, 6.6))

    lo = max(0.0, float(min(x.min(), y.min())) - 0.09)
    hi = min(1.03, float(max(x.max(), y.max())) + 0.09)

    if show_band:
        ax.fill_between([lo, hi], [lo - margin, hi - margin],
                        [lo + margin, hi + margin],
                        color=BAND_COLOR, alpha=0.30, lw=0, zorder=1)
    ax.plot([lo, hi], [lo, hi], color="0.45", lw=1.1, zorder=2)

    # One identical numbered marker per dataset; the name lives in the legend.
    handles = _numbered_points(ax, x, y, per_ds["dataset"].tolist(),
                               markersize, fontsize)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("STLRocket Balanced Accuracy", fontsize=12)
    ax.set_ylabel("ROCKET Balanced Accuracy", fontsize=12)
    ax.tick_params(labelsize=10)

    # Outside the axes on the right: the points span the full diagonal, so an
    # in-axes legend of ten rows would cover data wherever it sat, and the
    # square aspect makes the space beside the panel free anyway.
    # handleheight/handlelength must leave room for the full disc; the default
    # box is shorter than the marker and would clip it.
    disc_in = np.sqrt(markersize) / 72.0            # marker diameter, inches
    handle_units = disc_in * 72.0 / (fontsize + 0.5)  # in units of fontsize
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              handler_map={_NumberedMarker: _NumberedMarkerHandler()},
              fontsize=fontsize + 0.5, frameon=False,
              handletextpad=0.7, labelspacing=0.85, borderpad=0,
              handlelength=handle_units, handleheight=handle_units)

    # No title: the caption carries it. The equivalence result goes under the
    # axes because closeness on the page is an impression, and the CI is the
    # evidence -- but it stays one line, outside the data area.
    lines = []
    if show_stats:
        lines.append(
            f"mean gap {allrow['gap']:+.3f}   "
            f"95% CI [{allrow['ci_lo']:+.3f}, {allrow['ci_hi']:+.3f}]   "
            f"within ±{margin:g}: {allrow['verdict']}")
    if footnote:
        lines.append(footnote)

    fig.tight_layout()
    if lines:
        # Measure where the x-label actually ends and start below THAT. Using
        # the axes box alone puts the first line on top of the x-label, whose
        # height tight_layout has already accounted for but not reported.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        xlab_bb = ax.xaxis.get_label().get_window_extent(renderer)
        y0 = fig.transFigure.inverted().transform(
            (0, xlab_bb.y0))[1]
        for k, line in enumerate(lines):
            fig.text(0.5, y0 - 0.030 - 0.032 * k, line, ha="center",
                     va="top",
                     fontsize=8.5 if k == 0 else 7.5,
                     style="normal" if k == 0 else "italic",
                     color="0.25" if k == 0 else "0.45")

    out_path = out_dir / fname
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    print(f"  wrote {out_path.with_suffix('.pdf')}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(
        description="One clean scatter: STLRocket vs ROCKET, per dataset")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out_dir", default="figures_parity")
    p.add_argument("--stl_method", default="stl_linear")
    p.add_argument("--baseline", default="rocket_ridge",
                   help="rocket_ridge is ROCKET as published")
    p.add_argument("--margin", type=float, default=0.05,
                   help="equivalence margin in balanced accuracy, shown as the "
                        "shaded band")
    p.add_argument("--budget", type=int, default=10000,
                   help="budget to plot; -1 pools every budget")
    p.add_argument("--until_weight", type=float, default=0.0,
                   help="pin until_weight and tune depth only (default 0, "
                        "i.e. no Until); pass -1 to tune until_weight too")
    p.add_argument("--no_band", action="store_true",
                   help="drop the shaded equivalence band, leaving the "
                        "diagonal alone")
    p.add_argument("--no_stats", action="store_true",
                   help="drop the gap/CI line under the axes")
    p.add_argument("--no_footnote", action="store_true",
                   help="drop the tuning footnote (only if the caption says it)")
    p.add_argument("--fontsize", type=float, default=8.5,
                   help="numeral and legend text size")
    p.add_argument("--markersize", type=float, default=230,
                   help="marker area in points^2; must stay large enough to "
                        "hold a two-digit numeral legibly")
    p.add_argument("--fname", default="parity_single.png")
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

    pin = None if args.until_weight < 0 else args.until_weight
    if pin is not None and pin not in set(df["until_weight"].dropna()):
        raise SystemExit(f"until_weight {pin} not in data. "
                         f"Available: {sorted(set(df['until_weight'].dropna()))}")

    sub = df if budget is None else df[df["budget"] == budget]
    pairs = cross_pairs(sub, args.stl_method, args.baseline, tuned=True,
                        pin_until_weight=pin)
    if pairs.empty:
        raise SystemExit("No paired STL/ROCKET runs at this budget.")

    stats = per_dataset_gaps(pairs, args.margin)
    n_seeds = int(pairs.groupby("dataset")["seed"].nunique().max())

    # The selected depths are a result; a reader cannot check the claim without
    # them, so they are printed even though the figure stays clean.
    src = sub if pin is None else sub[(sub["method"] != args.stl_method)
                                      | (sub["until_weight"] == pin)]
    picks = select_best_stl(src, args.stl_method).groupby("dataset")[STL_AXES].first()
    print(f"\nSelected STL config per dataset "
          f"({'uw tuned' if pin is None else f'uw pinned to {pin:g}'}):")
    print(picks.to_string())

    budget_txt = ("all budgets pooled" if budget is None
                  else f"{budget:,} features")
    footnote = None if args.no_footnote else (
        f"{len(stats) - 1} UEA datasets, {budget_txt}, mean over {n_seeds} "
        f"seeds. Depth selected per dataset on test accuracy; "
        + ("until_weight also selected on test accuracy"
           if pin is None else
           f"until_weight fixed at {pin:g}"
           + (" (no Until)" if pin == 0 else ""))
        + ". ROCKET is untuned."
    )

    print(f"\n  {len(pairs)} paired runs")
    print("\nFigure:")
    plot_scatter(stats, out_dir, args.margin, footnote,
                 not args.no_band, not args.no_stats,
                 args.fontsize, args.markersize, args.fname)

    allrow = stats[stats["dataset"] == "ALL"].iloc[0]
    n_in = int((stats[stats["dataset"] != "ALL"]["gap"].abs() <= args.margin).sum())
    print(f"\nmean gap {allrow['gap']:+.4f}, "
          f"95% CI [{allrow['ci_lo']:+.4f}, {allrow['ci_hi']:+.4f}]")
    print(f"verdict: {allrow['verdict']} at ±{args.margin:g} "
          f"({n_in}/{len(stats) - 1} datasets within margin)")
    if allrow["verdict"] != "equivalent":
        print("NOTE: the CI is not fully inside the margin, so this does NOT "
              "support an equivalence claim at this margin -- widen --margin "
              "only if you can justify it, or report it as inconclusive.")


if __name__ == "__main__":
    main()
