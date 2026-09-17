#!/usr/bin/env python
"""
The method in one figure: trajectory -> classifier -> local explanation.

Three panels, left to right:

  (a) the raw multivariate trajectory, one line per state variable;
  (b) the classifier's decision scores over the classes, the predicted one filled;
  (c) the same trajectory with the local explanation drawn on top -- the temporal
      window shaded, the threshold as a rule, and the witness point that makes the
      formula true circled.

Panel (c) is the point of the figure: the explanation is not a saliency heatmap
but an STL formula whose atoms are literally readable off the signal.

Everything is recomputed from scratch (the pipeline saves no checkpoints), so the
figure always matches the current code.

Usage
-----
    python scripts/plot_story.py
    python scripts/plot_story.py --dataset BasicMotions --sample_idx 10
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# LaTeX-style type without requiring a LaTeX install: matplotlib's own Computer
# Modern via mathtext. usetex=True would be a truer match but needs pdflatex,
# which is not available here.
plt.rcParams.update({
    "font.family": "serif",
    # STIX rather than cmr10: cmr10 ships no bold weight, so a bold request
    # silently falls back to regular. STIX is the same Times-like LaTeX look and
    # has a real 700, and its mathtext set has bold math to match.
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    # Bold everywhere, including inside math. mathtext.default="bf" is what makes
    # the formulae and the axis numbers bold too -- setting only font.weight
    # leaves every $...$ expression light.
    "font.weight": "bold",
    "mathtext.default": "bf",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.features import build_formula_bank
from stlrocket.classifier import train_classifier
from stlrocket.explanations import build_local_explanation

# Reference categorical palette, fixed order (see the dataviz skill). Assigned by
# slot, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#7a7a7a"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d8d4"
SURFACE = "#ffffff"
HILITE = "#eda100"   # the explanation's window / threshold


def parse_atoms(formula: str) -> list[dict]:
    """Pull (var, op, threshold) and any enclosing time window out of a formula.

    Deliberately a regex over the printed form rather than an AST walk: the figure
    only needs the atoms it can draw, and a formula whose shape it cannot parse
    should degrade to "no annotation" rather than raise.
    """
    atoms = []
    for m in re.finditer(r"\(x_(\d+)\s*(=>|<=)\s*(-?[\d.]+)\)", formula):
        atoms.append({"var": int(m.group(1)), "op": m.group(2),
                      "thr": float(m.group(3)), "span": m.span()})
    # Attach the innermost G[a,b]/F[a,b] window that encloses each atom.
    windows = [(m.start(), m.group(1), int(m.group(2)), int(m.group(3)))
               for m in re.finditer(r"([GF])\[(\d+),(\d+)\]", formula)]
    for a in atoms:
        enclosing = [w for w in windows if w[0] < a["span"][0]]
        a["win"] = (enclosing[-1][1], enclosing[-1][2], enclosing[-1][3]) if enclosing else None
    return atoms


def pretty(formula: str) -> str:
    """Formula text for a matplotlib title: ASCII ops, tightened spacing."""
    s = formula.replace("=>", "≥").replace("<=", "≤")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def mathify(formula: str) -> str:
    """Render a printed STL formula as a mathtext string.

    The whole formula becomes one math expression so operators, subscripts and
    temporal bounds typeset like the paper's notation rather than as plain text.
    Torcheck prints unicode ops; mathtext needs the LaTeX command names.
    """
    s = re.sub(r"\s+", " ", formula).strip()
    s = s.replace("=>", r" \geq ").replace("<=", r" \leq ")
    s = s.replace("∧", r" \wedge ").replace("∨", r" \vee ").replace("¬", r" \neg ")
    # Torcheck prints channels as x_0..x_{V-1}; rename to 1-indexed "channel" so
    # the formula agrees with the left panel's legend, and so x stays reserved
    # for the kernel's feature vector.
    s = re.sub(r"x_(\d+)", lambda m: rf"\mathrm{{channel}}_{{{int(m.group(1)) + 1}}}", s)
    s = re.sub(r"\b([GF])\[(\d+),\s*(\d+)\]", r"\\mathbf{\1}_{[\2,\3]}", s)
    s = re.sub(r"\b([GF])\[(\d+),\s*inf\]", r"\\mathbf{\1}_{[\2,\\infty)}", s)
    # One decimal on thresholds: the raw values carry four, which is noise in a
    # figure. Applied after the window rewrite so time bounds stay integers.
    s = re.sub(r"(-?\d+\.\d+)", lambda m: f"{float(m.group(1)):.1f}", s)
    s = re.sub(r"\s+", " ", s).strip()
    return f"${s}$"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BasicMotions")
    p.add_argument("--sample_idx", type=int, default=10,
                   help="index into the TEST set")
    p.add_argument("--depth_max", type=int, default=2)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool_size", type=int, default=10)
    p.add_argument("--precision_threshold", type=float, default=0.75)
    p.add_argument("--out", type=Path, default=Path("figures_story/method_story.png"))
    p.add_argument("--max_vars", type=int, default=6,
                   help="cap the lines in panel (a) on wide multivariate data")
    args = p.parse_args()

    config = ExperimentConfig(
        dataset=args.dataset, n_formulas=args.budget, depth_max=args.depth_max,
        only_temporal=True, until_weight=0.0, cv=5,
        pool_size=args.pool_size, precision_threshold=args.precision_threshold,
        simplify_agreement=0.98, simplify_min_gain=0.0, simplify_decimals=1,
        n_run=1, base_seed=args.seed, explain=True, output_dir=".", device="cpu",
    )

    X_tr, y_tr, X_te, y_te = load_dataset(args.dataset)
    formulas, X_tr_feats, X_te_feats = build_formula_bank(X_tr, X_te, config, args.seed)
    model = train_classifier(X_tr_feats, y_tr, config, seed=args.seed)

    W, b = model.coef_, model.intercept_
    i = args.sample_idx
    phi, target_class, picks = build_local_explanation(
        X_te_feats[i], X_te[i:i + 1], W, b, model, formulas, X_tr, y_tr, {},
        pool_size=args.pool_size, precision_threshold=args.precision_threshold,
    )
    if phi is None:
        raise SystemExit(f"no local explanation for test instance {i}")

    Wf = np.atleast_2d(np.squeeze(W))
    bf = np.atleast_1d(np.squeeze(b))
    if Wf.shape[0] == 1:  # binary: glmnet stores one row
        Wf = np.vstack([-Wf[0], Wf[0]])
        bf = np.array([-bf[0], bf[0]])
    scores = Wf @ X_te_feats[i] + bf
    classes = [str(c) for c in model.classes_]

    x = X_te[i]
    V, T = x.shape
    t = np.arange(T)
    atoms = parse_atoms(str(phi))

    # Equal width for the two data panels; the schematic between them is narrower.
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0),
                             gridspec_kw={"width_ratios": [1.0, 0.72, 1.0]})
    fig.patch.set_facecolor(SURFACE)

    # ---- (a) the trajectory -------------------------------------------------
    ax = axes[0]
    n_show = min(V, args.max_vars)
    for v in range(n_show):
        # "channel", not x: x is the feature vector the kernel produces.
        ax.plot(t, x[v], lw=2, solid_joinstyle="round", solid_capstyle="round",
                color=SERIES[v % len(SERIES)], label=f"channel {v + 1}")
    ax.set_title(f"Multivariate time series ({args.dataset} dataset)",
                 fontsize=16, color=INK, loc="center", pad=10)
    ax.set_xlabel("time", fontsize=15, color=INK)
    ax.legend(fontsize=12, ncol=min(n_show, 3), frameon=False,
              loc="upper right", labelcolor=INK, handlelength=1.4,
              columnspacing=1.0, handletextpad=0.5)

    # ---- (b) the STL kernel -------------------------------------------------
    # Schematic, not a plot: a bank of M sampled STL formulae is applied to the
    # signal, and each one's robustness becomes a feature. Drawn in axes
    # coordinates so it stays put whatever the data looks like.
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Upper box: the STL kernel turning the trajectory into features.
    ax.add_patch(FancyBboxPatch(
        (0.10, 0.545), 0.80, 0.40,
        boxstyle="round,pad=0.028,rounding_size=0.04",
        linewidth=1.4, edgecolor=SERIES[0], facecolor=SERIES[0],
        alpha=0.07, zorder=2))
    ax.text(0.5, 0.965, r"STL robustness features", ha="center", va="bottom",
            fontsize=15, color=INK, zorder=4)
    rows = [r"$\rho(\varphi_1,\, \tau) = x_1$",
            r"$\rho(\varphi_2,\, \tau) = x_2$",
            r"$\vdots$",
            r"$\rho(\varphi_M,\, \tau) = x_M$"]
    for r, lab in enumerate(rows):
        ax.text(0.5, 0.885 - r * 0.098, lab, ha="center", va="center",
                fontsize=14 if lab != r"$\vdots$" else 12, color=INK, zorder=4)

    # Lower box: the sparse linear model scoring those features. Separate box
    # because it is a separate stage -- the kernel is fixed and random, the
    # weights are what training actually learns.
    ax.add_patch(FancyBboxPatch(
        (0.10, 0.045), 0.80, 0.25,
        boxstyle="round,pad=0.028,rounding_size=0.04",
        linewidth=1.4, edgecolor=SERIES[1], facecolor=SERIES[1],
        alpha=0.07, zorder=2))
    ax.text(0.5, 0.315, r"Linear model", ha="center", va="bottom",
            fontsize=15, color=INK, zorder=4)
    # No intercept term: the classifier is fit with fit_intercept=False, so the
    # score really is a bare inner product.
    ax.text(0.5, 0.225, r"$s_k = \langle w_k,\, x \rangle$",
            ha="center", va="center", fontsize=14, color=INK, zorder=4)
    ax.text(0.5, 0.110, r"$\hat{y} = \arg\max_k\, s_k$",
            ha="center", va="center", fontsize=14, color=INK, zorder=4)

    # Trajectory in at the top, prediction out at the bottom right, with the
    # feature vector passing between the two stages.
    ax.add_patch(FancyArrowPatch((-0.03, 0.745), (0.085, 0.745),
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=1.8, color=INK_SOFT,
                                 shrinkA=0, shrinkB=0, zorder=3, clip_on=False))
    ax.add_patch(FancyArrowPatch((0.5, 0.530), (0.5, 0.400),
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=1.8, color=INK_SOFT,
                                 shrinkA=0, shrinkB=0, zorder=3))
    ax.add_patch(FancyArrowPatch((0.915, 0.175), (1.03, 0.175),
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=1.8, color=INK_SOFT,
                                 shrinkA=0, shrinkB=0, zorder=3, clip_on=False))

    # ---- (c) the explanation ------------------------------------------------
    ax = axes[2]
    shown = sorted({a["var"] for a in atoms}) or [0]
    for v in range(V):
        if v not in shown:
            ax.plot(t, x[v], lw=1, color=GRID, zorder=1)
    for v in shown:
        ax.plot(t, x[v], lw=2, solid_joinstyle="round", solid_capstyle="round",
                color=SERIES[v % len(SERIES)], zorder=3, label=f"$x_{{{v}}}$")

    for a in atoms:
        v, thr, op = a["var"], a["thr"], a["op"]
        bounded = a["win"] is not None
        lo, hi = (a["win"][1], min(a["win"][2], T - 1)) if bounded else (0, T - 1)

        # Only a genuinely bounded atom gets a shaded window. An unbounded atom
        # constrains the whole signal, and shading all of it would drown the
        # window that actually localizes the decision.
        if bounded:
            # [lo, hi] is closed, so the shading must include index hi itself --
            # otherwise a witness at the last step appears to fall outside it.
            ax.axvspan(lo - 0.5, hi + 0.5, color=HILITE, alpha=0.16, lw=0, zorder=0)
            ax.hlines(thr, lo - 0.5, hi + 0.5, color=SERIES[v % len(SERIES)],
                      lw=3, zorder=5)
            # Inside the axes, in the headroom added below for the legend --
            # above them it reads as part of the panel title.
            ax.annotate(rf"$\mathbf{{{a['win'][0]}}}_{{[{a['win'][1]},{a['win'][2]}]}}$",
                        xy=((lo + hi) / 2, 0.955), xycoords=("data", "axes fraction"),
                        xytext=(0, 0), textcoords="offset points",
                        ha="center", va="top", fontsize=16, color=INK, zorder=6)
        else:
            # Dashed and spanning the axis: a constraint that holds everywhere.
            # Coloured by channel, not by HILITE: two atoms can have nearly the
            # same threshold, and in one colour the rules merge into one line.
            ax.axhline(thr, color=SERIES[v % len(SERIES)], lw=1.8,
                       ls=(0, (5, 3)), zorder=4, alpha=0.9)

        # Mark the extreme point of the window. Under F this is a genuine
        # witness -- one qualifying step is what makes the formula true. Under G
        # every step holds, so nothing is witnessed and the extreme is instead
        # the tightest margin. The two meanings differ, which is why the dot is
        # deliberately left out of the legend rather than given one name.
        seg = x[v][lo:hi + 1]
        j = int(np.argmin(seg)) if op == "<=" else int(np.argmax(seg))
        ax.plot(lo + j, seg[j], "o", ms=9, color=SERIES[v % len(SERIES)],
                mec=SURFACE, mew=2, zorder=5)

        # Name the threshold on its own rule; two atoms can sit at nearly the
        # same height, and an unlabelled pair reads as a single line. A bounded
        # atom's label goes just right of its window (clear of the shading); an
        # unbounded one's rides the far right of the axis.
        # Mathtext throughout: the CM serif face has no unicode ≤/≥ glyph.
        label = (rf"$\mathrm{{channel}}_{{{v + 1}}}"
                 rf" \{'leq' if op == '<=' else 'geq'} {thr:.1f}$")
        # A surface-coloured bbox lifts the label off the signal it sits on;
        # without it the text is unreadable wherever a line crosses behind it.
        bbox = dict(boxstyle="round,pad=0.18", facecolor=SURFACE,
                    edgecolor="none", alpha=0.85)
        # Stagger vertically when another atom's threshold is within a few
        # percent of this one -- otherwise the two labels land on each other.
        span = abs(np.diff(ax.get_ylim())[0]) or 1.0
        close = sum(1 for o in atoms
                    if o is not a and abs(o["thr"] - thr) < 0.06 * span)
        dy = 10 if close else 5
        if bounded:
            ax.annotate(label, xy=(hi + 0.5, thr), xytext=(5, dy),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=13, color=SERIES[v % len(SERIES)], zorder=7,
                        bbox=bbox)
        else:
            ax.annotate(label, xy=(T - 1, thr), xytext=(-2, -dy - 1),
                        textcoords="offset points", ha="right", va="top",
                        fontsize=13, color=SERIES[v % len(SERIES)], zorder=7,
                        bbox=bbox)

    # Title carries the prediction and the formula itself -- the formula is the
    # result, so it belongs in the panel's heading rather than a figure caption.
    # One shared y-range across both data panels. They plot the same trajectory,
    # so unequal limits would put the same value at two different heights; the
    # top margin covers the left legend and the right window label alike.
    dmin, dmax = float(x.min()), float(x.max())
    pad = 0.06 * (dmax - dmin)
    shared = (dmin - pad, dmax + pad + 0.22 * (dmax - dmin))
    axes[0].set_ylim(*shared)
    ax.set_ylim(*shared)
    # Name the true class too: "running" alone does not say whether the model
    # got it right, which is the first thing a reader asks.
    correct = str(y_te[i]) == str(target_class)
    suffix = "correct" if correct else f"true: {y_te[i]}"
    ax.set_title(f"Predicted class: {target_class} ({suffix})",
                 fontsize=16, color=INK, loc="center", pad=16)
    ax.set_xlabel("time", fontsize=15, color=INK)
    # No legend: the shaded span is labelled with its own temporal operator and
    # each threshold rule is labelled inline, so a legend would only restate
    # what the annotations already say.

    # Panel (b) is a schematic with its axis switched off; styling only the two
    # real plots keeps it that way.
    for ax in (axes[0], axes[2]):
        ax.tick_params(labelsize=13, colors=INK)
        # tick_params has no weight argument, so the tick labels are the one
        # place the global bold setting does not reach.
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        ax.grid(True, color=GRID, lw=0.8, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

    fig.tight_layout()

    # The formula sits under the right panel, centred on it. Placed after
    # tight_layout so it can be positioned against the panel's final geometry.
    bb = axes[2].get_position()
    fig.text(bb.x0 + bb.width / 2, bb.y0 - 0.20,
             f"Local explanation:  $\\varphi$ = {mathify(str(phi))}",
             ha="center", va="top", fontsize=16, color=INK)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    print(f"predicted={target_class}  true={y_te[i]}  picks={picks}")
    print(f"phi = {phi}")
    print(f"saved {args.out}\nsaved {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
