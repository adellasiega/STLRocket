#!/usr/bin/env python
"""
Continuity of instance explanations (paper, Section "Explanations"): similar series should
receive similar explanations, delta(tau, tau') ~ 0 => K[E(tau), E(tau')] ~ 1.

Per run, as in the experiments (formula bank, lasso without intercept, 1-SE lambda, instance
explanations by ordered scan, as built), every test series tau gets its explanation E(tau). Then:

  delta  distance between two test series on the standardized series the classifier sees:
         l1, l2, linf on the flattened (variables x time) series, and multivariate DTW
         (aeon dtw_pairwise_distance, full window)
  K      normalized STL kernel between two explanations, with the train series as trajectories:
         k(phi, psi) = mean_i rho(phi, tau_i) rho(psi, tau_i) over the train series,
         K = k(phi, psi) / sqrt(k(phi, phi) k(psi, psi)), 1 for the same formula

Reported per (run, delta), over the pairs of distinct test series, also restricted to pairs with
the same prediction (explanations of different classes are expected to differ; there, Spearman and
the deciles of delta are taken within each predicted class, then pooled):
  spearman      Spearman correlation between delta and K (continuity: negative)
  k_nn          mean over tau of K with the explanation of its nearest test series under delta
  k_rand        mean K over all pairs (what k_nn is compared with)
  same_nn       share of series whose nearest neighbour has the same explanation (K > 1 - 1e-9)
and, for the figure, the mean K in each decile of delta.

    python scripts/run_continuity.py --dataset Epilepsy --runs 0 1 2 3 4 5 6 7 8 9
    python scripts/run_continuity.py --dataset Epilepsy --summary_only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.features import build_formula_bank, eval_robustness, set_device
from stlrocket.classifier import train_classifier
from stlrocket import instance_rules as ir

DELTAS = ("l1", "l2", "linf", "dtw")
SCOPES = ("all", "same_pred")
N_BINS = 10


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def distances(X: np.ndarray, n_jobs: int) -> dict[str, np.ndarray]:
    from aeon.distances import dtw_pairwise_distance
    flat = X.reshape(len(X), -1).astype(np.float64)
    return {"l1": cdist(flat, flat, "cityblock"), "l2": cdist(flat, flat, "euclidean"),
            "linf": cdist(flat, flat, "chebyshev"),
            "dtw": dtw_pairwise_distance(X.astype(np.float64), n_jobs=n_jobs)}


def stl_kernel(phis: list, X_tr: np.ndarray) -> np.ndarray:
    """Normalized STL kernel between formulae, the train series as trajectories."""
    R = np.stack([eval_robustness(p, X_tr) for p in phis]).astype(np.float64)   # (formulae, train series)
    k = R @ R.T / R.shape[1]
    d = np.sqrt(np.diag(k))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(k / np.outer(d, d), -1.0, 1.0)


def pair_stats(D: np.ndarray, Kmat: np.ndarray, groups: np.ndarray) -> dict:
    """Statistics over the pairs i < j of series in the same group. Correlation and distance
    deciles are taken within each group and then pooled (Spearman weighted by the pairs of the
    group): the scale of delta and the level of K differ between groups, and pooling raw values
    would mix them (Simpson's paradox). A single group: every pair."""
    n = len(D)
    iu = np.triu_indices(n, 1)
    same = groups[:, None] == groups[None, :]
    ok = same[iu] & np.isfinite(Kmat[iu])
    d, k, g = D[iu][ok], Kmat[iu][ok], groups[iu[0]][ok]
    ranks, rhos, weights = np.empty(len(d)), [], []
    for c in np.unique(g):
        m = g == c
        ranks[m] = pd.Series(d[m]).rank(pct=True).to_numpy()
        if m.sum() > 2:
            rhos.append(spearmanr(d[m], k[m]).statistic); weights.append(m.sum())
    Dn = np.where(same & ~np.eye(n, dtype=bool), D, np.inf)
    has = np.isfinite(Dn).any(axis=1)
    nn = np.argmin(Dn, axis=1)
    k_nn = Kmat[np.arange(n), nn][has]
    bins = np.minimum((ranks * N_BINS).astype(int), N_BINS - 1)
    return {"spearman": float(np.average(rhos, weights=weights)) if rhos else np.nan,
            "k_nn": float(np.nanmean(k_nn)), "k_rand": float(np.mean(k)),
            "same_nn": float(np.mean(k_nn > 1 - 1e-9)), "n_pairs": int(len(d)),
            **{f"bin{b}": float(k[bins == b].mean()) if (bins == b).any() else np.nan for b in range(N_BINS)}}


def run(args, run_id: int) -> list[dict]:
    seed = args.base_seed + run_id
    X_tr, y_tr, X_te, y_te = load_dataset(args.dataset)
    cfg = ExperimentConfig(
        dataset=args.dataset, n_formulas=args.n_formulas, depth_max=args.depth, until_weight=0.0,
        cv=args.cv, cut_point=args.cut_point, fit_intercept=False, n_run=1, base_seed=args.base_seed,
        output_dir="", device="cpu", explain=False, pool_size=0, precision_threshold=0.0,
        simplify_agreement=0.0, simplify_min_gain=0.0, simplify_decimals=0)
    formulas, F_tr, F_te, sigma = build_formula_bank(X_tr, X_te, y_tr, cfg, seed)
    model = train_classifier(F_tr, y_tr, cfg, seed=seed)
    W = ir.model_logits_matrix(model)
    if not np.any(W != 0):
        log(f"run {run_id}: empty model, skipped")
        return []
    pred_tr, pred_te = ir.predict_indices(F_tr, W), ir.predict_indices(F_te, W)
    rules = ir.explain_instances(F_te, W, F_tr, pred_tr, ir.RuleConfig())
    phis = [ir.to_stl(r["literals"], formulas, sigma) for r in rules]
    Kmat = stl_kernel(phis, X_tr)
    rows = []
    for name, D in distances(X_te, args.n_jobs).items():
        for scope, groups in (("all", np.zeros(len(pred_te), dtype=int)), ("same_pred", pred_te)):
            rows.append({"dataset": args.dataset, "run": run_id, "delta": name, "scope": scope,
                         **pair_stats(D, Kmat, groups)})
    distinct = len({str(p) for p in phis})
    log(f"run {run_id}: {len(rules)} explanations, {distinct} distinct; " + " ".join(
        f"{r['delta']}/{r['scope']}: rho={r['spearman']:+.2f} knn={r['k_nn']:.2f} krand={r['k_rand']:.2f}"
        for r in rows if r["scope"] == "all"))
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    stats = ["spearman", "k_nn", "k_rand", "same_nn"]
    g = df.groupby(["scope", "delta"], sort=False)[stats]
    mean, se = g.mean(), g.std() / np.sqrt(g.count())
    table = mean.round(3).astype(str) + " (" + se.round(3).astype(str) + ")"
    print(f"\n{df.dataset.iloc[0]}, {df.run.nunique()} runs: mean over runs (standard error)")
    print(table.to_string())
    return mean


def figure(df: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_comparison import INK, INK_2, style_axis
    from case_study import CLASS_COLORS
    plt.rcParams.update({"font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
    cols = [f"bin{b}" for b in range(N_BINS)]
    x = (np.arange(N_BINS) + 0.5) * 100 / N_BINS
    fig, axes = plt.subplots(1, len(DELTAS), figsize=(3.3 * len(DELTAS), 3.0), sharey=True)
    for ax, name in zip(axes, DELTAS):
        for scope, color, label in (("same_pred", CLASS_COLORS[0], "same prediction"),
                                    ("all", INK_2, "all pairs")):
            d = df[(df.delta == name) & (df.scope == scope)][cols]
            m, s = d.mean().to_numpy(), (d.std() / np.sqrt(len(d))).to_numpy()
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, lw=0)
            ax.plot(x, m, color=color, lw=2, marker="o", ms=4, label=label)
        ax.set_title(name.replace("linf", r"$\ell_\infty$").replace("l1", r"$\ell_1$").replace("l2", r"$\ell_2$")
                     .replace("dtw", "DTW"), color=INK)
        ax.set_xlabel(r"percentile of $\delta(\tau, \tau')$")
        style_axis(ax)
    axes[0].set_ylabel(r"mean $K[\mathcal{E}(\tau), \mathcal{E}(\tau')]$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor="white", bbox_inches="tight")
    log(f"figure: {out.with_suffix('.pdf')}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Epilepsy")
    p.add_argument("--runs", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--n_formulas", type=int, default=10000)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--cut_point", type=float, default=1.0)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--n_jobs", type=int, default=4, help="processes of the DTW distance matrix")
    p.add_argument("--out_dir", default="results/continuity")
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()
    out = Path(args.out_dir) / f"{args.dataset.lower()}.csv"
    if not args.summary_only:
        set_device("cpu")
        rows = []
        for run_id in args.runs:
            rows += run(args, run_id)
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.read_csv(out)
    summarise(df)
    figure(df, out.with_name(f"continuity_{args.dataset.lower()}"))


if __name__ == "__main__":
    main()
