#!/usr/bin/env python
"""
Table of the paper comparing STLRocket with the two STL learners (Bombara and Belta's decision
tree and TLINet) and MultiRocket-Hydra, on the resamples of scripts/run_resample_comparison.sh.

    python scripts/paper_stl_comparison.py [--res_dir results/comparison_resamples] \
        [--out paper/tables/stl_comparison.tex] [--n_runs 30]

Per dataset and method: mean and standard deviation of the test balanced accuracy over the
n_runs resamples, best of the four methods in bold; a method with fewer runs on a dataset is
shown as "--". TLINet is also shown read as exact STL (bacc_test_exact: the max and min over
the windows of its formulas, instead of the sparse softmax by which it is trained and scored).
The summary rows (mean, mean rank, wins / ties / losses and Holm-adjusted p-value against
STLRocket, model size, time) use the datasets where every method with at least one complete
dataset is complete; a method with none is left out of them. The tests are those of
scripts/plot_comparison.py: Friedman over the methods, then every pair with a two-sided
Wilcoxon signed-rank test over the datasets, Holm-adjusted over all pairs.

STELLE (Ferfoglia et al. 2025) is shown as reported in its paper (Table 2): test accuracy, not
balanced accuracy, with two decimals, on 10 of the datasets, under its own protocol (10 folds x
3 seeds); it is in neither the bold nor the summary rows. The script also prints the test
accuracy (acc_test) of the other methods on those 10 datasets, to compare on the same metric.

Model size: STLRocket, number of formulae with a nonzero weight; Bombara and Belta and TLINet,
total node count of the class formulae. Time: whole pipeline (features, training with model
selection, prediction) in seconds, mean over runs, median over datasets.
Prints the numbers quoted in the text. Needs only numpy and scipy.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

METHODS = {"stlrocket": "STLRocket", "bombara": "Bombara--Belta", "tlinet": "TLINet",
           "multirockethydra": "MultiRocket-Hydra"}
# STELLE, test accuracy as reported in Table 2 of Ferfoglia et al., arXiv:2511.04244 (2025)
STELLE = {"ArticularyWordRecognition": 0.95, "AtrialFibrillation": 0.33, "BasicMotions": 0.96, "Cricket": 0.88,
          "ERing": 0.84, "Epilepsy": 0.94, "EthanolConcentration": 0.32, "HandMovementDirection": 0.27,
          "Handwriting": 0.20, "Libras": 0.71}
REFERENCE = "stlrocket"
ALPHA = 0.05
SIZE_KEY = {"stlrocket": "n_selected", "bombara": "formula_size", "tlinet": "formula_size"}


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for i, j in enumerate(order):
        running = max(running, (len(p) - i) * p[j])
        adj[j] = min(running, 1.0)
    return adj


def load(res_dir: Path, n_runs: int):
    rows = defaultdict(lambda: defaultdict(dict))          # rows[method][dataset][run] = json
    for f in res_dir.glob("*/run*/*.json"):
        m = f.stem
        if m in METHODS:
            j = json.loads(f.read_text())
            if j["run"] < n_runs:
                rows[m][j["dataset"]][j["run"]] = j
    datasets = sorted({d for m in rows for d in rows[m]}, key=str.lower)
    complete = {m: {d for d in datasets if len(rows[m].get(d, {})) == n_runs} for m in METHODS}
    return rows, datasets, complete


def stat(rows, m, d, key="bacc_test"):
    v = np.array([j[key] for j in rows[m][d].values()], dtype=float)
    return v.mean(), v.std(ddof=1)


def fmt_p(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else "$<0.001$"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res_dir", default="results/comparison_resamples")
    ap.add_argument("--out", default="paper/tables/stl_comparison.tex")
    ap.add_argument("--n_runs", type=int, default=30)
    args = ap.parse_args()
    rows, datasets, complete = load(Path(args.res_dir), args.n_runs)
    shown = list(METHODS)
    summarized = [m for m in shown if complete[m]]
    common = sorted(set.intersection(*(complete[m] for m in summarized)), key=str.lower)
    for m in shown:
        print(f"{METHODS[m]}: complete on {len(complete[m])} / {len(datasets)} datasets")
    print(f"summary over {len(common)} datasets: {', '.join(summarized)}")

    cols = shown[:3] + ["tlinet_exact"] + shown[3:] + ["stelle"]
    head = [METHODS[m] for m in shown[:3]] + [r"TLINet (STL)"] + [METHODS[m] for m in shown[3:]] + [r"STELLE$^\dagger$"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Test balanced accuracy of STLRocket, of the STL learners of Bombara and Belta~\cite{bombara_offline_2021} "
             r"and TLINet~\cite{li_learning_2023}, and of MultiRocket-Hydra on each dataset: mean $\pm$ standard deviation "
             f"over {args.n_runs} resamples, best of the four methods in bold (-- : not all resamples complete). "
             r"TLINet (STL) is TLINet with its formulae read with the exact STL semantics. "
             r"$^\dagger$STELLE~\cite{ferfoglia_guided_2025}: test accuracy (not balanced) as reported by its authors, "
             r"on the $10$ datasets they evaluate, under their protocol; not included in the bold or summary rows. "
             f"Summary rows over the {len(common)} datasets where every method is complete: mean, mean rank (1 is best), "
             r"datasets where the method is better / equal / worse than STLRocket, $p$-value of the two-sided Wilcoxon "
             r"signed-rank test against STLRocket, Holm-adjusted over all pairs of methods; median over datasets of the "
             r"model size (STLRocket: formulae with nonzero weight; Bombara--Belta and TLINet: nodes of all class "
             r"formulae) and of the time of the whole pipeline in seconds.}",
             r"\label{tab:stl_comparison}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Dataset & " + " & ".join(head) + r" \\", r"\midrule"]
    for d in datasets:
        means = {m: stat(rows, m, d)[0] for m in shown if d in complete[m]}
        best = max(means.values()) if means else None
        cells = []
        for c in cols:
            if c == "stelle":
                cells.append(f"{STELLE[d]:.2f}" if d in STELLE else "--")
                continue
            m, key = ("tlinet", "bacc_test_exact") if c == "tlinet_exact" else (c, "bacc_test")
            if d not in complete[m]:
                cells.append("--")
                continue
            mu, sd = stat(rows, m, d, key)
            s = f"{mu:.3f} $\\pm$ {sd:.3f}"
            cells.append(f"\\textbf{{{s}}}" if c != "tlinet_exact" and round(mu, 3) == round(best, 3) else s)
        lines.append(f"{d} & " + " & ".join(cells) + r" \\")

    # summary over the common datasets
    acc = {m: np.array([stat(rows, m, d)[0] for d in common]) for m in summarized}
    summary = {}
    if len(common) >= 2 and len(summarized) >= 2:
        A = np.column_stack([acc[m] for m in summarized])
        ranks = np.empty_like(A)
        for i in range(len(A)):           # average ranks with ties
            vals = A[i]
            for j, m in enumerate(summarized):
                ranks[i, j] = 1 + (vals > vals[j]).sum() + 0.5 * ((vals == vals[j]).sum() - 1)
        rank = dict(zip(summarized, ranks.mean(0)))
        pairs = list(combinations(summarized, 2))
        pv = np.array([wilcoxon(acc[a], acc[b]).pvalue for a, b in pairs])
        p_holm = dict(zip(pairs, holm(pv)))
        if len(summarized) >= 3:
            chi2, p_f = friedmanchisquare(*(acc[m] for m in summarized))
            print(f"Friedman chi2({len(summarized) - 1}) = {chi2:.2f}, p = {p_f:.2e}")
        for m in summarized:
            s = {"mean": acc[m].mean(), "rank": rank[m]}
            if m != REFERENCE and REFERENCE in summarized:
                diff = acc[m] - acc[REFERENCE]
                s["wtl"] = (int((diff > 1e-9).sum()), int((np.abs(diff) <= 1e-9).sum()), int((diff < -1e-9).sum()))
                key = (REFERENCE, m) if (REFERENCE, m) in p_holm else (m, REFERENCE)
                s["p"] = p_holm[key]
                s["diff"] = diff.mean()
            summary[m] = s
        for (a, b), p in p_holm.items():
            print(f"  {METHODS[a]} vs {METHODS[b]}: mean diff {np.mean(acc[a] - acc[b]):+.3f}, Holm p = {p:.3g}")
        if "tlinet" in summarized:
            ex = np.array([stat(rows, "tlinet", d, "bacc_test_exact")[0] for d in common])
            summary["tlinet_exact"] = {"mean": ex.mean()}
            print(f"  TLINet exact STL: mean {ex.mean():.3f} (vs {acc['tlinet'].mean():.3f} as scored by its authors)")

    def med(m, key):
        if m not in summarized or not common:
            return None
        if key == "time":
            v = [np.mean([(j.get("features_s") or 0) + j["fit_s"] + (j.get("predict_s") or 0)
                          for j in rows[m][d].values()]) for d in common]
        else:
            if m not in SIZE_KEY:
                return None
            v = [np.mean([j[SIZE_KEY[m]] for j in rows[m][d].values()]) for d in common]
        return float(np.median(v))

    def row(label, f):
        return label + " & " + " & ".join(f(c) for c in cols) + r" \\"

    def get(c, k, fmt):
        s = summary.get(c, {})
        return fmt(s[k]) if k in s else "--"

    fmt_n = lambda x: "--" if x is None else (f"{x:.2g}" if x < 10 else f"{x:.0f}")
    lines += [r"\midrule",
              row("Mean", lambda c: get(c, "mean", lambda x: f"{x:.3f}")),
              row("Mean rank", lambda c: get(c, "rank", lambda x: f"{x:.2f}")),
              row("W / T / L vs STLRocket", lambda c: get(c, "wtl", lambda x: " / ".join(map(str, x)))),
              row(r"$p$ vs STLRocket (Holm)", lambda c: get(c, "p", lambda x: f"\\textbf{{{fmt_p(x)}}}" if x < ALPHA else fmt_p(x))),
              row("Model size", lambda c: "--" if c in ("tlinet_exact", "stelle") else fmt_n(med(c, "size"))),
              row("Time (s)", lambda c: "--" if c in ("tlinet_exact", "stelle") else fmt_n(med(c, "time"))),
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = Path(args.out)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    for m in summarized:
        print(f"  {METHODS[m]}: {summary.get(m)} size {med(m, 'size')} time {med(m, 'time')}")
    print(f"test accuracy on the {len(STELLE)} datasets of STELLE (reported mean {np.mean(list(STELLE.values())):.3f}):")
    for m in shown:
        if not set(STELLE) <= complete[m]:
            print(f"  {METHODS[m]}: not complete on them")
            continue
        acc = {d: stat(rows, m, d, "acc_test")[0] for d in STELLE}
        w = sum(round(acc[d], 2) > STELLE[d] for d in STELLE)
        l = sum(round(acc[d], 2) < STELLE[d] for d in STELLE)
        print(f"  {METHODS[m]}: mean {np.mean(list(acc.values())):.3f}, vs STELLE (2 decimals) W/T/L {w}/{len(STELLE) - w - l}/{l}")


if __name__ == "__main__":
    main()
