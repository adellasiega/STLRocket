#!/usr/bin/env python
"""
Case study of the explanations on datasets with intuitive classes (paper, case study): what the
rules say, whether they are readable, and whether what they say is true of the data.

    python scripts/case_study.py [--datasets BasicMotions Epilepsy] [--run 0] [--stability_runs 0 1 2 3 4]

Reads the runs of scripts/run_explanations.py (results/explanations_balanced/<dataset>/run<r>/), rebuilds
the model and the rules from what they saved, and writes to results/case_study/:
  <dataset>_global.md / .tex   global explanation of each class in readable form (thresholds in
                               the units of the original series, windows in seconds), with its
                               test precision and coverage, size and number of variables
  <dataset>_global.pdf/.png    per class, one panel per atom of its global explanation: the
                               variable it reads on every test series (grey; the series predicted
                               as the class in colour), the threshold and the time span read
  <dataset>_instances.pdf/.png one correctly classified test series per class and one
                               misclassified: per literal of its rule, the robustness of the
                               formula on the train series by predicted class, the explained
                               series and the cut
  <dataset>_readability.csv    size, depth, atoms, variables and temporal operators of the test
                               instance rules and of the global disjuncts (simplified formulae)
  <dataset>_variables.pdf/.png share of the test instance rules of each predicted class that
                               read each variable
  <dataset>_atoms.csv          for each atom of the global explanations: threshold and, per
                               predicted class, mean and std of the variable over the span it
                               reads (original units), on test
  <dataset>_stability.csv      per class, across --stability_runs (different formula banks):
                               mean pairwise Jaccard of the variables used by the global
                               explanation and of the test series satisfying it
All metrics are w.r.t. the classes predicted by the model.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aeon.datasets import load_classification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_comparison import INK, INK_2, GRID, style_axis  # also sets the paper's rcParams
from stlrocket.data import load_dataset
from stlrocket.features import extract_features, eval_robustness
from stlrocket import instance_rules as ir
from stlrocket.pretty import DATASET_INFO, atom_spans, atom_threshold, formula_stats, to_text

CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MAX_ATOMS = 4   # panels per class in the global figure
MAX_LINES = 15  # above this many series a group is drawn as its 10-90% band and median


def label(name: str) -> str:
    """Variable name as math text (the paper font has no '_' in text mode): acc_x -> acc$_x$."""
    base, _, sub = name.partition("_")
    return f"$\\mathrm{{{base}}}_{{{sub}}}$" if sub else name


# ---------------------------------------------------------------- loading
def load_run(res_dir: Path, dataset: str, run: int) -> dict:
    """Everything of one run of run_explanations.py, with the model and the rules rebuilt."""
    d = res_dir / dataset / f"run{run}"
    cfg = json.loads((d / "config.json").read_text())
    model = json.loads((d / "model.json").read_text())
    with gzip.open(d / "formulas.pkl.gz", "rb") as f:
        formulas = pickle.load(f)
    sigma = np.load(d / "bank_stats.npz")["sigma"]
    X_tr, y_tr, X_te, y_te, mean, std = load_dataset(dataset, return_stats=True)
    classes = np.array(model["classes"])
    S = np.array(model["selected"], dtype=int)
    W = np.zeros((len(classes), len(formulas)))
    W[:, S] = np.array(model["weights"])
    # Features of the selected formulae only (the others have zero weight and never enter a rule).
    # The stored formulae are already centred: x_j = rho(phi_j) / sigma_j.
    F = {}
    for s, X in (("train", X_tr), ("test", X_te)):
        F[s] = np.zeros((len(X), len(formulas)), dtype=np.float32)
        if len(S):
            F[s][:, S] = extract_features(X, [formulas[j] for j in S]) / sigma[S]
    pred = {s: ir.predict_indices(F[s], W) for s in F}
    index = {c: i for i, c in enumerate(classes)}
    true = {"train": np.array([index[str(c)] for c in y_tr]), "test": np.array([index[str(c)] for c in y_te])}
    rules = {s: pd.read_csv(d / f"rules_{s}.csv") for s in ("train", "test")}
    for s in rules:
        rules[s]["literals"] = rules[s]["literals"].map(lambda x: [tuple(l) for l in json.loads(x)])
    rcfg = ir.RuleConfig(**cfg["rules"])
    train_rules = [dict(k=int(r.k), literals=r.literals, tp=int(r.tp_train) if r.n_literals else 0)
                   for r in rules["train"].itertuples()]
    G = ir.global_explanations(train_rules, F["train"], X_tr, pred["train"], formulas, sigma, rcfg)
    # The saved accuracy must be reproduced, or the rebuilt model is not the one explained
    acc = float((pred["test"] == true["test"]).mean())
    assert abs(acc - model["accuracy"]["test"]) < 1e-9, (acc, model["accuracy"]["test"])
    X_raw = {s: load_classification(dataset, split=s.upper())[0] for s in ("train", "test")}
    return dict(dataset=dataset, run=run, dir=d, classes=classes, formulas=formulas, sigma=sigma, W=W, S=S,
                X={"train": X_tr, "test": X_te}, X_raw=X_raw, mean=mean, std=std, F=F, pred=pred, true=true,
                rules=rules, G=G, info=DATASET_INFO[dataset], rcfg=rcfg)


def or_mask(phis: list, X: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(X), dtype=bool)
    for phi in phis:
        mask |= eval_robustness(phi, X) > 0
    return mask


def text(r: dict, phi) -> str:
    return to_text(phi, r["info"], r["mean"], r["std"])


# ---------------------------------------------------------------- 1. global table
def global_table(r: dict, out: Path) -> pd.DataFrame:
    rows = []
    for k, phis in sorted(r["G"].items()):
        m = ir.rule_metrics(or_mask(phis, r["X"]["test"]), k, r["pred"]["test"])
        for n, phi in enumerate(phis):
            rows.append({"class": r["classes"][k], "disjunct": n, "rule": text(r, phi), "stl": str(phi),
                         "precision_test": m["precision"], "coverage_test": m["coverage"], **formula_stats(phi)})
    df = pd.DataFrame(rows)
    # Must agree with the metrics run_explanations.py wrote for the simplified global explanations
    saved = pd.read_csv(r["dir"] / "global_metrics.csv")
    saved = saved[(saved.version == "simplified") & (saved.evaluated_on == "test")].set_index("class")
    for c, g in df.groupby("class"):
        assert np.isclose(g.precision_test.iloc[0], saved.loc[c, "precision"], equal_nan=True), c
    with open(f"{out}_global.md", "w") as f:
        f.write(f"# {r['dataset']} run {r['run']}: global explanations\n\n")
        for c, g in df.groupby("class", sort=False):
            f.write(f"## {c}  (test precision {g.precision_test.iloc[0]:.3f}, coverage {g.coverage_test.iloc[0]:.3f})\n\n")
            f.write("\n".join(f"- {row.rule}   [size {row.size}, {row.variables} variable(s)]" for row in g.itertuples()))
            f.write("\n\n")
    lines = [r"\begin{tabular}{llcc}", r"\toprule", r"Class & Global explanation & Precision & Coverage \\", r"\midrule"]
    for c, g in df.groupby("class", sort=False):
        body = r" \\ & $\vee$ ".join(row.rule.replace("<=", r"$\le$").replace(">=", r"$\ge$").replace("_", r"\_")
                                    for row in g.itertuples())
        lines.append(f"{c} & {body} & {g.precision_test.iloc[0]:.2f} & {g.coverage_test.iloc[0]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(f"{out}_global.tex").write_text("\n".join(lines) + "\n")
    return df


# ---------------------------------------------------------------- 2. global figure
def _draw_atom(ax, r, atom, a, b, k, split="test"):
    """Variable of the atom on every series of split (predicted class k in colour), its
    threshold and the span of time steps it is read on."""
    v = atom.var_index
    X, hz = r["X_raw"][split][:, v], r["info"].hz
    t = np.arange(X.shape[1]) / hz
    is_k = r["pred"][split] == k
    for group, color, z in ((X[~is_k], "#bdbdbd", 1), (X[is_k], CLASS_COLORS[k % len(CLASS_COLORS)], 2)):
        if len(group) > MAX_LINES:
            lo, med, hi = np.quantile(group, [0.1, 0.5, 0.9], axis=0)
            ax.fill_between(t, lo, hi, color=color, alpha=0.35, linewidth=0, zorder=z)
            ax.plot(t, med, color=color, linewidth=1.0, zorder=z)
        else:
            for x in group:
                ax.plot(t, x, color=color if z == 2 else GRID, linewidth=0.6 if z == 2 else 0.5,
                        alpha=0.8 if z == 2 else 1.0, zorder=z)
    end = (X.shape[1] - 1) if b is None else min(b, X.shape[1] - 1)
    ax.axvspan(a / hz, end / hz, color=INK, alpha=0.07, linewidth=0, zorder=0)
    ax.axhline(atom_threshold(atom, r["mean"], r["std"]), color=INK, linewidth=0.9, linestyle="--", zorder=3)
    ax.set_xlim(0, (X.shape[1] - 1) / hz)
    style_axis(ax)


def global_figure(r: dict, out: Path):
    ks = sorted(r["G"])
    spans = {k: [s for phi in r["G"][k] for s in atom_spans(phi)][:MAX_ATOMS] for k in ks}
    ncols = max(len(s) for s in spans.values())
    fig, axes = plt.subplots(len(ks), ncols, figsize=(2.6 * ncols, 1.9 * len(ks)), squeeze=False,
                             constrained_layout=True)
    for row, k in zip(axes, ks):
        for ax, (atom, a, b, neg) in zip(row, spans[k]):
            _draw_atom(ax, r, atom, a, b, k)
            ax.set_title(f"{r['classes'][k]}: {label(r['info'].var_names[atom.var_index])} "
                         f"{'$<$' if atom.lte else '$>$'} {atom_threshold(atom, r['mean'], r['std']):.3g}", fontsize=9)
        for ax in row[len(spans[k]):]:
            ax.set_visible(False)
        row[0].set_ylabel("value (raw units)")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    return fig


# ---------------------------------------------------------------- 3. instance figure
def pick_instances(r: dict) -> list[int]:
    """First correctly classified test series of each predicted class, then the first
    misclassified one, among those with a rule."""
    rt = r["rules"]["test"]
    has = rt.n_literals.values > 0
    ok = r["pred"]["test"] == r["true"]["test"]
    picks = [int(np.flatnonzero(has & ok & (r["pred"]["test"] == k))[0])
             for k in np.unique(r["pred"]["test"]) if (has & ok & (r["pred"]["test"] == k)).any()]
    wrong = np.flatnonzero(has & ~ok)
    return picks + ([int(wrong[0])] if len(wrong) else [])


def instance_figure(r: dict, out: Path):
    picks = pick_instances(r)
    rt = r["rules"]["test"]
    ncols = int(rt.n_literals.iloc[picks].max())
    fig, axes = plt.subplots(len(picks), ncols, figsize=(2.6 * ncols, 1.9 * len(picks)), squeeze=False,
                             constrained_layout=True)
    lines = []
    for row, i in zip(axes, picks):
        lits, k = rt.literals.iloc[i], int(rt.k.iloc[i])
        phi = ir.rule_formula(lits, r["formulas"], r["sigma"], r["X"]["train"])
        tag = "" if r["pred"]["test"][i] == r["true"]["test"][i] else f", true {r['classes'][r['true']['test'][i]]}"
        lines.append(f"test series {i} (predicted {r['classes'][k]}{tag}): {text(r, phi)}")
        for ax, (j, e, s) in zip(row, lits):
            rho_tr = r["F"]["train"][:, j] * r["sigma"][j]
            for c in np.unique(r["pred"]["train"]):
                y = rho_tr[r["pred"]["train"] == c]
                ax.scatter(y, np.full(len(y), c) + np.random.default_rng(c).uniform(-0.2, 0.2, len(y)), s=8,
                           color=CLASS_COLORS[c % len(CLASS_COLORS)], linewidths=0, alpha=0.8)
            cut = e * s * r["sigma"][j]
            ax.axvline(cut, color=INK, linestyle="--", linewidth=0.9)
            ax.axvline(0, color=INK_2, linewidth=0.5)
            ax.plot(r["F"]["test"][i, j] * r["sigma"][j], k, marker="*", markersize=11, color=INK,
                    markeredgecolor="white", linestyle="none", zorder=5)
            ax.set_yticks(range(len(r["classes"])), r["classes"], fontsize=8)
            ax.set_title(f"series {i}: $\\rho(\\varphi_{{{j}}}) {'>' if e > 0 else '<'} {cut:.2f}$", fontsize=9)
            style_axis(ax)
        for ax in row[len(lits):]:
            ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(r"robustness $\rho(\varphi_j, \tau)$")
    Path(f"{out}_instances.txt").write_text("\n".join(lines) + "\n")
    return fig


# ---------------------------------------------------------------- 4-5. readability and variables
def readability(r: dict, gdf: pd.DataFrame, out: Path) -> pd.DataFrame:
    rt = r["rules"]["test"]
    rows = []
    for row in rt[rt.n_literals > 0].itertuples():
        phi = ir.rule_formula(row.literals, r["formulas"], r["sigma"], r["X"]["train"])
        rows.append({"kind": "instance", "class": r["classes"][row.k], "literals": row.n_literals,
                     "size_raw": row.size_raw, "vars": sorted({a.var_index for a, *_ in atom_spans(phi)}),
                     **formula_stats(phi)})
    for row in gdf.itertuples():
        rows.append({"kind": "global", "class": row._1, "literals": np.nan, "size_raw": np.nan,
                     **{c: getattr(row, c) for c in ("size", "depth", "atoms", "variables", "n_G", "n_F", "n_U")}})
    df = pd.DataFrame(rows)
    df.drop(columns="vars").to_csv(f"{out}_readability.csv", index=False)
    return df


def variables_figure(r: dict, read: pd.DataFrame):
    inst = read[read.kind == "instance"]
    names = r["info"].var_names
    share = np.array([[np.mean([v in vs for vs in inst[inst["class"] == c].vars]) for v in range(len(names))]
                      for c in r["classes"]])
    fig, ax = plt.subplots(figsize=(0.8 * len(names) + 1.8, 0.45 * len(r["classes"]) + 1.0), constrained_layout=True)
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (a, b), v in np.ndenumerate(share):
        ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(names)), [label(n) for n in names])
    ax.set_yticks(range(len(r["classes"])), r["classes"])
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, label="share of test rules")
    return fig


# ---------------------------------------------------------------- 6. atoms vs simple statistics
def atom_statistics(r: dict, out: Path) -> pd.DataFrame:
    rows = []
    X, T = r["X_raw"]["test"], r["X_raw"]["test"].shape[2]
    for k, phis in sorted(r["G"].items()):
        for n, phi in enumerate(phis):
            for atom, a, b, neg in atom_spans(phi):
                end = T - 1 if b is None else min(b, T - 1)
                seg = X[:, atom.var_index, a:end + 1]
                row = {"class": r["classes"][k], "disjunct": n, "variable": r["info"].var_names[atom.var_index],
                       "direction": "<=" if atom.lte else ">=", "negated": neg,
                       "threshold": atom_threshold(atom, r["mean"], r["std"]),
                       "span_s": f"[{a / r['info'].hz:.3g}, {end / r['info'].hz:.3g}]"}
                for c in np.unique(r["pred"]["test"]):
                    s = seg[r["pred"]["test"] == c]
                    row[f"{r['classes'][c]} mean"] = float(s.mean())
                    row[f"{r['classes'][c]} min"] = float(s.min(axis=1).mean())
                    row[f"{r['classes'][c]} max"] = float(s.max(axis=1).mean())
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}_atoms.csv", index=False)
    return df


# ---------------------------------------------------------------- 7. stability
def stability(res_dir: Path, dataset: str, runs: list[int], out: Path) -> pd.DataFrame:
    rs = [load_run(res_dir, dataset, run) for run in runs]
    X_te = rs[0]["X"]["test"]
    rows = []
    for k, c in enumerate(rs[0]["classes"]):
        vars_, masks = [], []
        for r in rs:
            phis = r["G"].get(k, [])
            vars_.append({a.var_index for phi in phis for a, *_ in atom_spans(phi)})
            masks.append(or_mask(phis, X_te))
        jv = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(vars_, 2)]
        jm = [(a & b).sum() / max((a | b).sum(), 1) for a, b in itertools.combinations(masks, 2)]
        names = rs[0]["info"].var_names
        rows.append({"class": c, "runs": len(rs), "jaccard_variables": float(np.mean(jv)),
                     "jaccard_test_series": float(np.mean(jm)),
                     "variables_per_run": "; ".join(",".join(names[v] for v in sorted(s)) for s in vars_)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}_stability.csv", index=False)
    return df


def save(fig, path: str):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["BasicMotions", "Epilepsy"])
    ap.add_argument("--run", type=int, default=0, help="run shown in the tables and figures")
    ap.add_argument("--stability_runs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--res_dir", default="results/explanations_balanced")
    ap.add_argument("--out_dir", default="results/case_study")
    args = ap.parse_args()
    res_dir, out_dir = Path(args.res_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 120)

    for dataset in args.datasets:
        out = out_dir / dataset
        r = load_run(res_dir, dataset, args.run)
        print(f"\n===== {dataset} run {args.run}: {len(r['S'])} formulae selected")
        gdf = global_table(r, out)
        print(gdf[["class", "rule", "precision_test", "coverage_test", "size"]].to_string(index=False))
        save(global_figure(r, out), f"{out}_global")
        save(instance_figure(r, out), f"{out}_instances")
        print(Path(f"{out}_instances.txt").read_text())
        read = readability(r, gdf, out)
        print(read.groupby("kind")[["literals", "size_raw", "size", "depth", "atoms", "variables"]]
              .describe().T.round(2).to_string())
        save(variables_figure(r, read), f"{out}_variables")
        print(atom_statistics(r, out).round(2).to_string(index=False))
        print(stability(res_dir, dataset, args.stability_runs, out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
