#!/usr/bin/env python
"""
Ablation of the feature normalisation (features._center_and_scale) on BasicMotions, as given
(10 train / 10 test series per class) and undersampled to an imbalanced version (--counts
series per class, sorted class order, on train and test alike; the series are restandardized
on the subsampled train). Per run, one formula bank is sampled and its raw train/test
robustness F computed once; the variants only differ in how F becomes features:

  scale       x = F / std(F)                     no centring (sign of x is not satisfaction)
  center      x = (F - mean(F)) / std(F)         centred, every series weighted equally
  center_bal  the same, class-balanced weights   ours: weights N / (K N_k)

By the shift property F - mu is the robustness of the shifted formula phi^{-mu}, so the
feature-space rules are exactly STL rules on shifted formulae for every variant. Then, as in
the experiments: lasso without intercept (1-SE lambda), instance rules of every train and test
series (feature space, model-faithful), global explanation by greedy cover. Reported per
(setting, variant, run): test balanced accuracy, recall of the majority and minority classes,
share of test series predicted as the majority class, selected formulae, test pooled precision,
mean coverage and length of the instance rules of the test series, global macro F1 on test.

    python scripts/run_centering_ablation.py --runs 0 1 2 3 4 5 6 7 8 9
    python scripts/run_centering_ablation.py --summary_only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_instance_comparison import global_scores
from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset, standardize_series
from stlrocket.features import _make_generator, extract_features, set_device
from stlrocket.classifier import train_classifier
from stlrocket import instance_rules as ir

VARIANTS = ("scale", "center", "center_bal")
METRICS = ("bacc_test", "recall_major", "recall_minor", "pred_major_share", "n_selected",
           "prec_test", "coverage_test", "length", "global_f1_test")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def undersample(X, y, counts, rng):
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=False)
                          for c, n in zip(np.unique(y), counts)])
    return X[idx], y[idx]


def normalise(F_tr, F_te, y_tr, variant):
    """Features of a variant from raw robustness, with the constant-column rule of
    _center_and_scale."""
    F_tr, F_te = F_tr.astype(np.float64), F_te.astype(np.float64)
    w = compute_sample_weight("balanced", y_tr) if variant == "center_bal" else np.ones(len(y_tr))
    mean = np.average(F_tr, axis=0, weights=w)
    sigma = np.sqrt(np.average((F_tr - mean) ** 2, axis=0, weights=w))
    mu = np.zeros_like(mean) if variant == "scale" else mean
    constant = sigma <= 1e-6 * np.maximum(np.abs(F_tr).max(axis=0), 1.0)
    sigma[constant] = 1.0
    X_tr, X_te = (F_tr - mu) / sigma, (F_te - mu) / sigma
    X_tr[:, constant] = X_te[:, constant] = 0.0
    return X_tr.astype(np.float32), X_te.astype(np.float32)


def evaluate(X_tr, X_te, y_tr, y_te, cfg, seed, rcfg, major, minor):
    model = train_classifier(X_tr, y_tr, cfg, seed=seed)
    W = ir.model_logits_matrix(model)
    index = {c: i for i, c in enumerate(model.classes_)}
    t_te = np.array([index[c] for c in y_te])
    pred = {"train": ir.predict_indices(X_tr, W), "test": ir.predict_indices(X_te, W)}
    row = {"bacc_test": balanced_accuracy_score(t_te, pred["test"]),
           "recall_major": recall_score(t_te, pred["test"], labels=[index[major]], average="macro", zero_division=0),
           "recall_minor": recall_score(t_te, pred["test"], labels=[index[c] for c in minor], average="macro",
                                        zero_division=0),
           "pred_major_share": float((pred["test"] == index[major]).mean()),
           "n_selected": int(np.any(W != 0, axis=0).sum())}
    if row["n_selected"] == 0:          # empty model: every logit 0, nothing to explain
        return row | {m: np.nan for m in METRICS[5:]}
    tr_rules = ir.explain_instances(X_tr, W, X_tr, pred["train"], rcfg, train=True)
    te_rules = ir.explain_instances(X_te, W, X_tr, pred["train"], rcfg)
    found = [r for r in te_rules if r["literals"]]
    tp = sup = 0
    cov = []
    for i, r in enumerate(te_rules):
        if r["literals"]:
            m = ir.rule_metrics(ir.covered(r["literals"], X_te), r["k"], pred["test"], exclude=i)
            tp, sup = tp + m["tp"], sup + m["support"]
            cov.append(m["coverage"])
    rules = [{"k": r["k"], "length": len(r["literals"]), "tp_train": r["tp"],
              "mask_train": ir.covered(r["literals"], X_tr) if r["literals"] else None,
              "mask_test": ir.covered(r["literals"], X_te) if r["literals"] else None} for r in tr_rules]
    g = global_scores(rules, pred["train"], pred["test"], rcfg.min_new_share)
    return row | {"prec_test": tp / sup if sup else np.nan, "coverage_test": np.nanmean(cov) if cov else np.nan,
                  "length": np.mean([len(r["literals"]) for r in found]) if found else np.nan,
                  "global_f1_test": float(np.mean([x["f1"] for x in g])) if g else np.nan}


def run(args, run_id, rows):
    seed = args.base_seed + run_id
    X_tr0, y_tr0, X_te0, y_te0 = load_dataset(args.dataset)
    cfg = ExperimentConfig(
        dataset=args.dataset, n_formulas=args.n_formulas, depth_max=args.depth, until_weight=0.0,
        cv=args.cv, cut_point=args.cut_point, fit_intercept=False, n_run=1, base_seed=args.base_seed,
        output_dir="", device="cpu", explain=False, pool_size=0, precision_threshold=0.0,
        simplify_agreement=0.0, simplify_min_gain=0.0, simplify_decimals=0)
    rcfg = ir.RuleConfig()
    classes = np.unique(y_tr0)
    major, minor = classes[0], classes[1:]
    rng = np.random.default_rng(seed)
    for setting in ("balanced", "imbalanced"):
        if setting == "balanced":
            X_tr, y_tr, X_te, y_te = X_tr0, y_tr0, X_te0, y_te0
        else:
            X_tr, y_tr = undersample(X_tr0, y_tr0, args.counts, rng)
            X_te, y_te = undersample(X_te0, y_te0, args.counts, rng)
            X_tr, X_te = standardize_series(X_tr, X_te)
        np.random.seed(seed)
        formulas = _make_generator(X_tr, cfg, seed).sample(cfg.n_formulas)
        F_tr, F_te = extract_features(X_tr, formulas), extract_features(X_te, formulas)
        for variant in VARIANTS:
            r = evaluate(*normalise(F_tr, F_te, y_tr, variant), y_tr, y_te, cfg, seed, rcfg, major, minor)
            rows.append({"setting": setting, "variant": variant, "run": run_id, **r})
            log(f"run {run_id} {setting:10s} {variant:10s} " + " ".join(
                f"{m}={r[m]:.3f}" for m in METRICS))


def summarise(df):
    for setting, d in df.groupby("setting", sort=False):
        print(f"\n== {setting} ({d.run.nunique()} runs): mean (s.e.); p = paired Wilcoxon vs center_bal")
        ref = d[d.variant == "center_bal"].set_index("run")
        lines = []
        for variant in VARIANTS:
            v = d[d.variant == variant].set_index("run")
            cells = []
            for m in METRICS:
                x = v[m]
                cell = f"{x.mean():.3f} ({x.std() / np.sqrt(x.count()):.3f})"
                if variant != "center_bal":
                    a, b = x.align(ref[m], join="inner")
                    ok = a.notna() & b.notna()
                    diff = (a[ok] - b[ok]).to_numpy()
                    p = wilcoxon(diff).pvalue if ok.sum() > 1 and np.any(diff != 0) else np.nan
                    cell += f" p={p:.3f}"
                cells.append(cell)
            lines.append([variant, *cells])
        print(pd.DataFrame(lines, columns=["variant", *METRICS]).set_index("variant").T.to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="BasicMotions")
    p.add_argument("--runs", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--counts", type=int, nargs="+", default=[10, 3, 3, 3],
                   help="series per class (sorted class order) in the imbalanced setting")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--n_formulas", type=int, default=10000)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--cut_point", type=float, default=1.0)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--out", default="results/centering_ablation/basicmotions.csv")
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    if not args.summary_only:
        set_device("cpu")
        rows = []
        for run_id in args.runs:
            run(args, run_id, rows)
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    summarise(pd.read_csv(out))


if __name__ == "__main__":
    main()
