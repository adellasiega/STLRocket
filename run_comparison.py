#!/usr/bin/env python
"""
Three-way classification comparison on a single time-series dataset.

Methods:
  1. rocket      -- aeon RocketClassifier (random convolutional kernels + ridge head)
  2. stl_linear  -- STL robustness features -> glmnet LogitNet (L1 logistic regression)
  3. stl_tree    -- same STL robustness features -> sklearn DecisionTreeClassifier (CV-tuned)

For each "budget" b in {10,100,1000,10000} (kernel count for rocket, formula count for
STL) we run n_run seeds.  The STL approaches additionally sweep depth_max in {1,2,3}; the
two STL heads share a single robustness feature matrix per (budget, depth, seed) that is
computed exactly once.

Hyperparameter fairness: each head CV-tunes its primary capacity/regularization knob with
the same --cv fold count (ridge alpha for rocket, L1 lambda for glmnet, tree structure for
the decision tree).

Cost control: each configuration gets a wall-clock budget (--config_budget). A configuration
is one rocket (budget) or one STL (budget, depth) pair-of-heads. Seeds are the inner loop;
before each seed we check elapsed time and skip the remaining seeds once the budget is spent.
Completed runs are written to results.csv immediately, so partial results survive.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.preprocessing import StateVariableStandardScaler
from stlrocket.features import build_formula_bank
from stlrocket.classifier import train_classifier, evaluate_classifier


ROW_FIELDS = [
    "dataset", "method", "budget", "depth", "seed",
    "balanced_accuracy", "time_fit_s", "time_feats_s", "time_total_s", "status",
]


# ---------------------------------------------------------------------------
# Per-method fits
# ---------------------------------------------------------------------------

def run_rocket(X_tr_raw, y_tr, X_te_raw, y_te, n_kernels: int, seed: int, cv: int) -> dict:
    """ROCKET on raw (N,V,T) arrays with a cv-tuned ridge head."""
    estimator = RidgeClassifierCV(
        alphas=np.logspace(-3, 3, 10), cv=cv, class_weight="balanced"
    )
    clf = RocketClassifier(
        n_kernels=n_kernels, random_state=seed, n_jobs=8, estimator=estimator
    )
    t0 = time.perf_counter()
    clf.fit(X_tr_raw, y_tr)
    t1 = time.perf_counter()
    y_pred = clf.predict(X_te_raw)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "time_fit_s": round(t1 - t0, 4),
        "time_feats_s": None,
        "time_total_s": round(time.perf_counter() - t0, 4),
        "status": "ok",
    }


def build_stl_features(X_tr_raw, y_tr, X_te_raw, n_formulas, depth_max, seed, args):
    """Build STL robustness features once for both STL heads.

    Returns (config, X_tr_feats, X_te_feats, time_feats_s).
    """
    scaler = StateVariableStandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    config = ExperimentConfig(
        dataset=args.dataset,
        n_formulas=n_formulas,
        depth_max=depth_max,
        only_temporal=args.only_temporal,
        until_weight=args.until_weight,
        cv=args.cv,
        explain=False,
    )
    t0 = time.perf_counter()
    _formulas, X_tr_feats, X_te_feats = build_formula_bank(X_tr, X_te, config, seed)
    time_feats_s = round(time.perf_counter() - t0, 4)
    return config, X_tr_feats, X_te_feats, time_feats_s


def run_stl_linear(X_tr_feats, y_tr, X_te_feats, y_te, config, time_feats_s) -> dict:
    t0 = time.perf_counter()
    model = train_classifier(X_tr_feats, y_tr, config)
    t1 = time.perf_counter()
    metrics = evaluate_classifier(model, X_te_feats, y_te)
    return {
        "balanced_accuracy": metrics["balanced_accuracy"],
        "time_fit_s": round(t1 - t0, 4),
        "time_feats_s": time_feats_s,
        "time_total_s": round(time_feats_s + (t1 - t0), 4),
        "status": "ok",
    }


def run_stl_tree(X_tr_feats, y_tr, X_te_feats, y_te, seed, cv, time_feats_s) -> dict:
    # Guard against tiny per-class counts breaking StratifiedKFold.
    _, counts = np.unique(y_tr, return_counts=True)
    n_splits = max(2, min(cv, int(counts.min())))

    param_grid = {
        "max_depth": [None, 3, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "ccp_alpha": [0.0, 1e-3, 1e-2],
    }
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
        param_grid,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        scoring="balanced_accuracy",
        n_jobs=8,
    )
    t0 = time.perf_counter()
    search.fit(X_tr_feats, y_tr)
    t1 = time.perf_counter()
    y_pred = search.predict(X_te_feats)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "time_fit_s": round(t1 - t0, 4),
        "time_feats_s": time_feats_s,
        "time_total_s": round(time_feats_s + (t1 - t0), 4),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def make_row(dataset, method, budget, depth, seed, metrics) -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "budget": budget,
        "depth": depth if depth is not None else "",
        "seed": seed,
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "time_fit_s": metrics.get("time_fit_s"),
        "time_feats_s": metrics.get("time_feats_s"),
        "time_total_s": metrics.get("time_total_s"),
        "status": metrics["status"],
    }


def skipped_metrics() -> dict:
    return {
        "balanced_accuracy": float("nan"),
        "time_fit_s": None,
        "time_feats_s": None,
        "time_total_s": None,
        "status": "skipped",
    }


class ResultsWriter:
    """Appends rows to results.csv incrementally so partial runs survive."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=ROW_FIELDS).writeheader()

    def append(self, row: dict) -> None:
        self.rows.append(row)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=ROW_FIELDS).writerow(row)


def summarize(rows: list[dict], config: dict, path: Path, dataset: str) -> None:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["method"], r["budget"], r["depth"]), []).append(r)

    per_config = {}
    print(f"\n=== {dataset} summary (mean bal_acc over completed runs) ===")
    print(f"  {'method':<11} {'budget':>7} {'depth':>5} {'bal_acc':>16} {'runs':>7}")
    for key in sorted(groups, key=lambda k: (k[0], k[1], str(k[2]))):
        method, budget, depth = key
        grp = groups[key]
        ok = [r for r in grp if r["status"] == "ok"]
        n_ok = len(ok)
        n_skipped = sum(1 for r in grp if r["status"] == "skipped")
        accs = [r["balanced_accuracy"] for r in ok]
        mean_acc = float(np.mean(accs)) if accs else None
        std_acc = float(np.std(accs)) if accs else None
        per_config[f"{method}|{budget}|{depth}"] = {
            "method": method, "budget": budget, "depth": depth,
            "mean_balanced_accuracy": mean_acc,
            "std_balanced_accuracy": std_acc,
            "n_ok": n_ok, "n_skipped": n_skipped,
            "mean_time_total_s": float(np.mean([r["time_total_s"] for r in ok])) if ok else None,
        }
        acc_str = f"{mean_acc:.4f}±{std_acc:.4f}" if mean_acc is not None else "n/a"
        print(f"  {method:<11} {budget:>7} {str(depth):>5} {acc_str:>16} {f'{n_ok}/{len(grp)}':>7}")

    with open(path, "w") as f:
        json.dump({"config": config, "per_config": per_config}, f, indent=2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def parse_args():
    _d = ExperimentConfig()
    p = argparse.ArgumentParser(description="Three-way TS classification comparison")
    p.add_argument("--dataset", default=_d.dataset)
    p.add_argument("--budgets", default="10,100,1000,10000",
                   help="comma list of feature budgets (kernels / formulae)")
    p.add_argument("--depths", default="1,2,3",
                   help="comma list of STL depth_max values")
    p.add_argument("--n_run", type=int, default=_d.n_run)
    p.add_argument("--base_seed", type=int, default=_d.base_seed)
    p.add_argument("--config_budget", type=float, default=1800.0,
                   help="wall-clock seconds per configuration's run loop")
    p.add_argument("--cv", type=int, default=_d.cv,
                   help="shared CV folds for all three heads")
    p.add_argument("--only_temporal", type=lambda x: x.lower() != "false", default=_d.only_temporal)
    p.add_argument("--until_weight", type=float, default=_d.until_weight)
    p.add_argument("--output_dir", default="comparison_results")
    return p.parse_args()


def main() -> None:
    global RocketClassifier
    # Imported here so the rest of the module loads even if aeon is unavailable
    # (e.g. running only the STL methods on a minimal env).
    from aeon.classification.convolution_based import RocketClassifier as _Rocket
    RocketClassifier = _Rocket

    args = parse_args()
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    seeds = [args.base_seed + i for i in range(args.n_run)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    out_dir = Path(args.output_dir) / f"{args.dataset}_{timestamp}_{uid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Output directory: {out_dir}")
    print(f"Dataset: {args.dataset}  budgets: {budgets}  depths: {depths}  "
          f"n_run: {args.n_run}  cv: {args.cv}  config_budget: {args.config_budget}s\n")

    print("Loading dataset...")
    X_tr_raw, y_tr, X_te_raw, y_te = load_dataset(args.dataset)
    print(f"  train: {X_tr_raw.shape}  test: {X_te_raw.shape}")

    writer = ResultsWriter(out_dir / "results.csv")
    feat_build_calls = 0  # invariant check: should equal len(budgets)*len(depths)*n_run

    # --- ROCKET configurations: one per budget ---
    for b in budgets:
        t_start = time.perf_counter()
        for seed in seeds:
            if time.perf_counter() - t_start >= args.config_budget:
                writer.append(make_row(args.dataset, "rocket", b, None, seed, skipped_metrics()))
                continue
            print(f"[rocket  b={b:<6} seed={seed}] fitting...")
            m = run_rocket(X_tr_raw, y_tr, X_te_raw, y_te, b, seed, args.cv)
            writer.append(make_row(args.dataset, "rocket", b, None, seed, m))
            print(f"    bal_acc={m['balanced_accuracy']:.4f}  fit={m['time_fit_s']:.2f}s")

    # --- STL configurations: one per (budget, depth), both heads share features ---
    for b in budgets:
        for d in depths:
            t_start = time.perf_counter()
            for seed in seeds:
                if time.perf_counter() - t_start >= args.config_budget:
                    writer.append(make_row(args.dataset, "stl_linear", b, d, seed, skipped_metrics()))
                    writer.append(make_row(args.dataset, "stl_tree", b, d, seed, skipped_metrics()))
                    continue
                print(f"[stl     b={b:<6} d={d} seed={seed}] building features...")
                config, X_tr_feats, X_te_feats, time_feats_s = build_stl_features(
                    X_tr_raw, y_tr, X_te_raw, b, d, seed, args
                )
                feat_build_calls += 1

                m_lin = run_stl_linear(X_tr_feats, y_tr, X_te_feats, y_te, config, time_feats_s)
                writer.append(make_row(args.dataset, "stl_linear", b, d, seed, m_lin))
                m_tree = run_stl_tree(X_tr_feats, y_tr, X_te_feats, y_te, seed, args.cv, time_feats_s)
                writer.append(make_row(args.dataset, "stl_tree", b, d, seed, m_tree))
                print(f"    feats={time_feats_s:.2f}s  linear={m_lin['balanced_accuracy']:.4f}"
                      f"  tree={m_tree['balanced_accuracy']:.4f}")

    expected_builds = len(budgets) * len(depths) * args.n_run
    print(f"\nFeature builds: {feat_build_calls} (full sweep would be {expected_builds}; "
          f"fewer if any STL configuration hit its time budget)")

    summarize(writer.rows, vars(args), out_dir / "summary.json", args.dataset)
    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
