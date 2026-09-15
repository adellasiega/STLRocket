#!/usr/bin/env python
"""
Two-way classification comparison on a single time-series dataset.

Methods:
  1. stl_linear  -- STL robustness features -> glmnet LogitNet (L1 logistic regression)
  2. stl_tree    -- same STL robustness features -> sklearn DecisionTreeClassifier (CV-tuned)

For each "budget" b in {10,100,1000,10000} (formula count for STL) we run n_run seeds.
The STL approaches additionally sweep depth_max in {1,2,3}; the two STL heads share a
single robustness feature matrix per (budget, depth, seed) that is computed exactly once.

Hyperparameter fairness: each head CV-tunes its primary capacity/regularization knob with
the same --cv fold count (L1 lambda for glmnet, tree structure for the decision tree).

Cost control: each configuration gets a wall-clock budget (--config_budget). A configuration
is one STL (budget, depth) pair-of-heads. Seeds are the inner loop; before each seed we
check elapsed time and skip the remaining seeds once the budget is spent. Completed runs
are written to results.csv immediately, so partial results survive.
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

import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# Run correctly regardless of CWD (the SLURM sweep invokes this by absolute path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.features import build_formula_bank
from stlrocket.classifier import available_cpus, train_classifier, evaluate_classifier


ROW_FIELDS = [
    "dataset", "method", "budget", "depth", "until_weight", "only_temporal", "seed",
    "balanced_accuracy", "time_fit_s", "time_feats_s", "time_total_s", "status",
]


# ---------------------------------------------------------------------------
# Per-method fits
# ---------------------------------------------------------------------------

def build_stl_features(X_tr_raw, y_tr, X_te_raw, n_formulas, depth_max, seed, args):
    """Build STL robustness features once for both STL heads.

    Returns (config, X_tr_feats, X_te_feats, time_feats_s).
    """
    config = ExperimentConfig(
        dataset=args.dataset,
        n_formulas=n_formulas,
        depth_max=depth_max,
        only_temporal=args.only_temporal,
        until_weight=args.until_weight,
        cv=args.cv,
        pool_size=0,
        precision_threshold=0.0,
        simplify_agreement=0.0,
        simplify_min_gain=0.0,
        simplify_decimals=0,
        n_run=args.n_run,
        base_seed=args.base_seed,
        explain=False,
        output_dir=args.output_dir,
        device="cpu",
    )
    t0 = time.perf_counter()
    _formulas, X_tr_feats, X_te_feats = build_formula_bank(X_tr_raw, X_te_raw, config, seed)
    time_feats_s = round(time.perf_counter() - t0, 4)
    return config, X_tr_feats, X_te_feats, time_feats_s


def run_stl_linear(X_tr_feats, y_tr, X_te_feats, y_te, config, seed, time_feats_s) -> dict:
    t0 = time.perf_counter()
    model = train_classifier(X_tr_feats, y_tr, config, seed=seed)
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
    # Guard against tiny per-class counts breaking StratifiedKFold. Floor of 3
    # matches train_classifier, so both heads CV-tune over identical folds.
    _, counts = np.unique(y_tr, return_counts=True)
    n_splits = max(3, min(cv, int(counts.min())))

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
        # 36 param combos x n_splits folds, so the full allocation is usable here.
        n_jobs=available_cpus(),
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

def make_row(args, method, budget, depth, seed, metrics) -> dict:
    return {
        "dataset": args.dataset,
        "method": method,
        "budget": budget,
        "depth": depth if depth is not None else "",
        # Ablation axes are recorded per row so results.csv is self-describing:
        # many single-config runs can be concatenated without joining config.json.
        "until_weight": args.until_weight,
        "only_temporal": args.only_temporal,
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
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=ROW_FIELDS).writeheader()

    def append(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=ROW_FIELDS).writerow(row)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Two-way TS classification comparison")
    p.add_argument("--dataset", default="BasicMotions")
    p.add_argument("--budgets", default="10,100,1000,10000",
                   help="comma list of feature budgets (formula count)")
    p.add_argument("--depths", default="1,2,3",
                   help="comma list of STL depth_max values")
    p.add_argument("--n_run", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--config_budget", type=float, default=39600.0,
                   help="wall-clock seconds per configuration's run loop "
                        "(default 11h, sized to fit inside a 12h SLURM job)")
    p.add_argument("--cv", type=int, default=5,
                   help="shared CV folds for both heads; each head clamps it down "
                        "when a class has fewer members than folds")
    p.add_argument("--only_temporal", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--until_weight", type=float, default=0.0)
    p.add_argument("--output_dir", default="comparison_results")
    return p.parse_args()


def main() -> None:
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

    # --- STL configurations: one per (budget, depth), both heads share features ---
    for b in budgets:
        for d in depths:
            t_start = time.perf_counter()
            for seed in seeds:
                if time.perf_counter() - t_start >= args.config_budget:
                    writer.append(make_row(args, "stl_linear", b, d, seed, skipped_metrics()))
                    writer.append(make_row(args, "stl_tree", b, d, seed, skipped_metrics()))
                    continue
                print(f"[stl     b={b:<6} d={d} seed={seed}] building features...")
                config, X_tr_feats, X_te_feats, time_feats_s = build_stl_features(
                    X_tr_raw, y_tr, X_te_raw, b, d, seed, args
                )
                feat_build_calls += 1

                m_lin = run_stl_linear(X_tr_feats, y_tr, X_te_feats, y_te, config, seed, time_feats_s)
                writer.append(make_row(args, "stl_linear", b, d, seed, m_lin))
                m_tree = run_stl_tree(X_tr_feats, y_tr, X_te_feats, y_te, seed, args.cv, time_feats_s)
                writer.append(make_row(args, "stl_tree", b, d, seed, m_tree))
                print(f"    feats={time_feats_s:.2f}s  linear={m_lin['balanced_accuracy']:.4f}"
                      f"  tree={m_tree['balanced_accuracy']:.4f}")

    expected_builds = len(budgets) * len(depths) * args.n_run
    print(f"\nFeature builds: {feat_build_calls} (full sweep would be {expected_builds}; "
          f"fewer if any STL configuration hit its time budget)")

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
