#!/usr/bin/env python
"""
Baseline ROCKET classification on a single time-series dataset.

Methods:
  1. rocket_ridge   -- ROCKET random convolutional kernels -> RidgeClassifierCV
                       (the canonical head from the original ROCKET paper)
  2. rocket_linear  -- same ROCKET features -> glmnet LogitNet (L1 logistic
                       regression), i.e. the identical head STLRocket's
                       stl_linear uses, so the only thing that differs between
                       stl_linear and rocket_linear is the feature map

This mirrors run_stlinear_stltree.py so the two results.csv files concatenate
directly: same ROW_FIELDS, same balanced-accuracy metric, same seed protocol,
same per-configuration wall-clock budget, same incremental writer.

Budget axis: --budgets is a comma list of ROCKET *kernel* counts, matched against
the STL formula counts. ROCKET emits 2 features per kernel (PPV and max), so a
budget of b kernels yields 2b columns; the STL bank yields b. Compare on the
budget axis, not the raw column count -- the honest cost comparison is wall-clock.

The depth and until_weight axes are STL-specific and do not apply here, so those
columns are written empty. That keeps one concatenated frame where a groupby on
(dataset, method, budget) puts STL and ROCKET side by side.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import sys

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# Run correctly regardless of CWD (the SLURM sweep invokes this by absolute path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.classifier import available_cpus, train_classifier, evaluate_classifier

# Identical to run_stlinear_stltree.py so the two CSVs concatenate without a schema fix.
ROW_FIELDS = [
    "dataset", "method", "budget", "depth", "until_weight", "only_temporal", "seed",
    "balanced_accuracy", "time_fit_s", "time_feats_s", "time_total_s", "status",
]

# The alpha grid from the ROCKET paper's reference implementation.
RIDGE_ALPHAS = np.logspace(-3, 3, 10)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_rocket_features(X_tr_raw, X_te_raw, n_kernels, seed, args):
    """Fit ROCKET on train only, then apply to both splits.

    Fitting on train alone matters: ROCKET's kernel biases are sampled from the
    fitted data's convolution outputs, so fit_transform on the full set would
    leak test distribution into the feature map.
    """
    from aeon.transformations.collection.convolution_based import Rocket

    t0 = time.perf_counter()
    rocket = Rocket(
        n_kernels=n_kernels,
        random_state=seed,
        n_jobs=available_cpus(),
    )
    X_tr_feats = np.asarray(rocket.fit_transform(X_tr_raw), dtype=np.float64)
    X_te_feats = np.asarray(rocket.transform(X_te_raw), dtype=np.float64)
    time_feats_s = round(time.perf_counter() - t0, 4)

    # ROCKET's PPV/max pooling can emit non-finite values when a kernel produces a
    # constant response on a degenerate channel; both heads below would raise on
    # those, so neutralise them here rather than losing the whole seed.
    np.nan_to_num(X_tr_feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_te_feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return X_tr_feats, X_te_feats, time_feats_s


# ---------------------------------------------------------------------------
# Per-method fits
# ---------------------------------------------------------------------------

def run_rocket_ridge(X_tr_feats, y_tr, X_te_feats, y_te, seed, cv, time_feats_s) -> dict:
    # Mirror the fold clamping both STL heads use, so every head in the comparison
    # CV-tunes over the same number of folds on the same dataset.
    _, counts = np.unique(y_tr, return_counts=True)
    n_splits = max(3, min(cv, int(counts.min())))

    model = RidgeClassifierCV(
        alphas=RIDGE_ALPHAS,
        class_weight="balanced",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
    )
    t0 = time.perf_counter()
    model.fit(X_tr_feats, y_tr)
    t1 = time.perf_counter()
    y_pred = model.predict(X_te_feats)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "time_fit_s": round(t1 - t0, 4),
        "time_feats_s": time_feats_s,
        "time_total_s": round(time_feats_s + (t1 - t0), 4),
        "status": "ok",
    }


def run_rocket_linear(X_tr_feats, y_tr, X_te_feats, y_te, config, seed, time_feats_s) -> dict:
    """Same glmnet head as stl_linear, reusing the shared helpers verbatim."""
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


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def make_row(args, method, budget, seed, metrics) -> dict:
    return {
        "dataset": args.dataset,
        "method": method,
        "budget": budget,
        # depth and until_weight are STL-only axes; left empty so a concatenated
        # frame can filter ROCKET rows with a simple isna() on either column.
        "depth": "",
        "until_weight": "",
        "only_temporal": "",
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
    p = argparse.ArgumentParser(description="ROCKET baseline for the STLRocket comparison")
    p.add_argument("--dataset", default="BasicMotions")
    p.add_argument("--budgets", default="10,100,1000,10000",
                   help="comma list of ROCKET kernel counts (matched to STL formula counts)")
    p.add_argument("--n_run", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--config_budget", type=float, default=39600.0,
                   help="wall-clock seconds per configuration's run loop "
                        "(default 11h, sized to fit inside a 12h SLURM job)")
    p.add_argument("--cv", type=int, default=5,
                   help="shared CV folds for both heads; each head clamps it down "
                        "when a class has fewer members than folds")
    p.add_argument("--output_dir", default="comparison_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    seeds = [args.base_seed + i for i in range(args.n_run)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    out_dir = Path(args.output_dir) / f"rocket_{args.dataset}_{timestamp}_{uid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Output directory: {out_dir}")
    print(f"Dataset: {args.dataset}  budgets(kernels): {budgets}  "
          f"n_run: {args.n_run}  cv: {args.cv}  config_budget: {args.config_budget}s\n")

    print("Loading dataset...")
    X_tr_raw, y_tr, X_te_raw, y_te = load_dataset(args.dataset)
    print(f"  train: {X_tr_raw.shape}  test: {X_te_raw.shape}")

    # train_classifier reads only cv/n_run-independent fields off the config, but
    # it is a frozen dataclass, so build one valid instance and reuse it.
    config = ExperimentConfig(
        dataset=args.dataset,
        n_formulas=max(budgets),
        depth_max=1,
        only_temporal=True,
        until_weight=0.0,
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

    writer = ResultsWriter(out_dir / "results.csv")
    feat_build_calls = 0  # invariant check: should equal len(budgets)*n_run

    # --- one configuration per budget; both heads share a feature matrix ---
    for b in budgets:
        t_start = time.perf_counter()
        for seed in seeds:
            if time.perf_counter() - t_start >= args.config_budget:
                writer.append(make_row(args, "rocket_ridge", b, seed, skipped_metrics()))
                writer.append(make_row(args, "rocket_linear", b, seed, skipped_metrics()))
                continue
            print(f"[rocket  k={b:<6} seed={seed}] building features...")
            X_tr_feats, X_te_feats, time_feats_s = build_rocket_features(
                X_tr_raw, X_te_raw, b, seed, args
            )
            feat_build_calls += 1

            m_ridge = run_rocket_ridge(X_tr_feats, y_tr, X_te_feats, y_te,
                                       seed, args.cv, time_feats_s)
            writer.append(make_row(args, "rocket_ridge", b, seed, m_ridge))
            m_lin = run_rocket_linear(X_tr_feats, y_tr, X_te_feats, y_te,
                                      config, seed, time_feats_s)
            writer.append(make_row(args, "rocket_linear", b, seed, m_lin))
            print(f"    feats={time_feats_s:.2f}s  ridge={m_ridge['balanced_accuracy']:.4f}"
                  f"  linear={m_lin['balanced_accuracy']:.4f}")

    expected_builds = len(budgets) * args.n_run
    print(f"\nFeature builds: {feat_build_calls} (full sweep would be {expected_builds}; "
          f"fewer if any configuration hit its time budget)")

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
