#!/usr/bin/env python
"""
Validation-only experiment: ridge vs lasso on correlation-filtered STL banks. The TEST
split is never loaded.

Protocol as in scripts/run_ablation.py (and run_corr_ablation.sh): per run, a stratified
fit/validation split of TRAIN seeded by base_seed + run, series standardized on the fit
part, one formula stream per run. For each correlation threshold ("none" = unfiltered
bank) the bank is built once and both classifiers are fitted on it, so all arms are
paired on split and formula stream.

Classifiers:
  lasso  stlrocket.classifier.train_classifier (glmnet, class-balanced weights),
         validation at the 1-SE lambda and at the best-CV lambda
  ridge  sklearn RidgeClassifierCV (one-vs-rest least squares, as in ROCKET), alpha by
         efficient leave-one-out over --ridge_alphas, class-balanced sample weights

Rows are appended to the CSV as soon as they are computed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

# Run correctly regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stlrocket.data import standardize_series
from stlrocket.features import build_formula_bank, build_decorrelated_formula_bank
from stlrocket.classifier import train_classifier, evaluate_classifier, selection_diagnostics
from run_ablation import load_train, make_config, n_selected, split_ok, float_or_none


ROW_FIELDS = [
    "dataset", "run", "seed", "val_ratio", "depth_max", "n_formulas", "corr_threshold",
    "n_sampled", "n_bank", "classifier", "val_balanced_accuracy", "fit_balanced_accuracy",
    "n_selected", "val_balanced_accuracy_cvmax", "n_selected_cvmax", "ridge_alpha",
    "ridge_alpha_at_edge", "time_feats_s", "time_fit_s", "status",
]


def bacc(y, y_pred) -> float:
    return float(balanced_accuracy_score(y, y_pred))


def run_dataset(args, dataset: str, write) -> None:
    X, y = load_train(dataset)
    classes = np.unique(y)
    M = args.n_formulas

    for run in (args.runs if args.runs is not None else range(args.n_runs)):
        seed = args.base_seed + run
        base = {"dataset": dataset, "run": run, "seed": seed, "val_ratio": args.val_ratio,
                "depth_max": args.depth, "n_formulas": M}
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=seed)
        fit_idx, val_idx = next(splitter.split(X, y))
        y_fit, y_val = y[fit_idx], y[val_idx]
        X_fit, X_val = standardize_series(X[fit_idx], X[val_idx])
        if not split_ok(y_fit, y_val, classes):
            write({**base, "status": "skipped: too few samples per class"})
            continue

        for corr_threshold in args.corr_thresholds:
            cfg = make_config(args, dataset, M, args.depth, 0.0, args.fit_intercept)
            t0 = time.perf_counter()
            if corr_threshold is None:
                _, F_fit, F_val, _ = build_formula_bank(X_fit, X_val, y_fit, cfg, seed)
                n_sampled = M
            else:
                _, F_fit, F_val, _, n_sampled = build_decorrelated_formula_bank(
                    X_fit, X_val, y_fit, cfg, seed, corr_threshold, max_rounds=args.corr_max_rounds)
            t_feats = time.perf_counter() - t0
            bank = {**base, "corr_threshold": corr_threshold, "n_sampled": n_sampled,
                    "n_bank": F_fit.shape[1], "time_feats_s": round(t_feats, 4)}

            # Lasso: the STLRocket classifier.
            row = {**bank, "classifier": "lasso"}
            try:
                t0 = time.perf_counter()
                model = train_classifier(F_fit, y_fit, cfg, seed=seed)
                row["time_fit_s"] = round(time.perf_counter() - t0, 4)
                row["val_balanced_accuracy"] = evaluate_classifier(model, F_val, y_val)["balanced_accuracy"]
                row["fit_balanced_accuracy"] = evaluate_classifier(model, F_fit, y_fit)["balanced_accuracy"]
                row["n_selected"] = n_selected(model)
                row["val_balanced_accuracy_cvmax"] = evaluate_classifier(
                    model, F_val, y_val, lamb=model.lambda_max_)["balanced_accuracy"]
                sel = selection_diagnostics(model)
                row["n_selected_cvmax"] = sel["n_selected_path"][sel["index_cvmax"]]
                row["status"] = "ok"
            except Exception as e:
                row["status"] = f"error: {type(e).__name__}: {e}"
            write(row)

            # Ridge.
            row = {**bank, "classifier": "ridge"}
            try:
                t0 = time.perf_counter()
                ridge = RidgeClassifierCV(alphas=args.ridge_alphas, fit_intercept=args.ridge_fit_intercept)
                ridge.fit(F_fit, y_fit, sample_weight=compute_sample_weight("balanced", y_fit))
                row["time_fit_s"] = round(time.perf_counter() - t0, 4)
                row["val_balanced_accuracy"] = bacc(y_val, ridge.predict(F_val))
                row["fit_balanced_accuracy"] = bacc(y_fit, ridge.predict(F_fit))
                row["n_selected"] = int(np.any(np.atleast_2d(ridge.coef_) != 0, axis=0).sum())
                row["ridge_alpha"] = float(ridge.alpha_)
                row["ridge_alpha_at_edge"] = bool(ridge.alpha_ in (min(args.ridge_alphas), max(args.ridge_alphas)))
                row["status"] = "ok"
            except Exception as e:
                row["status"] = f"error: {type(e).__name__}: {e}"
            write(row)
            print(f"{dataset} run={run} corr={corr_threshold} n_bank={F_fit.shape[1]} feats={t_feats:.0f}s", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--runs", type=int, nargs="+", default=None)
    p.add_argument("--val_ratio", type=float, default=0.25)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--n_formulas", type=int, default=10000)
    p.add_argument("--corr_thresholds", type=float_or_none, nargs="+", default=[None, 0.99, 0.95])
    p.add_argument("--corr_max_rounds", type=int, default=100)
    # Lasso settings of run_corr_ablation.sh.
    p.add_argument("--fit_intercept", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--cut_point", type=float, default=1.0)
    # ROCKET uses logspace(-3, 3, 10); with M=10000 unit-variance columns the Gram
    # eigenvalues are ~1e4 and above, so the grid is extended to where the penalty bites.
    p.add_argument("--ridge_alphas", type=float, nargs="+", default=np.logspace(-3, 7, 21).tolist())
    p.add_argument("--ridge_fit_intercept", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, default="results/ablation/ridge_corr_d2/ridge_corr.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".args.json").write_text(json.dumps(vars(args), indent=2))
    new_file = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if new_file:
            writer.writeheader()

        def write(row: dict) -> None:
            writer.writerow(row)
            f.flush()

        for dataset in args.datasets:
            run_dataset(args, dataset, write)


if __name__ == "__main__":
    main()
