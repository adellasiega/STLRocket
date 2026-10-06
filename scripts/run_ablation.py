#!/usr/bin/env python
"""
Validation-only ablation over depth_max, number of formulas M, until_weight,
correlation-filter threshold, threshold calibration, SCL operator weight and
fit_intercept. The TEST split is never loaded.

For each dataset and run, the TRAIN split is divided into fit/validation with a
stratified split seeded by base_seed + run. All configurations in a run share that
split and the formula seed, so results are paired across configurations.

Features are built once per (dataset, run, depth_max, until_weight, corr_threshold,
calibrate, scl_weight) at max(M): the sampler draws formulas i.i.d. from one seeded
stream, the correlation filter is greedy in stream order, calibration draws from its own stream in
formula order and standardization is per column, so the first M columns are exactly the
M-formula bank. The filter and calibration only look at the fit part. fit_intercept does
not affect features, so both values reuse the same matrix.

Rows are appended to the CSV as soon as they are computed, so partial results survive.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Run correctly regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket.config import ExperimentConfig
from stlrocket.data import _fill_missing
from stlrocket.features import build_formula_bank, build_decorrelated_formula_bank
from stlrocket.classifier import train_classifier, evaluate_classifier


ROW_FIELDS = [
    "dataset", "run", "seed", "val_ratio", "depth_max", "n_formulas", "until_weight",
    "corr_threshold", "n_sampled", "calibrate", "scl_weight",
    "fit_intercept", "val_balanced_accuracy", "fit_balanced_accuracy", "n_selected",
    "time_feats_s", "time_fit_s", "status",
]

# train_classifier never uses fewer than 3 CV folds, so each class needs at least
# 3 samples in the fit part.
MIN_FIT_PER_CLASS = 3


def load_train(name: str) -> tuple[np.ndarray, np.ndarray]:
    """TRAIN split only, missing values filled as in stlrocket.data.load_dataset."""
    from aeon.datasets import load_classification

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        X, y = load_classification(name, split="TRAIN")
    return _fill_missing(X.astype(np.float32)), y


def make_config(args, dataset: str, n_formulas: int, depth_max: int,
                until_weight: float, fit_intercept: bool, calibrate: bool = False,
                scl_weight: float = 0.0) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset,
        n_formulas=n_formulas,
        depth_max=depth_max,
        until_weight=until_weight,
        cv=args.cv,
        cut_point=args.cut_point,
        fit_intercept=fit_intercept,
        explain=False,
        pool_size=0,
        precision_threshold=0.0,
        simplify_agreement=0.0,
        simplify_min_gain=0.0,
        simplify_decimals=0,
        n_run=args.n_runs,
        base_seed=args.base_seed,
        output_dir=str(Path(args.output).parent),
        device=args.device,
        calibrate_thresholds=calibrate,
        scl_weight=scl_weight,
    )


def n_selected(model) -> int:
    """Number of formulas with a nonzero coefficient for at least one class."""
    coef = np.atleast_2d(model.coef_)
    return int(np.any(coef != 0, axis=0).sum())


def split_ok(y_fit: np.ndarray, y_val: np.ndarray, classes: np.ndarray) -> bool:
    fit_counts = np.array([(y_fit == c).sum() for c in classes])
    val_counts = np.array([(y_val == c).sum() for c in classes])
    return bool(fit_counts.min() >= MIN_FIT_PER_CLASS and val_counts.min() >= 1)


def run_dataset(args, dataset: str, writer, f) -> None:
    X, y = load_train(dataset)
    classes = np.unique(y)
    M_max = max(args.n_formulas)

    def write(row: dict) -> None:
        writer.writerow(row)
        f.flush()

    for run in (args.runs if args.runs is not None else range(args.n_runs)):
        seed = args.base_seed + run
        base = {"dataset": dataset, "run": run, "seed": seed, "val_ratio": args.val_ratio}

        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=seed)
            fit_idx, val_idx = next(splitter.split(X, y))
        except ValueError as e:  # too few samples for the requested ratio
            write({**base, "status": f"split_error: {e}"})
            continue
        X_fit, y_fit, X_val, y_val = X[fit_idx], y[fit_idx], X[val_idx], y[val_idx]
        if not split_ok(y_fit, y_val, classes):
            write({**base, "status": "skipped: too few samples per class"})
            continue

        for depth_max in args.depths:
            for until_weight, corr_threshold, calibrate, scl_weight in itertools.product(
                    args.until_weights, args.corr_thresholds, args.calibrate, args.scl_weights):
                cfg = make_config(args, dataset, M_max, depth_max, until_weight, True, calibrate, scl_weight)
                t0 = time.perf_counter()
                if corr_threshold is None:
                    _, F_fit, F_val, _, _ = build_formula_bank(X_fit, X_val, cfg, seed)
                    n_sampled = M_max
                else:
                    _, F_fit, F_val, _, _, n_sampled = build_decorrelated_formula_bank(
                        X_fit, X_val, cfg, seed, corr_threshold)
                t_feats = time.perf_counter() - t0
                if F_fit.shape[1] < M_max:
                    print(f"warning: only {F_fit.shape[1]} formulas passed the filter, "
                          f"M above that is evaluated on the smaller bank", flush=True)

                for M in args.n_formulas:
                    for fit_intercept in args.fit_intercept:
                        row = {
                            **base,
                            "depth_max": depth_max,
                            "n_formulas": M,
                            "until_weight": until_weight,
                            "corr_threshold": corr_threshold,
                            "n_sampled": n_sampled,
                            "calibrate": calibrate,
                            "scl_weight": scl_weight,
                            "fit_intercept": fit_intercept,
                            # Bank built once at M_max; cost is linear in M.
                            "time_feats_s": round(t_feats * M / M_max, 4),
                        }
                        cfg_M = dataclasses.replace(cfg, n_formulas=M, fit_intercept=fit_intercept)
                        try:
                            t0 = time.perf_counter()
                            model = train_classifier(F_fit[:, :M], y_fit, cfg_M, seed=seed)
                            row["time_fit_s"] = round(time.perf_counter() - t0, 4)
                            row["val_balanced_accuracy"] = evaluate_classifier(
                                model, F_val[:, :M], y_val)["balanced_accuracy"]
                            row["fit_balanced_accuracy"] = evaluate_classifier(
                                model, F_fit[:, :M], y_fit)["balanced_accuracy"]
                            row["n_selected"] = n_selected(model)
                            row["status"] = "ok"
                        except Exception as e:
                            row["status"] = f"error: {type(e).__name__}: {e}"
                        write(row)
                        print(f"{dataset} run={run} depth={depth_max} until={until_weight} corr={corr_threshold} calib={calibrate} scl={scl_weight} "
                              f"M={M} intercept={fit_intercept} -> "
                              f"val={row.get('val_balanced_accuracy')} [{row['status']}]", flush=True)


def str2bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def float_or_none(s: str) -> float | None:
    return None if s.lower() == "none" else float(s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--n_runs", type=int, default=5)
    # Run only these run indices (seed = base_seed + run), e.g. one per SLURM task.
    # Overrides --n_runs; a task with --runs 3 reproduces run 3 of a full run.
    p.add_argument("--runs", type=int, nargs="+", default=None)
    p.add_argument("--val_ratio", type=float, default=0.25)
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n_formulas", type=int, nargs="+", default=[100, 1000, 10000])
    p.add_argument("--until_weights", type=float, nargs="+", default=[0.0, 1.0])
    # Correlation-filter thresholds; "none" = no filter.
    p.add_argument("--corr_thresholds", type=float_or_none, nargs="+", default=[None])
    # Calibrate formula thresholds on the fit part (features.calibrate_thresholds).
    p.add_argument("--calibrate", type=str2bool, nargs="+", default=[False])
    # Weight of the SCL Fraction operator in the sampler, relative to G and F (1 each); 0 = off.
    p.add_argument("--scl_weights", type=float, nargs="+", default=[0.0])
    p.add_argument("--fit_intercept", type=str2bool, nargs="+", default=[True, False])
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--cut_point", type=float, default=1.0)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, default="results/ablation/ablation.csv")
    args = p.parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        p.error("--val_ratio must be in (0, 1)")
    return args


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
        for dataset in args.datasets:
            run_dataset(args, dataset, writer, f)


if __name__ == "__main__":
    main()
