#!/usr/bin/env python
"""
Validation-only experiment: does dropping duplicate feature columns change the lasso, and
how much fit time does it save? The TEST split is never loaded.

Many sampled formulae compute the same feature up to sign and scale: a single-atom
formula's robustness is affine in its threshold (G[a,b](x > c) = min x - c), and centring
removes the threshold, so formulae differing only in threshold give |r| = 1, as do the
duals G(x > c) and F(x <= c'). Exactly duplicated columns leave the lasso's fitted values
unchanged (only which copy carries the weight is arbitrary) and lambda_max = max |X^T y| / n,
so the path should be the same. This script checks that and times the fit.

Protocol as in scripts/run_ablation.py: per run, a stratified fit/validation split of TRAIN
seeded by base_seed + run, series standardized on the fit part, one formula stream per run.
The bank is built once per (dataset, run, depth) at max(M); the first M columns are the
M-formula bank. For each M and each dedup threshold ("none" = all M columns) the columns
are filtered with features.correlation_filter on the fit features in float64, keeping a
column iff its |r| with every earlier kept column is < threshold, and the lasso is fitted
on the kept columns. All arms of an (M, run) are paired on split, formulae and CV folds.

The filter is run on float64 features: on float32 ones, exact duplicates come out with
|r| within ~4e-7 of 1 on either side, so a threshold of 1.0 would drop only some of them.
Constant columns (all 0 after features._center_and_scale) are dropped by every filter.

Per arm the CSV gets validation balanced accuracy at the 1-SE and best-CV lambdas, the
selected sizes and lambda indices, and the fraction of validation series whose predicted
class equals that of the "none" arm (at both lambdas). Timing: time_dedup_s is the filter,
time_fit_s the lasso; with --fit_repeats R the fit is repeated R times and the minimum kept.
The arm order is reversed on odd runs, so that warm-up or cache effects do not favour one
side. n_struct_dup counts formulae that are a structural copy of an earlier one (a chain of
G/F over one atom with the same windows and variable, up to negation), i.e. those that
could be skipped before feature extraction.

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
from sklearn.model_selection import StratifiedShuffleSplit

# Run correctly regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from torcheck.stl import Atom, Globally, Eventually
from stlrocket.data import standardize_series
from stlrocket.features import build_formula_bank, correlation_filter
from stlrocket.classifier import train_classifier, evaluate_classifier, selection_diagnostics
from run_ablation import load_train, make_config, n_selected, split_ok, float_or_none, str2bool


ROW_FIELDS = [
    "dataset", "run", "seed", "val_ratio", "depth_max", "n_formulas", "dedup_threshold",
    "n_bank", "n_constant", "n_struct_dup", "fit_intercept",
    "val_balanced_accuracy", "fit_balanced_accuracy", "n_selected",
    "val_balanced_accuracy_cvmax", "n_selected_cvmax", "lambda_index", "lambda_index_cvmax",
    "n_lambda", "val_pred_agreement", "val_pred_agreement_cvmax",
    "time_feats_s", "time_dedup_s", "time_fit_s", "status",
]


def structural_key(phi):
    """Key shared by single-atom G/F chains that give the same feature up to sign and an
    additive constant: variable and (operator, window) of every level, with G/F swapped
    when the atom is x <= c (negating x <= c gives x > c and swaps G and F). None for
    every other formula."""
    chain, node = [], phi
    while not isinstance(node, Atom):
        if not isinstance(node, (Globally, Eventually)):
            return None
        window = (("unbound",) if node.unbound else
                  ("right_unbound", node.left_time_bound) if node.right_unbound else
                  (node.left_time_bound, node.right_time_bound))
        chain.append((isinstance(node, Globally), window))
        node = node.child
    return node.var_index, tuple((is_g != node.lte, w) for is_g, w in chain)


def struct_dup_mask(formulas: list) -> np.ndarray:
    seen, dup = set(), np.zeros(len(formulas), dtype=bool)
    for j, phi in enumerate(formulas):
        key = structural_key(phi)
        if key is not None:
            dup[j] = key in seen
            seen.add(key)
    return dup


def predict(model, F, lamb=None) -> np.ndarray:
    return np.asarray(model.predict(F, lamb=lamb)).reshape(len(F), -1)[:, 0]


def fit_arm(F_fit, F_val, y_fit, y_val, cfg, seed, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model = train_classifier(F_fit, y_fit, cfg, seed=seed)
        times.append(time.perf_counter() - t0)
    sel = selection_diagnostics(model)
    i_1se, i_max = sel["index_1se"], sel["index_cvmax"]
    row = {
        "time_fit_s": round(min(times), 4),
        "val_balanced_accuracy": evaluate_classifier(model, F_val, y_val)["balanced_accuracy"],
        "fit_balanced_accuracy": evaluate_classifier(model, F_fit, y_fit)["balanced_accuracy"],
        "n_selected": n_selected(model),
        "val_balanced_accuracy_cvmax": evaluate_classifier(
            model, F_val, y_val, lamb=model.lambda_max_)["balanced_accuracy"],
        "n_selected_cvmax": sel["n_selected_path"][i_max],
        "lambda_index": i_1se, "lambda_index_cvmax": i_max, "n_lambda": sel["n_lambda"],
        "status": "ok",
    }
    return row, predict(model, F_val), predict(model, F_val, lamb=model.lambda_max_)


def run_dataset(args, dataset: str, write) -> None:
    X, y = load_train(dataset)
    classes = np.unique(y)
    M_max = max(args.n_formulas)

    for run in (args.runs if args.runs is not None else range(args.n_runs)):
        seed = args.base_seed + run
        base = {"dataset": dataset, "run": run, "seed": seed, "val_ratio": args.val_ratio,
                "fit_intercept": args.fit_intercept}
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=seed)
            fit_idx, val_idx = next(splitter.split(X, y))
        except ValueError as e:  # too few samples for the requested ratio
            write({**base, "status": f"split_error: {e}"})
            continue
        y_fit, y_val = y[fit_idx], y[val_idx]
        if not split_ok(y_fit, y_val, classes):
            write({**base, "status": "skipped: too few samples per class"})
            continue
        X_fit, X_val = standardize_series(X[fit_idx], X[val_idx])

        for depth_max in args.depths:
            cfg = make_config(args, dataset, M_max, depth_max, 0.0, args.fit_intercept)
            t0 = time.perf_counter()
            formulas, F_fit_all, F_val_all, _ = build_formula_bank(X_fit, X_val, y_fit, cfg, seed)
            t_feats = time.perf_counter() - t0
            struct_dup = struct_dup_mask(formulas)
            F64 = F_fit_all.astype(np.float64)

            for M in args.n_formulas:
                bank = {**base, "depth_max": depth_max, "n_formulas": M,
                        "n_constant": int((F_fit_all[:, :M].std(axis=0) == 0).sum()),
                        "n_struct_dup": int(struct_dup[:M].sum()),
                        # Bank built once at M_max; cost is linear in M.
                        "time_feats_s": round(t_feats * M / M_max, 4)}
                arms = list(args.dedup_thresholds)
                if run % 2:
                    arms.reverse()
                rows, preds = {}, {}
                for th in arms:
                    row = {**bank, "dedup_threshold": th}
                    try:
                        t0 = time.perf_counter()
                        cols = np.arange(M) if th is None else correlation_filter(F64[:, :M], th)
                        row["time_dedup_s"] = round(time.perf_counter() - t0, 4)
                        row["n_bank"] = len(cols)
                        fitted, p_1se, p_max = fit_arm(F_fit_all[:, cols], F_val_all[:, cols], y_fit, y_val,
                                                       cfg, seed, args.fit_repeats)
                        row.update(fitted)
                        preds[th] = (p_1se, p_max)
                    except Exception as e:
                        row["status"] = f"error: {type(e).__name__}: {e}"
                    rows[th] = row

                ref = preds.get(None)
                for th in args.dedup_thresholds:
                    row = rows[th]
                    if ref is not None and th in preds:
                        row["val_pred_agreement"] = float(np.mean(preds[th][0] == ref[0]))
                        row["val_pred_agreement_cvmax"] = float(np.mean(preds[th][1] == ref[1]))
                    write(row)
                print(f"{dataset} run={run} depth={depth_max} M={M} -> " + ", ".join(
                    f"{th}: n_bank={rows[th].get('n_bank')} fit={rows[th].get('time_fit_s')}s "
                    f"val={rows[th].get('val_balanced_accuracy')} agree={rows[th].get('val_pred_agreement')}"
                    for th in args.dedup_thresholds), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--runs", type=int, nargs="+", default=None)
    p.add_argument("--val_ratio", type=float, default=0.25)
    p.add_argument("--depths", type=int, nargs="+", default=[2])
    p.add_argument("--n_formulas", type=int, nargs="+", default=[10000])
    # "none" (all columns) must be among them: it is the reference of val_pred_agreement.
    p.add_argument("--dedup_thresholds", type=float_or_none, nargs="+", default=[None, 0.999999, 0.9999, 0.999])
    p.add_argument("--fit_repeats", type=int, default=1)
    # Lasso settings of run_ablation.sh.
    p.add_argument("--fit_intercept", type=str2bool, default=False)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--cut_point", type=float, default=1.0)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, default="results/ablation/dedup/dedup.csv")
    args = p.parse_args()
    if None not in args.dedup_thresholds:
        p.error("--dedup_thresholds must include none, the reference arm")
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

        def write(row: dict) -> None:
            writer.writerow(row)
            f.flush()

        for dataset in args.datasets:
            run_dataset(args, dataset, write)


if __name__ == "__main__":
    main()
