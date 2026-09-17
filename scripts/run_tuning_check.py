#!/usr/bin/env python
"""
Check whether gradient tuning of the assembled *global* explanation (per class,
post-simplification) improves held-out macro F1, across datasets.

Runs the same explanation pipeline as run_explanations.py (best stl_linear
depth per dataset from tables/results_long.csv, M=10000, until_weight=0), then
for each class's global formula calls stlrocket.tuning.finetune_formula with
objective="f1" and re-scores on the test set. Writes one row per dataset to
<output_dir>/tuning_check_summary.csv with before/after macro F1 (train+test),
so per-dataset SLURM array tasks accumulate into one shared file.

This does not touch results/explanations or any existing script's output --
separate output_dir, separate script.

Usage
-----
    python scripts/run_tuning_check.py --datasets Libras
    python scripts/run_tuning_check.py   # every dataset in the results table
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.features import build_formula_bank
from stlrocket.classifier import train_classifier, evaluate_classifier
from stlrocket.explanations import build_global_explanations
from stlrocket.evaluation import evaluate_global
from stlrocket.tuning import finetune_formula
from stlrocket.formula_sampler import F0

SUMMARY_FIELDS = [
    "dataset", "n_formulas", "depth_max", "until_weight", "seed",
    "balanced_accuracy", "n_classes",
    "global_macro_f1_train_before", "global_macro_f1_test_before",
    "global_macro_f1_train_after", "global_macro_f1_test_after",
    "delta_macro_f1_test", "n_classes_changed", "n_classes_improved",
    "n_classes_worsened", "mean_depth_before", "mean_depth_after",
    "time_explain_s", "time_tune_s",
]


def load_best_depths(table_path: Path, method: str = "stl_linear") -> dict[str, int]:
    import pandas as pd
    df = pd.read_csv(table_path)
    sub = df[df["method"] == method]
    if sub.empty:
        raise SystemExit(f"no rows for method={method!r} in {table_path}")
    return {str(r.dataset): int(r.depth) for r in sub.itertuples()}


def run_dataset(dataset: str, depth_max: int, args) -> dict:
    print(f"\n=== {dataset} (M={args.budget}, depth_max={depth_max}, "
          f"until_weight={args.until_weight}, seed={args.seed}) ===", flush=True)

    config = ExperimentConfig(
        dataset=dataset, n_formulas=args.budget, depth_max=depth_max,
        only_temporal=args.only_temporal, until_weight=args.until_weight,
        cv=args.cv, pool_size=args.pool_size,
        precision_threshold=args.precision_threshold,
        simplify_agreement=args.simplify_agreement,
        simplify_min_gain=args.simplify_min_gain,
        simplify_decimals=args.simplify_decimals,
        n_run=1, base_seed=args.seed, explain=True,
        output_dir=str(args.output_dir), device=args.device,
    )

    X_tr, y_tr, X_te, y_te = load_dataset(dataset)
    print(f"  train {X_tr.shape}  test {X_te.shape}  "
          f"{len(np.unique(y_tr))} classes", flush=True)

    formulas, X_tr_feats, X_te_feats = build_formula_bank(X_tr, X_te, config, args.seed)
    model = train_classifier(X_tr_feats, y_tr, config, seed=args.seed)
    acc = evaluate_classifier(model, X_te_feats, y_te)["balanced_accuracy"]
    print(f"  fit: balanced_accuracy={acc:.4f}", flush=True)

    W, b = model.coef_, model.intercept_
    t0 = time.perf_counter()
    global_per_class, locals_per_class, n_unique = build_global_explanations(
        X_te_feats, X_te, X_tr_feats, X_tr, W, b, model, formulas, y_tr,
        pool_size=config.pool_size,
        precision_threshold=config.precision_threshold,
        simplify_agreement=config.simplify_agreement,
        simplify_min_gain=config.simplify_min_gain,
        simplify_decimals=config.simplify_decimals,
        rng=np.random.default_rng(args.seed),
    )
    time_explain_s = time.perf_counter() - t0

    train_eval_before = evaluate_global(global_per_class, X_tr, y_tr)
    test_eval_before = evaluate_global(global_per_class, X_te, y_te)
    depths_before = [F0.formula_depth(phi) for phi in global_per_class.values()]
    print(f"  before: macro-F1 train={train_eval_before['macro_avg']['f1']:.4f} "
          f"test={test_eval_before['macro_avg']['f1']:.4f}", flush=True)

    # Tune each class's assembled global formula for F1, gated so it can only
    # keep or improve train-set hard F1 (accept_only_if_better).
    t0 = time.perf_counter()
    tuned_per_class = {}
    n_changed = n_improved = n_worsened = 0
    for cls, phi in global_per_class.items():
        phi_tuned, log = finetune_formula(
            phi, X_tr, y_tr, cls, n_steps=args.tune_steps, objective="f1")
        tuned_per_class[cls] = phi_tuned
        if str(phi_tuned) != str(phi):
            n_changed += 1
            if log["f1_after"] > log["f1_before"] + 1e-9:
                n_improved += 1
            elif log["f1_after"] < log["f1_before"] - 1e-9:
                n_worsened += 1
    time_tune_s = time.perf_counter() - t0

    train_eval_after = evaluate_global(tuned_per_class, X_tr, y_tr)
    test_eval_after = evaluate_global(tuned_per_class, X_te, y_te)
    depths_after = [F0.formula_depth(phi) for phi in tuned_per_class.values()]
    print(f"  after:  macro-F1 train={train_eval_after['macro_avg']['f1']:.4f} "
          f"test={test_eval_after['macro_avg']['f1']:.4f}  "
          f"(changed {n_changed}/{len(global_per_class)} classes, "
          f"{n_improved} improved / {n_worsened} worsened on train)", flush=True)

    return {
        "dataset": dataset,
        "n_formulas": args.budget,
        "depth_max": depth_max,
        "until_weight": args.until_weight,
        "seed": args.seed,
        "balanced_accuracy": round(acc, 6),
        "n_classes": len(global_per_class),
        "global_macro_f1_train_before": round(train_eval_before["macro_avg"]["f1"], 6),
        "global_macro_f1_test_before": round(test_eval_before["macro_avg"]["f1"], 6),
        "global_macro_f1_train_after": round(train_eval_after["macro_avg"]["f1"], 6),
        "global_macro_f1_test_after": round(test_eval_after["macro_avg"]["f1"], 6),
        "delta_macro_f1_test": round(
            test_eval_after["macro_avg"]["f1"] - test_eval_before["macro_avg"]["f1"], 6),
        "n_classes_changed": n_changed,
        "n_classes_improved": n_improved,
        "n_classes_worsened": n_worsened,
        "mean_depth_before": round(float(np.mean(depths_before)), 4) if depths_before else float("nan"),
        "mean_depth_after": round(float(np.mean(depths_after)), 4) if depths_after else float("nan"),
        "time_explain_s": round(time_explain_s, 4),
        "time_tune_s": round(time_tune_s, 4),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default=None,
                   help="comma list; default: every dataset in the results table")
    p.add_argument("--results_table", type=Path, default=Path("tables/results_long.csv"))
    p.add_argument("--method", default="stl_linear")
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--until_weight", type=float, default=0.0)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--only_temporal", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--pool_size", type=int, default=10)
    p.add_argument("--precision_threshold", type=float, default=0.75)
    p.add_argument("--simplify_agreement", type=float, default=0.98)
    p.add_argument("--simplify_min_gain", type=float, default=0.0)
    p.add_argument("--simplify_decimals", type=int, default=1)
    p.add_argument("--tune_steps", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", type=Path, default=Path("results/tuning_check"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    best_depths = {}
    if args.depth is None or args.datasets is None:
        if not args.results_table.exists():
            raise SystemExit(
                f"{args.results_table} not found -- run scripts/make_table.py first, "
                f"or pass --datasets together with --depth.")
        best_depths = load_best_depths(args.results_table, args.method)

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = sorted(best_depths)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, failed = [], []
    for ds in datasets:
        depth = args.depth if args.depth is not None else best_depths.get(ds)
        if depth is None:
            print(f"!! {ds}: no best depth in {args.results_table}; "
                  f"pass --depth to force one. Skipping.", file=sys.stderr)
            failed.append(ds)
            continue
        try:
            rows.append(run_dataset(ds, depth, args))
        except Exception as exc:
            print(f"!! {ds} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(ds)

    if rows:
        summary_path = args.output_dir / "tuning_check_summary.csv"
        existing = {}
        if summary_path.exists():
            with open(summary_path, newline="") as f:
                existing = {r["dataset"]: r for r in csv.DictReader(f)}
        for r in rows:
            existing[r["dataset"]] = r
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            w.writeheader()
            for ds in sorted(existing):
                w.writerow(existing[ds])
        print(f"\nWrote {summary_path} ({len(existing)} datasets)")

    if failed:
        print(f"Failed/skipped: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
