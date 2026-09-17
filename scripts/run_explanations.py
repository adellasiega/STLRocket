#!/usr/bin/env python
"""
Build local + global STL explanations for the best STLRocket model per dataset.

The accuracy sweep (``run_stlinear_stltree.py``) saves only metrics -- no trained
models -- so "the best model" exists on disk as a *configuration record*, not as
weights. This script therefore reads each dataset's best ``stl_linear`` config
from ``tables/results_long.csv`` (written by ``make_table.py``), retrains at that
config, and runs the explanation pipeline from the ``stlrocket`` package.

Per dataset it writes ``<output_dir>/<dataset>/``:

    config.json                 chosen config + the retrained model's accuracy
    global_explanations.csv     one row per class: formula + train/test quality
    local_explanations.csv      one row per test instance: formula + precision
    metrics.json                the same numbers, shaped for plot_explanations.py

plus an aggregate ``<output_dir>/summary.csv``, one row per dataset.

Usage
-----
    # every dataset in the table, at its own best depth
    python scripts/run_explanations.py

    # one dataset, cheap config, for a smoke test
    python scripts/run_explanations.py --datasets BasicMotions --budget 1000 \
        --output_dir /tmp/expl_smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Run correctly regardless of CWD (the SLURM array invokes this by absolute path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.features import build_formula_bank
from stlrocket.classifier import train_classifier, evaluate_classifier
from stlrocket.explanations import build_global_explanations
from stlrocket.evaluation import evaluate_global
from stlrocket.formula_sampler import F0


GLOBAL_FIELDS = [
    "class", "formula", "n_unique_locals", "depth",
    "coverage_train", "fp_train", "precision_train", "f1_train",
    "coverage_test", "fp_test", "precision_test", "f1_test",
]
LOCAL_FIELDS = [
    "sample_idx", "target_class", "formula", "n_picks", "depth",
    "precision_train", "n_tp", "n_fp",
]
SUMMARY_FIELDS = [
    "dataset", "n_formulas", "depth_max", "until_weight", "seed",
    "balanced_accuracy", "n_classes",
    "mean_local_precision", "mean_local_depth",
    "mean_n_picks", "n_local_empty",
    "global_macro_f1_train", "global_macro_f1_test",
    "global_macro_precision_test", "global_macro_coverage_test",
    "mean_global_depth",
    "time_feats_s", "time_fit_s", "time_explain_s",
]


# ---------------------------------------------------------------------------
# Best-configuration lookup
# ---------------------------------------------------------------------------

def load_best_depths(table_path: Path, method: str = "stl_linear") -> dict[str, int]:
    """Map dataset -> best depth_max, from make_table.py's long-format output.

    The table is already filtered to one budget and one until_weight (make_table
    pins both), so ``depth`` is the only free axis recorded per dataset.
    """
    df = pd.read_csv(table_path)
    sub = df[df["method"] == method]
    if sub.empty:
        raise SystemExit(f"no rows for method={method!r} in {table_path}")
    return {str(r.dataset): int(r.depth) for r in sub.itertuples()}


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def build_config(dataset: str, depth_max: int, args) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset,
        n_formulas=args.budget,
        depth_max=depth_max,
        only_temporal=args.only_temporal,
        until_weight=args.until_weight,
        cv=args.cv,
        pool_size=args.pool_size,
        precision_threshold=args.precision_threshold,
        simplify_agreement=args.simplify_agreement,
        simplify_min_gain=args.simplify_min_gain,
        simplify_decimals=args.simplify_decimals,
        n_run=1,
        base_seed=args.seed,
        explain=True,
        output_dir=str(args.output_dir),
        device=args.device,
    )


def _mean(values) -> float:
    """Rounded mean over a possibly-empty iterable, NaN when there is nothing."""
    vals = list(values)
    return round(float(np.mean(vals)), 6) if vals else float("nan")


def summarize_locals(locals_per_class: dict) -> dict:
    """Per-class local precision, coverage, tree depth and formula length."""
    out = {}
    for cls, lst in locals_per_class.items():
        kept = [t for t in lst if t[1] is not None]
        precisions = [t[3] for t in kept]
        depths = [t[6] for t in kept]
        lengths = [len(t[2]) for t in kept]
        mean = lambda v: float(np.mean(v)) if v else float("nan")
        out[str(cls)] = {
            "n_instances": len(lst),
            "n_explained": len(kept),
            "mean_precision": mean(precisions),
            "std_precision": float(np.std(precisions)) if precisions else float("nan"),
            "mean_depth": mean(depths),
            # Kept under the historical key: plot_explanations.py plots it as
            # "mean formula length (# conjuncts)".
            "mean_length": mean(lengths),
        }
    return out


def save_global_csv(global_per_class, train_eval, test_eval, n_unique, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GLOBAL_FIELDS)
        w.writeheader()
        for cls, phi in global_per_class.items():
            tr, te = train_eval.get(cls, {}), test_eval.get(cls, {})
            w.writerow({
                "class": cls,
                "formula": str(phi),
                "n_unique_locals": n_unique.get(cls, 0),
                "depth": F0.formula_depth(phi),
                "coverage_train": tr.get("coverage", float("nan")),
                "fp_train": tr.get("fp_rate", float("nan")),
                "precision_train": tr.get("precision", float("nan")),
                "f1_train": tr.get("f1", float("nan")),
                "coverage_test": te.get("coverage", float("nan")),
                "fp_test": te.get("fp_rate", float("nan")),
                "precision_test": te.get("precision", float("nan")),
                "f1_test": te.get("f1", float("nan")),
            })


def save_local_csv(locals_per_class, path: Path) -> None:
    rows = []
    for cls, lst in locals_per_class.items():
        for sample_idx, phi, picks, precision, n_tp, n_fp, depth in lst:
            rows.append({
                "sample_idx": sample_idx,
                "target_class": cls,
                "formula": str(phi) if phi is not None else "",
                "n_picks": len(picks),
                "depth": depth,
                "precision_train": round(precision, 6),
                "n_tp": n_tp,
                "n_fp": n_fp,
            })
    rows.sort(key=lambda r: r["sample_idx"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOCAL_FIELDS)
        w.writeheader()
        w.writerows(rows)


def run_dataset(dataset: str, depth_max: int, args) -> dict:
    print(f"\n=== {dataset} (M={args.budget}, depth_max={depth_max}, "
          f"until_weight={args.until_weight}, seed={args.seed}) ===", flush=True)

    config = build_config(dataset, depth_max, args)
    X_tr, y_tr, X_te, y_te = load_dataset(dataset)
    print(f"  train {X_tr.shape}  test {X_te.shape}  "
          f"{len(np.unique(y_tr))} classes", flush=True)

    t0 = time.perf_counter()
    formulas, X_tr_feats, X_te_feats = build_formula_bank(X_tr, X_te, config, args.seed)
    time_feats_s = time.perf_counter() - t0
    print(f"  features: {time_feats_s:.1f}s", flush=True)

    t0 = time.perf_counter()
    model = train_classifier(X_tr_feats, y_tr, config, seed=args.seed)
    time_fit_s = time.perf_counter() - t0
    acc = evaluate_classifier(model, X_te_feats, y_te)["balanced_accuracy"]
    print(f"  fit: {time_fit_s:.1f}s  balanced_accuracy={acc:.4f}", flush=True)

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
    print(f"  explain: {time_explain_s:.1f}s  "
          f"{len(global_per_class)} global formulas", flush=True)

    train_eval = evaluate_global(global_per_class, X_tr, y_tr)
    test_eval = evaluate_global(global_per_class, X_te, y_te)
    local_summary = summarize_locals(locals_per_class)

    out_dir = args.output_dir / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps({
        "dataset": dataset,
        "n_formulas": args.budget,
        "depth_max": depth_max,
        "until_weight": args.until_weight,
        "only_temporal": args.only_temporal,
        "seed": args.seed,
        "cv": args.cv,
        "pool_size": args.pool_size,
        "precision_threshold": args.precision_threshold,
        "simplify_agreement": args.simplify_agreement,
        "simplify_min_gain": args.simplify_min_gain,
        "simplify_decimals": args.simplify_decimals,
        "balanced_accuracy": acc,
        "time_feats_s": round(time_feats_s, 4),
        "time_fit_s": round(time_fit_s, 4),
        "time_explain_s": round(time_explain_s, 4),
    }, indent=2))

    save_global_csv(global_per_class, train_eval, test_eval, n_unique,
                    out_dir / "global_explanations.csv")
    save_local_csv(locals_per_class, out_dir / "local_explanations.csv")

    (out_dir / "metrics.json").write_text(json.dumps({
        "local_per_class": local_summary,
        "global_per_class": {
            str(cls): {"train": train_eval[cls], "test": test_eval[cls]}
            for cls in global_per_class
        },
        "global_macro_f1_train": train_eval["macro_avg"]["f1"],
        "global_macro_f1_test": test_eval["macro_avg"]["f1"],
        "global_macro_precision_test": test_eval["macro_avg"]["precision"],
        "global_macro_coverage_test": test_eval["macro_avg"]["coverage"],
    }, indent=2))

    all_locals = [t for lst in locals_per_class.values() for t in lst]
    explained = [t for t in all_locals if t[1] is not None]
    print(f"  macro-F1 train={train_eval['macro_avg']['f1']:.3f} "
          f"test={test_eval['macro_avg']['f1']:.3f}  ->  {out_dir}", flush=True)

    return {
        "dataset": dataset,
        "n_formulas": args.budget,
        "depth_max": depth_max,
        "until_weight": args.until_weight,
        "seed": args.seed,
        "balanced_accuracy": round(acc, 6),
        "n_classes": len(global_per_class),
        "global_macro_f1_train": round(train_eval["macro_avg"]["f1"], 6),
        "global_macro_f1_test": round(test_eval["macro_avg"]["f1"], 6),
        "global_macro_precision_test": round(test_eval["macro_avg"]["precision"], 6),
        "global_macro_coverage_test": round(test_eval["macro_avg"]["coverage"], 6),
        "mean_global_depth": round(float(np.mean(
            [F0.formula_depth(phi) for phi in global_per_class.values()])), 6)
            if global_per_class else float("nan"),
        "mean_local_precision": _mean(t[3] for t in explained),
        "mean_local_depth": _mean(t[6] for t in explained),
        "mean_n_picks": _mean(len(t[2]) for t in explained),
        "n_local_empty": len(all_locals) - len(explained),
        "time_feats_s": round(time_feats_s, 4),
        "time_fit_s": round(time_fit_s, 4),
        "time_explain_s": round(time_explain_s, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default=None,
                   help="comma list; default: every dataset in the results table")
    p.add_argument("--results_table", type=Path, default=Path("tables/results_long.csv"),
                   help="make_table.py long-format output, for the best depth per dataset")
    p.add_argument("--method", default="stl_linear",
                   help="which method's best config to explain")
    p.add_argument("--budget", type=int, default=10000, help="n_formulas (M)")
    p.add_argument("--until_weight", type=float, default=0.0)
    p.add_argument("--depth", type=int, default=None,
                   help="override the per-dataset best depth from the table")
    p.add_argument("--only_temporal", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--pool_size", type=int, default=10)
    p.add_argument("--precision_threshold", type=float, default=0.75)
    p.add_argument("--simplify_agreement", type=float, default=0.98)
    p.add_argument("--simplify_min_gain", type=float, default=0.0)
    p.add_argument("--simplify_decimals", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_dir", type=Path, default=Path("results/explanations"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --depth alone is not enough to skip the table: we still need the dataset
    # list from it unless --datasets was given explicitly.
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
        except Exception as exc:  # one bad dataset must not sink a SLURM array task
            print(f"!! {ds} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(ds)

    # Aggregate across whatever is on disk, so per-dataset SLURM tasks writing
    # into a shared output_dir still build up one complete summary.
    if rows:
        summary_path = args.output_dir / "summary.csv"
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
