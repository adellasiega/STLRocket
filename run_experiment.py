#!/usr/bin/env python
"""
Run the STL-Rocket experiment pipeline.

For n_run seeds: generate formulas, train logistic regression, evaluate
balanced accuracy. Then one extra run computes and saves local/global
explanations.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.preprocessing import StateVariableStandardScaler
from stlrocket.features import build_formula_bank
from stlrocket.classifier import train_classifier, evaluate_classifier
from stlrocket.explanations import build_global_explanations
from stlrocket.evaluation import evaluate_global


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def save_config(config: ExperimentConfig, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)


def save_accuracy_summary(results: list[dict], path: Path) -> None:
    bal_accs = [r["balanced_accuracy"] for r in results]
    macro_f1s = [r["macro_f1"] for r in results]
    feat_times = [r["time_features_s"] for r in results]
    train_times = [r["time_train_s"] for r in results]
    total_times = [r["time_total_s"] for r in results]
    summary = {
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "std_balanced_accuracy": float(np.std(bal_accs)),
        "mean_macro_f1": float(np.mean(macro_f1s)),
        "mean_time_features_s": float(np.mean(feat_times)),
        "mean_time_train_s": float(np.mean(train_times)),
        "mean_time_total_s": float(np.mean(total_times)),
        "per_run": results,
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Accuracy — mean bal_acc: {summary['mean_balanced_accuracy']:.4f}  "
        f"std: {summary['std_balanced_accuracy']:.4f}  "
        f"mean macro_f1: {summary['mean_macro_f1']:.4f}"
    )
    print(
        f"Timing  — features: {summary['mean_time_features_s']:.2f}s  "
        f"train: {summary['mean_time_train_s']:.2f}s  "
        f"total: {summary['mean_time_total_s']:.2f}s  (means over {len(results)} runs)"
    )


def save_local_explanations(locals_per_class: dict, path: Path) -> None:
    rows = []
    for cls, lst in locals_per_class.items():
        for sample_idx, phi_local, picks, precision, n_tp in lst:
            rows.append({
                "sample_idx": sample_idx,
                "target_class": cls,
                "formula": str(phi_local),
                "n_picks": len(picks),
                "precision_train": round(precision, 6),
                "n_tp": n_tp,
            })
    rows.sort(key=lambda r: r["sample_idx"])
    fieldnames = ["sample_idx", "target_class", "formula", "n_picks", "precision_train", "n_tp"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_global_explanations(
    global_per_class: dict,
    global_results: dict,
    n_unique_per_class: dict,
    path: Path,
) -> None:
    fieldnames = ["class", "formula", "n_unique_locals", "f1"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cls, phi_global in global_per_class.items():
            metrics = global_results.get(cls, {})
            writer.writerow({
                "class": cls,
                "formula": str(phi_global),
                "n_unique_locals": n_unique_per_class.get(cls, 0),
                "f1": round(metrics.get("f1", float("nan")), 6),
            })


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_single(
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    X_te_raw: np.ndarray,
    y_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> dict:
    scaler = StateVariableStandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    t0 = time.perf_counter()
    formulas, X_tr_feats, X_te_feats = build_formula_bank(X_tr, X_te, config, seed)
    t1 = time.perf_counter()
    model = train_classifier(X_tr_feats, y_tr, config)
    t2 = time.perf_counter()

    metrics = evaluate_classifier(model, X_te_feats, y_te)
    metrics["time_features_s"] = round(t1 - t0, 4)
    metrics["time_train_s"] = round(t2 - t1, 4)
    metrics["time_total_s"] = round(t2 - t0, 4)
    return metrics


def run_extra(
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    X_te_raw: np.ndarray,
    y_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
    out_dir: Path,
) -> None:
    print(f"\n--- Extra run (seed={seed}) for explanations ---")
    scaler = StateVariableStandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)

    formulas, X_tr_feats, X_te_feats = build_formula_bank(X_tr, X_te, config, seed)
    model = train_classifier(X_tr_feats, y_tr, config)

    W = model.coef_
    b = model.intercept_

    print("\nBuilding local/global explanations...")
    global_per_class, locals_per_class, n_unique_per_class = build_global_explanations(
        X_te_feats, W, b, model, formulas, X_tr, y_tr,
        pool_size=config.pool_size,
        precision_threshold=config.precision_threshold,
    )

    global_results = evaluate_global(global_per_class, X_te, y_te)

    save_local_explanations(locals_per_class, out_dir / "local_explanations.csv")
    save_global_explanations(global_per_class, global_results, n_unique_per_class, out_dir / "global_explanations.csv")

    print("\nGlobal explanation metrics:")
    print(f"{'class':<14} {'n_unique':>8} {'f1':>8}")
    print("-" * 34)
    for cls, m in global_results.items():
        if cls == "macro_avg":
            continue
        print(f"{cls:<14} {n_unique_per_class.get(cls, 0):>8d} {m['f1']:>7.1%}")
    macro = global_results.get("macro_avg", {})
    if macro:
        print("-" * 34)
        print(f"{'macro_avg':<14} {'':>8} {macro['f1']:>7.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run STL-Rocket experiment")
    parser.add_argument("--dataset", default="BasicMotions")
    parser.add_argument("--n_formulas", type=int, default=1000)
    parser.add_argument("--depth_max", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--threshold_corr", type=float, default=0.98)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--only_temporal", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--until_weight", type=float, default=0.0)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--max_iter_lr", type=int, default=3000)
    parser.add_argument("--pool_size", type=int, default=10)
    parser.add_argument("--precision_threshold", type=float, default=0.75)
    parser.add_argument("--n_run", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    return ExperimentConfig(**vars(args))


def main() -> None:
    config = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output_dir) / f"{config.dataset}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, out_dir / "config.json")
    print(f"Output directory: {out_dir}")
    print(f"Dataset: {config.dataset}  n_run: {config.n_run}  n_formulas: {config.n_formulas}\n")

    print("Loading dataset...")
    X_tr_raw, y_tr, X_te_raw, y_te = load_dataset(config.dataset)
    print(f"  train: {X_tr_raw.shape}  test: {X_te_raw.shape}")

    # --- n_run accuracy loop ---
    run_results = []
    for run_idx in range(config.n_run):
        seed = config.base_seed + run_idx
        print(f"\n--- Run {run_idx} (seed={seed}) ---")
        metrics = run_single(X_tr_raw, y_tr, X_te_raw, y_te, config, seed)
        entry = {"run": run_idx, "seed": seed, **metrics}
        run_results.append(entry)
        print(
            f"  bal_acc={metrics['balanced_accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}"
            f"  features={metrics['time_features_s']:.2f}s  train={metrics['time_train_s']:.2f}s"
        )

    save_accuracy_summary(run_results, out_dir / "accuracy.json")

    # --- Extra explanation run ---
    extra_seed = config.base_seed + config.n_run
    run_extra(X_tr_raw, y_tr, X_te_raw, y_te, config, extra_seed, out_dir)

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
