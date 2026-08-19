#!/usr/bin/env python
"""
Run the STL-Rocket experiment pipeline.

For each of n_run seeds: generate formulas, train logistic regression,
evaluate balanced accuracy, and build local/global explanations.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.preprocessing import StateVariableStandardScaler
from stlrocket.features import build_formula_bank, set_device
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
    feat_times = [r["time_features_s"] for r in results]
    train_times = [r["time_train_s"] for r in results]
    total_times = [r["time_total_s"] for r in results]

    has_explanations = any("time_explanations_s" in r for r in results)

    all_classes = set()
    for r in results:
        all_classes.update(r.get("local_metrics", {}).keys())

    explanation_per_class = {}
    for cls in sorted(all_classes):
        local_runs = [r["local_metrics"][cls] for r in results if cls in r.get("local_metrics", {})]
        global_runs = [r["global_metrics"].get(cls, {}) for r in results if "global_metrics" in r]
        precs = [m["avg_precision"] for m in local_runs if m["avg_precision"] is not None]
        covs = [m["avg_coverage"] for m in local_runs if m["avg_coverage"] is not None]
        f1s = [m["f1"] for m in global_runs if m and m.get("f1") is not None]
        explanation_per_class[cls] = {
            "mean_local_avg_precision": float(np.mean(precs)) if precs else None,
            "mean_local_avg_coverage": float(np.mean(covs)) if covs else None,
            "mean_global_f1": float(np.mean(f1s)) if f1s else None,
        }

    summary = {
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "std_balanced_accuracy": float(np.std(bal_accs)),
        "mean_time_features_s": float(np.mean(feat_times)),
        "mean_time_train_s": float(np.mean(train_times)),
        "mean_time_total_s": float(np.mean(total_times)),
        "per_run": results,
    }
    if has_explanations:
        expl_times = [r["time_explanations_s"] for r in results if "time_explanations_s" in r]
        summary["mean_time_explanations_s"] = float(np.mean(expl_times))
        summary["explanation_per_class"] = explanation_per_class

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Accuracy — mean bal_acc: {summary['mean_balanced_accuracy']:.4f}  "
        f"std: {summary['std_balanced_accuracy']:.4f}"
    )
    expl_str = f"  explanations: {summary['mean_time_explanations_s']:.2f}s" if has_explanations else ""
    print(
        f"Timing  — features: {summary['mean_time_features_s']:.2f}s  "
        f"train: {summary['mean_time_train_s']:.2f}s"
        f"{expl_str}  "
        f"total: {summary['mean_time_total_s']:.2f}s  (means over {len(results)} runs)"
    )
    if explanation_per_class:
        print("\nExplanation metrics (mean over runs):")
        print(f"  {'class':<14} {'local_prec':>10} {'local_cov':>10} {'global_f1':>10}")
        for cls, m in explanation_per_class.items():
            fmt = lambda v: f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
            print(
                f"  {cls:<14} {fmt(m['mean_local_avg_precision'])}"
                f" {fmt(m['mean_local_avg_coverage'])}"
                f" {fmt(m['mean_global_f1'])}"
            )


def save_local_explanations(locals_per_class: dict, path: Path) -> None:
    rows = []
    for cls, lst in locals_per_class.items():
        for sample_idx, phi_local, picks, precision, n_tp in lst:
            rows.append({
                "sample_idx": sample_idx,
                "target_class": cls,
                "formula": str(phi_local) if phi_local is not None else "",
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
            f1 = metrics.get("f1")
            writer.writerow({
                "class": cls,
                "formula": str(phi_global),
                "n_unique_locals": n_unique_per_class.get(cls, 0),
                "f1": round(f1, 6) if f1 is not None else "",
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
    out_dir: Path,
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

    if not config.explain:
        metrics["time_total_s"] = round(t2 - t0, 4)
        return metrics

    W = model.coef_
    b = model.intercept_

    print("  Building explanations...")
    t3 = time.perf_counter()
    global_per_class, locals_per_class, n_unique_per_class = build_global_explanations(
        X_te_feats, X_te, W, b, model, formulas, X_tr, y_tr,
        pool_size=config.pool_size,
        precision_threshold=config.precision_threshold,
    )
    global_results = evaluate_global(global_per_class, X_te, y_te)
    t4 = time.perf_counter()

    metrics["time_explanations_s"] = round(t4 - t3, 4)
    metrics["time_total_s"] = round(t4 - t0, 4)

    local_metrics: dict = {}
    for cls, lst in locals_per_class.items():
        valid = [(prec, n_tp) for _, phi, _, prec, n_tp in lst if phi is not None]
        precs = [p for p, _ in valid]
        covs = [n / len(y_tr) for _, n in valid]
        local_metrics[str(cls)] = {
            "avg_precision": float(np.mean(precs)) if precs else None,
            "avg_coverage": float(np.mean(covs)) if covs else None,
            "n_explained": len(valid),
            "n_total": len(lst),
        }

    metrics["local_metrics"] = local_metrics
    metrics["global_metrics"] = {
        str(cls): m for cls, m in global_results.items() if cls != "macro_avg"
    }

    save_local_explanations(locals_per_class, out_dir / f"seed{seed}_local_explanations.csv")
    save_global_explanations(global_per_class, global_results, n_unique_per_class, out_dir / f"seed{seed}_global_explanations.csv")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    _d = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Run STL-Rocket experiment")
    parser.add_argument("--dataset", default=_d.dataset)
    parser.add_argument("--n_formulas", type=int, default=_d.n_formulas)
    parser.add_argument("--depth_max", type=int, default=_d.depth_max)
    parser.add_argument("--only_temporal", type=lambda x: x.lower() != "false", default=_d.only_temporal)
    parser.add_argument("--until_weight", type=float, default=_d.until_weight)
    parser.add_argument("--cv", type=int, default=_d.cv)
    parser.add_argument("--pool_size", type=int, default=_d.pool_size)
    parser.add_argument("--precision_threshold", type=float, default=_d.precision_threshold)
    parser.add_argument("--n_run", type=int, default=_d.n_run)
    parser.add_argument("--base_seed", type=int, default=_d.base_seed)
    parser.add_argument("--output_dir", default=_d.output_dir)
    parser.add_argument("--explain", type=lambda x: x.lower() != "false", default=_d.explain)
    parser.add_argument("--device", default=_d.device)
    args = parser.parse_args()
    return ExperimentConfig(**vars(args))


def main() -> None:
    config = parse_args()
    set_device(config.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    out_dir = Path(config.output_dir) / f"{config.dataset}_{timestamp}_{uid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, out_dir / "config.json")
    print(f"Output directory: {out_dir}")
    print(f"Dataset: {config.dataset}  n_run: {config.n_run}  n_formulas: {config.n_formulas}\n")

    print("Loading dataset...")
    X_tr_raw, y_tr, X_te_raw, y_te = load_dataset(config.dataset)
    print(f"  train: {X_tr_raw.shape}  test: {X_te_raw.shape}")

    # --- n_run loop ---
    run_results = []
    for run_idx in range(config.n_run):
        seed = config.base_seed + run_idx
        print(f"\n--- Run {run_idx} (seed={seed}) ---")
        metrics = run_single(X_tr_raw, y_tr, X_te_raw, y_te, config, seed, out_dir)
        entry = {"run": run_idx, "seed": seed, **metrics}
        run_results.append(entry)
        print(
            f"  bal_acc={metrics['balanced_accuracy']:.4f}"
            f"  features={metrics['time_features_s']:.2f}s  train={metrics['time_train_s']:.2f}s"
        )

    save_accuracy_summary(run_results, out_dir / "accuracy.json")
    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
