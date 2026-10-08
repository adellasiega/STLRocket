#!/usr/bin/env python
"""
Full explanation pipeline (stlrocket/instance_rules.py, paper §3.3) per dataset and run:
formula bank and lasso as in the experiments (no intercept, 1-SE lambda), instance rules
for every train and test series, and global explanations per class. Seeded by
base_seed + run.

Everything is written to OUT_DIR/<dataset>/run<r>/:
  config.json         all settings (ExperimentConfig, RuleConfig, seed)
  model.json          classes, balanced accuracy on train and test, lambda, selected
                      formulae (indices into the bank) and their logit weights
  formulas.txt.gz     the formula bank, one STL formula per line (row j = formula j)
  formulas.pkl.gz     the same bank as torcheck objects (pickle), to rebuild rules
  bank_stats.npz      mu, sigma: train mean and std of each formula's robustness; with
                      formulas.pkl.gz they rebuild every rule exactly from its literals
                      (instance_rules.rule_formula)
  rules_train.csv     one row per series: its instance rule as literals (formula index,
  rules_test.csv      direction, cut), its simplified STL formula, and the formula's
                      metrics on train and test (the explained series itself left out)
  global_rules.csv    the rules of each class's global explanation, raw and simplified
  global_metrics.csv  metrics of each global explanation (raw and simplified) on train/test
  summary.json        headline numbers; written last, so it marks the run as complete

All metrics are w.r.t. the classes predicted by the model: support (series satisfying
the formula), TP, precision and coverage (TP / series of the class). Runs whose
summary.json exists are skipped, so a resubmitted job only redoes unfinished runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.features import build_formula_bank, eval_robustness
from stlrocket.classifier import train_classifier
from stlrocket.formula_sampler import F0
from stlrocket import instance_rules as ir

SPLITS = ("train", "test")
METRICS = ("support", "tp", "precision", "coverage")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def or_mask(phis: list, X: np.ndarray) -> np.ndarray:
    """Mask of the series of X satisfying the OR of phis."""
    mask = np.zeros(len(X), dtype=bool)
    for phi in phis:
        mask |= eval_robustness(phi, X) > 0
    return mask


def metric_cols(prefix: str, m: dict) -> dict:
    return {f"support_{prefix}": m["support"], f"tp_{prefix}": m["tp"],
            f"prec_{prefix}": m["precision"], f"coverage_{prefix}": m["coverage"]}


def run(args, dataset: str, run_id: int) -> None:
    out = Path(args.out_dir) / dataset / f"run{run_id}"
    if (out / "summary.json").exists():
        log(f"{dataset} run {run_id}: already complete, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)
    seed = args.base_seed + run_id
    times = {}

    t = time.perf_counter()
    X_tr, y_tr_raw, X_te, y_te_raw = load_dataset(dataset)
    cfg = ExperimentConfig(
        dataset=dataset, n_formulas=args.n_formulas, depth_max=args.depth, until_weight=0.0,
        cv=args.cv, cut_point=args.cut_point, fit_intercept=False, n_run=1, base_seed=args.base_seed,
        output_dir=str(out), device=args.device,
        # fields of the old explanation pipeline, unused here
        explain=False, pool_size=0, precision_threshold=0.0, simplify_agreement=0.0, simplify_min_gain=0.0,
        simplify_decimals=0,
    )
    rcfg = ir.RuleConfig(n_min=args.n_min, n_max=args.n_max, beam_width=args.beam_width,
                         confidence=args.confidence, min_new_share=args.min_new_share, decimals=args.decimals)
    (out / "config.json").write_text(json.dumps(
        {"dataset": dataset, "run": run_id, "seed": seed, "experiment": dataclasses.asdict(cfg),
         "rules": dataclasses.asdict(rcfg)}, indent=2))
    log(f"{dataset} run {run_id}: {len(X_tr)} train / {len(X_te)} test series, shape {X_tr.shape[1:]}")

    # Formula bank and model
    formulas, F_tr, F_te, mu, sigma = build_formula_bank(X_tr, X_te, cfg, seed)
    times["features_s"] = time.perf_counter() - t
    t = time.perf_counter()
    model = train_classifier(F_tr, y_tr_raw, cfg, seed=seed)
    times["model_s"] = time.perf_counter() - t
    W = ir.model_logits_matrix(model)
    classes = model.classes_
    index = {c: i for i, c in enumerate(classes)}
    F = {"train": F_tr, "test": F_te}
    Xs = {"train": X_tr, "test": X_te}
    pred = {s: ir.predict_indices(F[s], W) for s in SPLITS}
    true = {s: np.array([index[c] for c in y]) for s, y in (("train", y_tr_raw), ("test", y_te_raw))}
    selected = np.flatnonzero(np.any(W != 0, axis=0))
    bacc = {s: float(balanced_accuracy_score(true[s], pred[s])) for s in SPLITS}

    with gzip.open(out / "formulas.txt.gz", "wt") as f:
        f.writelines(f"{phi}\n" for phi in formulas)
    with gzip.open(out / "formulas.pkl.gz", "wb") as f:
        pickle.dump(formulas, f)
    np.savez(out / "bank_stats.npz", mu=mu, sigma=sigma)
    (out / "model.json").write_text(json.dumps(
        {"classes": [str(c) for c in classes], "balanced_accuracy": bacc,
         "accuracy": {s: float((pred[s] == true[s]).mean()) for s in SPLITS},
         "lambda": float(np.atleast_1d(model.lambda_best_)[0]), "n_selected": int(len(selected)),
         "selected": selected.tolist(), "weights": W[:, selected].tolist()}, indent=1))
    log(f"model: {len(selected)} of {len(formulas)} formulae selected, balanced accuracy "
        f"train {bacc['train']:.3f} / test {bacc['test']:.3f}")

    # Instance rules, scored as their simplified STL formula
    rule_tables = {}
    for own in SPLITS:
        t = time.perf_counter()
        rules = ir.explain_instances(F[own], W, F_tr, pred["train"], rcfg, train=own == "train")
        times[f"rules_{own}_s"] = time.perf_counter() - t
        if own == "train":
            train_rules = rules
        t = time.perf_counter()
        rows = []
        for i, r in enumerate(rules):
            row = {"i": i, "true_class": str(classes[true[own][i]]), "pred_class": str(classes[r["k"]]),
                   "k": r["k"], "n_literals": len(r["literals"]), "literals": json.dumps(r["literals"]),
                   "lcb_train": r["lcb"], "size_raw": 0, "size": 0,
                   **{c: np.nan for s in SPLITS for c in metric_cols(s, dict.fromkeys(METRICS))}, "rule": ""}
            phi = ir.rule_formula(r["literals"], formulas, mu, sigma, X_tr, rcfg)
            if phi is not None:
                row["size_raw"] = F0.formula_size(ir.to_stl(r["literals"], formulas, mu, sigma))
                row["size"] = F0.formula_size(phi)
                for s in SPLITS:
                    m = ir.rule_metrics(eval_robustness(phi, Xs[s]) > 0, r["k"], pred[s],
                                        exclude=i if s == own else None)
                    row.update(metric_cols(s, m))
                row["rule"] = str(phi)
            rows.append(row)
        times[f"rules_{own}_eval_s"] = time.perf_counter() - t
        rule_tables[own] = pd.DataFrame(rows)
        rule_tables[own].to_csv(out / f"rules_{own}.csv", index=False)
        log(f"{own} rules: {len(rules)} in {times[f'rules_{own}_s']:.0f} s "
            f"(+ {times[f'rules_{own}_eval_s']:.0f} s to simplify and score)")

    # Global explanations, raw and simplified
    t = time.perf_counter()
    args_g = (train_rules, F_tr, X_tr, pred["train"], formulas, mu, sigma, rcfg)
    G = {"raw": ir.global_explanations(*args_g, simplify=False), "simplified": ir.global_explanations(*args_g)}
    rule_rows, metric_rows = [], []
    for version, Gv in G.items():
        for k, phis in sorted(Gv.items()):
            for n, phi in enumerate(phis):
                rule_rows.append({"class": str(classes[k]), "k": k, "version": version, "n": n,
                                  "size": F0.formula_size(phi), "rule": str(phi)})
            for s in SPLITS:
                m = ir.rule_metrics(or_mask(phis, Xs[s]), k, pred[s])
                metric_rows.append({"class": str(classes[k]), "k": k, "version": version, "evaluated_on": s,
                                    "n_rules": len(phis), "size": sum(F0.formula_size(p) for p in phis), **m})
    times["global_s"] = time.perf_counter() - t
    pd.DataFrame(rule_rows).to_csv(out / "global_rules.csv", index=False)
    gm = pd.DataFrame(metric_rows)
    gm.to_csv(out / "global_metrics.csv", index=False)

    # Summary (written last: marks the run as complete)
    summary = {"dataset": dataset, "run": run_id, "seed": seed, "depth": args.depth, "n_formulas": args.n_formulas,
               "n_train": len(X_tr), "n_test": len(X_te), "n_classes": len(classes),
               "n_selected": int(len(selected)), "balanced_accuracy": bacc, "times_s": times}
    for own, df in rule_tables.items():
        found = df[df.n_literals > 0]
        summary[f"rules_{own}"] = {
            "found": int(len(found)), "of": int(len(df)),
            **{f"pooled_prec_{s}": float(found[f"tp_{s}"].sum() / max(found[f"support_{s}"].sum(), 1)) for s in SPLITS},
            **{f"mean_coverage_{s}": float(found[f"coverage_{s}"].mean()) for s in SPLITS},
            "mean_lcb_train": float(found.lcb_train.mean()), "mean_length": float(found.n_literals.mean()),
            "mean_size": float(found["size"].mean()) if len(found) else float("nan")}
    simp = gm[gm.version == "simplified"]
    summary["global"] = {s: {"macro_precision": float(simp[simp.evaluated_on == s].precision.mean()),
                             "macro_coverage": float(simp[simp.evaluated_on == s].coverage.mean())} for s in SPLITS}
    summary["global"]["mean_rules"] = float(simp[simp.evaluated_on == "train"].n_rules.mean())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"{dataset} run {run_id} done: test rules pooled precision "
        f"{summary['rules_test']['pooled_prec_test']:.3f}, global macro precision "
        f"{summary['global']['test']['macro_precision']:.3f} on test")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--runs", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--n_formulas", type=int, default=10000)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--cut_point", type=float, default=1.0)
    defaults = ir.RuleConfig()
    p.add_argument("--n_min", type=int, default=defaults.n_min, help="min support of a rule")
    p.add_argument("--n_max", type=int, default=defaults.n_max, help="max literals per rule")
    p.add_argument("--beam_width", type=int, default=defaults.beam_width)
    p.add_argument("--confidence", type=float, default=defaults.confidence, help="confidence of the Wilson bound")
    p.add_argument("--min_new_share", type=float, default=defaults.min_new_share,
                   help="global: share of its class a rule must newly cover to enter the OR")
    p.add_argument("--decimals", type=int, default=defaults.decimals, help="simplification: threshold rounding")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out_dir", default="results/explanations")
    args = p.parse_args()
    for dataset in args.datasets:
        for run_id in args.runs:
            run(args, dataset, run_id)


if __name__ == "__main__":
    main()
