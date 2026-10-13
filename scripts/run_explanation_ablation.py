#!/usr/bin/env python
"""
Validation-only ablation of the explanation method (stlrocket/instance_rules.py). The split is
the one of scripts/run_ablation.py: the TRAIN split of each dataset is divided into a fit part
and a validation part (stratified, seed = base_seed + run) and the test set is never loaded.
The formula bank and the lasso are built once per run on the fit part, as in the experiments
(no intercept, 1-SE lambda), and every variant explains the same model.

    python scripts/run_explanation_ablation.py --datasets ERing --runs 0 1

Part A, variants: every combination of
  construction  how a rule is built (RuleConfig.construction): beam (beam search over
                formulae and cuts), ordered_prefix, ordered_scan and ordered_nocut (formulae
                in the order of their attribution, see instance_rules._ordered_rule)
  candidates    attribution score that selects and ranks the formulae (RuleConfig.candidates)
  confidence    confidence level of the Wilson bound
Each variant builds the instance rules of the fit series (each left out of its own counts) and
of the validation series, and the global explanation of each class from the rules of the fit
series. Rules are scored in feature space (instance_rules.covered), which is equivalent to
evaluating their STL formula, w.r.t. the classes predicted by the model. Faithfulness to the
model is measured on the explained series itself (faithfulness).

Part B, simplification: for the distinct rules of the reference variant (REFERENCE), the
STL formula is built and simplified (instance_rules.simplify_rule) on the fit series, and the
series satisfying it are compared before and after, on fit and on validation.

Written to OUT_DIR, one set of files per dataset and run (prefix <dataset>_run<r>):
  .variants.csv        one row per variant: rule length, precision lower bound, precision and
                       coverage of the instance rules on fit and validation, their
                       faithfulness, precision, coverage and F1 of the global explanations, times
  .instances.csv.gz    one row per (variant, explained series): its rule as literals
                       (formula index, direction, cut) and its counts on fit and validation
  .simplification.csv  one row per rule of part B: series whose satisfaction differs between
                       feature space and STL, and between the raw and the simplified formula;
                       formula sizes; counts on fit and validation before and after
  .done.json           settings and status; written last, so it marks the run as complete

Runs whose .done.json exists are skipped, so a resubmitted job only redoes unfinished runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ablation import load_train, split_ok
from stlrocket.config import ExperimentConfig
from stlrocket.data import standardize_series
from stlrocket.features import build_formula_bank, eval_robustness
from stlrocket.classifier import train_classifier
from stlrocket.formula_sampler import F0
from stlrocket import instance_rules as ir

SPLITS = ("fit", "val")
REFERENCE = ("ordered_scan", "gxi", 0.95)   # variant whose rules are simplified in part B (else the first one run)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def f1(m: dict) -> float:
    """F1 of a rule_metrics dict (0 with no true positive, nan if the class is absent)."""
    if np.isnan(m["coverage"]):
        return np.nan
    return 2 * m["tp"] / (m["support"] + m["tp"] / m["coverage"]) if m["tp"] else 0.0


def faithfulness(x: np.ndarray, W: np.ndarray, literals: list) -> dict:
    """How much of the model's decision on features x the formulae R of a rule carry. The
    measure is the same for every variant (Gradient x Input scores s_j, sum_j s_j > 0):
      attr_share      sum of s_j over R / sum of the positive s_j
      attr_share_top  the same for the |R| formulae with the largest s_j: the most a rule of
                      this length can carry
      margin_drop     1 - m'/m, with m the margin of the predicted class over the best other
                      class and m' the same margin once the features of R are set to 0, their
                      train mean (above 1 when the prediction changes)
      flip            whether that changes the prediction"""
    if not literals:
        return dict(attr_share=np.nan, attr_share_top=np.nan, margin_drop=np.nan, flip=np.nan)
    k, s, _ = ir.candidate_scores(x, W, "gxi")
    R = [j for j, _, _ in literals]
    total = s[s > 0].sum()

    def margin(z):
        return z[k] - np.delete(z, k).max()

    x0 = x.copy()
    x0[R] = 0.0
    m, m0 = margin(W @ x), margin(W @ x0)
    return dict(attr_share=float(s[R].sum() / total), attr_share_top=float(np.sort(s)[-len(R):].sum() / total),
                margin_drop=float(1 - m0 / m) if m > 0 else np.nan, flip=float(m0 <= 0))


def instance_rows(rules: list[dict], idx: np.ndarray, own: str, F: dict, pred: dict, W: np.ndarray) -> list[dict]:
    """One row per explained series: its rule, the rule's counts on both splits (the series
    itself left out of its own split) and its faithfulness."""
    rows = []
    for i, r in zip(idx.tolist(), rules):
        row = {"own": own, "i": i, "k": r["k"], "n_candidates": r["n_candidates"],
               "n_literals": len(r["literals"]), "literals": json.dumps(r["literals"]), "lcb": r["lcb"],
               **faithfulness(F[own][i], W, r["literals"])}
        for s in SPLITS:
            if r["literals"]:
                m = ir.rule_metrics(ir.covered(r["literals"], F[s]), r["k"], pred[s], exclude=i if s == own else None)
            else:
                m = dict(support=0, tp=0, precision=np.nan, coverage=np.nan)
            row.update({f"support_{s}": m["support"], f"tp_{s}": m["tp"],
                        f"prec_{s}": m["precision"], f"coverage_{s}": m["coverage"]})
        rows.append(row)
    return rows


def global_summary(fit_rules: list[dict], F: dict, pred: dict, n_classes: int, rcfg: ir.RuleConfig) -> dict:
    """Macro metrics over classes of the global explanations built from fit_rules. A class
    without rules has coverage and F1 0 and no precision."""
    G = ir.global_rules(fit_rules, F["fit"], pred["fit"], rcfg)
    out = {"global_rules_per_class": float(np.mean([len(G.get(k, [])) for k in range(n_classes)])),
           "global_literals_per_class": float(np.mean([sum(len(lits) for lits in G.get(k, []))
                                                       for k in range(n_classes)]))}
    for s in SPLITS:
        per_class = []
        for k in range(n_classes):
            mask = np.zeros(len(F[s]), dtype=bool)
            for lits in G.get(k, []):
                mask |= ir.covered(lits, F[s])
            m = ir.rule_metrics(mask, k, pred[s])
            per_class.append({"precision": m["precision"], "coverage": m["coverage"], "f1": f1(m)})
        df = pd.DataFrame(per_class)
        out.update({f"global_{c}_{s}": float(df[c].mean()) for c in df.columns})   # nan-skipping means
    return out


def variant_summary(df: pd.DataFrame) -> dict:
    """Aggregates of the instance rows of one variant, per split of the explained series."""
    out = {}
    for own in SPLITS:
        d = df[df.own == own]
        found = d[d.n_literals > 0]
        out.update({f"{own}_explained": len(d), f"{own}_found": len(found) / max(len(d), 1),
                    f"{own}_candidates": float(d.n_candidates.mean()),
                    f"{own}_length": float(found.n_literals.mean()), f"{own}_lcb": float(found.lcb.mean()),
                    # distinct rules (as sets of formulae) per explained series: 1 if no rule is shared
                    f"{own}_distinct_rules": found.literals.map(
                        lambda lits: tuple(sorted(j for j, _, _ in json.loads(lits)))).nunique() / max(len(found), 1),
                    **{f"{own}_{c}": float(found[c].mean()) for c in ("attr_share", "attr_share_top", "margin_drop", "flip")}})
        for s in SPLITS:
            out[f"{own}_pooled_prec_{s}"] = float(found[f"tp_{s}"].sum() / max(found[f"support_{s}"].sum(), 1))
            out[f"{own}_prec_{s}"] = float(found[f"prec_{s}"].mean())
            out[f"{own}_coverage_{s}"] = float(found[f"coverage_{s}"].mean())
    return out


def simplification_rows(rules: list[dict], formulas: list, sigma: np.ndarray, X: dict, F: dict,
                        pred: dict) -> list[dict]:
    """Part B. Masks are the series with positive robustness; every count is the number of
    series whose mask differs between two versions of the same rule: raw (to_stl) and
    pruned (simplify_rule)."""
    rows = []
    for r in rules:
        t = time.perf_counter()
        raw = ir.to_stl(r["literals"], formulas, sigma)
        pruned = ir.simplify_rule(raw, X["fit"])
        elapsed = time.perf_counter() - t
        row = {"k": r["k"], "n_literals": len(r["literals"]), "literals": json.dumps(r["literals"]),
               "size_raw": F0.formula_size(raw), "size_pruned": F0.formula_size(pruned),
               "time_s": elapsed,
               "idempotent": str(ir.simplify_rule(pruned, X["fit"])) == str(pruned)}
        for s in SPLITS:
            feat = ir.covered(r["literals"], F[s])
            masks = {v: eval_robustness(phi, X[s]) > 0 for v, phi in (("raw", raw), ("pruned", pruned))}
            row.update({f"n_{s}": len(feat), f"feat_vs_raw_{s}": int((feat != masks["raw"]).sum()),
                        f"raw_vs_pruned_{s}": int((masks["raw"] != masks["pruned"]).sum())})
            for v in masks:
                m = ir.rule_metrics(masks[v], r["k"], pred[s])
                row.update({f"support_{v}_{s}": m["support"], f"tp_{v}_{s}": m["tp"]})
        row.update({"rule_raw": str(raw), "rule_pruned": str(pruned)})
        rows.append(row)
    return rows


def run(args, dataset: str, run_id: int) -> None:
    out_dir = Path(args.out_dir)
    stem = f"{dataset}_run{run_id}"
    done = out_dir / f"{stem}.done.json"
    if done.exists():
        log(f"{dataset} run {run_id}: already complete, skipped")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.base_seed + run_id
    rng = np.random.default_rng(seed)
    info = {"dataset": dataset, "run": run_id, "seed": seed, "args": vars(args)}

    X_all, y = load_train(dataset)
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=seed)
        fit_idx, val_idx = next(splitter.split(X_all, y))
        ok = split_ok(y[fit_idx], y[val_idx], np.unique(y))
    except ValueError:   # too few samples for the requested ratio
        ok = False
    if not ok:
        done.write_text(json.dumps({**info, "status": "skipped: too few samples per class"}, indent=1))
        log(f"{dataset} run {run_id}: skipped, too few samples per class")
        return

    X = dict(zip(SPLITS, standardize_series(X_all[fit_idx], X_all[val_idx])))
    cfg = ExperimentConfig(
        dataset=dataset, n_formulas=args.n_formulas, depth_max=args.depth, until_weight=0.0,
        cv=args.cv, cut_point=args.cut_point, fit_intercept=False, n_run=1, base_seed=args.base_seed,
        output_dir=str(out_dir), device=args.device,
        # fields of the old explanation pipeline, unused here
        explain=False, pool_size=0, precision_threshold=0.0, simplify_agreement=0.0, simplify_min_gain=0.0,
        simplify_decimals=0,
    )
    t = time.perf_counter()
    formulas, F_fit, F_val, sigma = build_formula_bank(X["fit"], X["val"], cfg, seed)
    model = train_classifier(F_fit, y[fit_idx], cfg, seed=seed)
    W = ir.model_logits_matrix(model)
    index = {c: i for i, c in enumerate(model.classes_)}
    F = {"fit": F_fit, "val": F_val}
    pred = {s: ir.predict_indices(F[s], W) for s in SPLITS}
    true = {"fit": np.array([index[c] for c in y[fit_idx]]), "val": np.array([index[c] for c in y[val_idx]])}
    base = {"dataset": dataset, "run": run_id, "seed": seed, "n_fit": len(fit_idx), "n_val": len(val_idx),
            "n_classes": len(index), "n_selected": int(np.any(W != 0, axis=0).sum()),
            # formulae weighted in several classes: the only ones on which gxi, ig and union can differ
            "n_selected_multiclass": int(((W != 0).sum(axis=0) > 1).sum()),
            **{f"bacc_{s}": float(balanced_accuracy_score(true[s], pred[s])) for s in SPLITS}}
    log(f"{dataset} run {run_id}: {base['n_fit']} fit / {base['n_val']} val series, {base['n_selected']} formulae "
        f"selected, balanced accuracy fit {base['bacc_fit']:.3f} / val {base['bacc_val']:.3f} "
        f"({time.perf_counter() - t:.0f} s)")

    # Series to explain: all of them, or the same seeded subsample for every variant
    explained = {s: np.sort(rng.choice(len(F[s]), args.max_explained, replace=False))
                 if len(F[s]) > args.max_explained else np.arange(len(F[s])) for s in SPLITS}

    # Part A
    variant_rows, instance_frames, reference_rules = [], [], []
    for construction, candidates, confidence in itertools.product(args.constructions, args.candidates, args.confidences):
        rcfg = ir.RuleConfig(n_max=args.n_max, beam_width=args.beam_width, confidence=confidence,
                             min_new_share=args.min_new_share, candidates=candidates, construction=construction)
        variant = {"construction": construction, "candidates": candidates, "confidence": confidence}
        rows, times, rules = [], {}, {}
        for own in SPLITS:
            t = time.perf_counter()
            # each fit series is left out of its own counts
            rules[own] = ir.explain_instances(F[own][explained[own]], W, F_fit, pred["fit"], rcfg, n_jobs=args.n_jobs,
                                              exclude=explained[own].tolist() if own == "fit" else None)
            times[f"time_{own}_s"] = time.perf_counter() - t
            rows += instance_rows(rules[own], explained[own], own, F, pred, W)
        df = pd.DataFrame(rows)
        for col, value in reversed(list(variant.items())):
            df.insert(0, col, value)
        instance_frames.append(df)
        variant_rows.append({**base, **variant, **variant_summary(df),
                             **global_summary(rules["fit"], F, pred, len(index), rcfg), **times})
        if tuple(variant.values()) == REFERENCE or not reference_rules:
            reference_rules = rules["fit"] + rules["val"]
        v = variant_rows[-1]
        log(f"  {construction:14s} {candidates:5s} {confidence:.2f}: val rules precision {v['val_pooled_prec_val']:.3f}, "
            f"coverage {v['val_coverage_val']:.3f}, length {v['val_length']:.2f}, attribution share "
            f"{v['val_attr_share']:.2f}, global F1 val {v['global_f1_val']:.3f} "
            f"({times['time_fit_s'] + times['time_val_s']:.0f} s)")
    pd.DataFrame(variant_rows).to_csv(out_dir / f"{stem}.variants.csv", index=False)
    inst = pd.concat(instance_frames, ignore_index=True)
    inst.insert(0, "run", run_id)
    inst.insert(0, "dataset", dataset)
    inst.to_csv(out_dir / f"{stem}.instances.csv.gz", index=False)

    # Part B: distinct rules of the reference variant
    distinct = list({json.dumps(r["literals"]): r for r in reference_rules if r["literals"]}.values())
    if len(distinct) > args.max_simplified:
        distinct = [distinct[i] for i in np.sort(rng.choice(len(distinct), args.max_simplified, replace=False))]
    t = time.perf_counter()
    simp = pd.DataFrame(simplification_rows(distinct, formulas, sigma, X, F, pred))
    simp.insert(0, "run", run_id)
    simp.insert(0, "dataset", dataset)
    simp.to_csv(out_dir / f"{stem}.simplification.csv", index=False)
    if len(simp):
        log(f"  simplification of {len(simp)} rules ({time.perf_counter() - t:.0f} s): size "
            f"{simp.size_raw.mean():.1f} -> {simp.size_pruned.mean():.1f}; series changing satisfaction on fit "
            f"{int(simp.raw_vs_pruned_fit.sum())}, on val {int(simp.raw_vs_pruned_val.sum())}; feature space vs "
            f"STL on fit {int(simp.feat_vs_raw_fit.sum())}, on val {int(simp.feat_vs_raw_val.sum())}")
    done.write_text(json.dumps({**info, "status": "ok", **base}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--runs", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--val_ratio", type=float, default=0.25)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--n_formulas", type=int, default=10000)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--cut_point", type=float, default=1.0)
    p.add_argument("--constructions", nargs="+", default=list(ir.CONSTRUCTIONS), choices=ir.CONSTRUCTIONS)
    p.add_argument("--candidates", nargs="+", default=["gxi", "ig"], choices=ir.CANDIDATE_METHODS)
    p.add_argument("--confidences", type=float, nargs="+", default=[0.9, 0.95, 0.99],
                   help="confidence levels of the Wilson bound")
    defaults = ir.RuleConfig()
    p.add_argument("--n_max", type=int, default=defaults.n_max, help="max literals per rule")
    p.add_argument("--beam_width", type=int, default=defaults.beam_width)
    p.add_argument("--min_new_share", type=float, default=defaults.min_new_share,
                   help="global: share of its class a rule must newly cover to enter the OR")
    p.add_argument("--max_explained", type=int, default=1000,
                   help="series explained per split; larger splits are subsampled (same series for every variant)")
    p.add_argument("--max_simplified", type=int, default=300, help="rules simplified in part B")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="processes for the instance rules (default: the SLURM allocation, else 1)")
    p.add_argument("--out_dir", default="results/ablation/explanations")
    args = p.parse_args()
    for dataset in args.datasets:
        for run_id in args.runs:
            run(args, dataset, run_id)


if __name__ == "__main__":
    main()
