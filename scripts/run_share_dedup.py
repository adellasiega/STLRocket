#!/usr/bin/env python
"""
Validation-only experiment: the worst-case margin share of instance rules, with and without
removing duplicate feature columns before the lasso. The test set is never loaded.

For an explained series with features x, predicted class k and rule formulae E,
A_lj = (w_kj - w_lj) x_j and m_l = sum_j A_lj = z_k - z_l > 0 (no intercept), and

    share(E) = min_{l != k} sum_{j in E} A_lj / m_l

sum_{j in E} A_lj is the margin over l once every formula outside E is set to 0, its
class-balanced train mean, so share > 0 iff E alone (the rest at the mean) still predicts k.
share can exceed 1 (formulae outside E with A_lj < 0) or be negative (a formula of E with
A_lj < 0 for some l).

The lasso spreads weight over copies of the same feature (scripts/run_dedup_ablation.py), so a
rule holding one copy leaves the others' weight in the denominator only. Two arms on the same
fit/validation split and formula bank, as in scripts/run_explanation_ablation.py:
  full   lasso on all M columns
  dedup  lasso on the columns kept by features.correlation_filter on the fit features in
         float64 at --dedup_threshold (|r| with every earlier kept column below it)
Each arm explains the same series with the same RuleConfig. Per explained series the rows hold
the run_explanation_ablation columns (rule, counts on fit and validation, faithfulness) and:
  share          share(E) as above
  share_groups   share of E plus every selected formula whose |r| on fit with a formula of E
                 is >= --dedup_threshold (its copies in the model): within one model, the part of
                 a full-vs-dedup difference due to duplicates
  n_copies       how many selected formulae outside E that adds
  share_pos      share > 0

Written to OUT_DIR per dataset and run (prefix <dataset>_run<r>):
  .instances.csv.gz  one row per (arm, explained series)
  .done.json         settings, per-arm model and share summaries; written last, so it marks
                     the run as complete (complete runs are skipped)
"""
from __future__ import annotations

import argparse
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
from run_explanation_ablation import SPLITS, log, instance_rows
from stlrocket.config import ExperimentConfig
from stlrocket.data import standardize_series
from stlrocket.features import build_formula_bank, correlation_filter
from stlrocket.classifier import train_classifier
from stlrocket import instance_rules as ir

ARMS = ("full", "dedup")


def margin_share(x: np.ndarray, W: np.ndarray, cols) -> float:
    """min over competitors l of sum_{j in cols} A_lj / m_l (module docstring)."""
    z = W @ x
    k = int(np.argmax(z))
    others = np.arange(len(z)) != k
    A = (W[k] - W[others]) * x
    m = A.sum(axis=1)
    if np.any(m <= 0):   # tie for the top logit
        return np.nan
    return float((A[:, list(cols)].sum(axis=1) / m).min())


def share_rows(rules: list[dict], idx: np.ndarray, X_own: np.ndarray, W: np.ndarray, copies: dict) -> list[dict]:
    rows = []
    for i, r in zip(idx.tolist(), rules):
        E = [j for j, _, _ in r["literals"]]
        if not E:
            rows.append(dict(share=np.nan, share_groups=np.nan, n_copies=0, share_pos=np.nan))
            continue
        G = set(E).union(*(copies.get(j, ()) for j in E))
        s = margin_share(X_own[i], W, E)
        rows.append(dict(share=s, share_groups=margin_share(X_own[i], W, G), n_copies=len(G) - len(set(E)),
                         share_pos=float(s > 0) if np.isfinite(s) else np.nan))
    return rows


def selected_copies(W: np.ndarray, F_fit: np.ndarray, threshold: float) -> dict:
    """Selected column j -> the other selected columns with |r| >= threshold with it on fit."""
    S = np.flatnonzero(np.any(W != 0, axis=0))
    if len(S) < 2:
        return {}
    R = np.abs(np.corrcoef(F_fit[:, S].astype(np.float64).T))
    np.fill_diagonal(R, 0.0)
    return {int(S[a]): {int(S[b]) for b in np.flatnonzero(R[a] >= threshold)} for a in range(len(S))}


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
        explain=False, pool_size=0, precision_threshold=0.0,
    )
    rcfg = ir.RuleConfig(n_max=args.n_max, beam_width=args.beam_width, confidence=args.confidence,
                         candidates=args.candidates, construction=args.construction)
    t = time.perf_counter()
    _, F_fit_all, F_val_all, _ = build_formula_bank(X["fit"], X["val"], y[fit_idx], cfg, seed)
    log(f"{dataset} run {run_id}: bank of {F_fit_all.shape[1]} formulae ({time.perf_counter() - t:.0f} s)")
    cols = {"full": np.arange(F_fit_all.shape[1]),
            "dedup": correlation_filter(F_fit_all.astype(np.float64), args.dedup_threshold)}

    # Series to explain: the same seeded subsample for both arms
    n = {"fit": len(fit_idx), "val": len(val_idx)}
    explained = {s: np.sort(rng.choice(n[s], args.max_explained, replace=False))
                 if n[s] > args.max_explained else np.arange(n[s]) for s in SPLITS}

    frames, summary = [], {}
    for arm in ARMS:
        t = time.perf_counter()
        F = {"fit": F_fit_all[:, cols[arm]], "val": F_val_all[:, cols[arm]]}
        model = train_classifier(F["fit"], y[fit_idx], cfg, seed=seed)
        W = ir.model_logits_matrix(model)
        index = {c: i for i, c in enumerate(model.classes_)}
        pred = {s: ir.predict_indices(F[s], W) for s in SPLITS}
        true = {"fit": np.array([index[c] for c in y[fit_idx]]), "val": np.array([index[c] for c in y[val_idx]])}
        copies = selected_copies(W, F["fit"], args.dedup_threshold)
        rows = []
        for own in SPLITS:
            # each fit series is left out of its own counts
            rules = ir.explain_instances(F[own][explained[own]], W, F["fit"], pred["fit"], rcfg, n_jobs=args.n_jobs,
                                         exclude=explained[own].tolist() if own == "fit" else None)
            base_rows = instance_rows(rules, explained[own], own, F, pred, W)
            rows += [{**a, **b} for a, b in zip(base_rows, share_rows(rules, explained[own], F[own], W, copies))]
        df = pd.DataFrame(rows)
        df.insert(0, "arm", arm)
        frames.append(df)

        found = df[df.n_literals > 0]
        summary[arm] = {
            "n_bank": len(cols[arm]), "n_selected": int(np.any(W != 0, axis=0).sum()),
            "n_selected_multiclass": int(((W != 0).sum(axis=0) > 1).sum()),
            "n_selected_with_copy": sum(1 for c in copies.values() if c),
            **{f"bacc_{s}": float(balanced_accuracy_score(true[s], pred[s])) for s in SPLITS},
            **{f"{own}_{c}": float(found[found.own == own][c].median())
               for own in SPLITS for c in ("share", "share_groups")},
            **{f"{own}_share_pos": float(found[found.own == own].share_pos.mean()) for own in SPLITS},
            **{f"{own}_length": float(found[found.own == own].n_literals.mean()) for own in SPLITS},
            "time_s": time.perf_counter() - t,
        }
        v = summary[arm]
        log(f"  {arm:5s}: {v['n_bank']} columns, {v['n_selected']} selected ({v['n_selected_with_copy']} with a copy), "
            f"bacc val {v['bacc_val']:.3f}; val rules: median share {v['val_share']:.3f} "
            f"(with copies {v['val_share_groups']:.3f}), share > 0 {v['val_share_pos']:.3f}, "
            f"length {v['val_length']:.2f} ({v['time_s']:.0f} s)")

    inst = pd.concat(frames, ignore_index=True)
    inst.insert(0, "run", run_id)
    inst.insert(0, "dataset", dataset)
    inst.to_csv(out_dir / f"{stem}.instances.csv.gz", index=False)
    done.write_text(json.dumps({**info, "status": "ok", "arms": summary}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--runs", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--val_ratio", type=float, default=0.25)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--n_formulas", type=int, default=10000)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--cut_point", type=float, default=1.0)
    p.add_argument("--dedup_threshold", type=float, default=0.999999,
                   help="|r| at which two columns count as copies (dedup arm and share_groups)")
    defaults = ir.RuleConfig()
    p.add_argument("--construction", default=defaults.construction, choices=ir.CONSTRUCTIONS)
    p.add_argument("--candidates", default=defaults.candidates, choices=ir.CANDIDATE_METHODS)
    p.add_argument("--confidence", type=float, default=defaults.confidence)
    p.add_argument("--n_max", type=int, default=defaults.n_max, help="max literals per rule")
    p.add_argument("--beam_width", type=int, default=defaults.beam_width)
    p.add_argument("--max_explained", type=int, default=1000,
                   help="series explained per split; larger splits are subsampled (same series for both arms)")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="processes for the instance rules (default: the SLURM allocation, else 1)")
    p.add_argument("--out_dir", default="results/ablation/share_dedup")
    args = p.parse_args()
    for dataset in args.datasets:
        for run_id in args.runs:
            run(args, dataset, run_id)


if __name__ == "__main__":
    main()
