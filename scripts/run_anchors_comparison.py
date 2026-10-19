#!/usr/bin/env python
"""
Instance explanations of STLRocket (stlrocket/instance_rules.py, paper Section "Explanations")
against Anchors (Ribeiro et al. 2018, package anchor-exp), on the official train/test split.

The formula bank and the lasso are built on train as in scripts/run_explanations.py (same
settings and seed = base_seed + run, hence the same model), and both methods explain that
classifier on the same features: Anchors is AnchorTabularExplainer on the selected formulae
(the only features the classifier reads), with its default quartile discretizer, so each
anchor is a conjunction of thresholds on robustness, i.e. also an STL formula. The two
methods differ in the search and in how precision is estimated: Anchors samples
perturbations from train rows conditioned on the anchor and stops at an estimated precision
>= --anchor_threshold; ours counts train series and maximises the Wilson lower bound.

    python scripts/run_anchors_comparison.py --datasets ERing --runs 0 1

The same series are explained by both methods: at most --max_explained test series (instance
metrics) and as many train series (global explanations), drawn once per run. For every
explained series and method the rule is evaluated exactly on train and test as the mask of the
series satisfying it, and scored w.r.t. the classes predicted by the model (rule_metrics; the
explained series is left out of its own split). Global explanations: for each method, the rules
of the explained train series are made distinct by covered set (shorter, then more TP) and
selected by the same greedy cover (instance_rules.select_disjuncts, min_new_share), and the OR is
scored on test per class predicted on train (a class without rules has F1 0).

Written to OUT_DIR, one set of files per dataset and run (prefix <dataset>_run<r>):
  .instances.csv.gz  one row per (method, split, explained series): rule, length, counts,
                     precision and coverage on train and test, time; Anchors also its own
                     precision and coverage estimates and the number of classifier calls
  .global.csv        one row per (method, class): disjuncts, precision, coverage and F1 on test
  .summary.csv       one row per method: headline means
  .done.json         settings and status; written last, so it marks the run as complete

Runs whose .done.json exists are skipped, so a resubmitted job only redoes unfinished runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_explanation_ablation import f1
from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.features import build_formula_bank
from stlrocket.classifier import train_classifier
from stlrocket import instance_rules as ir

SPLITS = ("train", "test")
METHODS = ("stlrocket", "anchors")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Anchors:
    """Anchors on the selected features S of the classifier W, run as
    AnchorTabularExplainer.explain_instance does, but keeping the predicates: each is
    (f, op, v) on the quartile bin of feature S[f], 'leq' bin <= v or 'geq' bin > v
    (bin = number of quartile borders below the value, lime's np.searchsorted)."""

    def __init__(self, F_tr, W, S, classes, cfg: dict):
        from anchor.anchor_tabular import AnchorTabularExplainer
        self.S, self.WS, self.cfg = S, W[:, S], cfg
        self.explainer = AnchorTabularExplainer([str(c) for c in classes], [f"phi_{j}" for j in S], F_tr[:, S])
        self.n_calls = 0

    def predict(self, Z) -> np.ndarray:
        Z = np.asarray(Z)
        self.n_calls += len(Z)
        return np.argmax(Z @ self.WS.T, axis=1)

    def bins(self, F: np.ndarray) -> np.ndarray:
        return self.explainer.disc.discretize(F[:, self.S])

    def border(self, f: int, v: int) -> float:
        """Upper border of quartile bin v of feature S[f], in feature units."""
        return float(self.explainer.disc.lambdas[f].__defaults__[0][v])

    def explain(self, x: np.ndarray) -> dict:
        from anchor.anchor_base import AnchorBaseBeam
        sample_fn, mapping = self.explainer.get_sample_fn(x[self.S], self.predict)

        def sample_int8(present, n, compute_labels=True):
            # 0/1 indicators as int8: anchor_beam preallocates 10^6 rows of them
            raw, data, labels = sample_fn(present, n, compute_labels)
            return raw, data.astype(np.int8), labels

        self.n_calls = 0
        exp = AnchorBaseBeam.anchor_beam(sample_int8, delta=self.cfg["delta"], epsilon=self.cfg["tau"],
                                         batch_size=self.cfg["batch_size"],
                                         desired_confidence=self.cfg["threshold"],
                                         beam_size=self.cfg["beam_size"], max_anchor_size=self.cfg["n_max"])
        preds = [mapping[i] for i in exp["feature"]]
        # tightest bound per (feature, side): these are the literals of the anchor
        lo, hi = {}, {}
        for f, op, v in preds:
            if op == "geq":
                lo[f] = max(lo.get(f, -1), v)
            else:
                hi[f] = min(hi.get(f, 10 ** 9), v)
        literals = ([(f, "geq", v) for f, v in lo.items()] + [(f, "leq", v) for f, v in hi.items()])
        rule = " & ".join([f"x{self.S[f]} > {self.border(f, v):.4g}" for f, v in lo.items()]
                          + [f"x{self.S[f]} <= {self.border(f, v):.4g}" for f, v in hi.items()])
        est = exp["precision"][-1] if preds else exp["all_precision"]
        cov = exp["coverage"][-1] if preds else 1.0
        return dict(literals=literals, n_features=len({f for f, _, _ in literals}), rule=rule,
                    est_precision=float(est), est_coverage=float(cov), n_calls=self.n_calls)

    def mask(self, literals: list, D: np.ndarray) -> np.ndarray:
        m = np.ones(len(D), dtype=bool)
        for f, op, v in literals:
            m &= D[:, f] > v if op == "geq" else D[:, f] <= v
        return m


def global_scores(rules: list[dict], pred_tr: np.ndarray, pred_te: np.ndarray, min_new_share: float) -> list[dict]:
    """Greedy-cover global explanation per class from instance rules {k, length, tp_train,
    mask_train, mask_test}, scored on test. Classes are those predicted on test, as in
    run_explanation_ablation.global_summary: a class without rules (e.g. never predicted on
    train) has the empty explanation, coverage and F1 0; a class not predicted on test is skipped."""
    out = []
    for k in np.unique(np.r_[pred_tr, pred_te]):
        distinct = {}
        for r in rules:
            if r["k"] != k or r["mask_train"] is None:
                continue
            key = r["mask_train"].tobytes()
            cur = distinct.get(key)
            if cur is None or (r["length"], -r["tp_train"]) < (cur["length"], -cur["tp_train"]):
                distinct[key] = r
        R = list(distinct.values())
        pos = pred_tr == k
        # select_disjuncts reads only len() of a rule given its mask and returns the rules
        # themselves: pass placeholders of the rule's length and map them back by identity
        placeholders = [[None] * r["length"] for r in R]
        chosen = ir.select_disjuncts(placeholders, None, pos, max(1, int(np.ceil(min_new_share * pos.sum()))),
                                     masks=[r["mask_train"] for r in R]) if R else []
        sel = [R[next(i for i, p in enumerate(placeholders) if p is c)] for c in chosen]
        mask = np.zeros(len(pred_te), dtype=bool)
        for r in sel:
            mask |= r["mask_test"]
        m = ir.rule_metrics(mask, int(k), pred_te)
        if np.isnan(m["coverage"]):     # class not predicted on test: nothing to score
            continue
        out.append({"k": int(k), "n_rules": len(sel), "length": sum(r["length"] for r in sel), **m,
                    "f1": f1(m) if sel else 0.0})
    return out


def run(args, dataset: str, run_id: int) -> None:
    out_dir = Path(args.out_dir)
    prefix = out_dir / f"{dataset}_run{run_id}"
    if Path(f"{prefix}.done.json").exists():
        log(f"{dataset} run {run_id}: already complete, skipped")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.base_seed + run_id
    done = {"dataset": dataset, "run": run_id, "seed": seed, "args": vars(args)}

    t0 = time.perf_counter()
    X_tr, y_tr, X_te, y_te = load_dataset(dataset)
    cfg = ExperimentConfig(
        dataset=dataset, n_formulas=args.n_formulas, depth_max=args.depth, until_weight=0.0,
        cv=args.cv, cut_point=args.cut_point, fit_intercept=False, n_run=1, base_seed=args.base_seed,
        output_dir=str(out_dir), device=args.device,
        explain=False, pool_size=0, precision_threshold=0.0, simplify_agreement=0.0, simplify_min_gain=0.0,
        simplify_decimals=0,
    )
    rcfg = ir.RuleConfig(n_max=args.n_max, confidence=args.confidence, min_new_share=args.min_new_share)
    acfg = {"threshold": args.anchor_threshold, "delta": args.anchor_delta, "tau": args.anchor_tau,
            "batch_size": args.anchor_batch_size, "beam_size": args.anchor_beam_size, "n_max": args.n_max}
    done.update(rules=dataclasses.asdict(rcfg), anchors=acfg)

    formulas, F_tr, F_te, sigma = build_formula_bank(X_tr, X_te, y_tr, cfg, seed)
    model = train_classifier(F_tr, y_tr, cfg, seed=seed)
    W = ir.model_logits_matrix(model)
    classes = model.classes_
    index = {c: i for i, c in enumerate(classes)}
    F = {"train": F_tr, "test": F_te}
    pred = {s: ir.predict_indices(F[s], W) for s in SPLITS}
    true = {"train": np.array([index[c] for c in y_tr]), "test": np.array([index[c] for c in y_te])}
    S = np.flatnonzero(np.any(W != 0, axis=0))
    done.update(n_train=len(X_tr), n_test=len(X_te), n_classes=len(classes), n_selected=int(len(S)),
                balanced_accuracy={s: float(balanced_accuracy_score(true[s], pred[s])) for s in SPLITS},
                model_s=time.perf_counter() - t0)
    log(f"{dataset} run {run_id}: {len(S)} formulae selected, balanced accuracy test "
        f"{done['balanced_accuracy']['test']:.3f}")
    if len(S) == 0:
        done["status"] = "empty_model"
        Path(f"{prefix}.done.json").write_text(json.dumps(done, indent=1))
        log("no formula selected: nothing to explain")
        return

    rng = np.random.default_rng(seed)
    explained = {s: np.sort(rng.choice(len(F[s]), min(args.max_explained, len(F[s])), replace=False))
                 for s in SPLITS}
    anchors = Anchors(F_tr, W, S, classes, acfg)
    D = {s: anchors.bins(F[s]) for s in SPLITS}

    rows, rules = [], {m: [] for m in METHODS}
    for own in SPLITS:
        idx = explained[own]
        # STLRocket: batched (and possibly parallel), time per series is the mean
        t = time.perf_counter()
        ours = ir.explain_instances(F[own][idx], W, F_tr, pred["train"], rcfg, n_jobs=args.n_jobs,
                                    exclude=list(idx) if own == "train" else None)
        t_ours = (time.perf_counter() - t) / len(idx)
        np.random.seed(seed)    # anchor-exp samples with the global numpy stream
        for i, r in zip(idx, ours):
            t = time.perf_counter()
            a = anchors.explain(F[own][i])
            t_anchor = time.perf_counter() - t
            for method, k, length, masks, extra, secs in (
                ("stlrocket", r["k"], len(r["literals"]),
                 {s: ir.covered(r["literals"], F[s]) for s in SPLITS} if r["literals"] else None,
                 {"rule": json.dumps(r["literals"]), "lcb_train": r["lcb"]}, t_ours),
                ("anchors", int(pred[own][i]), len(a["literals"]),
                 {s: anchors.mask(a["literals"], D[s]) for s in SPLITS},
                 {"rule": a["rule"], "n_features": a["n_features"], "est_precision": a["est_precision"],
                  "est_coverage": a["est_coverage"], "n_calls": a["n_calls"]}, t_anchor),
            ):
                row = {"method": method, "split": own, "i": int(i), "k": int(k), "found": masks is not None,
                       "length": length, "time_s": secs, **extra}
                if masks is not None:
                    for s in SPLITS:
                        m = ir.rule_metrics(masks[s], k, pred[s], exclude=int(i) if s == own else None)
                        row.update({f"{c}_{s}": m[c] for c in ("support", "tp", "precision", "coverage")})
                rows.append(row)
                if own == "train":
                    rules[method].append({"k": k, "length": length, "tp_train": row.get("tp_train", 0),
                                          "mask_train": masks["train"] if masks else None,
                                          "mask_test": masks["test"] if masks else None})
        log(f"{own}: {len(idx)} series explained, {t_ours:.2f} s/series ours, "
            f"{np.mean([r['time_s'] for r in rows if r['method'] == 'anchors' and r['split'] == own]):.2f} s/series Anchors")

    inst = pd.DataFrame(rows)
    inst.to_csv(f"{prefix}.instances.csv.gz", index=False)
    glob = pd.DataFrame([{"method": m, **g} for m in METHODS
                         for g in global_scores(rules[m], pred["train"], pred["test"], args.min_new_share)])
    glob.to_csv(f"{prefix}.global.csv", index=False)

    summary = []
    for m in METHODS:
        d = inst[(inst.method == m) & (inst.split == "test")]
        found = d[d.found]
        g = glob[glob.method == m]
        summary.append({
            "dataset": dataset, "run": run_id, "method": m, "n_explained": len(d), "found": float(d.found.mean()),
            "pooled_precision_test": float(found.tp_test.sum() / max(found.support_test.sum(), 1)),
            "mean_precision_test": float(found.precision_test.mean()),
            "mean_coverage_test": float(found.coverage_test.mean()),
            "mean_precision_train": float(found.precision_train.mean()),
            "mean_length": float(found.length.mean()), "time_s": float(d.time_s.mean()),
            "global_precision_test": float(g.precision.mean()), "global_coverage_test": float(g.coverage.mean()),
            "global_f1_test": float(g.f1.mean()), "global_rules": float(g.n_rules.mean())})
    pd.DataFrame(summary).to_csv(f"{prefix}.summary.csv", index=False)
    done.update(status="ok", total_s=time.perf_counter() - t0)
    Path(f"{prefix}.done.json").write_text(json.dumps(done, indent=1))
    for s in summary:
        log(f"  {s['method']:10s} test precision {s['mean_precision_test']:.3f}, coverage "
            f"{s['mean_coverage_test']:.3f}, length {s['mean_length']:.2f}, global F1 {s['global_f1_test']:.3f}, "
            f"{s['time_s']:.2f} s/series")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--runs", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--n_formulas", type=int, default=10000)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--cut_point", type=float, default=1.0)
    defaults = ir.RuleConfig()
    p.add_argument("--n_max", type=int, default=defaults.n_max, help="max literals (ours) / predicates (Anchors)")
    p.add_argument("--confidence", type=float, default=defaults.confidence, help="confidence of the Wilson bound")
    p.add_argument("--min_new_share", type=float, default=defaults.min_new_share,
                   help="global: share of its class a rule must newly cover to enter the OR")
    p.add_argument("--anchor_threshold", type=float, default=0.95, help="Anchors: precision threshold")
    p.add_argument("--anchor_delta", type=float, default=0.1)
    p.add_argument("--anchor_tau", type=float, default=0.15)
    p.add_argument("--anchor_batch_size", type=int, default=100)
    p.add_argument("--anchor_beam_size", type=int, default=4)
    p.add_argument("--max_explained", type=int, default=200, help="series explained per split (same for both methods)")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="processes for our instance rules (Anchors runs serially)")
    p.add_argument("--out_dir", default="results/anchors")
    args = p.parse_args()
    for dataset in args.datasets:
        for run_id in args.runs:
            run(args, dataset, run_id)


if __name__ == "__main__":
    main()
