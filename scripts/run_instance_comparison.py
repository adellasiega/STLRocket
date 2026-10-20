#!/usr/bin/env python
"""
Instance and global explanations of STLRocket (stlrocket/instance_rules.py, paper Section
"Explanations") against other explainers of the same classifier, on the official train/test split.

The formula bank and the lasso are built on train as in scripts/run_explanations.py (same
settings and seed = base_seed + run, hence the same model). Every explainer works on the same
features, the formulae selected by the classifier (the only ones it reads), and every
explanation is a conjunction of thresholds on their robustness, i.e. an STL formula.

Instance explanations (one per explained series):
  stlrocket  ours: counts train series and maximises the Wilson lower bound, at most --n_max literals
  axp        the exact subset-minimal abductive explanation on the box of the train values of
             the selected formulae, as literals (abductive.axp): sufficient for the prediction on
             the whole box, whatever its length
  iaxp       the inflated AXp (abductive.inflated_axp): the bounds of axp widened as far as possible
  lore       LORE (Guidotti et al. 2018; package lore_sa, TabularGeneticGeneratorLore with its
             defaults): the root-to-leaf path of the series in a decision tree fitted to a
             genetic neighbourhood of 1000 synthetic samples
  anchors    Anchors (Ribeiro et al. 2018; package anchor-exp, AnchorTabularExplainer with its
             default quartile discretizer), at most --n_max predicates and threshold
             --anchor_threshold on its estimated precision
Global explanations, per class predicted on train:
  <instance method>  the greedy cover (instance_rules.select_disjuncts, --min_new_share) of the
                     method's explanations of the train series, made distinct by covered set
                     (shorter, then more TP)
  tree, ripper       global surrogates trained on the model's predictions (global_baselines)

    python scripts/run_instance_comparison.py --datasets ERing --runs 0 1

Every method explains the same series: all those of each split, or a random --max_explained of
them, in a random order drawn once per run (LORE and Anchors the first ones). LORE and Anchors run in --n_jobs processes; a series
one of them has not explained within its timeout (--lore_timeout, --anchor_timeout) is recorded
with status "timeout", and after --max_timeouts consecutive timeouts the method stops for the run
(status "skipped" for the remaining series). Every rule is evaluated exactly on train and test as
the mask of the series satisfying it, and scored w.r.t. the classes predicted by the model
(rule_metrics; the explained series is left out of its own split). Global explanations are
scored on test per class predicted on train (a class without rules has F1 0); those from
instance explanations from every explained train series (subset "all") and, for comparison on
the same series, from those LORE and Anchors attempted (subsets "lore", "anchors"). On the
largest models LORE and Anchors explain only the first --sampled_max_explained series per split.

Written to OUT_DIR, one set of files per dataset and run (prefix <dataset>_run<r>):
  .instances.csv.gz  one row per (method, split, explained series): status, rule, length,
                     counts, precision and coverage on train and test, time; Anchors also its own
                     precision and coverage estimates, LORE its fidelity on its neighbourhood
                     and the class of its rule (rule_class, that of its surrogate tree: every rule
                     is scored for the model's class), both the number of classifier calls; AXp its size in features (n_features)
  .global.csv        one row per (method, subset, class): rules, literals, precision, coverage
                     and F1 on test
  .summary.csv       one row per method: headline means
  .done.json         settings, model, surrogates and status; written last, so it marks the run
                     as complete

Runs whose .done.json exists are skipped, so a resubmitted job only redoes unfinished runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import random
import signal
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
from stlrocket.abductive import axp, inflated_axp
from stlrocket.global_baselines import ripper_surrogate, tree_surrogate

SPLITS = ("train", "test")
METHODS = ("stlrocket", "axp", "iaxp", "lore", "anchors")
SAMPLED = ("lore", "anchors")          # run in worker processes, with a timeout
SURROGATES = ("tree", "ripper")


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout


def with_timeout(seconds: float, fn, *args):
    """fn(*args), or None after `seconds`. SIGALRM interrupts it between two Python bytecodes,
    so at most one numpy call late."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*args)
    except Timeout:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


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

    def explain(self, x: np.ndarray, seed: int) -> dict | None:
        from anchor.anchor_base import AnchorBaseBeam
        np.random.seed(seed)    # anchor-exp samples with the global numpy stream
        sample_fn, mapping = self.explainer.get_sample_fn(x[self.S], self.predict)

        def sample_int8(present, n, compute_labels=True):
            # anchor_beam preallocates batch_size * 10^4 rows of samples: the 0/1 indicators as
            # int8, and no raw samples (float64, only used for the examples it returns)
            raw, data, labels = sample_fn(present, n, compute_labels)
            return raw[:, :0], data.astype(np.int8), labels

        self.n_calls = 0
        exp = with_timeout(self.cfg["timeout"], lambda: AnchorBaseBeam.anchor_beam(
            sample_int8, delta=self.cfg["delta"], epsilon=self.cfg["tau"], batch_size=self.cfg["batch_size"],
            desired_confidence=self.cfg["threshold"], beam_size=self.cfg["beam_size"],
            max_anchor_size=self.cfg["n_max"]))
        if exp is None:
            return None
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


class Lore:
    """LORE (lore_sa.TabularGeneticGeneratorLore, defaults) on the selected features S of the
    classifier W. Its training data are the train series with the classes predicted by the
    model, as class labels 'c<k>' (lore_sa needs a categorical target). The rule's conditions
    (feature > t or feature <= t, from the surrogate tree) are reduced to the tightest bound per
    feature and side, as literals (j, e, s) of instance_rules; x <= t becomes -x > -t - ulp."""

    def __init__(self, F_tr, W, S, pred_tr, cfg: dict):
        from lore_sa.bbox import AbstractBBox
        from lore_sa.dataset import TabularDataset
        from lore_sa.lore import TabularGeneticGeneratorLore
        WS, outer = W[:, S], self

        class BBox(AbstractBBox):
            def predict(self, X):
                X = np.asarray(X, dtype=float)
                outer.n_calls += len(X)
                return np.array([f"c{k}" for k in np.argmax(X @ WS.T, axis=1)])

            def predict_proba(self, X):
                z = np.asarray(X, dtype=float) @ WS.T
                p = np.exp(z - z.max(axis=1, keepdims=True))
                return p / p.sum(axis=1, keepdims=True)

        df = pd.DataFrame(F_tr[:, S], columns=[f"x{j}" for j in S])
        df["y"] = [f"c{k}" for k in pred_tr]
        self.S, self.cfg, self.n_calls = S, cfg, 0
        self.explainer = TabularGeneticGeneratorLore(BBox(None), TabularDataset(df, class_name="y"))

    def explain(self, x: np.ndarray, seed: int) -> dict | None:
        random.seed(seed)       # deap and lore_sa sample with the global random and numpy streams
        np.random.seed(seed)
        self.n_calls = 0
        e = with_timeout(self.cfg["timeout"], self.explainer.explain, x[self.S])
        if e is None:
            return None
        lo, hi = {}, {}
        for p in e["rule"]["premises"]:
            j, t = int(p["attr"][1:]), float(p["val"])
            if p["op"] == ">":
                lo[j] = max(lo.get(j, -np.inf), t)
            elif p["op"] == "<=":
                hi[j] = min(hi.get(j, np.inf), t)
            else:
                raise ValueError(f"unexpected LORE condition {p}")
        literals = ([(j, 1, t) for j, t in lo.items()]
                    + [(j, -1, float(np.nextafter(-t, -np.inf))) for j, t in hi.items()])
        return dict(literals=literals, k=int(str(e["rule"]["consequence"]["val"])[1:]),
                    fidelity=float(e["fidelity"]), n_calls=self.n_calls)


# the explainer of the worker processes, set before they are forked
_WORKER: dict = {}


def _work(task):
    from threadpoolctl import threadpool_limits
    pos, x, seed = task
    with threadpool_limits(1):      # one BLAS thread per process: n_jobs processes fill the allocation
        t = time.perf_counter()
        out = _WORKER["explainer"].explain(x, seed)
        return pos, out, time.perf_counter() - t


def explain_sampled(explainer, X: np.ndarray, seeds: list, n_jobs: int, max_timeouts: int, name: str):
    """explainer.explain on every row of X in order, in n_jobs forked processes, in batches;
    stops after max_timeouts consecutive timeouts. Returns [(status, result, seconds)]."""
    _WORKER["explainer"] = explainer
    out = [("skipped", None, np.nan)] * len(X)
    batch = max(1, 4 * n_jobs)
    ctx = mp.get_context("fork")
    with ctx.Pool(n_jobs) if n_jobs > 1 else _Serial() as pool:
        run = 0
        for start in range(0, len(X), batch):
            tasks = [(p, X[p], seeds[p]) for p in range(start, min(start + batch, len(X)))]
            for p, res, secs in pool.imap(_work, tasks):
                out[p] = ("ok" if res is not None else "timeout", res, secs)
            for p in range(start, start + len(tasks)):
                run = run + 1 if out[p][0] == "timeout" else 0
            if run >= max_timeouts:
                log(f"{name}: {run} consecutive timeouts, stopped for this run after {start + len(tasks)} series")
                break
    return out


class _Serial:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap(self, fn, tasks):
        return map(fn, tasks)


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


def surrogate_scores(sur: dict, pred_tr: np.ndarray, pred_te: np.ndarray) -> list[dict]:
    """global_scores for a global surrogate (global_baselines): its OR for each class."""
    out = []
    for k in np.unique(np.r_[pred_tr, pred_te]):
        c = sur["classes"].get(int(k))
        mask = c["masks"]["test"] if c else np.zeros(len(pred_te), dtype=bool)
        m = ir.rule_metrics(mask, int(k), pred_te)
        if np.isnan(m["coverage"]):
            continue
        rules = c["rules"] if c else []
        out.append({"k": int(k), "n_rules": len(rules), "length": sum(len(r) for r in rules), **m,
                    "f1": f1(m) if rules else 0.0})
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
    methods = [m for m in METHODS if m not in args.skip]

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
            "batch_size": args.anchor_batch_size, "beam_size": args.anchor_beam_size, "n_max": args.n_max,
            "timeout": args.anchor_timeout}
    lcfg = {"timeout": args.lore_timeout}
    done.update(rules=dataclasses.asdict(rcfg), anchors=acfg, lore=lcfg)

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
    explained = {s: rng.permutation(len(F[s]))[:args.max_explained] for s in SPLITS}
    anchors = Anchors(F_tr, W, S, classes, acfg) if "anchors" in methods else None
    lore = Lore(F_tr, W, S, pred["train"], lcfg) if "lore" in methods else None
    D = {s: anchors.bins(F[s]) for s in SPLITS} if anchors else None
    lo, hi = F_tr.min(axis=0), F_tr.max(axis=0)     # AXp domain: the box of the train values

    rows, rules = [], {m: [] for m in methods}
    times = {}
    for own in SPLITS:
        idx = explained[own]
        res = {}
        # STLRocket: batched (and possibly parallel), time per series is the mean
        t = time.perf_counter()
        ours = ir.explain_instances(F[own][idx], W, F_tr, pred["train"], rcfg, n_jobs=args.n_jobs,
                                    exclude=list(idx) if own == "train" else None)
        t_ours = (time.perf_counter() - t) / len(idx)
        res["stlrocket"] = [("ok" if r["literals"] else "not_found", r, t_ours) for r in ours]
        for name, fn in (("axp", axp), ("iaxp", inflated_axp)):
            if name in methods:
                res[name] = []
                for i in idx:
                    t = time.perf_counter()
                    r = fn(F[own][i], W, lo, hi)
                    res[name].append(("ok" if r["sufficient"] else "not_found", r, time.perf_counter() - t))
        seeds = [seed * 1_000_003 + int(i) + (0 if own == "train" else 500_009) for i in idx]
        for name, explainer in (("lore", lore), ("anchors", anchors)):
            if explainer is not None:
                # Anchors' samples take batch_size * 10^4 * |S| bytes per process
                n_jobs = args.n_jobs if name == "lore" else max(1, min(args.n_jobs, int(
                    args.anchor_mem_gb * 1e9 / (args.anchor_batch_size * 1e4 * len(S)))))
                t = time.perf_counter()
                n = len(idx) if args.sampled_max_explained is None else min(len(idx), args.sampled_max_explained)
                res[name] = explain_sampled(explainer, F[own][idx[:n]], seeds[:n], n_jobs, args.max_timeouts,
                                            f"{own} {name}")
                res[name] += [(None, None, np.nan)] * (len(idx) - n)     # not attempted: no row
                times[f"{name}_{own}_s"] = time.perf_counter() - t

        for p, i in enumerate(idx):
            attempted = {m: res[m][p][0] in ("ok", "timeout") for m in SAMPLED if m in res}
            for method in methods:
                status, r, secs = res[method][p]
                if status is None:
                    continue
                extra, k, literals, bins = {}, int(pred[own][i]), None, None
                if r is not None:
                    literals = r["literals"]     # scored for the model's class k, whatever the explainer's
                    if method == "stlrocket":
                        extra = {"rule": json.dumps(r["literals"]), "lcb_train": r["lcb"]}
                    elif method in ("axp", "iaxp"):
                        extra = {"rule": json.dumps(r["literals"]), "n_features": len(r["features"])}
                    elif method == "lore":
                        extra = {"rule": json.dumps(r["literals"]), "fidelity": r["fidelity"], "n_calls": r["n_calls"],
                                 "rule_class": r["k"]}
                    else:
                        extra = {"rule": r["rule"], "n_features": r["n_features"], "est_precision": r["est_precision"],
                                 "est_coverage": r["est_coverage"], "n_calls": r["n_calls"]}
                        bins = D
                if status != "ok":
                    masks = None
                elif bins is not None:
                    masks = {s: anchors.mask(literals, bins[s]) for s in SPLITS}
                else:   # an empty rule (LORE, Anchors: a pure neighbourhood) covers every series
                    masks = {s: ir.covered(literals, F[s]) for s in SPLITS}
                row = {"method": method, "split": own, "i": int(i), "k": int(k), "status": status,
                       "found": masks is not None, "length": len(literals) if literals else 0,
                       "time_s": secs, **extra}
                if masks is not None:
                    for s in SPLITS:
                        m = ir.rule_metrics(masks[s], k, pred[s], exclude=int(i) if s == own else None)
                        row.update({f"{c}_{s}": m[c] for c in ("support", "tp", "precision", "coverage")})
                rows.append(row)
                if own == "train":
                    rules[method].append({"k": k, "length": row["length"], "tp_train": row.get("tp_train", 0),
                                          "mask_train": masks["train"] if masks else None,
                                          "mask_test": masks["test"] if masks else None,
                                          "attempted": attempted})
        log(f"{own}: {len(idx)} series explained, s/series: "
            + ", ".join(f"{m} {np.nanmean([x[2] for x in res[m]]) if any(x[0] != 'skipped' for x in res[m]) else np.nan:.3g}"
                        f" ({sum(x[0] == 'ok' for x in res[m])} ok)" for m in methods))

    inst = pd.DataFrame(rows)
    inst.to_csv(f"{prefix}.instances.csv.gz", index=False)
    # subset "all": from every train series the method explained; "lore", "anchors": from the
    # train series that method attempted (found or timed out), the same for every method
    subsets = ["all"] + [m for m in SAMPLED if m in methods]
    glob = [{"method": m, "subset": sub, **g} for m in methods for sub in subsets
            for g in global_scores([r for r in rules[m] if sub == "all" or r["attempted"].get(sub)],
                                   pred["train"], pred["test"], args.min_new_share)
            if sub == "all" or any(r["attempted"].get(sub) for r in rules[m])]
    for name in SURROGATES:
        if name in args.skip:
            continue
        t = time.perf_counter()
        sur = (tree_surrogate(F, S, pred, args.cv, seed, n_jobs=args.n_jobs) if name == "tree"
               else ripper_surrogate(F, S, pred, seed))
        times[f"{name}_s"] = time.perf_counter() - t
        sur_pred = {s: np.full(len(F[s]), -1) for s in SPLITS}
        for k, c in sur["classes"].items():
            for s in SPLITS:
                sur_pred[s][c["masks"][s] & (sur_pred[s] < 0)] = k
        done[name] = {**{key: v for key, v in sur.items() if key != "classes"},
                      "fidelity": {s: float(balanced_accuracy_score(pred[s], sur_pred[s])) for s in SPLITS},
                      "time_s": times[f"{name}_s"]}
        glob += [{"method": name, "subset": "all", **g} for g in surrogate_scores(sur, pred["train"], pred["test"])]
        log(f"{name}: test fidelity {done[name]['fidelity']['test']:.3f}, {times[f'{name}_s']:.1f} s")
    glob = pd.DataFrame(glob)
    glob.to_csv(f"{prefix}.global.csv", index=False)

    summary = []
    for m in list(methods) + [s for s in SURROGATES if s not in args.skip]:
        d = inst[(inst.method == m) & (inst.split == "test") & (inst.status != "skipped")]
        found = d[d.found]
        g = glob[(glob.method == m) & (glob.subset == "all")]
        summary.append({
            "dataset": dataset, "run": run_id, "method": m, "n_explained": len(d),
            "found": float(d.found.mean()) if len(d) else np.nan,
            "pooled_precision_test": float(found.tp_test.sum() / found.support_test.sum())
                                     if len(found) and found.support_test.sum() else np.nan,
            "mean_precision_test": float(found.precision_test.mean()) if len(found) else np.nan,
            "mean_coverage_test": float(found.coverage_test.mean()) if len(found) else np.nan,
            "mean_precision_train": float(found.precision_train.mean()) if len(found) else np.nan,
            "mean_length": float(found.length.mean()) if len(found) else np.nan,
            "time_s": float(d.time_s.mean()) if len(d) else np.nan,
            "global_precision_test": float(g.precision.mean()), "global_coverage_test": float(g.coverage.mean()),
            "global_f1_test": float(g.f1.mean()), "global_rules": float(g.n_rules.mean()),
            "global_length": float(g.length.mean())})
    pd.DataFrame(summary).to_csv(f"{prefix}.summary.csv", index=False)
    done.update(status="ok", times_s=times, total_s=time.perf_counter() - t0)
    Path(f"{prefix}.done.json").write_text(json.dumps(done, indent=1, default=float))
    for s in summary:
        log(f"  {s['method']:10s} test precision {s['pooled_precision_test']:.3f}, coverage "
            f"{s['mean_coverage_test']:.3f}, length {s['mean_length']:.2f}, global F1 {s['global_f1_test']:.3f} "
            f"({s['global_rules']:.1f} rules), {s['time_s']:.3g} s/series")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--runs", type=int, nargs="+", default=list(range(10)))
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
    p.add_argument("--anchor_timeout", type=float, default=600.0, help="Anchors: seconds per series")
    p.add_argument("--anchor_mem_gb", type=float, default=24.0,
                   help="Anchors: memory for its samples, which bounds its processes")
    p.add_argument("--lore_timeout", type=float, default=900.0, help="LORE: seconds per series")
    p.add_argument("--max_timeouts", type=int, default=8,
                   help="LORE, Anchors: consecutive timeouts after which the method stops for the run")
    p.add_argument("--max_explained", type=int, default=None, help="series explained per split (default: all)")
    p.add_argument("--sampled_max_explained", type=int, default=None,
                   help="LORE, Anchors: the first series of those explained per split (default: all)")
    p.add_argument("--skip", nargs="*", default=[], choices=list(METHODS[1:]) + list(SURROGATES),
                   help="methods not to run")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
                   help="processes for ours, LORE, Anchors and the tree's CV")
    p.add_argument("--out_dir", default="results/instance_comparison")
    args = p.parse_args()
    for dataset in args.datasets:
        for run_id in args.runs:
            run(args, dataset, run_id)


if __name__ == "__main__":
    main()
