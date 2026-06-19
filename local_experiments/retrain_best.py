#!/usr/bin/env python
"""
Retrain the STL *linear* model at the best (n_formulae, depth_max) configuration
found by ``run_comparison.py`` and build local + global explanations exactly as in
``local_experiments/notebook.ipynb``.

Pipeline
--------
1. Read the best ``stl_linear`` configuration (budget -> n_formulae, depth ->
   depth_max) from a ``run_comparison.py`` output directory (``summary.json`` is
   preferred; falls back to averaging ``results.csv`` per-seed rows).
2. Retrain the STL linear model at that configuration, reproducing the notebook's
   feature standardization and ``glmnet.LogitNet`` classifier (``cut_point=0``,
   ``standardize=True``, balanced-accuracy CV scorer).
3. Build per-instance local explanations over the *training* set, aggregate them
   into per-class global explanations, and post-process with the notebook's
   data-aware simplification + threshold rounding.
4. Persist the chosen config + metrics, local/global explanation CSVs, and KDE
   robustness plots (PNG) to an output directory.

The explanation logic here is the refined notebook version, which differs from the
(older) ``stlrocket/explanations.py`` package. Feature extraction, classifier and
robustness primitives are reused from the ``stlrocket`` package.

Usage
-----
    python local_experiments/retrain_best.py \
        --comparison_dir results/comparison/Libras_20260618_160028_d1241ee6 \
        --dataset Libras
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Make the repo root importable when run as a script from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import torch
from glmnet import LogitNet
from glmnet.scorer import make_scorer
from sklearn.metrics import balanced_accuracy_score

from torcheck.stl import Atom, Not, And, Or
from torcheck import simplify

from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.preprocessing import StateVariableStandardScaler
from stlrocket.features import build_formula_bank, eval_robustness, shift_atom_thresholds


# Notebook defaults (cell 2f5bdf9e).
SEED_F0 = 0
LOCAL_EXPL_TOP_M = 5
LOCAL_EXPL_PRECISION_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Robustness helper (notebook signature: list of formulae -> stacked rhos)
# ---------------------------------------------------------------------------

def robustness(TS: np.ndarray, formulae: list) -> np.ndarray:
    """Robustness of each formula on each series. Mirrors the notebook helper but
    delegates the single-formula evaluation to ``stlrocket.features.eval_robustness``."""
    rhos = [eval_robustness(phi, TS) for phi in formulae]
    rhos = np.stack(rhos, axis=1)
    return rhos.ravel() if len(formulae) == 1 else rhos


# ---------------------------------------------------------------------------
# 1. Best-config lookup
# ---------------------------------------------------------------------------

def _best_from_summary(summary: dict) -> tuple[int, int] | None:
    per_config = summary.get("per_config", {})
    best_key, best_acc = None, -np.inf
    for key, rec in per_config.items():
        if rec.get("method") != "stl_linear":
            continue
        acc = rec.get("mean_balanced_accuracy")
        if acc is None:
            continue
        if acc > best_acc:
            best_acc, best_key = acc, rec
    if best_key is None:
        return None
    return int(best_key["budget"]), int(best_key["depth"])


def _best_from_results_csv(csv_path: Path) -> tuple[int, int] | None:
    """Average per-seed ``balanced_accuracy`` within each (budget, depth) group
    for stl_linear and return the argmax group. Mirrors
    ``plot_comparison.best_accuracy_per_method`` (status == 'ok' rows only)."""
    groups: dict[tuple[int, int], list[float]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != "stl_linear" or row["status"] != "ok":
                continue
            try:
                acc = float(row["balanced_accuracy"])
            except (TypeError, ValueError):
                continue
            if np.isnan(acc):
                continue
            groups[(int(row["budget"]), int(row["depth"]))].append(acc)
    if not groups:
        return None
    best = max(groups, key=lambda k: float(np.mean(groups[k])))
    return best


def pick_best_stl_linear(comparison_dir: Path) -> tuple[int, int]:
    """Return (n_formulae, depth_max) of the best stl_linear config."""
    summary_path = comparison_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        best = _best_from_summary(summary)
        if best is not None:
            return best
    csv_path = comparison_dir / "results.csv"
    if csv_path.exists():
        best = _best_from_results_csv(csv_path)
        if best is not None:
            return best
    raise SystemExit(
        f"No usable stl_linear results found in {comparison_dir} "
        "(checked summary.json and results.csv)."
    )


def comparison_dataset(comparison_dir: Path) -> str | None:
    cfg_path = comparison_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text()).get("dataset")
    return None


# ---------------------------------------------------------------------------
# 2. Retrain (notebook cells 2f5bdf9e, 2be66502)
# ---------------------------------------------------------------------------

def retrain(dataset: str, n_formulae: int, depth_max: int, seed: int, cv: int):
    """Returns (model, formulae, X_tr_feats, X_te_feats, std, TS_tr, TS_te,
    y_tr, y_te, classes, metrics)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    TS_tr, y_tr, TS_te, y_te = load_dataset(dataset)
    classes = np.unique(y_tr)

    svss = StateVariableStandardScaler()
    TS_tr = svss.fit_transform(TS_tr)
    TS_te = svss.transform(TS_te)

    config = ExperimentConfig(
        dataset=dataset,
        n_formulas=n_formulae,
        depth_max=depth_max,
        only_temporal=True,
        until_weight=0.0,
        cv=cv,
        explain=False,
    )
    # build_formula_bank samples F0 formulae and mean-centres each feature on the
    # training set (shifting atom thresholds accordingly) -- the notebook's
    # `means` step. Only the /std standardization remains to be applied below.
    formulae, X_tr, X_te = build_formula_bank(TS_tr, TS_te, config, seed)

    std = X_tr.std(axis=0)
    X_tr = X_tr / std
    X_te = X_te / std

    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
    model = LogitNet(
        alpha=1.0,
        n_splits=cv,
        cut_point=0,
        fit_intercept=False,
        standardize=True,
        n_jobs=8,
        max_iter=1_000_000,
        verbose=True,
        scoring=balanced_accuracy_scorer,
    )
    model.fit(X_tr, y_tr)

    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    W = model.coef_

    metrics = {
        "balanced_accuracy_train": float(balanced_accuracy_score(y_tr, y_pred_tr)),
        "balanced_accuracy_test": float(balanced_accuracy_score(y_te, y_pred_te)),
        "nonzero_columns": int((W != 0).any(axis=0).sum()),
        "n_columns": int(W.shape[1]),
    }
    print(f"Balanced accuracy train set: {metrics['balanced_accuracy_train']:.3f}")
    print(f"Balanced accuracy test set: {metrics['balanced_accuracy_test']:.3f}")
    print(f"Non-zero columns of W: {metrics['nonzero_columns']} / {metrics['n_columns']}")

    return model, formulae, X_tr, X_te, std, TS_tr, TS_te, y_tr, y_te, classes, metrics


# ---------------------------------------------------------------------------
# 3a. Local explanations (notebook cell 6e50960b)
# ---------------------------------------------------------------------------

def per_competitor_contributions(W, x, k):
    if W.shape[0] == 1:
        return np.stack([2 * W[k], -2 * W[k]])
    return np.stack([(W[k] - W[l]) * x for l in range(W.shape[0]) if l != k])


def get_top_m_features(C, m):
    aggregated = C.sum(axis=0)
    return np.argsort(-aggregated)[:m].tolist()


def _reparametrize(phi_original, rho, rho_target, k, y_tr):
    """rho = robustness of phi_original on the training set (precomputed)."""
    y = np.asarray(y_tr)
    phi = copy.deepcopy(phi_original)

    other_classes = np.unique(y[y != k])
    med_others = {c: np.median(rho[y == c]) for c in other_classes}
    below = {c: v for c, v in med_others.items() if v < rho_target}
    above = {c: v for c, v in med_others.items() if v > rho_target}

    if not below and not above:
        ref_class = other_classes[0]
    elif not below:
        ref_class = min(above, key=above.get)
    elif not above:
        ref_class = max(below, key=below.get)
    elif len(below) == len(above):
        ref_class = max(med_others, key=lambda c: abs(med_others[c] - rho_target))
    elif len(below) > len(above):
        ref_class = max(below, key=below.get)
    else:
        ref_class = min(above, key=above.get)

    med_ref = med_others[ref_class]
    if rho_target < med_ref:
        phi = Not(phi)
        rho, rho_target, med_ref = -rho, -rho_target, -med_ref

    delta_star = -(rho_target + med_ref) / 2
    shift_atom_thresholds(phi, delta_star)
    return phi, rho + delta_star


def reparametrize_formula_global(phi_original, k, TS_tr, y_tr):
    rho = robustness(TS_tr, [phi_original])
    rho_target = float(np.median(rho[np.asarray(y_tr) == k]))
    phi, _ = _reparametrize(phi_original, rho, rho_target, k, y_tr)
    return phi


def reparametrize_formula_local(phi_original, ts, k, TS_tr, y_tr):
    rho = robustness(TS_tr, [phi_original])
    rho_target = float(robustness(ts[None, ...], [phi_original])[0])
    phi, _ = _reparametrize(phi_original, rho, rho_target, k, y_tr)
    return phi


def greedy_precise_picks(x, X_tr, std, W, model, formulae, y_tr, k, m, precision_threshold):
    y_tr = np.asarray(y_tr)
    target_class = model.classes_[k]

    C = per_competitor_contributions(W, x, k)
    top_m_features = get_top_m_features(C, m)

    usable = {}
    for j in top_m_features:
        rho_j = X_tr[:, j] * std[j]
        rho_target = float(x[j] * std[j])
        phi_j, rho_j_new = _reparametrize(formulae[j], rho_j, rho_target, target_class, y_tr)
        usable[j] = (phi_j, rho_j_new)

    remaining = list(usable)
    picks, reparametrized = [], {}
    rho_current = np.full(len(y_tr), np.inf)

    while remaining:
        best_j, best_precision, best_rho, best_phi = None, -np.inf, None, None
        for j in remaining:
            phi_j, rho_j = usable[j]
            pos_mask = np.minimum(rho_current, rho_j) > 0
            tot = pos_mask.sum()
            precision = float((y_tr[pos_mask] == target_class).sum() / tot) if tot else 0.0
            if precision > best_precision:
                best_precision, best_j, best_rho, best_phi = precision, j, rho_j, phi_j

        if best_j is None:
            break
        picks.append(best_j)
        remaining.remove(best_j)
        reparametrized[best_j] = simplify(best_phi)
        rho_current = np.minimum(rho_current, best_rho)
        if best_precision >= precision_threshold:
            break

    return picks, reparametrized


def conjunction(phis):
    if not phis:
        raise ValueError("empty conjunction")
    out = phis[0]
    for phi in phis[1:]:
        out = And(out, phi)
    return out


def build_local_explanation(x, model, formulae, X_tr, std, y_tr, m, precision_threshold):
    W, b = model.coef_, model.intercept_
    if W.shape[0] == 1:
        W = np.vstack([-W, W])
        b = np.concatenate([-b, b])

    k = int(np.argmax(W @ x + b))
    y_pred = model.classes_[k]

    picks, reparametrized = greedy_precise_picks(
        x, X_tr, std, W, model, formulae, y_tr, k, m, precision_threshold
    )
    if not picks:
        return None, y_pred, []

    local_explanation = simplify(conjunction([reparametrized[j] for j in picks]))
    return local_explanation, y_pred, picks


def evaluate_local_explanation(phi_local, target_class, TS_tr, y_tr):
    y_tr = np.asarray(y_tr)
    pos_mask = robustness(TS_tr, [phi_local]) > 0
    tot_positive = int(pos_mask.sum())
    if tot_positive == 0:
        return 0.0, 0, 0
    true_positive = int((y_tr[pos_mask] == target_class).sum())
    false_positive = tot_positive - true_positive
    precision = true_positive / tot_positive
    return precision, true_positive, false_positive


def build_local_explanations(model, formulae, X_tr, std, TS_tr, y_tr, m, precision_threshold):
    """Build a local explanation for every training instance (notebook cell ff5fd13c)."""
    locals_per_class = defaultdict(list)
    for i in range(X_tr.shape[0]):
        x = X_tr[i]
        phi_local, pred_class, picks = build_local_explanation(
            x, model, formulae, X_tr, std, y_tr, m=m, precision_threshold=precision_threshold
        )
        if phi_local is not None:
            precision, n_tp, n_fp = evaluate_local_explanation(phi_local, pred_class, TS_tr, y_tr)
        else:
            precision, n_tp, n_fp = 0.0, 0, 0

        print(
            f"sample {i:>4} | pred_class={pred_class!s:>10} | true_class={y_tr[i]!s:>10} | "
            f"precision={precision:.3f} | n_tp={n_tp:>3} | n_fp={n_fp:>3} | "
            f"picks={picks} | formula={phi_local}"
        )
        locals_per_class[pred_class].append([i, phi_local, picks, precision, n_tp, n_fp])
    return locals_per_class


def summarize_locals_per_class(locals_per_class):
    """Per-class aggregate quality of local explanations (notebook cell ff5fd13c):
    mean precision / n_tp / n_fp / formula length over instances that got one."""
    print("\n" + "=" * 60)
    print("PER-CLASS LOCAL SUMMARY")
    print("=" * 60)
    print(f"  {'class':<10} {'n':>4} {'expl':>5} {'precision':>10} "
          f"{'n_tp':>6} {'n_fp':>6} {'length':>7}")
    summary = {}
    for cls in sorted(locals_per_class, key=str):
        entries = locals_per_class[cls]
        found = [e for e in entries if e[1] is not None]
        rec = {"n_total": len(entries), "n_explained": len(found)}
        if found:
            precisions = [p for (_, _, _, p, _, _) in found]
            tps = [tp for (_, _, _, _, tp, _) in found]
            fps = [fp for (_, _, _, _, _, fp) in found]
            lengths = [len(picks) for (_, _, picks, _, _, _) in found]
            rec.update({
                "mean_precision": float(np.mean(precisions)),
                "std_precision": float(np.std(precisions)),
                "mean_n_tp": float(np.mean(tps)),
                "mean_n_fp": float(np.mean(fps)),
                "mean_length": float(np.mean(lengths)),
            })
            print(f"  {cls!s:<10} {rec['n_total']:>4} {rec['n_explained']:>5} "
                  f"{rec['mean_precision']:>10.3f} {rec['mean_n_tp']:>6.1f} "
                  f"{rec['mean_n_fp']:>6.1f} {rec['mean_length']:>7.2f}")
        else:
            print(f"  {cls!s:<10} {rec['n_total']:>4} {rec['n_explained']:>5} "
                  f"{'n/a':>10} {'n/a':>6} {'n/a':>6} {'n/a':>7}")
        summary[str(cls)] = rec
    return summary


# ---------------------------------------------------------------------------
# 3b. Global explanations (notebook cells cbf5112c, eebde2c4)
# ---------------------------------------------------------------------------

def disjunction(phis):
    if not phis:
        raise ValueError("empty disjunction")
    out = phis[0]
    for phi in phis[1:]:
        out = Or(out, phi)
    return out


def dedup_disjuncts(phi_disjuncts, masks, agreement):
    order = sorted(range(len(masks)), key=lambda i: -masks[i].sum())
    kept = []
    for i in order:
        if all((masks[i] == masks[j]).mean() < agreement for j in kept):
            kept.append(i)
    return kept


def prune_disjuncts(idx, masks, target, min_gain):
    def score_of(indices):
        u = np.logical_or.reduce([masks[i] for i in indices])
        cov = (u & target).sum() / target.sum()
        fp = (u & ~target).sum() / (~target).sum()
        return cov - fp

    idx = list(idx)
    improved = True
    while improved and len(idx) > 1:
        improved = False
        base = score_of(idx)
        for i in list(idx):
            rest = [j for j in idx if j != i]
            if score_of(rest) >= base - min_gain:
                idx.remove(i)
                improved = True
                break
    return idx


def simplify_global(phi_disjuncts, TS_tr, y_tr, target_class, agreement, min_gain):
    y = np.asarray(y_tr)
    target = y == target_class
    masks = [robustness(TS_tr, [phi]) > 0 for phi in phi_disjuncts]
    idx = dedup_disjuncts(phi_disjuncts, masks, agreement)
    idx = prune_disjuncts(idx, masks, target, min_gain)
    return [phi_disjuncts[i] for i in idx]


def build_global_explanations(locals_per_class, TS_tr, y_tr, agreement, min_gain):
    global_per_class = {}
    n_unique_per_class = {}
    for cls, lst in locals_per_class.items():
        phis = [phi for _, phi, _, _, _, _ in lst if phi is not None]
        n_unique_per_class[cls] = len({str(p) for p in phis})
        if phis:
            kept = simplify_global(phis, TS_tr, y_tr, cls, agreement, min_gain)
            global_per_class[cls] = reparametrize_formula_global(
                disjunction(kept), cls, TS_tr, y_tr
            )
    return global_per_class, n_unique_per_class


# --- data-aware structural simplification + threshold rounding (cell eebde2c4) ---

def positive_mask(phi, TS):
    return robustness(TS, [phi]) > 0


def _children_replacements(node):
    if isinstance(node, (And, Or)):
        return [node.left_child, node.right_child]
    if isinstance(node, Not):
        if isinstance(node.child, Not):
            return [node.child.child]
    return []


def _iter_nodes(node):
    for attr in ("child", "left_child", "right_child"):
        child = getattr(node, attr, None)
        if child is not None and not isinstance(child, Atom):
            yield node, attr, child
            yield from _iter_nodes(child)
        elif isinstance(child, Atom):
            yield node, attr, child


def simplify_data_aware(phi, TS, agreement):
    phi = copy.deepcopy(phi)
    ref_mask = positive_mask(phi, TS)

    def ok(candidate):
        return (positive_mask(candidate, TS) == ref_mask).mean() >= agreement

    improved = True
    while improved:
        improved = False
        for rep in _children_replacements(phi):
            if ok(rep):
                phi = rep
                improved = True
                break
        if improved:
            continue
        for parent, attr, child in _iter_nodes(phi):
            for rep in _children_replacements(child):
                old = getattr(parent, attr)
                setattr(parent, attr, rep)
                if ok(phi):
                    improved = True
                    break
                setattr(parent, attr, old)
            if improved:
                break
    return phi


def round_thresholds(phi, TS, decimals, agreement):
    phi = copy.deepcopy(phi)
    ref_mask = positive_mask(phi, TS)

    def visit(node):
        if isinstance(node, Atom):
            old = node.threshold
            node.threshold = round(float(old), decimals)
            if (positive_mask(phi, TS) == ref_mask).mean() < agreement:
                node.threshold = old
            return
        for attr in ("child", "left_child", "right_child"):
            child = getattr(node, attr, None)
            if child is not None:
                visit(child)

    visit(phi)
    return phi


# ---------------------------------------------------------------------------
# 4. Evaluation + plotting (notebook cells 6efd1720, 86c98985)
# ---------------------------------------------------------------------------

def evaluate_global(global_per_class, TS_eval, y_eval, header: str = ""):
    """Print and return per-class coverage / FP-rate / precision / F1 on TS_eval.

    F1 is the harmonic mean of precision and coverage (recall); a macro average
    over classes is added under the "macro_avg" key."""
    y_eval = np.asarray(y_eval)
    if header:
        print(header)
    print(f"{'class':<12s} {'coverage':>10s} {'FP rate':>10s} {'precision':>10s} {'F1':>8s}")
    print("-" * 60)
    out = {}
    f1s = []
    for cls, phi_global in global_per_class.items():
        rho = robustness(TS_eval, [phi_global])
        target_mask = y_eval == cls
        pos_mask = rho > 0
        coverage = float((rho[target_mask] > 0).mean()) if target_mask.any() else float("nan")
        fp_rate = float((rho[~target_mask] > 0).mean()) if (~target_mask).any() else float("nan")
        total_pos = int(pos_mask.sum())
        tp = int((pos_mask & target_mask).sum())
        precision = float(tp / total_pos) if total_pos > 0 else float("nan")
        denom = precision + coverage
        f1 = float(2 * precision * coverage / denom) if denom > 0 else float("nan")
        out[cls] = {"coverage": coverage, "fp_rate": fp_rate,
                    "precision": precision, "f1": f1}
        if not np.isnan(f1):
            f1s.append(f1)
        print(f"{cls!s:<12s} {coverage:>9.1%} {fp_rate:>9.1%} {precision:>9.1%} "
              f"{f1:>8.3f} {phi_global}")
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    out["macro_avg"] = {"f1": macro_f1}
    print(f"{'macro_avg':<12s} {'':>10s} {'':>10s} {'':>10s} {macro_f1:>8.3f}")
    return out


def plot_robustness_kde(phi, X, y, classes, target_class, title, out_path):
    rho = robustness(X, [phi])
    grid = np.linspace(rho.min() - 1, rho.max() + 1, 400)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for c in classes:
        rho_c = rho[y == c]
        if rho_c.size < 2 or np.allclose(rho_c.std(), 0):
            ax.axvline(rho_c.mean(), linestyle="--", label=f"{c} (n={rho_c.size}, degenerate)")
            continue
        kde = gaussian_kde(rho_c, 0.1)
        ax.plot(grid, kde(grid), label=f"{c} (n={rho_c.size})")
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":", label="rho = 0")
    ax.set_xlabel("rho")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def save_local_explanations(locals_per_class, path: Path) -> None:
    rows = []
    for cls, lst in locals_per_class.items():
        for sample_idx, phi_local, picks, precision, n_tp, n_fp in lst:
            rows.append({
                "sample_idx": sample_idx,
                "target_class": cls,
                "formula": str(phi_local) if phi_local is not None else "",
                "n_picks": len(picks),
                "precision_train": round(precision, 6),
                "n_tp": n_tp,
                "n_fp": n_fp,
            })
    rows.sort(key=lambda r: r["sample_idx"])
    fieldnames = ["sample_idx", "target_class", "formula", "n_picks",
                  "precision_train", "n_tp", "n_fp"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_global_explanations(global_per_class, train_eval, test_eval,
                             n_unique_per_class, path: Path) -> None:
    fieldnames = ["class", "formula", "n_unique_locals",
                  "coverage_train", "fp_train", "precision_train", "f1_train",
                  "coverage_test", "fp_test", "precision_test", "f1_test"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cls, phi_global in global_per_class.items():
            tr = train_eval.get(cls, {})
            te = test_eval.get(cls, {})
            writer.writerow({
                "class": cls,
                "formula": str(phi_global),
                "n_unique_locals": n_unique_per_class.get(cls, 0),
                "coverage_train": round(tr.get("coverage", float("nan")), 6),
                "fp_train": round(tr.get("fp_rate", float("nan")), 6),
                "precision_train": round(tr.get("precision", float("nan")), 6),
                "f1_train": round(tr.get("f1", float("nan")), 6),
                "coverage_test": round(te.get("coverage", float("nan")), 6),
                "fp_test": round(te.get("fp_rate", float("nan")), 6),
                "precision_test": round(te.get("precision", float("nan")), 6),
                "f1_test": round(te.get("f1", float("nan")), 6),
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comparison_dir", required=True, type=Path,
                   help="a run_comparison.py output directory")
    p.add_argument("--dataset", default=None,
                   help="dataset name (default: read from comparison config.json)")
    p.add_argument("--seed", type=int, default=SEED_F0)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--pool_size", type=int, default=LOCAL_EXPL_TOP_M)
    p.add_argument("--precision_threshold", type=float, default=LOCAL_EXPL_PRECISION_THRESHOLD)
    p.add_argument("--output_dir", default=None,
                   help="output directory (default: local_experiments/explanations/<dataset>_<ts>)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    n_formulae, depth_max = pick_best_stl_linear(args.comparison_dir)
    dataset = args.dataset or comparison_dataset(args.comparison_dir)
    if not dataset:
        raise SystemExit("Could not determine dataset; pass --dataset.")
    print(f"Best stl_linear config from {args.comparison_dir.name}: "
          f"n_formulae={n_formulae}  depth_max={depth_max}  dataset={dataset}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parent / "explanations" / f"{dataset}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    (model, formulae, X_tr, X_te, std, TS_tr, TS_te,
     y_tr, y_te, classes, metrics) = retrain(
        dataset, n_formulae, depth_max, args.seed, args.cv
    )

    print("\n--- Building local explanations (train set) ---")
    locals_per_class = build_local_explanations(
        model, formulae, X_tr, std, TS_tr, y_tr,
        m=args.pool_size, precision_threshold=args.precision_threshold,
    )

    local_summary = summarize_locals_per_class(locals_per_class)

    print("\n--- Building global explanations ---")
    global_per_class, n_unique_per_class = build_global_explanations(
        locals_per_class, TS_tr, y_tr, agreement=1.00, min_gain=0.5
    )

    # Data-aware simplification + threshold rounding (notebook cell c18746bf).
    for cls in global_per_class:
        phi = global_per_class[cls]
        phi = simplify_data_aware(phi, TS_tr, agreement=0.98)
        phi = round_thresholds(phi, TS_tr, decimals=1, agreement=0.98)
        global_per_class[cls] = phi

    print()
    train_eval = evaluate_global(global_per_class, TS_tr, y_tr, header="Evaluation on train set:")
    print()
    test_eval = evaluate_global(global_per_class, TS_te, y_te, header="Evaluation on test set:")

    # --- persist ---
    config_out = {
        "dataset": dataset,
        "n_formulae": n_formulae,
        "depth_max": depth_max,
        "seed": args.seed,
        "cv": args.cv,
        "pool_size": args.pool_size,
        "precision_threshold": args.precision_threshold,
        "comparison_dir": str(args.comparison_dir),
        **metrics,
    }
    (out_dir / "config.json").write_text(json.dumps(config_out, indent=2))
    save_local_explanations(locals_per_class, out_dir / "local_explanations.csv")
    save_global_explanations(global_per_class, train_eval, test_eval,
                             n_unique_per_class, out_dir / "global_explanations.csv")

    metrics_out = {
        "local_per_class": local_summary,
        "global_per_class": {
            str(cls): {
                "train": {k: train_eval[cls][k] for k in train_eval[cls]},
                "test": {k: test_eval[cls][k] for k in test_eval[cls]},
            }
            for cls in global_per_class
        },
        "global_macro_f1_train": train_eval["macro_avg"]["f1"],
        "global_macro_f1_test": test_eval["macro_avg"]["f1"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    for cls, phi_global in global_per_class.items():
        plot_robustness_kde(phi_global, TS_tr, y_tr, classes, cls,
                            title=f"phi_{cls}_global on train set",
                            out_path=out_dir / f"kde_{cls}_train.png")
        plot_robustness_kde(phi_global, TS_te, y_te, classes, cls,
                            title=f"phi_{cls}_global on test set",
                            out_path=out_dir / f"kde_{cls}_test.png")

    print(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
