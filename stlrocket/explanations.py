from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
from torcheck.stl import Not, And, Or
from torcheck import simplify

from .features import eval_robustness, shift_atom_thresholds


def reparametrize_formula(phi_original, X: np.ndarray, y: np.ndarray, target_class):
    y = np.asarray(y)
    phi = copy.deepcopy(phi_original)
    rho = eval_robustness(phi, X)
    target_mask = y == target_class
    med_target = np.median(rho[target_mask])
    other_classes = np.unique(y[~target_mask])
    med_others = {c: np.median(rho[y == c]) for c in other_classes}
    below = {c: m for c, m in med_others.items() if m < med_target}
    above = {c: m for c, m in med_others.items() if m > med_target}

    if len(below) == 0:
        ref_class = min(above, key=lambda c: above[c])
    elif len(above) == 0:
        ref_class = max(below, key=lambda c: below[c])
    elif len(above) == len(below):
        ref_class = max(med_others, key=lambda c: abs(med_others[c] - med_target))
    elif len(below) > len(above):
        ref_class = max(below, key=lambda c: below[c])
    else:
        ref_class = min(above, key=lambda c: above[c])

    med_ref = med_others[ref_class]
    negated = med_target < med_ref
    if negated:
        phi = Not(phi)
        med_target = -med_target
        med_ref = -med_ref

    delta_star = -(med_target + med_ref) / 2
    phi_new = copy.deepcopy(phi)
    shift_atom_thresholds(phi_new, delta_star)
    return phi_new


def per_competitor_contributions(
    W: np.ndarray, x: np.ndarray, class_idx: int, others: list
) -> np.ndarray:
    if W.shape[0] == 1:
        return np.stack([W[0] * x])
    return np.stack([(W[class_idx] - W[k]) * x for k in others])


def get_top_m_features(contrib_matrix: np.ndarray, m: int) -> tuple[list, np.ndarray]:
    agg = contrib_matrix.sum(axis=0)
    order = np.argsort(-agg)
    return order[:m].tolist(), agg


def greedy_precise_picks(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    model,
    formulas: list,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    class_idx: int,
    pool_size: int = 10,
    precision_threshold: float = 0.9,
) -> tuple[list, dict]:
    target_class = model.classes_[class_idx]
    others = [k for k in range(len(model.classes_)) if k != class_idx]

    contrib = per_competitor_contributions(W, x, class_idx, others)
    pool_indices, _ = get_top_m_features(contrib, pool_size)

    remaining = list(pool_indices)
    picks = []
    reparametrized = {}
    rho_current = np.full(len(y_tr), np.inf)

    for _ in range(len(remaining)):
        best_j, best_precision, best_rho, best_phi = None, -np.inf, None, None

        for j in remaining:
            phi_j = reparametrize_formula(formulas[j], X_tr, y_tr, target_class)
            rho_j = eval_robustness(phi_j, X_tr)
            rho_combined = np.minimum(rho_current, rho_j)

            pos_mask = rho_combined > 0
            tot = pos_mask.sum()
            precision = float((y_tr[pos_mask] == target_class).sum() / tot) if tot > 0 else 0.0

            if precision > best_precision:
                best_precision, best_j, best_rho, best_phi = precision, j, rho_j, phi_j

        picks.append(best_j)
        remaining.remove(best_j)
        reparametrized[best_j] = best_phi
        rho_current = np.minimum(rho_current, best_rho)

        if best_precision >= precision_threshold:
            break

    return picks, reparametrized


def conjunction(phis: list):
    if not phis:
        raise ValueError("empty conjunction")
    out = phis[0]
    for phi in phis[1:]:
        out = And(out, phi)
    return simplify(out)


def disjunction(phis: list):
    if not phis:
        raise ValueError("empty disjunction")
    out = phis[0]
    for phi in phis[1:]:
        out = Or(out, phi)
    return out


def build_local_explanation(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    model,
    formulas: list,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    pool_size: int = 20,
    precision_threshold: float = 0.9,
) -> tuple[object, str, list]:
    if W.shape[0] == 1:
        W = np.vstack([-W, W])
        b = np.array([-b[0], b[0]])
    scores = W @ x + b
    class_idx = int(np.argmax(scores))
    target_class = model.classes_[class_idx]

    picks, reparametrized = greedy_precise_picks(
        x, W, b, model, formulas, X_tr, y_tr,
        class_idx, pool_size=pool_size, precision_threshold=precision_threshold,
    )

    phis = [reparametrized[j] for j in picks]
    return conjunction(phis), target_class, picks


def evaluate_local_explanation(
    phi_local,
    target_class: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
) -> tuple[float, int]:
    """Returns (precision_train, n_tp)."""
    rhos = eval_robustness(phi_local, X_tr)
    pos_mask = rhos > 0
    tot_positive = int(pos_mask.sum())
    if tot_positive == 0:
        return 0.0, 0
    true_positive = int((y_tr[pos_mask] == target_class).sum())
    precision = float(true_positive / tot_positive)
    return precision, true_positive


def build_global_explanations(
    X_te_feats: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    model,
    formulas: list,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    pool_size: int = 20,
    precision_threshold: float = 0.9,
) -> tuple[dict, dict, dict]:
    """Build global explanations from per-sample local explanations.

    locals_per_class maps class -> list of
        (sample_idx, phi_local, picks, precision, support)
    where support = n_pos / len(y_tr).
    """
    locals_per_class: dict = defaultdict(list)
    n_test = X_te_feats.shape[0]

    for i in range(n_test):
        phi_local, target_class, picks = build_local_explanation(
            X_te_feats[i], W, b, model, formulas,
            X_tr, y_tr, pool_size=pool_size, precision_threshold=precision_threshold,
        )
        precision, n_tp = evaluate_local_explanation(phi_local, target_class, X_tr, y_tr)
        print(
            f"  sample {i:3d} | class {target_class} | {len(picks)} formula(s) "
            f"| precision {precision:.2f} | n_tp {n_tp}"
        )
        locals_per_class[target_class].append((i, phi_local, picks, precision, n_tp))

    global_per_class = {}
    n_unique_per_class = {}
    for cls, lst in locals_per_class.items():
        seen: dict = {}
        for _, phi, _, _, _ in lst:
            key = str(phi)
            if key not in seen:
                seen[key] = phi
        unique_phis = list(seen.values())
        n_unique_per_class[cls] = len(unique_phis)
        global_per_class[cls] = disjunction(unique_phis)

    return global_per_class, locals_per_class, n_unique_per_class
