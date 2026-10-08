"""Instance rules and global explanations (paper §3.3).

An instance rule L(tau) for the prediction k = f(tau) is a conjunction of literals, one
per reparametrized formula phi_j. Literal (j, e, s), with direction e = sgn(x_j(tau))
and cut s in standardized units, holds on tau' iff e * x_j(tau') > s, i.e. iff
rho(phi_j, tau') lies beyond c = mu_j + e * s * (sigma_j + 1e-8) on the same side as
tau. By the robustness shift property this is the STL formula phi_j^{-c} (e > 0) or
Not(phi_j)^{-c} (e < 0), so the whole search runs on the standardized train feature
matrix and STL is only evaluated to build the final formula (to_stl).

Rules are consistent: every literal holds on tau (0 <= s < e * x_j(tau)). Counts are
taken on the train set w.r.t. reference labels, the model's predictions: the support of
a rule is the number of train series satisfying it, TP the number of those predicted as
k. Among the consistent rules with support >= n_min, the search maximises the Wilson
lower confidence bound of the precision TP / support (wilson_lower), which favours rules
that are both precise and supported by many series.

The STL formula of a rule (rule_formula) is simplified without changing the train series
that satisfy it (simplify_rule). A global explanation of class k (global_explanations) is
an OR of few, general instance rules of the train series predicted as k
(select_disjuncts), each simplified the same way.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import reduce
from statistics import NormalDist

import numpy as np
from torcheck.stl import And, Not

from .features import shift_atom_thresholds
from .simplification import simplify_data_aware, round_thresholds


@dataclass
class RuleConfig:
    """Hyperparameters of instance rules and global explanations."""
    n_min: int = 5                       # min support (train series satisfying a rule)
    n_max: int = 3                       # max literals per rule
    beam_width: int = 5                  # rules kept per beam search step
    confidence: float = 0.9              # confidence level of the Wilson lower bound
    min_new_share: float | None = 0.05   # global: a rule enters the OR only if it covers this share
                                         # of its class in new train series (None: every distinct rule)
    decimals: int = 2                    # global: thresholds rounded to this many decimals


def model_logits_matrix(model) -> np.ndarray:
    """(K, M) logit weights of a glmnet LogitNet without intercept. Binomial models
    store one row w with logit(class 1) = w x, i.e. logits (0, w x)."""
    assert np.all(np.asarray(model.intercept_) == 0), "instance rules assume fit_intercept=False"
    W = np.atleast_2d(model.coef_)
    return np.vstack([np.zeros_like(W), W]) if W.shape[0] == 1 else W


def predict_indices(X_feats: np.ndarray, W_full: np.ndarray) -> np.ndarray:
    """Predicted class indices (rows of W_full, i.e. positions in model.classes_)."""
    return np.argmax(X_feats @ W_full.T, axis=1)


def candidate_scores(x: np.ndarray, W_full: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Predicted class k, scores s_j = sum_l pi_l (w_kj - w_lj) x_j with Gradient x
    Input weights pi_l = P(l|x) / (1 - P(k|x)) (eq. 18-19), and candidates J = {s_j > 0}
    ranked by decreasing s_j (eq. 21). Unselected formulae (w_.j = 0) and columns
    constant on train (x_j = 0) have s_j = 0, so they are never candidates."""
    z = W_full @ x
    k = int(np.argmax(z))
    P = np.exp(z - z.max())
    P /= P.sum()
    others = np.arange(len(z)) != k
    pi = P[others] / max(1.0 - P[k], 1e-300)
    if pi.sum() == 0:   # P(k|x) == 1 in floating point: fall back to uniform weights
        pi = np.full(others.sum(), 1.0 / others.sum())
    A = (W_full[k] - W_full[others]) * x       # (K-1, M) margin attributions, eq. 17
    s = pi @ A
    J = np.flatnonzero(s > 0)
    return k, s, J[np.argsort(-s[J], kind="stable")]


def cuts(u: np.ndarray, u_tau: float) -> np.ndarray:
    """Cut grid of one literal, ascending. u: e * x_j over the (kept) train series,
    u_tau = e * x_j(tau) > 0. A cut s must satisfy 0 <= s < u_tau (consistency, and
    covered series on the same side of the train mean as tau). The covered set {u > s}
    only changes when s crosses a train value, so each distinct set is represented by
    the midpoint between consecutive distinct train values, plus s = 0. Midpoints also
    keep every train value away from the cut, so feature-space and STL evaluation agree."""
    v = np.unique(u)
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[0.0], mid[(mid > 0) & (mid < u_tau)]])


def covered(literals: list, X_feats: np.ndarray) -> np.ndarray:
    """Mask of rows of X_feats satisfying every literal (j, e, s)."""
    mask = np.ones(len(X_feats), dtype=bool)
    for j, e, s in literals:
        mask &= e * X_feats[:, j] > s
    return mask


def wilson_lower(tp, n, confidence: float = 0.9):
    """One-sided Wilson lower confidence bound of the precision tp / n (0 where n == 0):
    the smallest precision still plausible after tp hits in n series. It is close to
    tp / n for large n and far below it for small n (5/5: 0.75, 28/30: 0.85 at 0.9).
    Works elementwise on arrays."""
    tp, n = np.asarray(tp, dtype=float), np.asarray(n, dtype=float)
    z = NormalDist().inv_cdf(confidence)
    z2 = z * z
    with np.errstate(divide="ignore", invalid="ignore"):
        p = tp / n
        lb = (p + z2 / (2 * n) - z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)
    return np.where(n > 0, lb, 0.0)


def _cut_counts(values: np.ndarray, is_pos: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Support and true-positive counts of {values > s} for every s in grid."""
    all_sorted, pos_sorted = np.sort(values), np.sort(values[is_pos])
    n_cov = len(all_sorted) - np.searchsorted(all_sorted, grid, side="right")
    n_tp = len(pos_sorted) - np.searchsorted(pos_sorted, grid, side="right")
    return n_cov, n_tp


def explain_instance(
    x: np.ndarray,
    W_full: np.ndarray,
    X_tr_feats: np.ndarray,
    ref_labels: np.ndarray,
    cfg: RuleConfig,
    exclude: int | None = None,
) -> dict:
    """Instance rule for the prediction on features x (eq. 15), by beam search (§3.3.3).

    ref_labels: train class indices counts are taken against, the model's predictions
    f(tau^(i)). exclude: train index of tau when tau is a train series (left out of every
    count).

    Returns dict(k, literals [(j, e, s)], tp, support, precision, lcb), lcb the Wilson
    bound of the rule's train precision; literals is empty if no rule reaches support n_min.
    """
    k, score, J = candidate_scores(x, W_full)
    keep = np.ones(len(X_tr_feats), dtype=bool)
    if exclude is not None:
        keep[exclude] = False
    pos = (ref_labels == k) & keep
    empty = dict(k=k, literals=[], tp=0, support=0, precision=0.0, lcb=0.0)
    if len(J) == 0:   # cannot happen: sum_j s_j = sum_l pi_l m_l > 0
        return empty

    eps = np.sign(x[J])
    U = X_tr_feats[:, J] * eps                    # (N, |J|): literal p holds iff U[:, p] > s
    u_tau = x[J] * eps
    grids = [cuts(U[keep, p], u_tau[p]) for p in range(len(J))]

    def summary(lits, mask):
        return dict(literals=lits, mask=mask)

    def rank(c):    # larger bound, then shorter, then more TP
        return (c["lcb"], -len(c["literals"]), c["tp"])

    beam = [summary((), keep.copy())]              # B_0 = {T}
    best = None
    for _ in range(cfg.n_max):
        ext = {}                                   # literal set -> (sort key, literals, support, tp)
        for state in beam:
            used = {p for p, _ in state["literals"]}
            mask = state["mask"]
            for p in range(len(J)):
                if p in used:
                    continue
                grid = grids[p]
                n_cov, n_tp = _cut_counts(U[mask, p], pos[mask], grid)
                # Cuts giving the same covered set (same count, since sets are nested):
                # keep the middle one, i.e. the widest margin among the covered values.
                _, first, size = np.unique(-n_cov, return_index=True, return_counts=True)
                for i in first + size // 2:
                    n = int(n_cov[i])
                    if n < cfg.n_min:
                        continue
                    tp = int(n_tp[i])
                    lits = state["literals"] + ((p, float(grid[i])),)
                    key = frozenset(lits)
                    # By the bound, then TP, then the attribution score of the new formula.
                    sort_key = (float(wilson_lower(tp, n, cfg.confidence)), tp, score[J[p]])
                    if key not in ext or sort_key > ext[key][0]:
                        ext[key] = (sort_key, lits, n, tp)
        if not ext:
            break
        ranked = sorted(ext.values(), key=lambda e: e[0], reverse=True)
        beam = [summary(lits, covered([(p, 1.0, s) for p, s in lits], U) & keep)
                for _, lits, _, _ in ranked[:cfg.beam_width]]
        # Track the best rule over all extensions of all steps
        for sort_key, lits, n, tp in ranked:
            cand = dict(literals=lits, tp=tp, lcb=sort_key[0])
            if best is None or rank(cand) > rank(best):
                best = cand

    if best is None:
        return empty
    chosen = _generalize(list(best["literals"]), U, grids, keep, pos, cfg)
    literals = [(int(J[p]), float(eps[p]), float(s)) for p, s in chosen["literals"]]
    return dict(k=k, literals=literals, tp=chosen["tp"], support=chosen["support"],
                precision=chosen["precision"],
                lcb=float(wilson_lower(chosen["tp"], chosen["support"], cfg.confidence)))


def _generalize(lits: list, U, grids, keep, pos, cfg: RuleConfig) -> dict:
    """Greedy generalization pass: repeatedly apply the move (drop a literal, or loosen
    its cut to any looser cut toward the train mean) that keeps support >= n_min and
    gives the largest Wilson bound, on equal bound the shorter rule. Stops when no move
    raises the bound or shortens the rule."""
    def value(tp, n, n_lits):
        return (float(wilson_lower(tp, n, cfg.confidence)), -n_lits)

    def masks_without(i):
        m = keep.copy()
        for q, (p, s) in enumerate(lits):
            if q != i:
                m &= U[:, p] > s
        return m

    full = masks_without(-1)
    cur = value(int((full & pos).sum()), int(full.sum()), len(lits))
    while True:
        best_move, best_val = None, cur
        for i, (p, s) in enumerate(lits):
            others = masks_without(i)
            if len(lits) > 1:      # drop literal i
                n, tp = int(others.sum()), int((others & pos).sum())
                val = value(tp, n, len(lits) - 1)
                if n >= cfg.n_min and val > best_val:
                    best_move, best_val = (i, None), val
            looser = grids[p][grids[p] < s]
            if len(looser):        # loosen literal i
                n_cov, n_tp = _cut_counts(U[others, p], pos[others], looser)
                for c, n, tp in zip(looser, n_cov, n_tp):
                    val = value(int(tp), int(n), len(lits))
                    if n >= cfg.n_min and val > best_val:
                        best_move, best_val = (i, float(c)), val
        if best_move is None or best_val <= cur:
            break
        i, c = best_move
        lits = lits[:i] + lits[i + 1:] if c is None else lits[:i] + [(lits[i][0], c)] + lits[i + 1:]
        cur = best_val
    full = masks_without(-1)
    n, tp = int(full.sum()), int((full & pos).sum())
    return dict(literals=tuple(lits), tp=tp, support=n, precision=tp / n if n else 0.0)


def explain_instances(X_feats: np.ndarray, W_full: np.ndarray, X_tr_feats: np.ndarray,
                      ref_labels: np.ndarray, cfg: RuleConfig, train: bool = False) -> list[dict]:
    """Instance rules for every row of X_feats. train=True: the rows are the train
    series themselves, each left out of its own counts."""
    return [explain_instance(x, W_full, X_tr_feats, ref_labels, cfg, exclude=i if train else None)
            for i, x in enumerate(X_feats)]


def rule_metrics(satisfied: np.ndarray, k: int, labels: np.ndarray, exclude: int | None = None) -> dict:
    """TP, support, precision and coverage of a rule (or global explanation) for class k
    on any split, given the mask of the series satisfying it and their class indices.
    coverage = TP / number of series of class k. exclude: row of the explained series
    itself, left out."""
    mask, pos = satisfied.copy(), labels == k
    if exclude is not None:
        mask[exclude] = pos[exclude] = False
    n, tp = int(mask.sum()), int((mask & pos).sum())
    return dict(tp=tp, support=n, precision=tp / n if n else np.nan,
                coverage=tp / pos.sum() if pos.any() else np.nan)


def to_stl(literals: list, formulas: list, mu: np.ndarray, sigma: np.ndarray):
    """STL formula of a rule (eq. 22): literal (j, e, s) is phi_j^{-c} (e > 0) or
    Not(phi_j)^{-c} (e < 0) with c = e * mu_j + s * (sigma_j + 1e-8), the cut on
    e * rho(phi_j, .). mu, sigma: the raw train statistics from build_formula_bank."""
    if not literals:
        return None
    parts = []
    for j, e, s in literals:
        phi = copy.deepcopy(formulas[j])
        if e < 0:
            phi = Not(phi)
        shift_atom_thresholds(phi, -(e * mu[j] + s * (sigma[j] + 1e-8)))
        parts.append(phi)
    return reduce(And, parts)


def simplify_rule(phi, X_tr: np.ndarray, cfg: RuleConfig):
    """Data-aware simplification of a rule's STL formula, exact on the train series X_tr:
    an and/or replaced by a child, double negation removed, thresholds rounded to
    cfg.decimals, each only if the train series satisfying the formula stay exactly the
    same. A branch can be dead for one rule but not for another, because the cut decides
    which branch matters. On new series the simplified formula can differ slightly."""
    phi = simplify_data_aware(phi, X_tr, agreement=1.0)
    return round_thresholds(phi, X_tr, decimals=cfg.decimals, agreement=1.0)


def rule_formula(literals: list, formulas: list, mu: np.ndarray, sigma: np.ndarray, X_tr: np.ndarray,
                 cfg: RuleConfig, simplify: bool = True):
    """STL formula of an instance rule (to_stl), simplified (simplify_rule) unless simplify
    is False. None for the empty rule."""
    phi = to_stl(literals, formulas, mu, sigma)
    return simplify_rule(phi, X_tr, cfg) if simplify and phi is not None else phi


def global_rules(train_rules: list[dict], X_tr_feats: np.ndarray, ref_labels: np.ndarray,
                 cfg: RuleConfig) -> dict[int, list]:
    """Rules of the global explanation per class (eq. 24), in feature space: the distinct
    instance rules of the train series predicted as that class, then few, general ones of
    them (select_disjuncts). Returns {k: [literals, ...]}.

    Rules are distinct by the set of train series they cover, not by their literals: each
    train series' cut grid leaves that series out, so two series can give slightly
    different cuts for the same covered set. Of such rules the shorter one is kept, then
    the one with more TP. A rule is selected only if it covers at least
    ceil(cfg.min_new_share * class size) new train series of its class (at least 1); the
    class is w.r.t. ref_labels, the train class indices of the instance rules."""
    out: dict[int, dict] = {}
    for r in train_rules:
        if r["literals"]:
            key = covered(r["literals"], X_tr_feats).tobytes()
            cur = out.setdefault(r["k"], {}).get(key)
            if cur is None or (len(r["literals"]), -r["tp"]) < (len(cur["literals"]), -cur["tp"]):
                out[r["k"]][key] = r
    rules = {k: [r["literals"] for r in v.values()] for k, v in out.items()}
    if cfg.min_new_share is None:
        return rules
    return {k: select_disjuncts(R, X_tr_feats, ref_labels == k,
                                max(1, int(np.ceil(cfg.min_new_share * (ref_labels == k).sum()))))
            for k, R in rules.items()}


def select_disjuncts(rules: list, X_tr_feats: np.ndarray, pos: np.ndarray, min_new: int) -> list:
    """Greedy cover of the train series of the class (pos) by few, general rules: add the
    rule satisfied by the most positives not yet covered (ties: fewer false positives,
    then shorter), and stop when the best one adds fewer than min_new new positives.
    Every disjunct can bring false positives on new series, while the true positives of
    the rules mostly overlap, so a rule covering almost nothing new is not worth it."""
    masks = [covered(lits, X_tr_feats) for lits in rules]
    done = np.zeros(len(X_tr_feats), dtype=bool)
    left, chosen = list(range(len(rules))), []
    while left:
        def gain(i):
            return (int((masks[i] & pos & ~done).sum()), -int((masks[i] & ~pos).sum()), -len(rules[i]))
        best = max(left, key=gain)
        if gain(best)[0] < max(min_new, 1):
            break
        chosen.append(best)
        left.remove(best)
        done |= masks[best]
    return [rules[i] for i in chosen]


def global_explanations(train_rules: list[dict], X_tr_feats: np.ndarray, X_tr: np.ndarray,
                        ref_labels: np.ndarray, formulas: list, mu: np.ndarray, sigma: np.ndarray,
                        cfg: RuleConfig, simplify: bool = True) -> dict[int, list]:
    """Global explanation per class: {k: [STL formula, ...]}, the OR of the listed rules.

    Rules come from global_rules, each simplified (simplify_rule) unless simplify is
    False; rules that become identical are kept once."""
    out = {}
    for k, R in global_rules(train_rules, X_tr_feats, ref_labels, cfg).items():
        phis = [rule_formula(lits, formulas, mu, sigma, X_tr, cfg, simplify) for lits in R]
        out[k] = list({str(phi): phi for phi in phis}.values())
    return out
