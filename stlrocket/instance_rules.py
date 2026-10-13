"""Instance rules and global explanations (paper §3.3).

An instance rule L(tau) for the prediction k = f(tau) is a conjunction of literals, one
per reparametrized formula phi_j of the bank (already centred on the train series, see
features.build_formula_bank). Literal (j, e, s), with x_j = rho(phi_j, .) / sigma_j,
direction e = sgn(x_j(tau)) and cut s >= 0, holds on tau' iff e * x_j(tau') > s, i.e. iff
rho(phi_j, tau') lies beyond c = e * s * sigma_j on the same side of 0 as tau. By the
robustness shift property this is the STL formula phi_j^{-|c|} (e > 0) or
Not(phi_j)^{-|c|} (e < 0), so the whole search runs on the scaled train feature matrix
and STL is only evaluated to build the final formula (to_stl).

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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, reduce
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
    candidates: str = "gxi"              # candidate formulae: one of CANDIDATE_METHODS (candidate_scores)
    generalize: bool = True              # greedy generalization pass after the beam search


def model_logits_matrix(model) -> np.ndarray:
    """(K, M) logit weights of a glmnet LogitNet without intercept. Binomial models
    store one row w with logit(class 1) = w x, i.e. logits (0, w x)."""
    assert np.all(np.asarray(model.intercept_) == 0), "instance rules assume fit_intercept=False"
    W = np.atleast_2d(model.coef_)
    return np.vstack([np.zeros_like(W), W]) if W.shape[0] == 1 else W


def predict_indices(X_feats: np.ndarray, W_full: np.ndarray) -> np.ndarray:
    """Predicted class indices (rows of W_full, i.e. positions in model.classes_)."""
    return np.argmax(X_feats @ W_full.T, axis=1)


CANDIDATE_METHODS = ("gxi", "ig", "union", "all")
IG_STEPS = 64   # midpoint quadrature of the Integrated Gradients path integral


def candidate_scores(x: np.ndarray, W_full: np.ndarray, method: str = "gxi") -> tuple[int, np.ndarray, np.ndarray]:
    """Predicted class k, scores s_j and candidates J ranked by decreasing s_j, from the
    margin attributions A_lj = (w_kj - w_lj) x_j (eq. 17). Unselected formulae (w_.j = 0)
    and columns constant on train (x_j = 0) have A_.j = 0 and are never candidates.

    gxi    s_j = sum_l pi_l A_lj with Gradient x Input weights pi_l = P(l|x) / (1 - P(k|x)),
           J = {s_j > 0}
    ig     the same with Integrated Gradients weights: P(l | beta x) averaged over
           beta in [0, 1] (reference 0, where every class has probability 1/K), J = {s_j > 0}
    union  per-competitor candidates J = {j : A_lj > 0 for some l}, s_j = max_l A_lj
    all    every formula with some A_lj != 0, s_j = max_l |A_lj|: no attribution"""
    z = W_full @ x
    k = int(np.argmax(z))
    others = np.arange(len(z)) != k
    A = (W_full[k] - W_full[others]) * x       # (K-1, M) margin attributions, eq. 17
    if method == "gxi":
        P = np.exp(z - z.max())
        P /= P.sum()
        pi = P[others] / max(1.0 - P[k], 1e-300)
        if pi.sum() == 0:   # P(k|x) == 1 in floating point: fall back to uniform weights
            pi = np.full(others.sum(), 1.0 / others.sum())
        s = pi @ A
    elif method == "ig":
        beta = (np.arange(IG_STEPS) + 0.5) / IG_STEPS
        Z = np.outer(beta, z)                  # logits along the path from 0 to x
        P = np.exp(Z - Z.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        s = P.mean(axis=0)[others] @ A
    elif method == "union":
        s = A.max(axis=0)
    elif method == "all":
        s = np.abs(A).max(axis=0)
    else:
        raise ValueError(f"unknown candidate method {method!r}: expected one of {CANDIDATE_METHODS}")
    J = np.flatnonzero(s > 0)
    return k, s, J[np.argsort(-s[J], kind="stable")]


def cuts(u: np.ndarray, u_tau: float) -> np.ndarray:
    """Cut grid of one literal, ascending. u: e * x_j over the (kept) train series,
    u_tau = e * x_j(tau) > 0. A cut s must satisfy 0 <= s < u_tau (consistency, and
    covered series on the same side of 0 as tau). The covered set {u > s}
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


@lru_cache(maxsize=None)
def _z(confidence: float) -> float:
    return NormalDist().inv_cdf(confidence)


def wilson_lower(tp, n, confidence: float = 0.9):
    """One-sided Wilson lower confidence bound of the precision tp / n (0 where n == 0):
    the smallest precision still plausible after tp hits in n series. It is close to
    tp / n for large n and far below it for small n (5/5: 0.75, 28/30: 0.85 at 0.9).
    Works elementwise on arrays."""
    tp, n = np.asarray(tp, dtype=float), np.asarray(n, dtype=float)
    z = _z(confidence)
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

    Returns dict(k, literals [(j, e, s)], tp, support, precision, lcb, n_candidates), lcb
    the Wilson bound of the rule's train precision; literals is empty if no rule reaches
    support n_min. Candidates come from candidate_scores(method=cfg.candidates), and the
    beam's best rule is refined by _generalize unless cfg.generalize is False.
    """
    k, score, J = candidate_scores(x, W_full, cfg.candidates)
    keep = np.ones(len(X_tr_feats), dtype=bool)
    if exclude is not None:
        keep[exclude] = False
    pos = (ref_labels == k) & keep
    empty = dict(k=k, literals=[], tp=0, support=0, precision=0.0, lcb=0.0, n_candidates=len(J))
    if len(J) == 0:   # cannot happen: some A_lj > 0 for every l, since every margin is positive
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
            pos_mask = pos[mask]
            U_mask = np.ascontiguousarray(U[mask].T)  # (|J|, n covered): one row per formula
            for p in range(len(J)):
                if p in used:
                    continue
                grid = grids[p]
                n_cov, n_tp = _cut_counts(U_mask[p], pos_mask, grid)
                # Cuts giving the same covered set (same count, since sets are nested):
                # keep the middle one, i.e. the widest margin among the covered values.
                _, first, size = np.unique(-n_cov, return_index=True, return_counts=True)
                idx = first + size // 2
                idx = idx[n_cov[idx] >= cfg.n_min]
                if len(idx) == 0:
                    continue
                n, tp = n_cov[idx], n_tp[idx]
                lcb = wilson_lower(tp, n, cfg.confidence)
                # These extensions share the formula, hence the score: only the beam_width
                # best by (bound, TP), earlier on ties, can enter the beam or be the best rule.
                # (Any other one is beaten by beam_width distinct rules of this same set.)
                if len(idx) > cfg.beam_width:
                    top = np.sort(np.lexsort((np.arange(len(idx)), -tp, -lcb))[:cfg.beam_width])
                    idx, n, tp, lcb = idx[top], n[top], tp[top], lcb[top]
                for i, n_i, tp_i, lcb_i in zip(idx.tolist(), n.tolist(), tp.tolist(), lcb.tolist()):
                    lits = state["literals"] + ((p, float(grid[i])),)
                    key = frozenset(lits)
                    # By the bound, then TP, then the attribution score of the new formula.
                    sort_key = (lcb_i, tp_i, score[J[p]])
                    if key not in ext or sort_key > ext[key][0]:
                        ext[key] = (sort_key, lits, n_i, tp_i)
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
    if cfg.generalize:
        chosen = _generalize(list(best["literals"]), U, grids, keep, pos, cfg)
    else:
        mask = covered([(p, 1.0, s) for p, s in best["literals"]], U) & keep
        n, tp = int(mask.sum()), int((mask & pos).sum())
        chosen = dict(literals=best["literals"], tp=tp, support=n, precision=tp / n if n else 0.0)
    literals = [(int(J[p]), float(eps[p]), float(s)) for p, s in chosen["literals"]]
    return dict(k=k, literals=literals, tp=chosen["tp"], support=chosen["support"],
                precision=chosen["precision"],
                lcb=float(wilson_lower(chosen["tp"], chosen["support"], cfg.confidence)),
                n_candidates=len(J))


def _generalize(lits: list, U, grids, keep, pos, cfg: RuleConfig) -> dict:
    """Greedy generalization pass: repeatedly apply the move (drop a literal, or loosen
    its cut to any looser cut toward 0) that keeps support >= n_min and
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
            if len(looser):        # loosen literal i: the first cut with the largest bound
                n_cov, n_tp = _cut_counts(U[others, p], pos[others], looser)
                lcb = np.where(n_cov >= cfg.n_min, wilson_lower(n_tp, n_cov, cfg.confidence), -np.inf)
                c = int(np.argmax(lcb))
                val = (float(lcb[c]), -len(lits))
                if lcb[c] > -np.inf and val > best_val:
                    best_move, best_val = (i, float(looser[c])), val
        if best_move is None or best_val <= cur:
            break
        i, c = best_move
        lits = lits[:i] + lits[i + 1:] if c is None else lits[:i] + [(lits[i][0], c)] + lits[i + 1:]
        cur = best_val
    full = masks_without(-1)
    n, tp = int(full.sum()), int((full & pos).sum())
    return dict(literals=tuple(lits), tp=tp, support=n, precision=tp / n if n else 0.0)


def explain_instances(X_feats: np.ndarray, W_full: np.ndarray, X_tr_feats: np.ndarray,
                      ref_labels: np.ndarray, cfg: RuleConfig, train: bool = False,
                      n_jobs: int = 1, exclude: list | None = None) -> list[dict]:
    """Instance rules for every row of X_feats. train=True: the rows are the train
    series themselves, each left out of its own counts. exclude: for rows that are a
    subset of the train series, the train index of each row, left out likewise. n_jobs > 1: rows split over that
    many worker processes (same rules; workers run the functions as defined in this
    module, so a candidate_scores patched at runtime is not seen by them). Starting the
    workers takes ~15 s (each imports torch), so small problems stay in this process."""
    excl = exclude if exclude is not None else list(range(len(X_feats))) if train else [None] * len(X_feats)
    if n_jobs <= 1 or len(X_feats) * len(X_tr_feats) < MIN_PARALLEL_WORK:
        return [explain_instance(x, W_full, X_tr_feats, ref_labels, cfg, exclude=e)
                for x, e in zip(X_feats, excl)]
    chunks = [c for c in np.array_split(np.arange(len(X_feats)), 4 * n_jobs) if len(c)]
    ctx = multiprocessing.get_context("spawn")   # no fork of a process holding torch threads
    with ProcessPoolExecutor(n_jobs, mp_context=ctx, initializer=_init_worker,
                             initargs=(W_full, X_tr_feats, ref_labels, cfg)) as ex:
        parts = ex.map(_explain_chunk, [(X_feats[c], [excl[i] for i in c]) for c in chunks])
        return [r for part in parts for r in part]


# Rows x train series below which explain_instances ignores n_jobs: ~1 min of serial work
MIN_PARALLEL_WORK = 200_000
_WORKER: dict = {}


def _init_worker(W_full, X_tr_feats, ref_labels, cfg) -> None:
    _WORKER.update(W_full=W_full, X_tr_feats=X_tr_feats, ref_labels=ref_labels, cfg=cfg)


def _explain_chunk(args) -> list[dict]:
    X, excl = args
    w = _WORKER
    return [explain_instance(x, w["W_full"], w["X_tr_feats"], w["ref_labels"], w["cfg"], exclude=e)
            for x, e in zip(X, excl)]


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


def to_stl(literals: list, formulas: list, sigma: np.ndarray):
    """STL formula of a rule (eq. 22): literal (j, e, s) is phi_j^{-c} (e > 0) or
    Not(phi_j)^{-c} (e < 0) with c = s * sigma_j, the cut on e * rho(phi_j, .).
    sigma: the feature scales from build_formula_bank."""
    if not literals:
        return None
    parts = []
    for j, e, s in literals:
        phi = copy.deepcopy(formulas[j])
        if e < 0:
            phi = Not(phi)
        shift_atom_thresholds(phi, -s * sigma[j])
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


def rule_formula(literals: list, formulas: list, sigma: np.ndarray, X_tr: np.ndarray,
                 cfg: RuleConfig, simplify: bool = True):
    """STL formula of an instance rule (to_stl), simplified (simplify_rule) unless simplify
    is False. None for the empty rule."""
    phi = to_stl(literals, formulas, sigma)
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
                        ref_labels: np.ndarray, formulas: list, sigma: np.ndarray,
                        cfg: RuleConfig, simplify: bool = True) -> dict[int, list]:
    """Global explanation per class: {k: [STL formula, ...]}, the OR of the listed rules.

    Rules come from global_rules, each simplified (simplify_rule) unless simplify is
    False; rules that become identical are kept once."""
    out = {}
    for k, R in global_rules(train_rules, X_tr_feats, ref_labels, cfg).items():
        phis = [rule_formula(lits, formulas, sigma, X_tr, cfg, simplify) for lits in R]
        out[k] = list({str(phi): phi for phi in phis}.values())
    return out
