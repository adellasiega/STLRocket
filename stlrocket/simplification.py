"""Simplification of global explanations (paper §3.4).

Two independent layers:
  - dedup/prune the set of local-explanation disjuncts that make up G(k)
  - shrink a single formula's tree / round its thresholds while preserving
    its positive-robustness mask on a reference dataset

Ported from local_experiments/notebook.ipynb, adapted to reuse
stlrocket.features.eval_robustness instead of a duplicated robustness() helper.
"""
from __future__ import annotations
import copy
import numpy as np
from torcheck.stl import Atom, Not, And, Or
from .features import eval_robustness


def positive_mask(phi, X: np.ndarray) -> np.ndarray:
    return eval_robustness(phi, X) > 0


def dedup_disjuncts(phis: list, masks: list, agreement: float) -> list[int]:
    """Merge near-identical disjuncts: if two positive masks agree on
    >= `agreement` fraction of samples, keep only the first (larger) one."""
    order = sorted(range(len(masks)), key=lambda i: -masks[i].sum())
    kept: list[int] = []
    for i in order:
        if all((masks[i] == masks[j]).mean() < agreement for j in kept):
            kept.append(i)
    return kept


def prune_disjuncts(idx: list[int], masks: list, target: np.ndarray, min_gain: float) -> list[int]:
    """Greedily drop disjuncts whose removal costs less than `min_gain`
    in score (coverage - FP rate)."""
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


def simplify_global(
    phi_disjuncts: list,
    X: np.ndarray,
    y: np.ndarray,
    target_class,
    agreement: float,
    min_gain: float,
) -> list:
    """Dedup near-identical disjuncts, then prune low-value ones."""
    y = np.asarray(y)
    target = y == target_class
    masks = [positive_mask(phi, X) for phi in phi_disjuncts]

    idx = dedup_disjuncts(phi_disjuncts, masks, agreement)
    idx = prune_disjuncts(idx, masks, target, min_gain)
    return [phi_disjuncts[i] for i in idx]


def _children_replacements(node):
    """Candidate smaller versions of this node."""
    if isinstance(node, (And, Or)):
        return [node.left_child, node.right_child]
    if isinstance(node, Not):
        if isinstance(node.child, Not):
            return [node.child.child]      # ¬¬phi -> phi
    return []


def _iter_nodes(node):
    """Yield (parent, attr_name, child) for every edge in the tree."""
    for attr in ("child", "left_child", "right_child"):
        child = getattr(node, attr, None)
        if child is not None and not isinstance(child, Atom):
            yield node, attr, child
            yield from _iter_nodes(child)
        elif isinstance(child, Atom):
            yield node, attr, child


def simplify_data_aware(phi, X: np.ndarray, agreement: float):
    """Greedily shrink phi while its positive mask on X stays
    >= `agreement` identical to the original. agreement=1.0 -> exact."""
    phi = copy.deepcopy(phi)
    ref_mask = positive_mask(phi, X)

    def ok(candidate):
        return (positive_mask(candidate, X) == ref_mask).mean() >= agreement

    improved = True
    while improved:
        improved = False

        # root-level replacement
        for rep in _children_replacements(phi):
            if ok(rep):
                phi = rep
                improved = True
                break
        if improved:
            continue

        # internal-node replacement
        for parent, attr, child in _iter_nodes(phi):
            for rep in _children_replacements(child):
                old = getattr(parent, attr)
                setattr(parent, attr, rep)
                if ok(phi):
                    improved = True
                    break
                setattr(parent, attr, old)      # revert
            if improved:
                break

    return phi


def round_thresholds(phi, X: np.ndarray, decimals: int, agreement: float):
    """Round atom thresholds where it doesn't change behavior on X."""
    phi = copy.deepcopy(phi)
    ref_mask = positive_mask(phi, X)

    def visit(node):
        if isinstance(node, Atom):
            old = node.threshold
            node.threshold = round(float(old), decimals)
            if (positive_mask(phi, X) == ref_mask).mean() < agreement:
                node.threshold = old        # revert
            return
        for attr in ("child", "left_child", "right_child"):
            child = getattr(node, attr, None)
            if child is not None:
                visit(child)

    visit(phi)
    return phi
