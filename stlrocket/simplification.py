"""Data-aware simplification of an STL formula (paper, "Simplifying formulae"): shrink the
formula's tree while preserving its positive-robustness mask on a reference dataset."""
from __future__ import annotations
import copy
import numpy as np
from torcheck.stl import Atom, Not, And, Or
from .features import eval_robustness


def positive_mask(phi, X: np.ndarray) -> np.ndarray:
    return eval_robustness(phi, X) > 0


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
