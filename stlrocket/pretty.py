"""Readable form of the STL formulae of explanations (paper case study).

Formulae are built on standardized series (data.standardize_series), so atom thresholds are
in standard deviations of their variable and time bounds are sample indices. to_text prints
them in the units of the original series and in seconds: a threshold c on variable v is
mean[v] + c * std[v], and index i is i / hz seconds. The windows of a temporal operator at the
top of a formula are absolute (the formula is read at t = 0); those of a nested operator are
relative to the time at which it is read, and are printed as offsets ("+0.2 s").

Negations are first pushed down to the atoms (nnf), which changes no robustness value:
not G = F not, not F = G not, De Morgan for and/or, and not (x <= c) is x >= c. Satisfaction
is positive robustness, so atoms are printed with strict inequalities.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from torcheck.stl import Atom, Not, And, Or, Globally, Eventually, Until

from .average import Average
from .features import _atoms, _children
from .formula_sampler import F0


@dataclass
class DatasetInfo:
    var_names: list[str]
    hz: float
    units: list[str | None]     # None: undocumented, values are printed in the raw units of the archive


# Channel order checked on the data: BasicMotions has the accelerometer first (all six
# channels are near 0 for standing, accelerometer spread largest for running). The archive
# documents neither unit; BasicMotions' accelerometer has gravity removed (|acc| ~ 0.8 standing).
DATASET_INFO = {
    "BasicMotions": DatasetInfo(["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"], 10.0, [None] * 6),
    "Epilepsy": DatasetInfo(["acc_x", "acc_y", "acc_z"], 16.0, [None] * 3),
}


def _temporal(cls, node, child):
    return cls(child, unbound=node.unbound, right_unbound=node.right_unbound,
               left_time_bound=node.left_time_bound, right_time_bound=node.right_time_bound - 1)


def nnf(phi, negate: bool = False):
    """Copy of phi with every negation pushed down to the atoms (absorbed by flipping the
    atom direction); same robustness on every series. Not Until is kept as it is."""
    if isinstance(phi, Average):
        return Average(var_index=phi.var_index, threshold=phi.threshold, lte=phi.lte != negate,
                       left_time_bound=phi.left_time_bound, right_time_bound=phi.right_time_bound - 1)
    if isinstance(phi, Atom):
        return Atom(var_index=phi.var_index, threshold=phi.threshold, lte=phi.lte != negate)
    if isinstance(phi, Not):
        return nnf(phi.child, not negate)
    if isinstance(phi, (And, Or)):
        cls = (Or if isinstance(phi, And) else And) if negate else type(phi)
        return cls(nnf(phi.left_child, negate), nnf(phi.right_child, negate))
    if isinstance(phi, (Globally, Eventually)):
        cls = (Eventually if isinstance(phi, Globally) else Globally) if negate else type(phi)
        return _temporal(cls, phi, nnf(phi.child, negate))
    if isinstance(phi, Until):
        u = Until(nnf(phi.left_child), nnf(phi.right_child), unbound=phi.unbound, right_unbound=phi.right_unbound,
                  left_time_bound=phi.left_time_bound, right_time_bound=phi.right_time_bound - 1)
        return Not(u) if negate else u
    raise TypeError(f"unknown node type: {type(phi).__name__}")


def window(node) -> tuple[int, int | None]:
    """(a, b) in samples of a temporal node, b None when the window runs to the end."""
    if node.unbound:
        return 0, None
    if node.right_unbound:
        return node.left_time_bound, None
    return node.left_time_bound, node.right_time_bound - 1


def _seconds(i: int, hz: float) -> str:
    return f"{i / hz:.3g} s"


def _window_text(node, hz: float, nested: bool) -> str:
    a, b = window(node)
    sign = "+" if nested else ""
    end = "the end" if b is None else f"{sign}{_seconds(b, hz)}"
    if a == 0 and b is None and not nested:
        return "the whole series"
    return f"[{sign}{_seconds(a, hz)}, {end}]"


def atom_threshold(atom: Atom, mean: np.ndarray, std: np.ndarray) -> float:
    """Threshold of the atom in the units of the original series."""
    return float(mean[atom.var_index] + atom.threshold * std[atom.var_index])


def to_original_units(phi, mean: np.ndarray, std: np.ndarray):
    """Copy of phi with every atom threshold in the units of the original series, to be
    evaluated on the original series (missing values filled as in data.load_dataset).

    x_v > c on the standardized series is x~_v > mean[v] + c * std[v] on the original one, so
    every atom, hence the formula, is satisfied by the same series. Robustness is not kept: the
    robustness of an atom on v is multiplied by std[v], so it is only a rescaling of the
    original robustness when every atom reads the same variable."""
    phi = copy.deepcopy(phi)
    for atom in _atoms(phi):
        atom.threshold = atom_threshold(atom, mean, std)
    return phi


def to_text(phi, info: DatasetInfo, mean: np.ndarray, std: np.ndarray, _nested: bool = False) -> str:
    """Readable form of phi, thresholds in original units and windows in seconds."""
    def rec(node, nested):
        if isinstance(node, Atom):
            v = node.var_index
            unit = f" {info.units[v]}" if info.units[v] else ""
            name = info.var_names[v]
            if isinstance(node, Average):
                name = f"mean of {name} over {_window_text(node, info.hz, nested)}"
            return f"{name} {'<' if node.lte else '>'} {atom_threshold(node, mean, std):.3g}{unit}"
        if isinstance(node, Not):
            return f"not ({rec(node.child, nested)})"
        if isinstance(node, (And, Or)):
            op = "and" if isinstance(node, And) else "or"
            return f"({rec(node.left_child, nested)}) {op} ({rec(node.right_child, nested)})"
        if isinstance(node, Globally):
            return f"always in {_window_text(node, info.hz, nested)}: {rec(node.child, True)}"
        if isinstance(node, Eventually):
            return f"at some time in {_window_text(node, info.hz, nested)}: {rec(node.child, True)}"
        if isinstance(node, Until):
            return (f"({rec(node.left_child, True)}) until, within {_window_text(node, info.hz, nested)}, "
                    f"({rec(node.right_child, True)})")
        raise TypeError(f"unknown node type: {type(node).__name__}")
    return rec(nnf(phi), _nested)


def atom_spans(phi) -> list[tuple[Atom, int, int | None, bool]]:
    """Every atom of nnf(phi) with the span of absolute time steps it can be read on when phi is
    read at t = 0 (sum of the windows of the temporal operators above it, and of its own if
    it is an Average; end None: to the end of the series) and whether it lies under an odd
    number of negations."""
    out = []

    def rec(node, a, b, neg):
        if isinstance(node, Atom):
            if isinstance(node, Average):
                wa, wb = window(node)
                a, b = a + wa, (None if b is None else b + wb)
            out.append((node, a, b, neg))
            return
        if isinstance(node, Not):
            rec(node.child, a, b, not neg)
            return
        if isinstance(node, (Globally, Eventually, Until)):
            wa, wb = window(node)
            a, b = a + wa, (None if b is None or wb is None else b + wb)
        for c in _children(node):
            rec(c, a, b, neg)

    rec(nnf(phi), 0, 0, False)
    return out


def formula_stats(phi) -> dict:
    """Size (nodes), depth, atoms, distinct variables and temporal operators of phi
    (A: Average atoms, also counted in atoms)."""
    counts = {"G": 0, "F": 0, "U": 0, "A": 0}

    def rec(node):
        if isinstance(node, Average):
            counts["A"] += 1
        elif isinstance(node, Globally):
            counts["G"] += 1
        elif isinstance(node, Eventually):
            counts["F"] += 1
        elif isinstance(node, Until):
            counts["U"] += 1
        for c in _children(node):
            rec(c)

    rec(phi)
    atoms = _atoms(phi)
    return {"size": F0.formula_size(phi), "depth": F0.formula_depth(phi), "atoms": len(atoms),
            "variables": len({a.var_index for a in atoms}), **{f"n_{k}": v for k, v in counts.items()}}
