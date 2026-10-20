"""
Global surrogates of the classifier, the baselines of its global explanations: rule models on
the selected formulae, trained on train to reproduce the classes predicted by the model.
For each class k each returns a list of rules (conjunctions of literals (j, e, s) of
instance_rules, e * x_j > s) whose OR is the explanation of k, and the masks of the series of
each split satisfying that OR.

  tree    tree.train_tree (balanced class weights, max_depth in tree.DEPTHS by CV on balanced
          accuracy, here w.r.t. the predictions) on the selected formulae: class k is the OR of
          the root-to-leaf paths of its leaves
  ripper  RIPPER (Cohen 1995; package wittgenstein, default settings) one-vs-rest on the
          predictions: class k is its rule set for k against the others. wittgenstein bins each
          feature into quantile bins (n_discretize_bins); a condition on a bin with two finite
          ends is two literals
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tree import path_literals, train_tree


def tree_surrogate(F: dict, S: np.ndarray, pred: dict, cv: int, seed: int, n_jobs: int = 1) -> dict:
    """F, pred: split -> feature matrix (all formulae) and predicted class indices."""
    search = train_tree(F["train"][:, S], pred["train"], cv, seed, n_jobs=n_jobs)
    tree = search.best_estimator_
    leaves = path_literals(tree)
    out = {}
    for k in np.unique(pred["train"]):
        rules = [[(int(S[j]), e, s) for j, e, s in leaf["literals"]] for leaf in leaves if leaf["class"] == k]
        out[int(k)] = {"rules": rules, "masks": {s: tree.predict(F[s][:, S]) == k for s in F}}
    return {"classes": out, "max_depth": search.best_params_["max_depth"], "depth": int(tree.get_depth())}


def _cond_literals(cond, j: int) -> list:
    """Literals of a wittgenstein condition on feature j: its value is a bin of the training
    discretization, written '<a', '>b' or 'a-b' (numbers may be negative). Bins are
    half-open; the literals use strict thresholds at the bin edges, exact up to equality."""
    v = str(cond.val)
    if v.startswith("<"):
        return [(j, -1, -float(v[1:]))]
    if v.startswith(">"):
        return [(j, 1, float(v[1:]))]
    i = v.index("-", 1)      # the separator: the first '-' after a possible leading sign
    a, b = float(v[:i]), float(v[i + 1:])
    return [(j, 1, a), (j, -1, -b)]


def ripper_surrogate(F: dict, S: np.ndarray, pred: dict, seed: int) -> dict:
    import wittgenstein as lw
    cols = [f"x{j}" for j in S]
    X = {s: pd.DataFrame(F[s][:, S], columns=cols) for s in F}
    out = {}
    for k in np.unique(pred["train"]):
        clf = lw.RIPPER(random_state=seed)
        clf.fit(X["train"], (pred["train"] == k).astype(int), pos_class=1)
        rules = [[lit for c in r.conds for lit in _cond_literals(c, int(c.feature[1:]))] for r in clf.ruleset_.rules]
        out[int(k)] = {"rules": rules, "masks": {s: np.asarray(clf.predict(X[s]), dtype=bool) for s in F}}
    return {"classes": out}
