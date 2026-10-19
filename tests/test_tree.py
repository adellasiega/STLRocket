"""Checks of stlrocket/tree.py on synthetic data.

    python tests/test_tree.py        (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket.config import ExperimentConfig
from stlrocket.features import build_formula_bank, eval_robustness
from stlrocket.tree import DEPTHS, path_formulas, train_tree


def test_path_formulas():
    """The STL formula of each root-to-leaf path is satisfied by exactly the series that
    reach that leaf, on train and on new series, and the leaves partition the series."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 2, 20)).cumsum(axis=2).astype(np.float32)
    y = (X[:, 0, -1] > 0).astype(int) + (X[:, 1].max(axis=1) > 3)
    cfg = ExperimentConfig(
        dataset="synthetic", n_formulas=200, depth_max=2, until_weight=0.0, cv=3, cut_point=1.0,
        fit_intercept=False, n_run=1, base_seed=0, output_dir=".", device="cpu", explain=False, pool_size=0,
        precision_threshold=0.0, simplify_agreement=0.0, simplify_min_gain=0.0, simplify_decimals=0)
    formulas, F_tr, F_te, sigma = build_formula_bank(X[:80], X[80:], y[:80], cfg, 0)
    tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0).fit(F_tr, y[:80])
    leaves = path_formulas(tree, formulas, sigma)
    assert len(leaves) == tree.get_n_leaves() > 1
    for Xs, Fs in ((X[:80], F_tr), (X[80:], F_te)):
        reached = tree.apply(Fs)
        sat = np.stack([eval_robustness(leaf["formula"], Xs) > 0 for leaf in leaves])
        for leaf, s in zip(leaves, sat):
            assert np.array_equal(s, reached == leaf["leaf"])
            assert len(leaf["literals"]) <= tree.get_depth()
        assert np.all(sat.sum(axis=0) == 1)
        # each series is predicted the class of the leaf whose formula it satisfies
        assert np.array_equal(np.array([leaf["class"] for leaf in leaves])[sat.argmax(axis=0)], tree.predict(Fs))


def test_train_tree():
    rng = np.random.default_rng(1)
    F = rng.standard_normal((90, 10))
    y = (F[:, 0] > 0).astype(int) + (F[:, 1] > 0.5)
    search = train_tree(F, y, cv=3, seed=0)
    d = search.best_params_["max_depth"]
    assert d in DEPTHS
    assert d is None or search.best_estimator_.get_depth() <= d


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
