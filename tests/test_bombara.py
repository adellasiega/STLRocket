"""Checks of stlrocket/bombara.py on synthetic data.

    python tests/test_bombara.py        (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket import bombara as bb
from stlrocket.features import eval_robustness


def test_statistics():
    """Sparse-table statistics equal the direct ones, and a primitive's formula is satisfied
    (robustness > 0) exactly when its statistic exceeds mu."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 1, 40)).astype(np.float32)
    st_max, st_min = bb.sparse_table(x[:, 0], np.maximum), bb.sparse_table(x[:, 0], np.minimum)
    for kind in bb.KINDS:
        for _ in range(50):
            a, b = sorted(rng.integers(0, 40, 2).tolist())
            c = int(rng.integers(0, 40 - b)) if kind in ("GF", "FG") else 0
            s = bb.statistic(x[:, 0], kind, a, b, c)
            assert np.array_equal(s, bb.statistic_tables(x[:, 0], kind, a, b, c, st_max, st_min)), (kind, a, b, c)
            mu = float(np.median(s)) + 1e-3
            rho = eval_robustness(bb.primitive_formula(dict(kind=kind, v=0, a=a, b=b, c=c, mu=mu)), x)
            assert np.array_equal(rho > 0, s > mu), (kind, a, b, c)


def test_best_thresholds():
    """The exact threshold sweep finds a perfect split when one exists."""
    S = np.array([[0.1, 0.2, 0.9, 1.0], [5.0, 5.0, 5.0, 5.0]])
    Yw = np.eye(2)[[0, 0, 1, 1]]
    gain, mu = bb.best_thresholds(S, Yw)
    assert np.isclose(gain[0], np.log(2)) and 0.2 < mu[0] < 0.9
    assert gain[1] == -np.inf


def test_tree_formulae():
    """Each class formula is satisfied by exactly the series the tree assigns to that class,
    on train and new series, at every depth; deeper cuts of a tree equal the trees grown to
    that depth; a learnable temporal pattern is found."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((150, 2, 30)).astype(np.float32)
    y = np.zeros(150, dtype=int)
    X[50:100, 0, 10:15] += 3          # class 1: a bump on x_0 in [10, 14]
    y[50:100] = 1
    X[100:, 1, 20:] -= 2              # class 2: x_1 low from 20 on
    y[100:] = 2
    perm = rng.permutation(150)
    X, y = X[perm], y[perm]
    tr, te = slice(0, 100), slice(100, 150)
    cfg = bb.PSOConfig(n_particles=10, n_iter=15)
    tree = bb.fit_tree(X[tr], y[tr], 3, seed=0, cfg=cfg, n_jobs=2)
    assert (tree.predict(X[te]) == y[te]).mean() > 0.9
    for d in (1, 2, 3):
        for part in (tr, te):
            pred = tree.predict_index(X[part], d)
            for k in range(3):
                phi = tree.class_formula(k, d)
                sat = eval_robustness(phi, X[part]) > 0 if phi is not None else np.zeros(len(pred), bool)
                assert np.array_equal(sat, pred == k), (d, k)
    shallow = bb.fit_tree(X[tr], y[tr], 1, seed=0, cfg=cfg)
    assert np.array_equal(shallow.predict(X), tree.predict(X, 1))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
