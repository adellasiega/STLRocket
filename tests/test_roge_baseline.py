"""Checks of stlrocket/roge_baseline.py.

    python tests/test_roge_baseline.py        (or pytest tests/)

Needs the roge package (see stlrocket/roge_baseline.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket import roge_baseline as rb


def test_lazy_templates_match_roge():
    from roge.genetic import initial_templates
    for C in (1, 2, 3, 4):
        names = rb.variable_names(C)
        ref = [str(f) for f in initial_templates(names)]
        lazy = rb.LazyTemplates(names)
        assert len(lazy) == len(ref) == rb.n_templates(C)
        assert [str(f) for f in lazy] == ref
        assert str(lazy[-1]) == ref[-1]


def test_lazy_templates_same_population():
    """With the patch, ROGE's initial population is the one it builds from the full list."""
    rr = rb.roge_module()
    names = rb.variable_names(3)
    cfg = rr.RogeConfig(ne=30, seed=0)
    ref = [str(f) for f in rr.generate_initial_formulae(names, cfg, np.random.default_rng(5))]
    with rb.lazy_templates():
        lazy = [str(f) for f in rr.generate_initial_formulae(names, cfg, np.random.default_rng(5))]
    assert lazy == ref
    assert rr.genetic.initial_templates is not rb.LazyTemplates   # restored


def test_lazy_templates_many_variables():
    lazy = rb.LazyTemplates(rb.variable_names(1000))
    assert len(lazy) == rb.n_templates(1000) > 10 ** 7
    str(lazy[len(lazy) // 3]), str(lazy[-1])


def _toy(n, K, seed):
    """Class k: a bump of height 2 on variable k % 2 around time 5 + 4k, else noise."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), n)
    X = 0.3 * rng.standard_normal((len(y), 2, 30))
    for i, k in enumerate(y):
        X[i, k % 2, 5 + 4 * k: 9 + 4 * k] += 2.0
    return X, y


def test_binary_and_multiclass():
    fast = rb.ROGEOptions(ng=1, ne_max=12, gpucb={"n_init": 8, "max_iter": 5, "hyper_restarts": 1})
    for K in (2, 3):
        X, y = _toy(15, K, 0)
        Xte, yte = _toy(15, K, 1)
        clf = rb.ROGEClassifier(fast, seed=0, n_jobs=1).fit(X, y)
        assert len(clf.models_) == (1 if K == 2 else K)
        assert clf.robustness(Xte).shape == (len(yte), len(clf.models_))
        acc = np.mean(clf.predict(Xte) == yte)
        assert acc > 0.8, (K, acc)
        clf.to_json()


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
