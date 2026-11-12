"""Checks of stlrocket/tlinet.py on synthetic data.

    python tests/test_tlinet.py        (or pytest tests/)

The reference operators below are plain loops written from the definitions in the TLINet
papers, against which the vectorized ones of stlrocket.tlinet are compared.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket import tlinet as tl
from stlrocket.features import eval_robustness


# -- loop references -----------------------------------------------------------------------
def ref_window(T, t1, t2, tau):
    ramp = lambda s: min(max(s / tau, 0.0), 1.0)
    return [min(ramp(t - t1 + tau), ramp(t2 - t + tau)) for t in range(T)]


def ref_sparse_softmax(x, w, beta, h):
    """Weighted sparse softmax of one sequence x with weights w."""
    sw = sum(w)
    wn = [wi / sw for wi in w] if sw != 0 else list(w)
    rw = [xi * wi for xi, wi in zip(x, wn)]
    mx = abs(max(rw)) or 1.0
    e = [math.exp(beta * h * r / mx) for r in rw]
    s = [ei / sum(e) for ei in e]
    den = sum(si * wi for si, wi in zip(s, wn)) or 1.0
    return sum(si * wi * xi for si, wi, xi in zip(s, wn, x)) / den


def ref_averaged_max(x, p):
    """Expected largest kept input, each input i kept independently with probability p_i."""
    order = sorted(range(len(x)), key=lambda i: -x[i])
    out, none_before = 0.0, 1.0
    for i in order:
        out += x[i] * p[i] * none_before
        none_before *= 1 - p[i]
    return out


def test_operators():
    rng = np.random.default_rng(0)
    T = 15
    t1, t2 = torch.tensor([2.3, 0.0, 7.0]), torch.tensor([9.6, 4.0, 14.0])
    w = tl.time_weights(T, t1, t2, 1.0)
    for i in range(3):
        assert np.allclose(w[i].numpy(), ref_window(T, float(t1[i]), float(t2[i]), 1.0), atol=1e-6)
    x = torch.tensor(rng.standard_normal((4, 3, T)), dtype=torch.float64)
    out = tl.sparse_softmax(x, w.double(), 2.5, 1.0)
    for b in range(4):
        for i in range(3):
            ref = ref_sparse_softmax(x[b, i].tolist(), w[i].tolist(), 2.5, 1.0)
            assert abs(float(out[b, i]) - ref) < 1e-9, (b, i)
    x = torch.tensor(rng.standard_normal((5, 6)), dtype=torch.float64)
    p = torch.tensor(rng.uniform(size=6), dtype=torch.float64)
    out = tl.averaged_max(x, p)
    for b in range(5):
        assert abs(float(out[b]) - ref_averaged_max(x[b].tolist(), p.tolist())) < 1e-12
    # with binary p the averaged max is the max over the selected inputs
    pb = torch.tensor([1.0, 0, 1, 0, 0, 1], dtype=torch.float64)
    assert torch.allclose(tl.averaged_max(x, pb), x[:, pb.bool()].max(1).values)


def _random_model(n_clauses, seed=0, K=3, C=3, T=20):
    g = torch.Generator().manual_seed(seed)
    m = tl.TLINet(K, C, T, n_clauses, tl.TLINetConfig(), g)
    with torch.no_grad():   # spread the parameters over both readings of every operator
        m.a.uniform_(-1, 1, generator=g)
        m.b.uniform_(-1, 1, generator=g)
    m.project()
    return m


def test_exact_matches_formula():
    """The vectorized exact robustness equals torcheck's robustness of the extracted formula,
    for both architectures, including empty formulas (-inf)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((25, 3, 20)).astype(np.float32)
    n_empty = 0
    for n_clauses in (None, 2, 4):
        for seed in range(4):
            m = _random_model(n_clauses, seed)
            rho = m.exact(torch.from_numpy(X), slice(None)).numpy()
            for k in range(m.K):
                f = m.formula(k)
                if f is None:
                    n_empty += 1
                    assert np.all(np.isneginf(rho[:, k]))
                else:
                    assert np.allclose(rho[:, k], eval_robustness(f, X), atol=1e-5), (n_clauses, seed, k)
    print(f"  {n_empty} empty formulas among the random models")


def test_authors_hard():
    """The authors' hard evaluation equals the loop references composed by hand: sparse
    softmax over the soft window, then the averaged max with binary p (the max or min of the
    selected inputs, 0 when none is selected)."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((6, 3, 20)).astype(np.float32)
    for n_clauses in (None, 2):
        m = _random_model(n_clauses, seed=5)
        out = m.authors_hard(torch.from_numpy(X), slice(None)).numpy()
        thr = m.a.detach().numpy(), m.b.detach().numpy()
        for k in range(m.K):
            w = tl.time_weights(m.T, m.t1[k], m.t2[k], 1.0).detach().numpy()
            for b in range(len(X)):
                r1 = []
                for i in range(m.nf):
                    z = (thr[0][k, i] * X[b, int(m.var[i])] - thr[1][k, i]).tolist()
                    if m.tvar_t[k, i] > 0.5:
                        r1.append(ref_sparse_softmax(z, w[i].tolist(), 2.5, 1.0))
                    else:
                        r1.append(-ref_sparse_softmax([-v for v in z], w[i].tolist(), 2.5, 1.0))

                def clause(r, p, tvar):
                    pb = [1.0 if q >= 0.5 else 0.0 for q in p]
                    if tvar > 0.5:
                        return ref_averaged_max(r, pb)
                    return -ref_averaged_max([-v for v in r], pb)

                c = [clause(r1, m.p1[k, j].detach().tolist(), float(m.tvar_l1[k, j].detach())) for j in range(m.p1.shape[1])]
                ref = c[0] if n_clauses is None else clause(c, m.p2[k, 0].detach().tolist(), float(m.tvar_l2[k, 0].detach()))
                assert abs(out[b, k] - ref) < 1e-4, (n_clauses, k, b, out[b, k], ref)


def _toy(n=80, C=2, T=30, seed=0):
    """Class 1: a bump of height 3 on variable 1 at t in [12, 16]; class 0: noise only."""
    rng = np.random.default_rng(seed)
    X = 0.3 * rng.standard_normal((n, C, T)).astype(np.float32)
    y = np.arange(n) % 2
    X[y == 1, 1, 12:17] += 3.0
    return X, y


def test_toy_recovery():
    """Training reaches a formula that separates the toy classes and reads the bump."""
    X, y = _toy()
    cfg = tl.TLINetConfig(epochs=100)
    clf = tl.TLINetClassifier(None, cfg, random_state=0).fit(X, y)
    assert float(np.mean(clf.predict(X, "exact") == y)) >= 0.95
    pred = clf.predict(X)
    acc = float(np.mean(pred == y))
    print(f"  toy: train accuracy {acc:.3f}, formula {clf.formula_text(0)}")
    assert acc >= 0.95
    preds = clf._predicates(0)
    assert any(v == 1 and lo <= 16 and hi >= 12 for v, _, lo, hi, _, _ in preds), preds
    X_te, y_te = _toy(seed=1)
    assert float(np.mean(clf.predict(X_te) == y_te)) >= 0.9


def test_multiclass_and_chunking():
    """Three classes: one model per class; splitting the models into chunks does not change
    training (their gradients are disjoint)."""
    rng = np.random.default_rng(2)
    X = 0.3 * rng.standard_normal((60, 2, 20)).astype(np.float32)
    y = np.arange(60) % 3
    X[y == 1, 0, 5:9] += 3.0
    X[y == 2, 1, 12:16] -= 3.0
    full = tl.TLINetClassifier(2, tl.TLINetConfig(epochs=20), random_state=3).fit(X, y)
    chunked = tl.TLINetClassifier(2, tl.TLINetConfig(epochs=20, max_elements=1), random_state=3).fit(X, y)
    assert full.model_.K == 3
    for (n, p), (_, q) in zip(full.model_.named_parameters(), chunked.model_.named_parameters()):
        assert torch.allclose(p, q, atol=1e-5), n
    assert np.array_equal(full.predict(X), chunked.predict(X))
    assert set(full.predict(X)) <= {0, 1, 2}
    # string labels go through classes_
    labels = np.array(["a", "b", "c"])[y]
    assert set(tl.TLINetClassifier(None, tl.TLINetConfig(epochs=5)).fit(X, labels).predict(X)) <= {"a", "b", "c"}


def test_ovr_targets():
    y = np.array([0, 0, 0, 1, 2, 2])
    Y, W, models = tl.ovr_targets(y, 3)
    assert models == [0, 1, 2]
    for j in range(3):
        assert np.isclose(W[Y[:, j] > 0, j].sum(), 0.5) and np.isclose(W[Y[:, j] < 0, j].sum(), 0.5)
    Y, W, models = tl.ovr_targets(np.array([0, 1, 1, 1]), 2)
    assert models == [1] and Y.shape == (4, 1) and np.isclose(W[0, 0], 0.5)


def test_choose_architecture():
    X, y = _toy(n=40)
    folds = [(np.arange(40)[np.arange(40) % 3 != f], np.arange(40)[np.arange(40) % 3 == f]) for f in range(3)]
    arch, cv = tl.choose_architecture(X, y, folds, 0, tl.TLINetConfig(epochs=10), architectures=(None, 2))
    assert set(cv) == {None, 2} and arch in cv and cv[arch] == max(cv.values())


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            f()
    print("all passed")
