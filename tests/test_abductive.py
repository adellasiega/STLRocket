"""Checks of the probabilistic inflated AXp of stlrocket/abductive.py on synthetic linear models.

    python tests/test_abductive.py        (or pytest tests/)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket.abductive import MarginalSampler, axp, probabilistic_inflated_axp


def _model(seed=0, K=3, M=12, N=80):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((K, M))
    W[:, rng.choice(M, 3, replace=False)] = 0.0          # unselected features
    F = rng.standard_normal((N, M))
    return W, F


def test_incremental_logits():
    """The logits kept by the sampler equal those recomputed from its samples."""
    W, F = _model()
    s = MarginalSampler(F, W, n_samples=500)
    s.start(F[0], seed=1)
    rng = np.random.default_rng(2)
    for _ in range(30):
        c = int(rng.integers(len(s.S)))
        a, b = sorted(rng.standard_normal(2))
        s.set(c, a, b) if rng.random() < 0.7 else s.set(c, -np.inf, np.inf)
        assert np.allclose(s.L, s.Z @ s.WS.T)


def test_estimate_matches_exact_probability():
    """On a small product distribution, the estimate matches the probability by enumeration."""
    rng = np.random.default_rng(3)
    W = rng.standard_normal((2, 3))
    F = rng.standard_normal((6, 3))
    s = MarginalSampler(F, W, n_samples=200_000)
    s.start(F[0], seed=4)
    s.set(1, F[:, 1].min() + 1e-9, np.inf)                 # restricts feature 1: drops its smallest value
    vals = [F[:, 0], F[F[:, 1] >= F[:, 1].min() + 1e-9, 1], F[:, 2]]
    exact = np.mean([np.argmax(W @ np.array(z)) == 0 for z in itertools.product(*vals)])
    assert abs(s.precision(0) - exact) < 0.01, (s.precision(0), exact)


def test_explanation_properties():
    """From an AXp of probability 1, the result is no longer than the AXp, its bounds are those
    of the AXp or looser, and its estimate is at least delta; delta = 1 keeps probability 1."""
    for seed in range(5):
        W, F = _model(seed)
        lo, hi = F.min(axis=0), F.max(axis=0)
        s = MarginalSampler(F, W, n_samples=1000)
        for i in range(10):
            x = F[i]
            base = axp(x, W, lo, hi)
            if not base["sufficient"]:
                continue
            for delta in (0.9, 1.0):
                r = probabilistic_inflated_axp(x, W, lo, hi, s, delta=delta, seed=i)
                assert r["est_precision"] >= delta - 1e-12
                assert len(r["literals"]) <= len(base["literals"])
                axp_bounds = {(j, e): t for j, e, t in base["literals"]}
                for j, e, t in r["literals"]:
                    assert (j, e) in axp_bounds and t <= axp_bounds[(j, e)] + 1e-12   # looser or equal
                # x satisfies its own explanation
                assert all(e * x[j] > t for j, e, t in r["literals"])


def test_start_has_probability_one():
    """The literal AXp is sufficient on the box, so under the marginals its probability is 1."""
    W, F = _model(7)
    lo, hi = F.min(axis=0), F.max(axis=0)
    s = MarginalSampler(F, W, n_samples=2000)
    col = {int(j): c for c, j in enumerate(s.S)}
    for i in range(10):
        x = F[i]
        base = axp(x, W, lo, hi)
        if not base["sufficient"]:
            continue
        s.start(x, seed=i)
        for j, e, _ in base["literals"]:
            c = col[j]
            s.set(c, x[j] if e > 0 else s.a[c], s.b[c] if e > 0 else x[j])
        assert s.precision(base["k"]) == 1.0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            f()
    print("all passed")
