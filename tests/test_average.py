"""Checks of the Average atom (stlrocket/average.py) and its use in the sampler.

    python tests/test_average.py        (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torcheck.stl import Atom, Globally, Not

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket.average import Average
from stlrocket.config import ExperimentConfig
from stlrocket.features import _robustness_trace, build_formula_bank, eval_robustness, shift_atom_thresholds
from stlrocket.formula_sampler import F0
from stlrocket.instance_rules import formula_key
from stlrocket.pretty import DatasetInfo, atom_spans, nnf, to_text


def _data(n=30, v=2, t=40, seed=0):
    return np.random.default_rng(seed).standard_normal((n, v, t)).cumsum(axis=2).astype(np.float32)


def _config(**kw):
    return ExperimentConfig(
        dataset="synthetic", n_formulas=200, depth_max=2, until_weight=0.0, cv=3, cut_point=1.0,
        fit_intercept=False, n_run=1, base_seed=0, output_dir=".", device="cpu", explain=False, pool_size=0,
        precision_threshold=0.0, simplify_agreement=0.0, simplify_min_gain=0.0, simplify_decimals=0, **kw)


def test_robustness_is_moving_average():
    X = _data()
    a, b, c = 3, 9, 0.5
    for lte in (False, True):
        phi = Average(var_index=1, threshold=c, lte=lte, left_time_bound=a, right_time_bound=b)
        trace = _robustness_trace(phi, torch.from_numpy(X)).numpy()
        T = X.shape[2]
        ma = np.stack([X[:, 1, t + a:t + b + 1].mean(axis=1) for t in range(T - b)], axis=1)
        expected = (c - ma) if lte else (ma - c)
        assert trace.shape == expected.shape == (X.shape[0], T - phi.time_depth())
        assert np.allclose(trace, expected, atol=1e-5)
        # Same trace length as a bounded G over the plain atom
        g = Globally(Atom(1, c, lte), unbound=False, left_time_bound=a, right_time_bound=b)
        assert _robustness_trace(g, torch.from_numpy(X)).shape == trace.shape
        assert np.array_equal(phi.boolean(torch.from_numpy(X)).numpy(), eval_robustness(phi, X) >= 0)


def test_shift_and_negation():
    X = _data()
    phi = Globally(Average(0, 0.2, False, 1, 5), unbound=False, left_time_bound=0, right_time_bound=10)
    rho = eval_robustness(phi, X)
    shift_atom_thresholds(phi, 0.7)
    assert np.allclose(eval_robustness(phi, X), rho + 0.7, atol=1e-5)
    neg = Not(phi)
    assert np.allclose(eval_robustness(nnf(neg), X), eval_robustness(neg, X), atol=1e-5)
    assert isinstance(nnf(neg).child, Average)


def test_sampler_stream_unchanged_when_off():
    kw = dict(n_vars=2, v_min=np.full(2, -1.0), v_max=np.full(2, 1.0), t_max=39, depth_max=2, until_weight=0.0)
    off = [str(f) for f in F0(**kw, seed=3).sample(300)]
    assert off == [str(f) for f in F0(**kw, seed=3, average_prob=0.0).sample(300)]
    on = F0(**kw, seed=3, average_prob=0.5).sample(300)
    n_avg = sum("avg[" in str(f) for f in on)
    assert n_avg > 100
    # Average does not spend a depth level, and its window stays within the budget
    assert max(F0.formula_depth(f) for f in on) <= 2
    assert max(f.time_depth() for f in on) <= 39


def test_bank_centring_with_average():
    X = _data(n=60)
    y = np.r_[np.zeros(45, dtype=int), np.ones(15, dtype=int)]
    formulas, F_tr, _, sigma = build_formula_bank(X, X[:5], y, _config(average_prob=0.5), 0)
    assert any(isinstance(n, Average) for phi in formulas for n, *_ in atom_spans(phi))
    w = compute_sample_weight("balanced", y)
    live = np.any(F_tr != 0, axis=0)
    assert np.allclose(np.average(F_tr[:, live], axis=0, weights=w), 0, atol=1e-4)
    for j in np.flatnonzero(live)[:10]:
        assert np.allclose(eval_robustness(formulas[j], X) / sigma[j], F_tr[:, j], atol=1e-4)


def test_printing_and_keys():
    phi = Globally(Average(0, 0.0, False, 2, 6), unbound=False, left_time_bound=0, right_time_bound=10)
    text = to_text(phi, DatasetInfo(["acc", "gyr"], 10.0, [None, None]), np.zeros(2), np.ones(2))
    assert "mean of acc over [+0.2 s, +0.6 s] > 0" in text
    assert atom_spans(phi)[0][1:3] == (2, 16)
    other = Globally(Average(0, 0.0, False, 2, 7), unbound=False, left_time_bound=0, right_time_bound=10)
    plain = Globally(Atom(0, 0.0, False), unbound=False, left_time_bound=0, right_time_bound=10)
    assert len({formula_key(phi), formula_key(other), formula_key(plain)}) == 3


if __name__ == "__main__":
    test_robustness_is_moving_average()
    test_shift_and_negation()
    test_sampler_stream_unchanged_when_off()
    test_bank_centring_with_average()
    test_printing_and_keys()
    print("ok")
