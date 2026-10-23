"""Checks of the cross-channel LinearAtom (stlrocket/linear_atom.py) and its use in the sampler.

    python tests/test_linear_atom.py        (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torcheck.stl import Eventually, Globally, Not

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket.config import ExperimentConfig
from stlrocket.features import _atoms, _robustness_trace, build_formula_bank, eval_robustness, shift_atom_thresholds
from stlrocket.formula_sampler import F0
from stlrocket.instance_rules import formula_key
from stlrocket.linear_atom import LinearAtom
from stlrocket.pretty import DatasetInfo, formula_stats, nnf, to_original_units, to_text


def _data(n=30, v=3, t=40, seed=0):
    return np.random.default_rng(seed).standard_normal((n, v, t)).cumsum(axis=2).astype(np.float32)


def _config(**kw):
    return ExperimentConfig(
        dataset="synthetic", n_formulas=200, depth_max=2, until_weight=0.0, cv=3, cut_point=1.0,
        fit_intercept=False, n_run=1, base_seed=0, output_dir=".", device="cpu", explain=False, pool_size=0,
        precision_threshold=0.0, simplify_agreement=0.0, simplify_min_gain=0.0, simplify_decimals=0, **kw)


def test_robustness_is_linear_combination():
    X = _data()
    for lte in (False, True):
        phi = LinearAtom(var_index=2, var_index2=0, a=0.6, b=-0.8, threshold=0.3, lte=lte)
        trace = _robustness_trace(phi, torch.from_numpy(X)).numpy()
        z = 0.6 * X[:, 2] - 0.8 * X[:, 0]
        assert np.allclose(trace, (0.3 - z) if lte else (z - 0.3), atol=1e-5)
        assert np.array_equal(phi.boolean(torch.from_numpy(X)).numpy(), eval_robustness(phi, X) >= 0)


def test_shift_and_negation():
    X = _data()
    phi = Globally(LinearAtom(0, 1, 0.8, 0.6, 0.2), unbound=False, left_time_bound=0, right_time_bound=10)
    rho = eval_robustness(phi, X)
    shift_atom_thresholds(phi, 0.7)
    assert np.allclose(eval_robustness(phi, X), rho + 0.7, atol=1e-5)
    neg = Not(phi)
    assert np.allclose(eval_robustness(nnf(neg), X), eval_robustness(neg, X), atol=1e-5)
    assert isinstance(nnf(neg).child, LinearAtom)


def test_sampler_stream_unchanged_when_off():
    kw = dict(n_vars=3, v_min=np.full(3, -1.0), v_max=np.full(3, 1.0), t_max=39, depth_max=2, until_weight=0.0)
    off = [str(f) for f in F0(**kw, seed=3).sample(300)]
    assert off == [str(f) for f in F0(**kw, seed=3, cross_prob=0.0).sample(300)]
    # With Average on, adding cross_prob = 0 still changes nothing
    avg = [str(f) for f in F0(**kw, seed=3, average_prob=0.5).sample(300)]
    assert avg == [str(f) for f in F0(**kw, seed=3, average_prob=0.5, cross_prob=0.0).sample(300)]
    on = F0(**kw, seed=3, cross_prob=0.5).sample(300)
    atoms = [n for f in on for n in _atoms(f)]
    cross = [n for n in atoms if isinstance(n, LinearAtom)]
    assert 0.3 < len(cross) / len(atoms) < 0.7
    assert all(n.var_index != n.var_index2 and n.a > 0 and abs(n.a ** 2 + n.b ** 2 - 1) < 1e-9 for n in cross)
    # A single-variable dataset never gets cross atoms
    one = F0(n_vars=1, v_min=np.full(1, -1.0), v_max=np.full(1, 1.0), t_max=39, depth_max=2, until_weight=0.0,
             seed=3, cross_prob=1.0).sample(50)
    assert not any(isinstance(n, LinearAtom) for f in one for n in _atoms(f))


def test_bank_centring_with_cross():
    X = _data(n=60)
    y = np.r_[np.zeros(45, dtype=int), np.ones(15, dtype=int)]
    formulas, F_tr, _, sigma = build_formula_bank(X, X[:5], y, _config(cross_prob=0.5), 0)
    assert any(isinstance(n, LinearAtom) for phi in formulas for n in _atoms(phi))
    w = compute_sample_weight("balanced", y)
    live = np.any(F_tr != 0, axis=0)
    assert np.allclose(np.average(F_tr[:, live], axis=0, weights=w), 0, atol=1e-4)
    for j in np.flatnonzero(live)[:10]:
        assert np.allclose(eval_robustness(formulas[j], X) / sigma[j], F_tr[:, j], atol=1e-4)


def test_original_units_and_printing():
    mean, std = np.array([10.0, -3.0, 0.5]), np.array([2.0, 0.5, 4.0])
    X = _data()                                   # standardized series
    X_orig = (X * std[None, :, None] + mean[None, :, None]).astype(np.float32)
    phi = Eventually(LinearAtom(0, 1, 0.6, -0.8, 0.4, lte=False), unbound=False, left_time_bound=2,
                     right_time_bound=20)
    # Same satisfaction on the original series
    assert np.array_equal(eval_robustness(phi, X) > 0, eval_robustness(to_original_units(phi, mean, std), X_orig) > 0)
    text = to_text(phi, DatasetInfo(["u", "g", "r"], 1.0, [None] * 3), mean, std)
    assert "0.3 u - 1.6 g >" in text
    assert formula_stats(phi)["variables"] == 2 and formula_stats(phi)["n_X"] == 1
    other = Eventually(LinearAtom(0, 2, 0.6, -0.8, 0.4), unbound=False, left_time_bound=2, right_time_bound=20)
    assert formula_key(phi) != formula_key(other)


if __name__ == "__main__":
    test_robustness_is_linear_combination()
    test_shift_and_negation()
    test_sampler_stream_unchanged_when_off()
    test_bank_centring_with_cross()
    test_original_units_and_printing()
    print("ok")
