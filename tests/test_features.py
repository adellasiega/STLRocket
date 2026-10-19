"""Checks of stlrocket/features.py on synthetic data.

    python tests/test_features.py        (or pytest tests/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket.config import ExperimentConfig
from stlrocket.features import build_formula_bank, eval_robustness


def test_class_balanced_centring():
    """On imbalanced train labels, every feature has zero class-balanced mean and unit
    class-balanced std on train (or is a constant column of zeros), and is the robustness
    of its stored, shifted formula divided by sigma_j."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2, 20)).cumsum(axis=2).astype(np.float32)
    y = np.r_[np.zeros(80, dtype=int), np.ones(20, dtype=int)]
    X[y == 1] += 2.0     # the minority class shifts the plain mean away from the balanced one
    cfg = ExperimentConfig(
        dataset="synthetic", n_formulas=100, depth_max=2, until_weight=0.0, cv=3, cut_point=1.0,
        fit_intercept=False, n_run=1, base_seed=0, output_dir=".", device="cpu", explain=False, pool_size=0,
        precision_threshold=0.0, simplify_agreement=0.0, simplify_min_gain=0.0, simplify_decimals=0)
    formulas, F_tr, F_te, sigma = build_formula_bank(X, X[:5], y, cfg, 0)
    w = compute_sample_weight("balanced", y)
    live = np.any(F_tr != 0, axis=0)
    assert live.sum() > 50
    assert np.allclose(np.average(F_tr[:, live], axis=0, weights=w), 0, atol=1e-4)
    assert np.allclose(np.sqrt(np.average(F_tr[:, live] ** 2, axis=0, weights=w)), 1, atol=1e-4)
    # The plain mean is not zero: the weighting is actually applied
    assert np.abs(F_tr[:, live].mean(axis=0)).max() > 0.1
    for j in np.flatnonzero(live)[:10]:
        assert np.allclose(eval_robustness(formulas[j], X) / sigma[j], F_tr[:, j], atol=1e-4)


if __name__ == "__main__":
    test_class_balanced_centring()
    print("ok")
