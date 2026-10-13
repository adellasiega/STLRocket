"""Property checks of stlrocket/instance_rules.py on synthetic data.

    python tests/test_instance_rules.py        (or pytest tests/)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket import instance_rules as ir
from stlrocket.config import ExperimentConfig
from stlrocket.features import build_formula_bank, eval_robustness


def random_model(rng, n=200, m=30, k=4, density=0.5):
    """Scaled features, logit weights with some zero entries, and the model's predictions."""
    F = rng.standard_normal((n, m))
    W = rng.standard_normal((k, m)) * (rng.random((k, m)) < density)
    return F, W, ir.predict_indices(F, W)


def test_candidates():
    rng = np.random.default_rng(0)
    for k in (2, 4):
        F, W, _ = random_model(rng, k=k)
        for x in F[:50]:
            J = {m: set(ir.candidate_scores(x, W, m)[2].tolist()) for m in ir.CANDIDATE_METHODS}
            assert J["gxi"] and J["ig"] and J["union"]
            assert J["gxi"] <= J["union"] and J["ig"] <= J["union"] <= J["all"]
            if k == 2:   # a single competitor: nothing to aggregate
                assert J["gxi"] == J["ig"] == J["union"]
            # completeness of Integrated Gradients: sum_j s_j = log(K P(k|x))
            cls, s, _ = ir.candidate_scores(x, W, "ig")
            z = W @ x
            log_p = z[cls] - z.max() - np.log(np.exp(z - z.max()).sum())
            assert abs(s.sum() - (np.log(k) + log_p)) < 1e-3


def test_rules():
    rng = np.random.default_rng(1)
    F, W, pred = random_model(rng)
    for method, i in itertools.product(ir.CANDIDATE_METHODS, range(40)):
        r = {g: ir.explain_instance(F[i], W, F, pred, ir.RuleConfig(candidates=method, generalize=g), exclude=i)
             for g in (True, False)}
        assert r[True]["lcb"] >= r[False]["lcb"]            # the greedy pass never lowers the bound
        for rule in r.values():
            if not rule["literals"]:
                continue
            assert rule["k"] == pred[i]
            assert all(s >= 0 and e * F[i, j] > s for j, e, s in rule["literals"])   # consistency
            assert len({j for j, _, _ in rule["literals"]}) == len(rule["literals"]) <= 3
            m = ir.rule_metrics(ir.covered(rule["literals"], F), rule["k"], pred, exclude=i)
            assert (m["tp"], m["support"]) == (rule["tp"], rule["support"]) and m["support"] >= 5
            assert np.isclose(rule["lcb"], ir.wilson_lower(m["tp"], m["support"]))


def test_stl_and_simplification():
    """A rule's STL formula is satisfied by the series its literals cover, on the train
    series, and simplification does not change them."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 2, 20)).cumsum(axis=2).astype(np.float32)
    cfg = ExperimentConfig(
        dataset="synthetic", n_formulas=200, depth_max=2, until_weight=0.0, cv=3, cut_point=1.0,
        fit_intercept=False, n_run=1, base_seed=0, output_dir=".", device="cpu", explain=False, pool_size=0,
        precision_threshold=0.0, simplify_agreement=0.0, simplify_min_gain=0.0, simplify_decimals=0)
    formulas, F, _, sigma = build_formula_bank(X, X[:5], cfg, 0)
    W = rng.standard_normal((3, F.shape[1])) * (rng.random((3, F.shape[1])) < 0.1)
    pred = ir.predict_indices(F, W)
    rcfg = ir.RuleConfig()
    n_rules = 0
    for i in range(30):
        lits = ir.explain_instance(F[i], W, F, pred, rcfg, exclude=i)["literals"]
        if not lits:
            continue
        n_rules += 1
        mask = ir.covered(lits, F)
        raw = ir.rule_formula(lits, formulas, sigma, X, rcfg, simplify=False)
        assert np.array_equal(eval_robustness(raw, X) > 0, mask)
        assert np.array_equal(eval_robustness(ir.simplify_rule(raw, X, rcfg), X) > 0, mask)
    assert n_rules > 0


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: ok")
