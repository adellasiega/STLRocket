"""Random STL formula grammar sampler.

Ported from stlkernel.distribution_formulae.F0 (~/Projects/STLKernel), retargeted
to build stlrocket.stl formulas instead of torcheck.stl ones. The recursive
sampling logic and probabilities are unchanged; only the temporal-operator
construction is adapted to stlcg++'s single interval=[a, b] convention (with
b possibly torch.inf for the "extends to the end of the signal" case), which
replaces torcheck's separate unbound/right_unbound/left_time_bound/
right_time_bound flags.

Formula strings/exact trees produced for a given seed will differ from the
torcheck-backed version; the sampling grammar and its statistics are the same.
"""
import random
from typing import Optional

import torch

from . import stl


class F0:
    def __init__(
        self,
        n_vars: int,
        v_min,
        v_max,
        t_max: int = 100,
        depth_max: int = 5,
        p_base: float = 0.05,        # base chance to stop at depth 0
        only_temporal: bool = False, # if true, discard purely boolean formulae
        until_weight: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.n_vars = n_vars
        self.v_min = v_min
        self.v_max = v_max
        self.t_max = t_max
        self.depth_max = depth_max
        self.p_base = p_base
        self.only_temporal = only_temporal
        self.until_weight = until_weight

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def _get_term_probability(self, current_depth: int) -> float:
        """Probability to stop and create an atom; grows linearly from p_base to 1.0."""
        growth = current_depth / self.depth_max
        return min(1.0, self.p_base + (1.0 - self.p_base) * growth)

    def _sample_formula(self, remaining_time: int, current_depth: int = 0):
        if current_depth >= self.depth_max or random.random() < self._get_term_probability(current_depth):
            return self._sample_atomic_predicate()
        return self._sample_operator_node(remaining_time, current_depth)

    def _sample_atomic_predicate(self):
        var_idx = random.randint(0, self.n_vars - 1)
        v_low = self.v_min[var_idx].item()
        v_high = self.v_max[var_idx].item()

        threshold = random.uniform(v_low, v_high)
        lte = random.choice([True, False])

        return stl.atom(var_index=var_idx, threshold=threshold, lte=lte)

    def _sample_interval(self, remaining_time: int):
        """Sample an interval=[a, b] for a temporal operator (b may be torch.inf)."""
        variants = ["unbound"]
        if remaining_time > 2:
            variants += ["bounded", "right_unbound"]
        variant = random.choice(variants)

        if variant == "unbound":
            return None, remaining_time
        elif variant == "bounded":
            b = random.randint(1, remaining_time - 1)
            a = random.randint(0, b - 1)
            return [a, b], remaining_time - b
        else:  # right_unbound
            a = random.randint(0, remaining_time - 1)
            return [a, torch.inf], remaining_time - a

    def _sample_operator_node(self, remaining_time: int, current_depth: int):
        classes = ["And", "Or", "Not", "Globally", "Eventually", "Until"]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0, self.until_weight]
        op = random.choices(classes, weights=weights, k=1)[0]

        if op == "And":
            left = self._sample_formula(remaining_time, current_depth + 1)
            right = self._sample_formula(remaining_time, current_depth + 1)
            return stl.And(left, right)

        elif op == "Or":
            left = self._sample_formula(remaining_time, current_depth + 1)
            right = self._sample_formula(remaining_time, current_depth + 1)
            return stl.Or(left, right)

        elif op == "Not":
            return stl.Negation(self._sample_formula(remaining_time, current_depth + 1))

        elif op in ("Globally", "Eventually"):
            OpClass = stl.Always if op == "Globally" else stl.Eventually
            interval, child_remaining_time = self._sample_interval(remaining_time)
            child = self._sample_formula(child_remaining_time, current_depth + 1)
            return OpClass(child, interval=interval)

        else:  # Until
            interval, child_remaining_time = self._sample_interval(remaining_time)
            left = self._sample_formula(child_remaining_time, current_depth + 1)
            right = self._sample_formula(child_remaining_time, current_depth + 1)
            return stl.Until(left, right, interval=interval)

    def is_temporal(self, formula):
        """Recursive check for whether a formula contains any temporal operators."""
        if isinstance(formula, (stl.Always, stl.Eventually, stl.Until)):
            return True
        if isinstance(formula, stl.Negation):
            return self.is_temporal(formula.subformula)
        if isinstance(formula, (stl.And, stl.Or)):
            return self.is_temporal(formula.subformula1) or self.is_temporal(formula.subformula2)
        return False  # It's an atom

    def sample(self, n_formulae: int) -> list:
        """Returns a list of n_formulae sampled formulas."""
        initial_time = self.t_max - 1
        sampled_formulae = []
        while len(sampled_formulae) < n_formulae:
            formula = self._sample_formula(initial_time, current_depth=0)
            if self.only_temporal and not self.is_temporal(formula):
                continue  # Discard purely boolean formulae
            sampled_formulae.append(formula)

        return sampled_formulae
