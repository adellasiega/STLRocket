"""Random STL formula grammar sampler.

Ported from stlkernel.distribution_formulae.F0 (~/Projects/STLKernel) so it lives
in this project instead of depending on the external stlkernel package. Builds
formulas out of torcheck.stl node classes (Atom/Not/And/Or/Globally/Eventually/
Until) -- torcheck (~/Projects/TorCheck) remains STLRocket's STL AST/quantitative-
semantics engine; a prior attempt to replace it with a vendored copy of stlcg++
was reverted after stlcg++'s vmap-based batching turned out to scale as O(N*T^2)
(O(N*W*T^2) for Until) in sample count N and signal length T, causing OOMs on
long-time-series datasets. torcheck operates on truly-batched (N, V, T) tensors
with O(T) / O(T*W) algorithms for the unbounded/bounded temporal operators, which
is why we're back on it.
"""
import random
from typing import Optional

import torch
from torcheck import stl


class F0:
    def __init__(
        self,
        n_vars: int,
        v_min,
        v_max,
        t_max: int = 100,        # last valid time index of the signal (T - 1)
        depth_max: int = 3,
        p_base: float = 0.05,        # base chance to stop at depth 0
        only_temporal: bool = False, # if true, discard purely boolean formulae
        until_weight: float = 0.05,
        not_weight: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.n_vars = n_vars
        self.v_min = v_min
        self.v_max = v_max
        self.t_max = t_max
        self.depth_max = depth_max
        self.p_base = p_base
        self.only_temporal = only_temporal
        self.not_weight = not_weight
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

    @staticmethod
    def _split_budget(remaining_time: int) -> tuple[int, int]:
        """Split a time budget between two siblings whose depths add up."""
        left = random.randint(0, max(0, remaining_time))
        return left, max(0, remaining_time) - left

    def _sample_atomic_predicate(self):
        var_idx = random.randint(0, self.n_vars - 1)
        v_low = self.v_min[var_idx].item()
        v_high = self.v_max[var_idx].item()

        threshold = random.uniform(v_low, v_high)
        lte = random.choice([True, False])

        return stl.Atom(var_index=var_idx, threshold=threshold, lte=lte)

    def _sample_operator_node(self, remaining_time: int, current_depth: int):
        classes = ["And", "Or", "Not", "Globally", "Eventually", "Until"]
        weights = [1.0, 1.0, self.not_weight, 1.0, 1.0, self.until_weight]
        op = random.choices(classes, weights=weights, k=1)[0]

        if op in ("Globally", "Eventually", "Until"):
            variants = ["unbound"]
            if remaining_time > 2:
                variants += ["bounded", "right_unbound"]
            variant = random.choice(variants)

        if op == "And":
            left = self._sample_formula(remaining_time, current_depth + 1)
            right = self._sample_formula(remaining_time, current_depth + 1)
            return stl.And(left, right)

        elif op == "Or":
            left = self._sample_formula(remaining_time, current_depth + 1)
            right = self._sample_formula(remaining_time, current_depth + 1)
            return stl.Or(left, right)

        elif op == "Not":
            return stl.Not(self._sample_formula(remaining_time, current_depth + 1))

        elif op in ("Globally", "Eventually"):
            OpClass = stl.Globally if op == "Globally" else stl.Eventually
            if variant == "unbound":
                child = self._sample_formula(remaining_time, current_depth + 1)
                return OpClass(child, unbound=True)
            elif variant == "bounded":
                b = random.randint(1, remaining_time)
                a = random.randint(0, b - 1)
                child = self._sample_formula(remaining_time - b, current_depth + 1)
                return OpClass(child, unbound=False, left_time_bound=a, right_time_bound=b)
            else:  # right_unbound
                a = random.randint(0, remaining_time)
                child = self._sample_formula(remaining_time - a, current_depth + 1)
                return OpClass(child, right_unbound=True, left_time_bound=a)

        else:  # Until
            # Until's time_depth SUMS its children's depths (unlike And/Or, which
            # take the max), so the two children must split the remaining budget
            # rather than each receiving all of it -- otherwise nested Untils can
            # demand more signal than the trace provides.
            if variant == "unbound":
                left_budget, right_budget = self._split_budget(remaining_time)
                left = self._sample_formula(left_budget, current_depth + 1)
                right = self._sample_formula(right_budget, current_depth + 1)
                return stl.Until(left, right, unbound=True)
            elif variant == "bounded":
                b = random.randint(1, remaining_time)
                a = random.randint(0, b - 1)
                left_budget, right_budget = self._split_budget(remaining_time - b)
                left = self._sample_formula(left_budget, current_depth + 1)
                right = self._sample_formula(right_budget, current_depth + 1)
                return stl.Until(left, right, unbound=False, left_time_bound=a, right_time_bound=b)
            else:  # right_unbound
                a = random.randint(0, remaining_time)
                left_budget, right_budget = self._split_budget(remaining_time - a)
                left = self._sample_formula(left_budget, current_depth + 1)
                right = self._sample_formula(right_budget, current_depth + 1)
                return stl.Until(left, right, right_unbound=True, left_time_bound=a)

    @staticmethod
    def formula_depth(formula) -> int:
        """Max nesting depth of a formula tree (0 for a bare atom)."""
        if isinstance(formula, stl.Atom):
            return 0
        if isinstance(formula, stl.Not):
            return 1 + F0.formula_depth(formula.child)
        if isinstance(formula, (stl.Globally, stl.Eventually)):
            return 1 + F0.formula_depth(formula.child)
        if isinstance(formula, (stl.And, stl.Or, stl.Until)):
            return 1 + max(F0.formula_depth(formula.left_child), F0.formula_depth(formula.right_child))
        raise TypeError(f"Unknown formula node type: {type(formula)}")

    @staticmethod
    def formula_size(formula) -> int:
        """Total node count of a formula tree (1 for a bare atom)."""
        if isinstance(formula, stl.Atom):
            return 1
        if isinstance(formula, stl.Not):
            return 1 + F0.formula_size(formula.child)
        if isinstance(formula, (stl.Globally, stl.Eventually)):
            return 1 + F0.formula_size(formula.child)
        if isinstance(formula, (stl.And, stl.Or, stl.Until)):
            return 1 + F0.formula_size(formula.left_child) + F0.formula_size(formula.right_child)
        raise TypeError(f"Unknown formula node type: {type(formula)}")

    def is_temporal(self, formula):
        """Recursive check for whether a formula contains any temporal operators."""
        if isinstance(formula, (stl.Globally, stl.Eventually, stl.Until)):
            return True
        if isinstance(formula, stl.Not):
            return self.is_temporal(formula.child)
        if isinstance(formula, (stl.And, stl.Or)):
            return self.is_temporal(formula.left_child) or self.is_temporal(formula.right_child)
        return False  # It's an atom

    def sample(self, n_formulae: int) -> list:
        """Returns a list of n_formulae sampled formulas."""
        # t_max is the last valid time index, so the largest bound a formula may
        # reference is t_max itself; the budget is that index, not t_max - 1.
        initial_time = self.t_max
        sampled_formulae = []
        while len(sampled_formulae) < n_formulae:
            formula = self._sample_formula(initial_time, current_depth=0)
            if self.only_temporal and not self.is_temporal(formula):
                continue  # Discard purely boolean formulae
            sampled_formulae.append(formula)

        return sampled_formulae
