"""
Random STL formula grammar sampler.
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
        t_max: int,        # last valid time index of the signal (T - 1)
        depth_max: int,
        until_weight: float,
        seed: Optional[int],
    ):
        self.n_vars = n_vars
        self.v_min = v_min
        self.v_max = v_max
        self.t_max = t_max
        self.depth_max = depth_max
        self.until_weight = until_weight

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def _sample_target_depth(self) -> int:
        """Per-formula depth, uniform in 1..depth_max."""
        return random.randint(1, self.depth_max)

    def _sample_formula(self, remaining_time: int, current_depth: int, target_depth: int, must_reach: bool):
        """
            must_reach: this branch has to grow to exactly target_depth, so that every
            formula has the depth it was assigned. Other branches stop with probability
            growing linearly from 0 to 1 at target_depth.
        """
        if current_depth >= target_depth:
            return self._sample_atomic_predicate()
        if not must_reach and random.random() < current_depth / target_depth:
            return self._sample_atomic_predicate()
        return self._sample_operator_node(remaining_time, current_depth, target_depth, must_reach)

    @staticmethod
    def _sample_interval(remaining_time: int) -> tuple[int, int]:
        """
            Bounded interval [a, b] with b <= remaining_time. The width b - a is
            log-uniform in [1, remaining_time], so narrow windows are common and
            every time scale is equally represented; the offset a is then uniform.
        """
        width = int((remaining_time + 1) ** random.random())  # in [1, remaining_time]
        a = random.randint(0, remaining_time - width)
        return a, a + width

    def _sample_atomic_predicate(self):
        var_idx = random.randint(0, self.n_vars - 1)
        v_low = self.v_min[var_idx].item()
        v_high = self.v_max[var_idx].item()

        threshold = random.uniform(v_low, v_high)
        lte = random.choice([True, False])

        return stl.Atom(var_index=var_idx, threshold=threshold, lte=lte)

    def _sample_operator_node(
        self,
        remaining_time: int,
        current_depth: int,
        target_depth: int,
        must_reach: bool,
        temporal_only: bool = False,
        allow_not: bool = True,
    ):
        if temporal_only:
            classes = ["Globally", "Eventually", "Until"]
            weights = [1.0, 1.0, self.until_weight]
        else:
            classes = ["And", "Or", "Globally", "Eventually", "Until"]
            weights = [1.0, 1.0, 1.0, 1.0, self.until_weight]
            # Not must wrap an operator other than Not: Not(atom) is the atom with lte
            # flipped and Not(Not(phi)) is phi. So its child must still have room for
            # its own children (depth current_depth + 2).
            if allow_not and current_depth + 2 <= target_depth:
                classes.append("Not")
                weights.append(1.0)
        op = random.choices(classes, weights=weights, k=1)[0]

        if op in ("Globally", "Eventually", "Until"):
            # Bounded windows are favoured (3:1:1) since unbound and right_unbound
            # ones always extend to the end of the trace.
            variants, variant_weights = ["unbound"], [1.0]
            if remaining_time > 0:
                variants += ["bounded", "right_unbound"]
                variant_weights += [3.0, 1.0]
            variant = random.choices(variants, weights=variant_weights, k=1)[0]

        # Of two children, a random one inherits the obligation to reach target_depth.
        reach_left = random.random() < 0.5
        d = current_depth + 1

        def unary(budget):
            return self._sample_formula(budget, d, target_depth, must_reach)

        def binary(left_budget, right_budget):
            left = self._sample_formula(left_budget, d, target_depth, must_reach and reach_left)
            right = self._sample_formula(right_budget, d, target_depth, must_reach and not reach_left)
            return left, right

        if op == "And":
            left, right = binary(remaining_time, remaining_time)
            return stl.And(left, right)

        elif op == "Or":
            left, right = binary(remaining_time, remaining_time)
            return stl.Or(left, right)

        elif op == "Not":
            return stl.Not(self._sample_operator_node(remaining_time, d, target_depth, must_reach, allow_not=False))

        elif op in ("Globally", "Eventually"):
            OpClass = stl.Globally if op == "Globally" else stl.Eventually
            if variant == "unbound":
                child = unary(remaining_time)
                return OpClass(child, unbound=True)
            elif variant == "bounded":
                a, b = self._sample_interval(remaining_time)
                child = unary(remaining_time - b)
                return OpClass(child, unbound=False, left_time_bound=a, right_time_bound=b)
            else:  # right_unbound
                a = random.randint(0, remaining_time)
                child = unary(remaining_time - a)
                return OpClass(child, right_unbound=True, left_time_bound=a)

        else:  # Until
            # Until's time_depth is the MAX of its children's depths plus its own
            # bound (as for And/Or), so both children get the whole remaining budget.
            if variant == "unbound":
                left, right = binary(remaining_time, remaining_time)
                return stl.Until(left, right, unbound=True)
            elif variant == "bounded":
                a, b = self._sample_interval(remaining_time)
                left, right = binary(remaining_time - b, remaining_time - b)
                return stl.Until(left, right, unbound=False, left_time_bound=a, right_time_bound=b)
            else:  # right_unbound
                a = random.randint(0, remaining_time)
                left, right = binary(remaining_time - a, remaining_time - a)
                return stl.Until(left, right, right_unbound=True, left_time_bound=a)

    @staticmethod
    def formula_depth(formula) -> int:
        """
            Max nesting depth of a formula tree (0 for a bare atom).
        """
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
        """
            Total node count of a formula tree (1 for a bare atom).
        """
        if isinstance(formula, stl.Atom):
            return 1
        if isinstance(formula, stl.Not):
            return 1 + F0.formula_size(formula.child)
        if isinstance(formula, (stl.Globally, stl.Eventually)):
            return 1 + F0.formula_size(formula.child)
        if isinstance(formula, (stl.And, stl.Or, stl.Until)):
            return 1 + F0.formula_size(formula.left_child) + F0.formula_size(formula.right_child)
        raise TypeError(f"Unknown formula node type: {type(formula)}")

    def sample(self, n_formulae: int) -> list:
        """
            Returns a list of n_formulae sampled formulas.
        """
        # t_max is the last valid time index, so the largest bound a formula may
        # reference is t_max itself; the budget is that index, not t_max - 1.
        initial_time = self.t_max
        # The root is always a temporal operator, so every formula is temporal
        # by construction (no rejection sampling of purely boolean formulae).
        return [
            self._sample_operator_node(
                initial_time,
                current_depth=0,
                target_depth=self._sample_target_depth(),
                must_reach=True,
                temporal_only=True,
            )
            for _ in range(n_formulae)
        ]
