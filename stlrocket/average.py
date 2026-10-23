"""
Average temporal operator, restricted to sit directly above an atom: Average[a,b](x ~ c),
whose robustness at t is the mean robustness of the atom over [t+a, t+b], i.e.
mean(x[t+a..t+b]) - c (or c - mean for x <= c).

Over a general child the sign of a mean robustness is not the satisfaction of any STL
formula. Over an atom it is: Average[a,b](x > c) holds iff the moving average of x is > c,
an atom on a filtered signal. So the node is an Atom subclass, a leaf predicate, and every
formula above it keeps its Boolean STL reading: isinstance(node, Atom) code (threshold
shifts, calibration, Not as a flip of lte) applies unchanged, since robustness is affine in
the threshold with the same sign as for the atom. It does not count as a depth level.

Window conventions (slicing at left_time_bound, right_time_bound stored +1) are those of
torcheck.stl.Globally, so time_depth and trace lengths match a bounded G over the atom.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor
from torcheck.stl import Atom


class Average(Atom):
    """Moving-average atom; bounded windows only."""

    # Read by code that handles every temporal node through these flags (pretty.window).
    unbound: bool = False
    right_unbound: bool = False

    def __init__(self, var_index: int, threshold: float, lte: bool = False,
                 left_time_bound: int = 0, right_time_bound: int = 1) -> None:
        super().__init__(var_index, threshold, lte)
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        if self.right_time_bound <= self.left_time_bound:
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        return f"avg[{self.left_time_bound},{self.right_time_bound - 1}] ( {super().__str__()} )"

    def time_depth(self) -> int:
        return self.right_time_bound - 1

    def _moving_average(self, x: Tensor) -> Tensor:
        xj: Tensor = x[:, self.var_index:self.var_index + 1, self.left_time_bound:]
        return F.avg_pool1d(xj, kernel_size=self.right_time_bound - self.left_time_bound, stride=1)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        m: Tensor = self._moving_average(x)
        return (self.threshold - m) if self.lte else (m - self.threshold)

    def _boolean(self, x: Tensor) -> Tensor:
        return self._quantitative(x) >= 0
