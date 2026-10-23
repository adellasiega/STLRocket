"""
Cross-channel atom: a linear predicate on two variables, a*x_i + b*x_j >= c (or <= c), whose
robustness is a*x_i + b*x_j - c (or c - a*x_i - b*x_j).

Relations between channels (colour differences between passbands in LSST, relative energy
between frequency bands in PhonemeSpectra) are a single linear predicate, while
single-variable atoms can only approximate a diagonal boundary with And/Or of two thresholds.
Linear predicates are standard STL atoms, and robustness is affine in the threshold with the
same sign as for a single-variable atom, so the node is an Atom subclass and isinstance(node,
Atom) code (threshold shifts, calibration, Not as a flip of lte) applies unchanged. var_index
is the first variable.
"""
from __future__ import annotations

from torch import Tensor
from torcheck.stl import Atom


class LinearAtom(Atom):
    """a*x_i + b*x_j against threshold; the sampler draws (a, b) on the unit circle with a > 0."""

    def __init__(self, var_index: int, var_index2: int, a: float, b: float, threshold: float,
                 lte: bool = False) -> None:
        super().__init__(var_index, threshold, lte)
        self.var_index2: int = var_index2
        self.a: float = a
        self.b: float = b

    def __str__(self) -> str:
        sign = "-" if self.b < 0 else "+"
        return (f"{self.a:.3f}*x_{self.var_index} {sign} {abs(self.b):.3f}*x_{self.var_index2}"
                f"{' <= ' if self.lte else ' >= '}{round(self.threshold, 4)}")

    def _combination(self, x: Tensor) -> Tensor:
        return self.a * x[:, self.var_index:self.var_index + 1, :] + self.b * x[:, self.var_index2:self.var_index2 + 1, :]

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z: Tensor = self._combination(x)
        return (self.threshold - z) if self.lte else (z - self.threshold)

    def _boolean(self, x: Tensor) -> Tensor:
        return self._quantitative(x) >= 0
