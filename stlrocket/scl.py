"""
Signal Convolution Logic (Silvetti, Nenzi, Bortolussi, Bartocci, ATVA 2018) operator
with a flat kernel, as a torcheck node.

Fraction(phi, p) over a window holds when phi holds on at least a fraction p of the
window's steps: p = 1 is Globally, "more than 0" is Eventually. Window conventions
(slicing at left_time_bound, right_time_bound stored +1, adapt_unbound suffixes) are
those of torcheck.stl.Globally, so time_depth and trace lengths match.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torcheck.stl import Node


class Fraction(Node):
    """
        Quantitative semantics: frac(t) - p, where frac(t) is the share of window steps
        where the child's robustness is > 0. It is a step function of the child's
        thresholds (not smooth) and affine in p, so p behaves like an atom threshold.
        Boolean semantics: frac(t) >= p on the child's boolean trace.
    """

    def __init__(
            self,
            child: Node,
            p: float,
            unbound: bool = False,
            right_unbound: bool = False,
            left_time_bound: int = 0,
            right_time_bound: int = 1,
            adapt_unbound: bool = True,
    ) -> None:
        super().__init__()
        self.child: Node = child
        self.p: float = p
        self.unbound: bool = unbound
        self.right_unbound: bool = right_unbound
        self.left_time_bound: int = left_time_bound
        self.right_time_bound: int = right_time_bound + 1
        self.adapt_unbound: bool = adapt_unbound

        if (self.unbound is False) and (self.right_unbound is False) and \
                (self.right_time_bound <= self.left_time_bound):
            raise ValueError("Temporal thresholds are incorrect: right parameter is higher than left parameter")

    def __str__(self) -> str:
        s_left = "[" + str(self.left_time_bound) + ","
        s_right = str(self.right_time_bound - 1) if not self.right_unbound else "inf"
        s0: str = s_left + s_right + "]" if not self.unbound else ""
        return f"frac{s0}>={self.p:.2f} ( {self.child} )"

    def time_depth(self) -> int:
        if self.unbound:
            return self.child.time_depth()
        elif self.right_unbound:
            return self.child.time_depth() + self.left_time_bound
        else:
            return self.child.time_depth() + self.right_time_bound - 1

    def _window_fraction(self, s: Tensor) -> Tensor:
        """Share of ones of the (N, 1, T) 0/1 trace s over the window at every t."""
        if self.unbound or self.right_unbound:
            if self.adapt_unbound:
                # Mean over the suffix [t, end], as Globally's cummin over the flipped trace.
                suffix_sum = torch.flip(torch.cumsum(torch.flip(s, [2]), dim=2), [2])
                counts = torch.arange(s.shape[2], 0, -1, device=s.device, dtype=s.dtype)
                return suffix_sum / counts
            return s.mean(dim=2, keepdim=True)
        return F.avg_pool1d(s, kernel_size=self.right_time_bound - self.left_time_bound, stride=1)

    def _quantitative(self, x: Tensor, normalize: bool = False) -> Tensor:
        z1: Tensor = self.child._quantitative(x[:, :, self.left_time_bound:], normalize)
        return self._window_fraction((z1 > 0).to(z1.dtype)) - self.p

    def _boolean(self, x: Tensor) -> Tensor:
        z1: Tensor = self.child._boolean(x[:, :, self.left_time_bound:])
        return self._window_fraction(z1.float()) >= self.p
