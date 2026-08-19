"""STL formula representation and quantitative (robustness) semantics.

Vendored and trimmed from stlcg++ (https://github.com/UW-CTRL/stlcg-plus-plus,
PyPI package `stlcgpp`), by Karen Leung / UW-CTRL. That repository has no
explicit LICENSE file (all-rights-reserved by default) -- this is used here
for internal/research purposes only; consider forking the upstream repo
under your own account and requesting an explicit license grant.

Only the classes STLRocket actually needs are kept: Predicate-based formulas
(And, Or, Negation, LessThan, GreaterThan, Always, Eventually, Until). The
Expression-based formula path, Equal/Implies, Identity, the recurrent
(*Recurrent) operators and the Differentiable* soft-interval operators are
dropped as unused.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# maxish / minish -- exact ("true") and smooth (softmax / logsumexp) reductions
# ---------------------------------------------------------------------------

def maxish_softmax(signal, dim=0, keepdim=True, temperature=1.0, **kwargs):
    return (torch.nn.functional.softmax(temperature * signal, dim=dim) * signal).sum(
        dim=dim, keepdim=keepdim
    )


def minish_softmax(signal, dim=0, keepdim=True, temperature=1.0, **kwargs):
    return -maxish_softmax(-signal, dim=dim, keepdim=keepdim, temperature=temperature)


def maxish_logsumexp(signal, dim=0, keepdim=True, temperature=1.0, **kwargs):
    return torch.logsumexp(temperature * signal, dim=dim, keepdim=keepdim) / temperature


def minish_logsumexp(signal, dim=0, keepdim=True, temperature=1.0, **kwargs):
    return -maxish_logsumexp(-signal, dim=dim, keepdim=keepdim, temperature=temperature)


def maxish(signal, dim=0, keepdim=True, approx_method="true", temperature=1.0, **kwargs):
    match approx_method:
        case "true":
            return torch.max(signal, dim=dim, keepdim=keepdim)[0]
        case "softmax":
            return maxish_softmax(signal, dim=dim, keepdim=keepdim, temperature=temperature)
        case "logsumexp":
            return maxish_logsumexp(signal, dim=dim, keepdim=keepdim, temperature=temperature)


def minish(signal, dim=0, keepdim=True, approx_method="true", temperature=1.0, **kwargs):
    return -maxish(-signal, dim=dim, keepdim=keepdim, approx_method=approx_method, temperature=temperature)


def cond(pred, true_fun, false_fun, *operands):
    return true_fun(*operands) if pred else false_fun(*operands)


def separate_and(formula, signal, **kwargs):
    if formula.__class__.__name__ != "And":
        return formula(signal, **kwargs).unsqueeze(-1)
    return torch.cat(
        [separate_and(formula.subformula1, signal, **kwargs),
         separate_and(formula.subformula2, signal, **kwargs)],
        dim=-1,
    )


def separate_or(formula, signal, **kwargs):
    if formula.__class__.__name__ != "Or":
        return formula(signal, **kwargs).unsqueeze(-1)
    return torch.cat(
        [separate_or(formula.subformula1, signal, **kwargs),
         separate_or(formula.subformula2, signal, **kwargs)],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Predicate -- extracts a scalar time-series from the raw multi-variable signal
# ---------------------------------------------------------------------------

class Predicate(torch.nn.Module):
    def __init__(self, name, predicate_function):
        super().__init__()
        self.name = name
        self.predicate_function = predicate_function

    def forward(self, signal: torch.Tensor):
        return self.predicate_function(signal)

    def __lt__(self, rhs):
        return LessThan(self, rhs)

    def __gt__(self, rhs):
        return GreaterThan(self, rhs)

    def __str__(self):
        return str(self.name)


# ---------------------------------------------------------------------------
# STL formula base class
# ---------------------------------------------------------------------------

class STLFormula(torch.nn.Module):
    """Signal expected shape: [time_dim, state_dim]."""

    def robustness_trace(self, signal: torch.Tensor, **kwargs):
        raise NotImplementedError

    def robustness(self, signal: torch.Tensor, **kwargs):
        return self.forward(signal, **kwargs)[0]

    def forward(self, signal: torch.Tensor, **kwargs):
        return self.robustness_trace(signal, **kwargs)

    def __and__(self, psi):
        return And(self, psi)

    def __or__(self, psi):
        return Or(self, psi)

    def __invert__(self):
        return Negation(self)


class LessThan(STLFormula):
    def __init__(self, lhs: Predicate, rhs: float):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal, **kwargs):
        return self.rhs - self.lhs(signal)

    def __str__(self):
        return f"{self.lhs.name} < {self.rhs}"


class GreaterThan(STLFormula):
    def __init__(self, lhs: Predicate, rhs: float):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def robustness_trace(self, signal, **kwargs):
        return self.lhs(signal) - self.rhs

    def __str__(self):
        return f"{self.lhs.name} > {self.rhs}"


class Negation(STLFormula):
    def __init__(self, subformula: STLFormula):
        super().__init__()
        self.subformula = subformula

    def robustness_trace(self, signal, **kwargs):
        return -self.subformula(signal, **kwargs)

    def __str__(self):
        return f"¬({self.subformula})"


class And(STLFormula):
    def __init__(self, subformula1: STLFormula, subformula2: STLFormula):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, signal, **kwargs):
        xx = separate_and(self, signal, **kwargs)
        return minish(xx, dim=-1, keepdim=False, **kwargs)

    def __str__(self):
        return f"({self.subformula1}) ∧ ({self.subformula2})"


class Or(STLFormula):
    def __init__(self, subformula1: STLFormula, subformula2: STLFormula):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, signal, **kwargs):
        xx = separate_or(self, signal, **kwargs)
        return maxish(xx, dim=-1, keepdim=False, **kwargs)

    def __str__(self):
        return f"({self.subformula1}) ∨ ({self.subformula2})"


# ---------------------------------------------------------------------------
# Temporal operators -- interval=[a, b] time-step bounds; b may be torch.inf
# (adapts to the actual runtime signal length, i.e. the "right-unbound" case)
# ---------------------------------------------------------------------------

class Eventually(STLFormula):
    def __init__(self, subformula: STLFormula, interval=None):
        super().__init__()
        self.interval = interval
        self.subformula = subformula
        self._interval = [0, torch.inf] if interval is None else interval

    def robustness_trace(self, signal, padding=None, large_number=1e9, **kwargs):
        device = signal.device
        time_dim = 0
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = -large_number

        def true_func(_interval, T):
            return [_interval[0], T - 1], -1.0, _interval[0]

        def false_func(_interval, T):
            return _interval, 1.0, 0

        interval, _, offset = cond(self._interval[1] == torch.inf, true_func, false_func, self._interval, T)

        dtype = signal.dtype
        signal_matrix = signal.reshape([T, 1]) @ torch.ones([1, T], device=device, dtype=dtype)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = -large_number

        signal_pad = torch.ones([interval[1] + 1, T], device=device, dtype=dtype) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        subsignal_mask = torch.tril(torch.ones([T + interval[1] + 1, T], device=device, dtype=dtype))
        time_interval_mask = torch.triu(
            torch.ones([T + interval[1] + 1, T], device=device, dtype=dtype), -interval[-1] - offset
        ) * torch.tril(torch.ones([T + interval[1] + 1, T], device=device, dtype=dtype), -interval[0])
        masked_signal_matrix = torch.where(
            (time_interval_mask * subsignal_mask) == 1.0, signal_padded, mask_value
        )
        return maxish(masked_signal_matrix, dim=time_dim, keepdim=False, **kwargs)

    def __str__(self):
        return f"♢ {self._interval}( {self.subformula} )"


class Always(STLFormula):
    def __init__(self, subformula: STLFormula, interval=None):
        super().__init__()
        self.interval = interval
        self.subformula = subformula
        self._interval = [0, torch.inf] if interval is None else interval

    def robustness_trace(self, signal, padding=None, large_number=1e9, **kwargs):
        device = signal.device
        time_dim = 0
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = large_number

        def true_func(_interval, T):
            return [_interval[0], T - 1], -1.0, _interval[0]

        def false_func(_interval, T):
            return _interval, 1.0, 0

        interval, sign, offset = cond(self._interval[1] == torch.inf, true_func, false_func, self._interval, T)

        dtype = signal.dtype
        signal_matrix = signal.reshape([T, 1]) @ torch.ones([1, T], device=device, dtype=dtype)
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = -large_number

        signal_pad = torch.cat(
            [
                torch.ones([interval[1], T], device=device, dtype=dtype) * sign * pad_value,
                torch.ones([1, T], device=device, dtype=dtype) * pad_value,
            ],
            dim=time_dim,
        )
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)
        subsignal_mask = torch.tril(torch.ones([T + interval[1] + 1, T], device=device, dtype=dtype))
        time_interval_mask = torch.triu(
            torch.ones([T + interval[1] + 1, T], device=device, dtype=dtype), -interval[-1] - offset
        ) * torch.tril(torch.ones([T + interval[1] + 1, T], device=device, dtype=dtype), -interval[0])
        masked_signal_matrix = torch.where(
            (time_interval_mask * subsignal_mask) == 1.0, signal_padded, mask_value
        )
        return minish(masked_signal_matrix, dim=time_dim, keepdim=False, **kwargs)

    def __str__(self):
        return f"◻ {self._interval}( {self.subformula} )"


class Until(STLFormula):
    def __init__(self, subformula1: STLFormula, subformula2: STLFormula, interval=None):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.interval = interval
        self._interval = [0, torch.inf] if interval is None else interval

    def robustness_trace(self, signal, padding=None, large_number=1e9, **kwargs):
        device = signal.device
        time_dim = 0
        signal1 = self.subformula1(signal, padding=padding, large_number=large_number, **kwargs)
        signal2 = self.subformula2(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]

        mask_value = large_number
        if self.interval is None:
            interval = [0, T - 1]
        elif self.interval[1] == torch.inf:
            interval = [self.interval[0], T - 1]
        else:
            interval = self.interval

        signal1_matrix = signal1.unsqueeze(1).expand(-1, T)
        signal2_matrix = signal2.unsqueeze(1).expand(-1, T)

        if padding == "last":
            pad_value1 = signal1[-1]
            pad_value2 = signal2[-1]
        elif padding == "mean":
            pad_value1 = signal1.mean(dim=time_dim)
            pad_value2 = signal2.mean(dim=time_dim)
        else:
            pad_value1 = torch.tensor(-mask_value, device=device, dtype=signal1.dtype)
            pad_value2 = torch.tensor(-mask_value, device=device, dtype=signal2.dtype)

        signal1_pad = pad_value1.view(1, 1).expand(interval[1] + 1, T)
        signal2_pad = pad_value2.view(1, 1).expand(interval[1] + 1, T)
        signal1_padded = torch.cat([signal1_matrix, signal1_pad], dim=time_dim)
        signal2_padded = torch.cat([signal2_matrix, signal2_pad], dim=time_dim)

        rows = torch.arange(T + interval[1] + 1, device=device).view(-1, 1)
        cols = torch.arange(T, device=device).view(1, -1)

        phi1_mask = torch.stack(
            [((cols - rows >= -end_idx) & (cols - rows <= 0)) for end_idx in range(interval[0], interval[-1] + 1)],
            dim=0,
        )
        phi2_mask = torch.stack(
            [((cols - rows >= -end_idx) & (cols - rows <= -end_idx)) for end_idx in range(interval[0], interval[-1] + 1)],
            dim=0,
        )

        signal1_batched = signal1_padded.unsqueeze(0)
        signal2_batched = signal2_padded.unsqueeze(0)

        phi1_masked_signal = torch.where(phi1_mask, signal1_batched, mask_value)
        phi2_masked_signal = torch.where(phi2_mask, signal2_batched, mask_value)

        return maxish(
            torch.stack(
                [
                    minish(
                        torch.stack(
                            [
                                minish(s1, dim=0, keepdim=False, **kwargs),
                                minish(s2, dim=0, keepdim=False, **kwargs),
                            ],
                            dim=0,
                        ),
                        dim=0,
                        keepdim=False,
                        **kwargs,
                    )
                    for (s1, s2) in zip(phi1_masked_signal, phi2_masked_signal)
                ],
                dim=0,
            ),
            dim=0,
            keepdim=False,
            **kwargs,
        )

    def __str__(self):
        return f"({self.subformula1}) U {self._interval}({self.subformula2})"


# ---------------------------------------------------------------------------
# STLRocket's own convenience factory (not part of stlcg++)
# ---------------------------------------------------------------------------

def atom(var_index: int, threshold: float, lte: bool = False) -> LessThan | GreaterThan:
    """A leaf STL predicate on one variable of a [T, V]-shaped signal.

    lte=True  -> x_i <= threshold (LessThan)
    lte=False -> x_i >= threshold (GreaterThan)
    """
    pred = Predicate(f"x{var_index}", lambda signal, i=var_index: signal[..., i])
    return LessThan(pred, threshold) if lte else GreaterThan(pred, threshold)
