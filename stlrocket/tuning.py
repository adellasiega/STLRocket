"""Gradient-based fine-tuning of STL formula parameters.

torcheck's quantitative semantics is already differentiable torch code, but the
formulas STLRocket reports get their parameters from two unoptimized sources:
thresholds from the median heuristic in ``explanations.reparametrize_formula``,
and time intervals straight from the random sampler (never touched at all).
This module refines both by gradient descent on a differentiable surrogate of
F1.

Two smoothing approximations make that possible, both taken from STLCG++
(Kapoor et al., 2025):

* ``min``/``max`` are replaced by log-sum-exp.  Not softmax: LSE composes under
  nesting (its error is ``log(k)/beta`` per level, additive down the tree),
  whereas a softmax-weighted average is not idempotent and its bias compounds
  multiplicatively, so a deep formula's soft robustness drifts far from the
  hard value.  This is the paper's Sec. IV-D point.
* Time intervals are replaced by the smooth mask of the paper's eq. 6, so the
  window bounds become continuous parameters.

Only the *smoothing* is borrowed.  The paper's masking approach to evaluating
robustness -- unrolling the signal into a (T+K)xT array -- is O(N*T^2) and is
deliberately not used here.  The outermost temporal operator reduces over the
time axis once, in O(N*T), since only its value at t=0 is ever read.  A nested
operator does need a robustness trace, but only over a window of bounded width,
so it costs O(N*T*span) with span capped independently of the signal length --
which is what makes tuning nested time intervals affordable on long signals.

The smooth semantics exist solely to produce gradients.  Every number this
module reports, including the accept/reject decision, is computed with
torcheck's exact hard semantics via ``features.eval_robustness``.
"""

from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

from torcheck.stl import Atom, Not, And, Or, Globally, Eventually, Until

from .features import eval_robustness

_CHILD_ATTRS = ("child", "left_child", "right_child")

# Each smoothing temperature is expressed in the units of some quantity in the
# data, so all three are derived from the data rather than passed in; see
# _default_temps.  The constants below are the dimensionless ratios.

# Mask steepness, in sigmoid widths per timestep.  Both ends are failure modes
# and the usable band is narrow: too soft and a window is smeared across far
# more timesteps than it names, so the surrogate ranks windows differently from
# the true semantics and the optimizer converges neatly to the wrong answer;
# too sharp and the mask saturates into a hard rectangle, d(mask)/da is zero
# away from the edges, and the interval parameters stop learning entirely.
# Expressed relative to T: the transition should span a small number of steps.
_MASK_STEEPNESS_STEPS = 2.5

# Acceptance sigmoid width, as a fraction of the robustness IQR.  Wide enough
# to give gradient across the bulk of the data, narrow enough that it does not
# average over the overlap between classes and stop telling good windows from
# bad ones.
_ACCEPT_TEMP_IQR_FRAC = 0.25

# LSE temperature, likewise as a fraction of the robustness IQR.  The sharpness
# actually used is its reciprocal.  Small: LSE's error is log(k)/beta *per
# nesting level*, so it accumulates down a tree, and real sampled explanations
# run to depth 5-6 where a loose value leaves the surrogate far enough from the
# true semantics that tuning degrades the formula and every candidate is
# rejected.  Toy formulas of depth 1-2 are far more forgiving and give no
# warning of this.
_LSE_TEMP_IQR_FRAC = 0.005

# Narrowest window the optimizer may produce, in timesteps.  Purely a numerical
# guard: below roughly one step the smooth mask loses all support, m.sum()
# underflows to exactly zero, and the masked reduction returns a value with no
# relation to the formula.  This is a clamp on the width rather than a penalty
# in the loss -- F1 already penalises a genuinely bad narrow window, so a loss
# term would only add a coefficient to tune.
_MIN_WINDOW_STEPS = 1.0

# Smallest span a nested window is given to move within, so that even a window
# sampled one step wide has somewhere to grow into.
_MIN_NESTED_SPAN = 8

# Widest window a *nested* temporal operator may be tuned to, in timesteps.
# A nested window has to be applied at every offset (see _reduce_trace), which
# costs O(N*T*span), so the span is capped to keep that affordable on long
# signals -- the cap, not the signal length, bounds the cost.  Nested windows
# are typically short anyway ("stays high for a few steps"); one wider than
# this keeps its sampled bounds and only its thresholds are tuned.
_MAX_NESTED_SPAN = 16


# --------------------------------------------------------------------------
# Smooth min / max
# --------------------------------------------------------------------------

def soft_max(z: torch.Tensor, dim: int, beta: float, keepdim: bool = False) -> torch.Tensor:
    """log-sum-exp approximation of ``max``; exact as ``beta -> inf``."""
    return torch.logsumexp(beta * z, dim=dim, keepdim=keepdim) / beta


def soft_min(z: torch.Tensor, dim: int, beta: float, keepdim: bool = False) -> torch.Tensor:
    return -soft_max(-z, dim, beta, keepdim=keepdim)


# --------------------------------------------------------------------------
# Soft mirrors of the torcheck node types
#
# These subclass the originals rather than patching them: every traversal in
# the codebase (features.shift_atom_thresholds, simplification._iter_nodes,
# F0.formula_depth) dispatches on isinstance, and a subclass satisfies those
# checks.  Patching And._quantitative globally would silently make the formula
# bank's feature extraction soft too.
#
# Not needs no soft version -- rho(~phi) = -rho(phi) is already exact.
# Until is left hard: its bounded semantics has three specialised paths, and
# until_weight is 0.0 in the explanation runs, so no Until reaches this module
# in practice.  Thresholds inside an Until still tune, being plain arithmetic.
# --------------------------------------------------------------------------

class _Soft:
    """Mixin holding the annealed temperatures, shared by reference."""

    temps: dict

    @property
    def beta(self) -> float:
        """Sharpness of the LSE min/max (1 / temperature)."""
        return self.temps["beta_lse"]


class SoftAnd(And, _Soft):
    def _quantitative(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        z1 = self.left_child._quantitative(x, normalize)
        z2 = self.right_child._quantitative(x, normalize)
        size = min(z1.size(2), z2.size(2))
        z = torch.stack([z1[:, :, :size], z2[:, :, :size]], dim=-1)
        return soft_min(z, dim=-1, beta=self.beta)


class SoftOr(Or, _Soft):
    def _quantitative(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        z1 = self.left_child._quantitative(x, normalize)
        z2 = self.right_child._quantitative(x, normalize)
        size = min(z1.size(2), z2.size(2))
        z = torch.stack([z1[:, :, :size], z2[:, :, :size]], dim=-1)
        return soft_max(z, dim=-1, beta=self.beta)


class _SoftTemporal(_Soft):
    """Shared machinery for the smooth-interval Globally/Eventually.

    The interval is parametrised by a start ``a`` and a strictly positive width
    ``w = softplus(w_raw)``, with ``b = a + w``, both as fractions of the signal
    length.  Making the width structural means ``b > a`` holds by construction
    rather than by projection after each step.
    """

    a_raw: torch.Tensor
    w_raw: torch.Tensor
    tunable: bool
    min_width_frac: float = 0.0
    # A nested node keeps a robustness trace instead of collapsing to t=0, so
    # the operator enclosing it still has a time axis to slide along.
    nested: bool = False
    nested_span: int = 0
    # Full signal length, so a window whose bounds are fractions of the signal
    # still refers to the right timesteps when a nested child has shortened the
    # trace that reaches it.
    signal_len: int = 0

    def _signal_depth_for_t0(self):
        # The base classes return the *current* window's extent, and
        # Node.quantitative truncates the input signal to that many steps.  The
        # window moves during optimization -- a formula starting at F[12,14]
        # may need to reach t=3 -- so a stale bound must not decide what the
        # optimizer is allowed to see.  Returning inf disables the truncation;
        # the soft reduction below spans the whole signal anyway.
        return float("inf")

    def _bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.sigmoid(self.a_raw)
        # Width is structural (softplus, so positive) rather than a second free
        # bound, which makes b > a hold by construction instead of needing a
        # projection after every step.  The floor is a numerical guard: below
        # about one timestep the mask loses all support and the reduction stops
        # meaning anything.  It is deliberately not a loss term -- F1 already
        # penalises a window that is too narrow to be useful.
        w = torch.nn.functional.softplus(self.w_raw) + self.min_width_frac
        return a, torch.clamp(a + w, max=1.0)

    def _mask(self, T: int, dtype, device, span: int | None = None) -> torch.Tensor:
        """Smooth window mask, STLCG++ eq. 6.

        The steepness ``c`` trades gradient against fidelity, and both ends
        fail: too sharp and the mask saturates into a rectangle of exact 0s and
        1s, so d(mask)/da is zero everywhere except within ~1/c of an edge and
        the window parameters stop learning; too soft and a narrow window is
        smeared into a low bump spanning far more timesteps than it names, so
        the soft robustness no longer reflects the formula being evaluated.
        The caller anneals ``c`` between the two, soft first for gradient flow,
        sharp last for fidelity.

        ``span`` is the length the normalised bounds are measured against, and
        the length of the returned mask.  It differs from ``T`` for a nested
        window, whose bounds are a fraction of its own capped extent rather
        than of the whole signal.
        """
        a, b = self._bounds()
        n = T if span is None else span
        c = self.temps["c_mask"]
        eps = self.temps["mask_eps"]
        i = torch.arange(n, dtype=dtype, device=device)
        # STL bounds are inclusive: G[l,r] covers steps l..r, so the mask has to
        # cover the half-open interval [l, r+1).  Centring the falling edge on
        # b rather than b+1 gives step r only half weight and, once the mask
        # sharpens, drops it entirely -- the soft formula then evaluates a
        # window one step shorter than the one it names, and the error does not
        # shrink as the temperatures sharpen.  Likewise the rising edge sits a
        # half step below a so that step l gets full weight.
        lo = a * n - 0.5
        hi = b * n + 0.5
        m = torch.sigmoid(c * (i - lo)) - torch.sigmoid(c * (i - hi))
        # Subtracting eps cuts the sigmoid tails so the mask has genuinely zero
        # support outside the window, which is what makes the final rounding to
        # integer bounds well defined.  Keep it small: with a soft mask, a large
        # eps clips the informative shoulders along with the tails.
        return torch.relu(m - eps)

    @staticmethod
    def _masked_lse(z: torch.Tensor, m: torch.Tensor, dim: int, beta: float,
                    sign: float, keepdim: bool = False) -> torch.Tensor:
        """Softly reduce ``z`` along ``dim`` over the smooth window ``m``.

        ``sign`` is -1 for Globally (a min) and +1 for Eventually (a max).
        """
        # LSE approximates the true min/max with error log(K)/beta over K
        # effective terms, so at fixed beta a wider window is smoothed harder.
        # That bias is systematic and, for a Globally over a class whose
        # robustness is uniformly positive, it makes the surrogate *decrease*
        # with width while true F1 increases -- the gradient then points away
        # from the correct window.  Scaling beta by the effective width keeps
        # the approximation error constant as the window grows.
        m_max = m.max()
        k_eff = m.sum() / (m_max + 1e-12)
        b = beta * torch.clamp(k_eff, min=1.0)

        # The mask enters as an additive log-domain bias: ~0 inside the window,
        # strongly negative outside, so masked-out steps drop out of the LSE.
        # It must NOT be normalised to sum to 1 -- that would add log(1/K) for a
        # window of K steps, a width-dependent penalty on robustness that has
        # nothing to do with the data.
        #
        # A *fully* masked entry must be driven out of the LSE, and the floor
        # has to beat what it competes against, which is b * |z|.  A fixed
        # log(1e-12) = -27.6 looks large but is nothing next to b|z| once
        # b reaches the hundreds, so masked steps leak back in and the reduction
        # returns something that is not the windowed min/max at all.
        #
        # The floor is large but *finite*, scaled to the magnitudes actually in
        # play.  Literal -inf is tempting and does converge, but if the window
        # ever shrinks until no timestep survives the mask, every entry is -inf:
        # logsumexp then returns -inf forward and NaN backward, the NaN reaches
        # the parameters, and the next rounding to integer bounds dies on
        # "cannot convert float NaN to integer".  A finite floor degrades
        # gracefully instead -- an empty window just returns the softened
        # extremum of the whole axis.
        span_scale = (b * z.detach().abs().amax().clamp(min=1.0)).clamp(min=1.0)
        floor = -(span_scale * 4.0 + 64.0)
        log_mask = torch.where(
            m > 0,
            torch.log(m.clamp_min(1e-12) / (m_max + 1e-12)).clamp_min(floor),
            torch.full_like(m, 1.0) * floor,
        )

        z = torch.logsumexp(b * sign * z + log_mask, dim=dim, keepdim=keepdim) / b
        return sign * z

    def _reduce(self, z1: torch.Tensor, sign: float) -> torch.Tensor:
        """Masked LSE over the whole time axis, evaluated at t=0 only.

        Used by the outermost temporal operator, whose result
        ``quantitative(evaluate_at_all_times=False)`` reads at ``z[:, 0, 0]``.
        Collapsing here means no robustness trace is needed -- which is what
        would otherwise force the paper's O(N*T^2) unrolling.  One (N, 1, T)
        buffer and a (T,) mask suffice: O(N*T).
        """
        # The bounds are fractions of the full signal length, but the trace
        # reaching this node is shorter when a nested operator has already
        # consumed time (its sliding reduction returns T-span+1 values).  Build
        # the mask over the *signal* scale and use the part of it that lines up
        # with the trace, so the window still refers to the timesteps it names
        # rather than being silently rescaled onto whatever arrived.
        n = z1.size(2)
        m = self._mask(self.signal_len or n, z1.dtype, z1.device)[:n]
        return self._masked_lse(z1, m, dim=2, beta=self.beta, sign=sign, keepdim=True)

    def _reduce_trace(self, z1: torch.Tensor, sign: float) -> torch.Tensor:
        """Masked LSE over a sliding window, keeping the time axis.

        Used by a *nested* temporal operator, where collapsing to a single
        value would leave the enclosing window nothing to slide over.  The
        window is applied at every offset via ``unfold``, giving a trace of
        length ``T - span + 1``.

        Cost is O(N*T*span), not O(N*T^2): ``span`` is the nested window's
        capped extent (``_MAX_NESTED_SPAN``), independent of signal length.
        That is the distinction from the paper's masking approach, which
        unrolls to (T+K)xT because it supports an arbitrary window at every
        offset.  Here the nested window's *width* is bounded while its
        *position* stays free, which is what keeps this affordable on the long
        signals that ruled the full construction out.
        """
        T = z1.size(2)
        span = min(self.nested_span, T)
        # unfold only yields frames that fit entirely inside the signal, so it
        # stops span-1 steps early and the trace would be shorter than the one
        # the hard semantics produces -- the enclosing window then cannot see
        # the late offsets at all, and for any instance whose best window lies
        # there the soft value is simply wrong.  Pad with the edge value, which
        # the mask zeroes out for every window that does not reach into it, so
        # the padding never contributes to a reduction it should not.
        pad = span - 1
        if pad > 0:
            z1 = torch.cat([z1, z1[:, :, -1:].expand(-1, -1, pad)], dim=2)
        # (N, 1, T, span)
        windows = z1.unfold(2, span, 1)
        m = self._mask(T, z1.dtype, z1.device, span=span)
        out = self._masked_lse(windows, m, dim=3, beta=self.beta, sign=sign)

        # Trim to the length the hard semantics would produce, T - r for a
        # window ending at r.  The padding above restores the offsets unfold
        # drops, but leaving the trace at full length hands a parent operator
        # trailing values that do not exist in the true semantics -- harmless
        # for a bounded parent whose mask ignores them, but an unbound parent
        # reduces over the whole trace with cummin/cummax and those extra
        # entries change its result, so the soft value stops converging to the
        # hard one however sharp the temperatures get.
        with torch.no_grad():
            _, b = self._bounds()
            right = int(round(float(b) * span))
        keep = max(T - right, 1)
        return out[:, :, :keep]

    def _quantitative_soft(self, x: torch.Tensor, normalize: bool, sign: float) -> torch.Tensor:
        if not self.tunable:
            # Unbound, or nested with interval tuning switched off: keep the
            # exact semantics.  Dispatch explicitly to the torcheck base rather
            # than via super(): these classes are
            # SoftGlobally(Globally, _SoftTemporal), so from inside the mixin
            # super() resolves *past* Globally and finds no _quantitative.
            return self._hard_base._quantitative(self, x, normalize)
        z1 = self.child._quantitative(x, normalize)
        if self.nested:
            return self._reduce_trace(z1, sign)
        return self._reduce(z1, sign)


class SoftGlobally(Globally, _SoftTemporal):
    _hard_base = Globally

    def _quantitative(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        # Globally is a min over the window: -LSE(-z).
        return self._quantitative_soft(x, normalize, sign=-1.0)


class SoftEventually(Eventually, _SoftTemporal):
    _hard_base = Eventually

    def _quantitative(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        return self._quantitative_soft(x, normalize, sign=1.0)


_SOFT_TEMPORAL = (SoftGlobally, SoftEventually)


# --------------------------------------------------------------------------
# Tree conversion
# --------------------------------------------------------------------------

def _logical_right(node) -> int:
    """Undo the ``+1`` the Globally/Eventually constructor applies.

    ``__init__`` stores ``right_time_bound + 1`` and ``__str__`` prints
    ``right_time_bound - 1``, so the value a constructor expects is one less
    than the attribute.  Mishandling this shifts every window by one step with
    no error anywhere.
    """
    return node.right_time_bound - 1


def to_soft(node, temps: dict, T: int, params: list, *, under_temporal: bool = False):
    """Mirror a plain torcheck tree into one with smooth semantics.

    Interval parameters for each bounded temporal node are appended to
    ``params``.  Nodes whose bounds cannot be tuned keep the exact semantics.
    """
    if isinstance(node, Atom):
        return Atom(node.var_index, node.threshold, node.lte)

    if isinstance(node, Not):
        return Not(to_soft(node.child, temps, T, params, under_temporal=under_temporal))

    if isinstance(node, (And, Or)):
        cls = SoftAnd if isinstance(node, And) else SoftOr
        out = cls(
            to_soft(node.left_child, temps, T, params, under_temporal=under_temporal),
            to_soft(node.right_child, temps, T, params, under_temporal=under_temporal),
        )
        out.temps = temps
        return out

    if isinstance(node, Until):
        # Bounds stay fixed; thresholds underneath still tune.
        return Until(
            to_soft(node.left_child, temps, T, params, under_temporal=True),
            to_soft(node.right_child, temps, T, params, under_temporal=True),
            unbound=node.unbound,
            right_unbound=node.right_unbound,
            left_time_bound=node.left_time_bound,
            right_time_bound=_logical_right(node),
        )

    if isinstance(node, (Globally, Eventually)):
        cls = SoftGlobally if isinstance(node, Globally) else SoftEventually
        width_steps = _logical_right(node) - node.left_time_bound

        # A nested operator is tuned over a sliding window that preserves the
        # time axis, so the operator enclosing it still has something to slide
        # along.  That costs O(N*T*span), so it is offered only while the span
        # stays capped; a wider nested window keeps its sampled bounds and
        # contributes only its thresholds.  Unbound windows have no interval to
        # tune in the first place.
        nested = under_temporal
        # The span is the room the nested window has to move and grow in, not
        # the width it starts at -- sizing it to the sampled width would pin the
        # window to its initial extent and there would be nothing to tune.  It
        # must also cover the whole sampled interval, right bound included: a
        # window like G[7,9] is only 2 steps wide but reaches out to 9, and a
        # span sized from the width alone would silently clamp it to the span
        # edge, evaluating a different formula than the one it names.  Capped so
        # the O(N*T*span) sliding reduction stays affordable.
        need = _logical_right(node) + 1
        span = min(max(2 * width_steps, need, _MIN_NESTED_SPAN), _MAX_NESTED_SPAN, T - 1)
        tunable = (
            not node.unbound
            and not node.right_unbound
            # Only when the span can actually represent the sampled interval;
            # otherwise keep the exact semantics and tune thresholds only.
            and (not nested or need <= span)
        )
        out = cls(
            to_soft(node.child, temps, T, params, under_temporal=True),
            unbound=node.unbound,
            right_unbound=node.right_unbound,
            left_time_bound=node.left_time_bound,
            right_time_bound=_logical_right(node),
        )
        out.temps = temps
        out.tunable = tunable
        out.nested = nested
        out.nested_span = span
        out.signal_len = T
        # A nested window's bounds are a fraction of its own capped span, not
        # of the whole signal, so the floor and the initial width are measured
        # against that same extent.
        scale = span if nested else T
        # Half a step, not a whole one.  The floor exists only to stop the mask
        # losing all support; making it a full step means a window sampled one
        # step wide has its entire width supplied by the floor, leaving
        # softplus(w_raw) at ~0 where its gradient all but vanishes -- the width
        # then sits pinned at the rounding boundary and never reaches the next
        # integer, however long the optimizer runs.
        out.min_width_frac = 0.5 * _MIN_WINDOW_STEPS / scale
        if tunable:
            a0 = min(max(node.left_time_bound / scale, 1e-3), 1 - 1e-3)
            width0 = max(width_steps / scale, 1e-3)
            # Start the window wider than it was sampled.  The loss surface has
            # a genuine local optimum between a narrow and a wide window: a
            # narrow window can already fit some instances well, so from a
            # narrow start the gradient points at shrinking further and never
            # reaches the timesteps that actually separate the classes.
            # Starting open lets the optimizer place the window against the
            # signal and tighten from there.  This matters at least as much for
            # a nested window, whose initial width is often a single step --
            # right where softplus' gradient is smallest, so it would otherwise
            # sit pinned below the next integer however long the run.
            width0 = max(width0, temps.get("width_init", 0.0))
            a0 = min(a0, 1 - 1e-3)
            out.a_raw = torch.tensor(math.log(a0 / (1 - a0)), requires_grad=True)
            # softplus^-1, less the floor that _bounds adds back, so the window
            # actually starts at width0.
            free0 = max(width0 - out.min_width_frac, 1e-4)
            out.w_raw = torch.tensor(math.log(math.expm1(free0)), requires_grad=True)
            params.extend([out.a_raw, out.w_raw])
        return out

    raise TypeError(f"unknown node type: {type(node).__name__}")


def _int_bounds(
    a: float, b: float, T: int, min_width: int, child_depth: int = 0,
    scale: int | None = None, horizon_cap: int | None = None,
) -> tuple[int, int]:
    """Round the smooth window back to valid integer bounds.

    ``child_depth`` is the number of timesteps the subformula already consumes
    (``Node.time_depth``).  A nested operator such as ``F[a,b] G[0,2] mu`` needs
    ``b + 2`` steps of signal to evaluate at t=0, so the outer window cannot run
    to the end of the trace: torcheck would hand ``max_pool1d`` a negative
    output size and raise.  The smooth mask has no such constraint -- it spans
    the whole signal either way -- so the limit has to be imposed here, where
    the continuous window is turned back into bounds the hard semantics will
    actually evaluate.
    """
    horizon = max(T - 1 - child_depth, min_width)
    if horizon_cap is not None:
        horizon = max(min(horizon, horizon_cap), min_width)
    n = T if scale is None else scale
    # A non-finite bound means the optimization diverged.  Fall back to the full
    # window rather than raising: the caller re-scores every candidate with the
    # exact semantics and keeps the original unless it genuinely improved, so a
    # degenerate candidate is discarded a moment later.  Raising here instead
    # would take down the whole explanation run for one bad instance.
    if not (math.isfinite(a) and math.isfinite(b)):
        return 0, horizon
    left = int(round(a * n))
    right = int(round(b * n))
    left = max(0, min(left, horizon - min_width))
    right = max(left + min_width, min(right, horizon))
    return left, right


def to_hard(node, T: int, min_width: int = int(_MIN_WINDOW_STEPS)):
    """Convert a soft tree back to plain torcheck nodes with float/int params."""
    if isinstance(node, Atom):
        # A diverged threshold would otherwise be written into the formula and
        # only surface later as a nan robustness; the accept gate discards such
        # a candidate, but keep the tree well-formed in the meantime.
        thr = float(node.threshold)
        return Atom(node.var_index, thr if math.isfinite(thr) else 0.0, node.lte)

    if isinstance(node, Not):
        return Not(to_hard(node.child, T, min_width))

    if isinstance(node, Until):
        return Until(
            to_hard(node.left_child, T, min_width),
            to_hard(node.right_child, T, min_width),
            unbound=node.unbound,
            right_unbound=node.right_unbound,
            left_time_bound=node.left_time_bound,
            right_time_bound=_logical_right(node),
        )

    if isinstance(node, _SOFT_TEMPORAL):
        base = Globally if isinstance(node, Globally) else Eventually
        child = to_hard(node.child, T, min_width)
        if not node.tunable:
            return base(
                child,
                unbound=node.unbound,
                right_unbound=node.right_unbound,
                left_time_bound=node.left_time_bound,
                right_time_bound=_logical_right(node),
            )
        with torch.no_grad():
            a, b = node._bounds()
        left, right = _int_bounds(
            float(a), float(b), T, min_width,
            child_depth=child.time_depth(),
            # A nested window's bounds are a fraction of its capped span, and
            # it is bounded by that span rather than by the whole signal.
            scale=node.nested_span if node.nested else None,
            horizon_cap=node.nested_span - 1 if node.nested else None,
        )
        # right is the *logical* bound: the constructor re-applies its own +1.
        return base(child, left_time_bound=left, right_time_bound=right)

    if isinstance(node, (And, Or)):
        base = And if isinstance(node, And) else Or
        return base(
            to_hard(node.left_child, T, min_width),
            to_hard(node.right_child, T, min_width),
        )

    if isinstance(node, (Globally, Eventually)):
        base = Globally if isinstance(node, Globally) else Eventually
        return base(
            to_hard(node.child, T, min_width),
            unbound=node.unbound,
            right_unbound=node.right_unbound,
            left_time_bound=node.left_time_bound,
            right_time_bound=_logical_right(node),
        )

    raise TypeError(f"unknown node type: {type(node).__name__}")


# --------------------------------------------------------------------------
# Threshold binding
# --------------------------------------------------------------------------

@dataclass
class _ThresholdBinding:
    atom: Atom
    original: float
    param: torch.Tensor


def _iter_atoms(node):
    if isinstance(node, Atom):
        yield node
        return
    for attr in _CHILD_ATTRS:
        child = getattr(node, attr, None)
        if child is not None:
            yield from _iter_atoms(child)


def bind_thresholds(phi, device=None, dtype=torch.float32) -> list[_ThresholdBinding]:
    """Swap every ``Atom.threshold`` for an optimizable scalar tensor.

    Atom._quantitative only ever does ``xj - threshold``, so a tensor broadcasts
    without any change to torcheck.
    """
    bindings = []
    for atom in _iter_atoms(phi):
        original = float(atom.threshold)
        param = torch.tensor(original, dtype=dtype, device=device, requires_grad=True)
        atom.threshold = param
        bindings.append(_ThresholdBinding(atom, original, param))
    return bindings


def unbind_thresholds(bindings: list[_ThresholdBinding], *, restore: bool) -> None:
    """Put plain floats back on every atom."""
    for b in bindings:
        b.atom.threshold = b.original if restore else float(b.param.detach())


@contextlib.contextmanager
def bound_thresholds(phi, device=None, dtype=torch.float32):
    """Bind thresholds for the duration of the block, always unbinding after.

    The ``finally`` matters: a tensor-valued threshold makes ``str(phi)`` raise
    ``TypeError: type Tensor doesn't define __round__`` (torcheck rounds it for
    display), and ``str(phi)`` is used downstream as a dedup key and written to
    CSV.  Leaking one would surface far from here.
    """
    bindings = bind_thresholds(phi, device, dtype)
    try:
        yield bindings
    except Exception:
        unbind_thresholds(bindings, restore=True)
        raise
    else:
        unbind_thresholds(bindings, restore=False)


# --------------------------------------------------------------------------
# Data-driven smoothing temperatures
# --------------------------------------------------------------------------

def _default_temps(phi, X: np.ndarray, T: int) -> dict:
    """Derive the smoothing temperatures from the formula and the data.

    Every temperature here is in the units of something measurable, which is
    why hand-tuning them generalises so badly: the right acceptance width for a
    formula whose robustness spans ~10 is two orders of magnitude off for one
    whose robustness spans ~0.1, and the right mask steepness for a 20-step
    signal is wrong for a 500-step one.

    Scales used:

    * the robustness spread of the *starting* formula on this data, measured as
      an interquartile range (robust to the outliers that a badly-placed
      starting formula tends to produce), sets both the acceptance sigmoid and
      the LSE temperature;
    * the signal length sets the mask steepness, since it is per-timestep.
    """
    rho = eval_robustness(phi, X)
    iqr = float(np.percentile(rho, 75) - np.percentile(rho, 25))
    if not np.isfinite(iqr) or iqr <= 0:
        # Degenerate: every instance has the same robustness, so the IQR says
        # nothing.  Fall back to the overall spread, then to a bare unit scale.
        iqr = float(np.std(rho))
        if not np.isfinite(iqr) or iqr <= 0:
            iqr = 1.0

    return {
        "beta_lse": 1.0 / (_LSE_TEMP_IQR_FRAC * iqr),
        "c_mask": _MASK_STEEPNESS_STEPS,
        "accept_temp": _ACCEPT_TEMP_IQR_FRAC * iqr,
        "mask_eps": 1e-3,
        "rho_iqr": iqr,
    }


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------

def soft_f1(
    rho: torch.Tensor,
    target_mask: torch.Tensor,
    sigmoid_temp: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable F1 of ``rho > 0`` against the target class.

    Precision alone would be maximized by a formula that accepts nothing (it is
    TP/(TP+FP), undefined-then-zero on an empty acceptance set), so coverage has
    to be in the objective.
    """
    s = torch.sigmoid(rho / sigmoid_temp)
    tp = s[target_mask].sum()
    fp = s[~target_mask].sum()
    precision = tp / (tp + fp + eps)
    coverage = tp / (target_mask.sum() + eps)
    return 2 * precision * coverage / (precision + coverage + eps)


def soft_precision(
    rho: torch.Tensor,
    target_mask: torch.Tensor,
    sigmoid_temp: float,
    min_coverage: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable precision, with a floor under coverage.

    Local explanations are selected and reported on *precision alone*
    (``greedy_precise_picks`` and ``evaluation.evaluate_local_explanation``),
    with no coverage term.  Optimising F1 for them therefore trades precision
    away for coverage -- correctly, by its own lights -- and the reported number
    drops even as the gate records an improvement.

    Precision on its own is maximised by accepting nothing, so coverage enters
    as a hinge rather than as a term: keep at least ``min_coverage`` of the
    target class, and past that maximise precision freely.
    """
    s = torch.sigmoid(rho / sigmoid_temp)
    tp = s[target_mask].sum()
    fp = s[~target_mask].sum()
    precision = tp / (tp + fp + eps)
    coverage = tp / (target_mask.sum() + eps)
    return precision - torch.relu(min_coverage - coverage)


def _hard_score(
    phi, X: np.ndarray, y: np.ndarray, target_class,
    objective: str = "f1", min_coverage: float = 0.0,
) -> float:
    """Exact score under torcheck's true semantics -- never the soft surrogate.

    The gate scores whatever the optimizer is actually maximising, so that
    "accepted" always means "better by the objective the caller asked for".
    """
    rho = eval_robustness(phi, X)
    pos = rho > 0
    target = y == target_class
    n_pos = int(pos.sum())
    if n_pos == 0 or not target.any():
        return float("nan")
    tp = int((pos & target).sum())
    precision = tp / n_pos
    coverage = tp / int(target.sum())
    if objective == "precision":
        return float(precision - max(0.0, min_coverage - coverage))
    if precision + coverage == 0:
        return 0.0
    return float(2 * precision * coverage / (precision + coverage))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def finetune_formula(
    phi,
    X: np.ndarray,
    y: np.ndarray,
    target_class,
    *,
    n_steps: int = 300,
    lr: float = 0.05,
    width_init: float = 0.5,
    eval_every: int = 25,
    objective: str = "f1",
    min_coverage: float = 0.0,
    device: str | None = None,
    accept_only_if_better: bool = True,
    # Overrides for the data-driven temperatures.  Leave them as None unless
    # you are deliberately investigating the smoothing itself.
    lse_temp: float | None = None,
    accept_temp: float | None = None,
    mask_steepness: float | None = None,
    mask_eps: float | None = None,
    # Deprecated, accepted so existing callers keep working; see the note in
    # the docstring.  Annealed schedules for these turned out to be the wrong
    # tool: each is a scale in the units of the data, so it is derived rather
    # than scheduled.
    tau_init: float | None = None,
    tau_final: float | None = None,
    beta_init: float | None = None,
    beta_final: float | None = None,
    sigmoid_temp_init: float | None = None,
    sigmoid_temp_final: float | None = None,
    lambda_width: float | None = None,
    min_width: int | None = None,
) -> tuple[object, dict]:
    """Fine-tune a formula's thresholds and time intervals by gradient descent.

    Maximizes a smooth surrogate of F1 for ``target_class`` on ``(X, y)``.  The
    returned formula is a plain torcheck tree with float thresholds and integer
    time bounds, ready for ``eval_robustness`` / ``evaluate_global``.

    The input formula is never mutated.

    The three smoothing temperatures are **derived from the data**, not passed
    in, because each is a scale in the units of some measured quantity: the
    acceptance sigmoid and the LSE temperature scale with the robustness
    spread of the starting formula, and the mask steepness scales with the
    signal length.  A value hand-tuned on one formula is therefore wrong for a
    formula whose robustness spans a different order of magnitude, which is
    exactly the situation across a bank of sampled formulas.  See
    ``_default_temps``.  Pass ``lse_temp`` / ``accept_temp`` /
    ``mask_steepness`` only to investigate the smoothing itself.

    There is likewise no width penalty.  Soft F1 already penalises a window too
    narrow to capture the class, so a ``lambda_width`` term would only add a
    coefficient to tune; the one real failure mode -- a window collapsing below
    a single timestep, where the mask loses support and the reduction stops
    meaning anything -- is handled structurally by a floor in ``_bounds``.

    Parameters
    ----------
    width_init:
        Minimum starting window width, as a fraction of the signal.  The loss
        surface has a genuine local optimum between a narrow and a wide window
        -- a narrow window can already fit some instances well, so from a
        narrow start the gradient points at shrinking further and never reaches
        the timesteps that separate the classes.  Starting open and tightening
        avoids that.
    eval_every:
        Stride, in steps, at which the true hard-semantics F1 is checked so the
        best iterate can be kept.  The surrogate and the true objective are not
        the same function, so the last iterate is not reliably the best one.
    accept_only_if_better:
        Return the original formula if tuning did not improve hard-semantics
        F1.  Tuning can then never make a result worse, which is also what
        makes any residual gap between the soft and hard optima harmless.

    Returns
    -------
    (phi_tuned, log) where log has "loss", "hard_f1", "f1_before", "f1_after",
    "accepted", "n_steps" and "temps".

    Notes
    -----
    ``tau_*``, ``beta_*``, ``sigmoid_temp_*``, ``lambda_width`` and
    ``min_width`` are accepted and ignored, so older call sites keep running.
    """
    y = np.asarray(y)
    X = np.ascontiguousarray(X, dtype=np.float32)
    T = X.shape[2]

    original = phi
    phi = copy.deepcopy(phi)

    score_kw = dict(objective=objective, min_coverage=min_coverage)
    f1_before = _hard_score(phi, X, y, target_class, **score_kw)

    signal = torch.as_tensor(X, device=device)          # built once, not per step
    target_mask = torch.as_tensor(y == target_class, device=device)

    temps = _default_temps(phi, X, T)
    if lse_temp is not None:
        temps["beta_lse"] = 1.0 / lse_temp
    if accept_temp is not None:
        temps["accept_temp"] = accept_temp
    if mask_steepness is not None:
        temps["c_mask"] = mask_steepness
    if mask_eps is not None:
        temps["mask_eps"] = mask_eps
    temps["width_init"] = width_init
    interval_params: list[torch.Tensor] = []
    soft = to_soft(phi, temps, T, interval_params)

    log: dict = {
        "loss": [],
        "hard_f1": [],
        "f1_before": f1_before,
        "n_steps": n_steps,
        # Recorded so a surprising result can be traced back to the scales that
        # were derived for it.
        "temps": {k: v for k, v in temps.items() if k != "width_init"},
    }
    best_score: float | None = None
    best_phi = None

    with bound_thresholds(soft, device=device) as bindings:
        params = [b.param for b in bindings] + interval_params
        if not params:
            log.update(f1_after=f1_before, accepted=False)
            return original, log

        opt = torch.optim.Adam(params, lr=lr)
        sig_t = temps["accept_temp"]
        for step in range(n_steps):
            opt.zero_grad()
            rho = soft.quantitative(signal, evaluate_at_all_times=False, normalize=False)
            if objective == "precision":
                loss = -soft_precision(rho, target_mask, sig_t, min_coverage)
            else:
                loss = -soft_f1(rho, target_mask, sig_t)
            loss.backward()
            opt.step()
            log["loss"].append(float(loss.detach()))

            # Keep the best iterate by *hard* F1 rather than trusting the last
            # one.  The surrogate and the true objective are not the same
            # function, so descending the surrogate to convergence can walk
            # past the best formula and settle somewhere slightly worse; and
            # the anneal keeps changing the objective underneath the optimizer,
            # so "last" is not even a fixed point.  Checking costs one hard
            # evaluation, so do it on a stride.
            if step % eval_every == 0 or step == n_steps - 1:
                with torch.no_grad():
                    candidate = to_hard(soft, T)
                score = _hard_score(candidate, X, y, target_class, **score_kw)
                log["hard_f1"].append(score)
                if not math.isnan(score) and (
                    best_score is None or math.isnan(best_score) or score > best_score
                ):
                    best_score, best_phi = score, candidate

    phi_tuned = best_phi if best_phi is not None else to_hard(soft, T)
    f1_after = _hard_score(phi_tuned, X, y, target_class, **score_kw)

    # A NaN "before" means nothing satisfied the starting formula, so any finite
    # result is an improvement; without this the hardest starting points could
    # never be accepted.
    improved = not math.isnan(f1_after) and (
        math.isnan(f1_before) or f1_after >= f1_before
    )
    accepted = improved or not accept_only_if_better

    log.update(f1_after=f1_after, accepted=accepted)
    return (phi_tuned if accepted else original), log

