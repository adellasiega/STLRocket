"""
Abductive explanations (AXp) of the linear classifier, exact (Marques-Silva et al. 2020).

The model predicts argmax_k w_k . x over the selected features, so a set of conditions on the
features is sufficient for the prediction k of x iff, for every other class l, the margin
(w_k - w_l) . z stays positive for every z in the feature domain that satisfies the conditions.
The domain is the box [lo_j, hi_j] of the train values of each feature (widened to contain x),
and the conditions bound features from below (z_j >= x_j) or above (z_j <= x_j); each margin is
linear, so its minimum over the box is reached feature by feature at an end of its interval and
is computed exactly.

axp returns:
  features  the AXp proper: a subset-minimal set of features which, fixed to their values in x,
            are sufficient (a deletion pass over the features, freeing first those whose release
            costs the least margin; for two classes this order gives a minimum-size AXp)
  literals  the same explanation as literals (j, e, s) of instance_rules (e * z_j > s): each
            fixed feature is split into its two bounds, and a second deletion pass drops the
            bounds that are not needed. The result is a subset-minimal set of literals, sufficient
            on the whole box; with two classes exactly one bound per feature of the AXp is left.
inflated_axp widens each bound of the literal AXp as far as the others allow (Izza et al. 2024).
Every train series lies in the box, so both have precision 1 on train w.r.t. the predictions of
the model; test series may fall outside it.

probabilistic_inflated_axp relaxes sufficiency to probability (probabilistic AXps, Izza et al.
2023, inflated as in Izza et al. 2024): the literals must keep the prediction k with probability
at least delta when the features follow a distribution D, here the features independent, each
distributed as its values on train (the product of the empirical marginals; a literal restricts
its feature to the train values within its interval). The probability is estimated by Monte Carlo
on draws fixed for the whole explanation (MarginalSampler). Starting from the literal AXp, which
has probability 1, the literals are released one by one, cheapest first (the worst-case margin
their release costs, as in axp), while the estimate stays >= delta; then each remaining bound is
widened towards the end of the box over a grid of train quantiles, stopping at the first step
that brings the estimate below delta. Unlike the formal AXps, nothing is guaranteed on any series:
the probability is under D, which ignores the dependence between the features.
"""
from __future__ import annotations

import numpy as np


def axp(x: np.ndarray, W_full: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """AXp of the prediction of x (all features) for the model W_full (K, M) on the box
    [lo, hi] (M,). Only features with a nonzero weight can enter the explanation."""
    k = int(np.argmax(W_full @ x))
    S = np.flatnonzero(np.any(W_full != 0, axis=0))
    D = (W_full[k] - np.delete(W_full, k, axis=0))[:, S]        # (K-1, |S|) margins' weights
    v = x[S]
    lo, hi = np.minimum(lo[S], v), np.maximum(hi[S], v)
    fixed = D * v                                                 # contribution with z_j = x_j
    # worst contribution with z_j free on [lo, v] (lower bound released) or [v, hi] (upper)
    free_lo = np.minimum(D * lo, fixed)
    free_hi = np.minimum(D * hi, fixed)
    free = np.minimum(free_lo, free_hi)
    margin = fixed.sum(axis=1)                                    # worst margins, everything fixed
    if np.any(margin <= 0):                                       # tie: nothing is sufficient
        return dict(k=k, features=[], literals=[], sufficient=False)

    # pass 1: features, cheapest release first
    order = np.argsort(np.max(fixed - free, axis=0), kind="stable")
    keep = np.ones(len(S), dtype=bool)
    for i in order:
        m = margin - fixed[:, i] + free[:, i]
        if np.all(m > 0):
            margin, keep[i] = m, False
    # pass 2: the two bounds of each kept feature, cheapest release first
    has_lo, has_hi = keep.copy(), keep.copy()
    cur = fixed.copy()                                            # contribution of kept features
    bounds = [(i, side) for i in np.flatnonzero(keep) for side in (0, 1)]
    cost = [np.max(fixed[:, i] - (free_lo if side == 0 else free_hi)[:, i]) for i, side in bounds]
    for b in np.argsort(cost, kind="stable"):
        i, side = bounds[b]
        lo_on, hi_on = (False, has_hi[i]) if side == 0 else (has_lo[i], False)
        new = fixed[:, i] if lo_on and hi_on else free_lo[:, i] if hi_on else free_hi[:, i] if lo_on else free[:, i]
        m = margin - cur[:, i] + new
        if np.all(m > 0):
            margin, cur[:, i] = m, new
            has_lo[i], has_hi[i] = lo_on, hi_on

    # z_j >= v_j as e = +1, s just below v_j; z_j <= v_j as e = -1, s just below -v_j
    literals = ([(int(S[i]), 1, float(np.nextafter(v[i], -np.inf))) for i in np.flatnonzero(has_lo)]
                + [(int(S[i]), -1, float(np.nextafter(-v[i], -np.inf))) for i in np.flatnonzero(has_hi)])
    return dict(k=k, features=[int(j) for j in S[keep]], literals=literals, sufficient=True,
                min_margin=float(margin.min()))


def inflated_axp(x: np.ndarray, W_full: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Inflated AXp (Izza et al. 2024) of the prediction of x: the features of axp, each widened
    from its value in x to an interval [a_j, b_j] as large as the other intervals allow, so
    that the explanation stays sufficient on the box. Features are widened in the order of axp
    (cheapest release first), each first downwards and then upwards, each side as far as every
    margin allows (exactly: the margins are linear in a_j and b_j). A side widened to the end
    of the box is dropped, so the literals are a subset of those of axp with looser thresholds."""
    base = axp(x, W_full, lo, hi)
    if not base["sufficient"]:
        return dict(base, intervals={})
    k = base["k"]
    D = W_full[k] - np.delete(W_full, k, axis=0)                  # (K-1, M)
    lo, hi = np.minimum(lo, x), np.maximum(hi, x)
    a, b = lo.copy(), hi.copy()                                    # current interval of each feature
    tol = 1e-9 * np.maximum(hi - lo, 1e-12)                        # keeps every margin clear of 0
    for j, e, s in base["literals"]:                               # start from the literal AXp
        if e > 0:
            a[j] = x[j]
        else:
            b[j] = x[j]
    worst = lambda j: np.minimum(D[:, j] * a[j], D[:, j] * b[j])
    margin = sum(worst(j) for j in np.flatnonzero(np.any(W_full != 0, axis=0)))
    order = list(dict.fromkeys(j for j, _, _ in base["literals"]))
    for j in order:
        for side in (0, 1):
            if (side == 0 and a[j] <= lo[j]) or (side == 1 and b[j] >= hi[j]):
                continue
            d = D[:, j]
            # contribution of j with the side moved to t: d * t on the classes it now bounds
            # (d > 0 for the lower side, d < 0 for the upper), unchanged on the others
            rest = margin - worst(j)
            other = d * b[j] if side == 0 else d * a[j]
            act = d > 0 if side == 0 else d < 0
            # need rest + min(d t, other) > 0 on every class: other already fine, so d t > -rest
            with np.errstate(divide="ignore"):
                bound = -rest[act] / d[act]
            if side == 0:
                t = max(lo[j], bound.max() if act.any() else -np.inf)
                t = lo[j] if t <= lo[j] else min(x[j], t + tol[j])
                old, a[j] = a[j], t
            else:
                t = min(hi[j], bound.min() if act.any() else np.inf)
                t = hi[j] if t >= hi[j] else max(x[j], t - tol[j])
                old, b[j] = b[j], t
            new = rest + worst(j)
            if np.all(new > 0):
                margin = new
            elif side == 0:                                         # rounding: keep the old end
                a[j] = old
            else:
                b[j] = old
    literals = ([(int(j), 1, float(np.nextafter(a[j], -np.inf))) for j in order if a[j] > lo[j]]
                + [(int(j), -1, float(np.nextafter(-b[j], -np.inf))) for j in order if b[j] < hi[j]])
    return dict(k=k, features=[int(j) for j in order], literals=literals, sufficient=True,
                intervals={int(j): (float(a[j]), float(b[j])) for j in order}, min_margin=float(margin.min()))


class MarginalSampler:
    """Monte Carlo estimate of the probability that the model predicts k when the selected
    features S are independent, each distributed as its train values, and restricted to
    intervals [a_j, b_j]. The uniform draws U are fixed at construction, so successive estimates
    for the same explanation differ only through the intervals (common random numbers), and
    restricting a feature maps the same draws to its train values within the interval. A feature
    whose interval holds no train value (x outside the train range) takes its value in x."""

    def __init__(self, F_tr: np.ndarray, W_full: np.ndarray, n_samples: int = 2000):
        self.S = np.flatnonzero(np.any(W_full != 0, axis=0))
        self.WS = W_full[:, self.S]                                   # (K, |S|)
        self.V = np.sort(F_tr[:, self.S], axis=0)                     # (N, |S|) sorted train values
        self.n_samples = n_samples

    def start(self, x: np.ndarray, seed: int) -> None:
        """Fresh draws for the explanation of x; every feature free."""
        rng = np.random.default_rng(seed)
        self.U = rng.random((self.n_samples, len(self.S)))
        self.xS = x[self.S]
        N = len(self.V)
        self.a = np.full(len(self.S), -np.inf)
        self.b = np.full(len(self.S), np.inf)
        self.Z = np.take_along_axis(self.V, np.minimum((self.U * N).astype(int), N - 1), axis=0)
        self.L = self.Z @ self.WS.T                                     # (n, K) logits

    def _column(self, c: int, a: float, b: float) -> np.ndarray:
        v = self.V[:, c]
        i0, i1 = np.searchsorted(v, a, "left"), np.searchsorted(v, b, "right")
        if i1 <= i0:
            return np.full(self.n_samples, self.xS[c])
        return v[np.minimum(i0 + (self.U[:, c] * (i1 - i0)).astype(int), i1 - 1)]

    def set(self, c: int, a: float, b: float) -> None:
        """Restrict feature S[c] to [a, b] (-inf / inf: unbounded side)."""
        col = self._column(c, a, b)
        self.L += np.outer(col - self.Z[:, c], self.WS[:, c])
        self.Z[:, c], self.a[c], self.b[c] = col, a, b

    def precision(self, k: int) -> float:
        return float(np.mean(np.argmax(self.L, axis=1) == k))


def probabilistic_inflated_axp(x: np.ndarray, W_full: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                               sampler: MarginalSampler, delta: float = 0.95, seed: int = 0,
                               n_steps: int = 16) -> dict:
    """Probabilistic inflated AXp of the prediction of x (module docstring): literals keeping
    the prediction with estimated probability >= delta under the product of the empirical
    marginals, from the literal AXp by release, then widening of the remaining bounds."""
    base = axp(x, W_full, lo, hi)
    if not base["sufficient"]:
        return dict(base, est_precision=float("nan"))
    k = base["k"]
    col = {int(j): c for c, j in enumerate(sampler.S)}
    sampler.start(x, seed)
    for j, e, _ in base["literals"]:                                # the literal AXp: z_j >= x_j or <= x_j
        c = col[j]
        a, b = (x[j], sampler.b[c]) if e > 0 else (sampler.a[c], x[j])
        sampler.set(c, a, b)
    p = sampler.precision(k)
    # release cost: worst-case margin lost when the bound is freed to the end of the box
    D = W_full[k] - np.delete(W_full, k, axis=0)
    lo_b, hi_b = np.minimum(lo, x), np.maximum(hi, x)
    def cost(j, e):
        end = lo_b[j] if e > 0 else hi_b[j]
        return float(np.max(D[:, j] * x[j] - np.minimum(D[:, j] * end, D[:, j] * x[j])))
    order = sorted(((j, e) for j, e, _ in base["literals"]), key=lambda t: cost(*t))
    kept = []
    for j, e in order:                                              # pass 1: release
        c = col[j]
        old = (sampler.a[c], sampler.b[c])
        sampler.set(c, -np.inf if e > 0 else old[0], old[1] if e > 0 else np.inf)
        q = sampler.precision(k)
        if q >= delta:
            p = q
        else:
            sampler.set(c, *old)
            kept.append((j, e))
    for j, e in kept:                                               # pass 2: widen over train quantiles
        c = col[j]
        v = sampler.V[:, c]
        beyond = v[v < x[j]][::-1] if e > 0 else v[v > x[j]]       # train values past x, outwards
        if len(beyond) == 0:
            continue
        steps = beyond[np.unique(np.linspace(0, len(beyond) - 1, min(n_steps, len(beyond))).round().astype(int))]
        for t in steps:
            old = (sampler.a[c], sampler.b[c])
            sampler.set(c, t if e > 0 else old[0], old[1] if e > 0 else t)
            q = sampler.precision(k)
            if q >= delta:
                p = q
            else:
                sampler.set(c, *old)
                break
    literals = []
    for j, e in kept:
        c = col[j]
        if e > 0 and sampler.a[c] > lo_b[j]:
            literals.append((int(j), 1, float(np.nextafter(sampler.a[c], -np.inf))))
        elif e < 0 and sampler.b[c] < hi_b[j]:
            literals.append((int(j), -1, float(np.nextafter(-sampler.b[c], -np.inf))))
    return dict(k=k, features=sorted({j for j, _, _ in literals}), literals=literals, sufficient=True,
                est_precision=p, axp_length=len(base["literals"]))


def is_sufficient(literals: list, k: int, W_full: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    """Whether the literals (j, e, s) guarantee class k on the box [lo, hi]: brute check of the
    worst margins, for tests."""
    L, U = lo.astype(float).copy(), hi.astype(float).copy()
    for j, e, s in literals:
        if e > 0:
            L[j] = max(L[j], s)
        else:
            U[j] = min(U[j], -s)
    D = W_full[k] - np.delete(W_full, k, axis=0)
    return bool(np.all(np.minimum(D * L, D * U).sum(axis=1) > 0))
