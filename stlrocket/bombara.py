"""Decision tree of STL primitives, after Bombara and Belta, "Offline and Online Learning of
Signal Temporal Logic Formulae Using Decision Trees" (ACM TCPS 2021), offline algorithm: a
baseline of the comparison that learns STL formulae directly from the series.

Each internal node splits the series on one primitive, a PSTL formula of one variable x_v
whose parameters are fitted at that node (their Algorithm 1). The primitives are the first-
and second-level ones of the paper (P1 and P2, their Definitions 4.1-4.2), on the sample grid
(time in steps, windows inclusive):

  kind  formula                   satisfied iff                                params
  F     F[a,b] (x_v > mu)         max_{[a,b]} x_v > mu                         a <= b
  G     G[a,b] (x_v > mu)         min_{[a,b]} x_v > mu                         a <= b
  GF    G[a,b] F[0,c] (x_v > mu)  min_{t in [a,b]} max_{[t,t+c]} x_v > mu      a <= b, b + c <= T-1
  FG    F[a,b] G[0,c] (x_v > mu)  max_{t in [a,b]} min_{[t,t+c]} x_v > mu      a <= b, b + c <= T-1

The primitives with x_v <= mu are the negations of these (F(x <= mu) = not G(x > mu), and so
on), which split the series in the same two sets, so these four kinds cover P1 and P2. Each
primitive is a threshold on one statistic s(tau) of the series, so for given time parameters
the best mu is found exactly by sweeping the thresholds between consecutive values of s, as
in CART; Particle Swarm Optimization (the optimizer of the paper, in the inertia-weight form
of Shi and Eberhart 1998) searches only the time parameters, one run per (variable, kind).
The paper's PSO also searches mu, so this is a better optimizer of the same objective.

Departures from the paper, needed for the multiclass, imbalanced UEA datasets and the
comparison's protocol: the impurity is the multiclass information gain (the paper's IG with
the entropy over K classes, as its Section 7.3 suggests) on class-balanced weights (as
stl_tree, sklearn's class_weight="balanced"), leaves predict the weighted majority class, and
the maximum depth is chosen by cross-validation (choose_depth) instead of cost-complexity
pruning. Statistics use sparse tables (range max/min in O(1) per series).

The STL formula of class k (class_formula) is the OR over the leaves of class k of the AND of
the literals on the path, the multiclass form of their Algorithm 2; exactly one class
formula holds on any series, the one of the class the tree predicts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torcheck.stl import And, Atom, Eventually, Globally, Not, Or

from .formula_sampler import F0

KINDS = ("F", "G", "GF", "FG")
DEPTHS = (1, 3, 5)
MIN_GAIN = 1e-12      # a split must lower the weighted entropy by more than this


@dataclass
class PSOConfig:
    n_particles: int = 20
    n_iter: int = 30
    w_start: float = 0.9      # inertia weight, decreasing linearly to w_end (Shi and Eberhart 1998)
    w_end: float = 0.4
    c1: float = 2.0           # cognitive and social coefficients
    c2: float = 2.0


# ---------------------------------------------------------------- statistics

def sparse_table(x: np.ndarray, op) -> list[np.ndarray]:
    """ST[k][:, t] = op over x[:, t : t + 2^k] (x: (n, T)), for 2^k <= T."""
    st = [x]
    k = 1
    while (1 << k) <= x.shape[1]:
        h = 1 << (k - 1)
        prev = st[-1]
        st.append(op(prev[:, :-h], prev[:, h:]))
        k += 1
    return st


def _window(st, op, a: int, b: int) -> np.ndarray:
    """op over x[:, a..b] from a sparse table."""
    k = (b - a + 1).bit_length() - 1
    return op(st[k][:, a], st[k][:, b - (1 << k) + 1])


def _sliding(st, op, a: int, b: int, c: int) -> np.ndarray:
    """(n, b-a+1): op over x[:, t..t+c] for t = a..b, from a sparse table."""
    k = (c + 1).bit_length() - 1
    s = c + 1 - (1 << k)
    return op(st[k][:, a:b + 1], st[k][:, a + s:b + s + 1])


def statistic_tables(x: np.ndarray, kind: str, a: int, b: int, c: int, st_max, st_min) -> np.ndarray:
    if kind == "F":
        return _window(st_max, np.maximum, a, b)
    if kind == "G":
        return _window(st_min, np.minimum, a, b)
    if kind == "GF":
        return _sliding(st_max, np.maximum, a, b, c).min(axis=1)
    if kind == "FG":
        return _sliding(st_min, np.minimum, a, b, c).max(axis=1)
    raise ValueError(kind)


def statistic(x: np.ndarray, kind: str, a: int, b: int, c: int = 0) -> np.ndarray:
    """The statistic of a primitive on x (n, T), computed directly (prediction, tests)."""
    if kind == "F":
        return x[:, a:b + 1].max(axis=1)
    if kind == "G":
        return x[:, a:b + 1].min(axis=1)
    win = np.lib.stride_tricks.sliding_window_view(x[:, a:b + c + 1], c + 1, axis=1)
    if kind == "GF":
        return win.max(axis=2).min(axis=1)
    if kind == "FG":
        return win.min(axis=2).max(axis=1)
    raise ValueError(kind)


def decode(pos: np.ndarray, T: int) -> tuple[int, int, int]:
    """Particle position in [0, T-1]^d -> integer (a, b, c) with a <= b and b + c <= T-1."""
    p = np.rint(pos).astype(int)
    a, b = min(p[0], p[1]), max(p[0], p[1])
    c = min(int(p[2]), T - 1 - b) if len(p) > 2 else 0
    return int(a), int(b), int(c)


# ---------------------------------------------------------------- split search

def _entropy(M: np.ndarray) -> np.ndarray:
    """Entropy of the weighted class counts along the last axis, times their total."""
    tot = M.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = M / tot[..., None]
        H = -np.where(P > 0, P * np.log(P), 0.0).sum(axis=-1)
    return np.where(tot > 0, H * tot, 0.0)


def best_thresholds(S: np.ndarray, Yw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row of S (P, n) of statistics, the threshold mu with the largest information
    gain of the split {s > mu} / {s <= mu} on weighted one-hot labels Yw (n, K): (gain, mu),
    gain -inf where s is constant. mu is the midpoint between consecutive distinct values."""
    order = np.argsort(S, axis=1, kind="stable")
    Ss = np.take_along_axis(S, order, axis=1)
    C = np.cumsum(Yw[order], axis=1)                       # (P, n, K): the first i+1 sorted
    total = C[:, -1]
    left, right = C[:, :-1], total[:, None, :] - C[:, :-1]
    W = total.sum(axis=-1)
    gain = (_entropy(total)[:, None] - _entropy(left) - _entropy(right)) / W[:, None]
    gain = np.where(Ss[:, 1:] > Ss[:, :-1], gain, -np.inf)
    i = np.argmax(gain, axis=1)
    rows = np.arange(len(S))
    mu = (Ss[rows, i].astype(np.float64) + Ss[rows, i + 1]) / 2
    return gain[rows, i], mu


def pso_primitive(x: np.ndarray, Yw: np.ndarray, kind: str, st_max, st_min, cfg: PSOConfig,
                  rng: np.random.Generator) -> tuple[float, tuple, float]:
    """Best (gain, (a, b, c), mu) of one primitive kind on one variable x (n, T), by PSO over
    the time parameters with the exact threshold. One particle starts at the whole series
    (and c = 0), the others uniformly at random."""
    T = x.shape[1]
    dim = 2 if kind in ("F", "G") else 3
    hi = float(T - 1)
    cache: dict[tuple, tuple[float, float]] = {}

    def evaluate(P: np.ndarray) -> np.ndarray:
        params = [decode(p, T) for p in P]
        new = list(dict.fromkeys(q for q in params if q not in cache))
        if new:
            S = np.stack([statistic_tables(x, kind, *q, st_max, st_min) for q in new])
            g, mu = best_thresholds(S, Yw)
            cache.update(zip(new, zip(g.tolist(), mu.tolist())))
        return np.array([cache[q][0] for q in params])

    pos = rng.uniform(0, hi, (cfg.n_particles, dim))
    pos[0] = [0.0, hi, 0.0][:dim]
    vel = rng.uniform(-hi, hi, (cfg.n_particles, dim)) * 0.1
    val = evaluate(pos)
    pbest, pval = pos.copy(), val.copy()
    g = int(np.argmax(pval))
    for it in range(cfg.n_iter):
        w = cfg.w_start + (cfg.w_end - cfg.w_start) * it / max(cfg.n_iter - 1, 1)
        r1, r2 = rng.random((2, cfg.n_particles, dim))
        vel = w * vel + cfg.c1 * r1 * (pbest - pos) + cfg.c2 * r2 * (pbest[g] - pos)
        vel = np.clip(vel, -hi, hi)
        pos = np.clip(pos + vel, 0.0, hi)
        val = evaluate(pos)
        better = val > pval
        pbest[better], pval[better] = pos[better], val[better]
        g = int(np.argmax(pval))
    q = decode(pbest[g], T)
    return float(pval[g]), q, cache[q][1]


# ---------------------------------------------------------------- tree

@dataclass
class Node:
    id: int                      # heap numbering: root 1, children 2i (satisfied) and 2i+1
    depth: int
    weights: np.ndarray          # weighted class counts of the train series reaching it
    label: int                   # weighted majority class (index into classes)
    split: dict | None = None    # kind, v, a, b, c, mu, gain


@dataclass
class BombaraTree:
    classes: np.ndarray
    max_depth: int
    nodes: dict = field(default_factory=dict)

    # ------------------------------------------------------------ prediction
    def predict_index(self, X: np.ndarray, max_depth: int | None = None) -> np.ndarray:
        """Class index of each series of X (N, V, T), the tree cut at max_depth."""
        d = self.max_depth if max_depth is None else max_depth
        out = np.empty(len(X), dtype=int)

        def walk(nid: int, idx: np.ndarray) -> None:
            if len(idx) == 0:
                return
            node = self.nodes[nid]
            sp = node.split
            if sp is None or node.depth >= d:
                out[idx] = node.label
                return
            s = statistic(X[idx, sp["v"]], sp["kind"], sp["a"], sp["b"], sp["c"])
            sat = s > sp["mu"]
            walk(2 * nid, idx[sat])
            walk(2 * nid + 1, idx[~sat])

        walk(1, np.arange(len(X)))
        return out

    def predict(self, X: np.ndarray, max_depth: int | None = None) -> np.ndarray:
        return self.classes[self.predict_index(X, max_depth)]

    # ------------------------------------------------------------ formulae
    def leaves(self, max_depth: int | None = None) -> list[tuple[int, list]]:
        """(class index, literals) of each leaf, a literal being (split, satisfied)."""
        d = self.max_depth if max_depth is None else max_depth
        out = []

        def walk(nid: int, lits: list) -> None:
            node = self.nodes[nid]
            if node.split is None or node.depth >= d:
                out.append((node.label, lits))
                return
            walk(2 * nid, lits + [(node.split, True)])
            walk(2 * nid + 1, lits + [(node.split, False)])

        walk(1, [])
        return out

    def class_formula(self, k: int, max_depth: int | None = None):
        """STL formula of class index k: OR over its leaves of the AND of their path's literals.
        None if no leaf predicts k, or if the tree is a single leaf (the formula would be True)."""
        paths = [lits for label, lits in self.leaves(max_depth) if label == k]
        if not paths or not all(paths):
            return None
        conj = [_and([primitive_formula(sp) if sat else Not(primitive_formula(sp)) for sp, sat in lits])
                for lits in paths]
        return _or(conj)

    def n_leaves(self, max_depth: int | None = None) -> int:
        return len(self.leaves(max_depth))

    def depth(self, max_depth: int | None = None) -> int:
        return max((len(l) for _, l in self.leaves(max_depth)), default=0)

    def to_json(self) -> dict:
        return {"classes": [str(c) for c in self.classes], "max_depth": self.max_depth,
                "nodes": [{"id": n.id, "depth": n.depth, "label": n.label, "weights": n.weights.tolist(),
                           "split": n.split} for n in self.nodes.values()]}


def primitive_formula(sp: dict):
    """torcheck formula of a split: satisfied iff its statistic > mu."""
    atom = Atom(sp["v"], float(sp["mu"]))
    a, b, c = sp["a"], sp["b"], sp["c"]
    if sp["kind"] == "F":
        return Eventually(atom, left_time_bound=a, right_time_bound=b)
    if sp["kind"] == "G":
        return Globally(atom, left_time_bound=a, right_time_bound=b)
    if sp["kind"] == "GF":
        return Globally(Eventually(atom, left_time_bound=0, right_time_bound=c), left_time_bound=a, right_time_bound=b)
    if sp["kind"] == "FG":
        return Eventually(Globally(atom, left_time_bound=0, right_time_bound=c), left_time_bound=a, right_time_bound=b)
    raise ValueError(sp["kind"])


def _and(fs):
    out = fs[0]
    for f in fs[1:]:
        out = And(out, f)
    return out


def _or(fs):
    out = fs[0]
    for f in fs[1:]:
        out = Or(out, f)
    return out


def formula_sizes(tree: BombaraTree, max_depth: int | None = None) -> list[int]:
    """Node count of each class formula (0 for a class with no formula)."""
    out = []
    for k in range(len(tree.classes)):
        phi = tree.class_formula(k, max_depth)
        out.append(F0.formula_size(phi) if phi is not None else 0)
    return out


def _best_split(X: np.ndarray, Yw: np.ndarray, node_id: int, seed: int, cfg: PSOConfig, n_jobs: int):
    """Best split of the series X (n, V, T) over every (variable, kind), or None."""
    def one_variable(v: int):
        x = np.ascontiguousarray(X[:, v])
        st_max, st_min = sparse_table(x, np.maximum), sparse_table(x, np.minimum)
        best = None
        for ki, kind in enumerate(KINDS):
            rng = np.random.default_rng([seed, node_id, v, ki])
            gain, (a, b, c), mu = pso_primitive(x, Yw, kind, st_max, st_min, cfg, rng)
            if best is None or gain > best["gain"]:
                best = dict(kind=kind, v=v, a=a, b=b, c=c, mu=mu, gain=gain)
        return best

    V = X.shape[1]
    if n_jobs > 1 and V > 1:
        with ThreadPoolExecutor(min(n_jobs, V)) as ex:
            cands = list(ex.map(one_variable, range(V)))
    else:
        cands = [one_variable(v) for v in range(V)]
    best = max(cands, key=lambda s: s["gain"])     # first on ties: lowest variable, then KINDS order
    return best if best["gain"] > MIN_GAIN else None


def fit_tree(X: np.ndarray, y: np.ndarray, max_depth: int, seed: int, cfg: PSOConfig | None = None,
             n_jobs: int = 1, classes: np.ndarray | None = None, class_weight: np.ndarray | None = None) -> BombaraTree:
    """Grow the tree on X (N, V, T), y to max_depth (their Algorithm 1 with the stopping
    conditions: maximum depth, a pure node, or no split lowering the impurity). Each node's
    PSO is seeded by (seed, node id, variable, kind), so the tree cut at a smaller depth is
    exactly the tree grown to that depth."""
    cfg = cfg or PSOConfig()
    classes = np.unique(y) if classes is None else classes
    yi = np.searchsorted(classes, y)
    K = len(classes)
    if class_weight is None:   # sklearn's "balanced": n / (K * n_k)
        counts = np.bincount(yi, minlength=K)
        class_weight = np.where(counts > 0, len(y) / (K * np.maximum(counts, 1)), 0.0)
    Yw = np.eye(K)[yi] * class_weight[yi][:, None]
    X = np.asarray(X, dtype=np.float32)
    tree = BombaraTree(classes=classes, max_depth=max_depth)

    def grow(nid: int, depth: int, idx: np.ndarray) -> None:
        w = Yw[idx].sum(axis=0)
        node = Node(id=nid, depth=depth, weights=w, label=int(np.argmax(w)))
        tree.nodes[nid] = node
        if depth >= max_depth or len(idx) < 2 or np.count_nonzero(w) < 2:
            return
        sp = _best_split(X[idx], Yw[idx], nid, seed, cfg, n_jobs)
        if sp is None:
            return
        node.split = sp
        sat = statistic(X[idx, sp["v"]], sp["kind"], sp["a"], sp["b"], sp["c"]) > sp["mu"]
        grow(2 * nid, depth + 1, idx[sat])
        grow(2 * nid + 1, depth + 1, idx[~sat])

    grow(1, 0, np.arange(len(X)))
    return tree


def choose_depth(X: np.ndarray, y: np.ndarray, folds, seed: int, depths=DEPTHS, cfg: PSOConfig | None = None,
                 n_jobs: int = 1) -> tuple[int, dict]:
    """Max depth with the best mean balanced accuracy over the folds (ties: shallower), each
    fold growing one tree to max(depths) and cutting it at each depth. Class weights are
    balanced on each fold's train part, as sklearn's class_weight="balanced" in a CV."""
    classes = np.unique(y)
    scores = {d: [] for d in depths}
    for tr, va in folds:
        tree = fit_tree(X[tr], y[tr], max(depths), seed, cfg, n_jobs, classes=classes)
        for d in depths:
            scores[d].append(balanced_accuracy_score(y[va], tree.predict(X[va], d)))
    mean = {d: float(np.mean(s)) for d, s in scores.items()}
    best = max(depths, key=lambda d: (round(mean[d], 12), -d))
    return best, mean
