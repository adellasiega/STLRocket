"""TLINet, after Li, Cai, Vasile and Tron, "Learning Signal Temporal Logic through Neural Network
for Interpretable Classification" (ACC 2023; extended as "TLINet: Differentiable Neural Network
Temporal Logic Inference", arXiv 2405.06670): a baseline of the comparison that learns an STL
formula end to end by gradient descent. This is a vectorized reimplementation of the authors'
code (https://github.com/danyangl6/TLINet: neurons.py, naval/naval_models.py and the training
loop of naval/Naval.ipynb), which has no license, is binary only and loops in Python over
predicates and time steps.

The network of one binary classifier, on series x of C variables and T steps:
  predicates   z_i(t) = a_i x_{v_i}(t) - b_i, i = 1..nf, nf = 3C, predicate i on variable
               v_i = i mod C (the authors' dim = i % 2 with nf = 3 * dim, for any C)
  temporal     r_i = tvar_i smax_w(z_i) + (1 - tvar_i) (-smax_w(-z_i)): F (tvar = 1) or G
               (tvar = 0) over the soft window w(t) = min(ramp(t - t1 + tau), ramp(t2 - t + tau)),
               ramp(s) = clip(s / tau, 0, 1), where smax_w is the authors' sparse softmax
               (beta = 2.5, h = 1)
  logical      c_j = tvar_j am_p(r) + (1 - tvar_j) (-am_p(-r)): OR (tvar = 1) or AND (tvar = 0)
               of the inputs selected by p_j in [0, 1], where am_p is the authors' averaged
               max, the expected value of the largest input when each input is kept
               independently with probability p
  architecture n_clauses None (their Model1): one logical layer of one clause over the
               predicates; n_clauses = m (their Model2): m clauses, then one clause over them
The loss is theirs: mean(relu(eps - y r)) - w_margin eps + w_bi * mean over the tensors of
mean p (1 - p) (the bimodal regularizer of every p and tvar) + w_l1 * sum p, with eps
learned, Adam, and before every step the projection of their forward (p and tvar clipped to
[0, 1], t1 and t2 to [0, T - 1] with t1 < t2 - 1, all-zero rows of p redrawn, eps >= 1e-5).
Hyperparameters are those of their naval notebook for each model (TLINetConfig).

The formula of a trained network (formula) reads every operator as hard: F if tvar > 0.5,
else G, on the window [int(t1), int(t2)] as their translate_formula; x_v > b/a if a > 0, else
x_v < b/a; the predicates and clauses with p >= 0.5; OR if tvar > 0.5, else AND.

Two readings of that hard network give its robustness (TLINetConfig.semantics):
  "authors"  their formula_forward, by which their papers report the formula's accuracy: the
             hard operators, but each temporal operator still the sparse softmax over the
             soft window (so not the max or min over [int(t1), int(t2)]), and a clause that
             selects nothing has robustness 0; the default, so that TLINet is scored as by
             its authors
  "exact"    the STL semantics of the formula (exact, equal to torcheck's robustness of
             formula()): true max and min over the window; a clause that selects nothing is
             dropped, and a formula left empty has robustness -inf
Prediction and checkpointing use the configured one; the other is recorded as a diagnostic
(on BasicMotions and Epilepsy the exact reading lost up to 40 points of balanced accuracy).

Departures from the authors' code, needed for the multiclass, imbalanced UEA datasets and the
comparison's protocol:
  - multiclass by one-vs-rest: one binary network per class (a single one, for class 1, when
    there are two classes), all trained together as one module with a leading model axis on
    every parameter; each model's loss depends only on its own parameters and Adam is
    elementwise, so this equals independent models seeing the same mini-batches;
  - each model's hinge is a weighted mean, with half the weight on its class and half on the
    others, shared equally among them (class-balanced, as stl_tree and bombara);
  - a series goes to the class whose model has the largest robustness (two classes: class 1
    iff its robustness > 0, as theirs), and every eval_every epochs each model keeps its
    state if the weighted train accuracy of its hard network is at least the best so far
    (their rule, per model);
  - the architecture is chosen by cross-validation (choose_architecture), as bombara's depth,
    instead of being set by hand for each dataset.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torcheck.stl import Atom, Eventually, Globally

from .bombara import _and, _or
from .formula_sampler import F0

# n_clauses of the candidate architectures, simplest first (CV ties go to the earlier one):
# None is the authors' Model1 (one clause), m their Model2 with m clauses
ARCHITECTURES = (None, 2, 4)


@dataclass
class TLINetConfig:
    preds_per_channel: int = 3     # nf = preds_per_channel * C, as their nf = dim * 3
    lr: float = 0.1                # Adam
    batch_size: int = 64
    epochs: int = 500
    eval_every: int = 10           # checkpointing period, in epochs
    beta: float = 2.5              # sparse softmax of the temporal operators
    h: float = 1.0
    tau: float = 1.0               # slope of the window ramps (not learned)
    eps_init: float = 0.1
    # regularization weights of their naval notebook: Model1 (one clause) and Model2
    w_bi_1: float = 1.0
    w_l1_1: float = 1e-2
    w_margin_1: float = 1e-2
    w_bi_2: float = 1.0
    w_l1_2: float = 1e-4
    w_margin_2: float = 1e-3
    max_elements: int = 2 ** 25    # (batch x models x nf x T) elements per chunk of models
    semantics: str = "authors"     # robustness for prediction and checkpointing: "authors" or "exact"

    def weights(self, n_clauses: int | None) -> tuple[float, float, float]:
        """(w_bi, w_l1, w_margin) of an architecture."""
        if n_clauses is None:
            return self.w_bi_1, self.w_l1_1, self.w_margin_1
        return self.w_bi_2, self.w_l1_2, self.w_margin_2


# -- differentiable operators, vectorized over (batch, model, ...) -------------------------
def time_weights(T: int, t1: torch.Tensor, t2: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft window weights (..., T) of the intervals [t1, t2] (shape (...)), their TimeFunction."""
    w = torch.arange(T, dtype=t1.dtype, device=t1.device)
    t1, t2 = t1.unsqueeze(-1), t2.unsqueeze(-1)
    f1 = (torch.relu(w - t1 + tau) - torch.relu(w - t1)) / tau
    f2 = (torch.relu(-w + t2 + tau) - torch.relu(-w + t2)) / tau
    return torch.minimum(f1, f2)


def sparse_softmax(x: torch.Tensor, w: torch.Tensor, beta: float, h: float) -> torch.Tensor:
    """Their SparseSoftMax over the last axis: x (..., T), weights w broadcastable to x."""
    ws = w.sum(-1, keepdim=True)
    wn = w / torch.where(ws == 0, torch.ones_like(ws), ws)
    rw = x * wn
    mx = rw.max(-1, keepdim=True).values.abs()
    mx = torch.where(mx == 0, torch.ones_like(mx), mx)
    # exp(beta h rw / mx) / sum: rw / mx <= 1, so their direct form cannot overflow either
    s = torch.softmax(beta * h * rw / mx, dim=-1)
    sw = s * wn
    den = sw.sum(-1)
    den = torch.where(den == 0, torch.ones_like(den), den)
    return (sw * x).sum(-1) / den


def averaged_max(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Their AveragedMax over the last axis: x (..., n), p broadcastable to x. The inputs sorted
    in decreasing order, the i-th weighted by p_i prod_{j<i} (1 - p_j)."""
    xs, idx = torch.sort(x, dim=-1, descending=True)
    ps = torch.gather(p.expand_as(x), -1, idx)
    keep = torch.cat([torch.ones_like(ps[..., :1]), 1 - ps[..., :-1]], dim=-1)
    return (xs * ps * torch.cumprod(keep, dim=-1)).sum(-1)


def logical(x: torch.Tensor, p: torch.Tensor, tvar: torch.Tensor) -> torch.Tensor:
    """Clauses of a logical layer: x (B, k, n) inputs, p (k, m, n), tvar (k, m) -> (B, k, m)."""
    xm = x.unsqueeze(2).expand(-1, -1, p.shape[1], -1)
    return tvar * averaged_max(xm, p) - (1 - tvar) * averaged_max(-xm, p)


def _weighted_mean_p(ps: list) -> torch.Tensor:
    """Per model (leading axis): mean over the tensors of the mean of p (1 - p)."""
    return torch.stack([(q * (1 - q)).flatten(1).mean(1) for q in ps]).mean(0)


# -- the network -----------------------------------------------------------------------
class TLINet(nn.Module):
    """n_models binary TLINet networks of one architecture, on series of C variables, T steps."""

    def __init__(self, n_models: int, C: int, T: int, n_clauses: int | None, cfg: TLINetConfig,
                 generator: torch.Generator):
        super().__init__()
        if T < 2:
            raise ValueError("TLINet needs series of at least 2 time steps")
        K, nf, m = n_models, cfg.preds_per_channel * C, n_clauses or 1
        self.K, self.C, self.T, self.nf, self.n_clauses, self.cfg, self.gen = K, C, T, nf, n_clauses, cfg, generator
        self.register_buffer("var", torch.arange(nf) % C)
        g = generator

        def rand(*shape):
            return nn.Parameter(torch.rand(*shape, generator=g))

        # initialized as theirs: a, b, p, tvar uniform in [0, 1), t1 in [0, T/2), t2 in [T/2, T)
        self.t1 = nn.Parameter(torch.randint(0, T // 2, (K, nf), generator=g).float())
        self.t2 = nn.Parameter(torch.randint(T // 2, T, (K, nf), generator=g).float())
        self.a, self.b = rand(K, nf), rand(K, nf)
        self.tvar_t = rand(K, nf)
        self.p1, self.tvar_l1 = rand(K, m, nf), rand(K, m)
        if n_clauses is not None:
            self.p2, self.tvar_l2 = rand(K, 1, m), rand(K, 1)
        self.eps = nn.Parameter(torch.full((K,), cfg.eps_init))

    @property
    def two_layers(self) -> bool:
        return self.n_clauses is not None

    def probabilities(self) -> list:
        return [self.p1, self.p2] if self.two_layers else [self.p1]

    def type_vars(self) -> list:
        return [self.tvar_t, self.tvar_l1] + ([self.tvar_l2] if self.two_layers else [])

    @torch.no_grad()
    def project(self) -> None:
        """The projection at the start of their forward."""
        self.p1.clamp_(0, 1)
        empty = (self.p1 == 0).all(-1)                                  # (K, m)
        if self.two_layers:
            empty &= self.p2[:, 0, :] > 0.5
        if empty.any():
            self.p1[empty] = torch.rand(int(empty.sum()), self.nf, generator=self.gen).to(self.p1)
        if self.two_layers:
            self.p2.clamp_(0, 1)
            empty2 = (self.p2 == 0).flatten(1).all(1)                   # (K,)
            if empty2.any():
                self.p2[empty2] = torch.rand(int(empty2.sum()), *self.p2.shape[1:], generator=self.gen).to(self.p2)
        for q in self.type_vars():
            q.clamp_(0, 1)
        self.t1.clamp_(0, self.T - 1)
        self.t2.clamp_(0, self.T - 1)
        bad = self.t1 >= self.t2 - 1
        self.t1[bad] = self.t2[bad] - 1
        self.eps[self.eps < 0] = 1e-5

    # -- soft network (training) ---------------------------------------------------------
    def soft(self, x: torch.Tensor, ks: slice) -> torch.Tensor:
        """Robustness (B, k) of the soft networks of models ks on x (B, C, T)."""
        cfg = self.cfg
        z = self.a[ks, :, None] * x[:, self.var, :].unsqueeze(1) - self.b[ks, :, None]   # (B, k, nf, T)
        w = time_weights(self.T, self.t1[ks], self.t2[ks], cfg.tau)                     # (k, nf, T)
        tv = self.tvar_t[ks]
        r = tv * sparse_softmax(z, w, cfg.beta, cfg.h) - (1 - tv) * sparse_softmax(-z, w, cfg.beta, cfg.h)
        r = logical(r, self.p1[ks], self.tvar_l1[ks])                                    # (B, k, m)
        if self.two_layers:
            r = logical(r, self.p2[ks], self.tvar_l2[ks])                                # (B, k, 1)
        return r[..., 0]

    def loss(self, x: torch.Tensor, y: torch.Tensor, sw: torch.Tensor, ks: slice) -> torch.Tensor:
        """Sum over models ks of their loss; y in {-1, 1} and sample weights sw, both (B, K)."""
        w_bi, w_l1, w_margin = self.cfg.weights(self.n_clauses)
        r = self.soft(x, ks)
        y, sw, eps = y[:, ks], sw[:, ks], self.eps[ks]
        hinge = (sw * torch.relu(eps - y * r)).sum(0) / sw.sum(0)
        ps = [q[ks] for q in self.probabilities()]
        reg = w_bi * _weighted_mean_p([q[ks] for q in self.type_vars()] + ps)
        reg = reg + w_l1 * sum(q.flatten(1).sum(1) for q in ps)
        return (hinge - w_margin * eps + reg).sum()

    # -- hard formula (prediction, checkpointing) ----------------------------------------
    @torch.no_grad()
    def predicate_thresholds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(thr, gt), both (K, nf): predicate i reads x_v > thr if gt, else x_v < thr."""
        a = torch.where(self.a == 0, torch.full_like(self.a, 1e-12), self.a)
        return self.b / a, self.a > 0

    @torch.no_grad()
    def windows(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Integer windows [lo, hi] (K, nf) of the predicates, int(t1) and int(t2) as theirs."""
        lo = torch.floor(self.t1).clamp(0, self.T - 1).long()
        hi = torch.maximum(torch.floor(self.t2).clamp(0, self.T - 1).long(), lo)
        return lo, hi

    @torch.no_grad()
    def exact(self, x: torch.Tensor, ks: slice) -> torch.Tensor:
        """Exact STL robustness (B, k) of the formulas of models ks on x (B, C, T)."""
        thr, gt = self.predicate_thresholds()
        thr, gt = thr[ks, :, None], gt[ks, :, None]
        xv = x[:, self.var, :].unsqueeze(1)                                              # (B, 1, nf, T)
        z = torch.where(gt, xv - thr, thr - xv)                                          # (B, k, nf, T)
        lo, hi = self.windows()
        t = torch.arange(self.T, device=x.device)
        inside = (t >= lo[ks, :, None]) & (t <= hi[ks, :, None])                          # (k, nf, T)
        inf = torch.tensor(float("inf"), dtype=z.dtype, device=z.device)
        is_f = self.tvar_t[ks] > 0.5
        r = torch.where(is_f, torch.where(inside, z, -inf).amax(-1), torch.where(inside, z, inf).amin(-1))

        def layer(r, p, tvar):      # (B, k, n), (k, m, n), (k, m) -> (B, k, m) and nonempty (k, m)
            sel = p >= 0.5
            rm = r.unsqueeze(2)
            c_or = torch.where(sel, rm, -inf).amax(-1)
            c_and = torch.where(sel, rm, inf).amin(-1)
            return torch.where(tvar > 0.5, c_or, c_and), sel.any(-1)

        c, nonempty = layer(r, self.p1[ks], self.tvar_l1[ks])
        if self.two_layers:
            p2 = self.p2[ks] * nonempty.unsqueeze(1)             # empty clauses are dropped
            c, nonempty = layer(c, p2, self.tvar_l2[ks])
        return torch.where(nonempty[:, 0], c[..., 0], -inf)

    @torch.no_grad()
    def authors_hard(self, x: torch.Tensor, ks: slice) -> torch.Tensor:
        """Robustness (B, k) of the hard networks of models ks on x by their formula_forward:
        hard F/G, p and AND/OR, sparse softmax over the soft window in time."""
        cfg = self.cfg
        z = self.a[ks, :, None] * x[:, self.var, :].unsqueeze(1) - self.b[ks, :, None]
        w = time_weights(self.T, self.t1[ks], self.t2[ks], cfg.tau)
        r = torch.where(self.tvar_t[ks] > 0.5, sparse_softmax(z, w, cfg.beta, cfg.h),
                        -sparse_softmax(-z, w, cfg.beta, cfg.h))

        def layer(r, p, tvar):
            pb = (p >= 0.5).to(r.dtype)
            rm = r.unsqueeze(2).expand(-1, -1, p.shape[1], -1)
            return torch.where(tvar > 0.5, averaged_max(rm, pb), -averaged_max(-rm, pb))

        c = layer(r, self.p1[ks], self.tvar_l1[ks])
        if self.two_layers:
            c = layer(c, self.p2[ks], self.tvar_l2[ks])
        return c[..., 0]

    @torch.no_grad()
    def formula(self, k: int):
        """torcheck formula of model k (None: empty), the hard reading of exact()."""
        thr, gt = self.predicate_thresholds()
        lo, hi = self.windows()

        def predicate(i: int):
            atom = Atom(int(self.var[i]), float(thr[k, i]), lte=not bool(gt[k, i]))
            op = Eventually if self.tvar_t[k, i] > 0.5 else Globally
            return op(atom, left_time_bound=int(lo[k, i]), right_time_bound=int(hi[k, i]))

        def clause(children: list, tvar):
            return None if not children else (_or if tvar > 0.5 else _and)(children)

        clauses = [clause([predicate(i) for i in torch.nonzero(self.p1[k, j] >= 0.5).ravel().tolist()],
                          self.tvar_l1[k, j]) for j in range(self.p1.shape[1])]
        if not self.two_layers:
            return clauses[0]
        sel = [clauses[j] for j in torch.nonzero(self.p2[k, 0] >= 0.5).ravel().tolist() if clauses[j] is not None]
        return clause(sel, self.tvar_l2[k, 0])


# -- training ----------------------------------------------------------------------------
def _chunks(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def ovr_targets(y_idx: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Targets y in {-1, 1} and sample weights, both (N, n_models), and the class of each
    model: one model per class, or one for class 1 with two classes. Each model's weights put
    half the mass on its class and half on the others, shared equally among them."""
    pos_classes = [1] if n_classes == 2 else list(range(n_classes))
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    Y = np.empty((len(y_idx), len(pos_classes)), dtype=np.float32)
    W = np.empty_like(Y)
    for j, c in enumerate(pos_classes):
        pos = y_idx == c
        Y[:, j] = np.where(pos, 1.0, -1.0)
        W[:, j] = np.where(pos, 0.5 / counts[c], 0.5 / ((n_classes - 1) * counts[y_idx]))
    return Y, W, pos_classes


def _weighted_accuracy(rho: torch.Tensor, Y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Per model, weighted accuracy of the predictions rho > 0."""
    pred = torch.where(rho > 0, 1.0, -1.0)
    return (W * (pred == Y)).sum(0) / W.sum(0)


class TLINetClassifier:
    """TLINet of one architecture (n_clauses: None is one clause, m two layers of m clauses),
    multiclass by one-vs-rest, predicting with the exact semantics of the learned formulas."""

    def __init__(self, n_clauses: int | None = None, cfg: TLINetConfig | None = None,
                 random_state: int = 0, device: str = "cpu"):
        self.n_clauses = n_clauses
        self.cfg = cfg or TLINetConfig()
        self.random_state = random_state
        self.device = device

    def _model_chunk(self, B: int) -> int:
        m = self.model_
        return max(1, self.cfg.max_elements // max(1, B * m.nf * m.T))

    @torch.no_grad()
    def _robustness(self, X: np.ndarray, semantics: str | None = None) -> torch.Tensor:
        """(N, n_models) robustness on X: "authors", "exact" or "soft" (the trained network);
        None is the configured semantics."""
        m = self.model_
        x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        B = max(1, min(len(x), self.cfg.max_elements // max(1, m.nf * m.T)))
        kc = self._model_chunk(B)
        f = {"authors": m.authors_hard, "exact": m.exact, "soft": m.soft}[semantics or self.cfg.semantics]
        return torch.cat([torch.cat([f(x[s], ks) for ks in _chunks(m.K, kc)], dim=1)
                          for s in _chunks(len(x), B)], dim=0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        cfg, dev = self.cfg, self.device
        if cfg.semantics not in ("authors", "exact"):
            raise ValueError(f"unknown semantics {cfg.semantics!r}")
        t0 = time.perf_counter()
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        Yn, Wn, self.model_classes_ = ovr_targets(y_idx, len(self.classes_))
        x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
        Y, W = torch.as_tensor(Yn, device=dev), torch.as_tensor(Wn, device=dev)
        N, C, T = x.shape
        g = torch.Generator().manual_seed(int(self.random_state))
        torch.manual_seed(int(self.random_state))
        self.model_ = m = TLINet(Y.shape[1], C, T, self.n_clauses, cfg, g).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
        bs = min(cfg.batch_size, N)
        kc = self._model_chunk(bs)
        best = torch.full((m.K,), -1.0, device=dev)
        best_epoch = torch.zeros(m.K, dtype=torch.long)
        best_state = {n: q.detach().clone() for n, q in m.named_parameters()}
        for epoch in range(cfg.epochs):
            perm = torch.randperm(N, generator=g).to(dev)
            for idx in _chunks(N, bs):
                b = perm[idx]
                opt.zero_grad()
                m.project()
                for ks in _chunks(m.K, kc):        # gradients of disjoint models: chunking is exact
                    m.loss(x[b], Y[b], W[b], ks).backward()
                opt.step()
            if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
                m.project()                        # their evaluation runs the (projecting) forward
                acc = _weighted_accuracy(self._robustness(X), Y, W)
                better = acc >= best
                best = torch.where(better, acc, best)
                best_epoch[better.cpu()] = epoch + 1
                for n, q in m.named_parameters():
                    best_state[n][better] = q.detach()[better]
        with torch.no_grad():
            for n, q in m.named_parameters():
                q.copy_(best_state[n])
        self.fit_s_ = time.perf_counter() - t0
        self.diagnostics_ = {
            "n_clauses": self.n_clauses,
            "best_epoch": best_epoch.tolist(),
            "train_weighted_acc_models": best.tolist(),
            **{f"bacc_train_{sem}": float(balanced_accuracy_score(y, self.predict(X, sem)))
               for sem in ("soft", "authors", "exact")},
            "n_predicates": [len(self._predicates(k)) for k in range(m.K)],
        }
        return self

    def _decide(self, rho: torch.Tensor) -> np.ndarray:
        rho = rho.cpu().numpy()
        if len(self.classes_) == 2:
            return self.classes_[(rho[:, 0] > 0).astype(int)]
        return self.classes_[np.argmax(rho, axis=1)]

    def robustness(self, X: np.ndarray, semantics: str | None = None) -> np.ndarray:
        """(N, n_models) robustness of the models (configured semantics by default); model j
        is for class classes_[model_classes_[j]]."""
        return self._robustness(X, semantics).cpu().numpy()

    def predict(self, X: np.ndarray, semantics: str | None = None) -> np.ndarray:
        return self._decide(self._robustness(X, semantics))

    # -- formulas --------------------------------------------------------------------------
    def formula(self, j: int):
        """torcheck formula of model j (None: empty, never satisfied)."""
        return self.model_.formula(j)

    def _predicates(self, j: int) -> set:
        """Distinct predicates (variable, F/G, window, direction, threshold) of model j's formula."""
        out = set()

        def walk(node):
            if node is None:
                return
            if isinstance(node, (Eventually, Globally)):
                a = node.child
                out.add((a.var_index, type(node).__name__, node.left_time_bound, node.right_time_bound - 1,
                         a.lte, round(float(a.threshold), 6)))
                return
            walk(node.left_child)
            walk(node.right_child)

        walk(self.formula(j))
        return out

    def formula_sizes(self) -> list[int]:
        """Node count of each model's formula (0 when empty)."""
        return [F0.formula_size(f) if (f := self.formula(j)) is not None else 0 for j in range(self.model_.K)]

    def formula_text(self, j: int) -> str:
        f = self.formula(j)
        return "false" if f is None else str(f)

    def to_json(self) -> dict:
        return {"classes": [str(c) for c in self.classes_], "n_clauses": self.n_clauses,
                "models": [{"class": str(self.classes_[c]), "formula": self.formula_text(j),
                            "size": s} for j, (c, s) in enumerate(zip(self.model_classes_, self.formula_sizes()))],
                "config": asdict(self.cfg)}


def choose_architecture(X: np.ndarray, y: np.ndarray, folds: list, seed: int, cfg: TLINetConfig | None = None,
                        device: str = "cpu", architectures=ARCHITECTURES) -> tuple[int | None, dict]:
    """The architecture with the best mean balanced accuracy over the CV folds (ties to the
    earlier, simpler one), and the mean CV balanced accuracy of each."""
    cv = {}
    for arch in architectures:
        scores = []
        for tr, va in folds:
            clf = TLINetClassifier(arch, cfg, seed, device).fit(X[tr], y[tr])
            scores.append(balanced_accuracy_score(y[va], clf.predict(X[va])))
        cv[arch] = float(np.mean(scores))
    best = max(architectures, key=lambda a: (cv[a], -architectures.index(a)))
    return best, cv
