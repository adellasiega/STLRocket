"""ROGE, after Nenzi, Silvetti, Bartocci and Bortolussi, "A Robust Genetic Algorithm for Learning
Temporal Specifications from Data" (QEST 2018): a baseline of the comparison that learns one STL
formula per binary problem with a genetic algorithm on the formula structure and GP-UCB on its
parameters, both maximizing the discrimination G = (mean rho_p - mean rho_n) / (sd rho_p + sd rho_n),
then shifts the formula's thresholds so that rho > 0 separates the two classes.

The method itself is the package roge (https://github.com/LogArtLab/roge, MIT), used unchanged:
roge.roge for Alg. 1 with its default settings (RogeConfig: ng = 3 generations, 10 random
formulae, alpha = 0.1, GP-UCB with 20 initial points and 50 iterations) and roge.make_classifier
for the threshold shift (its default "weighted" alpha). Install with
    pip install --no-deps --ignore-requires-python git+https://github.com/LogArtLab/roge@1391e33489c9ddc0a7b08f92be807349534b4cf6

Departures, needed for the multiclass, multivariate UEA datasets:
  - multiclass by one-vs-rest: one ROGE per class, positive = the class, negative = all the
    others (a single one, for class 1, when there are two classes), fitted in parallel; a series
    goes to the class whose shifted formula has the largest robustness (two classes: class 1 iff
    rho > 0, as ROGE). The series are standardized per variable (load_dataset), so robustness is
    in comparable units across the per-class formulas. G compares the two class means, so it does
    not depend on the sizes of the two sides; within "rest" each class weighs by its count;
  - population size: ROGE keeps all its initial templates (F mu, G mu over atoms and pairs of
    atoms, mu1 U mu2: O(C^2) of them on C variables) plus 10 random formulae, which is what the
    paper does for few variables; when that exceeds ne_max (default 54, the population of the
    paper's 2-variable naval experiment) ROGE's own option ne = ne_max samples ne_max - 10 of the
    templates. LazyTemplates gives ROGE the same template list without building it, so that the
    sampling stays possible on hundreds of variables;
  - time is the sample index (times = 0, 1, ..., T - 1), as for the other STL baselines.
ROGE has no capacity hyperparameter to tune, so there is no cross-validation.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, field

import numpy as np

# children of the initial population that are random formulae (RogeConfig.n_random default)
N_RANDOM = 10


@dataclass
class ROGEOptions:
    ng: int = 3            # generations (RogeConfig.ng default)
    ne_max: int = 54       # population cap: 44 templates + 10 random on the 2-variable naval data
    alpha_rule: str = "weighted"   # threshold shift (make_classifier default)
    gpucb: dict = field(default_factory=dict)   # overrides of roge.GPUCBOptions (tests only)


def variable_names(C: int) -> tuple[str, ...]:
    return tuple(f"x{c}" for c in range(C))


def to_traces(X: np.ndarray):
    from roge import Traces
    X = np.asarray(X, dtype=float)
    return Traces(np.arange(X.shape[2], dtype=float), X, variable_names(X.shape[1]))


def n_templates(C: int) -> int:
    """len(roge.genetic.initial_templates) on C variables: a = 2C atoms, mu over the a atoms and
    the And and Or of the a(a-1)/2 unordered pairs, F and G of every mu, and U over the a(a-1)
    ordered pairs."""
    a = 2 * C
    n_mu = a + a * (a - 1)
    return 2 * n_mu + a * (a - 1)


class LazyTemplates:
    """roge.genetic.initial_templates(variables) as a sequence that builds each template on
    access, in the same order (checked by tests/test_roge_baseline.py): with C variables the
    list has O(C^2) entries, millions on PEMS-SF or DuckDuckGeese, of which ROGE samples ne."""

    def __init__(self, variables):
        from roge.genetic import all_atoms
        self._atoms = all_atoms(variables)
        a = len(self._atoms)
        self._a = a
        self._n_mu = a + a * (a - 1)
        # first flat index of the unordered pairs (i, j > i) with first element i
        self._pair_start = np.concatenate([[0], np.cumsum(np.arange(a - 1, 0, -1))]).astype(np.int64)
        self._len = 2 * self._n_mu + a * (a - 1)

    def __len__(self) -> int:
        return self._len

    def _atom(self, i: int):
        return self._atoms[i].copy()

    def _mu(self, m: int):
        from roge.stl import And, Or
        if m < self._a:
            return self._atom(m)
        q = m - self._a
        p, is_or = divmod(q, 2)
        i = int(np.searchsorted(self._pair_start, p, side="right") - 1)
        j = i + 1 + int(p - self._pair_start[i])
        return (Or if is_or else And)(self._atom(i), self._atom(j))

    def __getitem__(self, k: int):
        from roge.stl import Eventually, Globally, Until
        k = int(k)
        if k < 0:
            k += self._len
        if not 0 <= k < self._len:
            raise IndexError(k)
        if k < 2 * self._n_mu:
            m, is_g = divmod(k, 2)
            return (Globally if is_g else Eventually)(self._mu(m))
        i, r = divmod(k - 2 * self._n_mu, self._a - 1)
        j = r if r < i else r + 1
        return Until(self._atom(i), self._atom(j))

    def __iter__(self):
        return (self[k] for k in range(self._len))


@contextlib.contextmanager
def lazy_templates():
    """Make roge's generate_initial_formulae read LazyTemplates (same list, built on access)."""
    rr = roge_module()
    original = rr.genetic.initial_templates
    rr.genetic.initial_templates = LazyTemplates
    try:
        yield
    finally:
        rr.genetic.initial_templates = original


def roge_module():
    """The module roge.roge (the package's attribute roge is the function of the same name)."""
    import importlib
    return importlib.import_module("roge.roge")


def _model_seed(seed: int, j: int) -> int:
    return int(np.random.SeedSequence([seed, j]).generate_state(1)[0])


def _fit_one(X: np.ndarray, positive: np.ndarray, seed: int, opts: ROGEOptions) -> dict:
    """One binary ROGE (positive vs the rest), in a worker: the shifted formula and diagnostics."""
    from roge import GPUCBOptions, RogeConfig, make_classifier, roge
    from roge.roge import ParameterSpace
    traces = to_traces(X)
    Dp, Dn = traces.subset(np.flatnonzero(positive)), traces.subset(np.flatnonzero(~positive))
    C = X.shape[1]
    ne = None if n_templates(C) + N_RANDOM <= opts.ne_max else opts.ne_max
    cfg = RogeConfig(ng=opts.ng, ne=ne, seed=seed, gpucb=GPUCBOptions(**opts.gpucb))
    t = time.perf_counter()
    # one BLAS thread per model, as the roge README advises (small matrices; the models
    # already run in parallel over the task's CPUs)
    from threadpoolctl import threadpool_limits
    # ne=None: ROGE keeps every template and concatenates the list with its random formulae,
    # which LazyTemplates does not support; the full list is small then, so build it as ROGE does
    lazy = lazy_templates() if ne is not None else contextlib.nullcontext()
    with lazy, threadpool_limits(1):
        res = roge(Dp, Dn, space=ParameterSpace.from_data(traces), cfg=cfg)
    clf = make_classifier(res.best, rule=opts.alpha_rule, Dp=Dp, Dn=Dn)
    return {"classifier": clf, "fit_s": time.perf_counter() - t, "G": float(res.best.G),
            "learned_formula": str(res.best.formula), "formula": str(clf.formula),
            "size": int(clf.formula.size()), "ne": len(res.population), "ne_option": ne,
            "seed": seed, "history": res.history}


class ROGEClassifier:
    """ROGE, multiclass by one-vs-rest (module docstring)."""

    def __init__(self, opts: ROGEOptions | None = None, seed: int = 0, n_jobs: int = 1):
        self.opts, self.seed, self.n_jobs = opts or ROGEOptions(), seed, n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray):
        from joblib import Parallel, delayed
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        self.model_classes_ = [1] if K == 2 else list(range(K))
        jobs = [delayed(_fit_one)(X, y == self.classes_[k], _model_seed(self.seed, j), self.opts)
                for j, k in enumerate(self.model_classes_)]
        n_jobs = max(1, min(self.n_jobs, len(jobs)))
        self.models_ = Parallel(n_jobs=n_jobs)(jobs) if n_jobs > 1 else [j[0](*j[1], **j[2]) for j in jobs]
        return self

    def robustness(self, X: np.ndarray) -> np.ndarray:
        """(N, n_models) robustness of the shifted formulas; column j is for class
        classes_[model_classes_[j]]."""
        traces = to_traces(X)
        return np.stack([m["classifier"].robustness(traces) for m in self.models_], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        rho = self.robustness(X)
        if len(self.classes_) == 2:
            return self.classes_[np.where(rho[:, 0] > 0, 1, 0)]
        return self.classes_[np.argmax(rho, axis=1)]

    def formula_sizes(self) -> list[int]:
        return [m["size"] for m in self.models_]

    def to_json(self) -> dict:
        return {"options": asdict(self.opts), "seed": self.seed,
                "models": [{"class": self.classes_[k].item() if hasattr(self.classes_[k], "item")
                            else self.classes_[k],
                            **{key: v for key, v in m.items() if key != "classifier"}}
                           for k, m in zip(self.model_classes_, self.models_)]}
