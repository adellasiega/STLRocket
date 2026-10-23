from __future__ import annotations
import numpy as np
import torch
from .formula_sampler import F0
from torcheck.stl import Atom, Not, And, Or, Globally, Eventually, Until
from sklearn.utils.class_weight import compute_sample_weight
from .config import ExperimentConfig


_DEFAULT_DEVICE = "cpu"

# Samples evaluated per forward pass. Until builds (batch, 1, T, T) tensors in
# torcheck, so evaluating the whole dataset at once runs out of memory on long T.
_BATCH_SIZE = 128


def set_device(device: str) -> None:
    """Set the default torch device used by eval_robustness/extract_features."""
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


def _as_signal(X: np.ndarray, device: str | None = None) -> torch.Tensor:
    """(N, V, T) numpy array -> (N, V, T) tensor, torcheck's expected layout."""
    return torch.from_numpy(X).to(device or _DEFAULT_DEVICE)


def _robustness(phi, signal: torch.Tensor, batch_size: int = _BATCH_SIZE) -> torch.Tensor:
    """Robustness at t=0 of every sample, evaluated batch_size samples at a time."""
    with torch.no_grad():
        return torch.cat([
            phi.quantitative(signal[i:i + batch_size], evaluate_at_all_times=False, normalize=False)
            for i in range(0, signal.shape[0], batch_size)
        ])


def eval_robustness(phi, X: np.ndarray, device: str | None = None) -> np.ndarray:
    rho = _robustness(phi, _as_signal(X, device))
    return rho.cpu().numpy().ravel()


def shift_atom_thresholds(node, delta: float, sign: int = 1) -> None:
    if isinstance(node, Atom):
        effective_delta = delta * sign
        if node.lte:
            node.threshold += effective_delta   # rho = threshold - x
        else:
            node.threshold -= effective_delta   # rho = x - threshold
    elif isinstance(node, Not):
        shift_atom_thresholds(node.child, delta, sign=-sign)
    elif isinstance(node, (And, Or)):
        shift_atom_thresholds(node.left_child, delta, sign)
        shift_atom_thresholds(node.right_child, delta, sign)
    elif isinstance(node, (Globally, Eventually)):
        shift_atom_thresholds(node.child, delta, sign)
    elif isinstance(node, Until):
        shift_atom_thresholds(node.left_child, delta, sign)
        shift_atom_thresholds(node.right_child, delta, sign)
    else:
        raise TypeError(f"unknown node type: {type(node).__name__}")


def extract_features(X: np.ndarray, formulas: list, device: str | None = None) -> np.ndarray:
    signal = _as_signal(X, device)
    feats = torch.stack([_robustness(phi, signal) for phi in formulas], dim=1)
    return feats.cpu().numpy()


def _robustness_trace(phi, signal: torch.Tensor, batch_size: int = _BATCH_SIZE) -> torch.Tensor:
    """(N, T') robustness at every time step where phi is defined."""
    with torch.no_grad():
        return torch.cat([
            phi.quantitative(signal[i:i + batch_size], evaluate_at_all_times=True, normalize=False)[:, 0]
            for i in range(0, signal.shape[0], batch_size)
        ])


def _children(node) -> list:
    if isinstance(node, Atom):
        return []
    if isinstance(node, (Not, Globally, Eventually)):
        return [node.child]
    return [node.left_child, node.right_child]


def _atoms(node) -> list:
    return [node] if isinstance(node, Atom) else [a for c in _children(node) for a in _atoms(c)]


def calibrate_thresholds(formulas: list, X: np.ndarray, rng: np.random.Generator) -> None:
    """Reset in place the atom thresholds of every multi-atom formula from the data X.

    Each atom belongs to a largest single-atom subformula psi (e.g. G[0,50](x > c)) that
    sits directly under a binary node. Robustness is linear in the threshold,
    rho_psi(c) = rho_psi(0) + k*c with k = +-1, so psi computes a statistic of X (here the
    window minimum of x) and compares it to c. c is set to a random quantile u of that
    statistic over X, so psi is true on a fraction 1-u of the points where its parent reads
    it: t=0 if no temporal operator is above, every time step otherwise. A threshold drawn
    from the raw values of x instead ignores the operators (G compares c to window minima),
    which leaves many subformulae always true or false and their binary parent decided by
    one child. Thresholds stay in the units of x. Single-atom formulae are skipped: their
    threshold only shifts the feature by a constant, which centring removes.

    Use the fit/train split only. rng is separate from the sampler's stream so that a
    formula's thresholds depend only on its position in the list.
    """
    signal = _as_signal(X)
    for phi in formulas:
        if len(_atoms(phi)) > 1:
            _calibrate_node(phi, signal, rng, read_at_all_times=False)


def _calibrate_node(node, signal: torch.Tensor, rng: np.random.Generator, read_at_all_times: bool) -> None:
    # node has at least two atoms. Its children are read at every time step if node is
    # temporal (Until) or a temporal operator is above it, at t=0 only otherwise.
    all_times = read_at_all_times or isinstance(node, (Globally, Eventually, Until))
    for child in _children(node):
        atoms = _atoms(child)
        if len(atoms) > 1:
            _calibrate_node(child, signal, rng, all_times)
            continue
        # A single-atom subformula is a chain of unary operators down to the atom.
        # rho(c) = rho(0) + k*c: k = +1 for x <= c, -1 for x >= c, flipped by each Not.
        atom = atoms[0]
        n_not, n = 0, child
        while not isinstance(n, Atom):
            n_not += isinstance(n, Not)
            n = n.child
        k = (1 if atom.lte else -1) * (-1) ** n_not
        atom.threshold = 0.0
        rho = _robustness_trace(child, signal)
        if not all_times:
            rho = rho[:, :1]
        q = float(np.quantile(rho.cpu().numpy(), rng.random()))
        atom.threshold = -k * q  # psi true iff rho(0) > q


def _make_generator(X_tr: np.ndarray, config: ExperimentConfig, seed: int) -> F0:
    N, V, T = X_tr.shape
    return F0(
        n_vars=V,
        v_min=np.nanmin(X_tr, axis=(0, 2)),
        v_max=np.nanmax(X_tr, axis=(0, 2)),
        t_max=T - 1,
        depth_max=config.depth_max,
        seed=seed,
        until_weight=config.until_weight,
        average_prob=config.average_prob,
    )


def _center_and_scale(formulas: list, X_tr: np.ndarray, X_te: np.ndarray, y_tr: np.ndarray, F_tr: np.ndarray):
    """Centre every formula on the train series with the robustness shift property, then
    scale its robustness to unit train std, both class-balanced.

    F_tr: train robustness of formulas as sampled. Each formula is shifted in place by
    the class-balanced train mean mu_j of its robustness (phi_j <- phi_j^{-mu_j}), so that
    the robustness of the stored formula has zero class-balanced train mean and the sign
    of a feature is the satisfaction of that formula. Means and stds weight each train
    series by N / (K N_k), the sample weights of the classifier, so that every class
    counts equally whatever its size. Features are recomputed from the shifted formulae
    rather than as F - mu, so that they are exactly what the stored formulae evaluate to.
    Returns (X_tr_feats, X_te_feats, sigma) with x_j = rho(phi_j, .) / sigma_j.
    """
    w = compute_sample_weight("balanced", y_tr)
    for phi, mu in zip(formulas, np.average(F_tr.astype(np.float64), axis=0, weights=w)):
        shift_atom_thresholds(phi, -float(mu))
    X_tr_feats, X_te_feats = extract_features(X_tr, formulas), extract_features(X_te, formulas)
    # The class-balanced mean of the shifted robustness is 0 (up to float rounding), so the
    # class-balanced std is the root of its weighted mean square.
    sigma = np.sqrt(np.average(X_tr_feats.astype(np.float64) ** 2, axis=0, weights=w)).astype(X_tr_feats.dtype)
    # A column constant on train (up to float rounding) carries no information, and
    # dividing by its std would blow up any test value that differs. Set such columns
    # to 0 on both sides, with scale 1.
    constant = sigma <= 1e-6 * np.maximum(np.abs(F_tr).max(axis=0), 1.0)
    sigma[constant] = 1.0
    X_tr_feats, X_te_feats = X_tr_feats / sigma, X_te_feats / sigma
    X_tr_feats[:, constant] = 0.0
    X_te_feats[:, constant] = 0.0
    return X_tr_feats, X_te_feats, sigma


def _build_raw_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    formulas = _make_generator(X_tr, config, seed).sample(config.n_formulas)
    if config.calibrate_thresholds:
        calibrate_thresholds(formulas, X_tr, np.random.default_rng(seed))
    return formulas, *_center_and_scale(formulas, X_tr, X_te, y_tr, extract_features(X_tr, formulas))


def correlation_filter(F: np.ndarray, threshold: float, n_keep: int = 0, block: int = 1000) -> np.ndarray:
    """
        Greedy filter over the columns of F in order: a column is kept if its |Pearson
        correlation| with every previously kept column is < threshold. Constant columns
        are dropped. The first n_keep columns are assumed already kept. Returns kept indices.
    """
    std = F.std(axis=0)
    Z = np.zeros_like(F, dtype=np.float64)
    nz = std > 1e-12
    Z[:, nz] = (F[:, nz] - F[:, nz].mean(axis=0)) / (std[nz] * np.sqrt(len(F)))  # Z.T @ Z = r
    keep = list(range(n_keep))
    for start in range(n_keep, F.shape[1], block):
        cand = np.flatnonzero(nz[start:min(start + block, F.shape[1])]) + start
        if keep:
            cand = cand[(np.abs(Z[:, keep].T @ Z[:, cand]) < threshold).all(axis=0)]
        R = np.abs(Z[:, cand].T @ Z[:, cand])
        alive = np.ones(len(cand), dtype=bool)
        for i in range(len(cand)):
            if alive[i]:
                alive[i + 1:] &= R[i, i + 1:] < threshold
        keep.extend(cand[alive].tolist())
    return np.array(keep, dtype=int)


def build_decorrelated_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    config: ExperimentConfig,
    seed: int,
    threshold: float,
    max_rounds: int = 20,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, int]:
    """Like build_formula_bank, but formulas whose train robustness has |corr| >= threshold
    with an earlier kept formula are dropped and replaced by new samples, until
    config.n_formulas are kept (or max_rounds resampling rounds are spent).

    The formulas come from the same seeded stream as build_formula_bank, so the kept
    bank is that stream filtered. Returns build_formula_bank's tuple plus n_sampled, the
    number of formulas drawn from the stream up to the last kept one.
    """
    np.random.seed(seed)
    set_device(config.device)

    M = config.n_formulas
    generator = _make_generator(X_tr, config, seed)
    rng = np.random.default_rng(seed)

    def sample(n):
        new = generator.sample(n)
        if config.calibrate_thresholds:
            calibrate_thresholds(new, X_tr, rng)
        return new

    formulas = sample(M)
    F_tr = extract_features(X_tr, formulas)
    keep = correlation_filter(F_tr, threshold)
    for _ in range(max_rounds):
        if len(keep) >= M:
            break
        # Draw enough to fill the gap at the acceptance rate seen so far, plus 20%.
        n_old = len(formulas)
        new = sample(int(min(M, max(1000, 1.2 * (M - len(keep)) * n_old / max(len(keep), 1)))))
        formulas += new
        F_tr = np.hstack([F_tr, extract_features(X_tr, new)])
        kept_new = correlation_filter(F_tr[:, np.r_[keep, n_old:len(formulas)]], threshold, n_keep=len(keep))
        keep = np.r_[keep, kept_new[len(keep):] - len(keep) + n_old]

    n_sampled = int(keep[M - 1]) + 1 if len(keep) >= M else len(formulas)
    keep = keep[:M]
    formulas = [formulas[i] for i in keep]
    return formulas, *_center_and_scale(formulas, X_tr, X_te, y_tr, F_tr[:, keep]), n_sampled


def build_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Sample STL formulas and compute their standardized robustness features.

    Returns (formulas, X_tr_feats, X_te_feats, sigma). X_tr and X_te are expected
    standardized per variable (data.standardize_series); y_tr are the train labels, which
    weight the train statistics by class. The returned formulas are the sampled ones
    shifted to zero class-balanced mean train robustness (_center_and_scale), and the
    features are their robustness divided by its class-balanced train std,
    x_j = rho(phi_j, tau) / sigma_j; columns constant on train are 0, with sigma_j = 1.
    """

    # F0 seeds random and torch itself; numpy is seeded here for downstream code.
    np.random.seed(seed)
    # Also the default for later eval_robustness calls (explanations, simplification).
    set_device(config.device)

    return _build_raw_formula_bank(X_tr, X_te, y_tr, config, seed)
