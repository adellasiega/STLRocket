from __future__ import annotations
import numpy as np
import torch
from .formula_sampler import F0
from .scl import Fraction
from torcheck.stl import Atom, Not, And, Or, Globally, Eventually, Until
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
    elif isinstance(node, Fraction):
        # rho = frac - p. The child is not shifted: frac is not affine in its thresholds.
        node.p -= delta * sign
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
    if isinstance(node, (Not, Globally, Eventually, Fraction)):
        return [node.child]
    return [node.left_child, node.right_child]


def _atoms(node) -> list:
    return [node] if isinstance(node, Atom) else [a for c in _children(node) for a in _atoms(c)]


def _carriers(node) -> list:
    """Nodes holding a parameter that robustness is affine in: atoms (threshold) and SCL
    Fraction nodes (p). The interior of a Fraction is not looked into."""
    if isinstance(node, (Atom, Fraction)):
        return [node]
    return [c for child in _children(node) for c in _carriers(child)]


def calibrate_thresholds(formulas: list, X: np.ndarray, rng: np.random.Generator) -> None:
    """Reset in place the thresholds of every formula from the data X.

    Parameters are carried by atoms (threshold c) and SCL Fraction nodes (p). Each carrier
    ends a largest single-carrier subformula psi (e.g. G[0,50](x > c)) that sits directly
    under a binary node, under a Fraction or at the root. Robustness is linear in the
    parameter along psi, rho_psi(c) = rho_psi(0) + k*c with k = +-1, so psi computes a
    statistic of X (here the window minimum of x) and compares it to c. c is set to a
    random quantile u of that statistic over X, so psi is true on a fraction 1-u of the
    points where its parent reads it: t=0 if no temporal operator is above, every time
    step otherwise. A threshold drawn from the raw values of x instead ignores the
    operators (G compares c to window minima), which leaves many subformulae always true
    or false and their binary parent decided by one child. Thresholds stay in the units
    of x. The parameter of a psi that is the whole formula is skipped: it only shifts the
    feature by a constant, which standardization removes. The interior of a Fraction is
    always calibrated, since the fraction depends on its thresholds non-linearly.

    Use the fit/train split only. rng is separate from the sampler's stream so that a
    formula's thresholds depend only on its position in the list.
    """
    signal = _as_signal(X)
    for phi in formulas:
        _calibrate_region(phi, signal, rng, read_at_all_times=False, root=True)


def _calibrate_region(node, signal: torch.Tensor, rng: np.random.Generator, read_at_all_times: bool,
                      root: bool = False) -> None:
    carriers = _carriers(node)
    if len(carriers) == 1:
        _calibrate_piece(node, carriers[0], signal, rng, read_at_all_times, root)
        return
    # The children of a node with several carriers are read at every time step if the
    # node is temporal (Until) or a temporal operator is above it, at t=0 only otherwise.
    all_times = read_at_all_times or isinstance(node, (Globally, Eventually, Until))
    for child in _children(node):
        _calibrate_region(child, signal, rng, all_times)


def _calibrate_piece(piece, carrier, signal: torch.Tensor, rng: np.random.Generator, read_at_all_times: bool,
                     root: bool) -> None:
    # piece is a chain of unary operators down to its only carrier.
    if isinstance(carrier, Fraction):
        # The Fraction reads its child at every time step.
        _calibrate_region(carrier.child, signal, rng, read_at_all_times=True)
    if root:
        return
    # rho(c) = rho(0) + k*c: k = +1 for x <= c, -1 for x >= c and for Fraction
    # (frac - p), flipped by each Not on the chain.
    n_not, n = 0, piece
    while n is not carrier:
        n_not += isinstance(n, Not)
        n = n.child
    k = (1 if isinstance(carrier, Atom) and carrier.lte else -1) * (-1) ** n_not
    attr = "threshold" if isinstance(carrier, Atom) else "p"
    setattr(carrier, attr, 0.0)
    rho = _robustness_trace(piece, signal)
    if not read_at_all_times:
        rho = rho[:, :1]
    rho = rho.cpu().numpy()
    u = rng.random()
    q = float(np.quantile(rho, u))
    if isinstance(carrier, Fraction):
        # Fractions take few distinct values, so q usually falls on a tie, and
        # "rho(0) > q" would make the whole tied group false (e.g. every series with
        # fraction 0). Cut halfway to the next distinct value below q (group true) or
        # above it (group false), whichever true share is closer to the target 1-u.
        lo, hi = rho[rho < q], rho[rho > q]
        cuts = [(q + lo.max()) / 2 if lo.size else q - 0.5, (q + hi.min()) / 2 if hi.size else q + 0.5]
        q = min(cuts, key=lambda c: abs((rho > c).mean() - (1 - u)))
    setattr(carrier, attr, -k * q)  # psi true iff rho(0) > q


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
        scl_weight=config.scl_weight,
    )


def _standardize(X_tr_feats: np.ndarray, X_te_feats: np.ndarray):
    mu = X_tr_feats.mean(axis=0)
    sigma = X_tr_feats.std(axis=0)
    X_tr_feats = (X_tr_feats - mu) / (sigma + 1e-8)
    X_te_feats = (X_te_feats - mu) / (sigma + 1e-8)
    # A column constant on train (up to float rounding) carries no information, but
    # dividing by sigma + 1e-8 blows any test value that differs up to ~1e8 (e.g. an SCL
    # fraction that is 0 on every train series). Set such columns to 0 on both sides.
    constant = sigma <= 1e-6 * np.maximum(np.abs(mu), 1.0)
    X_tr_feats[:, constant] = 0.0
    X_te_feats[:, constant] = 0.0
    return X_tr_feats, X_te_feats, mu, sigma


def _build_raw_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    formulas = _make_generator(X_tr, config, seed).sample(config.n_formulas)
    if config.calibrate_thresholds:
        calibrate_thresholds(formulas, X_tr, np.random.default_rng(seed))
    return formulas, *_standardize(extract_features(X_tr, formulas), extract_features(X_te, formulas))


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
    config: ExperimentConfig,
    seed: int,
    threshold: float,
    max_rounds: int = 20,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
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
    F_tr, F_te = extract_features(X_tr, formulas), extract_features(X_te, formulas)
    keep = correlation_filter(F_tr, threshold)
    for _ in range(max_rounds):
        if len(keep) >= M:
            break
        # Draw enough to fill the gap at the acceptance rate seen so far, plus 20%.
        n_old = len(formulas)
        new = sample(int(min(M, max(1000, 1.2 * (M - len(keep)) * n_old / max(len(keep), 1)))))
        formulas += new
        F_tr = np.hstack([F_tr, extract_features(X_tr, new)])
        F_te = np.hstack([F_te, extract_features(X_te, new)])
        kept_new = correlation_filter(F_tr[:, np.r_[keep, n_old:len(formulas)]], threshold, n_keep=len(keep))
        keep = np.r_[keep, kept_new[len(keep):] - len(keep) + n_old]

    n_sampled = int(keep[M - 1]) + 1 if len(keep) >= M else len(formulas)
    keep = keep[:M]
    return [formulas[i] for i in keep], *_standardize(F_tr[:, keep], F_te[:, keep]), n_sampled


def build_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample STL formulas and compute their standardized robustness features.

    Returns (formulas, X_tr_feats, X_te_feats, mu, sigma), where mu and sigma are
    the per-formula train mean and std used to standardize: raw = feats * (sigma + 1e-8) + mu,
    except for columns constant on train, which are 0.
    """

    # F0 seeds random and torch itself; numpy is seeded here for downstream code.
    np.random.seed(seed)
    # Also the default for later eval_robustness calls (explanations, simplification).
    set_device(config.device)

    return _build_raw_formula_bank(X_tr, X_te, config, seed)
