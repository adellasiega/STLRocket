from __future__ import annotations
import numpy as np
import torch
from .formula_sampler import F0
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


def _build_raw_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, V, T = X_tr.shape

    v_min = np.nanmin(X_tr, axis=(0, 2))
    v_max = np.nanmax(X_tr, axis=(0, 2))

    generator = F0(
        n_vars=V,
        v_min=v_min,
        v_max=v_max,
        t_max=T - 1,
        depth_max=config.depth_max,
        seed=seed,
        until_weight=config.until_weight,
    )

    formulas = generator.sample(config.n_formulas)
    X_tr_feats = extract_features(X_tr, formulas)
    X_te_feats = extract_features(X_te, formulas)
    mu = X_tr_feats.mean(axis=0)
    sigma = X_tr_feats.std(axis=0)
    X_tr_feats = (X_tr_feats - mu) / (sigma + 1e-8)
    X_te_feats = (X_te_feats - mu) / (sigma + 1e-8)

    return formulas, X_tr_feats, X_te_feats, mu, sigma


def build_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample STL formulas and compute their standardized robustness features.

    Returns (formulas, X_tr_feats, X_te_feats, mu, sigma), where mu and sigma are
    the per-formula train mean and std used to standardize: raw = feats * (sigma + 1e-8) + mu.
    """

    # F0 seeds random and torch itself; numpy is seeded here for downstream code.
    np.random.seed(seed)
    # Also the default for later eval_robustness calls (explanations, simplification).
    set_device(config.device)

    return _build_raw_formula_bank(X_tr, X_te, config, seed)
