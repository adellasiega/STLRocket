from __future__ import annotations
import numpy as np
import torch
from .formula_sampler import F0
from torcheck.stl import Atom, Not, And, Or, Globally, Eventually, Until
from .config import ExperimentConfig


_DEFAULT_DEVICE = "cpu"


def set_device(device: str) -> None:
    """Set the default torch device used by eval_robustness/extract_features."""
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


def _as_signal(X: np.ndarray, device: str | None = None) -> torch.Tensor:
    """(N, V, T) numpy array -> (N, V, T) tensor, torcheck's expected layout."""
    return torch.from_numpy(X).to(device or _DEFAULT_DEVICE)


def eval_robustness(phi, X: np.ndarray, device: str | None = None) -> np.ndarray:
    signal = _as_signal(X, device)
    with torch.no_grad():
        rho = phi.quantitative(signal, evaluate_at_all_times=False, normalize=False)
    return rho.detach().cpu().numpy().ravel()


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
    with torch.no_grad():
        feats = torch.stack(
            [
                phi.quantitative(signal, evaluate_at_all_times=False, normalize=False)
                for phi in formulas
            ],
            dim=1,
        )
    return feats.detach().cpu().numpy()


def _build_raw_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray]:
    N, V, T = X_tr.shape

    v_min = np.min(X_tr, axis=(0, 2))
    v_max = np.max(X_tr, axis=(0, 2))

    generator = F0(
        n_vars=V,
        v_min=v_min,
        v_max=v_max,
        t_max=T - 1,
        depth_max=config.depth_max,
        seed=seed,
        only_temporal=config.only_temporal,
        until_weight=config.until_weight,
    )

    formulas = generator.sample(config.n_formulas)

    X_tr_feats = extract_features(X_tr, formulas)
    X_te_feats = extract_features(X_te, formulas)

    # Standardize features (eq. 3). Formulas are left untouched -- reported/
    # reparametrized formulas (explanations.py::reparametrize_formula) always
    # recompute their own threshold shift from raw robustness medians, so a
    # constant additive/multiplicative pre-shift of the base formula has no
    # effect on that downstream result.
    mu = X_tr_feats.mean(axis=0)
    sigma = X_tr_feats.std(axis=0)
    X_tr_feats = (X_tr_feats - mu) / (sigma + 1e-8)
    X_te_feats = (X_te_feats - mu) / (sigma + 1e-8)

    return formulas, X_tr_feats, X_te_feats


def build_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Sample and center STL formulas.

    Returns (formulas, X_tr_feats, X_te_feats).
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    return _build_raw_formula_bank(X_tr, X_te, config, seed)
