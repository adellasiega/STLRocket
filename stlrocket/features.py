from __future__ import annotations
import numpy as np
import torch
from .formula_sampler import F0
from .stl import LessThan, GreaterThan, Negation, And, Or, Always, Eventually, Until
from .config import ExperimentConfig


_DEFAULT_DEVICE = "cpu"


def set_device(device: str) -> None:
    """Set the default torch device used by eval_robustness/extract_features."""
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


def _as_signal(X: np.ndarray, device: str | None = None) -> torch.Tensor:
    """(N, V, T) numpy array -> (N, T, V) tensor, as expected by stlrocket.stl formulas."""
    return torch.from_numpy(X).permute(0, 2, 1).to(device or _DEFAULT_DEVICE)


def eval_robustness(phi, X: np.ndarray, device: str | None = None) -> np.ndarray:
    signal = _as_signal(X, device)
    with torch.no_grad():
        rho = torch.func.vmap(phi.robustness)(signal)
    return rho.detach().cpu().numpy().ravel()


def shift_atom_thresholds(node, delta: float, sign: int = 1) -> None:
    if isinstance(node, (LessThan, GreaterThan)):
        effective_delta = delta * sign
        if isinstance(node, LessThan):
            node.rhs += effective_delta   # rho = threshold - x
        else:
            node.rhs -= effective_delta   # rho = x - threshold
    elif isinstance(node, Negation):
        shift_atom_thresholds(node.subformula, delta, sign=-sign)
    elif isinstance(node, (And, Or)):
        shift_atom_thresholds(node.subformula1, delta, sign)
        shift_atom_thresholds(node.subformula2, delta, sign)
    elif isinstance(node, (Always, Eventually)):
        shift_atom_thresholds(node.subformula, delta, sign)
    elif isinstance(node, Until):
        shift_atom_thresholds(node.subformula1, delta, sign)
        shift_atom_thresholds(node.subformula2, delta, sign)
    else:
        raise TypeError(f"unknown node type: {type(node).__name__}")


def extract_features(X: np.ndarray, formulas: list, device: str | None = None) -> np.ndarray:
    signal = _as_signal(X, device)
    with torch.no_grad():
        feats = torch.stack(
            [torch.func.vmap(phi.robustness)(signal) for phi in formulas],
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

    v_min = np.percentile(X_tr, 2,  axis=(0, 2))
    v_max = np.percentile(X_tr, 98, axis=(0, 2))

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
    for i, phi in enumerate(formulas):
        delta = -X_tr_feats[:, i].mean()
        X_tr_feats[:, i] += delta
        shift_atom_thresholds(phi, delta)

    X_te_feats = extract_features(X_te, formulas)

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
