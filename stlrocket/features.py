from __future__ import annotations
import numpy as np
import torch
from stlkernel.distribution_formulae import F0
from torcheck.stl import Atom, Not, And, Or, Globally, Eventually, Until
from .config import ExperimentConfig


class FFTFormula:
    """Wraps an STL formula to evaluate on the FFT magnitude spectrum of the input."""

    def __init__(self, formula):
        self.formula = formula

    def quantitative(self, X: torch.Tensor, **kwargs):
        X_fft = torch.abs(torch.fft.rfft(X, dim=2))
        return self.formula.quantitative(X_fft, **kwargs)


def eval_robustness(phi, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        rho = phi.quantitative(
            torch.from_numpy(X), evaluate_at_all_times=False, normalize=False
        )
    return rho.detach().cpu().numpy().ravel()


def shift_atom_thresholds(node, delta: float, sign: int = 1) -> None:
    if isinstance(node, FFTFormula):
        shift_atom_thresholds(node.formula, delta, sign)
    elif isinstance(node, Atom):
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


def extract_features(X: np.ndarray, formulas: list) -> np.ndarray:
    with torch.no_grad():
        feats = [
            phi.quantitative(
                torch.from_numpy(X), evaluate_at_all_times=False, normalize=False
            ).detach().cpu().numpy()
            for phi in formulas
        ]
    return np.stack(feats, axis=1)


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


def _build_fft_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray]:
    X_tr_fft = np.abs(np.fft.rfft(X_tr, axis=2))
    X_te_fft = np.abs(np.fft.rfft(X_te, axis=2))

    N, V, T_fft = X_tr_fft.shape

    v_min = np.percentile(X_tr_fft, 2,  axis=(0, 2))
    v_max = np.percentile(X_tr_fft, 98, axis=(0, 2))

    generator = F0(
        n_vars=V,
        v_min=v_min,
        v_max=v_max,
        t_max=T_fft - 1,
        depth_max=config.depth_max,
        seed=seed + 1,
        only_temporal=config.only_temporal,
        until_weight=config.until_weight,
    )

    inner_formulas = generator.sample(config.n_formulas)
    formulas = [FFTFormula(phi) for phi in inner_formulas]

    X_tr_feats = extract_features(X_tr_fft, inner_formulas)
    for i, phi in enumerate(formulas):
        delta = -X_tr_feats[:, i].mean()
        X_tr_feats[:, i] += delta
        shift_atom_thresholds(phi, delta)

    X_te_feats = extract_features(X_te_fft, inner_formulas)

    return formulas, X_tr_feats, X_te_feats


def build_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Sample and center STL formulas (and optionally FFT-domain formulas).

    Returns (formulas, X_tr_feats, X_te_feats).  When use_fourier=True the
    feature matrices are the horizontal concatenation of raw-domain and
    FFT-domain features; the formula list contains both, with FFTFormula
    wrappers for the FFT-domain entries so that eval_robustness works on raw
    signals throughout.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    formulas, X_tr_feats, X_te_feats = _build_raw_formula_bank(
        X_tr, X_te, config, seed
    )

    if config.use_fourier:
        fft_formulas, X_tr_fft_feats, X_te_fft_feats = _build_fft_formula_bank(
            X_tr, X_te, config, seed
        )
        formulas = formulas + fft_formulas
        X_tr_feats = np.concatenate([X_tr_feats, X_tr_fft_feats], axis=1)
        X_te_feats = np.concatenate([X_te_feats, X_te_fft_feats], axis=1)

    return formulas, X_tr_feats, X_te_feats
