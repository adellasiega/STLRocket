from __future__ import annotations

import numpy as np
import torch

from torcheck.stl import Atom, Not, And, Or, Globally, Eventually, Until

from .config import ExperimentConfig


def eval_robustness(phi, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        rho = phi.quantitative(
            torch.from_numpy(X), evaluate_at_all_times=False, normalize=False
        )
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


def extract_features(X: np.ndarray, formulas: list) -> np.ndarray:
    with torch.no_grad():
        feats = [
            phi.quantitative(
                torch.from_numpy(X), evaluate_at_all_times=False, normalize=False
            ).detach().cpu().numpy()
            for phi in formulas
        ]
    return np.stack(feats, axis=1)


def build_formula_bank(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Iteratively sample, center, and correlation-filter STL formulas.

    Returns (formulas, X_tr_feats, X_te_feats).
    """
    from stlkernel.distribution_formulae import F0

    np.random.seed(seed)
    torch.manual_seed(seed)

    N, V, T = X_tr.shape
    v_min = np.amin(X_tr, axis=(0, 2))
    v_max = np.amax(X_tr, axis=(0, 2))

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

    formulas: list = []
    X_tr_feats = np.zeros((X_tr.shape[0], 0))
    X_te_feats = np.zeros((X_te.shape[0], 0))

    for it in range(config.max_iter):
        needed = config.n_formulas - len(formulas)
        if needed <= 0:
            break

        batch = generator.sample(max(needed, config.batch_size))

        new_tr = extract_features(X_tr, batch)
        new_te = extract_features(X_te, batch)

        for i, phi in enumerate(batch):
            delta = -new_tr[:, i].mean()
            new_tr[:, i] += delta
            new_te[:, i] += delta
            shift_atom_thresholds(phi, delta)

        combined_tr = np.concatenate([X_tr_feats, new_tr], axis=1)
        combined_te = np.concatenate([X_te_feats, new_te], axis=1)
        combined_formulas = formulas + batch

        corr = np.corrcoef(combined_tr, rowvar=False)
        upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
        drop = [
            i for i in range(corr.shape[1])
            if np.any(np.abs(corr[upper[:, i], i]) > config.threshold_corr)
        ]
        keep = np.setdiff1d(np.arange(corr.shape[1]), drop)

        formulas = [combined_formulas[i] for i in keep]
        X_tr_feats = combined_tr[:, keep]
        X_te_feats = combined_te[:, keep]

        print(
            f"  iter {it}: kept {len(formulas)} / {config.n_formulas} "
            f"(dropped {len(drop)} this round)"
        )

    if len(formulas) > config.n_formulas:
        formulas = formulas[:config.n_formulas]
        X_tr_feats = X_tr_feats[:, :config.n_formulas]
        X_te_feats = X_te_feats[:, :config.n_formulas]

    return formulas, X_tr_feats, X_te_feats
