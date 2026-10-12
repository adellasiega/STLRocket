from __future__ import annotations
import warnings
import numpy as np


def _fill_missing(X: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along time, then back-fill leading NaNs; all-NaN series become 0."""
    mask = np.isnan(X)
    if not mask.any():
        return X
    T = X.shape[2]
    # Index of the last observed sample at or before each t (forward fill).
    idx = np.where(mask, 0, np.arange(T))
    np.maximum.accumulate(idx, axis=2, out=idx)
    X = np.take_along_axis(X, idx, axis=2)
    # Leading NaNs remain where no earlier sample exists: back-fill them.
    mask = np.isnan(X)
    idx = np.where(mask, T - 1, np.arange(T))
    idx = np.flip(np.minimum.accumulate(np.flip(idx, axis=2), axis=2), axis=2)
    X = np.take_along_axis(X, idx, axis=2)
    return np.nan_to_num(X, nan=0.0)


def standardize_series(X_tr: np.ndarray, X_te: np.ndarray, return_stats: bool = False):
    """Standardize each variable with its mean and std over all train series and time
    steps, so that atom thresholds and robustness are in standard deviations of the
    variable. A variable constant on train is only centred.

    Returns (X_tr, X_te), plus the per-variable (mean, std) used if return_stats: a
    threshold c on variable v is mean[v] + c * std[v] in the original units."""
    m = X_tr.mean(axis=(0, 2), dtype=np.float64, keepdims=True)
    s = X_tr.std(axis=(0, 2), dtype=np.float64, keepdims=True)
    s[s == 0] = 1.0
    out = ((X_tr - m) / s).astype(np.float32), ((X_te - m) / s).astype(np.float32)
    return (*out, m.ravel(), s.ravel()) if return_stats else out


def load_dataset(name: str, return_stats: bool = False):
    """Load a dataset from the aeon library.

    Returns X_tr, y_tr, X_te, y_te where X arrays have shape (N, V, T).
    Missing values are filled per series (forward fill, then back fill), then every
    variable is standardized on the train split (standardize_series). With return_stats,
    also returns the per-variable train (mean, std) of the original series.
    """
    from aeon.datasets import load_classification

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        X_tr, y_tr = load_classification(name, split="TRAIN")
        X_te, y_te = load_classification(name, split="TEST")

    X_tr = _fill_missing(X_tr.astype(np.float32))
    X_te = _fill_missing(X_te.astype(np.float32))
    X_tr, X_te, m, s = standardize_series(X_tr, X_te, return_stats=True)
    return (X_tr, y_tr, X_te, y_te, m, s) if return_stats else (X_tr, y_tr, X_te, y_te)
