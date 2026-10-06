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


def add_sg_derivative(X: np.ndarray, window_frac: float, polyorder: int = 2) -> np.ndarray:
    """Append the Savitzky-Golay first derivative of every channel: (N, V, T) -> (N, 2V, T).

    The window is window_frac * T, rounded to an odd length in [polyorder + 1, T].
    Each series is filtered on its own, so nothing leaks between samples. mode="interp"
    fits the polynomial to the edge windows instead of padding, since robustness is
    evaluated at t=0 and the first samples weigh a lot.
    """
    from scipy.signal import savgol_filter

    T = X.shape[2]
    w = int(window_frac * T) | 1
    w = min(max(w, polyorder + 1 + polyorder % 2), T if T % 2 else T - 1)
    dX = savgol_filter(X, w, polyorder, deriv=1, axis=-1, mode="interp").astype(X.dtype)
    return np.concatenate([X, dX], axis=1)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset from the aeon library.

    Returns X_tr, y_tr, X_te, y_te where X arrays have shape (N, V, T).
    Missing values are filled per series (forward fill, then back fill).
    """
    from aeon.datasets import load_classification

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        X_tr, y_tr = load_classification(name, split="TRAIN")
        X_te, y_te = load_classification(name, split="TEST")

    X_tr = _fill_missing(X_tr.astype(np.float32))
    X_te = _fill_missing(X_te.astype(np.float32))
    return X_tr, y_tr, X_te, y_te
