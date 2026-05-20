from __future__ import annotations

import warnings

import numpy as np


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset from the aeon library.

    Returns X_tr, y_tr, X_te, y_te where X arrays have shape (N, V, T).
    """
    from aeon.datasets import load_classification

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        X_tr, y_tr = load_classification(name, split="TRAIN")
        X_te, y_te = load_classification(name, split="TEST")

    return X_tr.astype(np.float64), y_tr, X_te.astype(np.float64), y_te
