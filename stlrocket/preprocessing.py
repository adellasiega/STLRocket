from __future__ import annotations

import numpy as np


class StateVariableStandardScaler:
    def __init__(self, eps: float = 1e-8):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, X: np.ndarray) -> "StateVariableStandardScaler":
        self.mean = np.mean(X, axis=(0, 2), keepdims=True)
        self.std = np.std(X, axis=(0, 2), keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / (self.std + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
