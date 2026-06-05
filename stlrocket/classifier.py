from __future__ import annotations
import numpy as np
from glmnet import LogitNet
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from .config import ExperimentConfig


def train_classifier(
    X_tr_feats: np.ndarray,
    y_tr: np.ndarray,
    config: ExperimentConfig,
) -> LogitNet:
    model = LogitNet(
        alpha=1.0,
        n_splits=config.cv,
        cut_point=0,
        fit_intercept=True,
        standardize=False,
        n_jobs=8,
        max_iter=1_000_000,
        verbose=True
    )
    sample_weight = compute_sample_weight("balanced", y_tr)
    model.fit(X_tr_feats, y_tr, sample_weight=sample_weight)
    
    return model


def evaluate_classifier(
    model: LogitNet,
    X_te_feats: np.ndarray,
    y_te: np.ndarray,
) -> dict:
    y_pred = model.predict(X_te_feats)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
    }
