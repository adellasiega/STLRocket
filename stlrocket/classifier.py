from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, classification_report

from .config import ExperimentConfig


def train_classifier(
    X_tr_feats: np.ndarray,
    y_tr: np.ndarray,
    config: ExperimentConfig,
) -> LogisticRegressionCV:
    model = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 10),
        l1_ratios=[1.0],
        fit_intercept=False,
        solver="saga",
        cv=config.cv,
        max_iter=config.max_iter_lr,
        class_weight="balanced",
        use_legacy_attributes=False,
    )
    model.fit(X_tr_feats, y_tr)
    return model


def evaluate_classifier(
    model: LogisticRegressionCV,
    X_te_feats: np.ndarray,
    y_te: np.ndarray,
) -> dict:
    y_pred = model.predict(X_te_feats)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
        "accuracy": float(model.score(X_te_feats, y_te)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            str(cls): {
                "precision": round(v["precision"], 6),
                "recall": round(v["recall"], 6),
                "f1": round(v["f1-score"], 6),
                "support": int(v["support"]),
            }
            for cls, v in report.items()
            if cls not in ("accuracy", "macro avg", "weighted avg")
        },
    }
