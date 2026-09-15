from __future__ import annotations
import os
import numpy as np
from glmnet import LogitNet
from glmnet.scorer import make_scorer
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from .config import ExperimentConfig


def available_cpus() -> int:
    """CPUs this process may actually use.

    Prefer SLURM's per-task allocation: on an HPC node os.cpu_count() reports the
    whole machine, not the cgroup, so trusting it oversubscribes the allocation.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    try:  # honours cgroup/affinity limits where available
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def train_classifier(
    X_tr_feats: np.ndarray,
    y_tr: np.ndarray,
    config: ExperimentConfig,
    seed: int | None = None,
) -> LogitNet:
    # glmnet passes n_splits straight to StratifiedKFold, which raises if a class
    # has fewer members than folds (e.g. StandWalkJump: 4 per class), so clamp as
    # the tree head does. Below 3 splits glmnet silently skips CV altogether and
    # leaves lambda untuned, so never go under that floor.
    n_splits = max(3, min(config.cv, int(np.unique(y_tr, return_counts=True)[1].min())))

    model = LogitNet(
        alpha=1.0,          # LASSO
        standardize=False,  # Features are already standardized in features.build_formula_bank
        fit_intercept=False,
        cut_point=1,
        n_splits=n_splits,  # Cross validation for lambda hyperparameter
        # glmnet parallelises the lambda-path scoring one thread PER FOLD, so
        # more threads than folds are simply idle.
        n_jobs=max(1, min(n_splits, available_cpus())),
        max_iter=100_000,
        # Seed for determining cv folds. Use the per-run seed so that repeated
        # runs vary the fold split too; falls back to base_seed when unset.
        random_state=config.base_seed if seed is None else seed,
        scoring=make_scorer(balanced_accuracy_score),
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
