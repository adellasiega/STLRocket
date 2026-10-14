from __future__ import annotations
import os
import numpy as np
from glmnet import LogitNet
from glmnet.scorer import make_scorer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
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


def cv_n_splits(y_tr: np.ndarray, cv: int) -> int:
    """Number of CV folds used to select lambda."""
    # glmnet passes n_splits straight to StratifiedKFold, which raises if a class
    # has fewer members than folds (e.g. StandWalkJump: 4 per class), so clamp as
    # the tree head does. Below 3 splits glmnet silently skips CV altogether and
    # leaves lambda untuned, so never go under that floor.
    return max(3, min(cv, int(np.unique(y_tr, return_counts=True)[1].min())))


def cv_splitter(y_tr: np.ndarray, cv: int, seed: int) -> StratifiedKFold:
    """The folds glmnet uses inside train_classifier (LogitNet.fit builds exactly this
    splitter from n_splits and random_state), for other models to share them."""
    return StratifiedKFold(n_splits=cv_n_splits(y_tr, cv), shuffle=True, random_state=seed)


def train_classifier(
    X_tr_feats: np.ndarray,
    y_tr: np.ndarray,
    config: ExperimentConfig,
    seed: int | None = None,
) -> LogitNet:
    n_splits = cv_n_splits(y_tr, config.cv)

    model = LogitNet(
        alpha=1.0,          # LASSO
        standardize=False,  # Features are already standardized in features.build_formula_bank
        fit_intercept=config.fit_intercept,
        cut_point=config.cut_point,
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
    lamb: float | None = None,
) -> dict:
    """Balanced accuracy at lamb (default: the selected lambda_best_)."""
    y_pred = np.asarray(model.predict(X_te_feats, lamb=lamb)).reshape(len(X_te_feats), -1)[:, 0]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
    }


def path_balanced_accuracy(model: LogitNet, X_feats: np.ndarray, y: np.ndarray) -> list:
    """Balanced accuracy at every lambda of the path, in path order."""
    y_pred = np.asarray(model.predict(X_feats, lamb=model.lambda_path_)).reshape(len(X_feats), -1)
    return [float(balanced_accuracy_score(y, y_pred[:, i])) for i in range(y_pred.shape[1])]


def selection_diagnostics(model: LogitNet) -> dict:
    """Cross-validation curve over the lambda path and the two candidate lambdas:
    index_1se is the selected one (lambda_best_, sparsest within cut_point SEs of the
    best CV score), index_cvmax the one with the best CV score (lambda_max_).
    n_selected_path counts the formulae with a nonzero weight for at least one class."""
    return {
        "n_splits": int(model.n_splits),
        "n_lambda": int(model.n_lambda_),
        "index_1se": int(np.atleast_1d(model.lambda_best_inx_)[0]),
        "index_cvmax": int(model.lambda_max_inx_),
        "lambda_path": model.lambda_path_.tolist(),
        "cv_mean_score": model.cv_mean_score_.tolist(),
        "cv_standard_error": model.cv_standard_error_.tolist(),
        "n_selected_path": np.any(model.coef_path_ != 0, axis=0).sum(axis=0).tolist(),
    }
