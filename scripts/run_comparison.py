#!/usr/bin/env python
"""
Test-set comparison of STLRocket's classifier with other classifiers (no explanations). Every
method is trained on the official train split and evaluated on the official test split of
each dataset, both preprocessed as in the experiments (stlrocket.data.load_dataset: missing
values filled, each variable standardized on train). Run r uses seed = base_seed + r for
every random choice of every method.

    python scripts/run_comparison.py --datasets ERing --runs 0 1 [--methods stlrocket minirocket ...]

Methods (METHODS):
  stlrocket          the pipeline of the experiments: formula bank (depth, M) and lasso
                     (classifier.train_classifier: no intercept, balanced weights,
                     stratified CV on balanced accuracy, cut_point standard errors)
  stl_tree           DecisionTreeClassifier(class_weight="balanced") on the same STL feature
                     matrix as stlrocket (built once per run and shared), max_depth in
                     stlrocket.tree.DEPTHS (1, 3, 5, 7 or unlimited) chosen by CV on balanced
                     accuracy over the same folds, ties to the shallower tree
  stl_ridge          ablation: the same STL feature matrix with the classifier of
                     MiniRocketClassifier, StandardScaler(with_mean=False) and
                     RidgeClassifierCV(alphas=logspace(-3, 3, 10)); differs from minirocket only
                     in the feature map and from stlrocket only in the classifier
  minirocket         aeon MiniRocketClassifier, default parameters; series shorter than 9
                     time steps (PenDigits) are zero-padded to 9, as for every Rocket method
  multirockethydra   aeon MultiRocketHydraClassifier, default parameters
  mrseql             aeon MrSEQLClassifier, default parameters; univariate only, so on
                     multivariate data one MrSEQL per channel, fitted in parallel, with
                     probabilities summed as in aeon's ClassifierChannelEnsemble; recorded
                     as "mode"
  minirocket_lasso   ablation: aeon MiniRocket transform fitted on train, each feature
                     standardized with its class-balanced train mean and std, as the STL
                     features (constant features set to 0), then exactly the lasso of
                     stlrocket; only the feature map differs

Written to OUT_DIR/<dataset>/run<r>/:
  <method>.json      balanced accuracy and accuracy on train and test, fit and predict times,
                     and per-method details (stlrocket and minirocket_lasso: lambda and
                     number of selected features; stl_tree: chosen depth (null: unlimited), depth,
                     leaves and CV scores; stl_ridge: alpha; mrseql: mode). Written last, so it marks the method as complete
  formulas.pkl.gz,   the formula bank and feature scales (sigma) of the run, written with the
  bank_stats.npz     STL features
  stl_tree.pkl       the fitted tree, which with the bank gives the STL formula of each leaf
                     (stlrocket.tree.path_formulas)

Methods whose json exists are skipped. The STL feature matrix is computed only if a method
needing it is not complete; with --feature_cache it is also saved, and loaded by later jobs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stlrocket.classifier import available_cpus, train_classifier
from stlrocket.config import ExperimentConfig
from stlrocket.data import load_dataset
from stlrocket.features import build_formula_bank
from stlrocket.tree import train_tree

ROCKET_MIN_LENGTH = 9   # shortest series MiniRocket and MultiRocket accept
METHODS = ("stlrocket", "stl_tree", "stl_ridge", "minirocket", "multirockethydra", "mrseql", "minirocket_lasso")
# settings of the experiments each method depends on, recorded in its json
LASSO_SETTINGS = {"stlrocket": ("depth", "n_formulas", "cv", "cut_point"), "stl_tree": ("depth", "n_formulas", "cv"),
                  "minirocket_lasso": ("cv", "cut_point")}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scores(y_tr, p_tr, y_te, p_te) -> dict:
    return {"bacc_train": float(balanced_accuracy_score(y_tr, p_tr)),
            "bacc_test": float(balanced_accuracy_score(y_te, p_te)),
            "acc_train": float(np.mean(np.asarray(y_tr) == np.asarray(p_tr))),
            "acc_test": float(np.mean(np.asarray(y_te) == np.asarray(p_te)))}


def lasso_predict(model, F):
    return np.asarray(model.predict(F)).reshape(len(F), -1)[:, 0]


def lasso_details(model) -> dict:
    coef = np.asarray(model.coef_)
    return {"lambda": float(np.atleast_1d(model.lambda_best_)[0]), "n_features": int(coef.shape[1]),
            "n_selected": int(np.any(coef != 0, axis=0).sum())}


class STLFeatures:
    """The STL feature matrix of a run, built at most once (or loaded from the cache)."""

    def __init__(self, args, out: Path, X_tr, y_tr, X_te, cfg: ExperimentConfig, seed: int, stem: str):
        self.args, self.out, self.X_tr, self.y_tr, self.X_te, self.cfg, self.seed = args, out, X_tr, y_tr, X_te, cfg, seed
        self.cache = Path(args.feature_cache) / f"{stem}.npz" if args.feature_cache else None
        self._F = None

    def get(self):
        if self._F is None:
            t = time.perf_counter()
            if self.cache is not None and self.cache.exists():
                z = np.load(self.cache)
                self._F = (z["F_tr"], z["F_te"], float(z["time_s"]))
                log(f"  STL features loaded from {self.cache}")
            else:
                formulas, F_tr, F_te, sigma = build_formula_bank(self.X_tr, self.X_te, self.y_tr, self.cfg, self.seed)
                elapsed = time.perf_counter() - t
                with gzip.open(self.out / "formulas.pkl.gz", "wb") as f:
                    pickle.dump(formulas, f)
                np.savez(self.out / "bank_stats.npz", sigma=sigma)
                if self.cache is not None:
                    self.cache.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(self.cache, F_tr=F_tr, F_te=F_te, time_s=elapsed)
                self._F = (F_tr, F_te, elapsed)
                log(f"  STL features: {F_tr.shape[1]} formulae in {elapsed:.0f} s")
        return self._F


def run_stlrocket(ctx) -> dict:
    F_tr, F_te, feat_s = ctx["stl"].get()
    t = time.perf_counter()
    model = train_classifier(F_tr, ctx["y_tr"], ctx["cfg"], seed=ctx["seed"])
    fit_s = time.perf_counter() - t
    t = time.perf_counter()
    p_te = lasso_predict(model, F_te)
    return {**scores(ctx["y_tr"], lasso_predict(model, F_tr), ctx["y_te"], p_te),
            "features_s": feat_s, "fit_s": fit_s, "predict_s": time.perf_counter() - t, **lasso_details(model)}


def run_stl_tree(ctx) -> dict:
    F_tr, F_te, feat_s = ctx["stl"].get()
    t = time.perf_counter()
    search = train_tree(F_tr, ctx["y_tr"], ctx["args"].cv, ctx["seed"], n_jobs=ctx["n_jobs"])
    fit_s = time.perf_counter() - t
    tree = search.best_estimator_
    with open(ctx["out"] / "stl_tree.pkl", "wb") as f:
        pickle.dump(tree, f)
    t = time.perf_counter()
    p_te = tree.predict(F_te)
    return {**scores(ctx["y_tr"], tree.predict(F_tr), ctx["y_te"], p_te),
            "features_s": feat_s, "fit_s": fit_s, "predict_s": time.perf_counter() - t,
            "max_depth": search.best_params_["max_depth"], "depth": int(tree.get_depth()),
            "n_leaves": int(tree.get_n_leaves()), "n_splits": int(search.n_splits_),
            "cv_bacc": {str(d): float(s) for d, s in zip(search.cv_results_["param_max_depth"],
                                                        search.cv_results_["mean_test_score"])}}


def run_stl_ridge(ctx) -> dict:
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    F_tr, F_te, feat_s = ctx["stl"].get()
    t = time.perf_counter()
    # the classifier of aeon's MiniRocketClassifier with its defaults (no class weights)
    clf = make_pipeline(StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    clf.fit(F_tr, ctx["y_tr"])
    fit_s = time.perf_counter() - t
    t = time.perf_counter()
    p_te = clf.predict(F_te)
    return {**scores(ctx["y_tr"], clf.predict(F_tr), ctx["y_te"], p_te),
            "features_s": feat_s, "fit_s": fit_s, "predict_s": time.perf_counter() - t,
            "alpha": float(clf[-1].alpha_)}


def pad_series(X_tr, X_te, min_length: int):
    """Zero-pad the series at the end up to min_length time steps (MiniRocket and MultiRocket
    need at least ROCKET_MIN_LENGTH, more than PenDigits has). The series are standardized,
    so 0 is the train mean of each variable. Returns X_tr, X_te and the details to record."""
    T = X_tr.shape[2]
    if T >= min_length:
        return X_tr, X_te, {}
    pad = ((0, 0), (0, 0), (0, min_length - T))
    return np.pad(X_tr, pad), np.pad(X_te, pad), {"padded_from": int(T), "padded_to": int(min_length)}


def run_aeon(ctx, make, min_length: int = 0) -> dict:
    """Fit and evaluate the aeon classifier make(ctx) returns, as (classifier, details), on
    the series zero-padded to min_length (pad_series)."""
    clf, details = make(ctx)
    X_tr, X_te, padded = pad_series(ctx["X_tr"], ctx["X_te"], min_length)
    t = time.perf_counter()
    clf.fit(X_tr, ctx["y_tr"])
    fit_s = time.perf_counter() - t
    t = time.perf_counter()
    p_te = clf.predict(X_te)
    predict_s = time.perf_counter() - t
    return {**scores(ctx["y_tr"], clf.predict(X_tr), ctx["y_te"], p_te),
            "fit_s": fit_s, "predict_s": predict_s, **details, **padded}


def make_minirocket(ctx):
    from aeon.classification.convolution_based import MiniRocketClassifier
    return MiniRocketClassifier(n_jobs=ctx["n_jobs"], random_state=ctx["seed"]), {}


def make_multirockethydra(ctx):
    from aeon.classification.convolution_based import MultiRocketHydraClassifier
    return MultiRocketHydraClassifier(n_jobs=ctx["n_jobs"], random_state=ctx["seed"]), {}


def _channel_proba(make, X_tr, y_tr, X_te):
    """Fit make() on one channel; its classes and class probabilities on train and test."""
    clf = make().fit(X_tr, y_tr)
    return list(clf.classes_), clf.predict_proba(X_tr), clf.predict_proba(X_te)


def channel_ensemble(make, X_tr, y_tr, X_te, n_jobs: int, seed: int):
    """Train and test predictions of aeon's ClassifierChannelEnsemble(channels=["all-split"],
    majority_vote=False, random_state=seed), with the channels fitted in parallel (aeon fits
    them one after the other): one make() per channel, class probabilities summed, ties
    broken at random as aeon does. Each worker fits and predicts its own channel and returns
    only probabilities, so the fitted models never have to be pickled."""
    from joblib import Parallel, delayed
    from sklearn.utils import check_random_state
    out = Parallel(n_jobs=n_jobs)(delayed(_channel_proba)(make, X_tr[:, [c]], y_tr, X_te[:, [c]])
                                  for c in range(X_tr.shape[1]))
    classes = np.unique(y_tr)
    assert all(o[0] == list(classes) for o in out)

    def predict(i):
        dists = sum(o[i] for o in out)
        rng = check_random_state(seed)   # a fresh generator per predict call, as in aeon
        return np.array([classes[int(rng.choice(np.flatnonzero(p == p.max())))]
                         for p in dists / dists.sum(axis=1, keepdims=True)])

    return predict(1), predict(2)


def run_mrseql(ctx) -> dict:
    """MrSEQLClassifier with default parameters; it is univariate only, so on multivariate
    data one per channel (channel_ensemble). The times of the channel ensemble include
    prediction, which runs in the same workers."""
    from aeon.classification.dictionary_based import MrSEQLClassifier
    if MrSEQLClassifier.get_class_tag("capability:multivariate") or ctx["X_tr"].shape[1] == 1:
        return run_aeon(ctx, lambda _: (MrSEQLClassifier(), {"mode": "native"}))
    t = time.perf_counter()
    p_tr, p_te = channel_ensemble(MrSEQLClassifier, ctx["X_tr"], ctx["y_tr"], ctx["X_te"], ctx["n_jobs"],
                                  ctx["seed"])
    return {**scores(ctx["y_tr"], p_tr, ctx["y_te"], p_te), "fit_s": time.perf_counter() - t, "predict_s": None,
            "mode": "channel_ensemble", "n_channels": int(ctx["X_tr"].shape[1])}


def run_minirocket_lasso(ctx) -> dict:
    from aeon.transformations.collection.convolution_based import MiniRocket
    t = time.perf_counter()
    transform = MiniRocket(n_jobs=ctx["n_jobs"], random_state=ctx["seed"])
    X_tr, X_te, padded = pad_series(ctx["X_tr"], ctx["X_te"], ROCKET_MIN_LENGTH)
    Z_tr = np.asarray(transform.fit_transform(X_tr), dtype=np.float64)
    Z_te = np.asarray(transform.transform(X_te), dtype=np.float64)
    # standardized with the class-balanced train statistics, as the STL features
    # (features._center_and_scale); constant columns are 0
    w = compute_sample_weight("balanced", ctx["y_tr"])
    mean = np.average(Z_tr, axis=0, weights=w)
    std = np.sqrt(np.average((Z_tr - mean) ** 2, axis=0, weights=w))
    const = std <= 1e-12 * np.maximum(np.abs(Z_tr).max(axis=0), 1.0)
    std[const] = 1.0
    Z_tr, Z_te = (Z_tr - mean) / std, (Z_te - mean) / std
    Z_tr[:, const], Z_te[:, const] = 0.0, 0.0
    feat_s = time.perf_counter() - t
    t = time.perf_counter()
    model = train_classifier(Z_tr, ctx["y_tr"], ctx["cfg"], seed=ctx["seed"])
    fit_s = time.perf_counter() - t
    t = time.perf_counter()
    p_te = lasso_predict(model, Z_te)
    return {**scores(ctx["y_tr"], lasso_predict(model, Z_tr), ctx["y_te"], p_te),
            "features_s": feat_s, "fit_s": fit_s, "predict_s": time.perf_counter() - t,
            "n_constant": int(const.sum()), **lasso_details(model), **padded}


RUNNERS = {
    "stlrocket": run_stlrocket,
    "stl_tree": run_stl_tree,
    "stl_ridge": run_stl_ridge,
    "minirocket": lambda ctx: run_aeon(ctx, make_minirocket, ROCKET_MIN_LENGTH),
    "multirockethydra": lambda ctx: run_aeon(ctx, make_multirockethydra, ROCKET_MIN_LENGTH),
    "mrseql": run_mrseql,
    "minirocket_lasso": run_minirocket_lasso,
}


def run(args, dataset: str, run_id: int) -> None:
    out = Path(args.out_dir) / dataset / f"run{run_id}"
    todo = [m for m in args.methods if not (out / f"{m}.json").exists()]
    if not todo:
        log(f"{dataset} run {run_id}: already complete, skipped")
        return
    out.mkdir(parents=True, exist_ok=True)
    seed = args.base_seed + run_id
    X_tr, y_tr, X_te, y_te = load_dataset(dataset)
    cfg = ExperimentConfig(
        dataset=dataset, n_formulas=args.n_formulas, depth_max=args.depth, until_weight=0.0,
        cv=args.cv, cut_point=args.cut_point, fit_intercept=False, n_run=1, base_seed=args.base_seed,
        output_dir=str(out), device=args.device,
        # fields of the old explanation pipeline, unused here
        explain=False, pool_size=0, precision_threshold=0.0, simplify_agreement=0.0, simplify_min_gain=0.0,
        simplify_decimals=0,
    )
    ctx = {"args": args, "out": out, "seed": seed, "cfg": cfg, "n_jobs": args.n_jobs,
           "X_tr": X_tr, "y_tr": y_tr, "X_te": X_te, "y_te": y_te,
           "stl": STLFeatures(args, out, X_tr, y_tr, X_te, cfg, seed, f"{dataset}_run{run_id}")}
    log(f"{dataset} run {run_id}: {len(X_tr)} train / {len(X_te)} test series, shape {X_tr.shape[1:]}, "
        f"methods {todo}")
    base = {"dataset": dataset, "run": run_id, "seed": seed, "n_train": len(X_tr), "n_test": len(X_te),
            "n_channels": int(X_tr.shape[1]), "length": int(X_tr.shape[2]), "n_classes": int(len(np.unique(y_tr)))}
    for method in todo:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                res = RUNNERS[method](ctx)
        except Exception as e:   # one failing method must not lose the others
            log(f"  {method}: FAILED, {type(e).__name__}: {e}")
            if args.raise_errors:
                raise
            continue
        if method in LASSO_SETTINGS:
            res["settings"] = {k: getattr(args, k) for k in LASSO_SETTINGS[method]}
        (out / f"{method}.json").write_text(json.dumps({**base, "method": method, **res}, indent=1))
        log(f"  {method}: balanced accuracy test {res['bacc_test']:.3f} (train {res['bacc_train']:.3f}), "
            f"fit {res['fit_s']:.0f} s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--runs", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--n_formulas", type=int, default=10000)
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--cut_point", type=float, default=1.0)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_jobs", type=int, default=available_cpus())
    p.add_argument("--feature_cache", default=None, help="directory where STL feature matrices are saved and reused")
    p.add_argument("--raise_errors", action="store_true", help="stop at the first failing method")
    p.add_argument("--out_dir", default="results/comparison_balanced")
    args = p.parse_args()
    for dataset in args.datasets:
        for run_id in args.runs:
            run(args, dataset, run_id)


if __name__ == "__main__":
    main()
