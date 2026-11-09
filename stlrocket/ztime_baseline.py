"""Z-Time (Lee, Lindgren and Papapetrou, "Z-Time: efficient and effective interpretable
multivariate time series classification", DMKD 2024) as a baseline of the comparison, on
the authors' code (https://github.com/zedshape/ztime), which is not a package and has no
license: clone it and point ZTIME_DIR to the checkout (the directory holding ztime/).

The authors' entry points (helper.simpleTrial, helper.randomizedSearch) return only the test
accuracy, so ZTimeClassifier re-assembles their pipeline (helper.trialSequence_multivariate)
from the same building blocks and exposes fit/predict:
  - each series z-normalized per channel (helper.znorm; a constant channel is only centred,
    where the authors' code would divide by 0);
  - the time axis cut into split_no parts; on each part, the event-interval database of
    helper.createMultivariateDatabase (SAX and PAA-SAX of the series, SAX of its first and
    second differences; window_size_slope = 1, glob = False, as in their trials) and the
    Z-Time arrangement counts (ZTime.ZTime(step).train/test);
  - StandardScaler(with_mean=False) and RidgeClassifierCV(alphas=logspace(-3, 3, 10)),
    their default classifier.
The feature matrix is built from the arrangement counts with the train arrangements as
columns, in a fixed order, for train and test alike (the authors refit a DictVectorizer on
the test counts, which gives the same columns as long as every series has some arrangement).

Hyperparameters are chosen as in helper.randomizedSearch, with its default grids: `cycle`
random configurations, each scored by accuracy on a stratified 67/33 holdout of the train
set, then the best one refitted on the whole train set. Here the draws and the holdout are
seeded (random_state); configurations whose parts are shorter than the PAA window are
skipped (the authors' code fails on them and draws again), as is any configuration that
fails, up to MAX_ATTEMPTS draws.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# helper.randomizedSearch defaults
STEPS = (10, 20, 30, 50, 100)
ALPHABET_SIZES = (3, 5, 7, 10)
ALPHABET_SIZES_SLOPE = (3, 5, 7, 10)
WINDOW_SIZES = (3, 5, 7, 10)
SPLIT_NOS = (1, 2, 4, 8)
CYCLE = 5
MAX_ATTEMPTS = 50


def _import_ztime():
    """The authors' helper and ZTime modules, from ZTIME_DIR."""
    root = os.environ.get("ZTIME_DIR")
    if not root:
        raise RuntimeError("set ZTIME_DIR to a checkout of https://github.com/zedshape/ztime")
    root = os.path.expanduser(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from ztime import ZTime, helper
    return helper, ZTime


def znorm(X: np.ndarray) -> np.ndarray:
    """Per series and channel z-normalization along time, exactly as helper.znorm (same
    operations, in the input's dtype: the SAX bins are sensitive to rounding), except that
    a constant channel is only centred."""
    X = np.asarray(X)
    m = X.mean(axis=-1)[..., np.newaxis]
    s = X.std(axis=-1)[..., np.newaxis]
    s = np.where(s == 0, np.ones_like(s), s)
    return (X - m) / s


def _counts(FL: dict, keys: list, n: int) -> csr_matrix:
    """(n, len(keys)) matrix of arrangement counts FL[key][series]."""
    rows, cols, vals = [], [], []
    for j, key in enumerate(keys):
        for sid, v in FL.get(key, {}).items():
            rows.append(sid)
            cols.append(j)
            vals.append(v)
    return csr_matrix((np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(n, len(keys)))


class ZTimeClassifier:
    """One Z-Time configuration, or (params=None) the randomized search of the authors."""

    def __init__(self, params: dict | None = None, cycle: int = CYCLE, random_state: int | None = None):
        self.params = params
        self.cycle = cycle
        self.random_state = random_state

    # -- one configuration -------------------------------------------------------------
    def _database(self, part: np.ndarray):
        p = self.params_
        return self._helper.createMultivariateDatabase(
            part, window_size=p["window_size"], window_size_slope=1, alphabet_size=p["alphabet_size"],
            alphabet_size_slope=p["alphabet_size_slope"], glob=False)

    def _fit_config(self, X: np.ndarray, y: np.ndarray):
        self._helper, ZTime = _import_ztime()
        p = self.params_
        self.algos_, self.keys_, mats = [], [], []
        for part in np.array_split(X, p["split_no"], axis=-1):
            algo = ZTime.ZTime(step=p["step"])
            FL = algo.train(self._database(part), y)
            algo.database = None
            keys = list(FL.keys())
            self.algos_.append(algo)
            self.keys_.append(keys)
            mats.append(_counts(FL, keys, len(X)))
        F = hstack(mats).tocsr()
        self.n_features_ = int(F.shape[1])
        self.scaler_ = StandardScaler(with_mean=False)
        self.clf_ = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.clf_.fit(self.scaler_.fit_transform(F), y)
        return self

    def _features(self, X: np.ndarray):
        mats = []
        for part, algo, keys in zip(np.array_split(X, self.params_["split_no"], axis=-1), self.algos_, self.keys_):
            mats.append(_counts(algo.test(self._database(part)), keys, len(X)))
        return self.scaler_.transform(hstack(mats).tocsr())

    # -- sklearn-like interface ----------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = znorm(X), np.asarray(y)
        t = time.perf_counter()
        if self.params is None:
            self.params_, self.search_ = self._search(X, y)
        else:
            self.params_, self.search_ = dict(self.params), []
        self.search_s_ = time.perf_counter() - t
        return self._fit_config(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf_.predict(self._features(znorm(X)))

    def _search(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        try:
            X_a, X_b, y_a, y_b = train_test_split(X, y, test_size=0.33, stratify=y, random_state=self.random_state)
        except ValueError:   # a class with a single series: as the authors, without stratification
            X_a, X_b, y_a, y_b = train_test_split(X, y, test_size=0.33, random_state=self.random_state)
        T = X.shape[-1]
        tried, best = [], None
        for _ in range(MAX_ATTEMPTS):
            if len(tried) == self.cycle:
                break
            p = {"step": int(rng.choice(STEPS)), "alphabet_size": int(rng.choice(ALPHABET_SIZES)),
                 "alphabet_size_slope": int(rng.choice(ALPHABET_SIZES_SLOPE)),
                 "window_size": int(rng.choice(WINDOW_SIZES)), "split_no": int(rng.choice(SPLIT_NOS))}
            if T // p["split_no"] < p["window_size"]:   # shortest part shorter than the PAA window
                continue
            try:
                model = ZTimeClassifier(p)
                model.params_ = p
                model._fit_config(X_a, y_a)
                acc = float(np.mean(model.clf_.predict(model._features(X_b)) == y_b))
            except Exception:   # the authors' search also skips failing configurations
                continue
            tried.append({**p, "holdout_acc": acc})
            if best is None or acc > best[1]:
                best = (p, acc)
        if best is None:
            raise RuntimeError(f"no Z-Time configuration could be fitted in {MAX_ATTEMPTS} draws")
        return best[0], tried
