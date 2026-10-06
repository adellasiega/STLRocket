"""Classifier balanced accuracy vs. correlation-filter threshold, over several seeds.

For each seed the formula stream of F0(seed) is sampled once into a pool, extended
on demand; each threshold keeps the first n_formulas formulae that pass the greedy
filter of updated_notebook.ipynb (identical to its build_decorrelated_bank). So, for a
seed, the thresholds differ only in the filter, not in the random formulae.

Usage: python corr_threshold_comparison.py [--seeds 0 1 ...] [--thresholds 0.95 0.99 1.0]
Rows are appended to the output CSV; (seed, threshold) pairs already there are skipped.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from stlrocket.data import load_dataset
from stlrocket.config import ExperimentConfig
from stlrocket.formula_sampler import F0
from stlrocket.features import extract_features, set_device
from stlrocket.classifier import train_classifier, evaluate_classifier


def correlation_filter(X, threshold, n_keep=0, block=1000):
    # Same as in updated_notebook.ipynb
    std = X.std(axis=0)
    Z = np.zeros_like(X, dtype=np.float64)
    nz = std > 1e-12
    Z[:, nz] = (X[:, nz] - X[:, nz].mean(axis=0)) / (std[nz] * np.sqrt(len(X)))  # Z.T @ Z = r
    keep = list(range(n_keep))
    for start in range(n_keep, X.shape[1], block):
        cand = np.flatnonzero(nz[start:min(start + block, X.shape[1])]) + start
        if keep:
            cand = cand[(np.abs(Z[:, keep].T @ Z[:, cand]) < threshold).all(axis=0)]
        R = np.abs(Z[:, cand].T @ Z[:, cand])
        alive = np.ones(len(cand), dtype=bool)
        for i in range(len(cand)):
            if alive[i]:
                alive[i + 1:] &= R[i, i + 1:] < threshold
        keep.extend(cand[alive].tolist())
    return np.array(keep, dtype=int)


class Pool:
    # Formulae of F0(seed) in sampling order, with raw train/test robustness features
    def __init__(self, X_tr, X_te, config, seed):
        np.random.seed(seed)
        self.X_tr, self.X_te = X_tr, X_te
        self.generator = F0(n_vars=X_tr.shape[1], v_min=np.nanmin(X_tr, axis=(0, 2)),
                            v_max=np.nanmax(X_tr, axis=(0, 2)), t_max=X_tr.shape[2] - 1,
                            depth_max=config.depth_max, seed=seed, until_weight=config.until_weight)
        self.F_tr = np.empty((len(X_tr), 0), dtype=np.float32)   # extract_features' dtype
        self.F_te = np.empty((len(X_te), 0), dtype=np.float32)

    def extend(self, n):
        new = self.generator.sample(n)
        self.F_tr = np.hstack([self.F_tr, extract_features(self.X_tr, new)])
        self.F_te = np.hstack([self.F_te, extract_features(self.X_te, new)])


def decorrelated_bank(pool, threshold, M, max_rounds=20):
    if pool.F_tr.shape[1] < M:
        pool.extend(M - pool.F_tr.shape[1])
    keep = correlation_filter(pool.F_tr, threshold)
    for _ in range(max_rounds):
        if len(keep) >= M:
            break
        n_old = pool.F_tr.shape[1]
        rate = len(keep) / n_old
        pool.extend(int(min(M, max(1000, 1.2 * (M - len(keep)) / rate))))
        new = correlation_filter(pool.F_tr[:, np.r_[keep, n_old:pool.F_tr.shape[1]]], threshold, n_keep=len(keep))
        keep = np.r_[keep, new[len(keep):] - len(keep) + n_old]
    n_sampled = int(keep[M - 1]) + 1 if len(keep) >= M else pool.F_tr.shape[1]
    keep = keep[:M]
    F_tr, F_te = pool.F_tr[:, keep], pool.F_te[:, keep]
    mu, sigma = F_tr.mean(axis=0), F_tr.std(axis=0)
    return (F_tr - mu) / (sigma + 1e-8), (F_te - mu) / (sigma + 1e-8), n_sampled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HandMovementDirection")
    p.add_argument("--n_formulas", type=int, default=10000)
    p.add_argument("--depth_max", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.95, 0.99, 1.0])
    p.add_argument("--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "results", "corr_threshold_comparison.csv"))
    args = p.parse_args()

    config = ExperimentConfig(
        dataset=args.dataset, n_formulas=args.n_formulas, depth_max=args.depth_max, until_weight=0.0,
        cv=3, cut_point=1.0, fit_intercept=False, explain=False, pool_size=10, precision_threshold=0.9,
        simplify_agreement=0.95, simplify_min_gain=0.0, simplify_decimals=1, n_run=len(args.seeds),
        base_seed=0, output_dir="results", device="cpu")
    set_device(config.device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    done = set()
    if os.path.exists(args.output):
        prev = pd.read_csv(args.output)
        done = set(zip(prev["seed"], prev["threshold"]))

    X_tr_raw, y_tr, X_te_raw, y_te = load_dataset(config.dataset)
    for seed in args.seeds:
        todo = [t for t in args.thresholds if (seed, t) not in done]
        if not todo:
            continue
        pool = Pool(X_tr_raw, X_te_raw, config, seed)
        # strictest threshold first: it needs the largest pool, the others reuse it
        for t in sorted(todo):
            t0 = time.time()
            X_tr, X_te, n_sampled = decorrelated_bank(pool, t, config.n_formulas)
            model = train_classifier(X_tr, y_tr, config)
            row = {"dataset": config.dataset, "n_formulas": config.n_formulas, "seed": seed, "threshold": t,
                   "n_sampled": n_sampled,
                   "train_bacc": evaluate_classifier(model, X_tr, y_tr)["balanced_accuracy"],
                   "test_bacc": evaluate_classifier(model, X_te, y_te)["balanced_accuracy"],
                   "n_selected": int((model.coef_ != 0).any(axis=0).sum()),
                   "seconds": round(time.time() - t0, 1)}
            print(row, flush=True)
            pd.DataFrame([row]).to_csv(args.output, mode="a", header=not os.path.exists(args.output), index=False)

    df = pd.read_csv(args.output)
    df = df[(df["dataset"] == config.dataset) & (df["n_formulas"] == config.n_formulas)]
    print(df.groupby("threshold")[["train_bacc", "test_bacc", "n_selected", "n_sampled"]]
            .agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
