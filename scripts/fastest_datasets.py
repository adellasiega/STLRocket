#!/usr/bin/env python
"""
Print the N datasets with the lowest mean time (feature build + fit) per run at
M=10000 in a previous ablation, summed over depths, space separated.

    python scripts/fastest_datasets.py results/ablation/full/*.csv --n 10
"""
import argparse

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("csvs", nargs="+")
p.add_argument("--n", type=int, default=10)
p.add_argument("--n_formulas", type=int, default=10000)
args = p.parse_args()

df = pd.concat(pd.read_csv(f) for f in args.csvs)
df = df[(df.status == "ok") & (df.n_formulas == args.n_formulas)]
t = (df.time_feats_s + df.time_fit_s).groupby([df.dataset, df.depth_max]).mean().groupby("dataset").sum()
print(" ".join(t.nsmallest(args.n).index))
