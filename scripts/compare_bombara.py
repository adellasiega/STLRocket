#!/usr/bin/env python
"""STLRocket vs the decision tree of STL primitives of Bombara and Belta (method "bombara" of
run_comparison.py): balanced accuracy on test and time, per dataset and overall.

    python scripts/compare_bombara.py [--res_dir results/comparison_bombara] [--csv out.csv]

Reads RES_DIR/<dataset>/run<r>/{stlrocket,bombara}.json and keeps the (dataset, run) pairs
where both methods completed. Time is end to end, from the series to the test predictions:
stlrocket features (both splits) + fit with model selection + predict; bombara fit with
model selection (CV over the depth and refit) + predict. Per dataset: mean (std) over runs.
Overall, over the dataset means: mean, wins/ties/losses of stlrocket, two-sided Wilcoxon
signed-rank test on balanced accuracy, and median time ratio bombara / stlrocket.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METHODS = ("stlrocket", "bombara")


def total_time(r: dict) -> float:
    return sum(r.get(k) or 0.0 for k in ("features_s", "fit_s", "predict_s"))


def load(res_dir: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(res_dir.glob("*/run*")):
        res = {m: json.loads((run / f"{m}.json").read_text()) for m in METHODS if (run / f"{m}.json").exists()}
        if len(res) < len(METHODS):
            continue
        s, b = res["stlrocket"], res["bombara"]
        rows.append({"dataset": s["dataset"], "run": s["run"],
                     "bacc_stlrocket": s["bacc_test"], "bacc_bombara": b["bacc_test"],
                     "time_stlrocket": total_time(s), "time_bombara": total_time(b),
                     "n_selected": s.get("n_selected"), "bb_depth": b["max_depth"], "bb_leaves": b["n_leaves"],
                     "bb_size": b["formula_size"]})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res_dir", default="results/comparison_bombara")
    p.add_argument("--csv", default=None, help="also write the per-dataset table here")
    args = p.parse_args()

    df = load(Path(args.res_dir))
    if df.empty:
        raise SystemExit(f"no run with both {METHODS} in {args.res_dir}")
    g = df.groupby("dataset")
    table = g.mean(numeric_only=True).drop(columns="run")
    for m in METHODS:
        table[f"bacc_{m}_std"] = g[f"bacc_{m}"].std()
    table["runs"] = g.size()
    table["time_ratio"] = table.time_bombara / table.time_stlrocket
    table = table.loc[sorted(table.index, key=str.lower)]

    show = pd.DataFrame({
        "runs": table.runs,
        "STLRocket": [f"{m:.3f} ({s:.3f})" for m, s in zip(table.bacc_stlrocket, table.bacc_stlrocket_std.fillna(0))],
        "Bombara": [f"{m:.3f} ({s:.3f})" for m, s in zip(table.bacc_bombara, table.bacc_bombara_std.fillna(0))],
        "diff": (table.bacc_stlrocket - table.bacc_bombara).map("{:+.3f}".format),
        "time STLR (s)": table.time_stlrocket.map("{:.1f}".format),
        "time BB (s)": table.time_bombara.map("{:.1f}".format),
        "ratio BB/STLR": table.time_ratio.map("{:.2f}".format),
        "BB depth": table.bb_depth.map("{:.1f}".format),
        "BB leaves": table.bb_leaves.map("{:.1f}".format),
        "BB nodes": table.bb_size.map("{:.1f}".format),
    }, index=table.index)
    pd.set_option("display.width", 250)
    print("Balanced accuracy on test, mean (std) over runs; time per run, end to end\n")
    print(show.to_string())

    d = table.bacc_stlrocket - table.bacc_bombara
    eps = 1e-9
    wins, ties, losses = int((d > eps).sum()), int((d.abs() <= eps).sum()), int((d < -eps).sum())
    print(f"\n{len(df)} runs on {len(table)} datasets (mean over the dataset means)")
    print(f"balanced accuracy  STLRocket {table.bacc_stlrocket.mean():.3f}   Bombara {table.bacc_bombara.mean():.3f}   "
          f"diff {d.mean():+.3f}")
    print(f"STLRocket wins / ties / losses: {wins} / {ties} / {losses}")
    if (d.abs() > eps).sum() >= 1:
        print(f"Wilcoxon signed-rank (two-sided): p = {wilcoxon(d).pvalue:.3g}"
              + ("  (fewer than 6 datasets: p cannot go below 0.05)" if len(d) < 6 else ""))
    print(f"time per run, median over datasets: STLRocket {table.time_stlrocket.median():.1f} s, "
          f"Bombara {table.time_bombara.median():.1f} s; median ratio Bombara / STLRocket "
          f"{table.time_ratio.median():.2f} (range {table.time_ratio.min():.2f}-{table.time_ratio.max():.2f})")
    if args.csv:
        table.to_csv(args.csv)
        print(f"per-dataset table -> {args.csv}")


if __name__ == "__main__":
    main()
