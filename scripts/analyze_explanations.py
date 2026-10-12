#!/usr/bin/env python
"""
Aggregates the runs of scripts/run_explanations.py: one row per dataset, mean (and std)
over its complete runs (those with summary.json), plus the run counts.

    python scripts/analyze_explanations.py [--res_dir results/explanations] [--out FILE.csv]

Rule metrics are those of the test-series instance rules (rules_test), evaluated on test;
global metrics are those of the simplified global explanations on test. All are w.r.t.
the model's predictions (see run_explanations.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def flatten(s: dict) -> dict:
    r, g, t = s["rules_test"], s["global"], s["times_s"]
    return {
        "dataset": s["dataset"], "run": s["run"], "n_train": s["n_train"], "n_test": s["n_test"],
        "n_classes": s["n_classes"], "n_selected": s["n_selected"],
        "bacc_test": s["balanced_accuracy"]["test"],
        "found": r["found"] / r["of"],
        "prec_train": r["pooled_prec_train"], "prec_test": r["pooled_prec_test"],
        "cov_test": r["mean_coverage_test"], "lcb": r["mean_lcb_train"],
        "length": r["mean_length"], "size": r["mean_size"],
        "g_prec_train": g["train"]["macro_precision"], "g_prec_test": g["test"]["macro_precision"],
        "g_cov_test": g["test"]["macro_coverage"], "g_rules": g["mean_rules"],
        "hours": sum(t.values()) / 3600,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res_dir", default="results/explanations")
    p.add_argument("--out", default=None, help="write the per-dataset table to this CSV")
    args = p.parse_args()

    runs, status = [], []
    for d in sorted(Path(args.res_dir).iterdir(), key=lambda d: d.name.lower()):
        for r in sorted(d.glob("run*")):
            if (r / "summary.json").exists():
                runs.append(flatten(json.loads((r / "summary.json").read_text())))
                state = "complete"
            elif (r / "model.json").exists() and json.loads((r / "model.json").read_text())["n_selected"] == 0:
                state = "empty_model"
            else:
                state = "unfinished"
            status.append({"dataset": d.name, "state": state})
    df = pd.DataFrame(runs)
    counts = pd.DataFrame(status).pivot_table(index="dataset", columns="state", aggfunc="size", fill_value=0)

    cols = [c for c in df.columns if c not in ("dataset", "run", "n_train", "n_test", "n_classes")]
    g = df.groupby("dataset")
    table = g[["n_train", "n_test", "n_classes"]].first().join(g[cols].mean())
    table = table.join(g[["bacc_test", "prec_test", "g_prec_test"]].std().add_suffix("_std"))
    table = counts.join(table).sort_index(key=lambda i: i.str.lower())
    if args.out:
        table.to_csv(args.out)

    pd.set_option("display.width", 250, "display.max_columns", 50)
    show = ["n_train", "n_classes", "n_selected", "bacc_test", "found", "prec_train", "prec_test", "cov_test",
            "length", "size", "g_prec_test", "g_cov_test", "g_rules", "hours"]
    print(counts.join(table[show]).to_string(float_format=lambda x: f"{x:.3f}"))
    done = df[df.n_selected > 0]
    print(f"\n{len(df)} complete runs on {df.dataset.nunique()} datasets; "
          f"means over runs (non-empty models):")
    print(done[show[2:]].mean().to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
