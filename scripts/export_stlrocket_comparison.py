#!/usr/bin/env python
"""
Write the STLRocket results of scripts/run_explanations.py into the folder of
scripts/run_comparison.py, as <dataset>/run<r>/stlrocket.json, so that plot_comparison.py reads
every method from one place. run_explanations.py trains exactly the model of the comparison's
stlrocket method (same code, settings and seed), so its accuracy and times are those of that
method; predict_s is not measured there and is left null.

    python scripts/export_stlrocket_comparison.py [--exp_dir results/explanations_balanced]
                                                  [--out_dir results/comparison_balanced]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default="results/explanations_balanced")
    ap.add_argument("--out_dir", default="results/comparison_balanced")
    args = ap.parse_args()
    n = 0
    for summary_path in sorted(Path(args.exp_dir).glob("*/run*/summary.json")):
        d = summary_path.parent
        s = json.loads(summary_path.read_text())
        model = json.loads((d / "model.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        out = Path(args.out_dir) / s["dataset"] / f"run{s['run']}"
        out.mkdir(parents=True, exist_ok=True)
        row = {"dataset": s["dataset"], "run": s["run"], "seed": s["seed"], "n_train": s["n_train"],
               "n_test": s["n_test"], "n_classes": s["n_classes"], "method": "stlrocket",
               "bacc_train": model["balanced_accuracy"]["train"], "bacc_test": model["balanced_accuracy"]["test"],
               "acc_train": model["accuracy"]["train"], "acc_test": model["accuracy"]["test"],
               "features_s": s["times_s"]["features_s"], "fit_s": s["times_s"]["model_s"], "predict_s": None,
               "lambda": model["lambda"], "n_selected": model["n_selected"],
               "settings": {"depth": s["depth"], "n_formulas": s["n_formulas"], "cv": cfg["experiment"]["cv"],
                            "cut_point": cfg["experiment"]["cut_point"]},
               "source": str(d)}
        (out / "stlrocket.json").write_text(json.dumps(row, indent=1))
        n += 1
    print(f"wrote {n} stlrocket.json files to {args.out_dir}")


if __name__ == "__main__":
    main()
