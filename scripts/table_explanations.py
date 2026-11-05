"""Per-dataset table of the explanations of run_explanations.py (paper, tab:expl_datasets).

One row per dataset, mean over its runs with a non-empty model (a run counts if its
summary.json exists), and a last row with the mean over datasets:
  Nonzero        nonzero entries of the weight matrix W (classes x selected formulae)
  instance rules of the test series, scored as their simplified STL formula on test,
  w.r.t. the classifier's predictions:
    Prec.        pooled precision (summary.json, pooled_prec_test)
    Cov.         mean coverage (mean_coverage_test)
    beta<0       fraction of the rules with negative share beta (series with a tied
                 top logit, whose beta is undefined, left out)
    beta         median share beta over the same rules (the mean is dominated by the few
                 series whose margin is near 0, e.g. LSST -6.6, FaceDetection 8.0)
    Nodes        mean node count of the simplified formula (mean_size)
  global explanations (simplified), on test, w.r.t. the classifier's predictions, averaged
  over the classes predicted on train:
    Prec.        precision of the class's disjunction (classes satisfied by no test series
                 left out, as in summary.json)
    Cov.         coverage
    F1           2 P R / (P + R), 0 when no test series of the class satisfies it
    Nodes        node count of the disjunction: sum of its rules' sizes + (rules - 1) ORs

    python scripts/table_explanations.py [--res_dir results/explanations_v3]
                                          [--out paper/tables/expl_datasets_body.tex]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_row(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    W = np.array(json.loads((run / "model.json").read_text())["weights"])
    r = s["rules_test"]
    rules = pd.read_csv(run / "rules_test.csv", usecols=["n_literals", "share"])
    beta = rules.loc[rules.n_literals > 0, "share"].dropna()
    g = pd.read_csv(run / "global_metrics.csv")
    g = g[(g.version == "simplified") & (g.evaluated_on == "test")]
    p, c = g.precision.fillna(0.0), g.coverage
    f1 = np.where(g.tp > 0, 2 * p * c / (p + c).where(p + c > 0, 1.0), 0.0)
    return {"dataset": s["dataset"], "nonzero": int((W != 0).sum()),
            "prec": r["pooled_prec_test"], "cov": r["mean_coverage_test"],
            "beta_neg": float((beta < 0).mean()), "beta": float(beta.median()), "nodes": r["mean_size"],
            "g_prec": float(g.precision.mean()), "g_cov": float(c.mean()), "g_f1": float(f1.mean()),
            "g_nodes": float((g["size"] + g.n_rules - 1).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res_dir", default="results/explanations_v3")
    p.add_argument("--out", default="paper/tables/expl_datasets_body.tex")
    args = p.parse_args()

    runs = sorted(f.parent for f in Path(args.res_dir).glob("*/run*/summary.json"))
    df = pd.DataFrame([run_row(r) for r in runs])
    table = df.groupby("dataset").mean()
    table = table.loc[sorted(table.index, key=str.lower)]
    n_runs = df.groupby("dataset").size()
    pd.set_option("display.width", 250)
    print(table.join(n_runs.rename("runs")).to_string(float_format=lambda x: f"{x:.3f}"))

    fmt = {"nonzero": "{:.1f}", "prec": "{:.3f}", "cov": "{:.3f}", "beta_neg": "{:.3f}", "beta": "{:.2f}",
           "nodes": "{:.1f}", "g_prec": "{:.3f}", "g_cov": "{:.3f}", "g_f1": "{:.3f}", "g_nodes": "{:.1f}"}

    def line(name, row):
        return " & ".join([name, *(fmt[c].format(row[c]) for c in fmt)]) + r" \\"

    lines = [r"\begin{tabular}{lrccccccccc}", r"\toprule",
             r" & & \multicolumn{5}{c}{Instance} & \multicolumn{4}{c}{Global} \\",
             r"\cmidrule(lr){3-7} \cmidrule(lr){8-11}",
             r"Dataset & Nonzero & Prec. & Cov. & $\beta < 0$ & $\beta$ & Nodes & Prec. & Cov. & $F_1$ & Nodes \\",
             r"\midrule"]
    lines += [line(d, row) for d, row in table.iterrows()]
    lines += [r"\midrule", line(f"Mean ({len(table)} datasets)", table.mean()), r"\bottomrule", r"\end{tabular}"]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\n{len(df)} runs on {len(table)} datasets -> {args.out}")


if __name__ == "__main__":
    main()
