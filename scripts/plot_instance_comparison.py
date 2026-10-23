#!/usr/bin/env python
"""
Plots the comparison of scripts/run_instance_comparison.py (STLRocket instance and global
explanations against other explainers of the same classifier and features).

    python scripts/plot_instance_comparison.py [--res_dir results/instance_comparison]
                                               [--out_dir results/figures] [--exclude DATASET ...]

Only complete runs (with a .done.json) with a non-empty model enter.

Instance metrics are computed from the explained test series, on the test set and w.r.t. the
model's predictions, per (dataset, run, method) and then averaged over the runs of a dataset.
Precision is pooled over the explanations found (TP / support, as in the explanation ablation);
coverage and length are means over them; found is the share of series explained. Comparisons:
  all      ours, AXp and inflated AXp on every explained test series
  lore     every method on the test series LORE explained (it may be capped or time out)
  anchors  every method on the test series Anchors explained within its timeout
LORE and Anchors enter only on datasets where they explained at least --min_share of the series
they attempted. Global explanations: F1 on test (mean over classes), number of rules and of
literals of the greedy cover of each instance method (subset "all") and of the surrogates.
Writes:
  instance_precision_coverage  (PDF, PNG) test precision against test coverage of the instance
                               explanations, one point per dataset and method, each method on
                               the series it explained
  instance_table.tex           per dataset and method: precision, coverage, length, time per
                               series, each method on the series it explained
  instance_summary.tex         the paper's table: per comparison, one row per method: the number of
                               datasets with a result and, per metric, the mean over
                               datasets (the two-sided Wilcoxon signed-rank p-value
                               of STLRocket against each method is only printed)
  global_summary.tex           the same for the global explanations, built from every explained
                               train series and from those LORE and Anchors attempted
and prints how often the precision claims of Anchors and AXp hold on test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_comparison import INK_2, style_axis  # also sets the paper's rcParams

# method -> (label, color, marker)
METHOD_STYLE = {"stlrocket": ("STLRocket", "#2a78d6", "o"), "axp": ("AXp", "#3a9e5c", "^"),
                "iaxp": ("Inflated AXp", "#8bc34a", "v"), "lore": ("LORE", "#9c27b0", "D"),
                "anchors": ("Anchors", "#eb6834", "s"), "tree": ("Surrogate tree", "#795548", "P"),
                "ripper": ("RIPPER", "#607d8b", "X")}
# short labels for the columns of the per-dataset table
SHORT = {"stlrocket": "Ours", "axp": "AXp", "iaxp": "IAXp", "lore": "LORE", "anchors": "Anch."}
INSTANCE = ("stlrocket", "axp", "iaxp", "lore", "anchors")
SAMPLED = ("lore", "anchors")
METRICS = [("precision", "Precision", "{:.3f}"), ("coverage", "Coverage", "{:.3f}"),
           ("length", "Length", "{:.2f}"), ("found", "Found", "{:.2f}"), ("time_s", "Time (s)", "{:.2g}")]
GLOBAL_METRICS = [("f1", "$F_1$", "{:.3f}"), ("n_rules", "Rules", "{:.1f}"), ("length", "Literals", "{:.1f}")]
GLOBAL_LABEL = {"all": "All datasets", "lore": "Datasets on which LORE ran",
                "anchors": "Datasets on which Anchors ran"}
SUBSET_LABEL = {"all": "All test series", "lore": "Test series explained by LORE",
                "anchors": "Test series explained by Anchors"}


def load(res_dir: Path, exclude: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Results of the complete runs (with a .done.json) with a non-empty model, of the datasets
    not in exclude."""
    done = [json.loads(f.read_text()) for f in sorted(res_dir.glob("*.done.json"))]
    done = [d for d in done if d["dataset"] not in exclude]
    empty = [(d["dataset"], d["run"]) for d in done if d["status"] != "ok"]
    if empty:
        print(f"runs with an empty model, left out: {empty}")
    ok = {(d["dataset"], d["run"]) for d in done if d["status"] == "ok"}

    def name(f: Path) -> tuple[str, int]:
        return f.name.split("_run")[0], int(f.name.split("_run")[1].split(".")[0])

    def read(pattern: str) -> pd.DataFrame:
        files = sorted(res_dir.glob(pattern))
        unfinished = sorted({name(f) for f in files if name(f) not in ok and name(f)[0] not in exclude
                             and name(f) not in empty})
        if unfinished and pattern.startswith("*.instances"):
            print(f"unfinished runs (no .done.json), left out: {unfinished}")
        return pd.concat([pd.read_csv(f).assign(dataset=name(f)[0], run=name(f)[1])
                          for f in files if name(f) in ok], ignore_index=True)

    return read("*.instances.csv.gz"), read("*.global.csv")


def run_metrics(d: pd.DataFrame) -> pd.Series:
    f = d[d.found]
    return pd.Series({"precision": f.tp_test.sum() / f.support_test.sum() if f.support_test.sum() else np.nan,
                      "coverage": f.coverage_test.mean(), "length": f.length.mean(), "found": d.found.mean(),
                      "time_s": d.time_s.mean(), "n": len(d)})


def done_share(inst: pd.DataFrame) -> pd.DataFrame:
    """(dataset) x sampled method: share of the test series it attempted that it explained."""
    a = inst[(inst.split == "test") & inst.method.isin(SAMPLED) & (inst.status != "skipped")]
    return a.groupby(["dataset", "method"]).status.apply(lambda s: (s == "ok").mean()).unstack("method")


def per_dataset(inst: pd.DataFrame, subset: str, share: pd.DataFrame, min_share: float) -> pd.DataFrame:
    """(dataset, method) x metrics, mean over runs, on the test series of the subset: "all" every
    series each method was run on (not skipped), else those the subset's method explained."""
    t = inst[(inst.split == "test") & (inst.status != "skipped")]
    if subset != "all":
        ok = t[(t.method == subset) & (t.status == "ok")][["dataset", "run", "i"]]
        t = t.merge(ok, on=["dataset", "run", "i"])
    m = t.groupby(["dataset", "run", "method"]).apply(run_metrics, include_groups=False)
    m = m.groupby(["dataset", "method"]).mean()
    d, meth = m.index.get_level_values("dataset"), m.index.get_level_values("method")
    if subset == "all":
        return m[~meth.isin(SAMPLED)]
    # a sampled method enters only where it explained enough of what it attempted
    good = lambda name: share[name][share[name] >= min_share].index if name in share else []
    keep = d.isin(good(subset)) & (~meth.isin(SAMPLED) | (meth == subset) | d.isin(good("lore" if subset == "anchors" else "anchors")))
    return m[keep]


def global_per_dataset(glob: pd.DataFrame, subset: str) -> pd.DataFrame:
    """(dataset, method) x global metrics, mean over runs and classes. subset "all": the instance
    methods from every explained train series, without the sampled ones; otherwise every instance
    method from the train series the subset's method attempted, on the datasets where it ran, and
    the surrogates (always trained on every train series) on the same datasets."""
    if subset == "all":
        g = glob[(glob.subset == "all") & ~glob.method.isin(SAMPLED)]
    else:
        ran = glob[glob.method == subset].dataset.unique()
        g = glob[glob.dataset.isin(ran) & ((glob.subset == subset) | glob.method.isin(["tree", "ripper"]))]
    g = g.groupby(["dataset", "run", "method"])[["f1", "n_rules", "length"]].mean()
    return g.groupby(["dataset", "method"]).mean()


def plot_precision_coverage(p: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    p = p.unstack("method")
    methods = [m for m in METHOD_STYLE if ("coverage", m) in p.columns]
    # each dataset: a segment from STLRocket to each other method
    for d in p.index:
        for m in methods:
            if m != "stlrocket":
                ax.plot([p.loc[d, ("coverage", "stlrocket")], p.loc[d, ("coverage", m)]],
                        [p.loc[d, ("precision", "stlrocket")], p.loc[d, ("precision", m)]],
                        color=INK_2, linewidth=0.3, zorder=1)
    for m in methods:
        label, color, marker = METHOD_STYLE[m]
        ax.scatter(p[("coverage", m)], p[("precision", m)], s=24, color=color, marker=marker,
                   edgecolors="white", linewidths=0.5, zorder=3, label=label)
    ax.set_xlabel("Test coverage of the instance explanations")
    ax.set_ylabel("Test precision of the instance explanations")
    ax.legend(loc="lower right", fontsize=7)
    style_axis(ax)
    return fig


def cell(v, fmt):
    return fmt.format(v) if np.isfinite(v) else "--"


def write_table(p: pd.DataFrame, share: pd.DataFrame, path: Path):
    p = p.unstack("method")
    methods = [m for m in INSTANCE if ("precision", m) in p.columns]
    cols = [c for c in METRICS if c[0] != "found"]
    nm = len(methods)
    head = " & ".join(f"\\multicolumn{{{nm}}}{{c}}{{{label}}}" for _, label, _ in cols)
    sub = " & ".join(" & ".join(SHORT[m] for m in methods) for _ in cols)
    lines = [r"\begin{tabular}{l" + "c" * nm * len(cols) + "}", r"\toprule", f" & {head} \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + nm * i}-{1 + nm * (i + 1)}}}" for i in range(len(cols))),
             f"Dataset & {sub} \\\\", r"\midrule"]
    for d in p.index:
        lines.append(d + " & " + " & ".join(cell(p.loc[d, (c, m)], fmt) for c, _, fmt in cols for m in methods) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def pvalue(a: pd.Series, b: pd.Series) -> float:
    diff = (a - b).dropna()
    return wilcoxon(diff).pvalue if len(diff) > 1 and diff.abs().sum() > 0 else np.nan


def write_summary(blocks: dict, metrics: list, methods: list, path: Path):
    """blocks: label -> per_dataset frame. One block per comparison, one row per method with a
    value: the number of datasets on which it has a result, then per metric the mean over
    datasets. The two-sided Wilcoxon signed-rank p-value against STLRocket, over the datasets
    where both have a value, is only printed."""
    ncol = 2 + len(metrics)
    lines = [r"\begin{tabular}{lr" + "c" * len(metrics) + "}", r"\toprule",
             "Method & Datasets & " + " & ".join(name for _, name, _ in metrics) + r" \\"]
    for label, p in blocks.items():
        p = p.unstack("method")
        if len(blocks) > 1:
            lines += [r"\midrule", f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{label}}}}} \\\\"]
        else:
            lines.append(r"\midrule")
        print(f"--- {label}")
        for m in methods:
            cols = [(c, m) for c, _, _ in metrics if (c, m) in p.columns]
            n = int(p[cols].notna().any(axis=1).sum()) if cols else 0
            if n == 0:
                continue
            cells, out = [], []
            for c, _, fmt in metrics:
                v = p[(c, m)].mean() if (c, m) in p.columns else np.nan
                pv = np.nan if m == "stlrocket" or (c, m) not in p.columns else pvalue(p[(c, "stlrocket")], p[(c, m)])
                cells.append(cell(v, fmt))
                out.append(f"{cell(v, fmt):>8s} (p {pv:.2g})" if m != "stlrocket" else f"{cell(v, fmt):>8s}")
            lines.append(f"{METHOD_STYLE[m][0]} & {n} & " + " & ".join(cells) + r" \\")
            print(f"{METHOD_STYLE[m][0]:15s} {n:3d} datasets " + " ".join(out))
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def claims(inst: pd.DataFrame, threshold: float):
    """How often the precision a method claims holds on test."""
    d = inst[(inst.split == "test") & inst.found]
    a = d[d.method == "anchors"]
    if len(a):
        c = a[a.est_precision >= threshold]
        print(f"Anchors: estimated precision >= {threshold} for {len(c) / len(a):.1%} of the explanations; "
              f"test precision >= {threshold} for {(c.precision_test >= threshold).mean():.1%} of those; "
              f"estimated minus test precision {(a.est_precision - a.precision_test).mean():+.3f}")
    for m in ("axp", "iaxp"):
        x = d[(d.method == m) & (d.support_test > 0)]
        if len(x):
            print(f"{m}: test precision 1 for {(x.precision_test == 1).mean():.1%} of the explanations satisfied by a "
                  f"test series ({len(x) / max((d.method == m).sum(), 1):.1%} of them)")
    lo = d[d.method == "lore"]
    if len(lo):
        print(f"LORE: mean fidelity on its neighbourhood {lo.fidelity.mean():.3f}, mean test precision "
              f"{lo.precision_test.mean():.3f}; rule class = model class for {(lo.rule_class == lo.k).mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default="results/instance_comparison")
    ap.add_argument("--out_dir", default="results/figures")
    ap.add_argument("--threshold", type=float, default=0.95, help="Anchors precision threshold of the runs")
    ap.add_argument("--min_share", type=float, default=0.5,
                    help="LORE, Anchors enter a comparison on datasets where they explained this share of "
                         "the series they attempted")
    ap.add_argument("--exclude", nargs="*", default=[], help="datasets left out")
    args = ap.parse_args()
    res_dir, out_dir = Path(args.res_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst, glob = load(res_dir, args.exclude)
    print(inst.groupby(["dataset", "method"]).run.nunique().unstack().to_string())
    share = done_share(inst)
    print("share of attempted test series explained:\n" + share.round(2).to_string())
    methods = [m for m in INSTANCE if (inst.method == m).any()]
    blocks = {SUBSET_LABEL[s]: per_dataset(inst, s, share, args.min_share)
              for s in ("all",) + tuple(m for m in SAMPLED if m in methods)}

    own = [per_dataset(inst, m, share, 0.0).xs(m, level="method", drop_level=False)
           for m in SAMPLED if m in methods]
    by_method = pd.concat([blocks[SUBSET_LABEL["all"]]] + own).sort_index()
    fig = plot_precision_coverage(by_method)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"instance_precision_coverage.{ext}", dpi=200, facecolor="white", bbox_inches="tight")
    print(f"wrote {out_dir / 'instance_precision_coverage'}.{{pdf,png}}")
    write_table(by_method, share, out_dir / "instance_table.tex")
    write_summary(blocks, METRICS, methods, out_dir / "instance_summary.tex")

    g_methods = [m for m in METHOD_STYLE if (glob.method == m).any()]
    g_blocks = {GLOBAL_LABEL[s]: global_per_dataset(glob, s)
                for s in ("all",) + tuple(m for m in SAMPLED if m in g_methods)}
    write_summary(g_blocks, GLOBAL_METRICS, g_methods, out_dir / "global_summary.tex")
    claims(inst, args.threshold)


if __name__ == "__main__":
    main()
