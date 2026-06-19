#!/usr/bin/env python
"""Visualize explanation quality from a ``retrain_best.py`` output directory.

``local_experiments/retrain_best.py`` writes, per dataset, a run directory:

    <run_dir>/
        config.json                 <- dataset, n_formulae, depth_max
        metrics.json                <- the quality numbers visualized here
        local_explanations.csv
        global_explanations.csv
        kde_<class>_{train,test}.png

This script reads ``metrics.json`` and saves two figures per run directory:

    explanation_global_<dataset>.png   global per-class coverage / FP-rate /
                                       precision / F1 (train vs test)
    explanation_local_<dataset>.png    local per-class mean precision, mean
                                       formula length, fraction of instances
                                       explained

It mirrors the CLI of ``scripts/plot_comparison.py``: pass a single run directory,
or a parent directory with ``--glob`` to process every run subdirectory under it.

Usage
-----
    # one run directory
    python scripts/plot_explanations.py results/explanations/Libras

    # every run directory under a parent
    python scripts/plot_explanations.py results/explanations --glob

    # show interactively in addition to saving
    python scripts/plot_explanations.py <run_dir> --show
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="tab10")

# train/test colors, stable across both figures.
SPLIT_COLORS = {"train": sns.color_palette("tab10")[0],
                "test": sns.color_palette("tab10")[1]}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _class_sort_key(c: str):
    """Sort classes numerically when possible ('1','2',...,'10'), else lexically."""
    try:
        return (0, float(c))
    except (TypeError, ValueError):
        return (1, str(c))


def load_metrics(run_dir: Path) -> tuple[dict, str, str]:
    """Return (metrics, dataset, subtitle) for a run directory.

    ``dataset`` / ``subtitle`` come from config.json when present, falling back to
    the directory name for the dataset.
    """
    metrics = json.loads((run_dir / "metrics.json").read_text())

    dataset, n_formulae, depth_max = run_dir.name, None, None
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        dataset = cfg.get("dataset", dataset)
        n_formulae = cfg.get("n_formulae")
        depth_max = cfg.get("depth_max")

    parts = []
    if n_formulae is not None:
        parts.append(f"n={n_formulae}")
    if depth_max is not None:
        parts.append(f"d={depth_max}")
    subtitle = ", ".join(parts)
    return metrics, dataset, subtitle


# ---------------------------------------------------------------------------
# Figure 1 — global per-class quality
# ---------------------------------------------------------------------------

def plot_global_quality(metrics, dataset, subtitle, out_path, show):
    gpc = metrics.get("global_per_class", {})
    classes = sorted(gpc, key=_class_sort_key)
    if not classes:
        print("  (no global_per_class metrics; skipping global figure)")
        return

    x = np.arange(len(classes))
    width = 0.4

    # (metric key, axis title, lower-is-better, optional macro reference)
    panels = [
        ("coverage", "coverage (recall)", False, None),
        ("fp_rate", "FP rate (lower is better)", True, None),
        ("precision", "precision", False, None),
        ("f1", "F1", False, ("global_macro_f1_train", "global_macro_f1_test")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (key, title, lower_better, macro) in zip(axes.ravel(), panels):
        for i, split in enumerate(("train", "test")):
            vals = [gpc[c].get(split, {}).get(key, np.nan) for c in classes]
            ax.bar(x + (i - 0.5) * width, vals, width,
                   label=split, color=SPLIT_COLORS[split])
        if macro is not None:
            for split, mkey in zip(("train", "test"), macro):
                mv = metrics.get(mkey)
                if mv is not None and not np.isnan(mv):
                    ax.axhline(mv, color=SPLIT_COLORS[split], linestyle="--",
                               linewidth=1, label=f"macro {split} ({mv:.2f})")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("class")
    title = f"{dataset} — global explanation quality"
    if subtitle:
        title += f" ({subtitle})"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"  saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — local per-class quality
# ---------------------------------------------------------------------------

def plot_local_quality(metrics, dataset, subtitle, out_path, show):
    lpc = metrics.get("local_per_class", {})
    classes = sorted(lpc, key=_class_sort_key)
    if not classes:
        print("  (no local_per_class metrics; skipping local figure)")
        return

    x = np.arange(len(classes))
    palette = sns.color_palette("tab10")

    mean_prec = [lpc[c].get("mean_precision", np.nan) for c in classes]
    std_prec = [lpc[c].get("std_precision", 0.0) or 0.0 for c in classes]
    mean_len = [lpc[c].get("mean_length", np.nan) for c in classes]
    frac_expl = [
        (lpc[c]["n_explained"] / lpc[c]["n_total"]) if lpc[c].get("n_total") else np.nan
        for c in classes
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(x, mean_prec, yerr=std_prec, capsize=3,
                color=palette[2], error_kw={"elinewidth": 1, "alpha": 0.7})
    axes[0].set_title("mean local precision (±std)")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(x, mean_len, color=palette[3])
    axes[1].set_title("mean formula length (# conjuncts)")
    axes[1].set_ylim(bottom=0)

    axes[2].bar(x, frac_expl, color=palette[4])
    axes[2].set_title("fraction of instances explained")
    axes[2].set_ylim(0, 1.05)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("class")

    title = f"{dataset} — local explanation quality"
    if subtitle:
        title += f" ({subtitle})"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"  saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_run_dir(run_dir: Path, show: bool) -> None:
    metrics, dataset, subtitle = load_metrics(run_dir)
    n_classes = len(metrics.get("global_per_class", {}))
    f1_tr = metrics.get("global_macro_f1_train")
    f1_te = metrics.get("global_macro_f1_test")
    f1_str = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else "n/a"
    print(f"{run_dir.name} (dataset={dataset}, {n_classes} classes, "
          f"macro-F1 train={f1_str(f1_tr)} test={f1_str(f1_te)})")

    plot_global_quality(metrics, dataset, subtitle,
                        run_dir / f"explanation_global_{dataset}.png", show)
    plot_local_quality(metrics, dataset, subtitle,
                       run_dir / f"explanation_local_{dataset}.png", show)


def find_run_dirs(root: Path, use_glob: bool) -> list[Path]:
    if not use_glob:
        return [root]
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and (d / "metrics.json").exists())


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("path", type=Path,
                   help="a retrain_best.py run directory, or a parent dir with --glob")
    p.add_argument("--glob", action="store_true",
                   help="treat PATH as a parent and plot every run subdirectory")
    p.add_argument("--show", action="store_true", help="display figures interactively")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    run_dirs = find_run_dirs(args.path, args.glob)
    if not run_dirs:
        raise SystemExit(f"no run directories with metrics.json under {args.path}")

    for run_dir in run_dirs:
        if not (run_dir / "metrics.json").exists():
            print(f"  skip {run_dir.name}: no metrics.json")
            continue
        process_run_dir(run_dir, args.show)


if __name__ == "__main__":
    main()
