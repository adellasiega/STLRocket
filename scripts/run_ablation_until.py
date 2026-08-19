#!/usr/bin/env python
"""Ablation: does the Until temporal operator improve balanced accuracy?

For each dataset, runs run_experiment.py twice (same seeds) with until_weight=0.0
("without Until") and until_weight=0.2 ("with Until"), then compares balanced
accuracy per condition via median/IQR and a paired Wilcoxon signed-rank test on
the matched per-seed differences.

Each (dataset, condition) run is a separate subprocess so a crash/OOM on one
(e.g. Until's known O(T^3) cost on long-T datasets) doesn't take down the sweep --
that dataset is just marked "failed" and the driver moves on.

Usage:
    python scripts/run_ablation_until.py --device cuda
    python scripts/run_ablation_until.py --datasets BasicMotions Epilepsy --n_run 3
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "StandWalkJump",
    "UWaveGestureLibrary",
]

OUTPUT_DIR_RE = re.compile(r"^Output directory: (.+)$", re.MULTILINE)


def run_condition(
    dataset: str,
    until_weight: float,
    n_run: int,
    base_seed: int,
    n_formulas: int,
    depth_max: int,
    device: str,
    output_dir: Path,
    timeout: float,
) -> list[float] | None:
    """Run run_experiment.py for one (dataset, until_weight) pair.

    Returns the list of per-seed balanced_accuracy values (ordered by seed), or
    None if the run failed/timed out/produced unreadable output.
    """
    cmd = [
        sys.executable, str(REPO_ROOT / "run_experiment.py"),
        "--dataset", dataset,
        "--until_weight", str(until_weight),
        "--n_run", str(n_run),
        "--base_seed", str(base_seed),
        "--n_formulas", str(n_formulas),
        "--depth_max", str(depth_max),
        "--device", device,
        "--explain", "false",
        "--output_dir", str(output_dir),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    [{dataset} until_weight={until_weight}] TIMEOUT after {timeout}s")
        return None

    if proc.returncode != 0:
        print(f"    [{dataset} until_weight={until_weight}] FAILED (exit {proc.returncode})")
        print(proc.stderr[-2000:])
        return None

    match = OUTPUT_DIR_RE.search(proc.stdout)
    if not match:
        print(f"    [{dataset} until_weight={until_weight}] could not find output directory in stdout")
        return None

    accuracy_path = Path(match.group(1)) / "accuracy.json"
    try:
        summary = json.loads(accuracy_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"    [{dataset} until_weight={until_weight}] could not read {accuracy_path}: {e}")
        return None

    per_run = sorted(summary["per_run"], key=lambda r: r["seed"])
    return [r["balanced_accuracy"] for r in per_run]


def iqr(values: list[float]) -> float:
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--n_run", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--n_formulas", type=int, default=100)
    parser.add_argument("--depth_max", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=float, default=1800, help="Per-condition subprocess timeout, seconds")
    parser.add_argument("--output_dir", default="results/ablation_until")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    without_dir = output_root / "without_until"
    with_dir = output_root / "with_until"
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        without = run_condition(
            dataset, 0.0, args.n_run, args.base_seed, args.n_formulas, args.depth_max,
            args.device, without_dir, args.timeout,
        )
        with_until = run_condition(
            dataset, 0.2, args.n_run, args.base_seed, args.n_formulas, args.depth_max,
            args.device, with_dir, args.timeout,
        )

        if without is None or with_until is None:
            status = "failed_without" if without is None else "failed_with"
            rows.append({
                "dataset": dataset, "median_without": "", "iqr_without": "",
                "median_with": "", "iqr_with": "", "delta_median": "",
                "wilcoxon_p": "", "n_pairs": 0, "status": status,
            })
            continue

        n_pairs = min(len(without), len(with_until))
        without, with_until = without[:n_pairs], with_until[:n_pairs]
        med_without, med_with = float(np.median(without)), float(np.median(with_until))

        diffs = np.array(with_until) - np.array(without)
        if np.all(diffs == 0):
            p_value = 1.0
        else:
            _, p_value = wilcoxon(diffs)

        rows.append({
            "dataset": dataset,
            "median_without": round(med_without, 4),
            "iqr_without": round(iqr(without), 4),
            "median_with": round(med_with, 4),
            "iqr_with": round(iqr(with_until), 4),
            "delta_median": round(med_with - med_without, 4),
            "wilcoxon_p": round(float(p_value), 4),
            "n_pairs": n_pairs,
            "status": "ok",
        })
        print(f"    without: median={med_without:.4f}  with: median={med_with:.4f}  "
              f"delta={med_with - med_without:+.4f}  p={p_value:.4f}")

    summary_path = output_root / "summary.csv"
    fieldnames = ["dataset", "median_without", "iqr_without", "median_with", "iqr_with",
                  "delta_median", "wilcoxon_p", "n_pairs", "status"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    failed_rows = [r for r in rows if r["status"] != "ok"]
    significant = [r for r in ok_rows if r["wilcoxon_p"] < 0.05]

    print(f"\n{'=' * 60}")
    print(f"Summary saved to: {summary_path}")
    print(f"{len(ok_rows)}/{len(rows)} datasets completed both conditions "
          f"({len(failed_rows)} failed: {[r['dataset'] for r in failed_rows]})")
    print(f"{len(significant)}/{len(ok_rows)} datasets show a significant difference (p<0.05):")
    for r in significant:
        direction = "Until helps" if r["delta_median"] > 0 else "Until hurts"
        print(f"    {r['dataset']}: delta={r['delta_median']:+.4f}  p={r['wilcoxon_p']:.4f}  ({direction})")


if __name__ == "__main__":
    main()
