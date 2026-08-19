#!/usr/bin/env python
"""Benchmark Eventually/Always/Until robustness computation cost vs signal length T.

Until's mask construction stacks one [~2T, T] mask per offset in its interval, so an
unbounded Until (interval=None, spanning the whole signal) scales roughly O(T^3) in time
and memory, versus O(T^2) for Eventually/Always. This script measures that directly on
whatever --device is given, to confirm the blowup on long-time-series datasets.

Usage:
    python scripts/benchmark_until.py --device cuda
    python scripts/benchmark_until.py --device cpu --t_values 50 100 200 400
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the repo root importable when run as a script from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from stlrocket.stl import Always, Eventually, Until, atom


def format_bytes_mb(n: int) -> float:
    return n / 1e6


def benchmark(device: str, t_values: list[int], n_vars: int = 3) -> None:
    a = atom(0, 0.0, lte=False)
    b = atom(1, 0.0, lte=True)

    has_cuda_mem_stats = device.startswith("cuda") and torch.cuda.is_available()

    print(f"{'T':>6}  {'operator':<10}  {'time_ms':>10}  {'peak_mem_MB':>12}")
    print("-" * 46)

    for T in t_values:
        signal = torch.randn(T, n_vars, device=device)

        operators = [
            ("Eventually", Eventually(a, interval=None)),
            ("Always", Always(a, interval=None)),
            ("Until", Until(a, b, interval=None)),
        ]

        for name, phi in operators:
            if has_cuda_mem_stats:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    phi.robustness(signal)
                if has_cuda_mem_stats:
                    torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) * 1000
                peak_mb = format_bytes_mb(torch.cuda.max_memory_allocated()) if has_cuda_mem_stats else float("nan")
                print(f"{T:6d}  {name:<10}  {dt_ms:10.2f}  {peak_mb:12.1f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{T:6d}  {name:<10}  {'OOM':>10}  {'OOM':>12}")


def count_unbounded_until(n_vars: int, t_max: int, n_formulas: int, depth_max: int, until_weight: float, seed: int) -> None:
    import numpy as np

    from stlrocket.formula_sampler import F0

    v_min = np.array([-1.0] * n_vars)
    v_max = np.array([1.0] * n_vars)
    gen = F0(
        n_vars=n_vars, v_min=v_min, v_max=v_max, t_max=t_max, depth_max=depth_max,
        seed=seed, only_temporal=True, until_weight=until_weight,
    )
    formulas = gen.sample(n_formulas)

    def has_unbounded_until(node) -> bool:
        if isinstance(node, Until) and node.interval is None:
            return True
        for attr in ("subformula", "subformula1", "subformula2"):
            child = getattr(node, attr, None)
            if child is not None and has_unbounded_until(child):
                return True
        return False

    n_unbounded = sum(has_unbounded_until(f) for f in formulas)
    print(f"\n{n_unbounded}/{len(formulas)} sampled formulas (until_weight={until_weight}, "
          f"depth_max={depth_max}, t_max={t_max}) contain an unbounded Until")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--t_values", type=int, nargs="+", default=[50, 100, 200, 400, 800, 1600])
    parser.add_argument("--n_vars", type=int, default=3)
    parser.add_argument("--until_weight", type=float, default=0.1)
    parser.add_argument("--n_formulas", type=int, default=200)
    parser.add_argument("--depth_max", type=int, default=3)
    args = parser.parse_args()

    benchmark(args.device, args.t_values, args.n_vars)
    count_unbounded_until(
        n_vars=args.n_vars,
        t_max=max(args.t_values),
        n_formulas=args.n_formulas,
        depth_max=args.depth_max,
        until_weight=args.until_weight,
        seed=0,
    )


if __name__ == "__main__":
    main()
