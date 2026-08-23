"""Adversarial FVG benchmark — pathological Theta(n) gaps with d~n.

Dataset:
- n bullish FVGs (g = n-2) where Low[i]=11 > High[i-2]=10 for all i>=2
- Final bar Low[n-1]=9 mitigates all prior gaps at last index => d ~ n/2 avg

Reports n, g, avg distance, runtime, growth ratio, empirical p without asserting p≈2.
This is separate from main benchmark headline (typical random data).
CI-safe: default n=1000,2000,5000,10000 (not 80k).

Usage:
    python benchmarks/bench_adversarial.py
    python benchmarks/bench_adversarial.py --sizes 1000,2000,5000,10000,20000
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

import pyvsmc


def make_adversarial(n: int):
    # Only bullish gaps, no bearish, to avoid shared mitigated overwrite
    # Low increases by 1 per bar, High = Low+1, so bullish Low[i] > High[i-2] true, bearish High[i] < Low[i-2] false
    base = 10.0 + np.arange(n, dtype=np.float64) * 1.0
    low = base
    high = base + 1.0
    low[-1] = 9.0
    high[-1] = 10.0
    close = np.full(n, 10.5, dtype=np.float64)
    close[-1] = 9.0
    return high, low, close


def bench_one(n: int, runs: int = 3):
    high, low, close = make_adversarial(n)
    # warmup numba (first call compiles)
    pyvsmc.detect_fvg(high[:10], low[:10], compute_mitigation=True, close=close[:10])
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        res = pyvsmc.detect_fvg(high, low, compute_mitigation=True, close=close)
        times.append(time.perf_counter() - t0)
    # correctness check
    g = int(res.bullish.sum())
    mitig = int(res.mitigated.sum())
    # avg distance
    idx = np.where(res.bullish)[0]
    dists = []
    for i in idx:
        mi = res.mitigated_index[i]
        if mi != -1:
            dists.append(mi - i)
    avg_d = float(np.mean(dists)) if dists else float("nan")
    return {
        "t_med": float(np.median(times)),
        "t_best": float(np.min(times)),
        "g": g,
        "mitigated": mitig,
        "avg_d": avg_d,
        "first_idx_all_999": bool(np.all(res.mitigated_index[res.bullish] == n - 1)) if n <= 5000 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="1000,2000,5000,10000")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print(f"pyvsmc {pyvsmc.__version__} adversarial FVG - g=Theta(n), d~n/2, all mitigate at last bar")
    print("n      g      avg_d   t_med     t_best   growth  p_empir")
    print("-" * 70)
    prev_n = None
    prev_t = None
    for n in sizes:
        r = bench_one(n, runs=3)
        growth = r["t_med"] / prev_t if prev_t else float("nan")
        # empirical p = log(growth)/log(n/prev_n)
        if prev_n and prev_t and prev_t > 0:
            p = math.log(growth) / math.log(n / prev_n) if growth > 0 else float("nan")
            p_str = f"{p:.2f}"
        else:
            p_str = "-"
        growth_str = f"{growth:.2f}x" if not np.isnan(growth) else "-"
        print(f"{n:6d} {r['g']:6d} {r['avg_d']:7.1f} {r['t_med']*1000:8.2f}ms {r['t_best']*1000:8.2f}ms {growth_str:>7} {p_str:>6}")
        prev_n, prev_t = n, r["t_med"]
    print("\nNote: No assertion on p - quadratic is current limitation, not desired behavior.")
    print("Typical random 100k fvg_mit ~16ms (57x vs 0.3.1) for d small; adversarial worst remains Theta(n^2).")


if __name__ == "__main__":
    main()
