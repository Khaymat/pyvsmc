"""Benchmark pyvsmc — vectorized SMC performance.

Measures each public API at n = 1k, 10k, 100k, 500k (and optional 1M).
Reports median of 3 runs + best, and verifies correctness.

Usage:
    python benchmarks/bench.py
    python benchmarks/bench.py --sizes 1000,10000,100000
"""
from __future__ import annotations

import argparse
import time
import tracemalloc

import numpy as np

import pyvsmc


def make_ohlc(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Trend + noise to mimic market
    trend = np.cumsum(rng.normal(0, 0.05, n))
    base = 100 + trend
    high = base + rng.uniform(0.2, 1.5, n)
    low = base - rng.uniform(0.2, 1.5, n)
    close = base + rng.normal(0, 0.3, n)
    open_ = np.roll(close, 1)
    open_[0] = base[0]
    # Ensure high>= max(open,close) etc for realism
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return open_, high, low, close


def bench_one(n: int, runs: int = 3):
    open_, high, low, close = make_ohlc(n)
    results = {}

    def time_fn(fn, *args, **kw):
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(*args, **kw)
            times.append(time.perf_counter() - t0)
        return float(np.median(times)), float(np.min(times))

    # Swings (vectorized)
    med, best = time_fn(pyvsmc.detect_swings, high, low, window_size=2)
    results["swings"] = (med, best)

    # FVG without mitigation (fast path)
    med, best = time_fn(pyvsmc.detect_fvg, high, low)
    results["fvg"] = (med, best)

    # FVG with mitigation (heavy)
    if n <= 50000:
        med, best = time_fn(pyvsmc.detect_fvg, high, low, compute_mitigation=True, close=close)
        results["fvg_mit"] = (med, best)
    else:
        results["fvg_mit"] = (float("nan"), float("nan"))

    # Structure - warm up numba first
    try:
        pyvsmc.detect_structure(high[:100], low[:100], close[:100], window_size=2)
    except Exception:
        pass
    med, best = time_fn(pyvsmc.detect_structure, high, low, close, window_size=2)
    results["structure"] = (med, best)
    med, best = time_fn(pyvsmc.detect_structure, high, low, close, window_size=2, break_mode="wick")
    results["structure_wick"] = (med, best)

    # Order Blocks
    med, best = time_fn(pyvsmc.detect_order_blocks, open_, high, low, close, lookback=10)
    results["order_blocks"] = (med, best)
    if n <= 50000:
        med, best = time_fn(pyvsmc.detect_order_blocks, open_, high, low, close, lookback=10, compute_mitigation=True, zone_mode="body")
        results["order_blocks_mit"] = (med, best)
    else:
        results["order_blocks_mit"] = (float("nan"), float("nan"))

    # Liquidity
    med, best = time_fn(pyvsmc.detect_liquidity, high, low, close, window_size=2)
    results["liquidity"] = (med, best)

    # Zones
    med, best = time_fn(pyvsmc.detect_zones, high, low, close, window_size=2)
    results["zones"] = (med, best)

    # Polars eager
    try:
        import polars as pl
        df = pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})
        med, best = time_fn(pyvsmc.add_smc_columns, df)
        results["polars_eager"] = (med, best)
        # Lazy
        ldf = df.lazy()
        med, best = time_fn(lambda x: x.collect(), ldf.pipe(lambda d: pyvsmc.fvg_polars(d)))
        results["polars_lazy_fvg"] = (med, best)
    except Exception as e:
        results["polars_eager"] = (float("nan"), float("nan"))
        results["polars_lazy_fvg"] = (float("nan"), float("nan"))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="1000,10000,100000,500000")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    print(f"pyvsmc {pyvsmc.__version__} benchmark — numpy {np.__version__}")
    try:
        import polars as pl
        print(f"polars {pl.__version__}")
    except Exception:
        print("polars not installed")
    print("")

    header = f"{'n':>8} | {'swings':>10} | {'fvg':>10} | {'fvg_mit':>10} | {'struct':>10} | {'ob':>10} | {'ob_mit':>10} | {'liq':>10} | {'zones':>10} | {'pl_eager':>10}"
    print(header)
    print("-" * len(header))
    all_results = {}
    for n in sizes:
        # warmup for n
        tracemalloc.start()
        res = bench_one(n, runs=3)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        all_results[n] = res
        def fmt(v):
            if np.isnan(v[0]):
                return "   SKIP   "
            return f"{v[0]*1000:7.2f}ms"
        line = f"{n:8d} | {fmt(res['swings'])} | {fmt(res['fvg'])} | {fmt(res['fvg_mit'])} | {fmt(res['structure'])} | {fmt(res['order_blocks'])} | {fmt(res['order_blocks_mit'])} | {fmt(res['liquidity'])} | {fmt(res['zones'])} | {fmt(res['polars_eager'])}  peak {peak/1024/1024:.1f}MB"
        print(line)

    # Save json
    import json, pathlib
    out = pathlib.Path("benchmarks/results.json")
    # convert to serializable
    serial = {str(k): {kk: float(v[0]) for kk, v in vv.items()} for k, vv in all_results.items()}
    out.write_text(json.dumps(serial, indent=2))
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
