# pyvsmc Benchmark — 0.3.1

**Env:** `pyvsmc 0.3.1`, `numpy 2.4.4`, `polars 1.43.2`, `Python 3.11.5`, Windows 11, tracemalloc peak.

**Dataset:** Synthetic OHLC trend + noise (`make_ohlc`), `window_size=2`.

| n | swings | fvg | fvg_mit | struct (close) | order_blocks | ob_mit | liquidity | zones | polars eager | peak |
|---|---|---|---|---|---|---|---|---|---|---|
| 1k | 0.79ms | 1.16ms | 1.96ms | 17.5ms* | 19.68ms | 23.51ms | 11.30ms | 1.79ms | 58.75ms | 43.7MB |
| 10k | 2.57ms | 1.90ms | 2.40ms | 19.79ms* | 28.87ms | 42.10ms | 154.32ms | 4.58ms | 206.55ms | 3.6MB |
| 100k | 23.37ms | 8.42ms | SKIP | 47.82ms | 74.02ms | SKIP | 1306ms | 39.11ms | 1452ms | 36MB |
| 500k | 105.5ms | 80.0ms | — | 1506ms | 4283ms | — | 411ms | 187ms | — | — |

\* `struct` 1k/10k includes numba cold compile (~15ms). 100k 47ms, 500k 1.5s after warmup — `nb.njit(cache=True)` helps next run 0.05s/100k.

**Catatan Mit:** `fvg_mit` & `ob_mit` di-SKIP untuk 100k+ di bench utama karena `O(m*n)` per-gap (3× threshold). 10k mit masih OK (<3ms). Untuk 100k+ gunakan `compute_mitigation=False` (default) → fvg 8ms.

**Bottleneck:**
- `liquidity` 1.3s/100k — loop `for k in range(sh_idx.size)` di `liquidity.py:109` untuk EQH/EQL swing (20k swings). Ganti vectorized `np.abs(diff)/denom <= thr`.
- `polars eager` 1.4s — `add_smc_columns` panggil semua indikator (termasuk liquidity) + `to_numpy()` copy.
- `order_blocks` 4.2s/500k — `_find_last_opposing` per-impulse `window scan` O(m*lookback) masih berat untuk 500k random (banyak impulse). Workaround: `lookback=10` sudah minimal.

**Rekomendasi 0.3.2:** vectorize `equal_swing_*`, lazy-init `structure` warmup, dan `fvg_mit` single-pass.

Run: `python benchmarks/bench.py --sizes 1000,10000,100000`
Results: `benchmarks/results.json`
