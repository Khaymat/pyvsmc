# pyvsmc

[![Python Version](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/badge/pypi-v0.3.1-orange.svg)](https://pypi.org/project/pyvsmc/)
[![Tests](https://img.shields.io/badge/tests-92%20passed-brightgreen.svg)](#testing)
[![Type Checked](https://img.shields.io/badge/mypy-strict-blue.svg)](#testing)
[![Ruff](https://img.shields.io/badge/lint-ruff-red.svg)](https://github.com/astral-sh/ruff)

**High-performance NumPy/Numba market-structure engine for Smart Money Concepts (SMC).**

`pyvsmc` provides NumPy/Numba + Polars implementations of SMC/ICT concepts — Fair Value Gaps, fractal swings, BOS/CHOCH, Order Blocks, Liquidity sweeps, Premium/Discount & OTE — with appropriate data structures and low asymptotic complexity.

Architecture: NumPy for bulk transforms, Numba `njit(cache=True)` for stateful/first-passage scans, Polars for DataFrame integration. Not “zero loops” — correctness and complexity first.

---

## Features

| Module | Concept | Key Function |
|--------|---------|--------------|
| `fvg` | Fair Value Gap / Imbalance + CE 50% + IFVG | `detect_fvg()` |
| `swings` | Fractal Swing Highs & Lows | `detect_swings()` |
| `structure` | BOS & CHOCH (first-cross, break_mode) | `detect_structure()` |
| `order_blocks` | Order Block + zone_mode + breaker | `detect_order_blocks()` |
| `liquidity` | Equal highs/lows (swing-based) + sweeps | `detect_liquidity()` |
| `zones` | Premium/Discount + OTE 0.618/0.705/0.786 | `detect_zones()` |
| `polars_ext` | Polars `.smc` namespace | `df.smc.add_all()` |

- **Correctness:** Strict `mypy`, NaN-aware, first-cross BOS, CE50/full/inverted tracking.
- **Polars:** `pl.DataFrame.smc.*` + `add_smc_columns()`; `fvg`/`swings` have native `LazyFrame` expr, other modules fallback honest `collect()` (documented).
- **Tested:** 92 tests covering empty, flat, NaN, dense gaps, mitigation, EQH/EQL, OTE, LazyFrame.

---

## Installation

```bash
pip install pyvsmc
```

With Polars (recommended):

```bash
pip install "pyvsmc[dev]"   # includes polars, pytest, ruff, mypy, numba
# or
pip install pyvsmc polars numba
```

From source:

```bash
git clone https://github.com/Khaymat/pyvsmc
cd pyvsmc
pip install -e ".[dev]"
```

**Requirements:** Python >=3.10, `numpy>=1.24.0`, `polars>=0.20.0` (optional), `numba>=0.56` (optional, for JIT).

---

## Quickstart

### NumPy API

```python
import numpy as np
import pyvsmc as smc

high  = np.array([10.0, 11.2, 10.8, 12.5, 11.0, 13.0])
low   = np.array([ 9.5,  9.8, 10.0, 11.8, 10.5, 12.2])
close = np.array([10.0, 10.5, 10.2, 12.2, 11.1, 12.8])
open_ = np.array([ 9.8, 10.0, 10.4, 11.0, 11.5, 12.0])

# 1. FVG + CE/IFVG
fvg = smc.detect_fvg(high, low, close=close, compute_mitigation=True)
print(fvg.bullish, fvg.ce_level, fvg.mitigated_50, fvg.inverted)

# 2. Swings
swings = smc.detect_swings(high, low, window_size=2)

# 3. Structure — break_mode close/wick/both, first-cross
structure = smc.detect_structure(high, low, close, window_size=2, break_mode="close")
print(structure.bos_bullish, structure.trend)

# 4. Order Blocks — zone_mode full/body/mean_threshold + breaker
obs = smc.detect_order_blocks(open_, high, low, close, lookback=5, zone_mode="body", compute_mitigation=True)
print(obs.bullish_ob, obs.is_breaker)

# 5. Liquidity + Zones
liq = smc.detect_liquidity(high, low, close, equal_threshold=0.001)
zones = smc.detect_zones(high, low, close)
print(liq.equal_swing_high, liq.sweep_high, zones.in_ote)
```

### Polars API

```python
import polars as pl
import pyvsmc  # registers .smc namespace

df = pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})

# Eager
from pyvsmc.polars_ext import add_smc_columns
df = add_smc_columns(df, window_size=2, fvg_mitigation=True)
# Lazy (fvg/swings native, others collect fallback)
ldf = df.lazy()
ldf = ldf.pipe(lambda d: pyvsmc.fvg_polars(d))  # native expr

# Namespace
df = pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})
df = df.smc.add_all(window_size=2, include_liquidity=True)
df = df.smc.fvg(min_gap_size=0.5)
df = df.smc.swings(window_size=2)
df = df.smc.structure(window_size=2, break_mode="wick")
df = df.smc.order_blocks(lookback=5, zone_mode="body")
df = df.smc.liquidity(equal_threshold=0.001)
df = df.smc.zones(eq_threshold=0.02)
```

---

## API Reference

### `detect_fvg(high, low, min_gap_size=None, min_gap_size_pct=None, *, compute_mitigation=False, close=None)`
Bullish `Low[i] > High[i-2]` `[High[i-2], Low[i]]`, Bearish `High[i] < Low[i-2]`.
Returns `FVGResult` with `bullish/bearish`, `bullish_upper/lower`, `ce_level`, `gap_size`, `mitigated/mitigated_50/mitigated_full/inverted` + indices. `close` enables IFVG.

### `detect_swings(high, low, window_size=2)`
`High[i]==max(window)`, `Low[i]==min(window)`. Returns `SwingResult`.

### `detect_structure(high, low, close, window_size=2, *, break_mode="close")`
`break_mode="close"|"wick"|"both"`, first-cross only. Returns `bos_bullish/bearish`, `choch_*`, `bos_level`, `trend`.

### `detect_order_blocks(open_, high, low, close, *, lookback=10, zone_mode="full", compute_mitigation=False)`
`zone_mode="full"|"body"|"mean_threshold"`. Returns `bullish_ob/bearish_ob`, `ob_high/low`, `mitigated/is_breaker`.

### `detect_liquidity(high, low, close, *, equal_threshold=0.001, sweep_lookback=20, window_size=2)`
Returns `equal_high/low` (adjacent) + `equal_swing_high/low` (EQH/EQL), `sweep_high/low`.

### `detect_zones(high, low, close, window_size=2, eq_threshold=0.02)`
Returns `premium/discount/equilibrium`, `range_high/low`, `ote_high/low/705`, `in_ote`.

All have `*_polars` variants. Polars `add_smc_columns` params: `include_*`, `fvg_mitigation`, `break_mode`, `zone_mode` (see `polars_ext.py`).

---

## Testing

```bash
pip install -e ".[dev]"
pytest -v
pytest --cov=pyvsmc --cov-report=term-missing
```

Run type checks and lint:

```bash
mypy src/pyvsmc
ruff check src/pyvsmc tests
```

---

## Project Structure

```
pyvsmc/
├── pyproject.toml
├── README.md
├── src/pyvsmc/
│   ├── __init__.py
│   ├── py.typed
│   ├── fvg.py
│   ├── swings.py
│   ├── structure.py
│   ├── order_blocks.py
│   ├── liquidity.py
│   ├── zones.py
│   └── polars_ext.py
├── src/pysmc/          # shim deprecated → pyvsmc
├── benchmarks/
│   ├── bench.py
│   └── results.json
└── tests/
    ├── test_fvg.py + test_fvg_regression.py
    ├── test_swings.py
    ├── test_structure.py
    ├── test_order_blocks.py
    ├── test_liquidity.py
    ├── test_zones.py
    └── test_polars_ext.py
```

---

## Performance & Complexity

**Benchmark 0.3.1** (synthetic OHLC, `window_size=2`, 3 runs median, `tracemalloc` peak):

| n | swings | fvg | fvg+mit | struct | OB | liquidity | zones | polars eager |
|---|---|---|---|---|---|---|---|---|
| 1k | 0.79ms | 1.16ms | 1.96ms | 17.5ms* | 19.6ms | 11ms | 1.79ms | 58ms |
| 10k | 2.57ms | 1.90ms | 2.40ms | 19.7ms* | 28.8ms | 154ms | 4.5ms | 206ms |
| 100k | 23ms | 8.4ms | 19ms** | 47ms | 74ms | 1306ms | 39ms | 1452ms |

`* struct cold JIT ~15ms, warm 47ms/100k`, `** fvg_mit 0.3.1 Numba single-scan, 0.3.0 SKIP`

**Complexity:**
- `swings` `O(n)`, `fvg` w/o mit `O(n)`, `fvg+mit` `O(n + g·d)` (`g` gaps, `d` avg distance, Numba early-break, worst `O(g·n)` documented), `structure` `O(n)` Numba `cache=True`, `OB` `O(m·L)` bounded `L=lookback`, `liquidity EQH` `O(s)` vectorized diff.

**Limitations:** `fvg mit` with `g=Θ(n)` still worst `O(n²)` — avoid `compute_mitigation=True` on 500k+ dense gaps; `polars Lazy` for `fvg/swings` native, others fallback `collect()`; `order_blocks` many impulses → `lookback` bound.

---

## Financial & Legal Disclaimer

**IMPORTANT — PLEASE READ CAREFULLY**

`pyvsmc` is an **open-source analytics and research library**. It is provided solely for **educational, informational, and research purposes**.

- **Not Financial Advice.** Nothing in this library, its documentation, examples, or outputs constitutes financial, investment, trading, or other professional advice.
- **No Warranty of Accuracy or Fitness.** SMC are interpretive frameworks.
- **Use at Your Own Risk.** Trading involves substantial risk of loss.
- **No Liability.** Authors disclaim all liability.
- **Do Your Own Research (DYOR).**

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (c) 2026 pyvsmc contributors.

---

## Contributing

Issues and pull requests are welcome. Please run `ruff`, `mypy`, and `pytest` before submitting.

## Acknowledgements

Built with [NumPy](https://numpy.org), [Numba](https://numba.pydata.org) and [Polars](https://pola.rs). SMC concepts as described by the broader ICT / Smart Money community.
