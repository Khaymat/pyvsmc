# pysmc

[![Python Version](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-orange.svg)](https://pypi.org/project/pysmc/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![Type Checked](https://img.shields.io/badge/mypy-strict-blue.svg)](#type-safety)
[![Ruff](https://img.shields.io/badge/lint-ruff-red.svg)](https://github.com/astral-sh/ruff)

**Ultra-fast, fully vectorized market structure & Smart Money Concepts (SMC) for Python.**

`pysmc` provides pure NumPy + Polars implementations of the most widely used SMC / ICT concepts — Fair Value Gaps, fractal swings, Break of Structure (BOS), Change of Character (CHOCH), and Order Blocks — with **zero Python for-loops over time-series**, strict typing, and a clean Polars plugin.

---

## Features

| Module | Concept | Key Function |
|--------|---------|--------------|
| `fvg` | Fair Value Gap / Imbalance | `detect_fvg()` |
| `swings` | Fractal Swing Highs & Lows | `detect_swings()` |
| `structure` | BOS & CHOCH Engine | `detect_structure()` |
| `order_blocks` | Order Block Zones | `detect_order_blocks()` |
| `polars_ext` | Polars `.smc` namespace | `df.smc.add_all()` |

- **Performance:** 100% vectorized (`NumPy` / `Polars` vector expressions). No `.iterrows()`, `.apply()`, or Python loops over bars. Handles 100k+ candles in milliseconds.
- **Type Safety:** Strict `mypy` — all public APIs are fully typed with Google-style docstrings.
- **Polars Native:** Optional `pl.DataFrame.smc.*` namespace + `add_smc_columns()` helper.
- **Tested:** Comprehensive `pytest` suite covering normal, edge (empty, flat, NaN, length < 3), and benchmark cases.

---

## Installation

```bash
pip install pysmc
```

With Polars (recommended):

```bash
pip install "pysmc[dev]"   # includes polars, pytest, ruff, mypy
# or
pip install pysmc polars
```

From source:

```bash
git clone https://github.com/pysmc/pysmc
cd pysmc
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, `numpy>=1.24.0`, `polars>=0.20.0` (optional but recommended).

---

## Quickstart

### NumPy API

```python
import numpy as np
import pysmc as smc

# OHLC arrays (float)
high  = np.array([10.0, 11.2, 10.8, 12.5, 11.0, 13.0])
low   = np.array([ 9.5,  9.8, 10.0, 11.8, 10.5, 12.2])
close = np.array([10.0, 10.5, 10.2, 12.2, 11.1, 12.8])
open_ = np.array([ 9.8, 10.0, 10.4, 11.0, 11.5, 12.0])

# 1. Fair Value Gaps (with mitigation tracking)
fvg = smc.detect_fvg(high, low, min_gap_size=0.3, compute_mitigation=True)
print(fvg.bullish)          # boolean mask
print(fvg.bullish_upper)    # upper boundary (Low[i])
print(fvg.mitigated)        # has price revisited the gap?

# 2. Fractal Swings
swings = smc.detect_swings(high, low, window_size=2)
print(swings.swing_high)       # True where High[i] == max(window)
print(swings.swing_high_price)

# 3. Market Structure — BOS & CHOCH
structure = smc.detect_structure(high, low, close, window_size=2)
print(structure.bos_bullish)    # continuation breaks
print(structure.choch_bearish)  # reversal breaks
print(structure.trend)          # 1=bull, -1=bear, 0=neutral

# 4. Order Blocks
obs = smc.detect_order_blocks(open_, high, low, close, lookback=5)
print(obs.bullish_ob)  # True at the bearish candle before a bullish impulse
print(obs.ob_high, obs.ob_low)
```

### Polars API

```python
import polars as pl
import pysmc  # registers .smc namespace

df = pl.DataFrame({
    "open":  open_,
    "high":  high,
    "low":   low,
    "close": close,
    "volume": [100, 120, 80, 200, 150, 180],
})

# Functional helper — adds all SMC columns at once
from pysmc.polars_ext import add_smc_columns
df = add_smc_columns(df, window_size=2, fvg_mitigation=True)

# Or via the .smc namespace (more granular)
df = pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})
df = df.smc.add_all(window_size=2, ob_lookback=10)
df = df.smc.fvg(min_gap_size=0.5)
df = df.smc.swings(window_size=2)
df = df.smc.structure(window_size=2)
df = df.smc.order_blocks(lookback=5)

print(df)
```

### Re-using Swings in Structure

```python
from pysmc.swings import detect_swings
from pysmc.structure import detect_structure

swings = detect_swings(high, low, window_size=3)
structure = detect_structure(
    high, low, close,
    swing_high=swings.swing_high,
    swing_low=swings.swing_low,
)
```

---

## API Reference

### `detect_fvg(high, low, min_gap_size=None, min_gap_size_pct=None, *, compute_mitigation=False)`

Detects 3-candle Fair Value Gaps.

- **Bullish FVG:** `Low[i] > High[i-2]` — gap zone `[High[i-2], Low[i]]`
- **Bearish FVG:** `High[i] < Low[i-2]` — gap zone `[High[i], Low[i-2]]`

Returns `FVGResult` with `bullish`, `bearish`, `bullish_upper/lower`, `bearish_upper/lower`, `gap_size`, `gap_size_pct`, `mitigated`, `mitigated_index`.

### `detect_swings(high, low, window_size=2)`

Fractal swing detection. `High[i] == max(High[i-N:i+N+1])`, `Low[i] == min(Low[i-N:i+N+1])`. Returns `SwingResult`.

### `detect_structure(high, low, close, window_size=2, *, swing_high=None, swing_low=None)`

BOS (continuation) vs CHOCH (reversal) classification with trend tracking. Returns `StructureResult` with `bos_bullish/bearish`, `choch_bullish/bearish`, `bos_level`, `choch_level`, `trend`.

### `detect_order_blocks(open_, high, low, close, *, lookback=10, ...)`

Finds last opposing candle before FVG/BOS impulses. Returns `OrderBlockResult` with `bullish_ob/bearish_ob`, `ob_high/low`, `validated_index`, `impulse_type`.

All functions also have `*_polars(df, ...)` variants and are available via `df.smc.*`.

---

## Testing

```bash
pip install -e ".[dev]"
pytest -v
pytest --cov=pysmc --cov-report=term-missing
```

Run type checks and lint:

```bash
mypy src/pysmc
ruff check src/pysmc tests
```

---

## Project Structure

```
pysmc/
├── pyproject.toml
├── README.md
├── src/pysmc/
│   ├── __init__.py
│   ├── py.typed
│   ├── fvg.py
│   ├── swings.py
│   ├── structure.py
│   ├── order_blocks.py
│   └── polars_ext.py
└── tests/
    ├── test_fvg.py
    ├── test_swings.py
    ├── test_structure.py
    ├── test_order_blocks.py
    └── test_polars_ext.py
```

---

## Performance Notes

- All indicators use `numpy.lib.stride_tricks.sliding_window_view`, `np.maximum.accumulate`, broadcasting, and chunked evaluation — **no Python loops over bars**.
- The single exception is the BOS/CHOCH trend tracker, which requires sequential state. It is JIT-compiled with `numba` when available and falls back to a single O(n) scan otherwise.
- Benchmark: ~100k candles — FVG < 10ms, swings < 20ms, structure < 30ms (CPython 3.11, NumPy 1.26).

---

## Financial & Legal Disclaimer

**IMPORTANT — PLEASE READ CAREFULLY**

`pysmc` is an **open-source analytics and research library**. It is provided solely for **educational, informational, and research purposes**.

- **Not Financial Advice.** Nothing in this library, its documentation, examples, or outputs constitutes financial, investment, trading, or other professional advice. No recommendation to buy, sell, or hold any financial instrument is made or implied.
- **No Warranty of Accuracy or Fitness.** Market structure and Smart Money Concepts are *interpretive frameworks*; their definitions vary across practitioners. The library implements one set of rules that may not match your trading methodology. Outputs may be incorrect, incomplete, or inappropriate for your use case.
- **Use at Your Own Risk.** Trading and investing involve substantial risk of loss, including loss of principal. Past simulated or historical performance is not indicative of future results. You are solely responsible for your own trading decisions, risk management, and compliance with applicable laws and regulations.
- **No Liability.** To the fullest extent permitted by law, the authors, contributors, and distributors of `pysmc` disclaim all liability for any loss, damage, cost, or expense arising directly or indirectly from use of this software.
- **Do Your Own Research (DYOR).** Always validate any signal or analysis with independent research, additional data sources, and, where appropriate, advice from a qualified professional.

By using this software you acknowledge that you have read, understood, and agree to this disclaimer.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (c) 2026 pysmc contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Contributing

Issues and pull requests are welcome. Please run `ruff`, `mypy`, and `pytest` before submitting.

## Acknowledgements

Built with [NumPy](https://numpy.org) and [Polars](https://pola.rs). SMC concepts as described by the broader ICT / Smart Money community.
