"""pyvsmc — ultra-fast vectorized Smart Money Concepts.

Public API
----------
* :mod:`pyvsmc.fvg` — Fair Value Gap detection
* :mod:`pyvsmc.swings` — Fractal swing highs/lows
* :mod:`pyvsmc.structure` — BOS & CHOCH engine
* :mod:`pyvsmc.order_blocks` — Order block identification
* :mod:`pyvsmc.liquidity` — Liquidity sweeps & equal highs/lows
* :mod:`pyvsmc.zones` — Premium/discount & dealing range
* :mod:`pyvsmc.polars_ext` — Polars ``.smc`` namespace

Quickstart
----------
.. code-block:: python

    import numpy as np
    import pyvsmc as smc

    high  = np.array([...], dtype=float)
    low   = np.array([...], dtype=float)
    close = np.array([...], dtype=float)
    open_ = np.array([...], dtype=float)

    fvg = smc.detect_fvg(high, low)
    swings = smc.detect_swings(high, low, window_size=2)
    structure = smc.detect_structure(high, low, close)
    obs = smc.detect_order_blocks(open_, high, low, close)
    liq = smc.detect_liquidity(high, low, close)
    zones = smc.detect_zones(high, low, close)

    # Polars (optional)
    import polars as pl
    df = pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    df = smc.add_smc_columns(df)          # functional
    df = df.smc.add_all()                 # namespace
"""

from __future__ import annotations

from .fvg import FVGResult, detect_fvg, fvg_polars
from .liquidity import LiquidityResult, detect_liquidity, liquidity_polars
from .order_blocks import OrderBlockResult, detect_order_blocks, order_blocks_polars
from .structure import StructureResult, detect_structure, structure_polars
from .swings import SwingResult, detect_swings, swings_polars
from .zones import ZoneResult, detect_zones, zones_polars

try:
    from .polars_ext import add_smc_columns  # noqa: F401
except ImportError:
    add_smc_columns = None  # type: ignore[assignment]

__all__ = [
    "detect_fvg", "FVGResult", "fvg_polars",
    "detect_swings", "SwingResult", "swings_polars",
    "detect_structure", "StructureResult", "structure_polars",
    "detect_order_blocks", "OrderBlockResult", "order_blocks_polars",
    "detect_liquidity", "LiquidityResult", "liquidity_polars",
    "detect_zones", "ZoneResult", "zones_polars",
    "add_smc_columns",
]

__version__ = "0.2.0"
