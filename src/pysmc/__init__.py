"""pysmc — ultra-fast vectorized Smart Money Concepts.

Public API
----------
* :mod:`pysmc.fvg` — Fair Value Gap detection
* :mod:`pysmc.swings` — Fractal swing highs/lows
* :mod:`pysmc.structure` — BOS & CHOCH engine
* :mod:`pysmc.order_blocks` — Order block identification
* :mod:`pysmc.polars_ext` — Polars ``.smc`` namespace

Quickstart
----------
.. code-block:: python

    import numpy as np
    import pysmc as smc

    high  = np.array([...], dtype=float)
    low   = np.array([...], dtype=float)
    close = np.array([...], dtype=float)
    open_ = np.array([...], dtype=float)

    fvg = smc.detect_fvg(high, low)
    swings = smc.detect_swings(high, low, window_size=2)
    structure = smc.detect_structure(high, low, close)
    obs = smc.detect_order_blocks(open_, high, low, close)

    # Polars (optional)
    import polars as pl
    df = pl.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    df = smc.add_smc_columns(df)          # functional
    df = df.smc.add_all()                 # namespace
"""

from __future__ import annotations

from .fvg import FVGResult, detect_fvg, fvg_polars
from .order_blocks import OrderBlockResult, detect_order_blocks, order_blocks_polars
from .structure import StructureResult, detect_structure, structure_polars
from .swings import SwingResult, detect_swings, swings_polars

# Polars helpers — imported lazily to keep polars optional, but re-exported
# here for convenience.  Importing pysmc should not fail if polars is
# absent; polars_ext handles that gracefully.
try:
    from .polars_ext import add_smc_columns  # noqa: F401
except ImportError:
    add_smc_columns = None  # type: ignore[assignment]

__all__ = [
    # FVG
    "detect_fvg",
    "FVGResult",
    "fvg_polars",
    # Swings
    "detect_swings",
    "SwingResult",
    "swings_polars",
    # Structure
    "detect_structure",
    "StructureResult",
    "structure_polars",
    # Order blocks
    "detect_order_blocks",
    "OrderBlockResult",
    "order_blocks_polars",
    # Polars
    "add_smc_columns",
]

__version__ = "0.1.0"
