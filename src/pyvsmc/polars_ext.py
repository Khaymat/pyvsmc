"""Polars extension / helper for pyvsmc — ``.smc`` namespace.

Provides a clean DataFrame helper that attaches all SMC indicator columns
via a single call, as well as individual method access through a
``.smc`` namespace object.

Usage
-----
.. code-block:: python

    import polars as pl
    import pyvsmc  # registers the extension

    df = pl.DataFrame({
        "open":  [...],
        "high":  [...],
        "low":   [...],
        "close": [...],
        "volume": [...],
    })

    # Option A: functional helper
    from pyvsmc.polars_ext import add_smc_columns
    df2 = add_smc_columns(df)

    # Option B: namespace (if you prefer OO style)
    df2 = df.smc.add_all()
    df2 = df.smc.fvg(min_gap_size=0.5)
    df2 = df.smc.swings(window_size=3)
    df2 = df.smc.structure(window_size=2)
    df2 = df.smc.order_blocks(lookback=5)

The extension is *optional* — importing ``pyvsmc`` does not require
``polars`` to be installed.  The namespace is only registered when
``polars`` is available; otherwise helper functions raise ``ImportError``
with an actionable message.

All underlying computations are fully vectorized (NumPy) and then
appended as Polars Series, so the performance characteristics are identical
to the NumPy API.
"""

from __future__ import annotations

from typing import Any

try:
    import polars as pl  # type: ignore[import-untyped]

    _POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore[assignment]
    _POLARS_AVAILABLE = False


def _require_polars() -> None:
    if not _POLARS_AVAILABLE:
        raise ImportError(
            "polars is required for the pyvsmc polars extension. "
            "Install it with `pip install polars` or `pip install pyvsmc[dev]`."
        )


# ---------------------------------------------------------------------------
# Functional helpers — usable without the namespace
# ---------------------------------------------------------------------------


def add_smc_columns(
    df: Any,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    open_col: str = "open",
    window_size: int = 2,
    tie: str = "all",
    fvg_min_gap_size: float | None = None,
    fvg_min_gap_size_pct: float | None = None,
    fvg_mitigation: bool = False,
    ob_lookback: int = 10,
    ob_use_fvg: bool = True,
    ob_use_bos: bool = True,
    ob_zone_mode: str = "full",
    ob_tie: str | None = None,
    structure_break_mode: str = "close",
    include_fvg: bool = True,
    include_swings: bool = True,
    include_structure: bool = True,
    include_order_blocks: bool = True,
    include_liquidity: bool = True,
    include_zones: bool = True,
    equal_threshold: float = 0.001,
    sweep_lookback: int = 20,
    eq_threshold: float = 0.02,
) -> Any:
    """Add all SMC indicator columns to a Polars DataFrame.

    This is the primary functional entry point.  It composes the individual
    indicator helpers (FVG, swings, structure, order blocks) and appends
    their columns in a single pipeline.

    Args:
        df: ``polars.DataFrame`` with OHLCV columns.
        high_col, low_col, close_col, open_col: Column names.
        window_size: Swing/structure window size ``N``.
        fvg_min_gap_size: FVG absolute gap filter.
        fvg_min_gap_size_pct: FVG percentage gap filter (fraction).
        fvg_mitigation: Whether to compute FVG mitigation state.
        ob_lookback: Order block lookback window.
        ob_use_fvg, ob_use_bos: OB impulse sources.
        include_fvg, include_swings, include_structure, include_order_blocks:
            Toggle individual indicator groups.

    Returns:
        New ``polars.DataFrame`` with SMC columns appended.  The input
        DataFrame is not mutated.

    Raises:
        ImportError: If ``polars`` is not installed.
        TypeError: If ``df`` is not a ``polars.DataFrame``.
    """
    _require_polars()

    # Import here to avoid circular imports at module load time and to keep
    # polars optional at import time for the core package.
    from .fvg import fvg_polars  # noqa: WPS433
    from .liquidity import liquidity_polars  # noqa: WPS433
    from .order_blocks import order_blocks_polars  # noqa: WPS433
    from .structure import structure_polars  # noqa: WPS433
    from .swings import swings_polars  # noqa: WPS433
    from .zones import zones_polars  # noqa: WPS433

    if isinstance(df, pl.LazyFrame):  # type: ignore[union-attr]
        # LazyFrame path: use lazy-capable helpers where possible, else collect
        # For now collect for complex helpers, keep FVG/Swings lazy
        # Simple: collect, apply eager, return lazy
        df_eager = df.collect()
        out_eager = add_smc_columns(df_eager, high_col=high_col, low_col=low_col, close_col=close_col, open_col=open_col, window_size=window_size, tie=tie, fvg_min_gap_size=fvg_min_gap_size, fvg_min_gap_size_pct=fvg_min_gap_size_pct, fvg_mitigation=fvg_mitigation, ob_lookback=ob_lookback, ob_use_fvg=ob_use_fvg, ob_use_bos=ob_use_bos, ob_zone_mode=ob_zone_mode, ob_tie=ob_tie, structure_break_mode=structure_break_mode, include_fvg=include_fvg, include_swings=include_swings, include_structure=include_structure, include_order_blocks=include_order_blocks, include_liquidity=include_liquidity, include_zones=include_zones, equal_threshold=equal_threshold, sweep_lookback=sweep_lookback, eq_threshold=eq_threshold)
        return out_eager.lazy()

    if not isinstance(df, pl.DataFrame):  # type: ignore[union-attr]
        raise TypeError(f"Expected polars.DataFrame, got {type(df)}")

    # Default ob_tie to tie if not specified (preserve 0.3.x default "all")
    if ob_tie is None:
        ob_tie = tie

    out = df

    if include_fvg:
        out = fvg_polars(
            out,
            high_col=high_col,
            low_col=low_col,
            min_gap_size=fvg_min_gap_size,
            min_gap_size_pct=fvg_min_gap_size_pct,
            compute_mitigation=fvg_mitigation,
        )

    if include_swings:
        out = swings_polars(out, high_col=high_col, low_col=low_col, window_size=window_size, tie=tie)

    if include_structure:
        out = structure_polars(
            out,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            window_size=window_size,
            break_mode=structure_break_mode,
            tie=tie,
        )

    if include_order_blocks:
        out = order_blocks_polars(
            out,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            lookback=ob_lookback,
            min_fvg_size=fvg_min_gap_size,
            min_fvg_size_pct=fvg_min_gap_size_pct,
            use_fvg=ob_use_fvg,
            use_bos=ob_use_bos,
            window_size=window_size,
            compute_mitigation=False,
            zone_mode=ob_zone_mode,
            tie=ob_tie,
        )

    if include_liquidity:
        out = liquidity_polars(out, high_col=high_col, low_col=low_col, close_col=close_col, equal_threshold=equal_threshold, sweep_lookback=sweep_lookback, window_size=window_size, tie=tie)

    if include_zones:
        out = zones_polars(out, high_col=high_col, low_col=low_col, close_col=close_col, window_size=window_size, eq_threshold=eq_threshold, tie=tie)

    return out


# ---------------------------------------------------------------------------
# ``.smc`` namespace — registered on ``pl.DataFrame`` via ``pl.api.register_dataframe_namespace``
# ---------------------------------------------------------------------------

if _POLARS_AVAILABLE:
    try:
        import polars.api as _pl_api  # type: ignore[import-untyped]

        @_pl_api.register_dataframe_namespace("smc")  # type: ignore[misc]
        class SMCNamespace:
            """Polars DataFrame ``.smc`` namespace.

            Access via ``df.smc.<method>()``.

            Attributes:
                _df: The underlying DataFrame.
            """

            def __init__(self, df: Any) -> None:
                self._df = df

            # -- individual indicators -------------------------------------------------

            def fvg(
                self,
                *,
                high_col: str = "high",
                low_col: str = "low",
                min_gap_size: float | None = None,
                min_gap_size_pct: float | None = None,
                compute_mitigation: bool = False,
            ) -> Any:
                """Append FVG columns.

                Args:
                    high_col: High column name.
                    low_col: Low column name.
                    min_gap_size: Absolute gap filter.
                    min_gap_size_pct: Percentage gap filter (fraction).
                    compute_mitigation: Whether to compute mitigation.

                Returns:
                    DataFrame with FVG columns.
                """
                from .fvg import fvg_polars  # noqa: WPS433

                return fvg_polars(
                    self._df,
                    high_col=high_col,
                    low_col=low_col,
                    min_gap_size=min_gap_size,
                    min_gap_size_pct=min_gap_size_pct,
                    compute_mitigation=compute_mitigation,
                )

            def swings(
                self,
                *,
                high_col: str = "high",
                low_col: str = "low",
                window_size: int = 2,
                tie: str = "all",
            ) -> Any:
                """Append swing columns.

                Args:
                    high_col: High column name.
                    low_col: Low column name.
                    window_size: Window size ``N``.
                    tie: Plateau handling.

                Returns:
                    DataFrame with swing columns.
                """
                from .swings import swings_polars  # noqa: WPS433

                return swings_polars(self._df, high_col=high_col, low_col=low_col, window_size=window_size, tie=tie)

            def structure(
                self,
                *,
                high_col: str = "high",
                low_col: str = "low",
                close_col: str = "close",
                window_size: int = 2,
                break_mode: str = "close",
                tie: str = "all",
            ) -> Any:
                """Append BOS/CHOCH structure columns.

                Args:
                    high_col: High column name.
                    low_col: Low column name.
                    close_col: Close column name.
                    window_size: Swing window size.
                    break_mode: How to define break.
                    tie: Swing tie handling.

                Returns:
                    DataFrame with structure columns.
                """
                from .structure import structure_polars  # noqa: WPS433

                return structure_polars(
                    self._df,
                    high_col=high_col,
                    low_col=low_col,
                    close_col=close_col,
                    window_size=window_size,
                    break_mode=break_mode,  # type: ignore[arg-type]
                    tie=tie,
                )

            def order_blocks(
                self,
                *,
                open_col: str = "open",
                high_col: str = "high",
                low_col: str = "low",
                close_col: str = "close",
                lookback: int = 10,
                min_fvg_size: float | None = None,
                min_fvg_size_pct: float | None = None,
                use_fvg: bool = True,
                use_bos: bool = True,
                window_size: int = 2,
            ) -> Any:
                """Append Order Block columns.

                Args:
                    open_col, high_col, low_col, close_col: Column names.
                    lookback: Lookback for opposing candle.
                    min_fvg_size: Minimum FVG size filter.
                    min_fvg_size_pct: Minimum FVG pct filter.
                    use_fvg, use_bos: Impulse sources.
                    window_size: Swing window for BOS.

                Returns:
                    DataFrame with OB columns.
                """
                from .order_blocks import order_blocks_polars  # noqa: WPS433

                return order_blocks_polars(
                    self._df,
                    open_col=open_col,
                    high_col=high_col,
                    low_col=low_col,
                    close_col=close_col,
                    lookback=lookback,
                    min_fvg_size=min_fvg_size,
                    min_fvg_size_pct=min_fvg_size_pct,
                    use_fvg=use_fvg,
                    use_bos=use_bos,
                    window_size=window_size,
                )

            def liquidity(self, *, high_col: str = "high", low_col: str = "low", close_col: str = "close", equal_threshold: float = 0.001, sweep_lookback: int = 20) -> Any:
                from .liquidity import liquidity_polars  # noqa: WPS433

                return liquidity_polars(self._df, high_col=high_col, low_col=low_col, close_col=close_col, equal_threshold=equal_threshold, sweep_lookback=sweep_lookback)

            def zones(self, *, high_col: str = "high", low_col: str = "low", close_col: str = "close", window_size: int = 2, eq_threshold: float = 0.02) -> Any:
                from .zones import zones_polars  # noqa: WPS433

                return zones_polars(self._df, high_col=high_col, low_col=low_col, close_col=close_col, window_size=window_size, eq_threshold=eq_threshold)

            def add_all(
                self,
                *,
                high_col: str = "high",
                low_col: str = "low",
                close_col: str = "close",
                open_col: str = "open",
                window_size: int = 2,
                fvg_min_gap_size: float | None = None,
                fvg_min_gap_size_pct: float | None = None,
                fvg_mitigation: bool = False,
                ob_lookback: int = 10,
                ob_use_fvg: bool = True,
                ob_use_bos: bool = True,
                include_fvg: bool = True,
                include_swings: bool = True,
                include_structure: bool = True,
                include_order_blocks: bool = True,
                include_liquidity: bool = True,
                include_zones: bool = True,
                equal_threshold: float = 0.001,
                sweep_lookback: int = 20,
                eq_threshold: float = 0.02,
            ) -> Any:
                """Append all SMC columns (composes all indicators).

                Args:
                    high_col, low_col, close_col, open_col: Column names.
                    window_size: Swing window size.
                    fvg_min_gap_size: FVG absolute filter.
                    fvg_min_gap_size_pct: FVG pct filter.
                    fvg_mitigation: Compute FVG mitigation.
                    ob_lookback: OB lookback.
                    ob_use_fvg, ob_use_bos: OB impulse sources.
                    include_fvg, include_swings, include_structure, include_order_blocks:
                        Toggles.

                Returns:
                    DataFrame with all SMC columns.
                """
                return add_smc_columns(
                    self._df,
                    high_col=high_col,
                    low_col=low_col,
                    close_col=close_col,
                    open_col=open_col,
                    window_size=window_size,
                    fvg_min_gap_size=fvg_min_gap_size,
                    fvg_min_gap_size_pct=fvg_min_gap_size_pct,
                    fvg_mitigation=fvg_mitigation,
                    ob_lookback=ob_lookback,
                    ob_use_fvg=ob_use_fvg,
                    ob_use_bos=ob_use_bos,
                    include_fvg=include_fvg,
                    include_swings=include_swings,
                    include_structure=include_structure,
                    include_order_blocks=include_order_blocks,
                    include_liquidity=include_liquidity,
                    include_zones=include_zones,
                    equal_threshold=equal_threshold,
                    sweep_lookback=sweep_lookback,
                    eq_threshold=eq_threshold,
                )

    except Exception:
        # If registration fails (e.g. older polars without api), silently skip.
        pass


__all__ = ["add_smc_columns"]
