"""Tests for pyvsmc.polars_ext namespace."""

from __future__ import annotations

import numpy as np
import pytest


class TestPolarsExt:
    def test_add_smc_columns(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from pyvsmc.polars_ext import add_smc_columns

        df = pl.DataFrame(
            {
                "open": [10.0, 10.5, 10.0, 10.2, 10.8, 11.0],
                "high": [10.2, 10.6, 10.3, 10.4, 12.0, 12.5],
                "low": [9.8, 10.0, 9.9, 10.8, 11.5, 11.8],
                "close": [10.1, 10.1, 10.1, 11.5, 11.8, 12.0],
                "volume": [100, 120, 80, 200, 150, 180],
            }
        )
        out = add_smc_columns(df)
        # Check all indicator groups present
        assert "fvg_bullish" in out.columns
        assert "swing_high" in out.columns
        assert "bos_bullish" in out.columns
        assert "ob_bullish" in out.columns
        assert out.shape[0] == df.shape[0]

    def test_namespace_smc(self) -> None:
        pytest.importorskip("polars")
        import polars as pl
        import pyvsmc  # noqa: F401  triggers namespace registration

        df = pl.DataFrame(
            {
                "open": [10.0, 10.5, 10.0, 10.2, 10.8],
                "high": [10.2, 10.6, 10.3, 10.4, 12.0],
                "low": [9.8, 10.0, 9.9, 10.8, 11.5],
                "close": [10.1, 10.1, 10.1, 11.5, 11.8],
            }
        )
        out = df.smc.fvg()
        assert "fvg_bullish" in out.columns
        out2 = df.smc.swings(window_size=1)
        assert "swing_high" in out2.columns
        out3 = df.smc.structure(window_size=1)
        assert "bos_bullish" in out3.columns
        out4 = df.smc.order_blocks(lookback=3)
        assert "ob_bullish" in out4.columns
        out5 = df.smc.add_all()
        assert "fvg_bullish" in out5.columns
        assert "swing_high" in out5.columns

    def test_add_smc_columns_toggles(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from pyvsmc.polars_ext import add_smc_columns

        df = pl.DataFrame(
            {
                "open": [10.0, 10.0, 10.0, 10.0, 10.0],
                "high": [10.2, 10.2, 10.2, 10.2, 10.2],
                "low": [9.8, 9.8, 9.8, 9.8, 9.8],
                "close": [10.0, 10.0, 10.0, 10.0, 10.0],
            }
        )
        out = add_smc_columns(df, include_order_blocks=False, include_structure=False)
        assert "fvg_bullish" in out.columns
        assert "ob_bullish" not in out.columns
        assert "bos_bullish" not in out.columns

    def test_custom_column_names(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from pyvsmc.polars_ext import add_smc_columns

        df = pl.DataFrame(
            {
                "o": [10.0, 10.5, 10.0, 10.2, 10.8],
                "h": [10.2, 10.6, 10.3, 10.4, 12.0],
                "l": [9.8, 10.0, 9.9, 10.8, 11.5],
                "c": [10.1, 10.1, 10.1, 11.5, 11.8],
            }
        )
        out = add_smc_columns(df, open_col="o", high_col="h", low_col="l", close_col="c")
        assert "fvg_bullish" in out.columns

    def test_not_dataframe_raises(self) -> None:
        pytest.importorskip("polars")
        from pyvsmc.polars_ext import add_smc_columns

        with pytest.raises(TypeError, match="polars.DataFrame"):
            add_smc_columns([1, 2, 3])  # type: ignore[arg-type]
