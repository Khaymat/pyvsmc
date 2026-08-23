"""Tests for pysmc.order_blocks."""

from __future__ import annotations

import numpy as np
import pytest

from pysmc.order_blocks import detect_order_blocks


class TestOrderBlocksBasic:
    def test_bullish_ob_found(self) -> None:
        # Bearish candle at 1, then bullish FVG at 3 -> OB at 1
        # FVG bullish at 3 requires Low[3] > High[1]
        open_ = np.array([10.0, 10.5, 10.0, 10.2, 10.8])
        high = np.array([10.2, 10.6, 10.3, 10.4, 12.0])
        low = np.array([9.8, 10.0, 9.9, 10.8, 11.5])  # Low[3]=10.8 > High[1]=10.6 => FVG at 3
        close = np.array([10.1, 10.1, 10.1, 11.5, 11.8])  # candle 1 is bearish (10.1 <10.5)
        # FVG bullish at 3, last bearish before it is candle 1
        res = detect_order_blocks(open_, high, low, close, lookback=5, use_fvg=True, use_bos=False)
        # Should have bullish OB at index 1
        assert res.bullish_ob[1] == True  # noqa: E712
        assert res.ob_high[1] == pytest.approx(high[1])
        assert res.validated_index[1] == 3
        assert res.impulse_type[1] == "fvg"

    def test_bearish_ob_found(self) -> None:
        # Bullish candle at 1, then bearish FVG at 3 — ensure candle 2 is NOT bullish so OB is at 1
        open_ = np.array([10.0, 9.5, 10.2, 9.8, 9.0])
        high = np.array([10.2, 10.0, 10.3, 9.0, 9.2])  # High[3]=9.0 < Low[1]=9.5 => FVG bear at 3
        low = np.array([9.8, 9.5, 9.9, 8.5, 8.8])
        close = np.array([9.9, 9.9, 10.1, 8.8, 8.9])  # candle 1 bullish (9.9>9.5), candle 2 bearish (10.1<10.2)
        res = detect_order_blocks(open_, high, low, close, lookback=5, use_fvg=True, use_bos=False)
        assert res.bearish_ob[1] == True  # noqa: E712

    def test_no_impulse_no_ob(self) -> None:
        # No FVG, no BOS -> no OB
        open_ = np.full(10, 10.0)
        high = np.full(10, 10.2)
        low = np.full(10, 9.8)
        close = np.full(10, 10.0)
        res = detect_order_blocks(open_, high, low, close, use_fvg=True, use_bos=True)
        assert not res.bullish_ob.any()
        assert not res.bearish_ob.any()

    def test_lookback_limits_search(self) -> None:
        # Bearish candle far away should not be found if lookback is small
        open_ = np.array([10.5, 10.0, 10.0, 10.0, 10.0])
        high = np.array([10.6, 10.2, 10.3, 10.3, 10.4])
        low = np.array([10.0, 9.9, 9.9, 10.8, 11.0])  # FVG at 3? Low3=10.8 > High1=10.2 -> yes at 3? Actually need High[1]=10.2, Low[3]=10.8
        close = np.array([10.1, 10.0, 10.0, 11.0, 11.2])  # candle 0 is bearish, but distance 3
        # With lookback=1, impulse at 3 can only look at index 2 (which is not bearish)
        res_small = detect_order_blocks(open_, high, low, close, lookback=1, use_fvg=True, use_bos=False)
        assert not res_small.bullish_ob[0]  # too far
        # With lookback=5, it should find it
        res_large = detect_order_blocks(open_, high, low, close, lookback=5, use_fvg=True, use_bos=False)
        # The last bearish before 3 is at 0 (close<open)
        # Check that some bullish OB is found
        assert res_large.bullish_ob.any()


class TestOrderBlocksEdge:
    def test_empty(self) -> None:
        res = detect_order_blocks(np.array([]), np.array([]), np.array([]), np.array([]))
        assert res.bullish_ob.size == 0

    def test_too_short(self) -> None:
        res = detect_order_blocks(np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]))
        assert not res.bullish_ob.any()

    def test_lookback_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="lookback"):
            detect_order_blocks(np.ones(5), np.ones(5), np.ones(5), np.ones(5), lookback=0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            detect_order_blocks(np.ones(3), np.ones(3), np.ones(3), np.ones(2))

    def test_nan_candle_not_considered_ob(self) -> None:
        open_ = np.array([10.0, np.nan, 10.0, 10.0, 10.0])
        high = np.array([10.2, 10.2, 10.2, 10.4, 12.0])
        low = np.array([9.8, 9.8, 9.8, 10.8, 11.0])
        close = np.array([9.9, 9.9, 9.9, 11.0, 11.2])
        res = detect_order_blocks(open_, high, low, close, use_fvg=True, use_bos=False)
        # Candle 1 has NaN open -> not valid OB
        if res.bullish_ob.any():
            assert not res.bullish_ob[1]

    def test_polars_basic(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from pysmc.order_blocks import order_blocks_polars

        df = pl.DataFrame(
            {
                "open": [10.0, 10.5, 10.0, 10.2, 10.8],
                "high": [10.2, 10.6, 10.3, 10.4, 12.0],
                "low": [9.8, 10.0, 9.9, 10.8, 11.5],
                "close": [10.1, 10.1, 10.1, 11.5, 11.8],
            }
        )
        out = order_blocks_polars(df)
        assert "ob_bullish" in out.columns
        assert "ob_bearish" in out.columns
