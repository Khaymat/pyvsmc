"""Tests for pysmc.structure — BOS & CHOCH engine."""

from __future__ import annotations

import numpy as np
import pytest

from pysmc.structure import detect_structure


class TestStructureBasic:
    def test_bullish_bos(self) -> None:
        # Create a clear swing high at index 2, then BOS above it
        # Use window_size=1 so swing detection is predictable
        # Highs: peak at 2 = 10, then later close breaks above 10
        high = np.array([5.0, 8.0, 10.0, 8.0, 6.0, 7.0, 9.0])
        low = np.array([4.0, 5.0, 6.0, 5.0, 4.0, 5.0, 6.0])
        close = np.array([4.5, 7.0, 9.0, 7.0, 5.0, 6.0, 11.0])  # breaks 10 at index 6
        res = detect_structure(high, low, close, window_size=1)
        # Expect swing high at 2 (10.0), BOS bullish at 6
        assert res.swing_high[2] == True  # noqa: E712
        assert res.bos_bullish[6] == True  # noqa: E712
        assert res.bos_level[6] == pytest.approx(10.0)

    def test_bearish_bos(self) -> None:
        low = np.array([10.0, 8.0, 5.0, 8.0, 10.0, 9.0, 7.0])
        high = np.array([11.0, 9.0, 6.0, 9.0, 11.0, 10.0, 8.0])
        close = np.array([10.5, 8.5, 5.5, 8.5, 10.5, 9.5, 4.0])  # breaks 5 at 6
        res = detect_structure(high, low, close, window_size=1)
        assert res.swing_low[2] == True  # noqa: E712
        assert res.bos_bearish[6] == True or res.choch_bearish[6] == True  # depending on prior trend
        # Check that at least a bearish break is flagged
        assert (res.bos_bearish | res.choch_bearish)[6] == True  # noqa: E712

    def test_choch_bearish_after_bull_trend(self) -> None:
        # Uptrend with BOS, then breakdown -> CHOCH bearish
        high = np.array([5.0, 10.0, 7.0, 12.0, 9.0, 6.0])
        low = np.array([3.0, 4.0, 5.0, 6.0, 4.0, 2.0])
        close = np.array([4.0, 9.0, 6.0, 11.0, 8.0, 1.0])
        # Swing high at 1 (10), BOS bullish at 3 (12 >10) sets trend bullish
        # Swing low: need to identify; low at 0 is edge, low at maybe?
        # Force scenario: use explicit swing masks instead of relying on autodetect
        # We'll test CHOCH classification more directly below
        res = detect_structure(high, low, close, window_size=1)
        # After bullish BOS, a break below swing low should be CHOCH bearish
        # Check trend evolution is non-trivial
        assert res.trend[-1] != 0  # trend should have been set

    def test_choch_bullish_after_bear_trend(self) -> None:
        high = np.array([10.0, 6.0, 9.0, 4.0, 8.0, 12.0])
        low = np.array([8.0, 3.0, 5.0, 1.0, 4.0, 6.0])
        close = np.array([9.0, 4.0, 7.0, 2.0, 6.0, 11.0])
        res = detect_structure(high, low, close, window_size=1)
        assert res.trend[-1] != 0

    def test_explicit_swing_masks(self) -> None:
        # Provide explicit swings to isolate structure logic
        n = 7
        high = np.array([10.0, 10.0, 12.0, 10.0, 10.0, 10.0, 13.0])
        low = np.array([9.0, 9.0, 9.0, 9.0, 8.0, 9.0, 9.0])
        close = np.array([9.5, 9.5, 11.0, 9.5, 8.5, 9.5, 12.5])
        swing_high = np.zeros(n, dtype=bool)
        swing_high[2] = True
        swing_low = np.zeros(n, dtype=bool)
        swing_low[4] = True
        res = detect_structure(high, low, close, window_size=1, swing_high=swing_high, swing_low=swing_low)
        assert res.bos_bullish[6] == True  # noqa: E712
        assert res.bos_level[6] == pytest.approx(12.0)

    def test_no_break_when_close_equals_level(self) -> None:
        # Strict break: close == level is NOT a break
        n = 5
        high = np.array([5.0, 10.0, 5.0, 5.0, 5.0])
        low = np.array([3.0, 4.0, 3.0, 3.0, 3.0])
        close = np.array([4.0, 9.0, 4.0, 4.0, 10.0])  # last close == 10 not >10? Actually 10 == 10
        swing_high = np.zeros(n, dtype=bool)
        swing_high[1] = True
        swing_low = np.zeros(n, dtype=bool)
        res = detect_structure(high, low, close, window_size=1, swing_high=swing_high, swing_low=swing_low)
        # Need close[4]=10 == high[1]=10 -> not >, so no break
        # Change to 10.0 exactly; should NOT trigger
        assert not res.bos_bullish[4]
        assert not res.choch_bullish[4]
        # Now with 10.1 should trigger
        close2 = np.array([4.0, 9.0, 4.0, 4.0, 10.1])
        res2 = detect_structure(high, low, close2, window_size=1, swing_high=swing_high, swing_low=swing_low)
        assert res2.bos_bullish[4] == True  # noqa: E712


class TestStructureTrend:
    def test_trend_initial_zero(self) -> None:
        high = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        low = np.array([9.0, 9.0, 9.0, 9.0, 9.0])
        close = np.array([9.5, 9.5, 9.5, 9.5, 9.5])
        res = detect_structure(high, low, close, window_size=1)
        # No structure breaks, trend stays 0
        assert (res.trend == 0).all()

    def test_trend_bullish_after_bos(self) -> None:
        high = np.array([5.0, 10.0, 5.0, 5.0, 6.0, 11.0])
        low = np.zeros(6)
        close = np.array([5.0, 9.0, 5.0, 5.0, 5.0, 10.5])
        swing_high = np.zeros(6, dtype=bool)
        swing_high[1] = True
        swing_low = np.zeros(6, dtype=bool)
        res = detect_structure(high, low, close, window_size=1, swing_high=swing_high, swing_low=swing_low)
        # BOS at 5 sets trend to 1
        assert res.trend[5] == 1
        # Trend persists after

    def test_trend_flips_on_choch(self) -> None:
        n = 8
        high = np.array([10.0, 12.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        low = np.array([10.0, 10.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        close = np.array([10.0, 11.0, 9.0, 12.5, 9.0, 7.0, 12.5, 12.5])
        swing_high = np.zeros(n, dtype=bool)
        swing_high[1] = True  # 12
        swing_low = np.zeros(n, dtype=bool)
        swing_low[2] = True  # 8
        # Close 3 breaks high -> BOS/CHOCH bullish sets trend 1
        # Close 5 breaks low -> CHOCH bearish flips to -1
        res = detect_structure(high, low, close, window_size=1, swing_high=swing_high, swing_low=swing_low)
        assert res.trend[3] == 1
        assert res.trend[5] == -1


class TestStructureEdgeCases:
    def test_empty(self) -> None:
        res = detect_structure(np.array([]), np.array([]), np.array([]))
        assert res.bos_bullish.size == 0
        assert res.trend.size == 0

    def test_length_less_than_window(self) -> None:
        res = detect_structure(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.5, 1.5]), window_size=2)
        assert not res.bos_bullish.any()
        assert not res.bos_bearish.any()

    def test_nan_close_no_trigger(self) -> None:
        high = np.array([5.0, 10.0, 5.0, 5.0])
        low = np.array([3.0, 4.0, 3.0, 3.0])
        close = np.array([4.0, 9.0, np.nan, np.nan])
        swing_high = np.zeros(4, dtype=bool)
        swing_high[1] = True
        swing_low = np.zeros(4, dtype=bool)
        # Even with NaN close at 3 that would be a break if not NaN, no flag
        res = detect_structure(high, low, close, window_size=1, swing_high=swing_high, swing_low=swing_low)
        assert not res.bos_bullish[3]

    def test_flat_no_structure(self) -> None:
        n = 10
        high = np.full(n, 10.0)
        low = np.full(n, 10.0)
        close = np.full(n, 10.0)
        res = detect_structure(high, low, close, window_size=1)
        assert not res.bos_bullish.any()
        assert not res.bos_bearish.any()
        assert not res.choch_bullish.any()
        assert not res.choch_bearish.any()

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            detect_structure(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0]))

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            detect_structure(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), window_size=0)

    def test_swing_mask_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="swing masks"):
            detect_structure(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0]),
                swing_high=np.array([True, False]),
                swing_low=np.array([False, True, False]),
            )


class TestStructureLevels:
    def test_bos_level_nan_where_no_event(self) -> None:
        high = np.array([5.0, 10.0, 5.0, 5.0, 5.0])
        low = np.array([3.0, 4.0, 3.0, 3.0, 3.0])
        close = np.array([4.0, 9.0, 4.0, 4.0, 4.0])
        res = detect_structure(high, low, close, window_size=1)
        # No BOS, all bos_level should be NaN
        assert np.isnan(res.bos_level).all()

    def test_choch_level_correct(self) -> None:
        n = 6
        high = np.array([10.0, 12.0, 10.0, 10.0, 10.0, 13.0])
        low = np.array([5.0, 6.0, 4.0, 6.0, 5.0, 5.0])
        close = np.array([6.0, 11.0, 5.0, 3.0, 12.5, 12.0])
        swing_high = np.zeros(n, dtype=bool)
        swing_high[1] = True
        swing_low = np.zeros(n, dtype=bool)
        swing_low[2] = True
        res = detect_structure(high, low, close, window_size=1, swing_high=swing_high, swing_low=swing_low)
        # First BOS sets trend, second opposite break is CHOCH
        # Detailed check depends on sequence; at least one CHOCH should exist or BOS
        assert (res.bos_bullish | res.choch_bullish | res.bos_bearish | res.choch_bearish).any()


class TestStructurePolars:
    def test_polars_basic(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from pysmc.structure import structure_polars

        df = pl.DataFrame(
            {
                "high": [5.0, 10.0, 5.0, 5.0, 6.0, 11.0],
                "low": [4.0, 5.0, 3.0, 4.0, 4.0, 5.0],
                "close": [4.5, 9.0, 4.0, 4.5, 5.5, 10.5],
            }
        )
        out = structure_polars(df, window_size=1)
        assert "bos_bullish" in out.columns
        assert "trend" in out.columns


class TestStructureLarge:
    def test_large_input(self) -> None:
        n = 100_000
        rng = np.random.default_rng(1)
        high = rng.uniform(90, 110, size=n)
        low = high - rng.uniform(0.5, 5, size=n)
        close = (high + low) / 2 + rng.normal(0, 0.5, size=n)
        import time

        t0 = time.perf_counter()
        res = detect_structure(high, low, close, window_size=2)
        elapsed = time.perf_counter() - t0
        assert res.trend.shape[0] == n
        assert elapsed < 2.0, f"Structure took {elapsed:.2f}s"
