"""Tests for pysmc.swings — fractal swing highs/lows."""

from __future__ import annotations

import numpy as np
import pytest

from pysmc.swings import detect_swings


class TestSwingsBasic:
    def test_swing_high_centered_peak(self) -> None:
        # Window=1: peak at index 2
        high = np.array([1.0, 3.0, 5.0, 3.0, 1.0])
        low = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        res = detect_swings(high, low, window_size=1)
        assert res.swing_high[2] == True  # noqa: E712
        assert res.swing_high_price[2] == pytest.approx(5.0)
        # neighbors are not swings
        assert not res.swing_high[1]
        assert not res.swing_high[3]

    def test_swing_low_centered_trough(self) -> None:
        high = np.array([5.0, 4.0, 3.0, 4.0, 5.0])
        low = np.array([4.0, 3.0, 0.0, 3.0, 4.0])
        res = detect_swings(high, low, window_size=1)
        assert res.swing_low[2] == True  # noqa: E712
        assert res.swing_low_price[2] == pytest.approx(0.0)

    def test_window_2(self) -> None:
        high = np.array([1.0, 2.0, 3.0, 5.0, 3.0, 2.0, 1.0])
        low = np.array([0.0, 0.5, 0.3, 1.0, 0.4, 0.2, 0.0])
        res = detect_swings(high, low, window_size=2)
        # Index 3 (5.0) is max of [1,2,3,5,3] -> swing high
        assert res.swing_high[3] == True  # noqa: E712
        # Edges cannot be swings
        assert not res.swing_high[0]
        assert not res.swing_high[1]
        assert not res.swing_high[5]
        assert not res.swing_high[6]

    def test_multiple_swings(self) -> None:
        high = np.array([5.0, 1.0, 5.0, 1.0, 5.0])
        low = np.array([4.0, 0.0, 4.0, 0.0, 4.0])
        res = detect_swings(high, low, window_size=1)
        # Swing highs at 0? No, edge excluded. So at 2 only?
        # Actually [5,1,5] at i=2: max of [1,5,1] =5 -> true
        # Check that at least index 2 is swing high
        assert res.swing_high[2] == True  # noqa: E712
        assert res.swing_low[1] == True  # noqa: E712
        assert res.swing_low[3] == True  # noqa: E712

    def test_plateau_equal_max_not_swing_unless_center_equals(self) -> None:
        # Plateau: two equal peaks
        high = np.array([1.0, 5.0, 5.0, 1.0])
        low = np.zeros(4)
        res = detect_swings(high, low, window_size=1)
        # At i=1: window [1,5,5] max=5, center=5 -> is swing (equal)
        # At i=2: window [5,5,1] max=5, center=5 -> also swing
        # This is the defined behavior: High[i] == max(window)
        assert res.swing_high[1] == True  # noqa: E712
        assert res.swing_high[2] == True  # noqa: E712


class TestSwingsEdgeCases:
    def test_empty(self) -> None:
        res = detect_swings(np.array([]), np.array([]), window_size=2)
        assert res.swing_high.size == 0

    def test_too_short(self) -> None:
        # n=3, window=2 -> need 5 -> no swings possible
        high = np.array([1.0, 3.0, 2.0])
        low = np.array([0.0, 1.0, 0.5])
        res = detect_swings(high, low, window_size=2)
        assert not res.swing_high.any()
        assert not res.swing_low.any()

    def test_exact_minimum_length(self) -> None:
        # n=5, window=2 -> width 5 -> exactly one window, center at 2
        high = np.array([1.0, 2.0, 10.0, 2.0, 1.0])
        low = np.array([0.0, 0.5, 0.2, 0.5, 0.0])
        res = detect_swings(high, low, window_size=2)
        assert res.swing_high[2] == True  # noqa: E712

    def test_flat_prices_all_equal(self) -> None:
        high = np.full(10, 5.0)
        low = np.full(10, 3.0)
        res = detect_swings(high, low, window_size=1)
        # All windows have max==center, so all non-edge bars are swings
        # This is correct per definition; check at least middle is True
        assert res.swing_high[5] == True  # noqa: E712
        assert res.swing_low[5] == True  # noqa: E712

    def test_nan_invalidates_window(self) -> None:
        high = np.array([1.0, 5.0, np.nan, 5.0, 1.0])
        low = np.array([0.0, 0.5, 0.3, 0.5, 0.0])
        res = detect_swings(high, low, window_size=1)
        # Windows containing NaN should not produce swings
        # Index 2 has NaN, windows [1,2,3] includes NaN
        assert not res.swing_high[1]
        assert not res.swing_high[2]
        assert not res.swing_high[3]

    def test_nan_at_center_not_swing(self) -> None:
        high = np.array([1.0, 3.0, np.nan, 3.0, 1.0])
        low = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        res = detect_swings(high, low, window_size=1)
        assert not res.swing_high[2]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            detect_swings(np.array([1.0, 2.0]), np.array([1.0]))

    def test_window_size_less_than_1_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            detect_swings(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), window_size=0)

    def test_window_size_1_minimal(self) -> None:
        high = np.array([2.0, 1.0, 3.0])
        low = np.array([1.0, 0.0, 2.0])
        res = detect_swings(high, low, window_size=1)
        # Only center index 1 can be swing; window [2,1,3] max=3 !=1 -> not high swing
        # low window [1,0,2] min=0 == center -> low swing true
        assert not res.swing_high[1]
        assert res.swing_low[1] == True  # noqa: E712


class TestSwingsPriceLevels:
    def test_price_levels_nan_where_no_swing(self) -> None:
        high = np.array([1.0, 5.0, 1.0])
        low = np.array([0.0, 0.2, 0.0])
        res = detect_swings(high, low, window_size=1)
        assert np.isnan(res.swing_high_price[0])
        assert res.swing_high_price[1] == pytest.approx(5.0)
        assert np.isnan(res.swing_high_price[2])

    def test_window_size_stored(self) -> None:
        res = detect_swings(np.array([1.0, 2.0, 3.0, 2.0, 1.0]), np.zeros(5), window_size=2)
        assert res.window_size == 2


class TestSwingsPolars:
    def test_polars_basic(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from pysmc.swings import swings_polars

        df = pl.DataFrame({"high": [1.0, 5.0, 1.0, 5.0, 1.0], "low": [0.0, 0.2, 0.0, 0.2, 0.0]})
        out = swings_polars(df, window_size=1)
        assert "swing_high" in out.columns
        assert "swing_low" in out.columns


class TestSwingsLarge:
    def test_large_input_vectorized(self) -> None:
        n = 200_000
        rng = np.random.default_rng(0)
        high = rng.uniform(90, 110, size=n)
        low = high - rng.uniform(0, 5, size=n)
        import time

        t0 = time.perf_counter()
        res = detect_swings(high, low, window_size=3)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.5, f"Swings took {elapsed:.2f}s"
        assert res.swing_high.shape[0] == n
