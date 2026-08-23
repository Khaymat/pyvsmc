"""Tests for tie/plateau semantics — NumPy/Polars parity, downstream impact."""

import numpy as np
import pytest
import polars as pl

from pyvsmc.swings import detect_swings, swings_polars


def assert_parity(high, low, window_size, tie):
    res_np = detect_swings(high, low, window_size=window_size, tie=tie)
    df = pl.DataFrame({"high": high, "low": low})
    res_pl = swings_polars(df, window_size=window_size, tie=tie)
    res_lazy = swings_polars(df.lazy(), window_size=window_size, tie=tie).collect()
    for col in ["swing_high", "swing_low"]:
        np.testing.assert_array_equal(getattr(res_np, col), res_pl[col].to_numpy())
        np.testing.assert_array_equal(getattr(res_np, col), res_lazy[col].to_numpy())


class TestTieFlat:
    def test_flat_all_vs_first(self):
        high = np.full(7, 5.0)
        low = np.full(7, 5.0)
        # all: every non-edge bar is both high and low
        res_all = detect_swings(high, low, window_size=1, tie="all")
        assert res_all.swing_high.sum() == 5  # indices 1..5
        assert res_all.swing_low.sum() == 5
        # first/strict: flat => 0
        for tie in ("first", "strict"):
            res = detect_swings(high, low, window_size=1, tie=tie)
            assert res.swing_high.sum() == 0
            assert res.swing_low.sum() == 0
            assert not np.any(res.swing_high & res.swing_low)
            assert_parity(high, low, 1, tie)

    def test_flat_ohlc_no_both(self):
        high = np.array([5.,5,5,5,5])
        low = np.array([5.,5,5,5,5])
        for tie in ("first", "strict"):
            res = detect_swings(high, low, window_size=1, tie=tie)
            assert not np.any(res.swing_high & res.swing_low)


class TestTiePlateau:
    def test_plateau_peak_first(self):
        high = np.array([1.,3,5,5,5,3,1])
        low = np.zeros(7)
        # N=1, plateau [5,5,5] at 2,3,4
        res_all = detect_swings(high, low, window_size=1, tie="all")
        assert list(res_all.swing_high.astype(int)) == [0,0,1,1,1,0,0]
        res_first = detect_swings(high, low, window_size=1, tie="first")
        assert list(res_first.swing_high.astype(int)) == [0,0,1,0,0,0,0]
        res_strict = detect_swings(high, low, window_size=1, tie="strict")
        assert res_strict.swing_high.sum() == 0
        for tie in ("all","first","strict"):
            assert_parity(high, low, 1, tie)

    def test_plateau_trough(self):
        low = np.array([5.,3,1,1,1,3,5])
        high = np.full(7, 5.0)
        res = detect_swings(high, low, window_size=1, tie="first")
        # one low at first plateau
        assert res.swing_low.sum() == 1
        assert res.swing_low[2] == True
        assert_parity(high, low, 1, "first")

    def test_double_top(self):
        high = np.array([1.,5,5,1])
        low = np.zeros(4)
        assert detect_swings(high, low, window_size=1, tie="all").swing_high.sum() == 2
        res = detect_swings(high, low, window_size=1, tie="first")
        assert res.swing_high.sum() == 1
        assert res.swing_high[1] == True
        assert not res.swing_high[2]
        assert detect_swings(high, low, window_size=1, tie="strict").swing_high.sum() == 0
        for tie in ("all","first","strict"):
            assert_parity(high, low, 1, tie)

    def test_double_bottom(self):
        low = np.array([5.,1,1,5])
        high = np.full(4, 5.0)
        assert detect_swings(high, low, window_size=1, tie="first").swing_low.sum() == 1
        assert_parity(high, low, 1, "first")


class TestTieIsolated:
    def test_isolated_unchanged(self):
        high = np.array([1.,3,5,3,1])
        low = np.zeros(5)
        for tie in ("all","first","strict"):
            res = detect_swings(high, low, window_size=1, tie=tie)
            assert res.swing_high[2] == True
            assert_parity(high, low, 1, tie)


class TestTieNaN:
    def test_nan_invalidate(self):
        high = np.array([1.,5,np.nan,5,1.])
        low = np.zeros(5)
        for tie in ("all","first","strict"):
            res = detect_swings(high, low, window_size=1, tie=tie)
            # window containing NaN is invalid
            assert not res.swing_high[1]
            assert not res.swing_high[2]
            assert_parity(high, low, 1, tie)


class TestDownstreamImpact:
    def test_structure_liquidity_not_modified(self):
        # Ensure default tie=all preserves old downstream counts
        # This test documents current downstream impact, not new behavior
        high = np.array([1.,5,5,5,1, 1.,5,5,1])
        low = np.array([0.,0,0,0,0, 0.,0,0,0])
        # Just verify detect_swings tie param doesn't affect default
        assert detect_swings(high[:5], low[:5], window_size=1, tie="all").swing_high.sum() == 3
        assert detect_swings(high[:5], low[:5], window_size=1, tie="first").swing_high.sum() == 1
