"""Tests for pysmc.fvg — Fair Value Gap detection."""

from __future__ import annotations

import numpy as np
import pytest

from pysmc.fvg import detect_fvg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_allclose_or_nan(a: np.ndarray, b: np.ndarray) -> None:
    """Compare arrays that may contain NaN — NaN locations must match."""
    assert a.shape == b.shape
    mask_a_nan = np.isnan(a)
    mask_b_nan = np.isnan(b)
    assert np.array_equal(mask_a_nan, mask_b_nan), f"NaN masks differ: {a} vs {b}"
    assert np.allclose(a[~mask_a_nan], b[~mask_b_nan], equal_nan=True)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestFVGBasic:
    def test_bullish_fvg_detected(self) -> None:
        high = np.array([10.0, 11.0, 10.5, 10.0, 12.0])
        low = np.array([9.0, 9.5, 9.0, 11.5, 11.4])
        # i=3: Low[3]=11.5 > High[1]=11.0 -> bullish FVG at 3
        res = detect_fvg(high, low)
        assert res.bullish[3] == True  # noqa: E712
        assert res.bearish[3] == False  # noqa: E712
        assert res.bullish_lower[3] == pytest.approx(11.0)
        assert res.bullish_upper[3] == pytest.approx(11.5)

    def test_bearish_fvg_detected(self) -> None:
        high = np.array([10.0, 12.0, 10.5, 8.0, 9.0])
        low = np.array([9.0, 10.0, 9.5, 7.0, 8.0])
        # i=3: High[3]=8.0 < Low[1]=10.0 -> bearish at 3
        res = detect_fvg(high, low)
        assert res.bearish[3] == True  # noqa: E712
        assert res.bullish[3] == False  # noqa: E712
        assert res.bearish_upper[3] == pytest.approx(10.0)
        assert res.bearish_lower[3] == pytest.approx(8.0)

    def test_no_fvg_monotonic_overlap(self) -> None:
        high = np.array([10.0, 10.5, 11.0, 11.5, 12.0])
        low = np.array([9.0, 9.5, 10.0, 10.5, 11.0])
        res = detect_fvg(high, low)
        assert not res.bullish.any()
        assert not res.bearish.any()

    def test_both_directions_in_same_series(self) -> None:
        # Craft series with both bullish and bearish FVGs
        high = np.array([10.0, 10.0, 10.0, 12.0, 10.0, 5.0])
        low = np.array([9.0, 9.0, 9.0, 11.0, 9.0, 4.0])
        # i=3: Low[3]=11.0 > High[1]=10.0 -> bullish at 3
        # i=5: High[5]=5.0 < Low[3]=11.0 -> bearish at 5
        res = detect_fvg(high, low)
        assert res.bullish[3]
        assert res.bearish[5]

    def test_gap_size_computed(self) -> None:
        high = np.array([10.0, 10.0, 10.0])
        low = np.array([9.0, 9.0, 12.0])
        res = detect_fvg(high, low)
        # bullish at 2: gap = 12 - 10 = 2
        assert res.bullish[2]
        assert res.gap_size[2] == pytest.approx(2.0)
        assert res.gap_size_pct[2] == pytest.approx(0.2)  # 2 / 10

    def test_bearish_gap_size(self) -> None:
        high = np.array([12.0, 15.0, 10.0])
        low = np.array([11.0, 14.0, 9.0])
        # Bearish at 2: High[2]=10 < Low[0]=11 -> gap = 11-10=1
        res = detect_fvg(high, low)
        assert res.bearish[2]
        assert res.gap_size[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestFVGFilters:
    def test_min_gap_size_filters_small_gaps(self) -> None:
        high = np.array([10.0, 10.0, 10.0])
        low = np.array([9.0, 9.0, 10.1])
        # gap = 0.1
        res_small = detect_fvg(high, low, min_gap_size=0.5)
        assert not res_small.bullish[2]
        res_no_filter = detect_fvg(high, low)
        assert res_no_filter.bullish[2]

    def test_min_gap_size_pct_filters(self) -> None:
        high = np.array([100.0, 100.0, 100.0])
        low = np.array([90.0, 90.0, 100.5])
        # gap = 0.5, pct = 0.005 (0.5%)
        res = detect_fvg(high, low, min_gap_size_pct=0.01)  # 1% threshold
        assert not res.bullish[2]
        res2 = detect_fvg(high, low, min_gap_size_pct=0.001)
        assert res2.bullish[2]

    def test_both_filters_must_pass(self) -> None:
        high = np.array([10.0, 10.0, 10.0])
        low = np.array([9.0, 9.0, 12.0])
        # gap=2, pct=0.2
        res = detect_fvg(high, low, min_gap_size=1.0, min_gap_size_pct=0.3)
        # pct fails -> filtered
        assert not res.bullish[2]
        res2 = detect_fvg(high, low, min_gap_size=1.0, min_gap_size_pct=0.1)
        assert res2.bullish[2]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestFVGEdgeCases:
    def test_empty_arrays(self) -> None:
        res = detect_fvg(np.array([]), np.array([]))
        assert res.bullish.size == 0
        assert res.bearish.size == 0

    def test_length_less_than_3(self) -> None:
        for n in [0, 1, 2]:
            high = np.ones(n)
            low = np.zeros(n)
            res = detect_fvg(high, low)
            assert not res.bullish.any()
            assert not res.bearish.any()

    def test_flat_prices_no_fvg(self) -> None:
        high = np.full(10, 10.0)
        low = np.full(10, 10.0)
        res = detect_fvg(high, low)
        assert not res.bullish.any()
        assert not res.bearish.any()

    def test_nan_handling(self) -> None:
        high = np.array([10.0, np.nan, 10.0, 11.0, 12.0])
        low = np.array([9.0, 9.0, np.nan, 12.0, 13.0])
        res = detect_fvg(high, low)
        # i=2 has NaN in window -> no FVG
        # i=3: Low[3]=12 > High[1]=NaN -> invalid -> no FVG
        assert not res.bullish[2]
        assert not res.bullish[3]

    def test_nan_low_invalidates(self) -> None:
        high = np.array([10.0, 10.0, 10.0])
        low = np.array([9.0, 9.0, np.nan])
        res = detect_fvg(high, low)
        assert not res.bullish[2]
        assert not res.bearish[2]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            detect_fvg(np.array([1.0, 2.0]), np.array([1.0]))

    def test_negative_filter_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            detect_fvg(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), min_gap_size=-1)
        with pytest.raises(ValueError, match="non-negative"):
            detect_fvg(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), min_gap_size_pct=-0.1)

    def test_2d_input_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            detect_fvg(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))

    def test_all_nan(self) -> None:
        high = np.full(5, np.nan)
        low = np.full(5, np.nan)
        res = detect_fvg(high, low)
        assert not res.bullish.any()
        assert not res.bearish.any()


# ---------------------------------------------------------------------------
# Mitigation
# ---------------------------------------------------------------------------


class TestFVGMitigation:
    def test_mitigation_bullish_gap_gets_pierced(self) -> None:
        # Bullish FVG at 2: High[0]=10, Low[2]=11 -> gap [10, 11]
        # Later low at 4 pierces it: low[4]=10.5 <= 11 ; low[3] must not pierce
        high = np.array([10.0, 10.2, 10.8, 11.5, 11.2, 12.0])
        low = np.array([9.0, 9.2, 11.0, 11.1, 10.5, 11.5])
        res = detect_fvg(high, low, compute_mitigation=True)
        assert res.bullish[2]
        assert res.mitigated[2] == True  # noqa: E712
        assert res.mitigated_index[2] == 4

    def test_mitigation_bearish_gap_gets_pierced(self) -> None:
        # Bearish FVG at 2: Low[0]=10, High[2]=8 -> gap [8,10]
        # Later high at 4 pierces: high[4]=9 >= 8
        high = np.array([11.0, 12.0, 8.0, 9.0, 9.0, 7.0])
        low = np.array([10.0, 11.0, 7.0, 8.0, 8.0, 6.0])
        res = detect_fvg(high, low, compute_mitigation=True)
        assert res.bearish[2]
        assert res.mitigated[2] == True  # noqa: E712
        assert res.mitigated_index[2] >= 3

    def test_no_mitigation_when_not_pierced(self) -> None:
        # Bullish FVG that never gets retested
        high = np.array([10.0, 10.0, 10.0, 12.0, 13.0, 14.0])
        low = np.array([9.0, 9.0, 11.0, 11.5, 12.0, 13.0])
        res = detect_fvg(high, low, compute_mitigation=True)
        assert res.bullish[2]
        assert not res.mitigated[2]
        assert res.mitigated_index[2] == -1

    def test_mitigation_disabled_returns_empty(self) -> None:
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 9.0, 12.0])
        res = detect_fvg(high, low, compute_mitigation=False)
        assert not res.mitigated.any()
        assert (res.mitigated_index == -1).all()

    def test_mitigation_short_series(self) -> None:
        res = detect_fvg(np.array([1.0]), np.array([1.0]), compute_mitigation=True)
        assert not res.mitigated.any()


# ---------------------------------------------------------------------------
# Polars integration
# ---------------------------------------------------------------------------


class TestFVGPolars:
    def test_polars_basic(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from pysmc.fvg import fvg_polars

        df = pl.DataFrame(
            {
                "high": [10.0, 11.0, 10.5, 12.0, 11.0],
                "low": [9.0, 9.5, 9.0, 11.5, 10.0],
            }
        )
        out = fvg_polars(df)
        assert "fvg_bullish" in out.columns
        assert "fvg_bearish" in out.columns
        assert out["fvg_bullish"][3] == True  # noqa: E712

    def test_polars_mitigation_columns(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from pysmc.fvg import fvg_polars

        df = pl.DataFrame({"high": [10.0, 10.2, 10.8, 11.0, 11.2], "low": [9.0, 9.2, 11.0, 10.8, 10.5]})
        out = fvg_polars(df, compute_mitigation=True)
        assert "fvg_mitigated" in out.columns
        assert "fvg_mitigated_index" in out.columns


# ---------------------------------------------------------------------------
# Benchmark / vectorization check (correctness over performance)
# ---------------------------------------------------------------------------


class TestFVGVectorization:
    def test_no_python_loops_large_input(self) -> None:
        """Ensure detection works on large input without slowdown due to loops."""
        n = 100_000
        rng = np.random.default_rng(42)
        high = rng.uniform(100, 110, size=n)
        low = high - rng.uniform(0.1, 2.0, size=n)
        # Should complete quickly (< 1s) due to vectorization
        import time

        t0 = time.perf_counter()
        res = detect_fvg(high, low)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"FVG detection took {elapsed:.2f}s, expected <1s (vectorization check)"
        assert res.bullish.shape[0] == n

    def test_deterministic_output(self) -> None:
        high = np.array([10.0, 11.0, 12.0, 10.0, 13.0])
        low = np.array([9.0, 9.5, 11.0, 9.0, 12.5])
        r1 = detect_fvg(high, low)
        r2 = detect_fvg(high, low)
        assert np.array_equal(r1.bullish, r2.bullish)
        assert np.array_equal(r1.bearish, r2.bearish)
