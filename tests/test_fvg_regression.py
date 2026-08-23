"""Regression tests for FVG 0.3.1 -> 0.3.2 Numba mitigation"""

import numpy as np
import pytest
from pyvsmc.fvg import detect_fvg


def test_first_mitigation_index():
    # Bullish FVG at 2: High[0]=10, Low[2]=11 -> gap [10,11], ce=10.5
    # Future lows: 11.1 (no mit), 10.8 (mit any), 10.4 (mit 50), 9.5 (full)
    high = np.array([10., 10.1, 10.8, 11.1, 11., 10.6, 9.8])
    low = np.array([9., 9., 11., 11.1, 10.8, 10.4, 9.5])
    close = np.array([9.5, 9.5, 10.5, 11., 10.9, 10.5, 9.6])
    res = detect_fvg(high, low, compute_mitigation=True, close=close)
    assert res.bullish[2]
    # first mit is low <=11 at index 4 (10.8)
    assert res.mitigated[2] and res.mitigated_index[2] == 4
    # ce 10.5 hit at 5 (10.4)
    assert res.mitigated_50[2] and res.mitigated_50_index[2] == 5
    # full <10 at 6 (9.5)
    assert res.mitigated_full[2] and res.mitigated_full_index[2] == 6


def test_bearish_ce_full():
    high = np.array([11., 12., 8., 9., 9., 10.])
    low = np.array([10., 11., 7., 8., 8., 6.])
    close = np.array([10.5, 11.5, 7.5, 8.5, 8.5, 6.5])
    res = detect_fvg(high, low, compute_mitigation=True, close=close)
    assert res.bearish[2]
    # bearish gap [8,10] ce=9
    # high 4=9 >= 8 mit, >=9 ce50, >=10 full at 5
    assert res.mitigated[2]
    assert res.mitigated_50[2]
    assert res.mitigated_full[2]


def test_inverted_ifvg():
    # bullish FVG then close below lower -> inverted
    high = np.array([10., 10.1, 10.8, 12., 13., 9.])
    low = np.array([9., 9., 11., 11.5, 12., 8.])
    close = np.array([9.5, 9.5, 10.5, 11.8, 12.5, 8.5])  # close 8.5 < lower 10 at mit 5
    res = detect_fvg(high, low, compute_mitigation=True, close=close)
    assert res.bullish[2]
    assert res.mitigated_full[2]
    assert res.inverted[2]  # close 8.5 < 10


def test_no_mitigation():
    high = np.array([10., 10., 10., 12., 13., 14.])
    low = np.array([9., 9., 11., 11.5, 12., 13.])
    close = np.array([9.5, 9.5, 10.5, 11.8, 12.5, 13.5])
    res = detect_fvg(high, low, compute_mitigation=True, close=close)
    assert res.bullish[2]
    assert not res.mitigated[2]
    assert not res.mitigated_50[2]
    assert not res.mitigated_full[2]
    assert not res.inverted[2]


def test_nan_poison():
    high = np.array([10., np.nan, 10., 11., 12.])
    low = np.array([9., 9., np.nan, 12., 13.])
    close = np.array([9.5, 9.5, 9.5, 11.5, 12.5])
    res = detect_fvg(high, low, compute_mitigation=True, close=close)
    # Gap at 2 is nan, should not be bullish, so no mit
    assert not res.bullish[2]
    # No crash, mitigated all false for nan gaps
    assert not res.mitigated[2]


def test_dense_gaps_scalability():
    # 200 bars dense gaps: alternating to create many FVGs
    n = 200
    high = np.tile([10., 11.], n//2).astype(float)
    low = np.tile([12., 9.], n//2).astype(float)  # low>high pattern creates many gaps
    # Adjust to ensure gaps: need low[i] > high[i-2]
    # Use simple pattern: high low sawtooth
    rng = np.random.default_rng(42)
    high = rng.uniform(100, 101, n)
    low = high + rng.uniform(1, 2, n)  # ensure low > high_prev? make high shift
    high[2:] = low[:-2] - 0.5  # force low[i] > high[i-2] for all i>=2
    res = detect_fvg(high, low, compute_mitigation=True)
    # should not hang, should have many gaps
    assert res.bullish.sum() > 100
    # mitigated should be computed without OOM and not exceed total gaps
    assert res.mitigated.sum() <= (res.bullish.sum() + res.bearish.sum())


def test_adversarial_small_correctness():
    # Pathological: many bullish gaps, mit at last bar — correctness only, CI-safe n=1000, no bearish to avoid overwrite
    n = 1000
    base = 10.0 + np.arange(n, dtype=np.float64) * 1.0
    low = base
    high = base + 1.0
    low[-1] = 9.0
    high[-1] = 10.0
    close = np.full(n, 10.5)
    close[-1] = 9.0
    res = detect_fvg(high, low, compute_mitigation=True, close=close)
    # First gaps are bullish only
    assert res.bullish[2]
    assert not res.bearish[2]
    assert res.mitigated_index[2] == n - 1
    # All bullish gaps should mitigate at last bar
    idx = np.where(res.bullish)[0]
    assert np.all(res.mitigated_index[idx] == n - 1)
    assert int(res.bullish.sum()) >= 990
    # Allow at most 1 bearish (last bar edge)
    assert int(res.bearish.sum()) <= 1
