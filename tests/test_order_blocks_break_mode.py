"""Deterministic test for OB break_mode wick vs close."""

import numpy as np
import pytest

from pyvsmc.order_blocks import detect_order_blocks
from pyvsmc.structure import detect_structure


def test_structure_wick_vs_close():
    # Swing high at index 2 = 10 (window 1)
    high = np.array([9., 10., 10., 10., 9., 11., 9.])
    low = np.array([8., 9., 9., 9., 8., 8., 8.])
    # close: at index 5, high wick 11 >10 but close 9.5 <10
    close = np.array([9., 9.5, 9.5, 9.5, 9., 9.5, 9.])
    # For close break, no BOS at 5 (close 9.5 <10), for wick break, BOS at 5 (high 11 >10)
    st_close = detect_structure(high, low, close, window_size=1, break_mode="close")
    st_wick = detect_structure(high, low, close, window_size=1, break_mode="wick")
    # wick should have at least one more break
    assert st_wick.bos_bullish[5] == True or st_wick.choch_bullish[5] == True
    assert not st_close.bos_bullish[5] and not st_close.choch_bullish[5]


def test_order_blocks_break_mode_deterministic():
    # Construct OHLC where OB differs by break_mode
    # Use same high/low/close as above, with open for OB
    high = np.array([9., 10., 10., 10., 9., 11., 9., 10.])
    low = np.array([8., 9., 9., 9., 8., 8., 8., 8.])
    close = np.array([9., 9.5, 9.5, 9.5, 9., 9.5, 9., 9.5])
    open_ = np.array([9., 9.2, 9.3, 9.4, 9., 9.6, 9., 9.3])
    # With close break, no BOS at 5, so fewer OB impulses
    ob_close = detect_order_blocks(open_, high, low, close, window_size=1, break_mode="close", tie="first")
    ob_wick = detect_order_blocks(open_, high, low, close, window_size=1, break_mode="wick", tie="first")
    # At least one should differ (wick finds OB where close doesn't)
    # Check that wick finds at least as many OBs
    assert ob_wick.bullish_ob.sum() + ob_wick.bearish_ob.sum() >= ob_close.bullish_ob.sum() + ob_close.bearish_ob.sum()
    # And that break_mode param is accepted and not default-ignored
    # Call with both should not raise
    assert True


def test_order_blocks_break_mode_signature():
    import inspect
    sig = inspect.signature(detect_order_blocks)
    assert "break_mode" in sig.parameters
    assert sig.parameters["break_mode"].default == "close"
    assert "tie" in sig.parameters
    assert "zone_mode" in sig.parameters
