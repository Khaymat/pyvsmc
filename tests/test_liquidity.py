"""Tests for pyvsmc.liquidity"""

import numpy as np
import pytest
from pyvsmc.liquidity import detect_liquidity

def test_equal_high():
    high=np.array([10.,10.01,10.,11.])
    low=np.array([9.,9.,9.,9.])
    close=np.array([9.5,9.5,9.5,9.5])
    res=detect_liquidity(high,low,close,equal_threshold=0.002)
    assert res.equal_high[1]==True
    assert not res.equal_high[3]

def test_sweep_high():
    # high sweep: max of 2 bars =10, then 10.5 >10 and close 9.8 <10
    high=np.array([10.,10.,10.,10.5,10.2])
    low=np.array([9.,9.,9.,9.,9.])
    close=np.array([9.5,9.5,9.5,9.8,9.5])
    res=detect_liquidity(high,low,close,sweep_lookback=3)
    assert res.sweep_high[3]==True
    assert res.sweep_level[3]==pytest.approx(10.)

def test_sweep_low():
    high=np.array([10.,10.,10.,10.,10.])
    low=np.array([10.,10.,10.,8.,10.])
    close=np.array([9.5,9.5,9.5,10.5,9.5])
    res=detect_liquidity(high,low,close,sweep_lookback=3)
    assert res.sweep_low[3]==True

def test_empty():
    res=detect_liquidity(np.array([]),np.array([]),np.array([]))
    assert res.equal_high.size==0
