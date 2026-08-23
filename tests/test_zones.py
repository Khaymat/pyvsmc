"""Tests for pyvsmc.zones"""

import numpy as np
from pyvsmc.zones import detect_zones

def test_premium_discount():
    # swing high at 1=10, swing low at 3=2 => range [2,10], eq=6
    # avoid flat plateau so later highs don't become swings
    high=np.array([5.,10.,6.,4.,3.,3.])
    low=np.array([4.,4.,4.,2.,4.,4.])
    close=np.array([4.,9.,4.,6.,8.,5.])
    res=detect_zones(high,low,close,window_size=1,eq_threshold=0.02)
    # At i=4 onward, range_high 10, range_low 2
    assert res.range_high[5]==10.
    assert res.range_low[5]==2.
    assert res.premium[4]==True  # 8 >6
    assert res.discount[5]==True  # 5 <6

def test_empty():
    res=detect_zones(np.array([]),np.array([]),np.array([]))
    assert res.premium.size==0
