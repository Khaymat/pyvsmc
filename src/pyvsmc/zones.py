"""Premium / Discount / Dealing Range zones — vectorized.

Premium = upper 50% of dealing range [swing_low, swing_high]
Discount = lower 50%
Equilibrium = 50% midpoint
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .swings import detect_swings


@dataclass(frozen=True, slots=True)
class ZoneResult:
    """Zones based on dealing range.

    Attributes:
        premium: Close in premium zone (> 50%)
        discount: Close in discount zone (< 50%)
        equilibrium: Close near equilibrium (within threshold)
        range_high: Forward-filled last swing high (top of range)
        range_low: Forward-filled last swing low (bottom)
        equilibrium_level: (high+low)/2
        premium_level: top of premium (same as range_high)
        discount_level: bottom of discount (range_low)
    """

    premium: npt.NDArray[np.bool_]
    discount: npt.NDArray[np.bool_]
    equilibrium: npt.NDArray[np.bool_]
    range_high: npt.NDArray[np.float64]
    range_low: npt.NDArray[np.float64]
    equilibrium_level: npt.NDArray[np.float64]


def _to_f(a: npt.ArrayLike) -> npt.NDArray[np.float64]:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D, got {arr.shape}")
    return arr


def _ffill_last_swing(mask: npt.NDArray[np.bool_], price: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = mask.shape[0]
    idx = np.where(mask, np.arange(n), -1)
    filled = np.maximum.accumulate(idx)
    res = np.full(n, np.nan, dtype=np.float64)
    valid = filled >= 0
    res[valid] = price[filled[valid]]
    return res


def detect_zones(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    window_size: int = 2,
    eq_threshold: float = 0.02,
) -> ZoneResult:
    """Detect premium/discount zones — vectorized.

    Dealing range defined by last swing high/low (forward-filled, shifted by 1 to avoid lookahead).
    Premium if close > equilibrium, discount if close < equilibrium.

    Args:
        high, low, close: OHLC (n,)
        window_size: swing detection N
        eq_threshold: fraction of range to consider equilibrium (0.02 = 2% from 50%)

    Returns:
        ZoneResult
    """
    h = _to_f(high); lo = _to_f(low); cl = _to_f(close)
    n = h.shape[0]
    if not (lo.shape[0]==n and cl.shape[0]==n):
        raise ValueError("high, low, close must same length")
    if window_size < 1:
        raise ValueError("window_size >=1")
    if n==0:
        return ZoneResult(premium=np.zeros(0,dtype=bool),discount=np.zeros(0,dtype=bool),equilibrium=np.zeros(0,dtype=bool),range_high=np.zeros(0,dtype=np.float64),range_low=np.zeros(0,dtype=np.float64),equilibrium_level=np.zeros(0,dtype=np.float64))
    swings = detect_swings(h, lo, window_size=window_size)
    sh = _ffill_last_swing(swings.swing_high, h)
    sl = _ffill_last_swing(swings.swing_low, lo)
    # shift by 1
    rh = np.empty(n, dtype=np.float64); rh[0]=np.nan; rh[1:]=sh[:-1] if n>1 else []
    rl = np.empty(n, dtype=np.float64); rl[0]=np.nan; rl[1:]=sl[:-1] if n>1 else []
    valid = ~np.isnan(rh) & ~np.isnan(rl) & ~np.isnan(cl) & (rh > rl)
    eq = np.full(n, np.nan, dtype=np.float64)
    eq[valid] = (rh[valid] + rl[valid]) / 2.0
    # range size for threshold
    rng = rh - rl
    premium = np.zeros(n, dtype=bool)
    discount = np.zeros(n, dtype=bool)
    equilibrium = np.zeros(n, dtype=bool)
    # equilibrium where |close - eq| / rng <= threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.abs(cl - eq) / rng
    is_eq = valid & (dist <= eq_threshold)
    equilibrium[is_eq] = True
    # premium: close > eq and not equilibrium
    premium[valid & (cl > eq) & ~is_eq] = True
    discount[valid & (cl < eq) & ~is_eq] = True
    return ZoneResult(premium, discount, equilibrium, rh, rl, eq)


def zones_polars(df: object, *, high_col="high", low_col="low", close_col="close", window_size=2, eq_threshold=0.02) -> object:
    try:
        import polars as pl
    except ImportError as e:
        raise ImportError("polars required") from e
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df)}")
    res = detect_zones(df[high_col].to_numpy().astype(float), df[low_col].to_numpy().astype(float), df[close_col].to_numpy().astype(float), window_size=window_size, eq_threshold=eq_threshold)
    return df.with_columns([pl.Series("premium", res.premium), pl.Series("discount", res.discount), pl.Series("equilibrium", res.equilibrium), pl.Series("range_high", res.range_high), pl.Series("range_low", res.range_low), pl.Series("equilibrium_level", res.equilibrium_level)])
