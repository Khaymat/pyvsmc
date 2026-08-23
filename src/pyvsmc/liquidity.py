"""Liquidity sweeps / equal highs-lows — fully vectorized."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class LiquidityResult:
    """Container for liquidity detection.

    All arrays shape (n,).

    Attributes:
        equal_high: True where High[i] ≈ High[i-1] within threshold.
        equal_low: True where Low[i] ≈ Low[i-1] within threshold.
        sweep_high: True where High[i] > recent swing high then close back below (liquidity grab).
        sweep_low: True where Low[i] < recent swing low then close back above.
        sweep_level: Level that was swept (prior high/low), np.nan elsewhere.
    """

    equal_high: npt.NDArray[np.bool_]
    equal_low: npt.NDArray[np.bool_]
    sweep_high: npt.NDArray[np.bool_]
    sweep_low: npt.NDArray[np.bool_]
    sweep_level: npt.NDArray[np.float64]


def _to_f(a: npt.ArrayLike) -> npt.NDArray[np.float64]:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D, got {arr.shape}")
    return arr


def detect_liquidity(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    *,
    equal_threshold: float = 0.001,
    sweep_lookback: int = 20,
    window_size: int = 2,
) -> LiquidityResult:
    """Detect equal highs/lows and liquidity sweeps — vectorized.

    * Equal highs: |High[i]-High[i-1]| / High[i-1] <= threshold (or abs if threshold>1)
    * Sweep high: High[i] > max(High[i-lookback:i]) and Close[i] < that max (grab)
    * Sweep low: Low[i] < min(Low[i-lookback:i]) and Close[i] > that min

    Args:
        high, low, close: OHLC arrays (n,).
        equal_threshold: Fraction for equal detection (0.001=0.1%). If >1 interpreted as absolute.
        sweep_lookback: Lookback for sweep level.
        window_size: Unused, kept for API symmetry.

    Returns:
        LiquidityResult
    """
    h = _to_f(high); lo = _to_f(low); cl = _to_f(close)
    n = h.shape[0]
    if not (lo.shape[0]==n and cl.shape[0]==n):
        raise ValueError("high, low, close must same length")
    equal_high = np.zeros(n, dtype=bool)
    equal_low = np.zeros(n, dtype=bool)
    sweep_high = np.zeros(n, dtype=bool)
    sweep_low = np.zeros(n, dtype=bool)
    sweep_level = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return LiquidityResult(equal_high, equal_low, sweep_high, sweep_low, sweep_level)

    # Equal detection vectorized
    valid_h = ~np.isnan(h[:-1]) & ~np.isnan(h[1:])
    valid_l = ~np.isnan(lo[:-1]) & ~np.isnan(lo[1:])
    if equal_threshold > 1:
        eq_h = np.abs(h[1:] - h[:-1]) <= equal_threshold
        eq_l = np.abs(lo[1:] - lo[:-1]) <= equal_threshold
    else:
        # fraction
        with np.errstate(divide="ignore", invalid="ignore"):
            denom_h = np.abs(h[:-1])
            denom_h[denom_h==0]=np.nan
            eq_h = np.abs(h[1:]-h[:-1])/denom_h <= equal_threshold
            denom_l = np.abs(lo[:-1])
            denom_l[denom_l==0]=np.nan
            eq_l = np.abs(lo[1:]-lo[:-1])/denom_l <= equal_threshold
        eq_h = np.nan_to_num(eq_h, nan=False)
        eq_l = np.nan_to_num(eq_l, nan=False)
    equal_high[1:][valid_h] = eq_h[valid_h]
    equal_low[1:][valid_l] = eq_l[valid_l]

    # Sweep detection: rolling max/min of lookback (excluding current)
    # Use stride view for efficiency
    if n > sweep_lookback:
        try:
            from numpy.lib.stride_tricks import sliding_window_view
        except ImportError as e:
            raise ImportError("sliding_window_view requires numpy>=1.20") from e
        # For i >= sweep_lookback, level = max(high[i-lookback:i])
        # sliding_window_view of size lookback gives windows [0:lb], [1:lb+1] ...
        # window j corresponds to high[j:j+lb] -> max for i = j+lb
        h_windows = sliding_window_view(h, window_shape=sweep_lookback)
        l_windows = sliding_window_view(lo, window_shape=sweep_lookback)
        # h_windows shape (n-lb+1, lb) -> we need up to n-1
        # valid where no nan in window
        h_valid = ~np.isnan(h_windows).any(axis=1)
        l_valid = ~np.isnan(l_windows).any(axis=1)
        h_max = np.nanmax(h_windows, axis=1)  # len n-lb+1
        l_min = np.nanmin(l_windows, axis=1)
        # Align: for i = lb .. n-1, level = h_max[i-lb]
        idx = np.arange(sweep_lookback, n)
        # High sweep: high[i] > h_max[i-lb] and close[i] < h_max[i-lb]
        # only where window valid and current not nan
        valid_curr = ~np.isnan(h[idx]) & ~np.isnan(cl[idx])
        cond_h = (h[idx] > h_max[idx - sweep_lookback]) & (cl[idx] < h_max[idx - sweep_lookback])
        sweep_high[idx] = cond_h & h_valid[idx - sweep_lookback] & valid_curr
        sweep_level[idx[sweep_high[idx]]] = h_max[idx[sweep_high[idx]] - sweep_lookback]
        valid_curr_l = ~np.isnan(lo[idx]) & ~np.isnan(cl[idx])
        cond_l = (lo[idx] < l_min[idx - sweep_lookback]) & (cl[idx] > l_min[idx - sweep_lookback])
        sweep_low[idx] = cond_l & l_valid[idx - sweep_lookback] & valid_curr_l
        # For sweep_low, set level where not already set (sweep_high has priority)
        mask_low = sweep_low[idx]
        low_idx = idx[mask_low]
        # Only set where not already high sweep (if both, keep high)
        need = np.isnan(sweep_level[low_idx])
        sweep_level[low_idx[need]] = l_min[low_idx[need] - sweep_lookback]

    return LiquidityResult(equal_high, equal_low, sweep_high, sweep_low, sweep_level)


def liquidity_polars(df: object, *, high_col="high", low_col="low", close_col="close", equal_threshold=0.001, sweep_lookback=20) -> object:
    try:
        import polars as pl
    except ImportError as e:
        raise ImportError("polars required") from e
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df)}")
    res = detect_liquidity(df[high_col].to_numpy().astype(float), df[low_col].to_numpy().astype(float), df[close_col].to_numpy().astype(float), equal_threshold=equal_threshold, sweep_lookback=sweep_lookback)
    return df.with_columns([pl.Series("equal_high", res.equal_high), pl.Series("equal_low", res.equal_low), pl.Series("sweep_high", res.sweep_high), pl.Series("sweep_low", res.sweep_low), pl.Series("sweep_level", res.sweep_level)])
