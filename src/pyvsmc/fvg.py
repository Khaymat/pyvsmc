"""Fair Value Gap (FVG) / Imbalance detection — fully vectorized.

A Fair Value Gap is a 3-candle price inefficiency pattern.

Definitions
-----------
* **Bullish FVG (buy-side imbalance):**  Low[i] > High[i-2] for i >= 2.
  The gap zone is [High[i-2], Low[i]].
* **Bearish FVG (sell-side imbalance):**  High[i] < Low[i-2] for i >= 2.
  The gap zone is [High[i], Low[i-2]].

This module provides a single vectorized entry point :func:`detect_fvg` that
returns a :class:`FVGResult` dataclass with boolean masks, boundaries, gap
sizes, and an optional vectorized mitigation tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np
import numpy.typing as npt

_HAS_NUMBA = importlib.util.find_spec("numba") is not None
if _HAS_NUMBA:
    import numba as nb  # type: ignore[import-untyped]

    @nb.njit(cache=True)
    def _fvg_first_idx_numba(
        n: int,
        gap_idx: np.ndarray,
        thresh: np.ndarray,
        target: np.ndarray,
        is_low: bool,
        out_idx: np.ndarray,
    ) -> None:
        # target is low if is_low else high; thresh per gap
        for k in range(gap_idx.shape[0]):
            idx = gap_idx[k]
            th = thresh[idx]
            if np.isnan(th):
                continue
            # need mitigated flag already true, but search anyway
            for j in range(idx + 1, n):
                v = target[j]
                if np.isnan(v):
                    continue
                if is_low:
                    if v <= th:
                        out_idx[idx] = j
                        break
                else:
                    if v >= th:
                        out_idx[idx] = j
                        break

    @nb.njit(cache=True)
    def _fvg_first_idx_three_numba(
        n: int,
        gap_idx: np.ndarray,
        mit_thresh: np.ndarray,
        ce_thresh: np.ndarray,
        full_thresh: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        close: np.ndarray,
        out_mit: np.ndarray,
        out_mit_idx: np.ndarray,
        out_50: np.ndarray,
        out_50_idx: np.ndarray,
        out_full: np.ndarray,
        out_full_idx: np.ndarray,
        out_inv: np.ndarray,
        is_bull: bool,
    ) -> None:
        for k in range(gap_idx.shape[0]):
            idx = gap_idx[k]
            th_mit = mit_thresh[idx]
            th_50 = ce_thresh[idx]
            th_full = full_thresh[idx]
            found_mit = False
            found_50 = False
            found_full = False
            for j in range(idx + 1, n):
                if not found_mit and not np.isnan(th_mit):
                    v = low[j] if is_bull else high[j]
                    if not np.isnan(v):
                        if is_bull:
                            if v <= th_mit:
                                out_mit[idx] = True
                                out_mit_idx[idx] = j
                                found_mit = True
                        else:
                            if v >= th_mit:
                                out_mit[idx] = True
                                out_mit_idx[idx] = j
                                found_mit = True
                if not found_50 and not np.isnan(th_50):
                    v = low[j] if is_bull else high[j]
                    if not np.isnan(v):
                        if is_bull:
                            if v <= th_50:
                                out_50[idx] = True
                                out_50_idx[idx] = j
                                found_50 = True
                        else:
                            if v >= th_50:
                                out_50[idx] = True
                                out_50_idx[idx] = j
                                found_50 = True
                if not found_full and not np.isnan(th_full):
                    v = low[j] if is_bull else high[j]
                    if not np.isnan(v):
                        if is_bull:
                            if v <= th_full:
                                out_full[idx] = True
                                out_full_idx[idx] = j
                                found_full = True
                        else:
                            if v >= th_full:
                                out_full[idx] = True
                                out_full_idx[idx] = j
                                found_full = True
                if found_mit and found_50 and found_full:
                    break
            if found_full:
                fj = out_full_idx[idx]
                if fj != -1 and fj < n and not np.isnan(close[fj]):
                    if is_bull:
                        if close[fj] < full_thresh[idx]:
                            out_inv[idx] = True
                    else:
                        if close[fj] > full_thresh[idx]:
                            out_inv[idx] = True



@dataclass(frozen=True, slots=True)
class FVGResult:
    """Container for Fair Value Gap detection results.

    All array attributes have shape ``(n,)`` where ``n`` is the input length.
    Indices ``0`` and ``1`` are never gaps (need 3 candles).

    Attributes:
        bullish: Boolean mask — ``True`` where a bullish FVG is present.
        bearish: Boolean mask — ``True`` where a bearish FVG is present.
        bullish_upper: Upper boundary of bullish gap (Low[i]); ``np.nan`` elsewhere.
        bullish_lower: Lower boundary of bullish gap (High[i-2]); ``np.nan`` elsewhere.
        bearish_upper: Upper boundary of bearish gap (Low[i-2]); ``np.nan`` elsewhere.
        bearish_lower: Lower boundary of bearish gap (High[i]); ``np.nan`` elsewhere.
        gap_size: Absolute gap size (upper - lower) for detected gaps; ``np.nan`` elsewhere.
        gap_size_pct: Gap size as percentage of lower boundary; ``np.nan`` elsewhere.
        ce_level: Consequent Encroachment 50% level ((upper+lower)/2); ``np.nan`` elsewhere.
        mitigated: Boolean mask — ``True`` where a gap has been mitigated
            (price pierced the zone) at any future bar.  All ``False`` when
            mitigation is not computed.
        mitigated_index: Index of the first mitigation bar for each FVG;
            ``-1`` when not mitigated or not computed.
        mitigated_50: True where price reached 50% CE level.
        mitigated_50_index: Index of first 50% mitigation.
        mitigated_full: True where price fully filled gap (beyond opposite side).
        mitigated_full_index: Index of first full fill.
        inverted: True where FVG was inverted (close beyond opposite side after mitigation) — IFVG.
    """

    bullish: npt.NDArray[np.bool_]
    bearish: npt.NDArray[np.bool_]
    bullish_upper: npt.NDArray[np.float64]
    bullish_lower: npt.NDArray[np.float64]
    bearish_upper: npt.NDArray[np.float64]
    bearish_lower: npt.NDArray[np.float64]
    gap_size: npt.NDArray[np.float64]
    gap_size_pct: npt.NDArray[np.float64]
    ce_level: npt.NDArray[np.float64]
    mitigated: npt.NDArray[np.bool_]
    mitigated_index: npt.NDArray[np.int64]
    mitigated_50: npt.NDArray[np.bool_]
    mitigated_50_index: npt.NDArray[np.int64]
    mitigated_full: npt.NDArray[np.bool_]
    mitigated_full_index: npt.NDArray[np.int64]
    inverted: npt.NDArray[np.bool_]


def _to_float64(arr: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce array-like to 1-D float64 ndarray."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {a.shape}")
    return a


def _compute_mitigation_vectorized(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    bullish_mask: npt.NDArray[np.bool_],
    bearish_mask: npt.NDArray[np.bool_],
    bullish_upper: npt.NDArray[np.float64],
    bullish_lower: npt.NDArray[np.float64],
    bearish_upper: npt.NDArray[np.float64],
    bearish_lower: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """Vectorized mitigation detection.

    A **bullish** FVG [lower, upper] is considered *mitigated* when a future
    candle's low pierces below the upper boundary (or fully through), i.e.
    ``low[j] <= bullish_upper[i]`` for some ``j > i``.  More conservatively
    some traders require ``low[j] < bullish_lower[i]`` (full fill).  Here we
    use the standard 50% / any-touch definition via ``low[j] < bullish_upper[i]``
    threshold and also flag full mitigation.

    A **bearish** FVG [lower, upper] is mitigated when ``high[j] >= bearish_lower[i]``.

    Complexity:
    - Mitigation existence flags via reverse-cumulative min/max are O(n)
      (NaN-aware, `n+1` aux).
    - First-occurrence indices are O(n + g·d) average/practical where
      `g` = number of gaps and `d` = average mitigation distance (early-break
      Numba scan, single pass for mit/CE50/full). Worst case O(g·n),
      Θ(n²) when g=Θ(n) and d=Θ(n) (e.g. all gaps mitigate only at final bar).
    - Do not confuse the O(n) flag pass with the first-index pass.

    Returns:
        mitigated: bool array shape (n,)
        mitigated_index: int array shape (n,) with -1 = never mitigated.
    """
    n = high.shape[0]
    mitigated = np.zeros(n, dtype=bool)
    mitigated_idx = np.full(n, -1, dtype=np.int64)

    if n < 3:
        return mitigated, mitigated_idx

    # --- Bullish mitigation ---
    # For each bullish FVG at i, we need min(low[i+1:]) < bullish_upper[i] ?
    # Compute reverse cumulative minimum of low.
    # rev_cummin_low[k] = min(low[k:])
    # Then mitigated if rev_cummin_low[i+1] <= bullish_upper[i]
    # To also get the *first* mitigation index we need a search. We use
    # broadcasting-safe argmax approach: build a 2-D comparison only for
    # indices where gaps exist to keep memory bounded.
    # Fallback to chunked vectorized search for memory safety.

    # Reverse cum min / max for fast mitigated flag (O(n))
    rev_cummin_low = np.empty(n + 1, dtype=np.float64)
    rev_cummin_low[n] = np.inf
    # Use numba-free vectorized accumulate: iterate reversed with numpy ufunc.accumulate
    # We can compute via: np.minimum.accumulate(low[::-1])[::-1]
    if n > 0:
        rev_cummin_low[:n] = np.minimum.accumulate(low[::-1])[::-1]
    rev_cummax_high = np.empty(n + 1, dtype=np.float64)
    rev_cummax_high[n] = -np.inf
    if n > 0:
        rev_cummax_high[:n] = np.maximum.accumulate(high[::-1])[::-1]

    bullish_indices = np.where(bullish_mask)[0]
    bearish_indices = np.where(bearish_mask)[0]

    # Fast flag using reverse cumulative
    if bullish_indices.size > 0:
        # rev_cummin_low[i+1] is min of low after i
        future_min = rev_cummin_low[bullish_indices + 1]
        # Use <= to detect any touch of the gap zone
        flag = future_min <= bullish_upper[bullish_indices]
        mitigated[bullish_indices[flag]] = True

    if bearish_indices.size > 0:
        future_max = rev_cummax_high[bearish_indices + 1]
        flag = future_max >= bearish_lower[bearish_indices]
        mitigated[bearish_indices[flag]] = True

    # Find first mitigation index per gap via 1-D scan (low memory, O(m*n) but m small)
    if bullish_indices.size > 0:
        bull_mitigated_idx = bullish_indices[mitigated[bullish_indices]]
        for idx in bull_mitigated_idx:
            thresh = bullish_upper[idx]
            # search future low <= thresh
            future = low[idx + 1 :]
            # use where to find first
            pos = np.where(future <= thresh)[0]
            if pos.size > 0:
                mitigated_idx[idx] = int(idx + 1 + pos[0])

    if bearish_indices.size > 0:
        bear_mitigated_idx = bearish_indices[mitigated[bearish_indices]]
        for idx in bear_mitigated_idx:
            thresh = bearish_lower[idx]
            future = high[idx + 1 :]
            pos = np.where(future >= thresh)[0]
            if pos.size > 0:
                mitigated_idx[idx] = int(idx + 1 + pos[0])

    return mitigated, mitigated_idx


def _compute_threshold_mitigation(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    bullish: npt.NDArray[np.bool_],
    bearish: npt.NDArray[np.bool_],
    bull_thresh: npt.NDArray[np.float64],
    bear_thresh: npt.NDArray[np.float64],
    bull_cond: str = "low_le",
    bear_cond: str = "high_ge",
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int64]]:
    """Generic threshold mitigation (vectorized for CE/full)."""
    n = high.shape[0]
    mitigated = np.zeros(n, dtype=bool)
    mitigated_idx = np.full(n, -1, dtype=np.int64)
    if n < 3:
        return mitigated, mitigated_idx
    # Use future min/max trick for flag, then chunked search
    rev_cummin_low = np.empty(n + 1, dtype=np.float64)
    rev_cummin_low[n] = np.inf
    if n > 0:
        rev_cummin_low[:n] = np.minimum.accumulate(low[::-1])[::-1]
    rev_cummax_high = np.empty(n + 1, dtype=np.float64)
    rev_cummax_high[n] = -np.inf
    if n > 0:
        rev_cummax_high[:n] = np.maximum.accumulate(high[::-1])[::-1]
    rev_cummin_close = np.empty(n + 1, dtype=np.float64)
    rev_cummin_close[n] = np.inf
    if n > 0:
        rev_cummin_close[:n] = np.minimum.accumulate(close[::-1])[::-1]
    rev_cummax_close = np.empty(n + 1, dtype=np.float64)
    rev_cummax_close[n] = -np.inf
    if n > 0:
        rev_cummax_close[:n] = np.maximum.accumulate(close[::-1])[::-1]
    bull_idx = np.where(bullish)[0]
    bear_idx = np.where(bearish)[0]
    # flag
    if bull_idx.size > 0:
        # choose cond
        if bull_cond == "low_le":
            future = rev_cummin_low[bull_idx + 1]
            flag = future <= bull_thresh[bull_idx]
        elif bull_cond == "close_lt":
            future = rev_cummin_close[bull_idx + 1]
            flag = future < bull_thresh[bull_idx]
        else:
            flag = np.zeros_like(bull_idx, dtype=bool)
        mitigated[bull_idx[flag]] = True
    if bear_idx.size > 0:
        if bear_cond == "high_ge":
            future = rev_cummax_high[bear_idx + 1]
            flag = future >= bear_thresh[bear_idx]
        elif bear_cond == "close_gt":
            future = rev_cummax_close[bear_idx + 1]
            flag = future > bear_thresh[bear_idx]
        else:
            flag = np.zeros_like(bear_idx, dtype=bool)
        mitigated[bear_idx[flag]] = True
    # per-gap 1-D search (low memory)
    if bull_idx.size > 0:
        b_m = bull_idx[mitigated[bull_idx]]
        for idx in b_m:
            thresh = bull_thresh[idx]
            if bull_cond == "low_le":
                future = low[idx + 1 :]
                pos = np.where(future <= thresh)[0]
            else:
                future = close[idx + 1 :]
                pos = np.where(future < thresh)[0]
            if pos.size > 0:
                mitigated_idx[idx] = int(idx + 1 + pos[0])
    if bear_idx.size > 0:
        b_m = bear_idx[mitigated[bear_idx]]
        for idx in b_m:
            thresh = bear_thresh[idx]
            if bear_cond == "high_ge":
                future = high[idx + 1 :]
                pos = np.where(future >= thresh)[0]
            else:
                future = close[idx + 1 :]
                pos = np.where(future > thresh)[0]
            if pos.size > 0:
                mitigated_idx[idx] = int(idx + 1 + pos[0])
    return mitigated, mitigated_idx


def detect_fvg(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    min_gap_size: float | None = None,
    min_gap_size_pct: float | None = None,
    *,
    compute_mitigation: bool = False,
    close: npt.ArrayLike | None = None,
) -> FVGResult:
    """Detect Fair Value Gaps (imbalances) — fully vectorized.

    Args:
        high: Array of bar highs, shape ``(n,)``.
        low: Array of bar lows, shape ``(n,)``.
        min_gap_size: Minimum absolute gap size filter.  Gaps smaller than
            this are suppressed.  ``None`` means no absolute filter.
        min_gap_size_pct: Minimum gap size as fraction of the lower boundary
            (e.g. ``0.001`` = 0.1 %).  ``None`` means no percentage filter.
            When both filters are set the gap must pass *both*.
        compute_mitigation: When ``True``, compute vectorized mitigation state
            (whether a future candle pierced the gap zone).
        close: Unused currently; reserved for future wick/body-aware logic.
            Kept for API forward-compatibility.

    Returns:
        :class:`FVGResult` with boolean masks and boundary arrays.

    Raises:
        ValueError: If ``high`` and ``low`` have different lengths or are not
            1-D, or if filter values are negative.

    Examples:
        >>> import numpy as np
        >>> high = np.array([10., 11., 12., 10., 15.])
        >>> low  = np.array([ 9.,  9.5, 11., 9.8, 14.])
        >>> res = detect_fvg(high, low)
        >>> res.bullish
        array([False, False, False, False,  True])
    """
    h = _to_float64(high)
    lo = _to_float64(low)

    if h.shape[0] != lo.shape[0]:
        raise ValueError(f"high and low must have same length: {h.shape[0]} vs {lo.shape[0]}")
    if min_gap_size is not None and min_gap_size < 0:
        raise ValueError("min_gap_size must be non-negative")
    if min_gap_size_pct is not None and min_gap_size_pct < 0:
        raise ValueError("min_gap_size_pct must be non-negative")

    n = h.shape[0]

    # Prepare outputs
    bullish = np.zeros(n, dtype=bool)
    bearish = np.zeros(n, dtype=bool)
    bullish_upper = np.full(n, np.nan, dtype=np.float64)
    bullish_lower = np.full(n, np.nan, dtype=np.float64)
    bearish_upper = np.full(n, np.nan, dtype=np.float64)
    bearish_lower = np.full(n, np.nan, dtype=np.float64)
    gap_size = np.full(n, np.nan, dtype=np.float64)
    gap_size_pct = np.full(n, np.nan, dtype=np.float64)
    ce_level = np.full(n, np.nan, dtype=np.float64)

    if n < 3:
        return FVGResult(
            bullish=bullish,
            bearish=bearish,
            bullish_upper=bullish_upper,
            bullish_lower=bullish_lower,
            bearish_upper=bearish_upper,
            bearish_lower=bearish_lower,
            gap_size=gap_size,
            gap_size_pct=gap_size_pct,
            ce_level=ce_level,
            mitigated=np.zeros(n, dtype=bool),
            mitigated_index=np.full(n, -1, dtype=np.int64),
            mitigated_50=np.zeros(n, dtype=bool),
            mitigated_50_index=np.full(n, -1, dtype=np.int64),
            mitigated_full=np.zeros(n, dtype=bool),
            mitigated_full_index=np.full(n, -1, dtype=np.int64),
            inverted=np.zeros(n, dtype=bool),
        )

    # Vectorized core logic — no Python loops over time-series
    # Align arrays: for index i, compare High[i-2] vs Low[i] (or Low[i-2] vs High[i])
    high_shift2 = h[:-2]  # High[i-2] aligned to i where i>=2
    low_shift2 = lo[:-2]

    high_current = h[2:]
    low_current = lo[2:]

    # NaN-aware: any NaN in the triplet invalidates the gap at i
    # low[i] > high[i-2]  (need valid values)
    valid_bull = ~(np.isnan(low_current) | np.isnan(high_shift2))
    valid_bear = ~(np.isnan(high_current) | np.isnan(low_shift2))

    raw_bullish = np.zeros(n, dtype=bool)
    raw_bearish = np.zeros(n, dtype=bool)

    # Only check where valid
    bull_condition = np.zeros(n - 2, dtype=bool)
    bear_condition = np.zeros(n - 2, dtype=bool)

    bull_condition[valid_bull] = low_current[valid_bull] > high_shift2[valid_bull]
    bear_condition[valid_bear] = high_current[valid_bear] < low_shift2[valid_bear]

    raw_bullish[2:] = bull_condition
    raw_bearish[2:] = bear_condition

    # Compute gap sizes for all i >=2 (vectorized)
    # Bullish gap size = Low[i] - High[i-2]  (positive when bullish)
    # Bearish gap size = Low[i-2] - High[i]  (positive when bearish)
    # Bull gaps
    bull_gap_abs = low_current - high_shift2  # shape (n-2,)
    # Bear gaps
    bear_gap_abs = low_shift2 - high_current

    # Fill for indices 2..n-1 where respective raw flag true, else keep nan
    # But also compute for filtering for all (we need sizes even before filtering)
    # We'll populate computed arrays for filtering
    tmp_size = np.full(n, np.nan, dtype=np.float64)
    tmp_size[2:][bull_condition] = bull_gap_abs[bull_condition]
    tmp_size[2:][bear_condition] = bear_gap_abs[bear_condition]

    # Compute pct relative to lower boundary
    tmp_pct = np.full(n, np.nan, dtype=np.float64)
    # Avoid division by zero / nan
    if bull_condition.any():
        idx_bull = np.where(raw_bullish)[0]
        lower_bull = h[idx_bull - 2]
        # pct = gap / lower * 100? spec says percentage filter; we use fraction
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_bull = np.where(lower_bull != 0, bull_gap_abs[bull_condition] / np.abs(lower_bull), np.nan)
        tmp_pct[idx_bull] = pct_bull
    if bear_condition.any():
        idx_bear = np.where(raw_bearish)[0]
        lower_bear = h[idx_bear]  # High[i] is lower of bear gap
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_bear = np.where(lower_bear != 0, bear_gap_abs[bear_condition] / np.abs(lower_bear), np.nan)
        tmp_pct[idx_bear] = pct_bear

    # Apply filters vectorized
    keep = np.ones(n, dtype=bool)
    if min_gap_size is not None:
        # Gaps must have size >= threshold; non-gaps keep==False? We set non-gaps to False via raw mask later
        # For filter, suppress gaps smaller than threshold
        size_ok = np.zeros(n, dtype=bool)
        # Only evaluate where gap exists
        gap_exists = raw_bullish | raw_bearish
        size_ok[gap_exists] = tmp_size[gap_exists] >= min_gap_size
        # Non-gap indices are not suppressed by this filter (they remain not-gap anyway)
        keep &= (~gap_exists | size_ok)

    if min_gap_size_pct is not None:
        pct_ok = np.zeros(n, dtype=bool)
        gap_exists = raw_bullish | raw_bearish
        # tmp_pct is fraction; compare directly
        pct_ok[gap_exists] = tmp_pct[gap_exists] >= min_gap_size_pct
        keep &= (~gap_exists | pct_ok)

    bullish = raw_bullish & keep
    bearish = raw_bearish & keep

    # Boundaries
    # Bullish: upper = Low[i], lower = High[i-2]
    idx_bull_final = np.where(bullish)[0]
    bullish_upper[idx_bull_final] = lo[idx_bull_final]
    bullish_lower[idx_bull_final] = h[idx_bull_final - 2]

    idx_bear_final = np.where(bearish)[0]
    bearish_upper[idx_bear_final] = lo[idx_bear_final - 2]
    bearish_lower[idx_bear_final] = h[idx_bear_final]

    # Gap sizes final (only for kept gaps)
    gap_size[idx_bull_final] = tmp_size[idx_bull_final]
    gap_size[idx_bear_final] = tmp_size[idx_bear_final]
    gap_size_pct[idx_bull_final] = tmp_pct[idx_bull_final]
    gap_size_pct[idx_bear_final] = tmp_pct[idx_bear_final]

    # CE 50% level
    ce_level[idx_bull_final] = (bullish_upper[idx_bull_final] + bullish_lower[idx_bull_final]) / 2.0
    ce_level[idx_bear_final] = (bearish_upper[idx_bear_final] + bearish_lower[idx_bear_final]) / 2.0

    # Mitigation — scalable: single Numba pass for all thresholds if available
    if compute_mitigation:
        mitigated = np.zeros(n, dtype=bool)
        mitigated_index = np.full(n, -1, dtype=np.int64)
        mitigated_50 = np.zeros(n, dtype=bool)
        mitigated_50_index = np.full(n, -1, dtype=np.int64)
        mitigated_full = np.zeros(n, dtype=bool)
        mitigated_full_index = np.full(n, -1, dtype=np.int64)
        inverted = np.zeros(n, dtype=bool)
        bull_idx = np.where(bullish)[0].astype(np.int64)
        bear_idx = np.where(bearish)[0].astype(np.int64)
        cl_arr = np.full(n, np.nan, dtype=np.float64) if close is None else _to_float64(close)
        if _HAS_NUMBA and n > 0 and (bull_idx.size > 0 or bear_idx.size > 0):
            # Numba single scan for all thresholds (mit, CE50, full, inverted) — O(n + g*avg_dist) with early break
            if bull_idx.size > 0:
                _fvg_first_idx_three_numba(
                    n, bull_idx, bullish_upper, ce_level, bullish_lower, lo, h, cl_arr,
                    mitigated, mitigated_index, mitigated_50, mitigated_50_index, mitigated_full, mitigated_full_index, inverted, True,
                )
            if bear_idx.size > 0:
                _fvg_first_idx_three_numba(
                    n, bear_idx, bearish_lower, ce_level, bearish_upper, lo, h, cl_arr,
                    mitigated, mitigated_index, mitigated_50, mitigated_50_index, mitigated_full, mitigated_full_index, inverted, False,
                )
        else:
            # Fallback: Python per-gap with reverse-cum flag + 1-D scan (still O(g*n) but low memory)
            # Use existing helpers but share flag logic to avoid triple work
            mitigated, mitigated_index = _compute_mitigation_vectorized(
                h, lo, bullish, bearish, bullish_upper, bullish_lower, bearish_upper, bearish_lower
            )
            mitigated_50, mitigated_50_index = _compute_threshold_mitigation(
                h, lo, cl_arr, bullish, bearish, ce_level, ce_level, "low_le", "high_ge"
            )
            mitigated_full, mitigated_full_index = _compute_threshold_mitigation(
                h, lo, cl_arr, bullish, bearish, bullish_lower, bearish_upper, "low_le", "high_ge"
            )
            if close is not None:
                for idx in np.where(bullish)[0]:
                    if mitigated_full[idx]:
                        mi = int(mitigated_full_index[idx])
                        if mi != -1 and mi < n and cl_arr[mi] < bullish_lower[idx]:
                            inverted[idx] = True
                for idx in np.where(bearish)[0]:
                    if mitigated_full[idx]:
                        mi = int(mitigated_full_index[idx])
                        if mi != -1 and mi < n and cl_arr[mi] > bearish_upper[idx]:
                            inverted[idx] = True
            else:
                inverted = mitigated_full.copy()
    else:
        mitigated = np.zeros(n, dtype=bool)
        mitigated_index = np.full(n, -1, dtype=np.int64)
        mitigated_50 = np.zeros(n, dtype=bool)
        mitigated_50_index = np.full(n, -1, dtype=np.int64)
        mitigated_full = np.zeros(n, dtype=bool)
        mitigated_full_index = np.full(n, -1, dtype=np.int64)
        inverted = np.zeros(n, dtype=bool)

    return FVGResult(
        bullish=bullish,
        bearish=bearish,
        bullish_upper=bullish_upper,
        bullish_lower=bullish_lower,
        bearish_upper=bearish_upper,
        bearish_lower=bearish_lower,
        gap_size=gap_size,
        gap_size_pct=gap_size_pct,
        ce_level=ce_level,
        mitigated=mitigated,
        mitigated_index=mitigated_index,
        mitigated_50=mitigated_50,
        mitigated_50_index=mitigated_50_index,
        mitigated_full=mitigated_full,
        mitigated_full_index=mitigated_full_index,
        inverted=inverted,
    )


# ---------------------------------------------------------------------------
# Polars helper — kept here for convenience; primary Polars API lives in
# polars_ext.py. This function is *not* imported by the package __init__ to
# avoid mandatory Polars dependency at import time.
# ---------------------------------------------------------------------------


def fvg_polars(
    df: object,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str | None = None,
    min_gap_size: float | None = None,
    min_gap_size_pct: float | None = None,
    compute_mitigation: bool = False,
) -> object:
    """Polars DataFrame version of :func:`detect_fvg`.

    Adds columns ``fvg_bullish``, ``fvg_bearish``, ``fvg_bullish_upper``,
    ``fvg_bullish_lower``, ``fvg_bearish_upper``, ``fvg_bearish_lower``,
    ``fvg_gap_size`` to the DataFrame.

    Args:
        df: ``polars.DataFrame`` with ``high_col`` and ``low_col`` columns.
        high_col: Name of the high column.
        low_col: Name of the low column.
        min_gap_size: Absolute gap filter.
        min_gap_size_pct: Percentage gap filter (fraction).
        compute_mitigation: Whether to add ``fvg_mitigated`` / ``fvg_mitigated_index``.

    Returns:
        New ``polars.DataFrame`` with FVG columns appended.
    """
    try:
        import polars as pl  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("polars is required for fvg_polars; install with `pip install polars`") from e

    if isinstance(df, pl.LazyFrame):
        # Native Lazy Expr path (streaming, no numpy conversion) — supports basic FVG without mitigation
        # For mitigation, fall back to collect (requires future scan)
        if compute_mitigation or min_gap_size is not None or min_gap_size_pct is not None:
            df_collected = df.collect()
            return fvg_polars(df_collected, high_col=high_col, low_col=low_col, close_col=close_col, min_gap_size=min_gap_size, min_gap_size_pct=min_gap_size_pct, compute_mitigation=compute_mitigation)
        # Pure lazy: low > high.shift(2) and high < low.shift(2)
        bullish_expr = (pl.col(low_col) > pl.col(high_col).shift(2)).fill_null(False)
        bearish_expr = (pl.col(high_col) < pl.col(low_col).shift(2)).fill_null(False)
        # gap size expr not needed for filter (already handled)
        return df.with_columns([
            bullish_expr.alias("fvg_bullish"),
            bearish_expr.alias("fvg_bearish"),
            pl.when(bullish_expr).then(pl.col(low_col)).otherwise(None).alias("fvg_bullish_upper"),
            pl.when(bullish_expr).then(pl.col(high_col).shift(2)).otherwise(None).alias("fvg_bullish_lower"),
            pl.when(bearish_expr).then(pl.col(low_col).shift(2)).otherwise(None).alias("fvg_bearish_upper"),
            pl.when(bearish_expr).then(pl.col(high_col)).otherwise(None).alias("fvg_bearish_lower"),
            pl.when(bullish_expr | bearish_expr).then((pl.col(low_col) - pl.col(high_col).shift(2)).abs()).otherwise(None).alias("fvg_gap_size"),
            pl.lit(None).alias("fvg_gap_size_pct"),
            pl.when(bullish_expr | bearish_expr).then((pl.col(low_col) + pl.col(high_col).shift(2))/2).otherwise(None).alias("fvg_ce_level"),
        ])

    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected polars.DataFrame, got {type(df)}")

    high = df[high_col].to_numpy().astype(float)
    low = df[low_col].to_numpy().astype(float)
    close = None
    if close_col is not None and close_col in df.columns:
        close = df[close_col].to_numpy().astype(float)
    res = detect_fvg(high, low, min_gap_size=min_gap_size, min_gap_size_pct=min_gap_size_pct, compute_mitigation=compute_mitigation, close=close)

    out = df.with_columns(
        [
            pl.Series("fvg_bullish", res.bullish),
            pl.Series("fvg_bearish", res.bearish),
            pl.Series("fvg_bullish_upper", res.bullish_upper),
            pl.Series("fvg_bullish_lower", res.bullish_lower),
            pl.Series("fvg_bearish_upper", res.bearish_upper),
            pl.Series("fvg_bearish_lower", res.bearish_lower),
            pl.Series("fvg_gap_size", res.gap_size),
            pl.Series("fvg_gap_size_pct", res.gap_size_pct),
            pl.Series("fvg_ce_level", res.ce_level),
        ]
    )
    if compute_mitigation:
        out = out.with_columns(
            [
                pl.Series("fvg_mitigated", res.mitigated),
                pl.Series("fvg_mitigated_index", res.mitigated_index),
                pl.Series("fvg_mitigated_50", res.mitigated_50),
                pl.Series("fvg_mitigated_50_index", res.mitigated_50_index),
                pl.Series("fvg_mitigated_full", res.mitigated_full),
                pl.Series("fvg_mitigated_full_index", res.mitigated_full_index),
                pl.Series("fvg_inverted", res.inverted),
            ]
        )
    return out
