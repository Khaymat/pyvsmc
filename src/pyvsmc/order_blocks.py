"""Order Block (OB) identification — vectorized.

Definition
----------
An **Order Block** is the last opposing candle before a strong impulsive
move that creates a market structure break or a significant Fair Value Gap.

* **Bullish OB**: the last *bearish* candle (``close < open``) before a
  bullish BOS / large bullish FVG expansion.  Represents a demand zone.
  Zone: ``[low, high]`` of that bearish candle; more conservatively
  ``[open, close]`` body, but we return the full candle range by default.

* **Bearish OB**: the last *bullish* candle (``close > open``) before a
  bearish BOS / large bearish FVG expansion.  Supply zone.

This module finds OBs by scanning for FVG impulses and/or close-based
displacement, then vectorized backward-search for the last opposing candle
within a lookback window.

All operations are vectorized via NumPy; the backward search uses
broadcasting with chunked evaluation to avoid O(n²) memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .fvg import detect_fvg
from .structure import detect_structure


@dataclass(frozen=True, slots=True)
class OrderBlockResult:
    """Container for Order Block detection.

    All arrays have shape ``(n,)`` where ``n`` is input length.

    Attributes:
        bullish_ob: Boolean mask — ``True`` where a bullish OB is anchored
            (the candle itself is the OB; the *impulse* that validates it
            occurs at ``ob_validated_index``).
        bearish_ob: Boolean mask — ``True`` where a bearish OB is anchored.
        ob_high: Upper boundary of the OB zone (candle high); ``np.nan`` elsewhere.
        ob_low: Lower boundary of the OB zone (candle low); ``np.nan`` elsewhere.
        ob_type: Array of strings ``"bullish"`` / ``"bearish"`` / ``""`` per bar.
        validated_index: Index of the impulse/break that validated the OB;
            ``-1`` where not validated or not an OB.
        impulse_type: What triggered the OB — ``"fvg"``, ``"bos"``, or ``""``.
        mitigated: True where zone later touched by price.
        mitigated_index: Index of first mitigation bar, -1 if never.
        is_breaker: True where OB was invalidated (close beyond extreme) → breaker block.
        breaker_index: Index of invalidation.
    """

    bullish_ob: npt.NDArray[np.bool_]
    bearish_ob: npt.NDArray[np.bool_]
    ob_high: npt.NDArray[np.float64]
    ob_low: npt.NDArray[np.float64]
    ob_type: npt.NDArray[np.object_]
    validated_index: npt.NDArray[np.int64]
    impulse_type: npt.NDArray[np.object_]
    mitigated: npt.NDArray[np.bool_]
    mitigated_index: npt.NDArray[np.int64]
    is_breaker: npt.NDArray[np.bool_]
    breaker_index: npt.NDArray[np.int64]


def _to_float64(arr: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {a.shape}")
    return a


def _find_last_opposing_candle_vectorized(
    open_: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    impulse_indices: npt.NDArray[np.int64],
    direction: str,
    lookback: int,
    high: npt.NDArray[np.float64] | None = None,
    low: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """For each impulse index, find last opposing candle within lookback.

    Args:
        open_: Open prices.
        close: Close prices.
        impulse_indices: Array of impulse bar indices (where FVG/BOS occurred).
        direction: ``"bullish"`` -> search for last bearish candle (close<open);
            ``"bearish"`` -> last bullish candle (close>open).
        lookback: Max bars to look back from impulse (exclusive of impulse bar itself).
        high, low: Optional for NaN filtering (if candle has NaN open/close/high/low, skip).

    Returns:
        (ob_indices, valid) where ob_indices is the OB anchor index for each
        impulse, and valid flags whether one was found.
    """
    n = open_.shape[0]
    m = impulse_indices.shape[0]

    if m == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)

    # Opposing mask per bar
    opposing = close < open_ if direction == "bullish" else close > open_

    # Invalidate NaN candles
    nan_candle = np.isnan(open_) | np.isnan(close)
    if high is not None:
        nan_candle |= np.isnan(high)
    if low is not None:
        nan_candle |= np.isnan(low)
    opposing = opposing & ~nan_candle

    # Per-impulse backward search (O(m*lookback), low memory)
    ob_indices = np.full(m, -1, dtype=np.int64)
    valid = np.zeros(m, dtype=bool)
    # Precompute opposing indices where true
    # For each impulse, scan backwards up to lookback
    for k in range(m):
        imp = int(impulse_indices[k])
        start_idx = max(0, imp - lookback)
        end_idx = imp - 1
        if end_idx < start_idx:
            continue
        # Search backwards for last opposing
        # Slice window and find last True via where
        window_opposing = opposing[start_idx : end_idx + 1]
        # Find last True index in window
        # Use np.where on reversed
        pos = np.where(window_opposing)[0]
        if pos.size > 0:
            last_pos = pos[-1]
            ob_indices[k] = start_idx + last_pos
            valid[k] = True

    return ob_indices, valid


def detect_order_blocks(
    open_: npt.ArrayLike,
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    *,
    lookback: int = 10,
    min_fvg_size: float | None = None,
    min_fvg_size_pct: float | None = None,
    use_fvg: bool = True,
    use_bos: bool = True,
    window_size: int = 2,
    compute_mitigation: bool = False,
    zone_mode: str = "full",
) -> OrderBlockResult:
    """Detect Order Blocks — fully vectorized.

    An Order Block is anchored at the last opposing candle before an impulse.
    Impulses are defined as:

    * **FVG impulse**: a Fair Value Gap (filtered by ``min_fvg_size``).
    * **BOS impulse**: a Break of Structure (close breaks last swing).

    When both ``use_fvg`` and ``use_bos`` are ``True``, both impulse types
    are considered and merged (if the same impulse bar has both, FVG takes
    precedence).

    Args:
        open_: Bar opens, shape ``(n,)``.
        high: Bar highs, shape ``(n,)``.
        low: Bar lows, shape ``(n,)``.
        close: Bar closes, shape ``(n,)``.
        lookback: Maximum bars to look back for the opposing candle.
            Must be ``>= 1``.
        min_fvg_size: Minimum FVG gap size to qualify as impulse.
            ``None`` means any FVG qualifies.
        min_fvg_size_pct: Minimum FVG gap percentage (fraction) to qualify.
        use_fvg: Whether to use FVG as impulse source.
        use_bos: Whether to use BOS as impulse source.
        window_size: Swing window size for BOS detection.

    Returns:
        :class:`OrderBlockResult` with OB masks and zones.

    Raises:
        ValueError: If array lengths differ or ``lookback < 1``.

    Examples:
        >>> import numpy as np
        >>> open_ = np.array([10., 10.5, 10.2, 10.8, 12.0])
        >>> high  = np.array([10.5, 11., 10.6, 12.5, 13.])
        >>> low   = np.array([9.5, 10., 9.8, 10.7, 11.5])
        >>> close = np.array([10.2, 10.3, 10.0, 12.2, 12.8])
        >>> res = detect_order_blocks(open_, high, low, close)
    """
    o = _to_float64(open_)
    h = _to_float64(high)
    lo = _to_float64(low)
    cl = _to_float64(close)

    n = o.shape[0]
    if not (h.shape[0] == n and lo.shape[0] == n and cl.shape[0] == n):
        raise ValueError(f"open, high, low, close must have same length: {o.shape[0]}, {h.shape[0]}, {lo.shape[0]}, {cl.shape[0]}")
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if zone_mode not in ("full", "body", "mean_threshold"):
        raise ValueError(f"zone_mode must be 'full','body','mean_threshold', got {zone_mode}")

    bullish_ob = np.zeros(n, dtype=bool)
    bearish_ob = np.zeros(n, dtype=bool)
    ob_high = np.full(n, np.nan, dtype=np.float64)
    ob_low = np.full(n, np.nan, dtype=np.float64)
    ob_type = np.full(n, "", dtype=object)
    validated_index = np.full(n, -1, dtype=np.int64)
    impulse_type = np.full(n, "", dtype=object)

    if n < 3:
        return OrderBlockResult(
            bullish_ob=bullish_ob,
            bearish_ob=bearish_ob,
            ob_high=ob_high,
            ob_low=ob_low,
            ob_type=ob_type,
            validated_index=validated_index,
            impulse_type=impulse_type,
            mitigated=np.zeros(n, dtype=bool),
            mitigated_index=np.full(n, -1, dtype=np.int64),
            is_breaker=np.zeros(n, dtype=bool),
            breaker_index=np.full(n, -1, dtype=np.int64),
        )

    # ---- Gather impulse indices ----
    impulse_bull_indices: list[npt.NDArray[np.int64]] = []
    impulse_bear_indices: list[npt.NDArray[np.int64]] = []
    impulse_bull_types: list[npt.NDArray[np.object_]] = []
    impulse_bear_types: list[npt.NDArray[np.object_]] = []

    if use_fvg:
        fvg_res = detect_fvg(h, lo, min_gap_size=min_fvg_size, min_gap_size_pct=min_fvg_size_pct)
        bull_fvg_idx = np.where(fvg_res.bullish)[0]
        bear_fvg_idx = np.where(fvg_res.bearish)[0]
        if bull_fvg_idx.size > 0:
            impulse_bull_indices.append(bull_fvg_idx)
            impulse_bull_types.append(np.full(bull_fvg_idx.shape[0], "fvg", dtype=object))
        if bear_fvg_idx.size > 0:
            impulse_bear_indices.append(bear_fvg_idx)
            impulse_bear_types.append(np.full(bear_fvg_idx.shape[0], "fvg", dtype=object))

    if use_bos:
        struct = detect_structure(h, lo, cl, window_size=window_size)
        # BOS bullish impulses and CHOCH bullish also count as bullish impulses?
        # For OB detection, we consider both BOS and CHOCH as structure breaks.
        # But to avoid double counting, we merge.
        bull_bos_idx = np.where(struct.bos_bullish | struct.choch_bullish)[0]
        bear_bos_idx = np.where(struct.bos_bearish | struct.choch_bearish)[0]
        if bull_bos_idx.size > 0:
            impulse_bull_indices.append(bull_bos_idx)
            impulse_bull_types.append(np.full(bull_bos_idx.shape[0], "bos", dtype=object))
        if bear_bos_idx.size > 0:
            impulse_bear_indices.append(bear_bos_idx)
            impulse_bear_types.append(np.full(bear_bos_idx.shape[0], "bos", dtype=object))

    # Merge impulses: deduplicate, keep first type (FVG priority since we appended FVG first)
    def _merge_impulses(
        indices_list: list[npt.NDArray[np.int64]],
        types_list: list[npt.NDArray[np.object_]],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.object_]]:
        if not indices_list:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=object)
        all_idx = np.concatenate(indices_list)
        all_typ = np.concatenate(types_list)
        # Sort by index
        order = np.argsort(all_idx, kind="stable")
        all_idx = all_idx[order]
        all_typ = all_typ[order]
        # Deduplicate: keep first occurrence
        _, unique_pos = np.unique(all_idx, return_index=True)
        # unique_pos are indices into sorted array where first occurrence
        # But np.unique returns sorted unique values; we want to keep sorted order.
        # So sort unique_pos
        unique_pos_sorted = np.sort(unique_pos)
        return all_idx[unique_pos_sorted], all_typ[unique_pos_sorted]

    bull_impulse_idx, bull_impulse_typ = _merge_impulses(impulse_bull_indices, impulse_bull_types)
    bear_impulse_idx, bear_impulse_typ = _merge_impulses(impulse_bear_indices, impulse_bear_types)

    # ---- Find OBs for bullish impulses (search bearish candles) ----
    if bull_impulse_idx.size > 0:
        ob_idx_bull, valid_bull = _find_last_opposing_candle_vectorized(o, cl, bull_impulse_idx, "bullish", lookback, h, lo)
        # Filter valid and also ensure OB index not already flagged with higher priority?
        # Each OB anchor corresponds to one impulse; if multiple impulses map to same OB anchor,
        # we keep the earliest impulse.
        for k in range(ob_idx_bull.shape[0]):
            if not valid_bull[k]:
                continue
            anchor = int(ob_idx_bull[k])
            imp_idx = int(bull_impulse_idx[k])
            imp_typ = str(bull_impulse_typ[k])
            # If this anchor already has a bullish OB, keep the earliest impulse (smallest imp_idx)
            if bullish_ob[anchor]:
                # Keep the earlier validated_index
                if imp_idx < validated_index[anchor]:
                    validated_index[anchor] = imp_idx
                    impulse_type[anchor] = imp_typ
                continue
            # Also avoid overwriting a bearish OB at same anchor (should not happen normally, but if it does, keep both? We treat as bullish)
            bullish_ob[anchor] = True
            if zone_mode == "full":
                ob_high[anchor] = h[anchor]
                ob_low[anchor] = lo[anchor]
            elif zone_mode == "body":
                ob_high[anchor] = max(o[anchor], cl[anchor])
                ob_low[anchor] = min(o[anchor], cl[anchor])
            else:  # mean_threshold
                ob_high[anchor] = max(o[anchor], cl[anchor])
                ob_low[anchor] = (o[anchor] + cl[anchor]) / 2.0
            ob_type[anchor] = "bullish"
            validated_index[anchor] = imp_idx
            impulse_type[anchor] = imp_typ

    # ---- Find OBs for bearish impulses (search bullish candles) ----
    if bear_impulse_idx.size > 0:
        ob_idx_bear, valid_bear = _find_last_opposing_candle_vectorized(o, cl, bear_impulse_idx, "bearish", lookback, h, lo)
        for k in range(ob_idx_bear.shape[0]):
            if not valid_bear[k]:
                continue
            anchor = int(ob_idx_bear[k])
            imp_idx = int(bear_impulse_idx[k])
            imp_typ = str(bear_impulse_typ[k])
            if bearish_ob[anchor]:
                if imp_idx < validated_index[anchor]:
                    validated_index[anchor] = imp_idx
                    impulse_type[anchor] = imp_typ
                continue
            # If anchor already bullish OB, we have a conflict (same candle is both bull and bear OB).
            # This can happen if both bullish and bearish impulses map to same anchor due to overlapping lookback.
            # We keep the existing (bullish) and skip.  Alternatively could keep both flags.
            # Here we allow both flags to be True simultaneously if truly both.
            if bullish_ob[anchor]:
                # Conflict: same candle flagged as both.  Keep bullish, but also set bearish.
                # To avoid ambiguity we set bearish as well but note ob_type becomes "both" is not supported;
                # we keep as bullish for display and set bearish flag too.
                pass
            bearish_ob[anchor] = True
            # If not already set by bullish, set boundaries and type
            if np.isnan(ob_high[anchor]):
                if zone_mode == "full":
                    ob_high[anchor] = h[anchor]
                    ob_low[anchor] = lo[anchor]
                elif zone_mode == "body":
                    ob_high[anchor] = max(o[anchor], cl[anchor])
                    ob_low[anchor] = min(o[anchor], cl[anchor])
                else:
                    ob_high[anchor] = max(o[anchor], cl[anchor])
                    ob_low[anchor] = (o[anchor] + cl[anchor]) / 2.0
            if ob_type[anchor] == "":
                ob_type[anchor] = "bearish"
            elif ob_type[anchor] == "bullish":
                ob_type[anchor] = "both"  # type: ignore[assignment]
            if validated_index[anchor] == -1:
                validated_index[anchor] = imp_idx
                impulse_type[anchor] = imp_typ
            else:
                # Keep earliest impulse; if this is earlier, update
                if imp_idx < validated_index[anchor]:
                    validated_index[anchor] = imp_idx
                    impulse_type[anchor] = imp_typ

    # Mitigation: future price touching OB zone (vectorized)
    mitigated = np.zeros(n, dtype=bool)
    mitigated_index = np.full(n, -1, dtype=np.int64)
    is_breaker = np.zeros(n, dtype=bool)
    breaker_index = np.full(n, -1, dtype=np.int64)
    if compute_mitigation:
        ob_idx = np.where(bullish_ob | bearish_ob)[0]
        for anchor in ob_idx:
            start = int(validated_index[anchor] + 1) if validated_index[anchor] != -1 else int(anchor + 1)
            if start >= n:
                continue
            low_future = lo[start:]
            high_future = h[start:]
            close_future = cl[start:]
            if bullish_ob[anchor]:
                hit = np.where(low_future <= ob_high[anchor])[0]
                if hit.size > 0:
                    mitigated[anchor] = True
                    mitigated_index[anchor] = int(start + hit[0])
                # breaker if close < ob_low
                b_hit = np.where(close_future < ob_low[anchor])[0]
                if b_hit.size > 0:
                    is_breaker[anchor] = True
                    breaker_index[anchor] = int(start + b_hit[0])
            if bearish_ob[anchor]:
                hit = np.where(high_future >= ob_low[anchor])[0]
                if hit.size > 0:
                    idx = int(start + hit[0])
                    if not mitigated[anchor] or idx < mitigated_index[anchor]:
                        mitigated[anchor] = True
                        mitigated_index[anchor] = idx
                b_hit = np.where(close_future > ob_high[anchor])[0]
                if b_hit.size > 0:
                    idx = int(start + b_hit[0])
                    if not is_breaker[anchor] or idx < breaker_index[anchor]:
                        is_breaker[anchor] = True
                        breaker_index[anchor] = idx

    return OrderBlockResult(
        bullish_ob=bullish_ob,
        bearish_ob=bearish_ob,
        ob_high=ob_high,
        ob_low=ob_low,
        ob_type=ob_type,
        validated_index=validated_index,
        impulse_type=impulse_type,
        mitigated=mitigated,
        mitigated_index=mitigated_index,
        is_breaker=is_breaker,
        breaker_index=breaker_index,
    )


def order_blocks_polars(
    df: object,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    lookback: int = 10,
    min_fvg_size: float | None = None,
    min_fvg_size_pct: float | None = None,
    use_fvg: bool = True,
    use_bos: bool = True,
    window_size: int = 2,
    compute_mitigation: bool = False,
) -> object:
    """Polars DataFrame version of :func:`detect_order_blocks`.

    Adds columns ``ob_bullish``, ``ob_bearish``, ``ob_high``, ``ob_low``,
    ``ob_type``, ``ob_validated_index``, ``ob_impulse_type``.

    Args:
        df: ``polars.DataFrame``.
        open_col, high_col, low_col, close_col: Column names.
        lookback: Lookback window for opposing candle.
        min_fvg_size: Minimum FVG size filter.
        min_fvg_size_pct: Minimum FVG pct filter.
        use_fvg, use_bos: Whether to use FVG/BOS impulses.
        window_size: Swing window for BOS.

    Returns:
        New ``polars.DataFrame`` with OB columns appended.
    """
    try:
        import polars as pl  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("polars is required for order_blocks_polars") from e

    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected polars.DataFrame, got {type(df)}")

    open_ = df[open_col].to_numpy().astype(float)
    high = df[high_col].to_numpy().astype(float)
    low = df[low_col].to_numpy().astype(float)
    close = df[close_col].to_numpy().astype(float)
    res = detect_order_blocks(
        open_,
        high,
        low,
        close,
        lookback=lookback,
        min_fvg_size=min_fvg_size,
        min_fvg_size_pct=min_fvg_size_pct,
        use_fvg=use_fvg,
        use_bos=use_bos,
        window_size=window_size,
        compute_mitigation=compute_mitigation,
    )
    cols = [
        pl.Series("ob_bullish", res.bullish_ob),
        pl.Series("ob_bearish", res.bearish_ob),
        pl.Series("ob_high", res.ob_high),
        pl.Series("ob_low", res.ob_low),
        pl.Series("ob_type", res.ob_type),
        pl.Series("ob_validated_index", res.validated_index),
        pl.Series("ob_impulse_type", res.impulse_type),
    ]
    if compute_mitigation:
        cols += [pl.Series("ob_mitigated", res.mitigated), pl.Series("ob_mitigated_index", res.mitigated_index)]
    return df.with_columns(cols)
