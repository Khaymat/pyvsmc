"""Fractal swing highs and lows — fully vectorized rolling-window detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class SwingResult:
    """Container for swing detection results.

    All array attributes have shape ``(n,)``.

    Attributes:
        swing_high: Boolean mask where ``High[i]`` is the maximum in
            ``[i-window_size, i+window_size]``.
        swing_low: Boolean mask where ``Low[i]`` is the minimum in the same window.
        swing_high_price: Price level of swing highs (``High[i]`` where mask True,
            ``np.nan`` elsewhere).
        swing_low_price: Price level of swing lows (``Low[i]`` where mask True,
            ``np.nan`` elsewhere).
        window_size: The window size ``N`` used for detection.
    """

    swing_high: npt.NDArray[np.bool_]
    swing_low: npt.NDArray[np.bool_]
    swing_high_price: npt.NDArray[np.float64]
    swing_low_price: npt.NDArray[np.float64]
    window_size: int


def _to_float64(arr: npt.ArrayLike) -> npt.NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {a.shape}")
    return a


def detect_swings(
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    window_size: int = 2,
) -> SwingResult:
    """Detect fractal swing highs and lows — fully vectorized.

    A bar ``i`` is a **swing high** iff::

        High[i] == max(High[i-N : i+N+1])

    and a **swing low** iff::

        Low[i] == min(Low[i-N : i+N+1])

    where ``N = window_size``.  Bars within ``N`` of the edges cannot be
    swings because the window would be incomplete.  ``NaN`` values in the
    window invalidate the swing.

    The implementation uses ``numpy.lib.stride_tricks.sliding_window_view``
    for O(n) vectorized windowed max/min without Python loops.

    Args:
        high: Array of bar highs, shape ``(n,)``.
        low: Array of bar lows, shape ``(n,)``.
        window_size: Number of bars to look left *and* right (``N``).
            Must be ``>= 1``.  Total window width is ``2*N+1``.

    Returns:
        :class:`SwingResult` with boolean masks and price arrays.

    Raises:
        ValueError: If inputs have different lengths, are not 1-D, or if
            ``window_size < 1``.

    Examples:
        >>> import numpy as np
        >>> high = np.array([1., 3., 2., 5., 4.])
        >>> low  = np.array([0., 1., 0.5, 2., 1.])
        >>> res = detect_swings(high, low, window_size=1)
        >>> res.swing_high
        array([False,  True, False,  True, False])
    """
    h = _to_float64(high)
    lo = _to_float64(low)

    if h.shape[0] != lo.shape[0]:
        raise ValueError(f"high and low must have same length: {h.shape[0]} vs {lo.shape[0]}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    n = h.shape[0]
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    swing_high_price = np.full(n, np.nan, dtype=np.float64)
    swing_low_price = np.full(n, np.nan, dtype=np.float64)

    if n < 2 * window_size + 1:
        return SwingResult(
            swing_high=swing_high,
            swing_low=swing_low,
            swing_high_price=swing_high_price,
            swing_low_price=swing_low_price,
            window_size=window_size,
        )

    width = 2 * window_size + 1

    # sliding_window_view returns shape (n - width + 1, width)
    # window i corresponds to original indices [i, i+width) -> center = i+window_size
    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except ImportError as exc:
        # fallback for very old numpy (should not happen with >=1.24)
        raise ImportError("numpy.lib.stride_tricks.sliding_window_view requires numpy>=1.20") from exc

    # High windows
    high_windows = sliding_window_view(h, window_shape=width)  # (n-width+1, width)
    low_windows = sliding_window_view(lo, window_shape=width)

    # Valid windows: no NaN in window
    high_valid = ~np.isnan(high_windows).any(axis=1)
    low_valid = ~np.isnan(low_windows).any(axis=1)

    # Center value of each window
    # high_windows[i] corresponds to center index c = i + window_size
    # Check if center equals max of window
    # Use vectorized max/min
    high_max = np.nanmax(high_windows, axis=1)  # but we already filtered NaN windows
    low_min = np.nanmin(low_windows, axis=1)

    center_high = h[window_size : n - window_size]
    center_low = lo[window_size : n - window_size]

    # Swing high: center == window max
    # Use exact equality; tolerance not applied to preserve strict fractal definition.
    # For floating point robustness we check with == (as per spec High[i] == max(...))
    high_is_max = center_high == high_max
    low_is_min = center_low == low_min

    # Combine with validity
    high_swing_window = high_is_max & high_valid
    low_swing_window = low_is_min & low_valid

    # Map back to original indices: window index i -> original index i+window_size
    swing_high[window_size : n - window_size] = high_swing_window
    swing_low[window_size : n - window_size] = low_swing_window

    # Price levels
    swing_high_price[swing_high] = h[swing_high]
    swing_low_price[swing_low] = lo[swing_low]

    return SwingResult(
        swing_high=swing_high,
        swing_low=swing_low,
        swing_high_price=swing_high_price,
        swing_low_price=swing_low_price,
        window_size=window_size,
    )


def swings_polars(
    df: object,
    *,
    high_col: str = "high",
    low_col: str = "low",
    window_size: int = 2,
) -> object:
    """Polars DataFrame version of :func:`detect_swings`.

    Adds columns ``swing_high``, ``swing_low``, ``swing_high_price``,
    ``swing_low_price``.

    Args:
        df: ``polars.DataFrame``.
        high_col: Name of high column.
        low_col: Name of low column.
        window_size: Window size ``N``.

    Returns:
        New ``polars.DataFrame`` with swing columns appended.
    """
    try:
        import polars as pl  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("polars is required for swings_polars") from e

    if isinstance(df, pl.LazyFrame):
        # Native lazy: rolling max/min centered
        width = 2 * window_size + 1
        # polars rolling uses window_size, need center: shift
        # For swing high: high == rolling_max centered
        swing_high_expr = (pl.col(high_col) == pl.col(high_col).rolling_max(window_size=width, min_samples=width, center=True)).fill_null(False)
        swing_low_expr = (pl.col(low_col) == pl.col(low_col).rolling_min(window_size=width, min_samples=width, center=True)).fill_null(False)
        return df.with_columns([
            swing_high_expr.alias("swing_high"),
            swing_low_expr.alias("swing_low"),
            pl.when(swing_high_expr).then(pl.col(high_col)).otherwise(None).alias("swing_high_price"),
            pl.when(swing_low_expr).then(pl.col(low_col)).otherwise(None).alias("swing_low_price"),
        ])

    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected polars.DataFrame, got {type(df)}")

    high = df[high_col].to_numpy().astype(float)
    low = df[low_col].to_numpy().astype(float)
    res = detect_swings(high, low, window_size=window_size)
    return df.with_columns(
        [
            pl.Series("swing_high", res.swing_high),
            pl.Series("swing_low", res.swing_low),
            pl.Series("swing_high_price", res.swing_high_price),
            pl.Series("swing_low_price", res.swing_low_price),
        ]
    )
