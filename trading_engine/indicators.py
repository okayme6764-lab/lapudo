"""Technical indicator calculations for XAUUSD scalping.

All functions are purely causal: the value at any point depends only on past
and present data (no lookahead), so they are safe to use in backtests.

Every function validates its inputs and raises descriptive exceptions on bad
input rather than silently returning garbage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_series(series: pd.Series, name: str) -> pd.Series:
    """Ensure ``series`` is a non-empty numeric pandas Series."""
    if not isinstance(series, pd.Series):
        raise TypeError(
            f"{name} must be a pandas Series, got {type(series).__name__}"
        )
    if series.empty:
        raise ValueError(f"{name} is empty")
    if series.isna().all():
        raise ValueError(f"{name} contains only NaN values")
    try:
        out = series.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} could not be converted to float: {exc}") from exc
    return out


def _check_period(period: int, name: str) -> None:
    if not isinstance(period, int):
        raise TypeError(f"{name} must be an int, got {type(period).__name__}")
    if period < 1:
        raise ValueError(f"{name} must be >= 1, got {period}")


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average.

    Uses ``adjust=False`` so the EMA is a recursive filter over all past data.
    """
    _check_period(period, "period")
    s = validate_series(series, "series")
    return s.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    - RSI = 100 when average loss is zero (strong uptrend).
    - RSI = 0 when average gain is zero (strong downtrend).
    - RSI = 50 when there is no movement at all.
    """
    _check_period(period, "period")
    s = validate_series(series, "series")

    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero: where avg_loss == 0 the RS is undefined.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))

    # avg_loss == 0 (and avg_gain > 0) -> RSI = 100
    out = out.where(avg_loss != 0.0, 100.0)
    # both averages zero (flat market) -> RSI = 50
    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    out = out.where(~both_zero, 50.0)

    return out.fillna(50.0)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence / Divergence.

    Returns ``(macd_line, signal_line, histogram)`` where the histogram is the
    difference between the MACD line and its signal line.
    """
    _check_period(fast, "fast")
    _check_period(slow, "slow")
    _check_period(signal, "signal")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")

    s = validate_series(series, "series")
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range using Wilder's smoothing.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    """
    _check_period(period, "period")
    h = validate_series(high, "high")
    l = validate_series(low, "low")
    c = validate_series(close, "close")

    prev_close = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_close).abs(), (l - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1.0 / period, adjust=False).mean()
