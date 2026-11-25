"""Candlestick pattern detection for entry confirmation.

Every pattern function takes a DataFrame with ``open``/``high``/``low``/
``close`` columns and returns a boolean Series aligned to the DataFrame index
(True where the pattern is present). ``candle_signal`` collapses the patterns
into a single directional score used by the strategy.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

_REQUIRED_COLUMNS = ("open", "high", "low", "close")


def _validate_candles(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if df.empty:
        raise ValueError("candlestick DataFrame is empty")


def _candle_parts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (body, upper_wick, lower_wick) absolute sizes."""
    o, h, l, c = (df["open"], df["high"], df["low"], df["close"])
    body = (c - o).abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    return body, upper, lower


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish engulfing: bearish candle fully engulfed by a bullish one."""
    _validate_candles(df)
    prev_bearish = df["close"].shift(1) < df["open"].shift(1)
    curr_bullish = df["close"] > df["open"]
    engulfs = (df["close"] >= df["open"].shift(1)) & (
        df["open"] <= df["close"].shift(1)
    )
    return prev_bearish & curr_bullish & engulfs


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Bearish engulfing: bullish candle fully engulfed by a bearish one."""
    _validate_candles(df)
    prev_bullish = df["close"].shift(1) > df["open"].shift(1)
    curr_bearish = df["close"] < df["open"]
    engulfs = (df["close"] <= df["open"].shift(1)) & (
        df["open"] >= df["close"].shift(1)
    )
    return prev_bullish & curr_bearish & engulfs


def hammer(df: pd.DataFrame) -> pd.Series:
    """Hammer: long lower wick, small body near the top of the range."""
    _validate_candles(df)
    body, upper, lower = _candle_parts(df)
    rng = df["high"] - df["low"]
    small_body = body <= 0.30 * rng
    long_lower = lower >= 2.0 * body
    small_upper = upper <= 0.30 * rng
    return small_body & long_lower & small_upper


def shooting_star(df: pd.DataFrame) -> pd.Series:
    """Shooting star: long upper wick, small body near the bottom."""
    _validate_candles(df)
    body, upper, lower = _candle_parts(df)
    rng = df["high"] - df["low"]
    small_body = body <= 0.30 * rng
    long_upper = upper >= 2.0 * body
    small_lower = lower <= 0.30 * rng
    return small_body & long_upper & small_lower


def doji(df: pd.DataFrame) -> pd.Series:
    """Doji: open and close nearly equal (indecision)."""
    _validate_candles(df)
    body, _, _ = _candle_parts(df)
    rng = df["high"] - df["low"]
    return body <= 0.10 * rng


def pin_bar_bullish(df: pd.DataFrame) -> pd.Series:
    """Bullish pin bar: long lower wick with body in the upper part."""
    _validate_candles(df)
    body, upper, lower = _candle_parts(df)
    long_lower = lower >= 2.0 * body
    body_upper = body >= upper  # body sits at the top of the range
    return long_lower & body_upper


def pin_bar_bearish(df: pd.DataFrame) -> pd.Series:
    """Bearish pin bar: long upper wick with body in the lower part."""
    _validate_candles(df)
    body, upper, lower = _candle_parts(df)
    long_upper = upper >= 2.0 * body
    body_lower = body >= lower  # body sits at the bottom of the range
    return long_upper & body_lower


def _pattern_map() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    return {
        "bullish_engulfing": bullish_engulfing,
        "bearish_engulfing": bearish_engulfing,
        "hammer": hammer,
        "shooting_star": shooting_star,
        "doji": doji,
        "pin_bar_bullish": pin_bar_bullish,
        "pin_bar_bearish": pin_bar_bearish,
    }


def latest_patterns(df: pd.DataFrame) -> dict[str, bool]:
    """Return a dict of pattern booleans for the most recent candle."""
    _validate_candles(df)
    out: dict[str, bool] = {}
    for name, fn in _pattern_map().items():
        out[name] = bool(fn(df).iloc[-1])
    return out


def candle_signal(df: pd.DataFrame) -> pd.Series:
    """Collapse candlestick patterns into a directional score per candle.

    Returns +1 for a strong bullish confirmation, -1 for a strong bearish
    confirmation, and 0 otherwise. Engulfing and pin bars count as strong
    signals; hammers/shooting stars count as mild; doji is neutral and never
    confirms an entry.
    """
    _validate_candles(df)
    signal = pd.Series(0, index=df.index, dtype=float)

    signal = signal.where(~bullish_engulfing(df), 1.0)
    signal = signal.where(~bearish_engulfing(df), -1.0)
    signal = signal.where(~pin_bar_bullish(df), 1.0)
    signal = signal.where(~pin_bar_bearish(df), -1.0)
    signal = signal.where(~hammer(df), 0.5)
    signal = signal.where(~shooting_star(df), -0.5)

    # Doji never confirms an entry (indecision).
    signal = signal.where(~doji(df), 0.0)
    return signal
