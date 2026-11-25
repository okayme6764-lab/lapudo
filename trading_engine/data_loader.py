"""Historical OHLCV data loading and validation."""

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close")


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an OHLCV DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("OHLCV data is empty")

    result = df.copy()

    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must contain timestamps")

    if result.index.has_duplicates:
        raise ValueError("Duplicate timestamps detected")

    result = result.sort_index()

    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise")

    values = result[list(REQUIRED_COLUMNS)].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("OHLC data contains missing or infinite values")

    if (result["high"] < result["low"]).any():
        raise ValueError("High cannot be lower than low")

    if (result["high"] < result[["open", "close"]].max(axis=1)).any():
        raise ValueError("High is inconsistent with open/close")

    if (result["low"] > result[["open", "close"]].min(axis=1)).any():
        raise ValueError("Low is inconsistent with open/close")

    if "volume" in result.columns:
        result["volume"] = pd.to_numeric(result["volume"], errors="raise")
        if (result["volume"] < 0).any():
            raise ValueError("Volume cannot be negative")

    return result


def load_ohlcv(path, timestamp_col="timestamp") -> pd.DataFrame:
    """Load OHLCV candles from a CSV file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col], errors="raise"
        )
        df = df.set_index(timestamp_col)

    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")

    return validate_ohlcv(df)
