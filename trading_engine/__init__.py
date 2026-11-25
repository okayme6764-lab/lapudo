"""XAUUSD AI Scalper - core trading engine.

Autonomous multi-indicator scalping strategy for Gold (XAUUSD) using
1-minute candles for entries and 5-minute candles for trend confirmation.

Modules
-------
indicators  : Technical indicator math (EMA, RSI, MACD, ATR).
signals     : Candlestick pattern detection and entry confirmation.
strategy    : Multi-timeframe signal generation with confidence scoring.
data_loader : Historical OHLCV loading/validation.
backtest    : Historical simulation harness for the strategy.
"""

from trading_engine.indicators import ema, rsi, macd, atr
from trading_engine.signals import (
    candle_signal,
    latest_patterns,
    bullish_engulfing,
    bearish_engulfing,
    hammer,
    shooting_star,
    doji,
    pin_bar_bullish,
    pin_bar_bearish,
)
from trading_engine.strategy import (
    ScalpingStrategy,
    StrategyConfig,
    TradeSignal,
    InsufficientDataError,
)
from trading_engine.data_loader import load_ohlcv, validate_ohlcv
from trading_engine.backtest import Backtester, BacktestResult

__all__ = [
    "ema",
    "rsi",
    "macd",
    "atr",
    "candle_signal",
    "latest_patterns",
    "bullish_engulfing",
    "bearish_engulfing",
    "hammer",
    "shooting_star",
    "doji",
    "pin_bar_bullish",
    "pin_bar_bearish",
    "ScalpingStrategy",
    "StrategyConfig",
    "TradeSignal",
    "InsufficientDataError",
    "load_ohlcv",
    "validate_ohlcv",
    "Backtester",
    "BacktestResult",
]

__version__ = "0.1.0"
