"""Multi-timeframe scalping strategy for XAUUSD.

The strategy combines:
  * 5-minute candles -> trend direction (EMA 50 + EMA 9/21 alignment)
  * 1-minute candles  -> entry signals (EMA 9/21, RSI, MACD, candlestick)

Each component contributes a score in [-1, 1]; the weighted sum is the signal
confidence. Trades are suppressed when the market is ranging, when indicators
conflict, when volatility is out of the tradable band, or when confidence is
below the configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from trading_engine.indicators import ema, rsi, macd, atr
from trading_engine.signals import candle_signal


class InsufficientDataError(Exception):
    """Raised when not enough history is available for a reliable signal."""


@dataclass
class StrategyConfig:
    """Tunable parameters for the scalping strategy."""

    # Indicator periods
    ema_fast: int = 9
    ema_mid: int = 21
    ema_trend: int = 50
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14

    # Component weights (must sum to 1.0)
    weight_trend: float = 0.35
    weight_ema: float = 0.25
    weight_rsi: float = 0.10
    weight_macd: float = 0.15
    weight_confirm: float = 0.15

    # Decision thresholds
    min_confidence: float = 0.55
    ema_deadband: float = 0.00005  # relative to price, avoids noise crossovers
    conflict_penalty: float = 0.5  # confidence multiplier on trend/entry conflict

    # Volatility band (ATR as a fraction of price)
    atr_max_frac: float = 0.004
    atr_min_frac: float = 0.0001

    # Risk
    stop_atr_mult: float = 1.5

    # Warmup
    min_bars_1m: int = 60
    min_bars_5m: int = 60

    def validate(self) -> None:
        w = (
            self.weight_trend
            + self.weight_ema
            + self.weight_rsi
            + self.weight_macd
            + self.weight_confirm
        )
        if not np.isclose(w, 1.0, atol=1e-6):
            raise ValueError(f"component weights must sum to 1.0, got {w:.4f}")
        if not 0.0 < self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1]")


@dataclass
class TradeSignal:
    """A single strategy decision for one bar."""

    action: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    reasons: List[str] = field(default_factory=list)
    timestamp: pd.Timestamp | None = None
    atr: float = 0.0
    atr_fraction: float = 0.0
    stop_distance: float = 0.0
    trend_state: str = "RANGING"
    price: float = 0.0

    @property
    def is_entry(self) -> bool:
        return self.action in ("BUY", "SELL")


_REQUIRED_COLUMNS = ("open", "high", "low", "close")


class ScalpingStrategy:
    """Autonomous multi-indicator scalping strategy."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self.config.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_frame(df: pd.DataFrame, name: str) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        if df.empty:
            raise ValueError(f"{name} is empty")

    def _check_warmup(self, df1: pd.DataFrame, df5: pd.DataFrame) -> None:
        if len(df1) < self.config.min_bars_1m:
            raise InsufficientDataError(
                f"1-minute data has {len(df1)} bars, need >= {self.config.min_bars_1m}"
            )
        if len(df5) < self.config.min_bars_5m:
            raise InsufficientDataError(
                f"5-minute data has {len(df5)} bars, need >= {self.config.min_bars_5m}"
            )

    # ------------------------------------------------------------------
    # Trend state (5-minute)
    # ------------------------------------------------------------------
    def _trend_state_series(self, df5: pd.DataFrame) -> pd.Series:
        cfg = self.config
        close = df5["close"]
        e50 = ema(close, cfg.ema_trend)
        e9 = ema(close, cfg.ema_fast)
        e21 = ema(close, cfg.ema_mid)

        # EMA50 slope over the last 3 bars (difference, not ratio).
        slope = e50.diff(3)
        bull = (slope > 0) & (close > e50) & (e9 > e21)
        bear = (slope < 0) & (close < e50) & (e9 < e21)

        state = pd.Series("RANGING", index=df5.index, dtype=object)
        state[bull] = "BULLISH"
        state[bear] = "BEARISH"
        return state

    # ------------------------------------------------------------------
    # Feature precomputation
    # ------------------------------------------------------------------
    def precompute(self, df1: pd.DataFrame, df5: pd.DataFrame) -> pd.DataFrame:
        """Augment the 1-minute frame with every feature a decision needs.

        All features are causal (only past data), so the returned frame can be
        used directly in a backtest without lookahead.
        """
        self._validate_frame(df1, "df1")
        self._validate_frame(df5, "df5")
        self._check_warmup(df1, df5)

        cfg = self.config
        out = df1.copy()

        close = out["close"]
        out["ema9"] = ema(close, cfg.ema_fast)
        out["ema21"] = ema(close, cfg.ema_mid)
        out["rsi"] = rsi(close, cfg.rsi_period)
        _, _, hist = macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        out["macd_hist"] = hist
        out["macd_hist_prev"] = hist.shift(1)
        out["atr"] = atr(out["high"], out["low"], close, cfg.atr_period)
        out["candle_sig"] = candle_signal(out)

        # 5-minute trend state carried onto each 1-minute bar (backward).
        trend = self._trend_state_series(df5)
        trend_score = trend.map({"BULLISH": 1.0, "BEARISH": -1.0}).fillna(0.0)

        left = out.reset_index()
        left.columns = ["timestamp"] + list(left.columns[1:])
        right = pd.DataFrame(
            {"timestamp": df5.index, "trend_state": trend.values, "trend_score": trend_score.values}
        )
        merged = pd.merge_asof(
            left.sort_values("timestamp"),
            right.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        out["trend_state"] = merged["trend_state"].values
        out["trend_score"] = merged["trend_score"].values
        return out

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def decide(self, row: pd.Series) -> TradeSignal:
        """Turn one feature row into a TradeSignal."""
        cfg = self.config
        ts = row.name
        price = float(row["close"])
        reasons: List[str] = []

        # --- component scores ---
        trend_score = float(row.get("trend_score", 0.0))
        trend_state = str(row.get("trend_state", "RANGING"))

        e9 = float(row["ema9"])
        e21 = float(row["ema21"])
        tol = cfg.ema_deadband * price
        if e9 > e21 + tol:
            ema_score = 1.0
        elif e9 < e21 - tol:
            ema_score = -1.0
        else:
            ema_score = 0.0

        rsi_val = float(row["rsi"])
        if rsi_val > 55:
            rsi_score = min(1.0, (rsi_val - 55) / 45.0)
        elif rsi_val < 45:
            rsi_score = max(-1.0, -(45 - rsi_val) / 45.0)
        else:
            rsi_score = 0.0

        h = float(row["macd_hist"])
        hp = float(row["macd_hist_prev"])
        if np.isnan(h):
            macd_score = 0.0
        elif h > 0 and h > hp:
            macd_score = 1.0
        elif h < 0 and h < hp:
            macd_score = -1.0
        elif h > 0:
            macd_score = 0.5
        else:
            macd_score = -0.5

        confirm_score = float(row["candle_sig"])

        confidence = (
            cfg.weight_trend * trend_score
            + cfg.weight_ema * ema_score
            + cfg.weight_rsi * rsi_score
            + cfg.weight_macd * macd_score
            + cfg.weight_confirm * confirm_score
        )

        atr_val = float(row["atr"])
        atr_frac = atr_val / price if price > 0 else 0.0

        # --- filters ---
        if trend_state == "RANGING":
            return TradeSignal(
                "HOLD", 0.0, ["market ranging"], ts, atr_val, atr_frac,
                0.0, trend_state, price,
            )

        if atr_frac > cfg.atr_max_frac:
            return TradeSignal(
                "HOLD", confidence, ["volatility too high"], ts, atr_val,
                atr_frac, 0.0, trend_state, price,
            )
        if atr_frac < cfg.atr_min_frac:
            return TradeSignal(
                "HOLD", confidence, ["volatility too low"], ts, atr_val,
                atr_frac, 0.0, trend_state, price,
            )

        entry_dir = 1.0 if ema_score > 0 else (-1.0 if ema_score < 0 else 0.0)
        if trend_score != 0.0 and entry_dir != 0.0 and (trend_score * entry_dir) < 0:
            confidence *= cfg.conflict_penalty
            reasons.append("trend/entry conflict")

        if confidence >= cfg.min_confidence:
            action = "BUY"
        elif confidence <= -cfg.min_confidence:
            action = "SELL"
        else:
            action = "HOLD"

        stop = cfg.stop_atr_mult * atr_val if action in ("BUY", "SELL") else 0.0
        if action != "HOLD":
            reasons.append(action.lower())

        return TradeSignal(action, confidence, reasons, ts, atr_val, atr_frac,
                           stop, trend_state, price)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, df1: pd.DataFrame, df5: pd.DataFrame) -> TradeSignal:
        """Produce a signal for the most recent bar of ``df1``."""
        feats = self.precompute(df1, df5)
        return self.decide(feats.iloc[-1])
