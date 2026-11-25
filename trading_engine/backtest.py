"""Historical backtest harness for the scalping strategy.

The backtester runs the strategy over a 1-minute OHLCV history (resampling to
5-minute internally for trend confirmation) and simulates a simple position
management model:

  * signals are computed on the close of bar ``t`` and executed at the open
    of bar ``t+1`` (no lookahead);
  * stop-loss and take-profit are checked against each bar's high/low;
  * if both stop and target fall inside one bar, the stop is assumed to be
    hit first (conservative);
  * a fixed fraction of equity is risked per trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from trading_engine.strategy import ScalpingStrategy, TradeSignal, InsufficientDataError

_OHLCV = ("open", "high", "low", "close")


@dataclass
class BacktestResult:
    """Summary statistics from a backtest run."""

    n_bars: int
    n_trades: int
    n_long: int
    n_short: int
    total_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    final_equity: float
    equity_curve: pd.Series
    trades: List[dict] = field(default_factory=list)


@dataclass
class _Position:
    direction: str  # "BUY" | "SELL"
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    stop: float
    target: float
    risk_amount: float


class Backtester:
    """Simulate the strategy over historical 1-minute data."""

    def __init__(
        self,
        strategy: Optional[ScalpingStrategy] = None,
        initial_capital: float = 10_000.0,
        risk_per_trade: float = 0.20,
        take_profit_rr: float = 1.5,
        commission_per_trade: float = 0.0,
        resample_rule: str = "5min",
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0.0 < risk_per_trade <= 1.0:
            raise ValueError("risk_per_trade must be in (0, 1]")
        if take_profit_rr <= 0:
            raise ValueError("take_profit_rr must be positive")

        self.strategy = strategy or ScalpingStrategy()
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.take_profit_rr = take_profit_rr
        self.commission_per_trade = commission_per_trade
        self.resample_rule = resample_rule

    # ------------------------------------------------------------------
    @staticmethod
    def _resample_5m(df1: pd.DataFrame) -> pd.DataFrame:
        agg = {c: "last" for c in _OHLCV}
        agg["high"] = "max"
        agg["low"] = "min"
        if "volume" in df1.columns:
            agg["volume"] = "sum"
        df5 = df1.resample("5min").agg(agg).dropna(subset=["open", "close"])
        return df5

    def run(self, df1: pd.DataFrame) -> BacktestResult:
        """Run the backtest on a 1-minute OHLCV frame."""
        if not isinstance(df1, pd.DataFrame):
            raise TypeError(f"expected a pandas DataFrame, got {type(df1).__name__}")
        missing = [c for c in _OHLCV if c not in df1.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        if len(df1) < self.strategy.config.min_bars_1m:
            raise InsufficientDataError(
                f"not enough 1-minute bars: {len(df1)} < "
                f"{self.strategy.config.min_bars_1m}"
            )

        df1 = df1.sort_index()
        df5 = self._resample_5m(df1)
        if len(df5) < self.strategy.config.min_bars_5m:
            raise InsufficientDataError(
                f"not enough 5-minute bars after resampling: {len(df5)} < "
                f"{self.strategy.config.min_bars_5m}"
            )

        feats = self.strategy.precompute(df1, df5)

        equity = self.initial_capital
        position: Optional[_Position] = None
        trades: List[dict] = []
        equity_curve = []

        for i in range(len(feats)):
            row = feats.iloc[i]
            bar_time = feats.index[i]
            o, h, l, c = (float(row["open"]), float(row["high"]),
                           float(row["low"]), float(row["close"]))

            # --- manage open position with the current bar ---
            if position is not None:
                closed = self._manage_position(position, bar_time, o, h, l)
                if closed is not None:
                    equity += closed["pnl"]
                    closed["equity"] = equity
                    trades.append(closed)
                    position = None

            # --- generate signal on this bar's close, execute next open ---
            signal = self.strategy.decide(row)
            if position is None and signal.is_entry and i + 1 < len(feats):
                next_open = float(feats.iloc[i + 1]["open"])
                position = self._open_position(signal, next_open, equity)

            equity_curve.append(equity)

        # Close any open position at the last close.
        if position is not None:
            last = feats.iloc[-1]
            closed = self._close_position(
                position, feats.index[-1], float(last["close"])
            )
            equity += closed["pnl"]
            closed["equity"] = equity
            trades.append(closed)
            if equity_curve:
                equity_curve[-1] = equity

        equity_series = pd.Series(equity_curve, index=feats.index)
        return self._summarize(equity_series, trades)

    # ------------------------------------------------------------------
    def _open_position(self, signal: TradeSignal, entry_price: float,
                       equity: float) -> _Position:
        direction = signal.action
        stop_distance = max(signal.stop_distance, 1e-9)
        risk_amount = equity * self.risk_per_trade
        size = risk_amount / stop_distance

        if direction == "BUY":
            stop = entry_price - stop_distance
            target = entry_price + stop_distance * self.take_profit_rr
        else:
            stop = entry_price + stop_distance
            target = entry_price - stop_distance * self.take_profit_rr

        return _Position(
            direction=direction,
            entry_time=signal.timestamp,
            entry_price=entry_price,
            size=size,
            stop=stop,
            target=target,
            risk_amount=risk_amount,
        )

    def _manage_position(self, pos: _Position, bar_time: pd.Timestamp,
                         o: float, h: float, l: float) -> Optional[dict]:
        """Check stop/target against a bar; returns a closed-trade dict or None."""
        if pos.direction == "BUY":
            stop_hit = l <= pos.stop
            target_hit = h >= pos.target
        else:
            stop_hit = h >= pos.stop
            target_hit = l <= pos.target

        if stop_hit and target_hit:
            # Conservative: assume stop is hit first.
            exit_price = pos.stop
            reason = "stop"
        elif stop_hit:
            exit_price = pos.stop
            reason = "stop"
        elif target_hit:
            exit_price = pos.target
            reason = "target"
        else:
            return None

        return self._close_position(pos, bar_time, exit_price, reason)

    def _close_position(self, pos: _Position, bar_time: pd.Timestamp,
                        exit_price: float, reason: str = "close") -> dict:
        if pos.direction == "BUY":
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_price) * pos.size
        pnl -= self.commission_per_trade
        return {
            "direction": pos.direction,
            "entry_time": pos.entry_time,
            "exit_time": bar_time,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "size": pos.size,
            "pnl": pnl,
            "return_pct": pnl / pos.risk_amount,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _summarize(equity: pd.Series, trades: List[dict]) -> BacktestResult:
        n = len(trades)
        n_long = sum(1 for t in trades if t["direction"] == "BUY")
        n_short = n - n_long

        total_return = (equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / n if n else 0.0
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = float(drawdown.min()) if len(drawdown) else 0.0

        return BacktestResult(
            n_bars=len(equity),
            n_trades=n,
            n_long=n_long,
            n_short=n_short,
            total_return=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            final_equity=float(equity.iloc[-1]) if len(equity) else 0.0,
            equity_curve=equity,
            trades=trades,
        )
