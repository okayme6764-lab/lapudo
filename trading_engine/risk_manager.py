"""Risk and position sizing for EGM Securities XAUUSD."""

from dataclasses import dataclass
from math import floor


@dataclass
class BrokerSpec:
    contract_size: float = 100.0
    min_volume: float = 0.01
    max_volume: float = 100.0
    volume_step: float = 0.01
    tick_size: float = 0.01
    tick_value: float = 1.0
    pip_size: float = 0.01


@dataclass
class TradeRisk:
    allowed: bool
    reason: str
    balance: float
    risk_amount: float
    lot_size: float
    stop_pips: float
    target_pips: float
    expected_loss: float
    expected_profit: float


class RiskManager:
    def __init__(
        self,
        broker: BrokerSpec | None = None,
        risk_percent: float = 20.0,
        reward_risk_ratio: float = 1.5,
        max_daily_loss_percent: float = 3.0,
    ):
        self.broker = broker or BrokerSpec()

        if not 0 < risk_percent <= 100:
            raise ValueError("risk_percent must be between 0 and 100")
        if reward_risk_ratio <= 0:
            raise ValueError("reward_risk_ratio must be positive")

        self.risk_percent = risk_percent
        self.reward_risk_ratio = reward_risk_ratio
        self.max_daily_loss_percent = max_daily_loss_percent

    def calculate(
        self,
        balance: float,
        entry_price: float,
        stop_price: float,
        direction: str,
    ) -> TradeRisk:

        b = self.broker

        if balance <= 0 or entry_price <= 0:
            raise ValueError("Balance and entry price must be positive")

        direction = direction.upper()
        if direction not in ("BUY", "SELL"):
            raise ValueError("Direction must be BUY or SELL")

        distance = (
            entry_price - stop_price
            if direction == "BUY"
            else stop_price - entry_price
        )

        if distance <= 0:
            raise ValueError("Stop loss is on the wrong side of entry")

        if min(
            b.contract_size, b.min_volume, b.volume_step,
            b.tick_size, b.tick_value, b.pip_size
        ) <= 0:
            raise ValueError("Broker specifications must be positive")

        risk_amount = balance * self.risk_percent / 100
        stop_pips = distance / b.pip_size
        target_pips = stop_pips * self.reward_risk_ratio

        # Tick value is assumed to be quoted per one standard lot.
        loss_per_lot = (distance / b.tick_size) * b.tick_value

        raw_lots = risk_amount / loss_per_lot
        lot_size = floor(raw_lots / b.volume_step) * b.volume_step

        if lot_size < b.min_volume:
            minimum_loss = b.min_volume * loss_per_lot
            return TradeRisk(
                False,
                f"Minimum lot risks {minimum_loss:.2f}, above budget {risk_amount:.2f}",
                balance, risk_amount, 0.0,
                stop_pips, target_pips, minimum_loss, 0.0
            )

        lot_size = min(lot_size, b.max_volume)
        expected_loss = lot_size * loss_per_lot
        expected_profit = expected_loss * self.reward_risk_ratio

        return TradeRisk(
            True,
            "Trade passes preliminary risk checks",
            balance,
            risk_amount,
            lot_size,
            stop_pips,
            target_pips,
            expected_loss,
            expected_profit,
        )
