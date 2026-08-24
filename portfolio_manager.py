"""Tracks account balance and open positions, keeping their market-data streams in sync."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Set

import numpy as np
from ib_insync import IB

from subscription_manager import SubscriptionManager

logger = logging.getLogger(__name__)

_BALANCE_TAGS = {"NetLiquidation", "AvailableFunds", "ExcessLiquidity"}
_PORTFOLIO_OWNER = "portfolio"


@dataclass
class PositionSnapshot:
    """A single open position, ready for display or downstream analytics."""

    symbol: str
    instrument_type: str
    expiry: str
    quantity: float
    avg_cost: float
    current_price: float
    pnl: float
    pnl_pct: float


class PortfolioManager:
    """Owns account balance and open-position bookkeeping."""

    def __init__(self, ib: IB, subscription_manager: SubscriptionManager) -> None:
        self._ib = ib
        self._subscriptions = subscription_manager

        self.net_liquidation: float = 0.0
        self.available_funds: float = 0.0
        self.excess_liquidity: float = 0.0
        self.positions: List[PositionSnapshot] = []

    def _update_balance(self) -> None:
        """Pulls the latest account summary values from IBKR."""
        try:
            summary = self._ib.accountSummary()
            values = {item.tag: item.value for item in summary if item.tag in _BALANCE_TAGS}

            self.net_liquidation = float(values.get("NetLiquidation", 0))
            self.available_funds = float(values.get("AvailableFunds", 0))
            self.excess_liquidity = float(values.get("ExcessLiquidity", 0))
        except Exception:
            logger.exception("Failed to update account balance.")

    def _release_closed_positions(self, current_con_ids: Set[int]) -> None:
        """Releases the 'portfolio' owner tag on contracts no longer held."""
        held_by_portfolio = {
            con_id for con_id, sub in self._subscriptions.subscriptions.items() if _PORTFOLIO_OWNER in sub.owners
        }

        for con_id in held_by_portfolio - current_con_ids:
            sub = self._subscriptions.subscriptions.get(con_id)
            if sub and _PORTFOLIO_OWNER in sub.owners:
                sub.owners.discard(_PORTFOLIO_OWNER)
                if not sub.owners:
                    self._ib.cancelMktData(sub.ticker.contract)
                    del self._subscriptions.subscriptions[con_id]

    def _analyze_positions(self) -> None:
        """Downloads open positions, reconciles subscriptions, and computes P&L."""
        try:
            raw_positions = self._ib.positions()

            contracts = []
            for pos in raw_positions:
                pos.contract.exchange = "SMART"
                contracts.append(pos.contract)

            if contracts:
                self._ib.qualifyContracts(*contracts)

            current_con_ids = {pos.contract.conId for pos in raw_positions}
            self._release_closed_positions(current_con_ids)

            self.positions = []

            for pos in raw_positions:
                ticker = self._subscriptions.subscribe(pos.contract, _PORTFOLIO_OWNER)

                current_price = ticker.last
                if np.isnan(current_price) or current_price <= 0:
                    current_price = ticker.close

                stale_price_flag = ""
                if np.isnan(current_price) or current_price <= 0:
                    current_price = pos.avgCost
                    stale_price_flag = "*"

                multiplier = (
                    float(pos.contract.multiplier)
                    if pos.contract.multiplier and pos.contract.multiplier.isdigit()
                    else 1.0
                )

                market_value = current_price * pos.position * multiplier
                total_cost = pos.avgCost * pos.position * multiplier
                pnl = market_value - total_cost
                pnl_pct = (pnl / total_cost * 100) if total_cost > 0 else 0

                if pos.contract.secType == "OPT":
                    instrument_type = "CALL" if pos.contract.right == "C" else "PUT"
                    raw_date = pos.contract.lastTradeDateOrContractMonth
                    expiry = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}" if len(raw_date) == 8 else raw_date
                    clean_symbol = f"{pos.contract.symbol} {pos.contract.strike}"
                else:
                    instrument_type = "STK"
                    expiry = "N/A"
                    clean_symbol = pos.contract.symbol

                self.positions.append(
                    PositionSnapshot(
                        symbol=clean_symbol + stale_price_flag,
                        instrument_type=instrument_type,
                        expiry=expiry,
                        quantity=pos.position,
                        avg_cost=pos.avgCost,
                        current_price=current_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
                )
        except Exception:
            logger.exception("Failed to process open positions.")

    def refresh(self) -> None:
        """Refreshes both account balance and open-position analytics."""
        self._update_balance()
        self._analyze_positions()
