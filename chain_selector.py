"""Selects the near-the-money option contracts to track for the volatility radar."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from ib_insync import IB, Option, Ticker

from subscription_manager import SubscriptionManager

logger = logging.getLogger(__name__)

_RADAR_OWNER = "radar"
_STRIKES_UNIVERSE = 35       # candidate strikes before filtering
_STRIKES_PER_EXPIRY = 8      # final strikes kept per expiry (one leg only, OTM)
_EXPIRIES_TRACKED = 10       # number of expirations tracked


class OptionChainSelector:
    """Selects and manages subscriptions for near-the-money option contracts."""

    def __init__(self, symbol: str, ib: IB, subscription_manager: SubscriptionManager) -> None:
        self.symbol = symbol
        self._ib = ib
        self._subscriptions = subscription_manager

        self.expiries: List[str] = []
        self.contracts_by_expiry: Dict[str, List[Option]] = {}

    def _select_target_expiries(self, chain_expirations: List[str]) -> List[str]:
        """Picks the nearest `_EXPIRIES_TRACKED` expirations at or beyond next Friday + 6 days."""
        target_date = datetime.now() + timedelta(days=6)
        while target_date.weekday() != 4:  # 4 == Friday
            target_date += timedelta(days=1)
        target = target_date.strftime("%Y%m%d")

        valid_expiries = [exp for exp in sorted(chain_expirations) if exp >= target]
        if valid_expiries:
            return valid_expiries[:_EXPIRIES_TRACKED]
        return sorted(chain_expirations)[-_EXPIRIES_TRACKED:]

    def refresh(self, underlying_ticker: Ticker, spot: float) -> None:
        """Rebuilds the tracked contract set and (re)subscribes to their streams.

        For each tracked expiry, selects the `_STRIKES_PER_EXPIRY` strikes
        closest to spot using only the out-of-the-money leg (calls ≥ spot,
        puts < spot) to avoid duplicating subscriptions.

        Args:
            underlying_ticker: Live ticker for the underlying stock.
            spot: Current underlying price.
        """
        self._subscriptions.release_owner(_RADAR_OWNER)

        contract = underlying_ticker.contract
        chains = self._ib.reqSecDefOptParams(contract.symbol, "", contract.secType, contract.conId)
        chain = next((c for c in chains if c.exchange == "SMART"), None)
        if not chain:
            logger.warning("No SMART option chain found for %s.", self.symbol)
            return

        self.expiries = self._select_target_expiries(chain.expirations)
        self.contracts_by_expiry = {}

        nearest_strikes = sorted(chain.strikes, key=lambda s: abs(s - spot))[:_STRIKES_UNIVERSE]
        ordered_strikes = sorted(nearest_strikes)

        for expiry in self.expiries:
            # Build candidates for OTM only: calls ≥ spot, puts < spot
            call_candidates = [Option(self.symbol, expiry, s, "C", "SMART") for s in ordered_strikes if s >= spot]
            put_candidates = [Option(self.symbol, expiry, s, "P", "SMART") for s in ordered_strikes if s < spot]
            qualified = self._ib.qualifyContracts(*(call_candidates + put_candidates))

            if not qualified:
                continue

            # Select the top `_STRIKES_PER_EXPIRY` by distance to spot
            by_distance = sorted(qualified, key=lambda c: abs(c.strike - spot))
            selected_contracts = by_distance[:_STRIKES_PER_EXPIRY]

            self.contracts_by_expiry[expiry] = sorted(selected_contracts, key=lambda c: (c.strike, c.right))

            for c in self.contracts_by_expiry[expiry]:
                self._subscriptions.subscribe(c, _RADAR_OWNER)
