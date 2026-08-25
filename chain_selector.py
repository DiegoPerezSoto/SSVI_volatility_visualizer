"""Selects an optimal grid of OTM option contracts to maximize SVI surface resolution."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from ib_insync import IB, Option, Ticker

from subscription_manager import SubscriptionManager

logger = logging.getLogger(__name__)

_RADAR_OWNER = "radar"

# OPTIMIZED QUOTA ALLOCATION (8 expiries * 10 strikes = 80 lines, safe for 100 limit)
_EXPIRIES_TRACKED = 14    # Consecutive maturities to build a dense term structure
_OTM_PUTS_PER_EXPIRY = 20  # Contiguous OTM puts for the left wing and skew
_OTM_CALLS_PER_EXPIRY = 20  # Contiguous OTM calls for the right wing


class OptionChainSelector:
    """Selects and manages subscriptions to build a high-resolution volatility surface."""

    def __init__(self, symbol: str, ib: IB, subscription_manager: SubscriptionManager) -> None:
        """Initializes the selector with IBKR client instances and tracking state."""
        self.symbol = symbol
        self._ib = ib
        self._subscriptions = subscription_manager

        self.expiries: List[str] = []
        self.contracts_by_expiry: Dict[str, List[Option]] = {}

    def _select_target_expiries(self, chain_expirations: List[str]) -> List[str]:
        """Picks the nearest `_EXPIRIES_TRACKED` expirations, preferring those >= next Friday + 6 days."""
        if not chain_expirations:
            return []
        
        # Try to get expirations starting from next Friday + 6 days
        target_date = datetime.now() + timedelta(days=6)
        while target_date.weekday() != 4:  # Friday
            target_date += timedelta(days=1)
        target = target_date.strftime("%Y%m%d")
        
        sorted_exps = sorted(chain_expirations)
        
        # Prefer expirations >= target
        valid_expiries = [exp for exp in sorted_exps if exp >= target]
        if valid_expiries and len(valid_expiries) >= _EXPIRIES_TRACKED:
            return valid_expiries[:_EXPIRIES_TRACKED]
        
        # If not enough, take the closest _EXPIRIES_TRACKED expirations
        if len(sorted_exps) >= _EXPIRIES_TRACKED:
            return sorted_exps[-_EXPIRIES_TRACKED:]
        
        # If less than _EXPIRIES_TRACKED available, return all
        logger.warning(f"Only {len(sorted_exps)} expirations available, wanted {_EXPIRIES_TRACKED}")
        return sorted_exps


    def _select_high_res_otm_strikes(self, available_strikes: List[float], spot: float) -> List[float]:
        """Selects a dense, contiguous window of strikes centered around the spot.
        
        Guarantees enough points (10 per expiry) for the Gatheral SVI algorithm 
        to perfectly anchor both the ATM curvature and the deep OTM asymptotes.
        """
        sorted_strikes = sorted(set(available_strikes))
        
        # OTM strictly defined
        puts_strikes = [k for k in sorted_strikes if k < spot]
        calls_strikes = [k for k in sorted_strikes if k >= spot]

        # Extract the closest N puts (highest strikes below spot)
        selected_puts = puts_strikes[-_OTM_PUTS_PER_EXPIRY:] if len(puts_strikes) >= _OTM_PUTS_PER_EXPIRY else puts_strikes
        # Extract the closest N calls (lowest strikes above spot)
        selected_calls = calls_strikes[:_OTM_CALLS_PER_EXPIRY] if len(calls_strikes) >= _OTM_CALLS_PER_EXPIRY else calls_strikes

        return sorted(selected_puts + selected_calls)

    def refresh(self, underlying_ticker: Ticker, spot: float) -> None:
        """Rebuilds the tracked contract set and (re)subscribes to their live streams.

        Args:
            underlying_ticker: Live ticker for the underlying stock.
            spot: Current underlying asset price.
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

        # Get our dense strike grid based on current spot
        target_strikes = self._select_high_res_otm_strikes(chain.strikes, spot)

        for expiry in self.expiries:
            call_candidates = [Option(self.symbol, expiry, s, "C", "SMART") for s in target_strikes if s >= spot]
            put_candidates = [Option(self.symbol, expiry, s, "P", "SMART") for s in target_strikes if s < spot]
            
            qualified = self._ib.qualifyContracts(*(call_candidates + put_candidates))
            if not qualified:
                continue

            self.contracts_by_expiry[expiry] = sorted(qualified, key=lambda c: (c.strike, c.right))

            # Register subscriptions dynamically
            for c in self.contracts_by_expiry[expiry]:
                self._subscriptions.subscribe(c, _RADAR_OWNER)