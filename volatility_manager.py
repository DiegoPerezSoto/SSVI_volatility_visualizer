"""Coordinates option chain selection and SVI volatility surface computation."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from ib_insync import IB, Ticker

from chain_selector import OptionChainSelector
from radar_engine import VolatilitySurfaceEngine
from subscription_manager import SubscriptionManager

_CHAIN_REFRESH_TTL_SEC = 600


class VolatilityManager:
    """Owns option chain selection and the per-underlying SVI radar."""

    def __init__(
        self,
        symbol: str,
        ib: IB,
        subscription_manager: SubscriptionManager,
        risk_free_rate: float,
        div_yield: float,
    ) -> None:
        """Args:
            symbol: Underlying ticker symbol.
            ib: Connected ib_insync IB instance.
            subscription_manager: Shared SubscriptionManager for market-data streams.
            risk_free_rate: Risk-free rate used in Black-Scholes calculations.
            div_yield: Continuous dividend yield used in the same calculations.
        """
        self.symbol = symbol
        self._chain_selector = OptionChainSelector(symbol, ib, subscription_manager)
        self._engine = VolatilitySurfaceEngine(subscription_manager, risk_free_rate, div_yield)

        self._last_chain_refresh: float = 0.0
        self.radar_data: Dict[str, pd.DataFrame] = {}
        self.surface_slices: List[Dict[str, Any]] = []

    @property
    def expiries(self) -> List[str]:
        """Currently tracked expiry strings, in display order."""
        return self._chain_selector.expiries

    def build_radar(self, underlying_ticker: Ticker) -> None:
        """Refreshes the options radar for the current underlying price.

        Rebuilds the tracked contract set every `_CHAIN_REFRESH_TTL_SEC`
        seconds and recomputes SVI/IV/bid-ask metrics on every call.

        Args:
            underlying_ticker: Live ib_insync ticker for the underlying stock.
        """
        if not underlying_ticker or np.isnan(underlying_ticker.marketPrice()):
            return

        spot = underlying_ticker.marketPrice()
        if np.isnan(spot) or spot <= 0:
            return

        now = time.time()
        if now - self._last_chain_refresh > _CHAIN_REFRESH_TTL_SEC or not self._chain_selector.contracts_by_expiry:
            self._chain_selector.refresh(underlying_ticker, spot)
            self._last_chain_refresh = now

        self.radar_data, self.surface_slices = self._engine.compute(
            spot, self._chain_selector.expiries, self._chain_selector.contracts_by_expiry
        )