"""SVI surface calibration and radar metric computation.

Takes a set of already-subscribed option contracts and computes per-expiry
DataFrames with IV, SVI-fitted IV, bid/ask prices, and order-book imbalance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ib_insync import Option

from options_pricer import OptionsPricer
from subscription_manager import SubscriptionManager

logger = logging.getLogger(__name__)

_SPREAD_THRESHOLD_PCT = 15.0


@dataclass
class OptionQuote:
    """Snapshot of a single option's quote-derived metrics for one radar cycle."""

    bid: float
    ask: float
    implied_vol: float
    order_book_imbalance: float
    spread_pct: float


class VolatilitySurfaceEngine:
    """Fits SVI per expiry and computes radar-ready metrics."""

    def __init__(
        self,
        subscription_manager: SubscriptionManager,
        risk_free_rate: float,
        div_yield: float,
    ) -> None:
        self._subscriptions = subscription_manager
        self._risk_free_rate = risk_free_rate
        self._div_yield = div_yield

    def _extract_quote(self, ticker) -> Optional[Tuple[float, float, float]]:
        """Returns (bid, ask, order_book_imbalance) for a ticker, or None if no two-sided quote."""
        bid = float(ticker.bid) if ticker.bid and not np.isnan(ticker.bid) else 0.0
        ask = float(ticker.ask) if ticker.ask and not np.isnan(ticker.ask) else 0.0
        if bid <= 0 and ask <= 0:
            return None

        bid_size = float(ticker.bidSize) if ticker.bidSize and not np.isnan(ticker.bidSize) else 0.0
        ask_size = float(ticker.askSize) if ticker.askSize and not np.isnan(ticker.askSize) else 0.0
        total_size = bid_size + ask_size
        obi = (bid_size - ask_size) / total_size if total_size > 0 else 0.0

        return bid, ask, obi

    @staticmethod
    def _format_obi(obi: float) -> str:
        """Formats order-book imbalance for display."""
        if obi > 0.4:
            return f"+{obi:.2f}"
        if obi < -0.4:
            return f"{obi:.2f}"
        return f"{obi:+.2f}"

    def compute(
        self,
        spot: float,
        expiries: List[str],
        contracts_by_expiry: Dict[str, List[Option]],
    ) -> Dict[str, pd.DataFrame]:
        """Computes the per-expiry radar DataFrame for the current cycle.

        Args:
            spot: Current underlying price.
            expiries: Tracked expiry strings, in display order.
            contracts_by_expiry: Subscribed option contracts per expiry.

        Returns:
            Mapping of expiry -> DataFrame with columns for bid/ask/IV/SVI per strike.
        """
        radar_data: Dict[str, pd.DataFrame] = {}
        if not expiries or not contracts_by_expiry:
            return radar_data

        for expiry in expiries:
            contracts = contracts_by_expiry.get(expiry, [])
            if not contracts:
                continue

            t = OptionsPricer.time_to_expiry(expiry)

            quotes_by_strike: Dict[float, Dict[str, OptionQuote]] = {}
            svi_strikes: List[float] = []
            svi_ivs: List[float] = []

            for contract in contracts:
                ticker = self._subscriptions.get_ticker(contract)
                if ticker is None:
                    continue

                quote = self._extract_quote(ticker)
                if quote is None:
                    continue
                bid, ask, obi = quote

                mid_price = (bid + ask) / 2
                strike, right = contract.strike, contract.right

                iv = max(
                    OptionsPricer.implied_vol_bisection(
                        mid_price, spot, strike, t, self._risk_free_rate, self._div_yield, right
                    ),
                    0.0001,
                )
                spread_pct = (ask - bid) / bid * 100 if bid > 0 else 0.0

                if strike not in quotes_by_strike:
                    quotes_by_strike[strike] = {}
                quotes_by_strike[strike][right] = OptionQuote(
                    bid=bid, ask=ask, implied_vol=iv, order_book_imbalance=obi, spread_pct=spread_pct
                )

                # Collect OTM quotes for SVI calibration
                is_otm = (right == "C" and strike >= spot) or (right == "P" and strike < spot)
                if is_otm:
                    if strike not in svi_strikes:
                        svi_strikes.append(strike)
                        svi_ivs.append(iv)

            if len(svi_strikes) < 5:
                continue

            order = np.argsort(svi_strikes)
            svi_strikes_sorted = [svi_strikes[i] for i in order]
            svi_ivs_sorted = [svi_ivs[i] for i in order]

            _, svi_params = OptionsPricer.calibrate_svi(svi_strikes_sorted, svi_ivs_sorted, spot, t)

            rows = []
            for strike in sorted(quotes_by_strike.keys()):
                quotes = quotes_by_strike[strike]

                if svi_params is not None:
                    k = np.log(strike / spot)
                    a, b, rho, m, sigma = svi_params
                    w_fit = OptionsPricer.svi_total_variance(k, a, b, rho, m, sigma)
                    iv_svi = np.sqrt(max(w_fit, 0) / t)
                else:
                    iv_svi = 0.0

                rows.append(
                    {
                        "Strike": f"{strike:.1f}",
                        "Call Bid": f"{quotes['C'].bid:.2f}" if "C" in quotes else "N/A",
                        "Call Ask": f"{quotes['C'].ask:.2f}" if "C" in quotes else "N/A",
                        "Call IV": f"{quotes['C'].implied_vol * 100:.1f}%" if "C" in quotes else "N/A",
                        "SVI IV": f"{iv_svi * 100:.1f}%",
                        "Put IV": f"{quotes['P'].implied_vol * 100:.1f}%" if "P" in quotes else "N/A",
                        "Put Ask": f"{quotes['P'].ask:.2f}" if "P" in quotes else "N/A",
                        "Put Bid": f"{quotes['P'].bid:.2f}" if "P" in quotes else "N/A",
                        "Call OBI": self._format_obi(quotes["C"].order_book_imbalance) if "C" in quotes else "N/A",
                        "Put OBI": self._format_obi(quotes["P"].order_book_imbalance) if "P" in quotes else "N/A",
                    }
                )

            if rows:
                radar_data[expiry] = pd.DataFrame(rows)

        return radar_data
