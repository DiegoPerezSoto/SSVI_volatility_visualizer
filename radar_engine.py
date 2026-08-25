"""SSVI surface calibration with time-dependent rho(theta) and radar metric computation.

Takes a set of already-subscribed option contracts, filters illiquid quotes,
anchors the ATM variance robustly, fits a unified global SSVI+rho(theta) surface, 
and computes radar-ready metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    """Extracts robust ATM anchors, filters spreads, fits global SSVI+rho(theta), and computes radar metrics."""

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
    ) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
        """Computes the per-expiry radar DataFrame and calibrates global SSVI+rho(theta).

        Args:
            spot: Current underlying price.
            expiries: Tracked expiry strings, in display order.
            contracts_by_expiry: Subscribed option contracts per expiry.

        Returns:
            Tuple of (DataFrame radar mapping, global surface slices for 3D visualization).
        """
        radar_data: Dict[str, pd.DataFrame] = {}
        surface_slices: List[Dict[str, Any]] = []
        if not expiries or not contracts_by_expiry:
            return radar_data, surface_slices

        # Phase 1: Collect valid quotes, filter illiquid spreads, and extract robust ATM variances (Theta)
        extracted_quotes = {}
        otm_points = []
        expiry_thetas = {}

        for expiry in expiries:
            contracts = contracts_by_expiry.get(expiry, [])
            if not contracts:
                continue

            t = OptionsPricer.time_to_expiry(expiry)
            quotes_by_strike = {}

            for contract in contracts:
                ticker = self._subscriptions.get_ticker(contract)
                if not ticker:
                    continue

                quote = self._extract_quote(ticker)
                if not quote:
                    continue
                
                bid, ask, obi = quote
                mid_price = (bid + ask) / 2
                strike, right = contract.strike, contract.right

                # Enforce strict liquidity threshold to protect global SSVI optimization from bad quotes
                spread_pct = (ask - bid) / bid * 100 if bid > 0 else 100.0
                if spread_pct > _SPREAD_THRESHOLD_PCT or bid <= 0:
                    continue

                iv = max(
                    OptionsPricer.implied_vol_bisection(
                        mid_price, spot, strike, t, self._risk_free_rate, self._div_yield, right
                    ),
                    0.0001,
                )

                if strike not in quotes_by_strike:
                    quotes_by_strike[strike] = {}
                quotes_by_strike[strike][right] = OptionQuote(bid, ask, iv, obi, spread_pct)

            if not quotes_by_strike:
                continue

            extracted_quotes[expiry] = quotes_by_strike

            # Derive robust Theta from a multi-strike ATM local average
            valid_atm_ivs = []
            for s_candidate in sorted(quotes_by_strike.keys(), key=lambda x: abs(x - spot))[:3]:
                for q_obj in quotes_by_strike[s_candidate].values():
                    valid_atm_ivs.append(q_obj.implied_vol)

            if valid_atm_ivs:
                atm_iv = sum(valid_atm_ivs) / len(valid_atm_ivs)
                theta = (atm_iv ** 2) * t
                expiry_thetas[expiry] = theta
            else:
                continue

            # Collect strict OTMs for global mesh calibration
            for strike, q_dict in quotes_by_strike.items():
                for right, q in q_dict.items():
                    is_otm = (right == "C" and strike >= spot) or (right == "P" and strike < spot)
                    if is_otm:
                        k = np.log(strike / spot)
                        w = (q.implied_vol ** 2) * t
                        otm_points.append((expiry, t, strike, k, q.implied_vol, w))

        # Enforce monotonic ATM variance across expiries: a locally noisy ATM IV
        # anchor can make theta(t) dip below the theta of an earlier expiry, baking
        # calendar arbitrage directly into the surface. Clamp each theta to the
        # running maximum of shorter-dated thetas.
        running_max_theta = 0.0
        for expiry in sorted(expiry_thetas.keys(), key=lambda e: OptionsPricer.time_to_expiry(e)):
            running_max_theta = max(running_max_theta, expiry_thetas[expiry])
            expiry_thetas[expiry] = running_max_theta

        # Phase 2: Global SSVI+rho(theta) Mesh Optimization
        global_params = None
        if len(otm_points) > 10:
            k_arr = np.array([p[3] for p in otm_points], dtype=float)
            theta_arr = np.array([expiry_thetas[p[0]] for p in otm_points], dtype=float)
            w_obs = np.array([p[5] for p in otm_points], dtype=float)

            global_params = OptionsPricer.calibrate_global_ssvi(k_arr, theta_arr, w_obs)
            if global_params is not None:
                rho_0_dbg, rho_infty_dbg, lam_dbg, eta_dbg, gamma_dbg = global_params
                logger.info(
                    "SSVI+rho(theta) fit -> rho0=%.4f, rho_inf=%.4f, lambda=%.4f, eta=%.4f, gamma=%.4f (points=%d, theta range=[%.5f, %.5f])",
                    rho_0_dbg, rho_infty_dbg, lam_dbg, eta_dbg, gamma_dbg, len(otm_points), theta_arr.min(), theta_arr.max(),
                )

        # Phase 3: Build Radar Outputs and Injection
        for expiry in expiries:
            if expiry not in extracted_quotes or expiry not in expiry_thetas:
                continue

            t = OptionsPricer.time_to_expiry(expiry)
            theta = expiry_thetas[expiry]
            quotes_by_strike = extracted_quotes[expiry]

            strikes = [p[2] for p in otm_points if p[0] == expiry]
            market_ivs = [p[4] for p in otm_points if p[0] == expiry]

            if global_params is not None and strikes:
                order = np.argsort(strikes)
                surface_slices.append({
                    "expiry": expiry,
                    "t": t,
                    "theta": theta,
                    "strikes": np.array([strikes[i] for i in order], dtype=float),
                    "market_ivs": np.array([market_ivs[i] for i in order], dtype=float),
                    "ssvi_params": global_params,  # [rho_0, rho_infty, lambda, eta, gamma]
                })

            rows = []
            for strike in sorted(quotes_by_strike.keys()):
                quotes = quotes_by_strike[strike]
                
                if global_params is not None:
                    k = np.log(strike / spot)
                    rho_0, rho_infty, lam, eta, gamma = global_params
                    w_fit = OptionsPricer.ssvi_total_variance(k, theta, rho_0, rho_infty, lam, eta, gamma)
                    iv_svi = np.sqrt(max(w_fit, 0.0) / t)
                else:
                    iv_svi = 0.0

                rows.append(
                    {
                        "Strike": f"{strike:.1f}",
                        "Call Bid": f"{quotes['C'].bid:.2f}" if "C" in quotes else "N/A",
                        "Call Ask": f"{quotes['C'].ask:.2f}" if "C" in quotes else "N/A",
                        "Call IV": f"{quotes['C'].implied_vol * 100:.1f}%" if "C" in quotes else "N/A",
                        "SSVI IV": f"{iv_svi * 100:.1f}%",
                        "Put IV": f"{quotes['P'].implied_vol * 100:.1f}%" if "P" in quotes else "N/A",
                        "Put Ask": f"{quotes['P'].ask:.2f}" if "P" in quotes else "N/A",
                        "Put Bid": f"{quotes['P'].bid:.2f}" if "P" in quotes else "N/A",
                        "Call OBI": self._format_obi(quotes["C"].order_book_imbalance) if "C" in quotes else "N/A",
                        "Put OBI": self._format_obi(quotes["P"].order_book_imbalance) if "P" in quotes else "N/A",
                    }
                )

            if rows:
                radar_data[expiry] = pd.DataFrame(rows)

        return radar_data, surface_slices