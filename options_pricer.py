"""Black-Scholes pricing and SVI volatility surface calibration.

Stateless module: every function receives inputs explicitly and returns
results without maintaining state. Safe for concurrent calls.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


class OptionsPricer:
    """Stateless collection of Black-Scholes and SVI pricing routines."""

    @staticmethod
    def compute_d1(spot: float, strike: float, t: float, r: float, sigma: float, div_yield: float) -> float:
        """Computes Black-Scholes d1 term."""
        sigma, t = max(sigma, 1e-4), max(t, 1e-6)
        return (np.log(spot / strike) + (r - div_yield + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    @staticmethod
    def compute_d2(d1: float, sigma: float, t: float) -> float:
        """Computes Black-Scholes d2 term from precomputed d1."""
        return d1 - max(sigma, 1e-4) * np.sqrt(max(t, 1e-6))

    @staticmethod
    def theoretical_call_price(spot: float, strike: float, t: float, r: float, sigma: float, div_yield: float) -> float:
        """European call price under Black-Scholes."""
        d1 = OptionsPricer.compute_d1(spot, strike, t, r, sigma, div_yield)
        d2 = OptionsPricer.compute_d2(d1, sigma, t)
        return spot * np.exp(-div_yield * t) * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)

    @staticmethod
    def theoretical_put_price(spot: float, strike: float, t: float, r: float, sigma: float, div_yield: float) -> float:
        """European put price under Black-Scholes."""
        d1 = OptionsPricer.compute_d1(spot, strike, t, r, sigma, div_yield)
        d2 = OptionsPricer.compute_d2(d1, sigma, t)
        return strike * np.exp(-r * t) * norm.cdf(-d2) - spot * np.exp(-div_yield * t) * norm.cdf(-d1)

    @staticmethod
    def compute_vega(spot: float, strike: float, t: float, r: float, sigma: float, div_yield: float) -> float:
        """Vega: sensitivity of option price to volatility change."""
        d1 = OptionsPricer.compute_d1(spot, strike, t, r, sigma, div_yield)
        return spot * np.sqrt(max(t, 1e-6)) * np.exp(-div_yield * t) * norm.pdf(d1)

    @staticmethod
    def implied_vol_bisection(
        market_price: float,
        spot: float,
        strike: float,
        t: float,
        r: float,
        div_yield: float,
        right: str,
        n_iter: int = 30,
        tol: float = 1e-5,
    ) -> float:
        """Solves for implied volatility via bisection on Black-Scholes price.

        Args:
            market_price: Observed mid price of the option (bid/ask average).
            spot: Current price of the underlying.
            strike: Strike price.
            t: Time to expiration, in years.
            r: Risk-free rate.
            div_yield: Continuous dividend yield.
            right: 'C' for call, 'P' for put.
            n_iter: Maximum number of bisection iterations.
            tol: Convergence tolerance on the volatility bracket width.

        Returns:
            Implied volatility as a decimal (e.g. 0.25 for 25%).
        """
        intrinsic = max(0.0, spot - strike) if right == "C" else max(0.0, strike - spot)
        if market_price <= intrinsic or np.isnan(market_price):
            return 0.001

        low, high = 0.001, 4.0
        mid = low
        for _ in range(n_iter):
            mid = (low + high) / 2
            if right == "C":
                theo_price = OptionsPricer.theoretical_call_price(spot, strike, t, r, mid, div_yield)
            else:
                theo_price = OptionsPricer.theoretical_put_price(spot, strike, t, r, mid, div_yield)

            if theo_price > market_price:
                high = mid
            else:
                low = mid
            if (high - low) < tol:
                break
        return mid

    @staticmethod
    def time_to_expiry(expiry_str: str) -> float:
        """Converts IBKR expiry string ('%Y%m%d') to year-fraction."""
        expiry_date = datetime.strptime(expiry_str, "%Y%m%d")
        delta = expiry_date - datetime.now()
        days = delta.days + delta.seconds / 86400
        return max(days / 365, 1e-6)

    @staticmethod
    def svi_total_variance(
        k: NDArray[np.float64], a: float, b: float, rho: float, m: float, sigma: float
    ) -> NDArray[np.float64]:
        """Raw SVI parameterization: w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))."""
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    @staticmethod
    def _svi_objective(params: NDArray[np.float64], k_arr: NDArray[np.float64], w_observed: NDArray[np.float64]) -> float:
        """Sum of squared errors between observed and SVI-model total variance."""
        a, b, rho, m, sigma = params
        w_model = OptionsPricer.svi_total_variance(k_arr, a, b, rho, m, sigma)
        return float(np.sum((w_observed - w_model) ** 2))

    @staticmethod
    def calibrate_svi(
        strikes: Sequence[float], ivs: Sequence[float], spot: float, t: float
    ) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """Calibrates raw SVI surface (a, b, rho, m, sigma) to observed IVs.

        Falls back to quadratic polynomial fit if L-BFGS-B does not converge,
        so callers always get a usable fitted curve back.

        Args:
            strikes: Strike prices used for calibration.
            ivs: Observed implied volatilities (decimal) at each strike.
            spot: Current underlying price.
            t: Time to expiry, in years.

        Returns:
            Tuple of (fitted implied vols at the input strikes, SVI params array or None).
        """
        strikes_arr = np.asarray(strikes, dtype=float)
        ivs_arr = np.asarray(ivs, dtype=float)
        k_arr = np.log(strikes_arr / spot)
        w_observed = (ivs_arr ** 2) * t

        x0 = [0.1 * t, 0.1, 0.0, 0.0, 0.1]
        bounds = [(1e-5, None), (0.0, 5.0), (-0.99, 0.99), (-2.0, 2.0), (1e-5, 1.0)]

        result = minimize(
            OptionsPricer._svi_objective, x0, args=(k_arr, w_observed), method="L-BFGS-B", bounds=bounds
        )

        if result.success:
            a_opt, b_opt, rho_opt, m_opt, sigma_opt = result.x
            w_fit = OptionsPricer.svi_total_variance(k_arr, a_opt, b_opt, rho_opt, m_opt, sigma_opt)
            iv_fit = np.sqrt(np.maximum(w_fit, 0) / t)
            return iv_fit, result.x

        logger.warning("SVI calibration did not converge for spot=%.2f, t=%.4f; using quadratic fallback.", spot, t)
        coeffs = np.polyfit(strikes_arr, ivs_arr, 2)
        return np.polyval(coeffs, strikes_arr), None
