"""Black-Scholes pricing and SSVI (Surface Stochastic Volatility Inspired) calibration.

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
    """Stateless collection of Black-Scholes and global SSVI pricing routines."""

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
        """Solves for implied volatility via bisection on Black-Scholes price."""
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
    def power_law_phi(theta: NDArray[np.float64] | float, eta: float, gamma: float) -> NDArray[np.float64] | float:
        """Computes the Power-Law scale function phi(theta) for SSVI."""
        theta = np.maximum(theta, 1e-6)
        return eta / ((theta ** gamma) * ((1.0 + theta) ** (1.0 - gamma)))

    @staticmethod
    def ssvi_total_variance(
        k: NDArray[np.float64], theta: NDArray[np.float64] | float, rho: float, eta: float, gamma: float
    ) -> NDArray[np.float64]:
        """Global SSVI parameterization: w(k, theta) = (theta/2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))."""
        phi = OptionsPricer.power_law_phi(theta, eta, gamma)
        sqrt_term = np.sqrt((phi * k + rho) ** 2 + 1.0 - rho ** 2)
        return (theta / 2.0) * (1.0 + rho * phi * k + sqrt_term)

    @staticmethod
    def _ssvi_objective(
        params: NDArray[np.float64], 
        k_arr: NDArray[np.float64], 
        theta_arr: NDArray[np.float64], 
        w_obs: NDArray[np.float64]
    ) -> float:
        """Sum of squared errors between observed market variance and global SSVI surface."""
        rho, eta, gamma = params
        
        # Absolute no-arbitrage boundary penalty
        if eta * (1.0 + abs(rho)) > 2.0:
            return 1e6
            
        w_model = OptionsPricer.ssvi_total_variance(k_arr, theta_arr, rho, eta, gamma)
        return float(np.sum((w_obs - w_model) ** 2))

    @staticmethod
    def calibrate_global_ssvi(
        k_arr: NDArray[np.float64], 
        theta_arr: NDArray[np.float64], 
        w_obs: NDArray[np.float64]
    ) -> Optional[NDArray[np.float64]]:
        """Calibrates the entire SSVI surface globally in a single L-BFGS-B pass.

        Args:
            k_arr: 1D array of log-moneyness for all OTM options across all expiries.
            theta_arr: 1D array of ATM total variances corresponding to each option's expiry.
            w_obs: 1D array of observed market total variances.

        Returns:
            Optimal parameters [rho, eta, gamma] if successful and arbitrage-free, else None.
        """
        if len(k_arr) < 5:
            return None

        # Initial seed: zero correlation, neutral curvature, linear decay
        x0 = [0.0, 1.0, 0.5]
        # Bounds: rho in (-1, 1), eta > 0, gamma in (0, 1)
        bounds = [(-0.99, 0.99), (1e-5, 5.0), (0.0, 1.0)]

        result = minimize(
            OptionsPricer._ssvi_objective, 
            x0, 
            args=(k_arr, theta_arr, w_obs), 
            method="L-BFGS-B", 
            bounds=bounds
        )

        if result.success and result.x[1] * (1.0 + abs(result.x[0])) <= 2.0:
            return result.x

        logger.warning("Global SSVI calibration failed or hit arbitrage bounds.")
        return None