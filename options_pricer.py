"""Black-Scholes pricing and SSVI with time-dependent rho(theta) calibration.

SSVI (Surface Stochastic Volatility Inspired) model augmented with an
exponential decay function for rho: rho(theta) = rho_infty + (rho_0 - rho_infty)*exp(-lambda*theta).
This allows the skew to change naturally across vencimientos: short-dated options
(small theta, affected by events) show stronger negative skew (low rho_0), while
longer-dated options (large theta, normal regime) show milder skew (higher rho_infty).

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

# Candidate starting points for the global optimizer: (rho0, rho_infty, lambda, eta, gamma).
# With rho(theta) parametrized via exponential decay, we explore different decay rates
# and skew regimes. Multi-start helps avoid local minima when theta ranges widely.
_SSVI_SEEDS = [
    (0.0, -0.1, 0.5, 1.0, 0.5),
    (-0.5, -0.1, 1.0, 1.0, 0.4),
    (-0.3, 0.0, 0.3, 0.8, 0.5),
    (-0.6, -0.05, 0.8, 1.2, 0.3),
    (-0.2, -0.15, 2.0, 0.9, 0.6),
]

_ETA_DEGENERATE_FLOOR = 0.05


class OptionsPricer:
    """Stateless collection of Black-Scholes and global SSVI+rho(theta) pricing routines."""

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
    def rho_theta(theta: NDArray[np.float64] | float, rho_0: float, rho_infty: float, lam: float) -> NDArray[np.float64] | float:
        """Exponential decay parameterization of rho as a function of theta (ATM total variance).

        rho(theta) = rho_infty + (rho_0 - rho_infty) * exp(-lambda * theta)

        At short theta (small ATM variance, e.g., pre-earnings): rho ≈ rho_0 (can be very negative for strong skew).
        At large theta (long-dated, normal regime): rho ≈ rho_infty (typically mildly negative or zero).
        This captures the natural decay of skew across the term structure.
        """
        theta = np.maximum(theta, 1e-6)
        return rho_infty + (rho_0 - rho_infty) * np.exp(-lam * theta)

    @staticmethod
    def ssvi_total_variance(
        k: NDArray[np.float64], 
        theta: NDArray[np.float64] | float, 
        rho_0: float, 
        rho_infty: float, 
        lam: float, 
        eta: float, 
        gamma: float
    ) -> NDArray[np.float64]:
        """Global SSVI with theta-dependent rho: w(k, theta) = (theta/2) * (1 + rho(theta)*phi*k + sqrt(...))."""
        rho = OptionsPricer.rho_theta(theta, rho_0, rho_infty, lam)
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
        """Sum of squared, theta-weighted errors for SSVI+rho(theta) surface.

        Params: [rho_0, rho_infty, lambda, eta, gamma].
        """
        rho_0, rho_infty, lam, eta, gamma = params
        
        # No-arbitrage boundary on (rho, eta) pairs: eta * (1 + |rho|) <= 2
        # For rho(theta), check bounds over the observed theta range.
        rho_range = OptionsPricer.rho_theta(np.array([theta_arr.min(), theta_arr.max()]), rho_0, rho_infty, lam)
        max_rho_bound = eta * (1.0 + np.max(np.abs(rho_range)))
        if max_rho_bound > 2.0:
            return 1e6
            
        w_model = OptionsPricer.ssvi_total_variance(k_arr, theta_arr, rho_0, rho_infty, lam, eta, gamma)
        weights = 1.0 / np.maximum(theta_arr, 1e-6)
        return float(np.sum(weights * (w_obs - w_model) ** 2))

    @staticmethod
    def calibrate_global_ssvi(
        k_arr: NDArray[np.float64], 
        theta_arr: NDArray[np.float64], 
        w_obs: NDArray[np.float64]
    ) -> Optional[NDArray[np.float64]]:
        """Calibrates SSVI+rho(theta) surface globally, multi-start L-BFGS-B.

        With rho now a function of theta, the skew can adapt across the term structure,
        allowing short-dated expiries (pre-earnings, high theta variation) and
        long-dated expiries (stable regime) to have naturally different skews within
        a single coherent surface.

        Args:
            k_arr: 1D array of log-moneyness for all OTM options across all expiries.
            theta_arr: 1D array of ATM total variances corresponding to each option's expiry.
            w_obs: 1D array of observed market total variances.

        Returns:
            Optimal parameters [rho_0, rho_infty, lambda, eta, gamma] if successful, else None.
        """
        if len(k_arr) < 5:
            return None

        # Bounds: rho_0, rho_infty in (-0.99, 0.99), lambda > 0, eta > 0, gamma in (0, 1)
        bounds = [(-0.99, 0.99), (-0.99, 0.99), (0.01, 10.0), (1e-5, 5.0), (0.0, 1.0)]

        best_result = None
        best_score = np.inf

        for x0 in _SSVI_SEEDS:
            result = minimize(
                OptionsPricer._ssvi_objective,
                x0,
                args=(k_arr, theta_arr, w_obs),
                method="L-BFGS-B",
                bounds=bounds,
            )

            if not result.success:
                continue

            rho_0_fit, rho_infty_fit, lam_fit, eta_fit, gamma_fit = result.x
            
            # Check arbitrage bound
            rho_range = OptionsPricer.rho_theta(np.array([theta_arr.min(), theta_arr.max()]), rho_0_fit, rho_infty_fit, lam_fit)
            if eta_fit * (1.0 + np.max(np.abs(rho_range))) > 2.0:
                continue

            is_degenerate = eta_fit < _ETA_DEGENERATE_FLOOR
            score = result.fun + (1e3 if is_degenerate else 0.0)

            if score < best_score:
                best_score = score
                best_result = result.x

        if best_result is None:
            logger.warning("Global SSVI+rho(theta) calibration failed across all seeds or hit arbitrage bounds.")
            return None

        if best_result[3] < _ETA_DEGENERATE_FLOOR:
            logger.warning(
                "SSVI+rho(theta) calibration converged but eta near-degenerate (eta=%.5f); "
                "surface curvature may be weak.", best_result[3]
            )

        return best_result