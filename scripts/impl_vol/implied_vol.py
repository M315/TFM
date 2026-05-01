"""
Black-Scholes-Merton (1973) call and put prices with continuous dividend yield,
shared vega, and implied volatility inversion.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_call(S: float, K: float, T: float, r: float, sigma: float,
            q: float = 0.0) -> float:
    """BSM European call:  S·e^{-qT}·N(d1) − K·e^{-rT}·N(d2)."""
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float,
           q: float = 0.0) -> float:
    """BSM European put:  K·e^{-rT}·N(−d2) − S·e^{-qT}·N(−d1)."""
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float,
            q: float = 0.0) -> float:
    """BSM vega  dV/dσ = S·e^{-qT}·√T·φ(d1)  — identical for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def implied_vol(
    V_mkt: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Compute implied volatility by inverting the BSM formula.

    option_type: "call" or "put".
    q: continuous dividend yield.
    Uses Newton–Raphson with Brent as fallback.
    Returns NaN if the market price lies outside the no-arbitrage bounds.
    """
    Se = S * np.exp(-q * T)   # dividend-adjusted spot
    Ke = K * np.exp(-r * T)   # discounted strike

    if option_type == "call":
        bs_price = bs_call
        if V_mkt <= max(Se - Ke, 0.0) or V_mkt >= Se:
            return np.nan
    else:
        bs_price = bs_put
        if V_mkt <= max(Ke - Se, 0.0) or V_mkt >= Ke:
            return np.nan

    sigma = 0.3
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, q)
        vega  = bs_vega(S, K, T, r, sigma, q)
        if abs(vega) < 1e-12:
            break
        sigma_new = sigma - (price - V_mkt) / vega
        if sigma_new <= 0:
            break
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new

    try:
        return brentq(
            lambda s: bs_price(S, K, T, r, s, q) - V_mkt,
            1e-6, 10.0, xtol=tol,
        )
    except ValueError:
        return np.nan
