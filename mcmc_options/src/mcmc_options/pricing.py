"""
pricing.py — Option pricing logic: Monte Carlo and closed-form Black-Scholes.

European Options
----------------
A European call pays max(S_T - K, 0) at maturity T.
A European put  pays max(K - S_T, 0) at maturity T.

Under risk-neutral pricing the fair value is the discounted expected payoff:

    Call = e^{-rT} · E^Q[max(S_T - K, 0)]
    Put  = e^{-rT} · E^Q[max(K - S_T, 0)]

Monte Carlo approximates E^Q[·] by the sample mean over n_simulations paths.

The Black-Scholes closed-form solution serves as the benchmark:

    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
    Call = S·N(d1) - K·e^{-rT}·N(d2)
    Put  = K·e^{-rT}·N(-d2) - S·N(-d1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .config import SimulationConfig
from .simulation import simulate_gbm_paths


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MCPricingResult:
    """Holds Monte Carlo pricing output with diagnostics."""
    call_price: float
    put_price: float
    call_stderr: float          # standard error of the MC estimator
    put_stderr: float
    n_simulations: int
    use_antithetic: bool

    def __str__(self) -> str:
        return (
            f"Monte Carlo Pricing Result\n"
            f"  Call : {self.call_price:>10.4f}  (±{self.call_stderr:.4f})\n"
            f"  Put  : {self.put_price:>10.4f}  (±{self.put_stderr:.4f})\n"
            f"  Paths: {self.n_simulations:,}  |  antithetic={self.use_antithetic}"
        )


@dataclass
class BSPricingResult:
    """Holds closed-form Black-Scholes pricing output."""
    call_price: float
    put_price: float
    d1: float
    d2: float

    def __str__(self) -> str:
        return (
            f"Black-Scholes Analytical Result\n"
            f"  Call : {self.call_price:>10.4f}\n"
            f"  Put  : {self.put_price:>10.4f}\n"
            f"  d1   : {self.d1:>10.4f}   d2: {self.d2:.4f}"
        )


@dataclass
class ComparisonResult:
    """Side-by-side comparison of MC vs analytical prices."""
    mc: MCPricingResult
    bs: BSPricingResult

    @property
    def call_error(self) -> float:
        return abs(self.mc.call_price - self.bs.call_price)

    @property
    def put_error(self) -> float:
        return abs(self.mc.put_price - self.bs.put_price)

    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"{self.mc}\n\n"
            f"{self.bs}\n\n"
            f"Absolute errors\n"
            f"  Call |MC - BS| = {self.call_error:.4f}\n"
            f"  Put  |MC - BS| = {self.put_error:.4f}\n"
            f"{'='*50}"
        )


# ---------------------------------------------------------------------------
# Monte Carlo pricer
# ---------------------------------------------------------------------------

def price_european_mc(
    cfg: SimulationConfig,
    terminal_prices: np.ndarray | None = None,
) -> MCPricingResult:
    """
    Price European call and put options via Monte Carlo simulation.

    Parameters
    ----------
    cfg : SimulationConfig
        Model and simulation configuration.
    terminal_prices : np.ndarray, optional
        Pre-computed terminal prices (shape: (n_simulations,)).
        If None, paths are simulated from cfg internally.

    Returns
    -------
    MCPricingResult
        Call/put prices with standard errors.

    Algorithm
    ---------
    1. Simulate n_simulations GBM paths (or use supplied terminal prices).
    2. Compute payoffs at maturity:
           call_payoff = max(S_T - K, 0)
           put_payoff  = max(K - S_T, 0)
    3. Discount the sample mean by e^{-rT}.
    4. Standard error = std(payoff) / sqrt(n) (CLT confidence interval).
    """
    if terminal_prices is None:
        paths = simulate_gbm_paths(cfg)
        terminal_prices = paths[:, -1]

    discount = np.exp(-cfg.r * cfg.T)

    call_payoffs = np.maximum(terminal_prices - cfg.K, 0.0)
    put_payoffs  = np.maximum(cfg.K - terminal_prices, 0.0)

    n = len(terminal_prices)
    call_price = discount * call_payoffs.mean()
    put_price  = discount * put_payoffs.mean()

    # Standard error of the estimator (not of individual payoffs)
    call_stderr = discount * call_payoffs.std(ddof=1) / np.sqrt(n)
    put_stderr  = discount * put_payoffs.std(ddof=1)  / np.sqrt(n)

    return MCPricingResult(
        call_price=float(call_price),
        put_price=float(put_price),
        call_stderr=float(call_stderr),
        put_stderr=float(put_stderr),
        n_simulations=n,
        use_antithetic=cfg.use_antithetic,
    )


# ---------------------------------------------------------------------------
# Analytical Black-Scholes pricer
# ---------------------------------------------------------------------------

def price_european_bs(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> BSPricingResult:
    """
    Price European call and put via the Black-Scholes closed-form formula.

    Parameters
    ----------
    S     : float  — Current spot price.
    K     : float  — Strike price.
    r     : float  — Risk-free rate (annualised, continuous compounding).
    sigma : float  — Volatility (annualised).
    T     : float  — Time to maturity (years).

    Returns
    -------
    BSPricingResult

    Raises
    ------
    ValueError
        If sigma == 0 or T == 0 (degenerate cases handled separately).
    """
    if sigma == 0.0 or T == 0.0:
        # Degenerate case: no uncertainty → payoff is deterministic
        fwd = S * np.exp(r * T)
        call = np.exp(-r * T) * max(fwd - K, 0.0)
        put  = np.exp(-r * T) * max(K - fwd, 0.0)
        return BSPricingResult(call_price=call, put_price=put, d1=0.0, d2=0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return BSPricingResult(
        call_price=float(call),
        put_price=float(put),
        d1=float(d1),
        d2=float(d2),
    )


def price_european_bs_from_config(cfg: SimulationConfig) -> BSPricingResult:
    """Convenience wrapper: build BSPricingResult directly from a SimulationConfig."""
    return price_european_bs(cfg.S0, cfg.K, cfg.r, cfg.sigma, cfg.T)


# ---------------------------------------------------------------------------
# Vectorised surface computation
# ---------------------------------------------------------------------------

def compute_price_surface(
    S_grid: np.ndarray,
    T_grid: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    method: str = "bs",
    n_simulations: int = 20_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute option prices over a 2-D (S, T) grid.

    Parameters
    ----------
    S_grid : np.ndarray, shape (n_S,)
        Array of spot prices.
    T_grid : np.ndarray, shape (n_T,)
        Array of maturities.
    K, r, sigma : float
        Fixed option parameters.
    option_type : {"call", "put"}
    method : {"bs", "mc"}
        "bs" uses the analytical formula (fast, exact).
        "mc" runs a full Monte Carlo per grid point (slow, approximate).
    n_simulations : int
        Used only when method="mc".
    seed : int
        Base random seed; each grid point uses seed + flat_index.

    Returns
    -------
    prices : np.ndarray, shape (n_T, n_S)
        Option prices on the grid.
    """
    n_S, n_T = len(S_grid), len(T_grid)
    prices = np.zeros((n_T, n_S))

    for i, T in enumerate(T_grid):
        for j, S in enumerate(S_grid):
            if method == "bs":
                result = price_european_bs(S, K, r, sigma, T)
                prices[i, j] = (
                    result.call_price if option_type == "call" else result.put_price
                )
            else:
                cfg = SimulationConfig(
                    S0=S, K=K, r=r, sigma=sigma, T=T,
                    n_simulations=n_simulations,
                    n_steps=max(int(T * 252), 1),
                    seed=seed + i * n_S + j,
                    use_antithetic=True,
                )
                result = price_european_mc(cfg)
                prices[i, j] = (
                    result.call_price if option_type == "call" else result.put_price
                )

    return prices
