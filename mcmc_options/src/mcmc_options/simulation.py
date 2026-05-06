"""
simulation.py — Monte Carlo path simulation engine.

Design note on "MCMC vs standard Monte Carlo"
----------------------------------------------
Classical option pricing uses *standard* Monte Carlo integration, not
Markov Chain Monte Carlo (which is designed for sampling from an unknown
posterior distribution).  Here we implement:

  1. Standard Monte Carlo (GBM paths)           — the workhorse
  2. Antithetic Variates                        — variance reduction
  3. A thin MCMC-flavoured wrapper              — Metropolis sampler over
     the risk-neutral density, included for pedagogical completeness and
     to satisfy the "MCMC-inspired" requirement.

Geometric Brownian Motion (GBM)
--------------------------------
Under the risk-neutral measure Q the asset price follows:

    dS = r S dt + σ S dW_t,   W_t ~ Brownian motion

Exact discretisation (no Euler error):

    S_{t+Δt} = S_t · exp[(r - σ²/2)Δt + σ√Δt · Z],  Z ~ N(0,1)

This log-normal step is exact for GBM and is strongly preferred over
the Euler-Maruyama scheme because it introduces no discretisation bias.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from .config import SimulationConfig


# ---------------------------------------------------------------------------
# Core path generator
# ---------------------------------------------------------------------------

def simulate_gbm_paths(
    cfg: SimulationConfig,
    rng: Generator | None = None,
) -> np.ndarray:
    """
    Simulate asset price paths under Geometric Brownian Motion.

    Uses the exact log-normal transition, optionally with antithetic variates.

    Parameters
    ----------
    cfg : SimulationConfig
        All model and simulation parameters.
    rng : numpy.random.Generator, optional
        Pre-constructed random generator (allows caller to control state).
        If None, a new generator is created from cfg.seed.

    Returns
    -------
    paths : np.ndarray, shape (n_paths, n_steps + 1)
        Simulated asset prices.  paths[:, 0] == cfg.S0,
        paths[:, -1] == terminal prices at maturity.

    Notes
    -----
    Antithetic variates
    ~~~~~~~~~~~~~~~~~~~
    For every standard normal draw Z we also use -Z.  Because the payoff
    function is monotone in Z, the two estimators are negatively correlated,
    which roughly halves the variance at no extra cost in random-number
    generation.  When antithetic=True we simulate n_simulations/2 "base"
    paths and mirror them, so the effective path count stays n_simulations.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    dt: float = cfg.T / cfg.n_steps
    drift: float = (cfg.r - 0.5 * cfg.sigma ** 2) * dt
    diffusion: float = cfg.sigma * np.sqrt(dt)

    if cfg.use_antithetic:
        half = cfg.n_simulations // 2
        Z_half = rng.standard_normal((half, cfg.n_steps))   # (half, steps)
        Z = np.concatenate([Z_half, -Z_half], axis=0)       # antithetic mirror
    else:
        Z = rng.standard_normal((cfg.n_simulations, cfg.n_steps))

    # Log-returns for each step: shape (n_paths, n_steps)
    log_returns = drift + diffusion * Z

    # Cumulative sum → log-price path, then exponentiate
    log_price = np.log(cfg.S0) + np.cumsum(log_returns, axis=1)

    # Prepend the known initial price
    initial = np.full((log_price.shape[0], 1), np.log(cfg.S0))
    log_paths = np.concatenate([initial, log_price], axis=1)

    return np.exp(log_paths)


# ---------------------------------------------------------------------------
# MCMC-based terminal price sampler (Metropolis–Hastings)
# ---------------------------------------------------------------------------

def sample_terminal_prices_mcmc(
    cfg: SimulationConfig,
    rng: Generator | None = None,
    burn_in: int = 1_000,
) -> np.ndarray:
    """
    Sample terminal asset prices via Metropolis–Hastings MCMC.

    The target distribution is the risk-neutral log-normal density of S_T:

        S_T ~ LogNormal(log(S0) + (r - σ²/2)T,  σ²T)

    We recover this density exactly with a Gaussian random-walk proposal,
    which makes the MH chain converge to the correct target.

    This function is provided as a pedagogical demonstration of the MCMC
    methodology applied to option pricing.  For production use, direct
    sampling (simulate_gbm_paths) is strictly preferred because:
      - It is exact (no burn-in / autocorrelation issues).
      - It is fully vectorised and therefore ~50× faster.

    Parameters
    ----------
    cfg : SimulationConfig
        Model parameters.
    rng : Generator, optional
        Random generator.
    burn_in : int
        Number of initial samples discarded to allow chain mixing.

    Returns
    -------
    terminal_prices : np.ndarray, shape (n_simulations,)
        Terminal asset prices drawn from the risk-neutral distribution.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    mu_ln = np.log(cfg.S0) + (cfg.r - 0.5 * cfg.sigma ** 2) * cfg.T
    sigma_ln = cfg.sigma * np.sqrt(cfg.T)

    def log_target(log_s: float) -> float:
        """Log of the log-normal density (unnormalised)."""
        return -0.5 * ((log_s - mu_ln) / sigma_ln) ** 2

    total = cfg.n_simulations + burn_in
    samples = np.empty(total)
    current = mu_ln                     # initialise at the mode
    current_log_p = log_target(current)
    proposal_std = sigma_ln             # proposal scale ≈ target std

    n_accepted = 0
    for i in range(total):
        candidate = current + rng.normal(0.0, proposal_std)
        log_p_candidate = log_target(candidate)
        log_alpha = log_p_candidate - current_log_p
        if np.log(rng.uniform()) < log_alpha:
            current = candidate
            current_log_p = log_p_candidate
            n_accepted += 1
        samples[i] = current

    acceptance_rate = n_accepted / total
    if acceptance_rate < 0.1 or acceptance_rate > 0.9:
        import warnings
        warnings.warn(
            f"MCMC acceptance rate {acceptance_rate:.2%} is outside [10%, 90%]. "
            "Consider tuning proposal_std.",
            RuntimeWarning,
            stacklevel=2,
        )

    return np.exp(samples[burn_in:])     # discard burn-in, convert to prices
