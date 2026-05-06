"""
conftest.py — Shared pytest fixtures.
"""

import pytest
from mcmc_options.config import SimulationConfig


@pytest.fixture
def base_cfg() -> SimulationConfig:
    """ATM option, standard BS parameters, reproducible seed."""
    return SimulationConfig(
        S0=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
        n_simulations=50_000, n_steps=252, seed=0, use_antithetic=True,
    )


@pytest.fixture
def itm_call_cfg() -> SimulationConfig:
    """Deep in-the-money call: S0 >> K."""
    return SimulationConfig(
        S0=150.0, K=100.0, r=0.05, sigma=0.20, T=1.0,
        n_simulations=50_000, n_steps=252, seed=1, use_antithetic=True,
    )


@pytest.fixture
def low_vol_cfg() -> SimulationConfig:
    """Near-zero volatility: MC should converge tightly to BS."""
    return SimulationConfig(
        S0=100.0, K=100.0, r=0.05, sigma=0.001, T=1.0,
        n_simulations=50_000, n_steps=252, seed=2, use_antithetic=True,
    )
