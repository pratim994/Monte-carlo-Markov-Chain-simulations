"""
tests/test_pricing.py — Unit tests for pricing logic.

Covers:
  - Black-Scholes analytical formula correctness
  - Monte Carlo vs Black-Scholes agreement
  - Put-Call Parity
  - Edge cases (deep ITM/OTM, zero vol)
  - Convergence: more paths → smaller error
"""

import dataclasses

import numpy as np
import pytest

from mcmc_options.config import SimulationConfig
from mcmc_options.pricing import (
    price_european_bs,
    price_european_bs_from_config,
    price_european_mc,
    ComparisonResult,
)


# ---------------------------------------------------------------------------
# Black-Scholes analytical tests
# ---------------------------------------------------------------------------

class TestBlackScholes:

    def test_atm_call_known_value(self) -> None:
        """ATM call with known closed-form result (S=K=100, r=5%, σ=20%, T=1)."""
        result = price_european_bs(S=100, K=100, r=0.05, sigma=0.20, T=1.0)
        # Accepted reference value ≈ 10.4506
        np.testing.assert_allclose(result.call_price, 10.4506, atol=1e-3)

    def test_atm_put_known_value(self) -> None:
        result = price_european_bs(S=100, K=100, r=0.05, sigma=0.20, T=1.0)
        np.testing.assert_allclose(result.put_price, 5.5735, atol=1e-3)

    def test_put_call_parity(self) -> None:
        """C - P = S - K·e^{-rT}  must hold exactly."""
        for S in [80, 100, 120]:
            res = price_european_bs(S=S, K=100, r=0.05, sigma=0.20, T=1.0)
            lhs = res.call_price - res.put_price
            rhs = S - 100 * np.exp(-0.05 * 1.0)
            np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_call_price_non_negative(self) -> None:
        for S in np.linspace(50, 150, 10):
            res = price_european_bs(S=float(S), K=100, r=0.05, sigma=0.20, T=1.0)
            assert res.call_price >= 0
            assert res.put_price >= 0

    def test_call_lower_bound(self) -> None:
        """Call price ≥ max(S - K·e^{-rT}, 0)."""
        res = price_european_bs(S=100, K=90, r=0.05, sigma=0.20, T=1.0)
        lower_bound = max(100 - 90 * np.exp(-0.05 * 1.0), 0)
        assert res.call_price >= lower_bound - 1e-10

    def test_zero_volatility_degenerate(self) -> None:
        """sigma=0: option value = max discounted intrinsic value."""
        S, K, r, T = 100.0, 95.0, 0.05, 1.0
        res = price_european_bs(S=S, K=K, r=r, sigma=0.0, T=T)
        fwd = S * np.exp(r * T)
        expected_call = np.exp(-r * T) * max(fwd - K, 0.0)
        np.testing.assert_allclose(res.call_price, expected_call, atol=1e-10)

    def test_deep_itm_call_approaches_intrinsic(self) -> None:
        """Deep ITM call (S >> K) → price ≈ S - K·e^{-rT}."""
        res = price_european_bs(S=500, K=100, r=0.05, sigma=0.20, T=1.0)
        intrinsic = 500 - 100 * np.exp(-0.05 * 1.0)
        np.testing.assert_allclose(res.call_price, intrinsic, rtol=0.001)

    def test_deep_otm_call_near_zero(self) -> None:
        """Deep OTM call (S << K) → price ≈ 0."""
        res = price_european_bs(S=10, K=200, r=0.05, sigma=0.20, T=1.0)
        assert res.call_price < 0.01

    def test_increasing_sigma_increases_option_value(self) -> None:
        """Higher volatility → higher option value (vega > 0)."""
        r1 = price_european_bs(S=100, K=100, r=0.05, sigma=0.10, T=1.0)
        r2 = price_european_bs(S=100, K=100, r=0.05, sigma=0.40, T=1.0)
        assert r2.call_price > r1.call_price
        assert r2.put_price > r1.put_price

    def test_increasing_T_increases_option_value(self) -> None:
        """Longer maturity → higher value (positive time value)."""
        r1 = price_european_bs(S=100, K=100, r=0.05, sigma=0.20, T=0.25)
        r2 = price_european_bs(S=100, K=100, r=0.05, sigma=0.20, T=2.00)
        assert r2.call_price > r1.call_price


# ---------------------------------------------------------------------------
# Monte Carlo pricing tests
# ---------------------------------------------------------------------------

class TestMonteCarloPricing:

    def test_mc_close_to_bs_atm(self, base_cfg: SimulationConfig) -> None:
        """ATM option: MC estimate should be within 0.30 of Black-Scholes."""
        mc = price_european_mc(base_cfg)
        bs = price_european_bs_from_config(base_cfg)
        np.testing.assert_allclose(mc.call_price, bs.call_price, atol=0.30)
        np.testing.assert_allclose(mc.put_price,  bs.put_price,  atol=0.30)

    def test_mc_call_within_2_stderr(self, base_cfg: SimulationConfig) -> None:
        """MC call price should be within 2 standard errors of BS most of the time."""
        mc = price_european_mc(base_cfg)
        bs = price_european_bs_from_config(base_cfg)
        assert abs(mc.call_price - bs.call_price) < 2 * mc.call_stderr + 0.05

    def test_mc_low_volatility_convergence(self, low_vol_cfg: SimulationConfig) -> None:
        """Near-zero volatility: MC should converge very tightly to BS."""
        mc = price_european_mc(low_vol_cfg)
        bs = price_european_bs_from_config(low_vol_cfg)
        np.testing.assert_allclose(mc.call_price, bs.call_price, atol=0.05)
        np.testing.assert_allclose(mc.put_price,  bs.put_price,  atol=0.05)

    def test_mc_stderr_decreases_with_more_paths(self, base_cfg: SimulationConfig) -> None:
        """Standard error should decrease with more simulations (CLT)."""
        cfg_small = dataclasses.replace(base_cfg, n_simulations=1_000)
        cfg_large = dataclasses.replace(base_cfg, n_simulations=50_000)
        mc_small = price_european_mc(cfg_small)
        mc_large = price_european_mc(cfg_large)
        assert mc_large.call_stderr < mc_small.call_stderr

    def test_mc_put_call_parity(self, base_cfg: SimulationConfig) -> None:
        """MC should approximately satisfy put-call parity."""
        mc = price_european_mc(base_cfg)
        lhs = mc.call_price - mc.put_price
        rhs = base_cfg.S0 - base_cfg.K * np.exp(-base_cfg.r * base_cfg.T)
        np.testing.assert_allclose(lhs, rhs, atol=0.30)

    def test_antithetic_reduces_stderr(self, base_cfg: SimulationConfig) -> None:
        """Antithetic variates should reduce standard error vs plain MC."""
        cfg_plain = dataclasses.replace(base_cfg, use_antithetic=False, seed=99)
        cfg_anti  = dataclasses.replace(base_cfg, use_antithetic=True,  seed=99)
        mc_plain = price_european_mc(cfg_plain)
        mc_anti  = price_european_mc(cfg_anti)
        # Antithetic should give lower or comparable stderr (not guaranteed in
        # finite samples but holds statistically for monotone payoffs).
        # We use a generous tolerance to avoid flakiness.
        assert mc_anti.call_stderr <= mc_plain.call_stderr * 1.10


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_negative_S0_raises(self) -> None:
        with pytest.raises(ValueError, match="S0"):
            SimulationConfig(S0=-10.0)

    def test_zero_T_raises(self) -> None:
        with pytest.raises(ValueError, match="T"):
            SimulationConfig(T=0.0)

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            SimulationConfig(sigma=-0.1)

    def test_zero_simulations_raises(self) -> None:
        with pytest.raises(ValueError, match="n_simulations"):
            SimulationConfig(n_simulations=0)
