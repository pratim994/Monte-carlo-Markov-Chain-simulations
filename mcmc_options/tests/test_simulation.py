"""
tests/test_simulation.py — Unit tests for the GBM path simulator.
"""

import numpy as np
import pytest
from mcmc_options.config import SimulationConfig
from mcmc_options.simulation import simulate_gbm_paths, sample_terminal_prices_mcmc


class TestSimulateGBMPaths:
    """Tests for simulate_gbm_paths()."""

    def test_output_shape(self, base_cfg: SimulationConfig) -> None:
        paths = simulate_gbm_paths(base_cfg)
        assert paths.shape == (base_cfg.n_simulations, base_cfg.n_steps + 1)

    def test_initial_price_correct(self, base_cfg: SimulationConfig) -> None:
        paths = simulate_gbm_paths(base_cfg)
        np.testing.assert_allclose(paths[:, 0], base_cfg.S0)

    def test_all_prices_positive(self, base_cfg: SimulationConfig) -> None:
        """GBM stays strictly positive by construction."""
        paths = simulate_gbm_paths(base_cfg)
        assert np.all(paths > 0), "GBM paths must remain positive"

    def test_antithetic_doubles_exact(self) -> None:
        """With antithetic=True, path count should equal n_simulations exactly."""
        cfg = SimulationConfig(n_simulations=10_000, seed=0, use_antithetic=True)
        paths = simulate_gbm_paths(cfg)
        assert paths.shape[0] == cfg.n_simulations

    def test_reproducibility(self, base_cfg: SimulationConfig) -> None:
        """Same seed → identical paths."""
        p1 = simulate_gbm_paths(base_cfg)
        p2 = simulate_gbm_paths(base_cfg)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self) -> None:
        cfg_a = SimulationConfig(seed=0)
        cfg_b = SimulationConfig(seed=1)
        p_a = simulate_gbm_paths(cfg_a)
        p_b = simulate_gbm_paths(cfg_b)
        assert not np.allclose(p_a, p_b)

    def test_terminal_price_distribution(self, base_cfg: SimulationConfig) -> None:
        """
        Terminal log-prices should be approximately N(mu, sigma^2).
        Check that sample mean and std are within 3 standard errors of theory.
        """
        paths = simulate_gbm_paths(base_cfg)
        log_terminal = np.log(paths[:, -1])
        n = base_cfg.n_simulations

        mu_theory = np.log(base_cfg.S0) + (base_cfg.r - 0.5 * base_cfg.sigma**2) * base_cfg.T
        sigma_theory = base_cfg.sigma * np.sqrt(base_cfg.T)

        # 3-sigma tolerance (should pass ~99.7% of the time)
        tol_mean = 3 * sigma_theory / np.sqrt(n)
        np.testing.assert_allclose(log_terminal.mean(), mu_theory, atol=tol_mean)

        tol_std = 3 * sigma_theory / np.sqrt(2 * (n - 1))
        np.testing.assert_allclose(log_terminal.std(ddof=1), sigma_theory, atol=tol_std)

    def test_zero_vol_paths_deterministic(self) -> None:
        """sigma=0 → all paths identical and equal to S0 * e^{rT}."""
        cfg = SimulationConfig(S0=100.0, r=0.05, sigma=0.0, T=1.0, n_simulations=100, seed=0)
        paths = simulate_gbm_paths(cfg)
        expected_terminal = 100.0 * np.exp(0.05 * 1.0)
        np.testing.assert_allclose(paths[:, -1], expected_terminal, rtol=1e-10)


class TestMCMCSampler:
    """Tests for the Metropolis-Hastings terminal price sampler."""

    def test_output_length(self, base_cfg: SimulationConfig) -> None:
        samples = sample_terminal_prices_mcmc(base_cfg, burn_in=500)
        assert len(samples) == base_cfg.n_simulations

    def test_all_prices_positive(self, base_cfg: SimulationConfig) -> None:
        samples = sample_terminal_prices_mcmc(base_cfg, burn_in=500)
        assert np.all(samples > 0)

    def test_distribution_matches_gbm(self, base_cfg: SimulationConfig) -> None:
        """
        MCMC log-prices should match the target log-normal distribution
        to within 3 standard errors.
        """
        samples = sample_terminal_prices_mcmc(base_cfg, burn_in=2_000)
        log_samples = np.log(samples)
        mu_theory = np.log(base_cfg.S0) + (base_cfg.r - 0.5 * base_cfg.sigma**2) * base_cfg.T
        sigma_theory = base_cfg.sigma * np.sqrt(base_cfg.T)
        n = len(samples)

        tol_mean = 3 * sigma_theory / np.sqrt(n)
        np.testing.assert_allclose(log_samples.mean(), mu_theory, atol=tol_mean)
