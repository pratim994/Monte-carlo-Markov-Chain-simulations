"""
main.py — Entry point for the MCMC / Monte Carlo option pricing application.

Run via:
    python -m mcmc_options.main
    # or, if installed with poetry:
    mcmc-options
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; switch to "TkAgg" or "Qt5Agg" for GUI

import matplotlib.pyplot as plt

from .config import SimulationConfig, SurfaceConfig
from .pricing import (
    price_european_mc,
    price_european_bs_from_config,
    ComparisonResult,
)
from .visualization import (
    plot_comparison_surface,
    plot_mc_vs_bs_convergence,
    plot_sample_paths,
)


OUTPUT_DIR = "outputs"


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_pricing_demo(cfg: SimulationConfig) -> ComparisonResult:
    """Run the core pricing demo and print results."""
    print("\n" + "=" * 60)
    print("  European Option Pricing — Monte Carlo vs Black-Scholes")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  S₀={cfg.S0}  K={cfg.K}  r={cfg.r:.0%}  σ={cfg.sigma:.0%}  T={cfg.T}yr")
    print(f"  Paths={cfg.n_simulations:,}  Steps={cfg.n_steps}  "
          f"Antithetic={cfg.use_antithetic}  Seed={cfg.seed}\n")

    print("Simulating paths …")
    mc_result = price_european_mc(cfg)
    bs_result = price_european_bs_from_config(cfg)

    comparison = ComparisonResult(mc=mc_result, bs=bs_result)
    print(comparison)
    return comparison


def run_surface_plots(cfg: SimulationConfig, scfg: SurfaceConfig) -> None:
    """Generate and save 3D surface plots."""
    print("\n--- 3D Surface Plots ---")

    fig = plot_comparison_surface(
        cfg, scfg,
        save_path=os.path.join(OUTPUT_DIR, "surface_call_put.png"),
    )
    plt.close(fig)

    fig = plot_sample_paths(
        cfg,
        n_display=60,
        save_path=os.path.join(OUTPUT_DIR, "sample_paths.png"),
    )
    plt.close(fig)


def run_convergence_plot(cfg: SimulationConfig) -> None:
    """Plot MC convergence to Black-Scholes as n_simulations grows."""
    print("\n--- Convergence Analysis ---")
    fig = plot_mc_vs_bs_convergence(
        cfg,
        n_sim_values=[200, 1_000, 5_000, 10_000, 50_000, 100_000],
        save_path=os.path.join(OUTPUT_DIR, "convergence.png"),
    )
    plt.close(fig)


def main() -> None:
    """Orchestrate the full pricing and visualisation pipeline."""
    _ensure_output_dir()

    # ── Core pricing configuration ──────────────────────────────────────────
    cfg = SimulationConfig(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.20,
        T=1.0,
        n_simulations=100_000,
        n_steps=252,
        seed=42,
        use_antithetic=True,
    )

    # ── Surface grid configuration ───────────────────────────────────────────
    scfg = SurfaceConfig(
        S_range=(60.0, 140.0),
        T_range=(0.1, 2.0),
        n_S=30,
        n_T=30,
        n_simulations_surface=20_000,
    )

    run_pricing_demo(cfg)
    run_surface_plots(cfg, scfg)
    run_convergence_plot(cfg)

    print(f"\nAll outputs written to: {os.path.abspath(OUTPUT_DIR)}/")
    print("Done.\n")


if __name__ == "__main__":
    main()
