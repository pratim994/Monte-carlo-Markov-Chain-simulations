"""
visualization.py — 3D surface plots for option prices.

Generates publication-quality matplotlib figures showing option price as a
function of spot price (S) and time to maturity (T).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from matplotlib.ticker import LinearLocator

from .config import SimulationConfig, SurfaceConfig
from .pricing import compute_price_surface


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_grids(scfg: SurfaceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (S_arr, T_arr, S_mesh, T_mesh) for the surface."""
    S_arr = np.linspace(scfg.S_range[0], scfg.S_range[1], scfg.n_S)
    T_arr = np.linspace(scfg.T_range[0], scfg.T_range[1], scfg.n_T)
    S_mesh, T_mesh = np.meshgrid(S_arr, T_arr)
    return S_arr, T_arr, S_mesh, T_mesh


def _style_3d_axis(
    ax: plt.Axes,
    title: str,
    z_label: str,
) -> None:
    """Apply consistent axis labels, tick formatting and title."""
    ax.set_xlabel("Spot Price  S", labelpad=10, fontsize=11)
    ax.set_ylabel("Time to Maturity  T (yrs)", labelpad=10, fontsize=11)
    ax.set_zlabel(z_label, labelpad=10, fontsize=11)
    ax.set_title(title, fontsize=13, pad=20)
    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter("{x:.2f}")
    ax.view_init(elev=28, azim=-55)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_option_surface(
    cfg: SimulationConfig,
    scfg: SurfaceConfig,
    option_type: str = "call",
    method: str = "bs",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Render a 3-D surface of option price vs (S, T).

    Parameters
    ----------
    cfg : SimulationConfig
        Provides K, r, sigma for pricing.
    scfg : SurfaceConfig
        Defines the S/T grid dimensions.
    option_type : {"call", "put"}
    method : {"bs", "mc"}
        "bs" — fast analytical (recommended for surface).
        "mc" — Monte Carlo at each grid point (slow but instructive).
    save_path : str, optional
        File path to save the figure (e.g., "surface.png").

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    S_arr, T_arr, S_mesh, T_mesh = _build_grids(scfg)

    print(f"Computing {option_type} price surface ({method.upper()}) …")
    Z = compute_price_surface(
        S_grid=S_arr,
        T_grid=T_arr,
        K=cfg.K,
        r=cfg.r,
        sigma=cfg.sigma,
        option_type=option_type,
        method=method,
        n_simulations=scfg.n_simulations_surface,
        seed=cfg.seed or 42,
    )

    fig = plt.figure(figsize=(12, 8))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        S_mesh, T_mesh, Z,
        cmap=cm.plasma,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1)
    cbar.set_label("Option Price ($)", fontsize=10)

    label_map = {"call": "Call Option Price ($)", "put": "Put Option Price ($)"}
    method_str = "Black-Scholes" if method == "bs" else "Monte Carlo"
    title = (
        f"European {option_type.capitalize()} Option Price Surface\n"
        f"K={cfg.K}, r={cfg.r:.0%}, σ={cfg.sigma:.0%}  [{method_str}]"
    )
    _style_3d_axis(ax, title, label_map[option_type])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    return fig


def plot_comparison_surface(
    cfg: SimulationConfig,
    scfg: SurfaceConfig,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot call and put surfaces side-by-side using Black-Scholes.

    Parameters
    ----------
    cfg : SimulationConfig
    scfg : SurfaceConfig
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    S_arr, T_arr, S_mesh, T_mesh = _build_grids(scfg)

    print("Computing call and put surfaces (Black-Scholes) …")
    Z_call = compute_price_surface(S_arr, T_arr, cfg.K, cfg.r, cfg.sigma, "call", "bs")
    Z_put  = compute_price_surface(S_arr, T_arr, cfg.K, cfg.r, cfg.sigma, "put",  "bs")

    fig = plt.figure(figsize=(18, 8))

    for idx, (Z, option_type, cmap_name) in enumerate(
        [(Z_call, "Call", "plasma"), (Z_put, "Put", "viridis")]
    ):
        ax: Axes3D = fig.add_subplot(1, 2, idx + 1, projection="3d")
        surf = ax.plot_surface(
            S_mesh, T_mesh, Z,
            cmap=cmap_name,
            linewidth=0,
            antialiased=True,
            alpha=0.92,
        )
        cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, pad=0.1)
        cbar.set_label("Price ($)", fontsize=9)

        title = (
            f"European {option_type} — Black-Scholes\n"
            f"K={cfg.K}, r={cfg.r:.0%}, σ={cfg.sigma:.0%}"
        )
        _style_3d_axis(ax, title, f"{option_type} Price ($)")

    fig.suptitle("Option Price Surfaces", fontsize=15, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    return fig


def plot_mc_vs_bs_convergence(
    cfg: SimulationConfig,
    n_sim_values: list[int] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Show how Monte Carlo call price converges to the Black-Scholes price
    as the number of simulations increases.

    Parameters
    ----------
    cfg : SimulationConfig
        Base configuration (seed, K, r, sigma, T, S0, …).
    n_sim_values : list[int], optional
        Path counts to test. Defaults to logarithmic sequence.
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from .pricing import price_european_mc, price_european_bs_from_config

    if n_sim_values is None:
        n_sim_values = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

    bs = price_european_bs_from_config(cfg)
    mc_calls, mc_puts, call_se, put_se = [], [], [], []

    for n in n_sim_values:
        import dataclasses
        test_cfg = dataclasses.replace(cfg, n_simulations=n)
        mc = price_european_mc(test_cfg)
        mc_calls.append(mc.call_price)
        mc_puts.append(mc.put_price)
        call_se.append(mc.call_stderr)
        put_se.append(mc.put_stderr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mc_prices, se, option_type, bs_price in zip(
        axes,
        [mc_calls, mc_puts],
        [call_se, put_se],
        ["Call", "Put"],
        [bs.call_price, bs.put_price],
    ):
        mc_arr = np.array(mc_prices)
        se_arr = np.array(se)

        ax.semilogx(n_sim_values, mc_arr, "o-", color="#e74c3c", label="Monte Carlo", lw=2)
        ax.fill_between(
            n_sim_values,
            mc_arr - 1.96 * se_arr,
            mc_arr + 1.96 * se_arr,
            alpha=0.2, color="#e74c3c", label="95% CI"
        )
        ax.axhline(bs_price, color="#2c3e50", ls="--", lw=1.5,
                   label=f"Black-Scholes = {bs_price:.4f}")
        ax.set_xlabel("Number of Simulations", fontsize=11)
        ax.set_ylabel("Option Price ($)", fontsize=11)
        ax.set_title(f"European {option_type} — MC Convergence", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    return fig


def plot_sample_paths(
    cfg: SimulationConfig,
    n_display: int = 50,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a subset of simulated GBM asset paths for visual inspection.

    Parameters
    ----------
    cfg : SimulationConfig
    n_display : int
        Number of paths to draw (for clarity; all paths used in pricing).
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from .simulation import simulate_gbm_paths

    paths = simulate_gbm_paths(cfg)
    t = np.linspace(0, cfg.T, cfg.n_steps + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(min(n_display, paths.shape[0])):
        ax.plot(t, paths[i], lw=0.6, alpha=0.5)

    ax.axhline(cfg.K, color="black", ls="--", lw=1.5, label=f"Strike K={cfg.K}")
    ax.set_xlabel("Time (years)", fontsize=11)
    ax.set_ylabel("Asset Price ($)", fontsize=11)
    ax.set_title(
        f"Simulated GBM Paths  (S₀={cfg.S0}, σ={cfg.sigma:.0%}, r={cfg.r:.0%})",
        fontsize=12,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    return fig
