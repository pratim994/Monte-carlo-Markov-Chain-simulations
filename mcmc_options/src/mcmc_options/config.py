"""
config.py — Parameter configuration for the Monte Carlo option pricing engine.

All simulation parameters are centralised here. Use SimulationConfig as the
single source of truth when instantiating the pricing and simulation modules.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationConfig:
    """
    Holds all configurable parameters for the Monte Carlo option pricer.

    Parameters
    ----------
    S0 : float
        Initial (spot) price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Continuously compounded risk-free interest rate (annualised).
    sigma : float
        Annualised volatility of the underlying asset (lognormal).
    T : float
        Time to maturity in years.
    n_simulations : int
        Number of independent Monte Carlo paths to generate.
    n_steps : int
        Number of discrete time steps per path (time discretisation).
    seed : Optional[int]
        Random seed for reproducibility. None = non-deterministic.
    use_antithetic : bool
        If True, apply antithetic variates variance reduction.

    Notes
    -----
    The model assumes:
    - Constant volatility and risk-free rate (standard Black-Scholes world).
    - No dividends on the underlying.
    - European-style exercise (payoff evaluated only at maturity T).
    - Continuous trading with no transaction costs.
    """

    S0: float = 100.0
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    T: float = 1.0
    n_simulations: int = 100_000
    n_steps: int = 252          # trading days in a year
    seed: Optional[int] = 42
    use_antithetic: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Raise ValueError for obviously invalid parameter combinations."""
        if self.S0 <= 0:
            raise ValueError(f"S0 must be positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {self.n_simulations}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")


@dataclass
class SurfaceConfig:
    """
    Configuration for the 3D option-price surface plot.

    Parameters
    ----------
    S_range : tuple[float, float]
        (min, max) range of spot prices for the surface.
    T_range : tuple[float, float]
        (min, max) range of maturities (years) for the surface.
    n_S : int
        Number of grid points along the S-axis.
    n_T : int
        Number of grid points along the T-axis.
    n_simulations_surface : int
        Simulations per grid point (lower than full precision for speed).
    """

    S_range: tuple = (60.0, 140.0)
    T_range: tuple = (0.1, 2.0)
    n_S: int = 25
    n_T: int = 25
    n_simulations_surface: int = 20_000
