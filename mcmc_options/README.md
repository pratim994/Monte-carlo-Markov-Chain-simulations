# MCMC Options — 3D Monte Carlo Option Pricing

A production-quality Python implementation of Monte Carlo simulation for pricing
European options under the Black-Scholes framework, with full 3D visualisation.

---

## Overview

### Stochastic Process

Asset prices are modelled as **Geometric Brownian Motion** under the risk-neutral measure:

```
dS = r·S·dt + σ·S·dW_t
```

Using the exact log-normal transition (no Euler discretisation error):

```
S_{t+Δt} = S_t · exp[(r - σ²/2)Δt + σ√Δt · Z],   Z ~ N(0,1)
```

### Pricing

European option prices are the discounted expected payoff under Q:

```
Call = e^{-rT} · E[max(S_T - K, 0)]
Put  = e^{-rT} · E[max(K - S_T, 0)]
```

Monte Carlo approximates the expectation by sample mean over N paths.

### Variance Reduction: Antithetic Variates

For every random draw Z we also simulate using −Z. Because the payoff is
monotone in Z, the two estimators are negatively correlated, roughly halving
the variance at no extra simulation cost.

### MCMC Component

A Metropolis-Hastings sampler is included that targets the risk-neutral
log-normal terminal-price distribution. For production pricing, direct GBM
sampling is preferred (exact, fully vectorised, ~50× faster). The MCMC
component is provided for pedagogical completeness.

---

## Project Structure

```
mcmc_options/
├── pyproject.toml
├── README.md
├── src/
│   └── mcmc_options/
│       ├── __init__.py
│       ├── config.py          # Parameter dataclasses
│       ├── simulation.py      # GBM path engine + MCMC sampler
│       ├── pricing.py         # MC & Black-Scholes pricers
│       ├── visualization.py   # 3D surface & convergence plots
│       └── main.py            # CLI entry point
└── tests/
    ├── conftest.py
    ├── test_simulation.py
    └── test_pricing.py
```

---

## Installation

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
cd mcmc_options
poetry install
```

Or with pip directly:

```bash
pip install numpy matplotlib scipy
```

---

## Running

```bash
# With Poetry
poetry run mcmc-options

# As a module
cd mcmc_options
python -m mcmc_options.main

# Output files are written to: outputs/
#   surface_call_put.png   — 3D call & put price surfaces
#   sample_paths.png       — simulated GBM asset paths
#   convergence.png        — MC convergence to Black-Scholes
```

---

## Running Tests

```bash
# With Poetry
poetry run pytest

# With coverage
poetry run pytest --cov=mcmc_options --cov-report=term-missing

# Run a specific test file
poetry run pytest tests/test_pricing.py -v
```

---

## Example Output

```
============================================================
  European Option Pricing — Monte Carlo vs Black-Scholes
============================================================

Parameters:
  S₀=100.0  K=100.0  r=5%  σ=20%  T=1.0yr
  Paths=100,000  Steps=252  Antithetic=True  Seed=42

==================================================
Monte Carlo Pricing Result
  Call :    10.4423  (±0.0297)
  Put  :     5.5652  (±0.0210)
  Paths: 100,000  |  antithetic=True

Black-Scholes Analytical Result
  Call :    10.4506
  Put  :     5.5735
  d1   :     0.3500   d2: 0.1500

Absolute errors
  Call |MC - BS| = 0.0083
  Put  |MC - BS| = 0.0083
==================================================
```

---

## Key Parameters

| Parameter      | Default  | Description                          |
|----------------|----------|--------------------------------------|
| `S0`           | 100.0    | Initial spot price                   |
| `K`            | 100.0    | Strike price                         |
| `r`            | 0.05     | Risk-free rate (annualised)          |
| `sigma`        | 0.20     | Volatility (annualised)              |
| `T`            | 1.0      | Time to maturity (years)             |
| `n_simulations`| 100,000  | Number of Monte Carlo paths          |
| `n_steps`      | 252      | Time steps per path                  |
| `use_antithetic`| True    | Antithetic variates variance reduction|
| `seed`         | 42       | Random seed for reproducibility      |

---

## Assumptions

- Constant volatility and risk-free rate (Black-Scholes world)
- No dividends on the underlying asset
- European-style exercise (payoff at maturity only)
- Continuous trading, no transaction costs
- Log-normal asset price distribution

---

## Performance Notes

- All path generation is **fully vectorised** with NumPy (no Python loops over paths)
- The 3D surface uses the analytical Black-Scholes formula for speed
- For Monte Carlo surfaces, reduce `n_simulations_surface` (default 20,000) to trade accuracy for speed
- Antithetic variates roughly double effective sample efficiency
