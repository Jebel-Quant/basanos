<div align="center">

# <img src="https://raw.githubusercontent.com/Jebel-Quant/rhiza/main/.rhiza/assets/rhiza-logo.svg" alt="Rhiza Logo" width="30" style="vertical-align: middle;"> Basanos

**Correlation-aware portfolio optimization and analytics for Python.**

![Synced with Rhiza](https://img.shields.io/badge/synced%20with-rhiza-2FA4A9?color=2FA4A9)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python versions](https://img.shields.io/badge/Python-3.11%20•%203.12%20•%203.13%20•%203.14-blue?logo=python)](https://www.python.org/)
[![CI](https://github.com/Jebel-Quant/basanos/actions/workflows/rhiza_ci.yml/badge.svg?event=push)](https://github.com/Jebel-Quant/basanos/actions/workflows/rhiza_ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?logo=ruff)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Last Updated](https://img.shields.io/github/last-commit/jebel-quant/basanos/main?label=Last%20updated&color=blue)

</div>

---

Basanos computes **correlation-adjusted risk positions** from price data and expected-return signals. It estimates time-varying EWMA correlations, applies shrinkage towards the identity matrix, and solves a normalized linear system per timestamp to produce stable, scale-invariant positions — implementing a first hurdle for expected returns.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Development](#development)
- [License](#license)

## Features

- **Correlation-Aware Optimization** — EWMA correlation estimation with shrinkage towards identity
- **Dynamic Risk Management** — Volatility-normalized positions with configurable clipping and variance scaling
- **Portfolio Analytics** — Sharpe, VaR, CVaR, drawdown, skew, kurtosis, and more
- **Performance Attribution** — Tilt/timing decomposition to isolate allocation vs. selection effects
- **Interactive Visualizations** — Plotly dashboards for NAV, drawdown, lead/lag analysis, and correlation heatmaps
- **Polars-Native** — Built on Polars DataFrames for high-performance, memory-efficient computation

## Installation

```bash
pip install basanos
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add basanos
```

## Quick Start

### Portfolio Optimization

```python
import numpy as np
import polars as pl
from basanos.math import BasanosConfig, BasanosEngine

n_days = 100
dates = pl.date_range(
    pl.date(2023, 1, 1),
    pl.date(2023, 1, 1) + pl.duration(days=n_days - 1),
    eager=True,
)
rng = np.random.default_rng(42)

prices = pl.DataFrame({
    "date": dates,
    "AAPL":  100.0 + np.cumsum(rng.normal(0, 1.0, n_days)),
    "GOOGL": 150.0 + np.cumsum(rng.normal(0, 1.2, n_days)),
})

# Expected-return signals in [-1, 1] (e.g. from a forecasting model)
mu = pl.DataFrame({
    "date": dates,
    "AAPL":  np.tanh(rng.normal(0, 0.5, n_days)),
    "GOOGL": np.tanh(rng.normal(0, 0.5, n_days)),
})

cfg = BasanosConfig(
    vola=16,    # EWMA lookback for volatility (days)
    corr=32,    # EWMA lookback for correlation (days, must be >= vola)
    clip=3.5,   # Clipping threshold for vol-adjusted returns
    shrink=0.5, # Shrinkage intensity towards identity [0, 1]
    aum=1e6,    # Assets under management
)

engine    = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
positions = engine.cash_position  # pl.DataFrame of optimized cash positions
portfolio = engine.portfolio      # Portfolio object for analytics
```

### Portfolio Analytics

```python
import numpy as np
import polars as pl
from basanos.analytics import Portfolio

n_days = 60
dates = pl.date_range(
    pl.date(2023, 1, 1),
    pl.date(2023, 1, 1) + pl.duration(days=n_days - 1),
    eager=True,
)
rng = np.random.default_rng(42)

prices = pl.DataFrame({
    "date": dates,
    "AAPL":  100.0 * np.cumprod(1 + rng.normal(0.001, 0.020, n_days)),
    "GOOGL": 150.0 * np.cumprod(1 + rng.normal(0.001, 0.025, n_days)),
})

positions = pl.DataFrame({
    "date": dates,
    "AAPL":  np.full(n_days, 10_000.0),
    "GOOGL": np.full(n_days, 15_000.0),
})

portfolio = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

# Performance metrics
nav      = portfolio.nav_accumulated   # Cumulative additive NAV
returns  = portfolio.returns           # Daily returns scaled by AUM
drawdown = portfolio.drawdown          # Distance from high-water mark

# Statistics
stats  = portfolio.stats
sharpe = stats.sharpe()["returns"]
vol    = stats.volatility()["returns"]
```

### Visualizations

```python
fig = portfolio.plots.snapshot()                          # NAV + drawdown dashboard
fig = portfolio.plots.lead_lag_ir_plot(start=-10, end=20) # Sharpe across position lags
fig = portfolio.plots.lagged_performance_plot(lags=[0, 1, 2, 3, 4])
fig = portfolio.plots.correlation_heatmap()
# fig.show()
```

## How It Works

The optimizer implements a three-step pipeline per timestamp:

1. **Volatility adjustment** — Log returns are normalized by an EWMA volatility estimate and clipped at `cfg.clip` standard deviations to limit the influence of outliers.

2. **Correlation estimation** — An EWMA correlation matrix is computed from the vol-adjusted returns using a lookback of `cfg.corr` days. The matrix is shrunk toward the identity matrix with intensity `cfg.shrink`:

   ```
   C_shrunk = (1 - shrink) · C_ewma + shrink · I
   ```

   Shrinkage stabilizes the matrix when assets are few or the lookback is short.

3. **Position solving** — For each timestamp, the system `C_shrunk · x = mu` is solved for `x` (the risk position vector). The solution is normalized by the inverse-matrix norm of `mu`, making positions scale-invariant with respect to signal magnitude. Positions are further scaled by a running profit-variance estimate to adapt risk dynamically.

Cash positions are obtained by dividing risk positions by per-asset EWMA volatility.

## API Reference

### `basanos.math`

```python
from basanos.math import BasanosConfig, BasanosEngine
```

| Class | Description |
|-------|-------------|
| `BasanosConfig` | Immutable configuration (Pydantic model) |
| `BasanosEngine` | Core optimizer; produces positions and a `Portfolio` |

**`BasanosEngine` properties**

| Property | Returns | Description |
|----------|---------|-------------|
| `assets` | `list[str]` | Numeric asset column names |
| `ret_adj` | `pl.DataFrame` | Vol-adjusted, clipped log returns |
| `vola` | `pl.DataFrame` | Per-asset EWMA volatility |
| `cor` | `dict[date, np.ndarray]` | EWMA correlation matrices keyed by date |
| `cash_position` | `pl.DataFrame` | Optimized cash positions |
| `portfolio` | `Portfolio` | Ready-to-use portfolio for analytics |

---

### `basanos.analytics`

```python
from basanos.analytics import Portfolio
```

| Class | Description |
|-------|-------------|
| `Portfolio` | Central data model for P&L, NAV, and attribution |
| `Stats` | Statistical risk/return metrics |
| `Plots` | Plotly-based interactive visualizations |

**`Portfolio` properties**

| Property | Description |
|----------|-------------|
| `profits` | Per-asset daily P&L |
| `profit` | Aggregate daily portfolio profit |
| `nav_accumulated` | Cumulative additive NAV |
| `nav_compounded` | Compounded NAV |
| `returns` | Daily returns scaled by AUM |
| `monthly` | Monthly compounded returns |
| `highwater` | Running NAV maximum |
| `drawdown` | Drawdown from high-water mark |
| `tilt` | Static allocation (average position) |
| `timing` | Dynamic timing (deviation from average) |
| `stats` | `Stats` instance |
| `plots` | `Plots` instance |

**`Stats` methods**

| Method | Description |
|--------|-------------|
| `sharpe(periods)` | Annualized Sharpe ratio |
| `volatility(periods, annualize)` | Standard deviation of returns |
| `skew()` | Skewness |
| `kurtosis()` | Excess kurtosis |
| `value_at_risk(alpha, sigma)` | Parametric VaR |
| `conditional_value_at_risk(alpha, sigma)` | Expected shortfall (CVaR) |
| `avg_return()` | Mean return (zeros excluded) |
| `avg_win()` | Mean positive return |
| `avg_loss()` | Mean negative return |
| `best()` | Maximum single-period return |
| `worst()` | Minimum single-period return |

## Configuration Reference

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `vola` | `int` | `> 0` | EWMA lookback for volatility (days) |
| `corr` | `int` | `>= vola` | EWMA lookback for correlation (days) |
| `clip` | `float` | `> 0` | Clipping threshold for vol-adjusted returns |
| `shrink` | `float` | `[0, 1]` | Shrinkage intensity — `0` = no shrinkage, `1` = identity |
| `aum` | `float` | `> 0` | Assets under management for position scaling |

```python
from basanos.math import BasanosConfig

# Conservative — longer lookbacks, stronger shrinkage
conservative = BasanosConfig(vola=32, corr=64, clip=3.0, shrink=0.7, aum=1e6)

# Responsive — shorter lookbacks, lighter shrinkage
responsive   = BasanosConfig(vola=8,  corr=16, clip=4.0, shrink=0.3, aum=1e6)
```

## Development

```bash
git clone https://github.com/Jebel-Quant/basanos.git
cd basanos
uv sync
```

| Command | Action |
|---------|--------|
| `make test` | Run the test suite |
| `make fmt` | Format and lint with ruff |
| `make typecheck` | Static type checking |
| `make deptry` | Audit declared dependencies |

Before submitting a PR, ensure all checks pass:

```bash
make fmt && make test && make typecheck
```

## License

See [LICENSE](LICENSE) for details.
