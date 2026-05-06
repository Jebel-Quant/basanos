<div align="center" markdown="1">

# <img src="https://raw.githubusercontent.com/Jebel-Quant/rhiza/main/.rhiza/assets/rhiza-logo.svg" alt="Rhiza Logo" width="30" style="vertical-align: middle;"> Basanos

**Correlation-aware portfolio optimization and analytics for Python.**

![Synced with Rhiza](https://img.shields.io/badge/synced%20with-rhiza-2FA4A9?color=2FA4A9)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python versions](https://img.shields.io/badge/Python-3.11%20•%203.12%20•%203.13%20•%203.14-blue?logo=python)](https://www.python.org/)
[![CI](https://github.com/Jebel-Quant/basanos/actions/workflows/rhiza_ci.yml/badge.svg?event=push)](https://github.com/Jebel-Quant/basanos/actions/workflows/rhiza_ci.yml)
[![Coverage](https://jebel-quant.github.io/basanos/coverage-badge.svg)](https://jebel-quant.github.io/basanos/reports/html-coverage/)
[![PyPI version](https://img.shields.io/pypi/v/basanos.svg)](https://pypi.org/project/basanos/)
[![Downloads](https://static.pepy.tech/personalized-badge/basanos?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/basanos)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg?logo=ruff)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Last Updated](https://img.shields.io/github/last-commit/jebel-quant/basanos/main?label=Last%20updated&color=blue)
[![Paper](https://img.shields.io/badge/paper-basanos.pdf-red?logo=adobeacrobatreader)](https://github.com/jebel-quant/basanos/blob/paper/basanos.pdf)

</div>

---

Basanos computes **correlation-adjusted risk positions** from price data and expected-return signals. It estimates time-varying EWMA correlations, applies shrinkage towards the identity matrix, and solves a normalized linear system per timestamp to produce stable, scale-invariant positions — implementing a first hurdle for expected returns.

## Table of Contents

- [Idea](#idea)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Notebooks](#notebooks)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

## Idea

Most systematic strategies produce a raw signal vector μ — one number per asset indicating how bullish or bearish the model is. Sizing each position in direct proportion to its signal ignores the fact that correlated assets will receive large, overlapping bets in the same direction, concentrating risk rather than diversifying it.

Basanos treats position sizing as a **linear system**:

```
C · x = μ
```

where C is the (shrunk, time-varying) correlation matrix and μ is the signal. Solving for x inverts the correlation structure — assets that share a lot of co-movement with the rest of the portfolio receive smaller positions, while idiosyncratic assets can carry more. The result is a set of *risk positions* that express the full information in the signal while respecting the portfolio's correlation geometry.

Three design choices keep the output stable and usable in practice:

1. **EWMA estimates** — both volatility and correlations are computed as exponentially weighted moving averages, so the optimizer adapts to changing regimes without requiring a fixed lookback window.
2. **Regularised correlation matrix** — the correlation matrix must be well-conditioned to invert reliably. Basanos offers two complementary strategies (see [Covariance Modes](#covariance-modes)):
   - *EWMA with shrinkage* (default): the estimated EWMA correlation is blended toward the identity matrix, pulling noisy eigenvalues toward a common value. The `cfg.shrink` parameter (λ) controls how much of the raw EWMA matrix is retained.
   - *Sliding-window factor model*: a rolling block of recent vol-adjusted returns is decomposed via truncated SVD into `k` latent factors, giving a low-rank-plus-diagonal covariance structure. **No shrinkage is needed**: the number of factors `k` is the sole regularisation knob — reducing `k` compresses more of the correlation structure into fewer factors, analogous to strong shrinkage.
3. **Scale invariance** — positions are normalised by the inverse-matrix norm of μ, so doubling the signal magnitude does not double the position. Sizing is driven instead by a running estimate of realised profit variance, which scales risk up in good regimes and down in bad ones.

The output of the solve is a *risk position* (units of volatility). Dividing by per-asset EWMA volatility converts it into a *cash position* — how many dollars to hold in each asset.

**Why not just run a full optimizer?** The primary use case for basanos is **signal assessment**, not production execution. A fully constrained Markowitz optimizer — with turnover limits, sector caps, leverage constraints, and factor neutrality targets — will bend positions away from what the signal actually implies. The resulting P&L reflects the interaction of the signal with all those constraints, making it hard to tell whether the underlying signal has edge. Basanos deliberately avoids hard constraints to give the signal room to express itself cleanly. By orthogonalizing μ to known risk factors before passing it in, you can further isolate the pure alpha component and measure how much of the return comes from the signal itself versus incidental factor exposures. This makes it a natural *first hurdle*: a signal that cannot generate a reasonable Sharpe through this minimal framework is unlikely to survive the additional friction of a production optimizer.

## Features

- **Correlation-Aware Optimization** — Two covariance modes: EWMA with linear shrinkage (default) or sliding-window factor model
- **Dynamic Risk Management** — Volatility-normalized positions with configurable clipping and variance scaling
- **Signal Evaluation** — IC and Rank IC time series, ICIR summary statistics, and a naïve equal-weight Sharpe benchmark to isolate signal skill
- **Diagnostic Properties** — Condition number, effective rank, solver residual, signal utilisation, risk position, and gross leverage for every timestamp
- **Factor Risk Model** — `FactorModel` decomposition Σ = B·F·Bᵀ + D fitted via truncated SVD; used as the covariance estimator in `sliding_window` mode and available for standalone low-rank covariance inspection
- **Portfolio Analytics** — Sharpe, VaR, CVaR, drawdown, skew, kurtosis, and more
- **Performance Attribution** — Tilt/timing decomposition to isolate allocation vs. selection effects
- **Interactive Visualizations** — Plotly dashboards for NAV, drawdown, lead/lag analysis, and correlation heatmaps
- **Trading Cost Analysis** — Estimate the impact of one-way trading costs on Sharpe ratio across a configurable basis-point range
- **Config Reports** — Self-contained HTML report for `BasanosConfig` with parameter table, shrinkage guidance, and interactive lambda-sweep chart
- **HTML Reports** — One-call self-contained dark-themed HTML report with statistics tables and interactive Plotly charts
- **Polars-Native** — Built on Polars DataFrames for high-performance, memory-efficient computation

## Installation

```bash
pip install basanos
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add basanos
```

Or with conda (via [conda-forge](https://conda-forge.org/)):

```bash
conda install -c conda-forge basanos
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

# Mode 1 — EWMA with shrinkage (default)
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

### Factor Model Mode

Use `SlidingWindowConfig` to switch to the sliding-window factor model. See the [Factor Models guide](https://jebel-quant.github.io/basanos/factor-models/) for a full explanation of the approach and how to choose `window` and `n_factors`.

```python
import numpy as np
import polars as pl
from basanos.math import BasanosConfig, BasanosEngine, SlidingWindowConfig

n_days = 200
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
    "MSFT":  200.0 + np.cumsum(rng.normal(0, 1.5, n_days)),
})
mu = pl.DataFrame({
    "date": dates,
    "AAPL":  np.tanh(rng.normal(0, 0.5, n_days)),
    "GOOGL": np.tanh(rng.normal(0, 0.5, n_days)),
    "MSFT":  np.tanh(rng.normal(0, 0.5, n_days)),
})

# Mode 2 — Sliding-window factor model (no shrinkage required)
cfg = BasanosConfig(
    vola=16,
    corr=32,
    clip=3.5,
    shrink=0.5,  # only used if covariance_mode is ewma_shrink; ignored here
    aum=1e6,
    covariance_config=SlidingWindowConfig(
        window=60,    # rolling window length W (rows); rule of thumb: W >= 2 * n_assets
        n_factors=2,  # number of latent factors k; fewer = stronger regularisation
    ),
)

engine    = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
positions = engine.cash_position
```

### Portfolio Analytics

```python
import numpy as np
import polars as pl
from jquantstats import Portfolio

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

### Generating Reports

`portfolio.report` returns a `Report` facade that produces a self-contained, dark-themed HTML document with a performance-statistics table and multiple interactive Plotly charts.

```python
report = portfolio.report

# Render to a string (e.g. to serve via an API or display in a notebook)
html_str = report.to_html()

# Or save directly to disk — a .html extension is added automatically
saved_path = report.save("output/report")
# → saves to output/report.html

# Customize the page title
report.save("output/my_report.html", title="My Strategy Report")
```

The generated report contains the following sections:

| Section | Content |
|---------|---------|
| **Performance** | Cumulative NAV + drawdown snapshot |
| **Risk Analysis** | Rolling Sharpe ratio and rolling volatility charts |
| **Annual Breakdown** | Sharpe ratio by calendar year |
| **Monthly Returns** | Monthly returns heatmap |
| **Performance Statistics** | Full statistics table (returns, drawdown, risk-adjusted, distribution) |
| **Correlation Analysis** | Asset correlation heatmap |
| **Lead / Lag** | Lead/lag information ratio chart |
| **Turnover Summary** | Portfolio turnover metrics |
| **Trading Cost Impact** | Sharpe ratio vs. one-way trading cost (basis points) |

### Trading Cost Analysis

`Portfolio` exposes two methods for understanding how trading costs erode strategy edge:

```python
# Net-of-cost daily returns (5 bps one-way cost)
adj_returns = portfolio.cost_adjusted_returns(cost_bps=5)

# Sharpe ratio sweep from 0 to 20 bps
impact = portfolio.trading_cost_impact(max_bps=20)
# Returns a pl.DataFrame with columns: cost_bps (Int64), sharpe (Float64)

# Interactive Plotly chart — Sharpe vs cost
fig = portfolio.plots.trading_cost_impact_plot(max_bps=20)
# fig.show()
```

### Config Reports

`BasanosConfig` and `BasanosEngine` each expose a report property that produces a self-contained, dark-themed HTML document with:

- a **parameter table** (all fields, values, constraints, and descriptions),
- a **shrinkage guidance table** (n/T regime heuristics), and
- a **theory section** on Ledoit-Wolf shrinkage.

When accessed from `BasanosEngine`, the report additionally includes an **interactive lambda-sweep chart** — the annualised Sharpe ratio as the shrinkage parameter λ is swept across [0, 1].

```python
import numpy as np
import polars as pl
from basanos.math import BasanosConfig, BasanosEngine

cfg = BasanosConfig(vola=16, corr=32, clip=3.5, shrink=0.5, aum=1e6)

# Config-only report (no lambda sweep)
html_str = cfg.report.to_html()
cfg.report.save("output/config_report")  # → output/config_report.html

# Engine report (includes lambda-sweep chart)
n = 100
_dates = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 1) + pl.duration(days=n - 1), eager=True)
_rng = np.random.default_rng(0)
_prices = pl.DataFrame({"date": _dates, "AAPL": 100.0 + np.cumsum(_rng.normal(0, 1.0, n)), "GOOGL": 150.0 + np.cumsum(_rng.normal(0, 1.2, n))})
_mu = pl.DataFrame({"date": _dates, "AAPL": np.tanh(_rng.normal(0, 0.5, n)), "GOOGL": np.tanh(_rng.normal(0, 0.5, n))})
cfg_engine = BasanosEngine(prices=_prices, mu=_mu, cfg=cfg)
cfg_engine.config_report.save("output/config_with_sweep")
```

## Notebooks

Five interactive [Marimo](https://marimo.io/) notebooks live under
`book/marimo/notebooks/`. They are self-contained — each embeds its own
dependency list ([PEP 723](https://peps.python.org/pep-0723/)), so `uv run`
installs everything automatically.

| Notebook | Description | Key concepts |
|---|---|---|
| **[`end_to_end.py`](book/marimo/notebooks/end_to_end.py)** | **Complete worked example from raw prices to HTML report using realistic synthetic equity data** | Data preparation, config selection with shrinkage guidance, engine instantiation, HTML report generation, trading cost analysis |
| [`demo.py`](book/marimo/notebooks/demo.py) | End-to-end interactive demo of the Basanos optimizer | Signal generation, correlation-aware position sizing, portfolio analytics, reactive UI |
| [`ewm_benchmark.py`](book/marimo/notebooks/ewm_benchmark.py) | Validates and benchmarks the NumPy/SciPy EWM correlation implementation against the legacy pandas version | EWM, `scipy.signal.lfilter`, NaN handling, performance comparison |
| [`shrinkage_guide.py`](book/marimo/notebooks/shrinkage_guide.py) | Theoretical and empirical guide to tuning the shrinkage parameter λ | Marchenko-Pastur law, linear shrinkage `C(λ) = λ·C_EWMA + (1−λ)·I`, Sharpe vs. λ sweep, turnover analysis |
| [`diagnostics.py`](book/marimo/notebooks/diagnostics.py) | Guided tour of all five engine diagnostic properties on a dataset with engineered edge cases | `position_status`, `condition_number`, `effective_rank`, `solver_residual`, `signal_utilisation` |

### Running the notebooks

```bash
# Launch all notebooks in the Marimo editor (opens http://localhost:2718)
make marimo

# Open a single notebook for interactive editing
marimo edit book/marimo/notebooks/end_to_end.py

# Run a single notebook read-only / presentation mode
marimo run book/marimo/notebooks/end_to_end.py

# Self-contained via uv — no prior install needed
uv run book/marimo/notebooks/end_to_end.py
uv run book/marimo/notebooks/demo.py
uv run book/marimo/notebooks/ewm_benchmark.py
uv run book/marimo/notebooks/shrinkage_guide.py
uv run book/marimo/notebooks/diagnostics.py
```

**Prerequisites**: Python ≥ 3.11 and `uv`. Each notebook's dependencies
(marimo, basanos, numpy, polars, plotly, …) are resolved automatically by `uv`.
If you are editing source code alongside the notebook, run `make install` first
so the local package is available.

## Documentation

Full documentation is available at **[jebel-quant.github.io/basanos](https://jebel-quant.github.io/basanos/)**.

| Topic | Description |
|-------|-------------|
| [Covariance Modes & How It Works](https://jebel-quant.github.io/basanos/concepts/) | Pipeline overview, EWMA shrinkage deep-dive, sliding-window factor model, choosing between modes |
| [Factor Models](https://jebel-quant.github.io/basanos/factor-models/) | `FactorModel` background, Σ = B·F·Bᵀ + D decomposition, truncated SVD |
| [Streaming Guide](https://jebel-quant.github.io/basanos/streaming/) | `BasanosStream` — incremental one-row-at-a-time interface for live systems |
| [Performance](https://jebel-quant.github.io/basanos/performance/) | Computational complexity, memory usage, practical limits, benchmark data |
| [API Reference](https://jebel-quant.github.io/basanos/api/) | `BasanosEngine`, `BasanosConfig`, `FactorModel`, `BasanosStream`, `Portfolio` |


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

For architectural decisions and their rationale, see [docs/adr/](docs/adr/).

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, commit conventions, and PR expectations.

## License

See [LICENSE](LICENSE) for details.
