<div align="center">

# <img src="https://raw.githubusercontent.com/Jebel-Quant/rhiza/main/.rhiza/assets/rhiza-logo.svg" alt="Rhiza Logo" width="30" style="vertical-align: middle;"> Basanos

**Correlation-aware portfolio optimization and analytics for Python.**

![Synced with Rhiza](https://img.shields.io/badge/synced%20with-rhiza-2FA4A9?color=2FA4A9)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python versions](https://img.shields.io/badge/Python-3.11%20•%203.12%20•%203.13%20•%203.14-blue?logo=python)](https://www.python.org/)
[![CI](https://github.com/Jebel-Quant/basanos/actions/workflows/rhiza_ci.yml/badge.svg?event=push)](https://github.com/Jebel-Quant/basanos/actions/workflows/rhiza_ci.yml)
[![Coverage](https://raw.githubusercontent.com/Jebel-Quant/basanos/refs/heads/gh-pages/coverage-badge.svg)](https://jebel-quant.github.io/basanos/tests/html-coverage/index.html)
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
- [How It Works](#how-it-works)
- [Covariance Modes](#covariance-modes)
  - [Mode 1 — EWMA with Shrinkage](#mode-1--ewma-with-shrinkage)
  - [Mode 2 — Sliding-Window Factor Model](#mode-2--sliding-window-factor-model)
  - [Choosing Between Modes](#choosing-between-modes)
- [Performance Characteristics](#performance-characteristics)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
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

Use `SlidingWindowConfig` to switch to the sliding-window factor model. No explicit shrinkage is needed — the number of factors `k` acts as the sole regularisation knob (fewer factors = stronger compression of the correlation structure).

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

Four interactive [Marimo](https://marimo.io/) notebooks live under
`book/marimo/notebooks/`. They are self-contained — each embeds its own
dependency list ([PEP 723](https://peps.python.org/pep-0723/)), so `uv run`
installs everything automatically.

| Notebook | Description | Key concepts |
|---|---|---|
| [`demo.py`](book/marimo/notebooks/demo.py) | End-to-end interactive demo of the Basanos optimizer | Signal generation, correlation-aware position sizing, portfolio analytics, reactive UI |
| [`ewm_benchmark.py`](book/marimo/notebooks/ewm_benchmark.py) | Validates and benchmarks the NumPy/SciPy EWM correlation implementation against the legacy pandas version | EWM, `scipy.signal.lfilter`, NaN handling, performance comparison |
| [`shrinkage_guide.py`](book/marimo/notebooks/shrinkage_guide.py) | Theoretical and empirical guide to tuning the shrinkage parameter λ | Marchenko-Pastur law, linear shrinkage `C(λ) = λ·C_EWMA + (1−λ)·I`, Sharpe vs. λ sweep, turnover analysis |
| [`diagnostics.py`](book/marimo/notebooks/diagnostics.py) | Guided tour of all five engine diagnostic properties on a dataset with engineered edge cases | `position_status`, `condition_number`, `effective_rank`, `solver_residual`, `signal_utilisation` |

### Running the notebooks

```bash
# Launch all notebooks in the Marimo editor (opens http://localhost:2718)
make marimo

# Open a single notebook for interactive editing
marimo edit book/marimo/notebooks/demo.py

# Run a single notebook read-only / presentation mode
marimo run book/marimo/notebooks/demo.py

# Self-contained via uv — no prior install needed
uv run book/marimo/notebooks/demo.py
uv run book/marimo/notebooks/ewm_benchmark.py
uv run book/marimo/notebooks/shrinkage_guide.py
uv run book/marimo/notebooks/diagnostics.py
```

**Prerequisites**: Python ≥ 3.11 and `uv`. Each notebook's dependencies
(marimo, basanos, numpy, polars, plotly, …) are resolved automatically by `uv`.
If you are editing source code alongside the notebook, run `make install` first
so the local package is available.

## How It Works

The optimizer implements a three-step pipeline per timestamp:

1. **Volatility adjustment** — Log returns are normalized by an EWMA volatility estimate and clipped at `cfg.clip` standard deviations to limit the influence of outliers.

2. **Covariance estimation** — A regularised correlation matrix is built from the vol-adjusted returns. Two modes are available (see [Covariance Modes](#covariance-modes)):

   - *EWMA with shrinkage* (default, `EwmaShrinkConfig`): An EWMA correlation matrix is computed over a `cfg.corr`-day lookback and blended toward the identity with weight `cfg.shrink` (λ):

     ```
     C_shrunk = λ · C_ewma + (1 − λ) · I
     ```

     See [Mode 1 — EWMA with Shrinkage](#mode-1--ewma-with-shrinkage) for guidance on choosing λ.

   - *Sliding-window factor model* (`SlidingWindowConfig`): The `window` most-recent vol-adjusted returns are decomposed via truncated SVD into `k` latent factors, giving a low-rank-plus-diagonal estimator solved efficiently via the Woodbury identity. **No explicit shrinkage is required** — `k` is the regularisation knob. See [Mode 2 — Sliding-Window Factor Model](#mode-2--sliding-window-factor-model).

3. **Position solving** — For each timestamp, the system `C · x = mu` is solved for `x` (the risk position vector). The solution is normalized by the inverse-matrix norm of `mu`, making positions scale-invariant with respect to signal magnitude. Positions are further scaled by a running profit-variance estimate to adapt risk dynamically.

Cash positions are obtained by dividing risk positions by per-asset EWMA volatility.

## Covariance Modes

Basanos offers two covariance-estimation strategies, selectable via the `covariance_config` field of `BasanosConfig`. Both solve the same linear system `C · x = μ` but differ in how `C` is regularised.

### Mode 1 — EWMA with Shrinkage

**Config class:** `EwmaShrinkConfig` (default; no extra fields needed beyond the top-level `BasanosConfig` parameters)

#### Why shrink?

Sample correlation matrices estimated from *T* observations of *n* assets are
poorly conditioned when *n* is large relative to *T* — the classical
*curse of dimensionality*. The Marchenko–Pastur law shows that extreme
eigenvalues of the sample matrix are severely biased (small eigenvalues are
deflated, large ones are inflated), making the matrix difficult to invert
reliably. Linear shrinkage toward the identity corrects this by pulling all
eigenvalues toward a common value, improving the numerical condition of the
matrix and reducing out-of-sample estimation error.

Basanos uses **convex linear shrinkage** (Ledoit & Wolf, 2004):

```
C_shrunk = λ · C_ewma + (1 − λ) · I_n
```

This is a special case of the general Ledoit–Wolf framework where the
shrinkage target is the identity matrix and the retention weight λ is
treated as a user-controlled hyperparameter. Unlike the analytically optimal
Ledoit–Wolf or Oracle Approximating Shrinkage (OAS) estimators, Basanos uses
a fixed λ — appropriate for *regularising a linear solver* rather than
*estimating a covariance matrix*, where practical stability often matters
more than minimum Frobenius loss.

Setting `λ = 0.0` (full shrinkage, C = I) is a meaningful corner case: the
system reduces to `x = μ`, i.e. signal-proportional sizing — the same result
a Markowitz optimizer produces when all assets are treated as uncorrelated.

#### How to choose `cfg.shrink` (= λ)

The key quantity is the **concentration ratio** *n / T*, where *n* = number
of assets and *T* = `cfg.corr` (the EWMA lookback).

| Regime | n / T ratio | Suggested λ | Rationale |
|--------|------------|-------------|-----------|
| Many assets, short lookback | > 0.5 | 0.3 – 0.5 | High noise; strong regularisation |
| Moderate assets and lookback | 0.1 – 0.5 | 0.5 – 0.7 | Balanced |
| Few assets, long lookback | < 0.1 | 0.7 – 0.9 | Well-conditioned sample; light regularisation |

A useful heuristic starting point is **λ ≈ 1 − n / (2·T)** (where *n* = number
of assets and *T* = `cfg.corr`), which approximates the Ledoit–Wolf optimal
intensity. Always validate on held-out data.

**Sensitivity notes:**

- Below λ ≈ 0.3 the matrix can become nearly singular for small portfolios
  (e.g., n > 10 with `corr` < 50), leading to numerically unstable positions.
- Above λ ≈ 0.8 the off-diagonal correlations are so heavily damped that the
  optimizer behaves almost as if all assets were independent.
- Shrinkage is most influential in the range λ ∈ [0.3, 0.8].

The `book/marimo/notebooks/shrinkage_guide.py` notebook shows the empirical
effect of different shrinkage levels on portfolio Sharpe ratio and position
stability for a realistic synthetic dataset.

### Mode 2 — Sliding-Window Factor Model

**Config class:** `SlidingWindowConfig(window=W, n_factors=k)`

#### How it works

At each timestamp *t*, the `W` most recent rows of vol-adjusted returns are
stacked into a matrix **R** ∈ ℝ^(W×n) and decomposed via **truncated SVD**
into *k* leading singular triplets. This yields a **factor risk model**:

```
Ĉ = (1/W) · V_k · Σ_k² · V_kᵀ + D̂
```

where **V_k** (n×k) contains the top-*k* right singular vectors (factor
loadings), **Σ_k** is the diagonal matrix of the top-*k* singular values, and
**D̂** = diag(d₁, …, dₙ) is chosen so that **Ĉ** has unit diagonal
(idiosyncratic variance).  The full decomposition is:

```
Σ = B · F · Bᵀ + D
```

| Term | Shape | Meaning |
|------|-------|---------|
| **B** = V_k | n × k | Factor loading matrix |
| **F** = Σ_k² / W | k × k | Factor covariance matrix |
| **D** | n × n (diagonal) | Per-asset idiosyncratic variance |

The linear system **Ĉ · x = μ** is then solved via the **Woodbury
(Sherman–Morrison) identity**, exploiting the low-rank-plus-diagonal structure
to reduce the per-step cost from O(n³) to **O(k³ + kn)** — a large saving
when k ≪ n.

#### Why no shrinkage is needed

The factor model is itself a **low-rank regularisation**. Truncating the SVD
to *k* factors discards all eigenvalues beyond rank *k*, replacing them with
the idiosyncratic floor **D̂**. This achieves the same goal as shrinkage but
through a different mechanism:

| Concept | EWMA + Shrinkage | Factor Model |
|---------|-----------------|--------------|
| **Regularisation mechanism** | Blend all eigenvalues toward 1 (identity target) | Retain only the top *k* eigenvalues; replace the rest with an asset-specific floor |
| **Regularisation knob** | `shrink` (λ ∈ [0, 1]) — closer to 0 = stronger | `n_factors` (k ≥ 1) — smaller k = stronger |
| **Extreme** | λ = 0 → C = I (fully uncorrelated) | k = 1 → single market factor |
| **Full structure** | λ = 1 → raw EWMA matrix | k = n → full sample covariance |
| **Ill-conditioning** | Can still be ill-conditioned at λ ≈ 1 with small T | Always well-conditioned; D is diagonal and strictly positive |
| **Solver cost** | O(n³) per step | **O(k³ + kn)** per step via Woodbury |

Because the idiosyncratic component **D** is strictly positive by construction,
**Ĉ** is always positive definite and invertible — regardless of `W` or `k`.
No additional regularisation (shrinkage) is required or applied.

#### How to choose `window` and `n_factors`

**`window` (W):**

- Rule of thumb: **W ≥ 2·n** keeps the sample covariance matrix over the window
  well-posed before truncation.
- Smaller W makes the estimator react faster to regime changes but increases
  estimation noise. Larger W is smoother but slower to adapt.
- The first W−1 rows of output will be zero (warm-up period).

**`n_factors` (k):**

- k controls how much of the correlation structure is captured. Think of it as
  the *effective rank* of the systematic component.
- **k = 1** recovers the single market-factor model (one global source of
  co-movement). This is the strongest regularisation short of the identity.
- **k = 2–5** is a natural range for diversified equity portfolios: market,
  sector, style factors.
- Increasing k captures finer cross-asset correlation at the cost of higher
  estimation noise in the factor loadings.
- Unlike shrinkage (a continuous [0, 1] dial), k is a discrete choice — use
  `engine.sharpe_at_window_factors(window=W, n_factors=k)` to sweep over
  candidate values on your dataset.

| Portfolio size | Typical k range | Rationale |
|---------------|-----------------|-----------|
| 2–5 assets | 1–2 | Very few independent sources of risk |
| 10–30 assets | 2–5 | Market + a handful of sector/style factors |
| 50–200 assets | 3–10 | Richer factor structure; keep k ≪ n |
| > 200 assets | 5–20 | Start from variance-explained criterion |

A practical starting point is to choose k such that the top-k factors explain
60–80% of the variance of the vol-adjusted returns over the window.

### Choosing Between Modes

| Criterion | EWMA + Shrinkage | Sliding-Window Factor Model |
|-----------|-----------------|----------------------------|
| **Tuning parameters** | λ (continuous) | W (window) + k (factors) |
| **Intuition** | "How much do I trust the raw EWMA correlations?" | "How many independent risk drivers exist in my universe?" |
| **Computational cost** | O(T·N²) for EWMA; O(N³) per solve | O(W·N·k) per step; O(k³ + kN) per solve |
| **Memory** | O(T·N²) peak | O(W·N) sliding buffer |
| **Warm-up** | Gradual (EWMA decay) | Hard cutoff at W rows |
| **Regime adaptability** | Smooth exponential decay | Hard rolling window (all rows equally weighted) |
| **Best for** | Moderate N (≤ 200), long histories, continuous λ search | Large N, short histories, interpretable factor structure |

When in doubt, start with **EWMA + shrinkage** (the default) — it requires
fewer design decisions and has a well-understood parameter space. Switch to
the **factor model** when:
- n is large and n/W approaches or exceeds 0.5 (sample covariance poorly conditioned),
- you want O(k³ + kn) per-step cost rather than O(n³),
- you want an interpretable factor structure (e.g., for risk attribution), or
- you prefer a single discrete knob (k) over a continuous one (λ).

### References

- Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for
  large-dimensional covariance matrices.* Journal of Multivariate Analysis,
  88(2), 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). *Shrinkage
  algorithms for MMSE covariance estimation.* IEEE Transactions on Signal
  Processing, 58(10), 5016–5029. https://doi.org/10.1109/TSP.2010.2053029
- Stein, C. (1956). *Inadmissibility of the usual estimator for the mean of a
  multivariate normal distribution.* Proceedings of the Third Berkeley
  Symposium, 1, 197–206.
- Woodbury, M. A. (1950). *Inverting modified matrices.* Memorandum Report 42,
  Statistical Research Group, Princeton University.

## Performance Characteristics

> **TL;DR** — the optimizer is practical for ≤ 250 assets with ≤ 10 years of
> daily data on a 16 GB workstation.  Beyond those limits, memory or compute
> time becomes the bottleneck.

### Computational complexity

Let *N* = number of assets and *T* = number of timestamps.

| Step | Complexity | Bottleneck |
|------|-----------|------------|
| Vol-adjustment (`ret_adj`, `vola`) | O(T·N) | EWMA per asset; scales linearly |
| EWM correlation (`cor`) | **O(T·N²)** | `lfilter` over all N² pairs in parallel |
| Linear solve per row (`cash_position`) | **O(N³) × T solves** | Cholesky/LU decomposition per timestamp |

For most practical portfolio sizes (N ≤ 200) the correlation step dominates.
At very large N (≥ 500) the per-solve cost O(N³) can also become significant.

### Memory usage

`_ewm_corr_numpy` allocates roughly **14 float64 arrays** of shape `(T, N, N)`
simultaneously at peak (input sequences fed to `scipy.signal.lfilter`, the IIR
filter outputs, the five EWM component arrays, and the result tensor):

```
Peak RAM ≈ 14 × 8 × T × N²  bytes  ≈  112 × T × N²  bytes
```

Practical working sizes:

| N (assets) | T (daily rows) | Approx. history | Peak memory |
|-----------|---------------|-----------------|-------------|
| 50 | 252 | ~1 year | ~70 MB |
| 100 | 252 | ~1 year | ~280 MB |
| 100 | 1 260 | ~5 years | ~1.4 GB |
| 100 | 2 520 | ~10 years | **~2.8 GB** |
| 200 | 1 260 | ~5 years | ~5.6 GB |
| 200 | 2 520 | ~10 years | **~11 GB** |
| 500 | 2 520 | ~10 years | **~70 GB ⚠** |
| 1 000 | 2 520 | ~10 years | **~280 GB ⛔** |

### Practical limits

| Zone | Condition | Guidance |
|------|-----------|----------|
| ✅ Comfortable | N ≤ 150, T ≤ 1 260 (~5 yr daily) | Runs on an 8 GB laptop in seconds |
| ⚠ Feasible with care | N ≤ 250, T ≤ 2 520 (~10 yr daily) | Requires ~11–12 GB RAM; plan for 10–60 s wall time |
| 🔴 Impractical | N > 500 or T > 5 000 | Peak memory exceeds 16 GB; consider mitigation strategies below |
| ⛔ Not supported | N > 1 000 with multi-year history | Solve cost and memory are prohibitive on commodity hardware |

> **Note on `cfg.corr`** — this is the EWM lookback window, not the total
> dataset length.  Even if you have 10 years of prices, keeping `cfg.corr`
> short (e.g., 63 days) does *not* reduce the peak memory cost of
> `_ewm_corr_numpy`: the function always allocates the full `(T, N, N)` tensor
> regardless of the lookback value.  To limit memory, reduce the number of rows
> passed in *T* itself (e.g., trim old prices) rather than adjusting `cfg.corr`.

### Mitigation strategies

When you hit memory or performance limits:

1. **Reduce the asset universe** — keep only the most liquid or relevant assets;
   pre-filter with univariate signal strength before running the optimizer.
2. **Shorten the price history** — `_ewm_corr_numpy` processes every row; trim
   older data to the minimum needed for the EWM warm-up (`cfg.corr` rows).
3. **Increase `cfg.shrink` toward 1.0** — stronger identity shrinkage reduces
   the sensitivity of the solve to noisy off-diagonal entries, allowing a
   shorter effective lookback without instability.
4. **Process in rolling windows** — run the optimizer on overlapping windows
   (e.g., 1-year chunks) and stitch results; correlation estimates will differ
   slightly at window boundaries but memory stays bounded.
5. **Use `cor_tensor` instead of `cor`** — returns a single `(T, N, N)` NumPy
   array rather than a Python dict, avoiding Python object overhead for large T.

### Benchmark data

Measured on a GitHub Actions runner (AMD EPYC 7763, 4 vCPUs, Python 3.12):

| Dataset | `cor` time | `cash_position` time |
|---------|-----------|---------------------|
| 5 assets, 252 rows (~1 yr) | 1.2 ms | 56 ms |
| 5 assets, 1 260 rows (~5 yr) | 5.4 ms | 222 ms |
| 20 assets, 252 rows (~1 yr) | 13.6 ms | — |

See [`BENCHMARKS.md`](BENCHMARKS.md) for full results and regression baselines.

## API Reference

### `basanos.math`

```python
from basanos.math import BasanosConfig, BasanosEngine, FactorModel
from basanos.math import EwmaShrinkConfig, SlidingWindowConfig, CovarianceMode
```

| Class | Description |
|-------|-------------|
| `BasanosConfig` | Immutable configuration (Pydantic model) |
| `BasanosEngine` | Core optimizer; produces positions and a `Portfolio` |
| `FactorModel` | Factor risk model decomposition Σ = B·F·Bᵀ + D |
| `EwmaShrinkConfig` | Covariance config for EWMA + shrinkage mode (default) |
| `SlidingWindowConfig` | Covariance config for sliding-window factor model mode |
| `CovarianceMode` | Enum: `ewma_shrink` / `sliding_window` |

**`BasanosEngine` properties**

| Property | Returns | Description |
|----------|---------|-------------|
| `assets` | `list[str]` | Numeric asset column names |
| `ret_adj` | `pl.DataFrame` | Vol-adjusted, clipped log returns |
| `vola` | `pl.DataFrame` | Per-asset EWMA volatility |
| `cor` | `dict[date, np.ndarray]` | EWMA correlation matrices keyed by date |
| `cor_tensor` | `np.ndarray` | All correlation matrices stacked as a `(T, N, N)` tensor; supports `.npy` round-trip |
| `cash_position` | `pl.DataFrame` | Optimized cash positions (risk divided by EWMA volatility) |
| `position_status` | `pl.DataFrame` | Per-row reason code for cash_position: `warmup`, `zero_signal`, `degenerate`, or `valid` |
| `risk_position` | `pl.DataFrame` | Risk positions before volatility scaling (= `cash_position × vola`) |
| `position_leverage` | `pl.DataFrame` | L1 norm of cash positions (gross leverage) per timestamp |
| `condition_number` | `pl.DataFrame` | Condition number κ of the shrunk correlation matrix per timestamp |
| `effective_rank` | `pl.DataFrame` | Entropy-based effective rank of the shrunk correlation matrix per timestamp |
| `solver_residual` | `pl.DataFrame` | Euclidean residual norm ‖C_shrunk·x − μ‖₂ per timestamp |
| `signal_utilisation` | `pl.DataFrame` | Per-asset fraction of μ surviving the correlation filter; 1 when C = I |
| `ic` | `pl.DataFrame` | Cross-sectional Pearson IC time series (signal vs. one-period forward return) |
| `rank_ic` | `pl.DataFrame` | Cross-sectional Spearman Rank IC time series |
| `ic_mean` | `float` | Mean IC across all timestamps (NaN-ignoring) |
| `ic_std` | `float` | Sample standard deviation of IC |
| `icir` | `float` | IC Information Ratio (`ic_mean / ic_std`) |
| `rank_ic_mean` | `float` | Mean Rank IC across all timestamps |
| `rank_ic_std` | `float` | Sample standard deviation of Rank IC |
| `naive_sharpe` | `float` | Sharpe ratio of the naïve equal-weight signal (μ = 1 benchmark) |
| `portfolio` | `Portfolio` | Ready-to-use portfolio for analytics |
| `config_report` | `ConfigReport` | HTML report with lambda-sweep chart + parameter table |

**`BasanosEngine` methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `sharpe_at_shrink` | `sharpe_at_shrink(shrink: float) → float` | Annualised Sharpe ratio for a given shrinkage weight λ ∈ [0, 1]; use for lambda sweeps (EWMA mode) |
| `sharpe_at_window_factors` | `sharpe_at_window_factors(window: int, n_factors: int) → float` | Annualised Sharpe ratio for given sliding-window `window` and `n_factors`; sweeps factor model hyperparameters |

**`BasanosConfig` properties**

| Property | Returns | Description |
|----------|---------|-------------|
| `report` | `ConfigReport` | HTML report with parameter table, shrinkage guidance, and theory (no lambda sweep) |

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
| `Report` | HTML report facade; produces self-contained dark-themed reports |

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
| `turnover` | Daily one-way turnover as a fraction of AUM |
| `turnover_weekly` | Weekly-aggregated turnover |
| `stats` | `Stats` instance |
| `plots` | `Plots` instance |
| `report` | `Report` instance for HTML report generation |

**`Portfolio` methods (trading costs)**

| Method | Description |
|--------|-------------|
| `cost_adjusted_returns(cost_bps)` | Daily returns net of estimated one-way trading costs (in bps) |
| `trading_cost_impact(max_bps=20)` | Sharpe ratio at each integer cost level 0 … max_bps (returns `pl.DataFrame`) |
| `turnover_summary()` | Summary DataFrame: mean daily/weekly turnover and turnover std |

**`Stats` methods**

| Method | Description |
|--------|-------------|
| `sharpe(periods)` | Annualized Sharpe ratio |
| `volatility(periods, annualize)` | Standard deviation of returns |
| `rolling_sharpe(window, periods)` | Rolling annualised Sharpe ratio time series |
| `rolling_volatility(window, periods, annualize)` | Rolling volatility time series |
| `annual_breakdown()` | Full summary statistics broken down by calendar year |
| `skew()` | Skewness |
| `kurtosis()` | Excess kurtosis |
| `value_at_risk(alpha, sigma)` | Parametric VaR |
| `conditional_value_at_risk(alpha, sigma)` | Expected shortfall (CVaR) |
| `avg_return()` | Mean return (zeros excluded) |
| `avg_win()` | Mean positive return |
| `avg_loss()` | Mean negative return |
| `win_rate()` | Fraction of profitable periods |
| `profit_factor()` | Gross wins / absolute gross losses |
| `payoff_ratio()` | Average win / absolute average loss |
| `monthly_win_rate()` | Fraction of profitable calendar months |
| `best()` | Maximum single-period return |
| `worst()` | Minimum single-period return |
| `worst_n_periods(n)` | *N* worst return periods (default 5) |
| `up_capture(benchmark)` | Up-market capture ratio vs. benchmark |
| `down_capture(benchmark)` | Down-market capture ratio vs. benchmark |
| `max_drawdown()` | Largest peak-to-trough decline as a fraction of peak |
| `avg_drawdown()` | Mean drawdown across all underwater periods |
| `max_drawdown_duration()` | Longest consecutive underwater period (calendar days) |
| `calmar(periods)` | Annualized return divided by max drawdown |
| `recovery_factor()` | Total return divided by max drawdown |

**`Plots` methods (rolling & sub-period)**

| Method | Description |
|--------|-------------|
| `rolling_sharpe_plot(window)` | Line chart of rolling Sharpe ratio |
| `rolling_volatility_plot(window)` | Line chart of rolling annualised volatility |
| `annual_sharpe_plot()` | Bar chart of Sharpe ratio by calendar year |
| `trading_cost_impact_plot(max_bps=20)` | Line chart of Sharpe ratio vs. one-way trading cost (0–max_bps bps) |

---

### `basanos.math.FactorModel`

A frozen dataclass representing a factor risk model decomposition:

```
Σ = B · F · Bᵀ + D
```

where **B** (n×k) is the factor loading matrix, **F** (k×k) is the factor covariance matrix, and **D** is the diagonal idiosyncratic variance.

```python
from basanos.math import FactorModel
import numpy as np

# Fit from a return matrix (T×n) using truncated SVD
returns = np.random.default_rng(0).normal(size=(200, 5))
fm = FactorModel.from_returns(returns, k=2)

fm.n_assets    # 5
fm.n_factors   # 2
fm.covariance  # reconstructed (5, 5) covariance matrix
```

**`FactorModel` attributes**

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `factor_loadings` | `(n, k)` | Factor loading matrix **B**; column *j* gives each asset's sensitivity to factor *j* |
| `factor_covariance` | `(k, k)` | Factor covariance matrix **F** (positive definite) |
| `idiosyncratic_var` | `(n,)` | Per-asset idiosyncratic variance (strictly positive) |

**`FactorModel` properties and methods**

| Name | Returns | Description |
|------|---------|-------------|
| `n_assets` | `int` | Number of assets *n* |
| `n_factors` | `int` | Number of factors *k* |
| `covariance` | `np.ndarray (n, n)` | Reconstructed full covariance matrix Σ = B·F·Bᵀ + D |
| `from_returns(returns, k)` | `FactorModel` | Class method; fits the model from a `(T, n)` return matrix via truncated SVD |

---

### `basanos.math.ConfigReport`

Accessed via `cfg.report` (config-only) or `engine.config_report` (includes lambda sweep).

```python
from basanos.math import BasanosConfig, BasanosEngine

cfg    = BasanosConfig(vola=16, corr=32, clip=3.5, shrink=0.5, aum=1e6)
engine = BasanosEngine(prices=_prices, mu=_mu, cfg=cfg)

cfg.report.to_html()          # parameter table + shrinkage guidance + theory
engine.config_report.to_html()  # above + interactive lambda-sweep chart
```

**`ConfigReport` methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_html` | `to_html(title="Basanos Config Report") -> str` | Returns the complete HTML document as a string. |
| `save` | `save(path, title="Basanos Config Report") -> Path` | Writes the HTML document to *path*. A `.html` suffix is appended when missing. |

---

### `basanos.analytics.Report`

Accessed via `portfolio.report`.  Produces a self-contained HTML document with
dark-themed styling, a statistics table, and embedded interactive Plotly charts.

**`Report` methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_html` | `to_html(title="Basanos Portfolio Report") -> str` | Returns the complete HTML document as a string.  Plotly.js is loaded once from the CDN. |
| `save` | `save(path, title="Basanos Portfolio Report") -> Path` | Writes the HTML document to *path*.  A `.html` suffix is appended when the path has no extension. |

## Configuration Reference

### `BasanosConfig` — core parameters

**Required parameters**

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `vola` | `int` | `> 0` | EWMA lookback for volatility (days) |
| `corr` | `int` | `>= vola` | EWMA lookback for correlation (days); used only in `ewma_shrink` mode |
| `clip` | `float` | `> 0` | Clipping threshold for vol-adjusted returns |
| `shrink` | `float` | `[0, 1]` | Shrinkage intensity (λ) — `0` = identity, `1` = raw EWMA; used only in `ewma_shrink` mode |
| `aum` | `float` | `> 0` | Assets under management for position scaling |

**Optional parameters** (sensible defaults for most use cases)

| Parameter | Type | Default | Constraint | Description |
|-----------|------|---------|------------|-------------|
| `covariance_config` | `EwmaShrinkConfig \| SlidingWindowConfig` | `EwmaShrinkConfig()` | — | Covariance estimation strategy; see below |
| `profit_variance_init` | `float` | `1.0` | `> 0` | Initial value for the profit-variance EMA |
| `profit_variance_decay` | `float` | `0.99` | `(0, 1)` | EMA decay factor λ for realised P&L variance; default gives ~69-period half-life |
| `denom_tol` | `float` | `1e-12` | `> 0` | Minimum normalisation denominator; positions are zeroed at or below this threshold |
| `position_scale` | `float` | `1e6` | `> 0` | Multiplicative factor applied to dimensionless risk positions to obtain base-currency cash positions |
| `min_corr_denom` | `float` | `1e-14` | `> 0` | Guard threshold for the EWMA correlation denominator; correlations below this are set to NaN |
| `max_nan_fraction` | `float` | `0.9` | `(0, 1)` | Maximum tolerated fraction of null values in any asset price column before raising `ExcessiveNullsError` |

### `EwmaShrinkConfig` — EWMA with shrinkage (default)

No additional fields beyond the top-level `BasanosConfig` parameters (`corr`, `shrink`).

```python
from basanos.math import BasanosConfig  # EwmaShrinkConfig is the default

# Conservative — longer lookbacks, stronger shrinkage
conservative = BasanosConfig(vola=32, corr=64, clip=3.0, shrink=0.7, aum=1e6)

# Responsive — shorter lookbacks, lighter shrinkage
responsive   = BasanosConfig(vola=8,  corr=16, clip=4.0, shrink=0.3, aum=1e6)
```

### `SlidingWindowConfig` — sliding-window factor model

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `window` | `int` | `> 0` | Rolling window length *W* (number of most-recent observations). Rule of thumb: *W* ≥ 2·n_assets. The first *W*−1 output rows are zero (warm-up). |
| `n_factors` | `int` | `> 0` | Number of latent factors *k*. Fewer factors = stronger regularisation. *k* = 1 = single market factor; *k* = 2–5 typical for diversified equity. |
| `max_components` | `int \| None` | `> 0` or `None` | Optional hard cap on SVD components used per streaming step. When set, the effective component count is `min(n_factors, window, n_valid_assets, max_components)`. Useful for large universes where only a few factors dominate and you want to limit SVD cost. Defaults to `None` (no extra cap). |

> **Effective component count:** at each streaming step the number of SVD components actually used is `k_eff = min(n_factors, window, n_valid_assets[, max_components])`. This implicit truncation ensures the SVD remains well-posed when assets temporarily drop out of the universe. Use `max_components` for explicit control over computational cost in large universes.

> **Note:** `corr` and `shrink` on `BasanosConfig` are ignored in `sliding_window` mode. They remain required fields for API consistency (and for `ewma_shrink` mode), but have no effect on factor-model positions. Any valid value (e.g. `corr=32, shrink=0.5`) is acceptable as a placeholder.

```python
from basanos.math import BasanosConfig, SlidingWindowConfig

# Factor model — 60-day window, 3 latent factors
cfg = BasanosConfig(
    vola=16,
    corr=32,   # unused in sliding_window mode but still required
    clip=3.5,
    shrink=0.5,  # unused in sliding_window mode
    aum=1e6,
    covariance_config=SlidingWindowConfig(window=60, n_factors=3),
)

# Single market-factor model — maximum regularisation
single_factor = BasanosConfig(
    vola=16, corr=32, clip=3.5, shrink=0.5, aum=1e6,
    covariance_config=SlidingWindowConfig(window=120, n_factors=1),
)
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

For architectural decisions and their rationale, see [docs/adr/](docs/adr/).

## License

See [LICENSE](LICENSE) for details.
