# The Problem Basanos Solves

## The Signal Sizing Problem

Every systematic strategy generates a **signal vector** μ — one number per asset that encodes how bullish or bearish the model is on each instrument. The naive approach is to size positions in direct proportion to the signal:

```
x_i = μ_i
```

This is simple but ignores a critical structural reality: **correlated assets receive overlapping bets**. If two assets move together 80% of the time and both have bullish signals, holding both at full signal size concentrates risk rather than diversifying it. The portfolio ends up with one large, correlated bet dressed up as two independent ones.

The outcome is poor: realized P&L is noisier than expected, drawdowns are deeper, and the Sharpe ratio understates the true quality of the signal. When the signal is evaluated in this naive framework, it looks weaker than it is — or, worse, a weak signal disguised by incidental factor exposure looks stronger than it is.

## The Approach

Basanos reframes position sizing as a **linear system**:

```
C · x = μ
```

where C is the time-varying, shrunk correlation matrix and μ is the signal. Solving for x inverts the correlation structure: assets that co-move heavily with the rest of the portfolio are automatically scaled down; idiosyncratic assets can carry more. The result is a set of *risk positions* that express the full information in μ while respecting the portfolio's correlation geometry.

This is not a new idea — it is the reduced form of mean-variance optimization when the objective is to maximize signal-to-noise subject to a risk budget. What Basanos adds is a set of practical design choices that make the approach stable and useful for **signal assessment** rather than production execution.

### Three stabilizing choices

**1. EWMA estimates.** Both volatility and correlations are computed as exponentially weighted moving averages. This allows the optimizer to adapt to changing regimes without requiring a fixed lookback window or periodic re-estimation.

**2. Identity shrinkage.** The estimated correlation matrix is blended toward the identity matrix with a user-controlled weight λ:

```
C_shrunk = λ · C_ewma + (1 − λ) · I
```

Shrinkage regularises the linear solve, guards against noise in the off-diagonal entries, and prevents extreme positions when the number of assets is large relative to the estimation window. At λ = 0 (full shrinkage), C = I and the system reduces to x = μ — signal-proportional sizing, identical to what a Markowitz optimizer produces when all assets are treated as uncorrelated.

**3. Scale invariance.** Positions are normalized by the inverse-matrix norm of μ, so doubling the signal magnitude does not double the position. Risk is instead scaled by a running estimate of realized profit variance — increasing exposure in good regimes and reducing it in bad ones.

### From risk positions to cash positions

The output of the linear solve is a *risk position* measured in units of volatility. Dividing by per-asset EWMA volatility converts it into a *cash position* — the dollar amount to hold in each asset.

## Why Not a Full Optimizer?

The primary use case for Basanos is **signal assessment**, not production execution. A fully constrained Markowitz optimizer — with turnover limits, sector caps, leverage constraints, and factor neutrality targets — bends positions away from what the signal implies. The resulting P&L reflects the signal's interaction with all those constraints, making it hard to isolate whether the underlying signal has edge.

Basanos deliberately avoids hard constraints so the signal has room to express itself cleanly. By orthogonalizing μ to known risk factors before passing it in, the pure alpha component can be further isolated.

This makes Basanos a natural **first hurdle**: a signal that cannot generate a reasonable Sharpe through this minimal framework is unlikely to survive the additional friction of a production optimizer.

## What Basanos Measures

Once positions are computed, the analytics layer evaluates:

- **Sharpe ratio**, volatility, drawdown, VaR, CVaR, and distribution moments
- **Tilt/timing decomposition** — separating the static allocation component from the dynamic timing component
- **Lead/lag analysis** — Sharpe ratio across position lags to detect execution delay sensitivity
- **Trading cost impact** — how quickly the edge degrades as one-way transaction costs increase
- **Configuration sensitivity** — how the Sharpe ratio responds to different shrinkage levels (lambda sweep), guiding hyperparameter choice

The goal throughout is to answer the question: *does this signal have real, robust edge before it encounters production constraints?*
