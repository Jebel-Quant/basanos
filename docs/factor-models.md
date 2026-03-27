# Factor Models

`FactorModel` provides a low-rank decomposition of the asset covariance matrix
and serves as the covariance estimator in **sliding-window mode**
(`SlidingWindowConfig`).  It is also available for standalone low-rank
covariance inspection.

---

## Mathematical background

The factor risk model represents a covariance matrix as:

```
Σ = B · F · Bᵀ + D
```

where:

| Term | Shape | Meaning |
|------|-------|---------|
| **B** | n × k | Factor loading matrix — column *j* gives each asset's sensitivity to factor *j* |
| **F** | k × k | Factor covariance matrix (positive definite) |
| **D** | n × n (diagonal) | Per-asset idiosyncratic variance (strictly positive) |

Fitting proceeds by truncated SVD of the vol-adjusted return matrix
**R** ∈ ℝ^(T×n):

```
Ĉ = (1/T) · V_k · Σ_k² · V_kᵀ + D̂
```

where **V_k** (n×k) contains the top-*k* right singular vectors and **Σ_k** is
the diagonal of the top-*k* singular values.  **D̂** is set so that **Ĉ** has
unit diagonal.

---

## Quick example

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

---

## `FactorModel` reference

### Attributes

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `factor_loadings` | `(n, k)` | Factor loading matrix **B**; column *j* gives each asset's sensitivity to factor *j* |
| `factor_covariance` | `(k, k)` | Factor covariance matrix **F** (positive definite) |
| `idiosyncratic_var` | `(n,)` | Per-asset idiosyncratic variance (strictly positive) |

### Properties and methods

| Name | Returns | Description |
|------|---------|-------------|
| `n_assets` | `int` | Number of assets *n* |
| `n_factors` | `int` | Number of factors *k* |
| `covariance` | `np.ndarray (n, n)` | Reconstructed full covariance matrix Σ = B·F·Bᵀ + D |
| `from_returns(returns, k)` | `FactorModel` | Class method; fits the model from a `(T, n)` return matrix via truncated SVD |

---

## Using FactorModel in BasanosEngine

Pass a `SlidingWindowConfig` to switch the engine to factor-model covariance:

```python
from basanos.math import BasanosConfig, BasanosEngine, SlidingWindowConfig

cfg = BasanosConfig(
    vola=16,
    corr=32,    # unused in sliding_window mode but still required
    clip=3.5,
    shrink=0.5, # unused in sliding_window mode
    aum=1e6,
    covariance_config=SlidingWindowConfig(
        window=60,    # rolling window length W; rule of thumb: W >= 2 * n_assets
        n_factors=2,  # number of latent factors k; fewer = stronger regularisation
    ),
)

engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
```

At each timestamp the engine fits a rolling `FactorModel` over the last `window`
rows and solves `Ĉ · x = μ` via the **Woodbury identity**, reducing the
per-step cost from O(n³) to **O(k³ + kn)**.

For guidance on choosing `window` and `n_factors` see
[Concepts — Mode 2](concepts.md#mode-2--sliding-window-factor-model).

For the full API documentation see [Risk Models & Helpers](api/models.md).
