# Covariance Modes & How It Works

This page explains the internal pipeline of the Basanos optimizer and provides
a deep dive into the two available covariance-estimation strategies.

---

## How It Works

The optimizer implements a three-step pipeline per timestamp:

1. **Volatility adjustment** — Log returns are normalized by an EWMA volatility estimate and clipped at `cfg.clip` standard deviations to limit the influence of outliers.

2. **Covariance estimation** — A regularised correlation matrix is built from the vol-adjusted returns. Two modes are available (see [Covariance Modes](#covariance-modes) below):

   - *EWMA with shrinkage* (default, `EwmaShrinkConfig`): blends the `cfg.corr`-day EWMA correlation toward the identity at rate `cfg.shrink` (λ). See [Mode 1](#mode-1--ewma-with-shrinkage) for guidance on λ.
   - *Sliding-window factor model* (`SlidingWindowConfig`): decomposes the `window` most-recent vol-adjusted returns via truncated SVD into `k` factors, producing a low-rank-plus-diagonal estimator solved via the Woodbury identity. See [Mode 2](#mode-2--sliding-window-factor-model).

3. **Position solving** — The system `C · x = mu` is solved for `x` (the risk position vector), normalised by the inverse-matrix norm of `mu` for scale invariance, and further scaled by a running profit-variance estimate.

Cash positions are obtained by dividing risk positions by per-asset EWMA volatility.

---

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

---

## References

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
