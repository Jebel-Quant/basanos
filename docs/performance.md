# Performance Characteristics

> **TL;DR** — the optimizer is practical for ≤ 250 assets with ≤ 10 years of
> daily data on a 16 GB workstation.  Beyond those limits, memory or compute
> time becomes the bottleneck.

## Computational complexity

Let *N* = number of assets and *T* = number of timestamps.

| Step | Complexity | Bottleneck |
|------|-----------|------------|
| Vol-adjustment (`ret_adj`, `vola`) | O(T·N) | EWMA per asset; scales linearly |
| EWM correlation (`cor`) | **O(T·N²)** | `lfilter` over all N² pairs in parallel |
| Linear solve per row (`cash_position`) | **O(N³) × T solves** | Cholesky/LU decomposition per timestamp |

For most practical portfolio sizes (N ≤ 200) the correlation step dominates.
At very large N (≥ 500) the per-solve cost O(N³) can also become significant.

## Memory usage

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

## Practical limits

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

## Mitigation strategies

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

## Benchmark data

Measured on a GitHub Actions runner (AMD EPYC 7763, 4 vCPUs, Python 3.12):

| Dataset | `cor` time | `cash_position` time |
|---------|-----------|---------------------|
| 5 assets, 252 rows (~1 yr) | 1.2 ms | 56 ms |
| 5 assets, 1 260 rows (~5 yr) | 5.4 ms | 222 ms |
| 20 assets, 252 rows (~1 yr) | 13.6 ms | — |

See [`BENCHMARKS.md`](../BENCHMARKS.md) for full results and regression baselines.
