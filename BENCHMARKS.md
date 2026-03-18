# Basanos — Benchmark Baseline Results

This document records the **baseline performance metrics** for `basanos` core
components.  Results are produced by the pytest-benchmark suite in
`tests/benchmarks/` and should be updated whenever a significant performance
change is introduced.

> **Regression detection** is automated in CI via the
> [`rhiza_benchmarks.yml`](.github/workflows/rhiza_benchmarks.yml) workflow,
> which alerts when any benchmark degrades by more than 150 % relative to the
> stored baseline.

---

## Environment

| Property           | Value                                  |
|--------------------|----------------------------------------|
| Date               | 2026-03-18                             |
| Python             | CPython 3.12 (`.python-version`)       |
| Benchmark Python   | CPython 3.12.3 (GitHub Actions runner) |
| OS                 | Linux (Ubuntu, Azure runner)           |
| CPU                | AMD EPYC 7763 64-Core Processor        |
| CPU speed          | 3.25 GHz                               |
| CPU cores          | 4 (virtual, Azure Standard DS2 v2)     |
| pytest-benchmark   | 5.2.3                                  |
| basanos version    | 0.3.0                                  |
| Commit             | `86fb095`                              |

Dataset sizes used in the benchmarks:

| Suffix | Rows  | Assets | Description            |
|--------|-------|--------|------------------------|
| 252_5  | 252   | 5      | ~1 year, 5 assets      |
| 1260_5 | 1260  | 5      | ~5 years, 5 assets     |
| 252_20 | 252   | 20     | ~1 year, 20 assets     |

---

## Results

Times are **mean wall-clock** over all benchmark rounds (min-of-runs strategy
used by pytest-benchmark to reduce scheduling noise).  The raw JSON is stored
in [`benchmarks/results/baseline.json`](benchmarks/results/baseline.json).

### Portfolio

| Benchmark                     | Mean      | Min       | Std Dev  | OPS   | Rounds |
| ----------------------------- | --------- | --------- | -------- | ----- | ------ |
| test_profits_252_5            | 3.257 ms  | 3.127 ms  | 89.3 µs  | 307   | 222    |
| test_profits_1260_5           | 3.303 ms  | 3.139 ms  | 89.2 µs  | 303   | 290    |
| test_profits_252_20           | 12.012 ms | 11.661 ms | 209.1 µs | 83    | 81     |
| test_nav_accumulated_252_5    | 3.613 ms  | 3.450 ms  | 125.3 µs | 277   | 253    |
| test_nav_compounded_252_5     | 4.051 ms  | 3.420 ms  | 619.5 µs | 247   | 260    |
| test_drawdown_252_5           | 3.860 ms  | 3.465 ms  | 360.4 µs | 259   | 243    |
| test_drawdown_1260_5          | 3.928 ms  | 3.730 ms  | 217.4 µs | 255   | 249    |
| test_monthly_252_5            | 4.396 ms  | 4.172 ms  | 120.6 µs | 227   | 177    |
| test_tilt_timing_decomp_252_5 | 15.758 ms | 15.416 ms | 223.0 µs | 64    | 61     |
| test_all_252_5                | 7.940 ms  | 7.671 ms  | 445.7 µs | 126   | 128    |

### Stats

| Benchmark              | Mean     | Min      | Std Dev | OPS    | Rounds |
| ---------------------- | -------- | -------- | ------- | ------ | ------ |
| test_volatility_252    | 196.2 µs | 178.7 µs | 11.8 µs | 5,098  | 3258   |
| test_sharpe_252        | 201.2 µs | 180.9 µs | 11.6 µs | 4,970  | 3745   |
| test_value_at_risk_252 | 72.0 µs  | 67.4 µs  | 6.8 µs  | 13,884 | 3463   |
| test_summary_252       | 4.138 ms | 4.004 ms | 71.9 µs | 242    | 199    |
| test_summary_1260      | 4.365 ms | 4.181 ms | 89.1 µs | 229    | 221    |

### BasanosEngine

| Benchmark                 | Mean       | Min        | Std Dev  | OPS   | Rounds |
| ------------------------- | ---------- | ---------- | -------- | ----- | ------ |
| test_ret_adj_252_5        | 427.3 µs   | 403.4 µs   | 13.8 µs  | 2,340 | 1804   |
| test_vola_252_5           | 251.9 µs   | 219.5 µs   | 19.0 µs  | 3,970 | 4209   |
| test_cor_252_5            | 1.236 ms   | 1.195 ms   | 20.1 µs  | 809   | 601    |
| test_cor_1260_5           | 5.047 ms   | 4.897 ms   | 99.1 µs  | 198   | 212    |
| test_cor_252_20           | 13.387 ms  | 13.096 ms  | 218.5 µs | 75    | 70     |
| test_cash_position_252_5  | 68.279 ms  | 67.143 ms  | 1.057 ms | 15    | 14     |
| test_cash_position_1260_5 | 343.011 ms | 341.686 ms | 1.129 ms | 3     | 5      |
| test_portfolio_252_5      | 68.545 ms  | 67.227 ms  | 769.7 µs | 15    | 15     |

---

## Running Benchmarks

```bash
make benchmark
```

This installs `pytest-benchmark` and `pygal`, runs the full benchmark suite,
and writes:

| Path                         | Content                              |
|------------------------------|--------------------------------------|
| `_tests/benchmarks/results.json` | Raw pytest-benchmark JSON output |
| `_tests/benchmarks/histogram.*`  | Per-test latency histograms      |
| `_tests/benchmarks/report.html`  | HTML summary report              |

The `_tests/` directory is excluded from version control (see `.gitignore`).
To publish a new baseline, copy `_tests/benchmarks/results.json` to
`benchmarks/results/baseline.json` and update the tables in this file.

---

## CI Regression Detection

The [`rhiza_benchmarks.yml`](.github/workflows/rhiza_benchmarks.yml) workflow:

- Runs on every push to `main` / `master` and on pull requests.
- Stores benchmark history in the `gh-pages` branch under `/benchmarks/`.
- **Posts a PR comment** when any benchmark regresses by more than 150 %.
- **Fails the workflow** on pushes to `main` when a regression is detected.

The regression threshold can be adjusted via the `alert-threshold` key in the
workflow file.
