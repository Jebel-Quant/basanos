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
| Date               | 2026-03-16                             |
| Python             | CPython 3.12 (`.python-version`)       |
| Benchmark Python   | CPython 3.12.3 (GitHub Actions runner) |
| OS                 | Linux (Ubuntu, Azure runner)           |
| CPU                | AMD EPYC 7763 64-Core Processor        |
| CPU speed          | 3.16 GHz                               |
| CPU cores          | 4 (virtual, Azure Standard DS2 v2)     |
| pytest-benchmark   | 5.2.3                                  |
| basanos version    | 0.2.2                                  |
| Commit             | `7fafc1d`                              |

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
| test_profits_252_5            | 3.340 ms  | 3.126 ms  | 93.2 µs  | 299   | 239    |
| test_profits_1260_5           | 3.361 ms  | 3.212 ms  | 76.6 µs  | 298   | 266    |
| test_profits_252_20           | 12.283 ms | 11.969 ms | 188.0 µs | 81    | 79     |
| test_nav_accumulated_252_5    | 3.719 ms  | 3.394 ms  | 162.8 µs | 269   | 239    |
| test_nav_compounded_252_5     | 3.790 ms  | 3.668 ms  | 75.0 µs  | 264   | 258    |
| test_drawdown_252_5           | 3.920 ms  | 3.766 ms  | 74.4 µs  | 255   | 223    |
| test_drawdown_1260_5          | 3.992 ms  | 3.855 ms  | 90.9 µs  | 251   | 230    |
| test_monthly_252_5            | 4.504 ms  | 4.382 ms  | 57.6 µs  | 222   | 200    |
| test_tilt_timing_decomp_252_5 | 16.214 ms | 15.747 ms | 272.7 µs | 62    | 61     |
| test_all_252_5                | 8.028 ms  | 7.855 ms  | 120.3 µs | 125   | 121    |

### Stats

| Benchmark              | Mean     | Min      | Std Dev | OPS    | Rounds |
| ---------------------- | -------- | -------- | ------- | ------ | ------ |
| test_volatility_252    | 204.3 µs | 185.2 µs | 11.0 µs | 4,894  | 3128   |
| test_sharpe_252        | 201.6 µs | 188.7 µs | 10.2 µs | 4,959  | 4152   |
| test_value_at_risk_252 | 72.8 µs  | 68.1 µs  | 7.1 µs  | 13,741 | 4013   |
| test_summary_252       | 868.5 µs | 827.8 µs | 20.4 µs | 1,151  | 754    |
| test_summary_1260      | 890.6 µs | 839.4 µs | 29.8 µs | 1,123  | 1032   |

### BasanosEngine

| Benchmark                 | Mean       | Min        | Std Dev  | OPS   | Rounds |
| ------------------------- | ---------- | ---------- | -------- | ----- | ------ |
| test_ret_adj_252_5        | 437.3 µs   | 394.8 µs   | 24.0 µs  | 2,287 | 1736   |
| test_vola_252_5           | 255.8 µs   | 218.0 µs   | 11.7 µs  | 3,910 | 4022   |
| test_cor_252_5            | 1.240 ms   | 1.196 ms   | 20.4 µs  | 807   | 613    |
| test_cor_1260_5           | 5.351 ms   | 5.273 ms   | 35.8 µs  | 187   | 189    |
| test_cor_252_20           | 13.590 ms  | 13.274 ms  | 677.4 µs | 74    | 71     |
| test_cash_position_252_5  | 56.321 ms  | 51.665 ms  | 2.644 ms | 18    | 18     |
| test_cash_position_1260_5 | 221.709 ms | 219.001 ms | 2.715 ms | 5     | 5      |
| test_portfolio_252_5      | 54.340 ms  | 50.828 ms  | 1.614 ms | 18    | 20     |

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
