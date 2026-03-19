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
| Date               | 2026-03-19                             |
| Python             | CPython 3.12 (`.python-version`)       |
| Benchmark Python   | CPython 3.12.3 (GitHub Actions runner) |
| OS                 | Linux (Ubuntu, Azure runner)           |
| CPU                | AMD EPYC 7763 64-Core Processor        |
| CPU speed          | 3.25 GHz                               |
| CPU cores          | 4 (virtual, Azure Standard DS2 v2)     |
| pytest-benchmark   | 5.2.3                                  |
| basanos version    | 0.4.2                                  |
| Commit             | `9aa4491`                              |

Dataset sizes used in the benchmarks:

| Suffix        | Rows  | Assets | Window | Factors | Description                      |
|---------------|-------|--------|--------|---------|----------------------------------|
| 252_5         | 252   | 5      | —      | —       | ~1 year, 5 assets (ewma_shrink)  |
| 1260_5        | 1260  | 5      | —      | —       | ~5 years, 5 assets (ewma_shrink) |
| 252_20        | 252   | 20     | —      | —       | ~1 year, 20 assets (ewma_shrink) |
| sw_252_5_60_3 | 252   | 5      | 60     | 3       | ~1 year, 5 assets (sliding_window)  |
| sw_252_20_60_5| 252   | 20     | 60     | 5       | ~1 year, 20 assets (sliding_window) |
| sw_1260_5_60_3| 1260  | 5      | 60     | 3       | ~5 years, 5 assets (sliding_window) |

---

## Results

Times are **mean wall-clock** over all benchmark rounds (min-of-runs strategy
used by pytest-benchmark to reduce scheduling noise).  The raw JSON is stored
in [`benchmarks/results/baseline.json`](benchmarks/results/baseline.json).

### Portfolio

| Benchmark                     | Mean      | Min       | Std Dev  | OPS   | Rounds |
| ----------------------------- | --------- | --------- | -------- | ----- | ------ |
| test_profits_252_5            | 3.316 ms  | 3.137 ms  | 267.3 µs | 302   | 225    |
| test_profits_1260_5           | 3.293 ms  | 3.157 ms  | 85.7 µs  | 304   | 288    |
| test_profits_252_20           | 12.063 ms | 11.693 ms | 213.1 µs | 83    | 80     |
| test_nav_accumulated_252_5    | 3.629 ms  | 3.477 ms  | 81.2 µs  | 276   | 246    |
| test_nav_compounded_252_5     | 3.727 ms  | 3.602 ms  | 70.4 µs  | 268   | 255    |
| test_drawdown_252_5           | 3.832 ms  | 3.704 ms  | 71.2 µs  | 261   | 253    |
| test_drawdown_1260_5          | 3.903 ms  | 3.763 ms  | 71.0 µs  | 256   | 244    |
| test_monthly_252_5            | 4.369 ms  | 4.268 ms  | 65.1 µs  | 229   | 213    |
| test_tilt_timing_decomp_252_5 | 15.864 ms | 15.531 ms | 217.1 µs | 63    | 63     |
| test_all_252_5                | 7.922 ms  | 7.726 ms  | 97.2 µs  | 126   | 125    |

### Stats

| Benchmark              | Mean     | Min      | Std Dev | OPS    | Rounds |
| ---------------------- | -------- | -------- | ------- | ------ | ------ |
| test_volatility_252    | 202.5 µs | 182.1 µs | 11.9 µs | 4,938  | 2874   |
| test_sharpe_252        | 199.8 µs | 181.6 µs | 12.4 µs | 5,005  | 3976   |
| test_value_at_risk_252 | 76.0 µs  | 70.9 µs  | 7.2 µs  | 13,156 | 3579   |
| test_summary_252       | 4.197 ms | 4.014 ms | 94.7 µs | 238    | 193    |
| test_summary_1260      | 4.391 ms | 4.216 ms | 97.2 µs | 228    | 224    |

### BasanosEngine (ewma_shrink)

| Benchmark                 | Mean       | Min        | Std Dev  | OPS   | Rounds |
| ------------------------- | ---------- | ---------- | -------- | ----- | ------ |
| test_ret_adj_252_5        | 445.9 µs   | 404.4 µs   | 25.4 µs  | 2,242 | 1805   |
| test_vola_252_5           | 257.5 µs   | 218.1 µs   | 12.0 µs  | 3,884 | 4151   |
| test_cor_252_5            | 1.242 ms   | 1.193 ms   | 26.7 µs  | 805   | 460    |
| test_cor_1260_5           | 5.731 ms   | 5.631 ms   | 52.5 µs  | 175   | 176    |
| test_cor_252_20           | 14.290 ms  | 13.987 ms  | 547.3 µs | 70    | 68     |
| test_cash_position_252_5  | 69.242 ms  | 68.734 ms  | 421.1 µs | 14    | 14     |
| test_cash_position_1260_5 | 351.091 ms | 349.983 ms | 1.583 ms | 3     | 5      |
| test_portfolio_252_5      | 70.508 ms  | 68.892 ms  | 1.261 ms | 14    | 15     |

### BasanosEngine (sliding_window)

Complexity: O(T·W·N·k) for rolling SVDs, O(T·(k³ + kN)) for Woodbury solves.
Memory: O(W·N) per step, independent of T.

| Benchmark                         | T    | N  | W  | k | Mean       | Min        | Std Dev  | OPS   | Rounds |
| --------------------------------- | ---- | -- | -- | - | ---------- | ---------- | -------- | ----- | ------ |
| test_cash_position_sw_252_5_60_3  | 252  | 5  | 60 | 3 | 57.706 ms  | 57.323 ms  | 246.2 µs | 17    | 18     |
| test_cash_position_sw_252_20_60_5 | 252  | 20 | 60 | 5 | 76.367 ms  | 75.868 ms  | 279.8 µs | 13    | 14     |
| test_cash_position_sw_1260_5_60_3 | 1260 | 5  | 60 | 3 | 340.503 ms | 339.861 ms | 471.5 µs | 3     | 5      |

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
