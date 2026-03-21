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

| Property           | Value                                              |
|--------------------|----------------------------------------------------|
| Date               | 2026-03-21                                         |
| Python             | CPython 3.12.3                                     |
| OS                 | Linux 6.14.0-1017-azure (GitHub Actions runner)    |
| CPU                | AMD EPYC 7763 64-Core Processor                    |
| CPU cores          | 4 vCPUs (GitHub Actions standard Ubuntu runner)    |
| pytest-benchmark   | 5.2.3                                              |
| basanos version    | 0.5.0                                              |
| Commit             | `06c0877`                                          |

> **Note**: These results were captured on a standard GitHub Actions Ubuntu
> runner (AMD EPYC 7763, 4 vCPUs) — the same hardware class used by CI
> regression detection.  Prior baselines captured on a local Apple M4 Pro have
> been retired; all future baselines should be re-captured on CI runners so
> regression thresholds remain meaningful.

Dataset sizes used in the benchmarks:

| Suffix              | Rows  | Assets | Window | Factors | Description                          |
|---------------------|-------|--------|--------|---------|--------------------------------------|
| 252_5               | 252   | 5      | —      | —       | ~1 year, 5 assets (ewma_shrink)      |
| 1260_5              | 1260  | 5      | —      | —       | ~5 years, 5 assets (ewma_shrink)     |
| 252_20              | 252   | 20     | —      | —       | ~1 year, 20 assets (ewma_shrink)     |
| sw_252_5_60_3       | 252   | 5      | 60     | 3       | ~1 year, 5 assets (sliding_window)   |
| sw_252_20_60_5      | 252   | 20     | 60     | 5       | ~1 year, 20 assets (sliding_window)  |
| sw_1260_5_60_3      | 1260  | 5      | 60     | 3       | ~5 years, 5 assets (sliding_window)  |
| fw_252_5            | 252   | 5      | —      | —       | from_warmup EWM, ~1 year, 5 assets   |
| fw_1260_5           | 1260  | 5      | —      | —       | from_warmup EWM, ~5 years, 5 assets  |
| fw_252_20           | 252   | 20     | —      | —       | from_warmup EWM, ~1 year, 20 assets  |
| fw_sw_252_5_60_3    | 252   | 5      | 60     | 3       | from_warmup SW, ~1 year, 5 assets    |
| fw_sw_1260_5_60_3   | 1260  | 5      | 60     | 3       | from_warmup SW, ~5 years, 5 assets   |

---

## Results

Times are **mean wall-clock** over all benchmark rounds (min-of-runs strategy
used by pytest-benchmark to reduce scheduling noise).  The raw JSON is stored
in [`benchmarks/results/baseline.json`](benchmarks/results/baseline.json).

### Portfolio

| Benchmark                     | Mean      | Min       | Std Dev  | OPS    | Rounds |
| ----------------------------- | --------- | --------- | -------- | ------ | ------ |
| test_profits_252_5            | 3.232 ms  | 3.075 ms  | 106.0 µs | 309    | 232    |
| test_profits_1260_5           | 3.245 ms  | 3.115 ms  | 79.5 µs  | 308    | 268    |
| test_profits_252_20           | 11.921 ms | 11.550 ms | 520.8 µs | 84     | 81     |
| test_nav_accumulated_252_5    | 3.563 ms  | 3.421 ms  | 77.8 µs  | 281    | 263    |
| test_nav_compounded_252_5     | 3.663 ms  | 3.499 ms  | 87.4 µs  | 273    | 270    |
| test_drawdown_252_5           | 3.749 ms  | 3.627 ms  | 54.1 µs  | 267    | 263    |
| test_drawdown_1260_5          | 3.819 ms  | 3.702 ms  | 87.9 µs  | 262    | 258    |
| test_monthly_252_5            | 4.320 ms  | 4.175 ms  | 90.1 µs  | 231    | 217    |
| test_tilt_timing_decomp_252_5 | 15.640 ms | 15.228 ms | 260.8 µs | 64     | 63     |
| test_all_252_5                | 7.808 ms  | 7.640 ms  | 102.9 µs | 128    | 125    |

### Stats

| Benchmark              | Mean     | Min      | Std Dev | OPS    | Rounds |
| ---------------------- | -------- | -------- | ------- | ------ | ------ |
| test_volatility_252    | 196.3 µs | 180.0 µs | 11.9 µs | 5,095  | 4348   |
| test_sharpe_252        | 202.0 µs | 184.0 µs | 12.7 µs | 4,949  | 3922   |
| test_value_at_risk_252 | 73.6 µs  | 68.1 µs  | 7.0 µs  | 13,593 | 3892   |
| test_summary_252       | 4.131 ms | 3.989 ms | 90.1 µs | 242    | 197    |
| test_summary_1260      | 4.345 ms | 4.159 ms | 115.9 µs| 230    | 208    |

### BasanosEngine (ewma_shrink)

| Benchmark                 | Mean       | Min        | Std Dev  | OPS   | Rounds |
| ------------------------- | ---------- | ---------- | -------- | ----- | ------ |
| test_ret_adj_252_5        | 432.4 µs   | 400.7 µs   | 19.0 µs  | 2,312 | 1719   |
| test_vola_252_5           | 251.8 µs   | 213.6 µs   | 13.7 µs  | 3,971 | 3882   |
| test_cor_252_5            | 1.256 ms   | 1.195 ms   | 86.8 µs  | 796   | 472    |
| test_cor_1260_5           | 5.678 ms   | 5.522 ms   | 83.6 µs  | 176   | 177    |
| test_cor_252_20           | 14.535 ms  | 13.963 ms  | 232.9 µs | 69    | 67     |
| test_cash_position_252_5  | 69.886 ms  | 69.322 ms  | 410.3 µs | 14    | 14     |
| test_cash_position_1260_5 | 351.111 ms | 349.679 ms | 1.492 ms | 3     | 5      |
| test_portfolio_252_5      | 70.143 ms  | 69.495 ms  | 372.5 µs | 14    | 15     |

### BasanosEngine (sliding_window)

Complexity: O(T·W·N·k) for rolling SVDs, O(T·(k³ + kN)) for Woodbury solves.
Memory: O(W·N) per step, independent of T.

| Benchmark                         | T    | N  | W  | k | Mean       | Min        | Std Dev  | OPS | Rounds |
| --------------------------------- | ---- | -- | -- | - | ---------- | ---------- | -------- | --- | ------ |
| test_cash_position_sw_252_5_60_3  | 252  | 5  | 60 | 3 | 61.656 ms  | 60.809 ms  | 369.0 µs | 16  | 16     |
| test_cash_position_sw_252_20_60_5 | 252  | 20 | 60 | 5 | 80.364 ms  | 79.773 ms  | 248.3 µs | 12  | 12     |
| test_cash_position_sw_1260_5_60_3 | 1260 | 5  | 60 | 3 | 364.392 ms | 362.199 ms | 1.507 ms | 3   | 5      |

### BasanosStream.from_warmup()

One-time initialisation cost for the incremental streaming API.  Runs the
O(T·N²) EWM correlation batch pass and extracts the IIR filter state (PR #349
eliminated the prior redundant lfilter sweeps, halving this cost for large T).

| Benchmark                      | T    | N  | W  | k | Mean       | Min        | Std Dev    | OPS | Rounds |
| ------------------------------ | ---- | -- | -- | - | ---------- | ---------- | ---------- | --- | ------ |
| test_from_warmup_252_5         | 252  | 5  | —  | — | 62.781 ms  | 61.419 ms  | 2.402 ms   | 16  | 17     |
| test_from_warmup_1260_5        | 1260 | 5  | —  | — | 338.479 ms | 337.557 ms | 1.307 ms   | 3   | 5      |
| test_from_warmup_252_20        | 252  | 20 | —  | — | 93.538 ms  | 92.492 ms  | 494.3 µs   | 11  | 11     |
| test_from_warmup_sw_252_5_60_3 | 252  | 5  | 60 | 3 | 64.614 ms  | 63.811 ms  | 305.4 µs   | 15  | 16     |
| test_from_warmup_sw_1260_5_60_3| 1260 | 5  | 60 | 3 | 368.756 ms | 366.942 ms | 1.733 ms   | 3   | 5      |

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
