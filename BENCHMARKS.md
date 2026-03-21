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
| Date               | 2026-03-21                             |
| Python             | CPython 3.14.2                         |
| OS                 | macOS Darwin 24.6.0 (local run)        |
| CPU                | Apple M4 Pro                           |
| CPU cores          | 14 physical / 14 logical               |
| pytest-benchmark   | 5.2.3                                  |
| basanos version    | 0.5.0                                  |
| Commit             | `75ebd14`                              |

> **Note**: These results were captured on a local Apple M4 Pro.  Absolute
> timings differ from prior CI baselines (Azure AMD EPYC) but the relative
> ordering and scaling behaviour are consistent.  CI regression detection
> compares each run against the stored `baseline.json` on the same runner.

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
| test_profits_252_5            | 995.3 µs  | 808.6 µs  | 121.6 µs | 1,005  | 90     |
| test_profits_1260_5           | 989.3 µs  | 784.2 µs  | 146.7 µs | 1,011  | 1020   |
| test_profits_252_20           | 3.269 ms  | 2.825 ms  | 280.0 µs | 306    | 319    |
| test_nav_accumulated_252_5    | 1.234 ms  | 951.4 µs  | 146.3 µs | 811    | 236    |
| test_nav_compounded_252_5     | 1.217 ms  | 954.0 µs  | 159.8 µs | 821    | 816    |
| test_drawdown_252_5           | 1.418 ms  | 1.078 ms  | 176.3 µs | 705    | 388    |
| test_drawdown_1260_5          | 1.424 ms  | 1.113 ms  | 220.2 µs | 702    | 735    |
| test_monthly_252_5            | 1.729 ms  | 1.326 ms  | 207.1 µs | 578    | 127    |
| test_tilt_timing_decomp_252_5 | 6.156 ms  | 5.034 ms  | 620.4 µs | 162    | 86     |
| test_all_252_5                | 3.056 ms  | 2.542 ms  | 278.0 µs | 327    | 346    |

### Stats

| Benchmark              | Mean     | Min      | Std Dev | OPS    | Rounds |
| ---------------------- | -------- | -------- | ------- | ------ | ------ |
| test_volatility_252    | 45.2 µs  | 33.6 µs  | 6.8 µs  | 22,114 | 13080  |
| test_sharpe_252        | 46.0 µs  | 30.9 µs  | 9.0 µs  | 21,756 | 15415  |
| test_value_at_risk_252 | 19.9 µs  | 17.6 µs  | 2.8 µs  | 50,163 | 993    |
| test_summary_252       | 1.583 ms | 1.304 ms | 197.5 µs| 632    | 85     |
| test_summary_1260      | 1.619 ms | 1.299 ms | 137.4 µs| 618    | 511    |

### BasanosEngine (ewma_shrink)

| Benchmark                 | Mean      | Min       | Std Dev  | OPS   | Rounds |
| ------------------------- | --------- | --------- | -------- | ----- | ------ |
| test_ret_adj_252_5        | 247.8 µs  | 151.0 µs  | 135.9 µs | 4,035 | 598    |
| test_vola_252_5           | 127.1 µs  | 57.8 µs   | 44.3 µs  | 7,867 | 4596   |
| test_cor_252_5            | 626.3 µs  | 479.9 µs  | 91.8 µs  | 1,597 | 362    |
| test_cor_1260_5           | 1.750 ms  | 1.402 ms  | 241.3 µs | 571   | 575    |
| test_cor_252_20           | 4.232 ms  | 3.638 ms  | 462.0 µs | 236   | 242    |
| test_cash_position_252_5  | 18.961 ms | 18.082 ms | 390.8 µs | 53    | 47     |
| test_cash_position_1260_5 | 89.985 ms | 88.583 ms | 631.0 µs | 11    | 12     |
| test_portfolio_252_5      | 19.131 ms | 18.373 ms | 318.7 µs | 52    | 50     |

### BasanosEngine (sliding_window)

Complexity: O(T·W·N·k) for rolling SVDs, O(T·(k³ + kN)) for Woodbury solves.
Memory: O(W·N) per step, independent of T.

| Benchmark                         | T    | N  | W  | k | Mean      | Min       | Std Dev  | OPS | Rounds |
| --------------------------------- | ---- | -- | -- | - | --------- | --------- | -------- | --- | ------ |
| test_cash_position_sw_252_5_60_3  | 252  | 5  | 60 | 3 | 16.545 ms | 15.990 ms | 276.4 µs | 60  | 58     |
| test_cash_position_sw_252_20_60_5 | 252  | 20 | 60 | 5 | 29.655 ms | 28.820 ms | 543.3 µs | 34  | 35     |
| test_cash_position_sw_1260_5_60_3 | 1260 | 5  | 60 | 3 | 93.002 ms | 91.818 ms | 816.9 µs | 11  | 11     |

### BasanosStream.from_warmup()

One-time initialisation cost for the incremental streaming API.  Runs the
O(T·N²) EWM correlation batch pass and extracts the IIR filter state (PR #349
eliminated the prior redundant lfilter sweeps, halving this cost for large T).

| Benchmark                      | T    | N  | W  | k | Mean      | Min       | Std Dev    | OPS | Rounds |
| ------------------------------ | ---- | -- | -- | - | --------- | --------- | ---------- | --- | ------ |
| test_from_warmup_252_5         | 252  | 5  | —  | — | 17.001 ms | 16.352 ms | 285.9 µs   | 59  | 54     |
| test_from_warmup_1260_5        | 1260 | 5  | —  | — | 87.876 ms | 86.769 ms | 758.9 µs   | 11  | 12     |
| test_from_warmup_252_20        | 252  | 20 | —  | — | 27.633 ms | 26.821 ms | 458.1 µs   | 36  | 35     |
| test_from_warmup_sw_252_5_60_3 | 252  | 5  | 60 | 3 | 17.599 ms | 17.080 ms | 257.5 µs   | 57  | 54     |
| test_from_warmup_sw_1260_5_60_3| 1260 | 5  | 60 | 3 | 97.020 ms | 93.433 ms | 6.405 ms   | 10  | 11     |

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
