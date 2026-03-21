# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.20.4",
#     "basanos",
#     "numpy>=2.0.0",
#     "polars>=1.0.0",
#     "plotly>=6.0.0",
# ]
# [tool.uv.sources]
# basanos = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import polars as pl

    from basanos.math import BasanosConfig, BasanosEngine


@app.cell
def cell_01():
    """Render the notebook introduction."""
    mo.md(
        r"""
        # 🏁 Basanos — End-to-End Worked Example

        This notebook walks through the **complete Basanos workflow** from raw prices
        to a self-contained HTML report, using a realistic synthetic dataset that
        mimics eight diversified equity sectors.

        ## What this notebook covers

        | Step | Topic |
        |------|-------|
        | 1 | 📈 **Data preparation** — synthetic multi-sector equity prices and momentum signals |
        | 2 | ⚙️ **Config selection** — choosing `vola`, `corr`, `clip`, and `shrink` with guidance |
        | 3 | 🔬 **Engine instantiation** — running `BasanosEngine` |
        | 4 | 📄 **HTML report generation** — one-call self-contained report |
        | 5 | 💸 **Trading cost analysis** — Sharpe degradation across basis-point sweep |

        > **Public data only.** All data is generated deterministically from a fixed seed —
        > no external data sources, API keys, or network access required.
        """
    )


@app.cell
def cell_02():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_03():
    """Introduce the data preparation section."""
    mo.md(
        r"""
        ## 📈 Step 1 — Data Preparation

        We generate **eight synthetic equity-like price series** representing broad
        market sectors (Technology, Healthcare, Financials, Energy, Consumer,
        Industrials, Materials, Utilities).  The returns share a common market factor
        (mimicking S&P 500 beta exposure) plus idiosyncratic noise, producing a
        realistic cross-sectional correlation structure.

        A **momentum signal** $\mu \in [-1, 1]$ is derived from the difference between
        a fast (5-day) and slow (60-day) exponentially weighted moving average of
        log returns, rescaled through $\tanh$ to keep values bounded.

        The dataset spans **five years** of daily data (~1 260 rows) — enough history
        for robust EWMA warmup and meaningful out-of-sample statistics.
        """
    )


@app.cell
def cell_04():
    """Generate synthetic multi-sector equity prices and momentum signals."""
    _rng = np.random.default_rng(2024)

    _assets = [
        "Technology",
        "Healthcare",
        "Financials",
        "Energy",
        "Consumer",
        "Industrials",
        "Materials",
        "Utilities",
    ]
    _n_assets = len(_assets)
    _n_days = 1_260  # ~5 years of daily data

    _start = pl.date(2019, 1, 1)
    _end = _start + pl.duration(days=_n_days - 1)
    _dates = pl.date_range(_start, _end, interval="1d", eager=True)

    # Per-sector parameters
    _annual_drift = np.array([0.14, 0.10, 0.09, 0.06, 0.08, 0.09, 0.07, 0.05])
    _annual_vol = np.array([0.22, 0.18, 0.20, 0.28, 0.17, 0.19, 0.24, 0.14])
    _betas = np.array([1.3, 0.9, 1.1, 0.8, 0.95, 1.0, 1.1, 0.5])  # market sensitivity
    _s0 = np.array([100.0] * _n_assets)

    _daily_drift = _annual_drift / 252
    _daily_vol = _annual_vol / np.sqrt(252)

    # Market factor (shared component → cross-sectional correlation)
    _market_vol_daily = 0.012
    _market_factor = _rng.normal(0.0, _market_vol_daily, _n_days)

    # Idiosyncratic shocks (sector-specific)
    _idio_shocks = _rng.normal(0.0, 1.0, (_n_days, _n_assets))

    # Combine: r_i = drift_i + beta_i * market + idio_vol_i * epsilon_i
    _idio_vol = np.sqrt(np.maximum(_daily_vol**2 - (_betas * _market_vol_daily) ** 2, 1e-6))
    _log_returns = (
        _daily_drift[np.newaxis, :]
        + _betas[np.newaxis, :] * _market_factor[:, np.newaxis]
        + _idio_vol[np.newaxis, :] * _idio_shocks
    )

    # Build price DataFrame
    _price_paths = _s0[np.newaxis, :] * np.exp(np.cumsum(_log_returns, axis=0))
    prices = pl.DataFrame({"date": _dates, **{a: _price_paths[:, i].tolist() for i, a in enumerate(_assets)}})

    # Momentum signal: tanh of fast-minus-slow EWMA log return spread
    _alpha_fast = 2.0 / (5 + 1)
    _alpha_slow = 2.0 / (60 + 1)
    _mu_cols: dict[str, object] = {"date": _dates}
    for _i, _a in enumerate(_assets):
        _lr = _log_returns[:, _i]
        # Vectorised EWMA using cumulative trick (approximate)
        _ew_fast = np.zeros(_n_days)
        _ew_slow = np.zeros(_n_days)
        _ew_fast[0] = _lr[0]
        _ew_slow[0] = _lr[0]
        for _t in range(1, _n_days):
            _ew_fast[_t] = _alpha_fast * _lr[_t] + (1 - _alpha_fast) * _ew_fast[_t - 1]
            _ew_slow[_t] = _alpha_slow * _lr[_t] + (1 - _alpha_slow) * _ew_slow[_t - 1]
        _mu_cols[_a] = np.tanh(100.0 * (_ew_fast - _ew_slow))

    mu = pl.DataFrame(_mu_cols)
    return mu, prices


@app.cell
def cell_05(mu, prices):
    """Preview the generated dataset."""
    mo.vstack(
        [
            mo.md("### Price series (first 5 rows)"),
            mo.ui.table(prices.head(5)),
            mo.md(
                f"*{prices.height:,} daily rows × {prices.width - 1} assets.  "
                "Date column `date` is a Polars `Date` type.*"
            ),
            mo.md("### Momentum signals µ (first 5 rows)"),
            mo.ui.table(mu.head(5)),
            mo.md(
                r"*$\mu \in [-1, 1]$ — fast EWMA vs. slow EWMA spread through $\tanh$.  "
                "Positive values indicate upward momentum.*"
            ),
        ]
    )


@app.cell
def cell_06():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_07():
    """Introduce the config selection section."""
    mo.md(
        r"""
        ## ⚙️ Step 2 — Config Selection

        `BasanosConfig` has four key parameters.  The table below summarises the
        selection guidelines built into the library.

        | Parameter | Role | Guidance |
        |-----------|------|----------|
        | `vola` | EWMA lookback for per-asset volatility (days) | 10 – 30 for daily equity data |
        | `corr` | EWMA lookback for correlation matrix (days; must be ≥ `vola`) | 30 – 90; longer → smoother |
        | `clip` | Clipping threshold for vol-adjusted returns (std devs) | 3 – 5; tighter → reduces outlier impact |
        | `shrink` | Shrinkage intensity towards identity `[0, 1]` | See guidance below |

        ### Shrinkage guidance

        The correct `shrink` value depends on the ratio of assets (`n`) to correlation
        lookback (`T = corr`):

        | Regime | Assets `n` | Lookback `T` | Recommended `shrink` |
        |--------|-----------|-------------|----------------------|
        | Many assets, short history | > 20 | < 40 | **0.3 – 0.5** (strong regularisation) |
        | Balanced | ~ 10 | ~ 60 | **0.5 – 0.7** |
        | Few assets, long history | < 10 | > 100 | **0.7 – 0.9** (light regularisation) |

        For this dataset we have **8 assets** and will use `corr = 60` → the *balanced*
        regime, so `shrink = 0.6` is a reasonable starting point.

        Use the sliders to explore how each parameter affects the strategy.
        """
    )


@app.cell
def cell_08():
    """Create interactive sliders for BasanosConfig parameters."""
    vola_slider = mo.ui.slider(
        start=4,
        stop=64,
        value=16,
        step=1,
        label="vola (volatility lookback, days):",
        show_value=True,
    )
    corr_slider = mo.ui.slider(
        start=4,
        stop=128,
        value=60,
        step=1,
        label="corr (correlation lookback, days):",
        show_value=True,
    )
    clip_slider = mo.ui.slider(
        start=1.0,
        stop=6.0,
        value=4.0,
        step=0.5,
        label="clip (vol-adjusted return clipping threshold):",
        show_value=True,
    )
    shrink_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.6,
        step=0.05,
        label="shrink (shrinkage towards identity):",
        show_value=True,
    )
    mo.vstack([vola_slider, corr_slider, clip_slider, shrink_slider])
    return clip_slider, corr_slider, shrink_slider, vola_slider


@app.cell
def cell_09(clip_slider, corr_slider, shrink_slider, vola_slider):
    """Build BasanosConfig from slider values and display the active configuration."""
    _effective_corr = max(corr_slider.value, vola_slider.value)
    _warn = _effective_corr > corr_slider.value

    # Shrinkage regime note
    _n_assets = 8
    _corr_lookback = _effective_corr
    if _n_assets > 20 and _corr_lookback < 40:
        _regime = "many assets / short history → `shrink` 0.3 – 0.5 recommended"
    elif _n_assets < 10 and _corr_lookback > 100:
        _regime = "few assets / long history → `shrink` 0.7 – 0.9 recommended"
    else:
        _regime = "balanced regime → `shrink` 0.5 – 0.7 recommended"

    cfg = BasanosConfig(
        vola=vola_slider.value,
        corr=_effective_corr,
        clip=clip_slider.value,
        shrink=shrink_slider.value,
        aum=10_000_000,
    )
    mo.callout(
        mo.md(
            f"""
            **Active configuration** — {_regime}

            `BasanosConfig(vola={cfg.vola}, corr={cfg.corr}, clip={cfg.clip}, shrink={cfg.shrink}, aum={cfg.aum:,.0f})`

            {"⚠️ `corr` was raised to match `vola` (constraint: corr ≥ vola)." if _warn else ""}
            """
        ),
        kind="info",
    )
    return (cfg,)


@app.cell
def cell_10():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_11():
    """Introduce the engine instantiation section."""
    mo.md(
        r"""
        ## 🔬 Step 3 — Engine Instantiation

        `BasanosEngine` implements a vectorised three-step pipeline across all
        timestamps in a single constructor call:

        1. **Volatility adjustment** — log returns are normalised by an EWMA
           volatility estimate and clipped at `cfg.clip` standard deviations.

        2. **Correlation estimation** — an EWMA correlation matrix is computed
           and regularised via linear shrinkage toward the identity:

           $$C_\text{shrunk} = \lambda \cdot C_\text{EWMA} + (1 - \lambda) \cdot I,
           \quad \lambda = \texttt{cfg.shrink}$$

        3. **Position solving** — for each timestamp the system
           $C_\text{shrunk} \cdot x = \mu$ is solved and the risk position $x$ is
           normalised, then converted to a cash position by dividing by per-asset
           EWMA volatility.
        """
    )


@app.cell
def cell_12(cfg, mu, prices):
    """Instantiate BasanosEngine."""
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    portfolio = engine.portfolio
    return engine, portfolio


@app.cell
def cell_13(engine):
    """Show key engine outputs: position status, cash positions, and IC."""
    _status_counts = (
        engine.position_status.group_by("status").agg(pl.len().alias("count")).sort("count", descending=True)
    )
    mo.vstack(
        [
            mo.md("### Position status breakdown"),
            mo.ui.table(_status_counts),
            mo.md(
                "*`warmup` — EWMA not yet converged; `valid` — position produced normally; "
                "`zero_signal` — signal was zero; `degenerate` — matrix ill-conditioned.*"
            ),
            mo.md("### Cash positions (last 5 rows)"),
            mo.ui.table(engine.cash_position.tail(5)),
            mo.md(
                f"*{engine.cash_position.height:,} rows × {len(engine.assets)} assets.  "
                "Values are in currency units (AUM-scaled).*"
            ),
            mo.md(
                f"### Signal quality\n\n"
                f"- **IC mean**: `{engine.ic_mean:.4f}` | **IC std**: `{engine.ic_std:.4f}`\n"
                f"- **ICIR**: `{engine.icir:.4f}`\n"
                f"- **Rank IC mean**: `{engine.rank_ic_mean:.4f}` | **Rank IC std**: `{engine.rank_ic_std:.4f}`\n"
            ),
        ]
    )


@app.cell
def cell_14(portfolio):
    """Display key performance statistics."""
    mo.vstack(
        [
            mo.md("### Performance Statistics"),
            mo.ui.table(portfolio.stats.summary()),
            mo.md(
                "*Metrics are computed on the daily profit stream scaled by AUM.  "
                "Sharpe, volatility, and VaR are annualised (×√252).*"
            ),
            mo.md("### Turnover Summary"),
            mo.ui.table(portfolio.turnover_summary()),
        ]
    )


@app.cell
def cell_15(portfolio):
    """Render the performance snapshot (NAV + drawdown)."""
    mo.vstack(
        [
            mo.md(
                r"""
                ### Performance Dashboard

                The upper panel shows cumulative NAV alongside the **tilt** component
                (static average allocation) and the **timing** component (dynamic
                deviation from average).  The lower panel shows the drawdown from the
                running high-water mark.
                """
            ),
            mo.ui.plotly(portfolio.plots.snapshot()),
        ]
    )


@app.cell
def cell_16():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_17():
    """Introduce the HTML report section."""
    mo.md(
        r"""
        ## 📄 Step 4 — HTML Report Generation

        `portfolio.report` is a lazy facade that assembles a **self-contained HTML
        report** containing:

        - Metadata (date range, asset list, AUM)
        - All key statistics and turnover tables
        - Eight interactive Plotly charts (NAV, rolling Sharpe, volatility, annual
          Sharpe, monthly returns heatmap, correlation heatmap, lead/lag IR,
          and trading cost impact)

        The report embeds Plotly from CDN and requires no additional dependencies
        to open in any modern browser.

        ```python
        # Save to disk
        saved_path = portfolio.report.save("basanos_report.html")

        # Or get the HTML string (e.g. to serve via an API)
        html_str = portfolio.report.to_html(title="My Strategy Report")
        ```

        The cell below saves the report to a temporary path and confirms success.
        """
    )


@app.cell
def cell_18(portfolio):
    """Generate and save the HTML report."""
    import tempfile
    from pathlib import Path

    _tmp = Path(tempfile.mkdtemp()) / "basanos_report.html"
    _saved = portfolio.report.save(_tmp)
    _size_kb = _saved.stat().st_size / 1_024

    mo.callout(
        mo.md(
            f"""
            ✅ **Report saved successfully**

            - **Path**: `{_saved}`
            - **File size**: `{_size_kb:,.1f} KB`
            - **Charts**: NAV + drawdown, rolling Sharpe, rolling volatility, annual Sharpe,
              monthly returns heatmap, correlation heatmap, lead/lag IR, trading cost impact

            To save to a custom location:
            ```python
            portfolio.report.save("my_report.html")
            ```
            """
        ),
        kind="success",
    )


@app.cell
def cell_19():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_20():
    """Introduce the trading cost analysis section."""
    mo.md(
        r"""
        ## 💸 Step 5 — Trading Cost Analysis

        High turnover strategies can erode edge when real-world trading costs are
        applied.  Basanos estimates cost impact via a **linear cost model**:

        $$\text{daily cost} = \text{turnover}_t \times \frac{\text{cost\_bps}}{10\,000}$$

        where $\text{turnover}_t$ is the fraction of AUM traded on day $t$.

        `portfolio.trading_cost_impact(max_bps)` sweeps one-way costs from 0 to
        `max_bps` basis points and returns the net Sharpe at each level — making it
        easy to see at what cost level the strategy's edge is fully consumed.

        Use the slider to set the maximum cost level to evaluate.
        """
    )


@app.cell
def cell_21():
    """Create slider for maximum trading cost."""
    max_bps_slider = mo.ui.slider(
        start=5,
        stop=50,
        value=30,
        step=5,
        label="Maximum one-way trading cost to evaluate (bps):",
        show_value=True,
    )
    max_bps_slider
    return (max_bps_slider,)


@app.cell
def cell_22(max_bps_slider, portfolio):
    """Run and display the trading cost impact analysis."""
    _impact = portfolio.trading_cost_impact(max_bps=max_bps_slider.value)

    # Find the breakeven cost (first level where Sharpe ≤ 0)
    _breakeven = _impact.filter(pl.col("sharpe") <= 0).select("cost_bps").head(1)
    _be_note = (
        f"The strategy's edge is **fully consumed at ~{_breakeven[0, 0]} bps** one-way trading cost."
        if _breakeven.height > 0
        else "The strategy retains positive Sharpe across the entire cost range evaluated."
    )

    mo.vstack(
        [
            mo.md("### Trading Cost Impact Table"),
            mo.ui.table(_impact),
            mo.md(f"*{_be_note}  One-way cost in basis points per unit of AUM traded.*"),
        ]
    )


@app.cell
def cell_23(portfolio):
    """Render the trading cost impact chart."""
    mo.vstack(
        [
            mo.md(
                r"""
                ### Sharpe vs. Trading Cost

                Each bar shows the annualised Sharpe ratio after deducting one-way
                trading costs at the given basis-point level.  The height of the
                zero-cost bar (0 bps) is the gross strategy Sharpe.
                """
            ),
            mo.ui.plotly(portfolio.plots.trading_cost_impact_plot()),
        ]
    )


@app.cell
def cell_24():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_25():
    """Render the conclusion."""
    mo.md(
        r"""
        ## 🎉 Summary

        This notebook demonstrated the complete **Basanos** workflow end-to-end:

        ✅ **Data preparation** — realistic synthetic sector prices with shared market
        factor and momentum signals

        ✅ **Config selection** — `BasanosConfig` with shrinkage guidance table and
        reactive parameter sliders

        ✅ **Engine instantiation** — `BasanosEngine` computes correlation-adjusted
        cash positions and IC/ICIR signal quality metrics

        ✅ **HTML report generation** — one-call `portfolio.report.save()` producing
        a self-contained browser report with eight interactive charts

        ✅ **Trading cost analysis** — `portfolio.trading_cost_impact()` sweep showing
        Sharpe degradation and the breakeven cost level

        ---

        ### Next steps

        - Replace the synthetic data with real price data from a public source
          (e.g. `yfinance`) and your own forecasting signal
        - Switch to `SlidingWindowConfig` for the factor-model covariance mode
          (see `book/marimo/notebooks/ewm_benchmark.py`)
        - Tune `shrink` empirically using the interactive sweep in
          `book/marimo/notebooks/shrinkage_guide.py`
        - Use `portfolio.truncate()` to analyse sub-periods in isolation
        - See the [basanos repository](https://github.com/Jebel-Quant/basanos) for
          the full API reference
        """
    )


if __name__ == "__main__":
    app.run()
