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
    """Render the basanos demo introduction."""
    mo.md(
        r"""
        # 📊 Basanos — Portfolio Optimization & Analytics

        **Basanos** computes **correlation-adjusted risk positions** from price data
        and expected-return signals. It estimates time-varying EWMA correlations,
        applies shrinkage towards the identity matrix, and solves a normalized
        linear system per timestamp to produce stable, scale-invariant positions —
        implementing a *first hurdle* for expected returns.

        ## What this notebook covers

        1. 📈 **Synthetic market data** — four equity-like price series and momentum signals
        2. ⚙️ **Interactive configuration** — reactive sliders for `BasanosConfig` parameters
        3. 🔬 **Optimization** — `BasanosEngine` computes correlation-adjusted cash positions
        4. 📉 **Portfolio analytics** — NAV, drawdown, tilt/timing decomposition, statistics
        5. 🎨 **Visualisations** — performance dashboard, lead/lag IR, and correlation heatmap
        """
    )


@app.cell
def cell_02():
    """Render horizontal rule separator."""
    mo.md(r"""---""")


@app.cell
def cell_03():
    """Introduce the synthetic data section."""
    mo.md(
        r"""
        ## 📈 Synthetic Market Data

        We generate four synthetic equity-like price series — **AAPL**, **GOOGL**,
        **MSFT**, and **AMZN** — over three years of daily data using geometric
        Brownian motion. A fixed random seed (`42`) ensures reproducibility.

        The expected-return signals $\mu \in [-1, 1]$ are constructed from a simple
        price-momentum rule: the difference between a 5-day and 20-day moving average,
        passed through $\tanh$ to keep values bounded.
        """
    )


@app.cell
def cell_04():
    """Generate synthetic prices and expected-return signals with a fixed seed."""
    _rng = np.random.default_rng(42)
    _assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    _n = 750  # roughly 3 years of daily data
    _start = pl.date(2021, 1, 1)
    _end = _start + pl.duration(days=_n - 1)
    _dates = pl.date_range(_start, _end, interval="1d", eager=True)

    # Geometric Brownian motion: different drift / vol per asset
    _drift = [0.0003, 0.0002, 0.0004, 0.0001]
    _sigma = [0.018, 0.022, 0.020, 0.025]
    _s0 = [150.0, 2800.0, 300.0, 3300.0]

    _prices_cols: dict[str, object] = {"date": _dates}
    for _i, _a in enumerate(_assets):
        _log_ret = _rng.normal(_drift[_i], _sigma[_i], _n)
        _prices_cols[_a] = _s0[_i] * np.exp(np.cumsum(_log_ret))

    prices = pl.DataFrame(_prices_cols)

    # Momentum signal: tanh(50 * (MA5 - MA20) / price)
    _mu_cols: dict[str, object] = {"date": _dates}
    for _a in _assets:
        _p = prices[_a].to_numpy()
        _ma5 = np.convolve(_p, np.ones(5) / 5, mode="same") / _p - 1.0
        _ma20 = np.convolve(_p, np.ones(20) / 20, mode="same") / _p - 1.0
        _mu_cols[_a] = np.tanh(50.0 * (_ma5 - _ma20))

    mu = pl.DataFrame(_mu_cols)
    return mu, prices


@app.cell
def cell_05(mu, prices):
    """Show a preview of the prices and signals DataFrames."""
    mo.vstack(
        [
            mo.md("### Price data (first 5 rows)"),
            mo.ui.table(prices.head(5)),
            mo.md("### Expected-return signals µ (first 5 rows)"),
            mo.ui.table(mu.head(5)),
        ]
    )


@app.cell
def cell_06():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_07():
    """Introduce the configuration section."""
    mo.md(
        r"""
        ## ⚙️ Configuration

        Adjust the `BasanosConfig` parameters using the sliders below.
        The optimizer and all downstream analytics **update reactively** whenever
        you move a slider.

        | Parameter | Constraint | Description |
        |-----------|------------|-------------|
        | `vola` | `> 0` | EWMA lookback for volatility estimation (days) |
        | `corr` | `≥ vola` | EWMA lookback for correlation estimation (days) |
        | `clip` | `> 0` | Clipping threshold for vol-adjusted returns |
        | `shrink` | `[0, 1]` | Shrinkage intensity towards identity — `0` = none, `1` = full |
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
        value=32,
        step=1,
        label="corr (correlation lookback, days):",
        show_value=True,
    )
    clip_slider = mo.ui.slider(
        start=1.0,
        stop=6.0,
        value=3.5,
        step=0.5,
        label="clip (vol-adjusted return clipping threshold):",
        show_value=True,
    )
    shrink_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.5,
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
    cfg = BasanosConfig(
        vola=vola_slider.value,
        corr=_effective_corr,
        clip=clip_slider.value,
        shrink=shrink_slider.value,
        aum=1_000_000,
    )
    mo.callout(
        mo.md(
            f"""
            **Active configuration**

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
    """Introduce the optimization section."""
    mo.md(
        r"""
        ## 🔬 Optimization

        `BasanosEngine` implements a three-step pipeline per timestamp:

        1. **Volatility adjustment** — log returns are normalized by an EWMA
           volatility estimate and clipped at `cfg.clip` standard deviations.

        2. **Correlation estimation** — an EWMA correlation matrix is computed
           and shrunk towards the identity matrix:

           $$C_\text{shrunk} = (1 - \text{shrink}) \cdot C_\text{EWMA} + \text{shrink} \cdot I$$

        3. **Position solving** — for each timestamp, solve $C_\text{shrunk} \cdot x = \mu$
           and normalize by the inverse-matrix norm of $\mu$. Positions are then scaled
           by EWMA volatility and a running profit-variance estimate.
        """
    )


@app.cell
def cell_12(cfg, mu, prices):
    """Run BasanosEngine and extract cash positions and portfolio."""
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    portfolio = engine.portfolio
    return engine, portfolio


@app.cell
def cell_13(engine):
    """Display the computed cash positions (last 5 rows)."""
    mo.vstack(
        [
            mo.md("### Cash positions (last 5 rows)"),
            mo.ui.table(engine.cash_position.tail(5)),
            mo.md(
                f"*{engine.cash_position.height} rows × {len(engine.assets)} assets. "
                "Values are in currency units (AUM-scaled).*"
            ),
        ]
    )


@app.cell
def cell_14():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_15():
    """Introduce the portfolio analytics section."""
    mo.md(
        r"""
        ## 📉 Portfolio Analytics

        The `Portfolio` object exposes a rich set of computed properties for
        evaluating strategy performance. All values update reactively as you
        adjust the configuration sliders above.
        """
    )


@app.cell
def cell_16(portfolio):
    """Display the full statistics summary table."""
    mo.vstack(
        [
            mo.md("### Statistics Summary"),
            mo.ui.table(portfolio.stats.summary()),
            mo.md(
                "*Metrics are computed on the daily profit stream scaled by AUM. "
                "Sharpe, volatility, and VaR are annualised.*"
            ),
        ]
    )


@app.cell
def cell_17(portfolio):
    """Display the tilt / timing decomposition (last 6 rows)."""
    mo.vstack(
        [
            mo.md(
                r"""
                ### Tilt / Timing Decomposition (last 6 rows)

                - **Portfolio** — total accumulated NAV
                - **Tilt** — static average allocation (constant weights over time)
                - **Timing** — dynamic deviation from the average weight
                """
            ),
            mo.ui.table(portfolio.tilt_timing_decomp.tail(6)),
        ]
    )


@app.cell
def cell_18(portfolio):
    """Display the monthly returns table."""
    mo.vstack(
        [
            mo.md("### Monthly Returns"),
            mo.ui.table(portfolio.monthly.select(["year", "month_name", "returns", "profit", "NAV_accumulated"])),
        ]
    )


@app.cell
def cell_19():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_20():
    """Introduce the visualisations section."""
    mo.md(
        r"""
        ## 🎨 Visualisations

        All charts are interactive Plotly figures — hover for tooltips, click
        legend items to toggle traces, and use the range-selector buttons for
        quick zoom presets.
        """
    )


@app.cell
def cell_21(portfolio):
    """Render the performance snapshot dashboard (NAV + drawdown)."""
    mo.vstack(
        [
            mo.md(
                r"""
                ### Performance Dashboard

                Three overlaid NAV curves show the total portfolio alongside its
                **tilt** (static average) and **timing** (dynamic deviation) components.
                The lower panel shows the drawdown as a percentage of the high-water mark.
                """
            ),
            mo.ui.plotly(portfolio.plots.snapshot()),
        ]
    )


@app.cell
def cell_22(portfolio):
    """Render the lead / lag information-ratio chart."""
    mo.vstack(
        [
            mo.md(
                r"""
                ### Lead / Lag Information Ratio

                Each bar shows the annualised Sharpe ratio when positions are shifted
                by a given number of days. A genuine signal should peak at **lag 0** and
                decay positively; negative lags test for look-ahead bias (should be flat).
                """
            ),
            mo.ui.plotly(portfolio.plots.lead_lag_ir_plot(start=-5, end=10)),
        ]
    )


@app.cell
def cell_23(portfolio):
    """Render the asset + portfolio correlation heatmap."""
    mo.vstack(
        [
            mo.md(
                r"""
                ### Correlation Heatmap

                Pairwise Pearson correlation of asset log-returns, with the portfolio
                profit series appended as an additional column. Helps identify
                concentration risk and diversification properties.
                """
            ),
            mo.ui.plotly(portfolio.plots.correlation_heatmap()),
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
        ## 🎉 Conclusion

        This notebook walked through the full **basanos** workflow:

        ✅ **Synthetic data** — GBM price series and momentum-based signals
        ✅ **Interactive configuration** — reactive `BasanosConfig` sliders
        ✅ **Optimization** — `BasanosEngine` generates correlation-adjusted positions
        ✅ **Analytics** — statistics summary, tilt/timing decomposition, monthly returns
        ✅ **Visualisations** — performance dashboard, lead/lag IR, and correlation heatmap

        ### Next steps

        - Replace the synthetic data with real price data and a forecasting signal
        - Tune `vola`, `corr`, `clip`, and `shrink` for your specific market
        - Use `Portfolio.truncate()` to analyse sub-periods in isolation
        - Explore `Portfolio.smoothed_holding()` to reduce portfolio turnover
        - See the [basanos repository](https://github.com/Jebel-Quant/basanos) for the full API reference
        """
    )


if __name__ == "__main__":
    app.run()
