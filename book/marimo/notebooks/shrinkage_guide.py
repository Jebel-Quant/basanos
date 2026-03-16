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
    import plotly.graph_objects as go
    import polars as pl

    from basanos.math import BasanosConfig, BasanosEngine
    from basanos.math._signal import shrink2id


@app.cell
def cell_01():
    """Render the shrinkage guide introduction."""
    mo.md(
        r"""
        # 🔬 Basanos Shrinkage Guide

        This notebook provides **theoretical background** and **empirical guidance**
        on how shrinkage toward the identity matrix affects the Basanos optimizer.

        ## What this notebook covers

        1. 📐 **Theory** — why shrinkage helps and what it does to eigenvalues
        2. 🎚️ **Interactive sweep** — Sharpe ratio vs. shrinkage intensity (λ)
        3. 📊 **Condition number** — how λ affects matrix conditioning
        4. 🗺️ **Parameter map** — joint effect of portfolio size × lookback × λ
        5. 💡 **Practical recommendations** — when to use strong vs. light shrinkage
        """
    )


@app.cell
def cell_02():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_03():
    """Render the theory section."""
    mo.md(
        r"""
        ## 📐 Theoretical Background

        ### The curse of dimensionality in covariance estimation

        Given $T$ observations of $n$ assets, the sample correlation matrix
        $\hat{C}$ is an unbiased estimator of the true correlation $C$ — but its
        **eigenvalues are biased** when $n / T$ is non-negligible.  The
        Marchenko–Pastur law (1967) characterises this bias: small eigenvalues
        are deflated and large eigenvalues are inflated, causing the matrix
        to be ill-conditioned and difficult to invert reliably.

        ### Linear shrinkage toward the identity

        The **convex linear shrinkage** estimator mixes the sample matrix with
        the identity:

        $$\hat{C}(\lambda) = \lambda \cdot \hat{C}_{\text{EWMA}} + (1 - \lambda) \cdot I_n$$

        This pulls all eigenvalues toward 1, reducing variance at the cost of a
        small bias.  Ledoit & Wolf (2004) showed that the optimal $\lambda^*$
        minimises the expected Frobenius loss and has a closed-form expression
        depending on $n$, $T$, and the spectral structure of the data.

        ### Why Basanos uses a fixed λ instead of optimal shrinkage

        The Basanos optimizer uses shrinkage to **regularise a linear solver**
        (the system $C x = \mu$ must be well-posed at every timestamp), not to
        estimate a covariance matrix for portfolio optimisation.  In this context:

        - A fixed, user-controlled λ is simpler and more predictable.
        - The EWMA lookback already provides implicit regularisation.
        - Optimal shrinkage minimises Frobenius loss, not position stability.

        ### Interpreting `cfg.shrink` (= λ)

        | `cfg.shrink` | Effect |
        |:---:|:---|
        | `1.0` | No shrinkage — raw EWMA correlation used |
        | `0.5` | Equal mix of EWMA correlation and identity |
        | `0.0` | Full shrinkage — identity only (assets treated as uncorrelated) |

        **Rule of thumb:** start with $\lambda \approx 1 - n / (2T)$, then tune
        on held-out data by sweeping λ and measuring out-of-sample Sharpe ratio.
        """
    )


@app.cell
def cell_04():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_05():
    """Generate synthetic prices and signals (reused across all experiments)."""
    _rng = np.random.default_rng(42)
    _assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"]
    _n = 750  # ~3 years of daily data
    _start = pl.date(2021, 1, 1)
    _end = _start + pl.duration(days=_n - 1)
    _dates = pl.date_range(_start, _end, interval="1d", eager=True)

    _drift = [0.0003, 0.0002, 0.0004, 0.0001, 0.0005, 0.0002]
    _sigma = [0.018, 0.022, 0.020, 0.025, 0.030, 0.024]
    _s0 = [150.0, 2800.0, 300.0, 3300.0, 700.0, 200.0]

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
def cell_06():
    """Introduce the Sharpe vs. shrinkage sweep."""
    mo.md(
        r"""
        ## 🎚️ Sharpe Ratio vs. Shrinkage Intensity

        The chart below sweeps $\lambda$ from 0.0 (full shrinkage) to 1.0 (no
        shrinkage) in steps of 0.05, computing the annualised Sharpe ratio at
        each level for two lookback regimes:

        - **Short lookback** (`corr=20`) — high concentration ratio, sample matrix is noisy
        - **Long lookback** (`corr=120`) — low concentration ratio, sample matrix is more reliable

        Use the interactive controls below to choose `vola` and `clip`.
        """
    )


@app.cell
def cell_07():
    """Create controls for the Sharpe sweep."""
    vola_sweep = mo.ui.slider(
        start=4,
        stop=32,
        value=16,
        step=2,
        label="vola (volatility lookback, days):",
        show_value=True,
    )
    clip_sweep = mo.ui.slider(
        start=1.0,
        stop=6.0,
        value=3.5,
        step=0.5,
        label="clip (vol-adjusted return clipping threshold):",
        show_value=True,
    )
    mo.vstack([vola_sweep, clip_sweep])
    return clip_sweep, vola_sweep


@app.cell
def cell_08(clip_sweep, mu, prices, vola_sweep):
    """Sweep lambda and compute Sharpe ratio for short and long lookbacks."""
    _lambdas = np.arange(0.0, 1.05, 0.05).round(2)
    _results: dict[str, list[float]] = {"lambda": list(_lambdas), "short_corr": [], "long_corr": []}

    for _lam in _lambdas:
        for _corr, _key in [(20, "short_corr"), (120, "long_corr")]:
            _eff_corr = max(_corr, vola_sweep.value)
            _cfg = BasanosConfig(
                vola=vola_sweep.value,
                corr=_eff_corr,
                clip=clip_sweep.value,
                shrink=float(_lam),
                aum=1_000_000,
            )
            try:
                _eng = BasanosEngine(prices=prices, mu=mu, cfg=_cfg)
                _sharpe = float(_eng.portfolio.stats.sharpe(periods=252).get("returns", float("nan")))
            except Exception:
                _sharpe = float("nan")
            _results[_key].append(_sharpe)

    sweep_df = pl.DataFrame(_results)
    return (sweep_df,)


@app.cell
def cell_09(sweep_df):
    """Render the Sharpe vs. lambda chart."""
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=sweep_df["lambda"].to_list(),
            y=sweep_df["short_corr"].to_list(),
            mode="lines+markers",
            name="Short lookback (corr=20)",
            line={"color": "#e74c3c", "width": 2},
            marker={"size": 6},
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=sweep_df["lambda"].to_list(),
            y=sweep_df["long_corr"].to_list(),
            mode="lines+markers",
            name="Long lookback (corr=120)",
            line={"color": "#2980b9", "width": 2},
            marker={"size": 6},
        )
    )
    _fig.update_layout(
        title="Annualised Sharpe Ratio vs. Shrinkage Retention Weight (λ)",
        xaxis_title="λ = cfg.shrink  (0 = full shrinkage, 1 = no shrinkage)",
        yaxis_title="Sharpe ratio (annualised)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        height=450,
    )
    mo.ui.plotly(_fig)


@app.cell
def cell_10():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_11():
    """Introduce the condition number section."""
    mo.md(
        r"""
        ## 📊 Matrix Condition Number vs. λ

        The **condition number** $\kappa = \lambda_{\max} / \lambda_{\min}$ measures
        how close a matrix is to singular.  High condition numbers (> 1000) indicate
        that the linear system $C x = \mu$ is ill-posed and solutions are dominated
        by numerical noise rather than the signal.

        The chart below shows $\kappa$ as a function of λ for sample EWMA
        correlation matrices with varying portfolio sizes.
        """
    )


@app.cell
def cell_12(prices):
    """Compute condition numbers for a range of lambda values and portfolio sizes."""
    _lambdas_cond = np.arange(0.0, 1.01, 0.02).round(2)
    _asset_subsets = {
        "2 assets": ["AAPL", "GOOGL"],
        "4 assets": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "6 assets": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"],
    }
    _cond_results: dict[str, list] = {"lambda": list(_lambdas_cond)}

    for _label, _subset in _asset_subsets.items():
        _conds = []
        # Build a simple sample correlation matrix for the subset
        _data = prices.select(_subset).to_numpy().astype(float)
        _log_ret = np.diff(np.log(_data), axis=0)
        # Use last 40 rows as "sample" (short lookback scenario)
        _sample = _log_ret[-40:]
        _corr = np.corrcoef(_sample.T)
        for _lam in _lambdas_cond:
            _shrunk = shrink2id(_corr, lamb=float(_lam))
            _eigvals = np.linalg.eigvalsh(_shrunk)
            _kappa = _eigvals[-1] / max(_eigvals[0], 1e-14)
            _conds.append(float(_kappa))
        _cond_results[_label] = _conds

    cond_df = pl.DataFrame(_cond_results)
    return (cond_df,)


@app.cell
def cell_13(cond_df):
    """Render the condition number chart."""
    _colors = {"2 assets": "#27ae60", "4 assets": "#e67e22", "6 assets": "#8e44ad"}
    _fig2 = go.Figure()
    for _col in ["2 assets", "4 assets", "6 assets"]:
        _fig2.add_trace(
            go.Scatter(
                x=cond_df["lambda"].to_list(),
                y=cond_df[_col].to_list(),
                mode="lines",
                name=_col,
                line={"color": _colors[_col], "width": 2},
            )
        )
    _fig2.add_hline(
        y=1000,
        line_dash="dash",
        line_color="red",
        annotation_text="κ = 1000 (ill-conditioned threshold)",
        annotation_position="top left",
    )
    _fig2.update_layout(
        title="Condition Number κ vs. Shrinkage Retention Weight (λ)  [corr lookback = 40 days]",
        xaxis_title="λ = cfg.shrink",
        yaxis_title="Condition number κ (log scale)",
        yaxis_type="log",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        height=450,
    )
    mo.ui.plotly(_fig2)


@app.cell
def cell_14():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_15():
    """Introduce the interactive parameter-sensitivity section."""
    mo.md(
        r"""
        ## 🗺️ Interactive Parameter Sensitivity

        Explore the **joint effect** of `shrink` (λ) alongside the other
        optimizer parameters on portfolio Sharpe ratio.  This helps build
        intuition for which combinations of lookback and shrinkage work
        well for your specific dataset.
        """
    )


@app.cell
def cell_16():
    """Create interactive controls for sensitivity exploration."""
    vola_sens = mo.ui.slider(
        start=4,
        stop=32,
        value=16,
        step=2,
        label="vola (days):",
        show_value=True,
    )
    corr_sens = mo.ui.slider(
        start=4,
        stop=128,
        value=32,
        step=4,
        label="corr (days):",
        show_value=True,
    )
    clip_sens = mo.ui.slider(
        start=1.0,
        stop=6.0,
        value=3.5,
        step=0.5,
        label="clip:",
        show_value=True,
    )
    shrink_sens = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.5,
        step=0.05,
        label="shrink (λ):",
        show_value=True,
    )
    mo.vstack([vola_sens, corr_sens, clip_sens, shrink_sens])
    return clip_sens, corr_sens, shrink_sens, vola_sens


@app.cell
def cell_17(clip_sens, corr_sens, mu, prices, shrink_sens, vola_sens):
    """Build engine from sensitivity controls and display key metrics."""
    _eff_corr_s = max(corr_sens.value, vola_sens.value)
    _cfg_s = BasanosConfig(
        vola=vola_sens.value,
        corr=_eff_corr_s,
        clip=clip_sens.value,
        shrink=shrink_sens.value,
        aum=1_000_000,
    )
    _eng_s = BasanosEngine(prices=prices, mu=mu, cfg=_cfg_s)
    port_s = _eng_s.portfolio
    _n_assets = len(_eng_s.assets)
    _concentration = _n_assets / _eff_corr_s
    _sharpe_s = float(port_s.stats.sharpe(periods=252).get("returns", float("nan")))
    _vol_s = float(port_s.stats.volatility(periods=252).get("returns", float("nan")))

    _concentration_msg = (
        "⚠️ High concentration ratio (n/T > 0.3): consider reducing λ for better regularisation."
        if _concentration > 0.3
        else "✅ Concentration ratio is within a manageable range."
    )
    _corr_raised_msg = "⚠️ `corr` was raised to match `vola`." if _eff_corr_s > corr_sens.value else ""
    mo.callout(
        mo.md(
            f"""
            **Active configuration:**
            `BasanosConfig(vola={_cfg_s.vola}, corr={_cfg_s.corr}, clip={_cfg_s.clip},
            shrink={_cfg_s.shrink}, aum={_cfg_s.aum:,.0f})`

            | Metric | Value |
            |--------|-------|
            | Number of assets *n* | {_n_assets} |
            | Concentration ratio *n/T* | {_concentration:.3f} |
            | Annualised Sharpe | **{_sharpe_s:.3f}** |
            | Annualised volatility | {_vol_s:.4f} |

            {_concentration_msg}
            {_corr_raised_msg}
            """
        ),
        kind="info",
    )
    return (port_s,)


@app.cell
def cell_18(port_s):
    """Show the NAV chart for the selected configuration."""
    mo.vstack(
        [
            mo.md("### Performance (NAV)"),
            mo.ui.plotly(port_s.plots.snapshot()),
        ]
    )


@app.cell
def cell_19():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_20():
    """Render practical recommendations."""
    mo.md(
        r"""
        ## 💡 Practical Recommendations

        ### When to prefer **strong shrinkage** (low λ, e.g. 0.2 – 0.5)

        - Portfolio has **many assets** (n > 15) and a **short lookback**
          (`corr` < 50 days): the EWMA matrix is noisy and the condition number
          will be high without regularisation.
        - During **crisis / high-correlation regimes**: off-diagonal correlations
          spike and the sample matrix becomes nearly rank-deficient.
        - When **position stability** is the priority: heavy shrinkage damps
          the optimizer's sensitivity to noisy correlation estimates, reducing
          turnover.

        ### When to prefer **light shrinkage** (high λ, e.g. 0.7 – 0.95)

        - Portfolio has **few assets** (n < 8) with a **long lookback**
          (`corr` > 100 days): the concentration ratio is low and the EWMA
          matrix is reliable.
        - When **diversification signals** are strong: if genuine off-diagonal
          correlation structure exists that the optimizer should exploit, heavy
          shrinkage discards this information.

        ### Quick-start heuristic

        $$\lambda_{\text{start}} \approx 1 - \frac{n}{2T}$$

        where $n$ = number of assets and $T$ = `cfg.corr`. This approximates the
        Ledoit–Wolf optimal intensity and provides a sensible starting point.
        Always validate on out-of-sample data by sweeping λ and measuring
        held-out Sharpe ratio (see the sweep chart at the top of this notebook).

        ### References

        - Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for
          large-dimensional covariance matrices.* Journal of Multivariate
          Analysis, 88(2), 365–411.
          https://doi.org/10.1016/S0047-259X(03)00096-4
        - Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010).
          *Shrinkage algorithms for MMSE covariance estimation.* IEEE
          Transactions on Signal Processing, 58(10), 5016–5029.
          https://doi.org/10.1109/TSP.2010.2053029
        - Marchenko, V. A., & Pastur, L. A. (1967). *Distribution of
          eigenvalues for some sets of random matrices.* Mathematics of the
          USSR-Sbornik, 1(4), 457–483.
        """
    )


if __name__ == "__main__":
    app.run()
