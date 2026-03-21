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
    from plotly.subplots import make_subplots

    from basanos.math import BasanosConfig, BasanosEngine, SlidingWindowConfig


@app.cell
def cell_01():
    """Render the diagnostics notebook introduction."""
    mo.md(
        r"""
        # 🩺 Basanos — Engine Diagnostics

        **Basanos** exposes rich per-timestamp diagnostic properties that help you
        understand *why* the optimizer produced a particular set of positions —
        and flag when something is going wrong.

        This notebook demonstrates all five diagnostic properties of
        :class:`~basanos.math.BasanosEngine` on a synthetic dataset that contains
        deliberately engineered edge cases:

        | Property | Description |
        |---|---|
        | `position_status` | Per-row label: `warmup` / `zero_signal` / `degenerate` / `valid` |
        | `condition_number` | Condition number κ of the correlation matrix |
        | `effective_rank` | Entropy-based effective rank of the correlation matrix |
        | `solver_residual` | Euclidean residual ‖C·x − μ‖₂ of the linear solve |
        | `signal_utilisation` | Fraction of μᵢ surviving the correlation filter per asset |

        ## What this notebook covers

        1. 🧪 **Synthetic dataset** — prices and signals with warm-up, zero-signal, and low-shrinkage periods
        2. 📊 **Position-status breakdown** — stacked bar chart showing when each status occurs
        3. 📈 **Condition number & effective rank** — time series with a configurable ill-conditioning threshold
        4. 🔢 **Solver residual** — residual norm over time, showing when the linear solve degrades
        5. 🎛️ **Signal utilisation** — per-asset attenuation/amplification from correlation structure
        """
    )


@app.cell
def cell_02():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_03():
    """Introduce the synthetic data section."""
    mo.md(
        r"""
        ## 🧪 Synthetic Dataset with Edge Cases

        We generate **six synthetic equity-like price series** over 500 trading days.
        The dataset is crafted to contain three distinct edge cases that exercise every
        diagnostic code path:

        * **Early EWMA period** (rows 0–~60): the EWMA correlation estimator has not
          yet accumulated enough history so the correlation matrix is near-zero / NaN.
          With low shrinkage (λ = 0.99), the normalisation denominator is degenerate,
          producing `degenerate` rows.  Once the EWMA converges, rows become `valid`.
          > **Note:** `'warmup'` is a *Sliding Window*-only status code.  EWMA mode
          > emits `'degenerate'` during its convergence period, not `'warmup'`.
        * **Zero-signal period** (rows 100–149): the expected-return vector μ is forced
          to zero for all assets — the optimizer short-circuits without solving, producing
          `zero_signal` rows.
        * **Low-shrinkage period** (rows 200–299): the engine is configured with λ = 0.99
          (near-zero shrinkage), which can drive large condition numbers especially when
          the correlation matrix is near-singular — diagnostic rows may show `degenerate`.

        Outside these windows the engine runs normally, producing `valid` positions.
        """
    )


@app.cell
def cell_04():
    """Generate synthetic prices, signals, and config."""
    _seed = 7  # fixed seed for reproducibility
    _rng = np.random.default_rng(_seed)
    _assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA"]
    _n = 500
    _start = pl.date(2022, 1, 1)
    _dates = pl.date_range(_start, _start + pl.duration(days=_n - 1), interval="1d", eager=True)

    # Geometric Brownian motion with moderate correlations
    _drift = [0.0003, 0.0002, 0.0004, 0.0001, 0.0003, 0.0005]
    _sigma = [0.018, 0.022, 0.020, 0.025, 0.030, 0.028]
    _s0 = [150.0, 2800.0, 300.0, 3300.0, 330.0, 500.0]

    # Introduce mild correlation via a common factor
    _common = _rng.normal(0, 0.01, _n)
    _prices_cols: dict[str, object] = {"date": _dates}
    for _i, _a in enumerate(_assets):
        _log_ret = _rng.normal(_drift[_i], _sigma[_i], _n) + 0.4 * _common
        _prices_cols[_a] = _s0[_i] * np.exp(np.cumsum(_log_ret))

    prices = pl.DataFrame(_prices_cols)

    # Momentum signal: tanh of (MA5 - MA20) / price, clipped to [-1, 1]
    _mu_cols: dict[str, object] = {"date": _dates}
    for _a in _assets:
        _p = prices[_a].to_numpy()
        _ma5 = np.convolve(_p, np.ones(5) / 5, mode="full")[:_n]
        _ma20 = np.convolve(_p, np.ones(20) / 20, mode="full")[:_n]
        _raw = np.tanh(50.0 * (_ma5 - _ma20) / _p)
        # Zero-signal window: rows 100-149
        _raw[100:150] = 0.0
        _mu_cols[_a] = _raw

    mu = pl.DataFrame(_mu_cols)

    mo.callout(
        mo.md(
            f"""
            **Dataset summary**

            | | Value |
            |---|---|
            | Assets | {", ".join(_assets)} |
            | Rows | {_n} |
            | Date range | {_dates[0]} → {_dates[-1]} |
            | Zero-signal window | rows 100-149 (50 rows) |
            """
        ),
        kind="info",
    )
    return mu, prices


@app.cell
def cell_05():
    """Introduce the configuration section."""
    mo.md(
        r"""
        ## ⚙️ Engine Configuration

        We use the default EWMA-shrinkage mode with **low shrinkage** (`λ = 0.99`)
        to deliberately surface ill-conditioning in the diagnostic charts.
        You can adjust the parameters below using the sliders.
        """
    )


@app.cell
def cell_06():
    """Create interactive sliders for BasanosConfig parameters."""
    shrink_slider = mo.ui.slider(
        start=0.1,
        stop=1.0,
        step=0.05,
        value=0.99,
        label="Shrinkage retention weight λ (higher → less regularisation)",
    )
    corr_slider = mo.ui.slider(
        start=20,
        stop=120,
        step=10,
        value=64,
        label="Correlation EWMA half-life (days)",
    )
    kappa_threshold = mo.ui.slider(
        start=50,
        stop=5000,
        step=50,
        value=500,
        label="Condition-number threshold κ* (dashed line on chart)",
    )
    mo.vstack([shrink_slider, corr_slider, kappa_threshold])
    return corr_slider, kappa_threshold, shrink_slider


@app.cell
def cell_07(corr_slider, kappa_threshold, mu, prices, shrink_slider):
    """Build BasanosEngine from slider values and extract all diagnostic properties."""
    cfg = BasanosConfig(
        vola=16,
        corr=corr_slider.value,
        clip=3.5,
        shrink=shrink_slider.value,
        aum=1_000_000,
    )
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    # Extract all five diagnostic properties
    pos_status = engine.position_status
    cond_num = engine.condition_number
    eff_rank = engine.effective_rank
    solv_res = engine.solver_residual
    sig_util = engine.signal_utilisation

    # Count status codes for callout
    _counts = pos_status.group_by("status").len().sort("status")
    _rows = [f"| `{r['status']}` | {r['len']} |" for r in _counts.iter_rows(named=True)]

    mo.callout(
        mo.md(
            f"""
            **Active configuration**: λ = {shrink_slider.value}, corr half-life = {corr_slider.value} days

            **Position status breakdown** ({prices.height} rows total):

            | Status | Count |
            |---|---|
            {"".join(_rows)}
            """
        ),
        kind="success",
    )
    return cfg, cond_num, eff_rank, engine, kappa_threshold, pos_status, sig_util, solv_res


@app.cell
def cell_08():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_09():
    """Introduce position-status chart."""
    mo.md(
        r"""
        ## 📊 Position Status Breakdown

        The stacked bar chart below shows how the four status codes are
        distributed over time. Each bar covers a 10-day bucket.

        * **warmup** (blue) — appears at the start when the EWMA estimator
          has insufficient history.
        * **zero_signal** (orange) — rows 100–149, where μ was forced to zero.
        * **degenerate** (red) — timestamps where the solve failed numerically.
        * **valid** (green) — normal rows where positions were computed.
        """
    )


@app.cell
def cell_10(pos_status):
    """Render the position-status stacked bar chart."""
    _bucket_size = 10  # days per bar
    _df = pos_status.with_columns((pl.arange(0, pos_status.height) // _bucket_size).alias("bucket"))
    _buckets = sorted(_df["bucket"].unique().to_list())
    _statuses = ["warmup", "zero_signal", "degenerate", "valid"]
    _colors = {
        "warmup": "#3498db",
        "zero_signal": "#e67e22",
        "degenerate": "#e74c3c",
        "valid": "#27ae60",
    }

    # Build a count matrix: rows = buckets, columns = statuses
    _counts: dict[str, list[int]] = {s: [] for s in _statuses}
    _bucket_labels: list[str] = []
    for _b in _buckets:
        _slice = _df.filter(pl.col("bucket") == _b)
        _start_row = int(_b * _bucket_size)
        _bucket_labels.append(f"Row {_start_row}")
        _vc = {r[0]: r[1] for r in _slice.group_by("status").len().iter_rows()}
        for _s in _statuses:
            _counts[_s].append(_vc.get(_s, 0))

    _fig1 = go.Figure()
    for _s in _statuses:
        _fig1.add_trace(
            go.Bar(
                x=_bucket_labels,
                y=_counts[_s],
                name=_s,
                marker_color=_colors[_s],
            )
        )
    _fig1.update_layout(
        barmode="stack",
        title="Position Status Breakdown (10-row buckets)",
        xaxis_title="Bucket start row",
        yaxis_title="Number of rows",
        legend_title="Status",
        height=380,
        xaxis={"tickangle": -45},
    )
    _fig1


@app.cell
def cell_11():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_12():
    """Introduce condition number and effective rank section."""
    mo.md(
        r"""
        ## 📈 Condition Number & Effective Rank

        * **Condition number κ** — the ratio of the largest to smallest eigenvalue of
          the correlation matrix.  κ ≫ 1 means the matrix is nearly singular and the
          linear solve may be inaccurate.  A threshold of κ* = 500 (adjustable above)
          is a common rule of thumb for flagging ill-conditioned matrices.
        * **Effective rank** — an entropy-based measure of the number of *effective*
          dimensions in the correlation matrix.  A value close to the number of assets
          indicates a well-spread spectrum; a value of 1 indicates a rank-1 matrix.

        With λ close to 1 (low shrinkage), the raw EWMA matrix dominates and condition
        numbers can spike dramatically, especially during short lookback windows.
        """
    )


@app.cell
def cell_13(cond_num, eff_rank, kappa_threshold):
    """Render the condition number and effective rank charts."""
    _fig2 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Condition Number κ (log scale)", "Effective Rank"),
        vertical_spacing=0.12,
    )

    # Condition number (row 1)
    _fig2.add_trace(
        go.Scatter(
            x=list(range(cond_num.height)),
            y=cond_num["condition_number"].to_list(),
            mode="lines",
            name="κ",
            line={"color": "#8e44ad", "width": 1.5},
        ),
        row=1,
        col=1,
    )
    _fig2.add_hline(
        y=kappa_threshold.value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"κ* = {kappa_threshold.value} (threshold)",
        annotation_position="top left",
        row=1,
        col=1,
    )

    # Effective rank (row 2)
    _fig2.add_trace(
        go.Scatter(
            x=list(range(eff_rank.height)),
            y=eff_rank["effective_rank"].to_list(),
            mode="lines",
            name="Effective rank",
            line={"color": "#2980b9", "width": 1.5},
        ),
        row=2,
        col=1,
    )

    _fig2.update_layout(
        title="Matrix Quality Diagnostics",
        height=550,
        showlegend=True,
    )
    _fig2.update_yaxes(type="log", row=1, col=1)
    _fig2.update_xaxes(title_text="Row index", row=2, col=1)
    _fig2


@app.cell
def cell_14():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_15():
    """Introduce solver residual section."""
    mo.md(
        r"""
        ## 🔢 Solver Residual ‖C·x − μ‖₂

        After solving **C · x = μ** the engine computes the Euclidean residual
        ‖C·x − μ‖₂.  For a well-conditioned system on modern hardware this value
        sits near machine epsilon (~10⁻¹⁵).

        Large residuals (> 10⁻⁶) indicate:

        * Near-singular correlation matrices (high κ), causing the Cholesky solve
          to accumulate rounding errors.
        * Fallback to LU decomposition when Cholesky fails, which can still leave
          a non-trivial residual.
        * `NaN` residuals appear during `warmup` or `degenerate` rows.

        The residual is reported as **zero** during `zero_signal` rows (no solve is
        performed when μ = 0).
        """
    )


@app.cell
def cell_16(solv_res):
    """Render the solver residual time series chart."""
    _fig3 = go.Figure()
    _fig3.add_trace(
        go.Scatter(
            x=list(range(solv_res.height)),
            y=solv_res["residual"].to_list(),
            mode="lines",
            name="‖C·x − μ‖₂",
            line={"color": "#c0392b", "width": 1.5},
        )
    )
    _fig3.update_layout(
        title="Solver Residual ‖C·x − μ‖₂ over Time",
        xaxis_title="Row index",
        yaxis_title="Residual norm (log scale)",
        yaxis_type="log",
        height=380,
    )
    _fig3


@app.cell
def cell_17():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_18():
    """Introduce signal utilisation section."""
    mo.md(
        r"""
        ## 🎛️ Signal Utilisation per Asset

        Signal utilisation **uᵢ** measures what fraction of the raw signal μᵢ
        survives after the correlation filter is applied:

        $$u_i = \frac{(C^{-1}\,\mu)_i}{\mu_i}$$

        * **uᵢ = 1** — asset *i* is uncorrelated with all others; the correlation
          filter leaves its signal unchanged.
        * **uᵢ < 1** — the asset's signal is attenuated because it is correlated
          with other assets that already carry similar information.
        * **uᵢ > 1** — the asset's signal is amplified because it is *negatively*
          correlated with others (diversification benefit).

        NaN values appear during `warmup` and `degenerate` rows, or when |μᵢ| is
        below machine precision.
        """
    )


@app.cell
def cell_19(sig_util):
    """Render the signal utilisation chart."""
    _assets = [c for c in sig_util.columns if c != "date"]
    _palette = [
        "#e74c3c",
        "#3498db",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
    ]

    _fig4 = go.Figure()
    for _j, _a in enumerate(_assets):
        _fig4.add_trace(
            go.Scatter(
                x=list(range(sig_util.height)),
                y=sig_util[_a].to_list(),
                mode="lines",
                name=_a,
                line={"color": _palette[_j % len(_palette)], "width": 1.2},
                opacity=0.85,
            )
        )

    _fig4.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="black",
        annotation_text="uᵢ = 1 (no attenuation)",
        annotation_position="top right",
    )
    _fig4.update_layout(
        title="Signal Utilisation uᵢ per Asset",
        xaxis_title="Row index",
        yaxis_title="Utilisation uᵢ",
        height=420,
        legend_title="Asset",
    )
    _fig4


@app.cell
def cell_20():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_sw_01():
    """Introduce the Sliding Window mode section."""
    mo.md(
        r"""
        ## 🪟 Sliding Window Mode Diagnostics

        The same five diagnostic properties are available when the engine is configured
        with :class:`~basanos.math.SlidingWindowConfig` (``covariance_mode = 'sliding_window'``).
        This mode fits a low-rank factor model over a rolling window of returns instead of
        maintaining an EWMA correlation matrix.

        **Key behavioural difference**: in sliding window mode the ``'warmup'`` status code
        is emitted explicitly for the first ``window - 1`` rows — there is simply not enough
        history to fill the window yet.  EWMA mode never emits ``'warmup'``; its convergence
        period appears as ``'degenerate'`` rows instead.

        Below we run the same dataset through a sliding-window engine
        (``window = 80``, ``n_factors = 3``) so you can compare the diagnostic output
        side-by-side with the EWMA results above.
        """
    )


@app.cell
def cell_sw_02(mu, prices):
    """Build a sliding-window BasanosEngine and extract its diagnostic properties."""
    _sw_cfg = BasanosConfig(
        vola=16,
        corr=64,
        clip=3.5,
        shrink=0.5,
        aum=1_000_000,
        covariance_config=SlidingWindowConfig(window=80, n_factors=3),
    )
    sw_engine = BasanosEngine(prices=prices, mu=mu, cfg=_sw_cfg)

    sw_pos_status = sw_engine.position_status
    sw_cond_num = sw_engine.condition_number
    sw_eff_rank = sw_engine.effective_rank
    sw_solv_res = sw_engine.solver_residual
    sw_sig_util = sw_engine.signal_utilisation

    _sw_counts = sw_pos_status.group_by("status").len().sort("status")
    _sw_rows = [f"| `{r['status']}` | {r['len']} |" for r in _sw_counts.iter_rows(named=True)]

    mo.callout(
        mo.md(
            f"""
            **Sliding Window config**: window = 80, n_factors = 3, shrink = 0.5

            **Position status breakdown** ({prices.height} rows total):

            | Status | Count |
            |---|---|
            {"".join(_sw_rows)}

            > The first **79 rows** carry `'warmup'` status — the window cannot
            > be filled until row 80.  Compare with EWMA mode above, where the
            > early rows appear as `'degenerate'` rather than `'warmup'`.
            """
        ),
        kind="neutral",
    )
    return sw_cond_num, sw_eff_rank, sw_engine, sw_pos_status, sw_sig_util, sw_solv_res


@app.cell
def cell_sw_03(sw_cond_num, sw_eff_rank, sw_pos_status):
    """Render SW position-status and condition-number charts side by side."""
    from plotly.subplots import make_subplots as _make_subplots

    # ── Position-status stacked bar ──────────────────────────────────────────
    _bucket_size = 10
    _df_sw = sw_pos_status.with_columns((pl.arange(0, sw_pos_status.height) // _bucket_size).alias("bucket"))
    _buckets_sw = sorted(_df_sw["bucket"].unique().to_list())
    _statuses_sw = ["warmup", "zero_signal", "degenerate", "valid"]
    _colors_sw = {
        "warmup": "#3498db",
        "zero_signal": "#e67e22",
        "degenerate": "#e74c3c",
        "valid": "#27ae60",
    }
    _counts_sw: dict[str, list[int]] = {s: [] for s in _statuses_sw}
    _labels_sw: list[str] = []
    for _b in _buckets_sw:
        _slice = _df_sw.filter(pl.col("bucket") == _b)
        _start_row = int(_b * _bucket_size)
        _labels_sw.append(f"Row {_start_row}")
        _vc = {r[0]: r[1] for r in _slice.group_by("status").len().iter_rows()}
        for _s in _statuses_sw:
            _counts_sw[_s].append(_vc.get(_s, 0))

    _fig_sw = _make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Position Status (SW, 10-row buckets)", "Condition Number κ (SW, log)"),
        horizontal_spacing=0.12,
    )

    for _s in _statuses_sw:
        _fig_sw.add_trace(
            go.Bar(x=_labels_sw, y=_counts_sw[_s], name=_s, marker_color=_colors_sw[_s]),
            row=1,
            col=1,
        )

    # ── Condition number ─────────────────────────────────────────────────────
    _fig_sw.add_trace(
        go.Scatter(
            x=list(range(sw_cond_num.height)),
            y=sw_cond_num["condition_number"].to_list(),
            mode="lines",
            name="κ (SW)",
            line={"color": "#8e44ad", "width": 1.5},
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    _fig_sw.update_layout(
        barmode="stack",
        height=400,
        title_text="Sliding Window Mode Diagnostics",
        legend_title="Status / Series",
    )
    _fig_sw.update_yaxes(type="log", row=1, col=2)
    _fig_sw.update_xaxes(tickangle=-45, row=1, col=1)
    _fig_sw.update_xaxes(title_text="Row index", row=1, col=2)
    _fig_sw


@app.cell
def cell_21():
    """Render closing remarks."""
    mo.md(
        r"""
        ## 💡 Summary & Practical Guidelines

        | Diagnostic | What to watch for | Action |
        |---|---|---|
        | `position_status` | `degenerate` (EWMA) or `warmup` (SW) persist | Increase shrinkage / window |
        | `condition_number` | κ > 500 in sustained periods | Reduce λ or switch to `sliding_window` mode |
        | `effective_rank` | Effective rank ≪ n_assets | Correlation is concentrated; few factors dominate |
        | `solver_residual` | Residuals > 10⁻⁶ | Numerical concern — inspect κ at those timestamps |
        | `signal_utilisation` | uᵢ < 0.2 consistently | Signal is heavily attenuated; consider excluding |

        All five properties are available on every :class:`~basanos.math.BasanosEngine`
        instance and are computed lazily (each property runs its own pass through
        the stored price and signal data).

        Both **EWMA-shrink** and **Sliding Window** covariance modes expose the same
        diagnostic API — but their ``position_status`` codes differ during the
        convergence / warmup period:

        * **EWMA mode**: early rows appear as ``'degenerate'`` (NaN matrix → solve fails).
        * **SW mode**: first ``window - 1`` rows appear as ``'warmup'`` (explicit guard).
        """
    )


if __name__ == "__main__":
    app.run()
