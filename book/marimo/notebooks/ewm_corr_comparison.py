# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.20.4",
#     "basanos",
#     "numpy>=2.0.0",
#     "polars>=1.0.0",
#     "pandas>=3.0.0",
#     "pyarrow>=23.0.1",
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
    import pandas as pd
    import plotly.graph_objects as go
    import polars as pl

    from basanos.math._signal import vol_adj
    from basanos.math.optimizer import _ewm_corr_numpy


@app.cell
def cell_01():
    """Render the notebook introduction."""
    mo.md(
        r"""
        # 🔬 EWMA Correlation: NumPy vs Pandas

        This notebook compares two implementations of the exponentially
        weighted moving average (EWMA) correlation matrix computation:

        | | Implementation | Dependency |
        |---|---|---|
        | **NumPy** | Custom accumulator using normalised cumulative weights | `numpy` only |
        | **Pandas** | `DataFrame.ewm(com=…).corr()` | `pandas` + `pyarrow` |

        Both use `com`-parameterised decay (`α = 1 / (1 + com)`) and an
        `adjust=True`-equivalent normalisation. The goal is to verify that
        the NumPy replacement produces numerically equivalent results so that
        removing `pandas` and `pyarrow` as runtime dependencies is safe.
        """
    )


@app.cell
def cell_02():
    """Render horizontal rule."""
    mo.md("---")


@app.cell
def cell_03():
    """Introduce synthetic data section."""
    mo.md(
        r"""
        ## 📈 Synthetic Data

        We generate three synthetic price series over **200 trading days**
        with a fixed seed for full reproducibility. Volatility-adjusted log
        returns are computed with the same `vol_adj` helper used by
        `BasanosEngine`, making the comparison directly representative of
        production inputs.
        """
    )


@app.cell
def cell_04():
    """Generate synthetic prices and vol-adjusted returns."""
    _rng = np.random.default_rng(7)
    _n = 200
    _start = pl.date(2023, 1, 1)
    _end = _start + pl.duration(days=_n - 1)
    _dates = pl.date_range(_start, _end, interval="1d", eager=True)
    _assets = ["A", "B", "C"]

    _prices_dict: dict[str, object] = {"date": _dates}
    for _a in _assets:
        _log_ret = _rng.normal(0.0002, 0.015, _n)
        _prices_dict[_a] = 100.0 * np.exp(np.cumsum(_log_ret))

    prices = pl.DataFrame(_prices_dict)

    # Compute vol-adjusted returns using the same helper as BasanosEngine
    _vola = 20
    _clip = 3.5
    ret_adj = prices.with_columns([vol_adj(pl.col(a), vola=_vola, clip=_clip) for a in _assets])

    mo.vstack(
        [
            mo.md("### Price data (first 5 rows)"),
            mo.ui.table(prices.head(5)),
            mo.md("### Vol-adjusted returns (first 5 rows)"),
            mo.ui.table(ret_adj.head(5)),
        ]
    )
    return prices, ret_adj


@app.cell
def cell_05():
    """Introduce the com slider."""
    mo.md(
        r"""
        ## ⚙️ EWMA Lookback

        Adjust the `com` (centre-of-mass) parameter to control the decay
        rate used by both implementations.  A larger value gives more weight
        to older observations.
        """
    )


@app.cell
def cell_06():
    """Interactive slider for com parameter."""
    com_slider = mo.ui.slider(
        start=5,
        stop=80,
        value=20,
        step=1,
        label="com (centre-of-mass for EWMA decay):",
        show_value=True,
    )
    com_slider
    return (com_slider,)


@app.cell
def cell_07():
    """Render separator."""
    mo.md("---")


@app.cell
def cell_08():
    """Introduce implementation comparison section."""
    mo.md(
        r"""
        ## ⚖️ Implementation Comparison

        Both implementations receive the same `(T, N)` array of
        vol-adjusted returns and produce a `(T, N, N)` correlation cube.

        ### NumPy implementation

        Accumulates EWMA sufficient statistics on-line using:

        $$
        \hat{\Sigma}_t =
        \frac{\sum_{s \le t} \alpha(1-\alpha)^{t-s} x_s x_s^\top}{\sum_{s \le t} \alpha(1-\alpha)^{t-s}}
        - \hat{\mu}_t \hat{\mu}_t^\top
        $$

        then normalises off-diagonal entries by $\sqrt{\sigma_{ii}\,\sigma_{jj}}$.
        Rows containing NaN are **skipped** and do not count toward the ``com``
        minimum-observations threshold, matching pandas' pairwise behaviour.

        ### Pandas implementation

        Delegates to `pandas.DataFrame.ewm(com=com, min_periods=com).corr()`,
        which computes pairwise EWMA covariance then normalises.
        """
    )


@app.cell
def cell_09(com_slider, ret_adj):
    """Run both implementations and compute correlation cubes for comparison."""
    _com = com_slider.value
    _x = ret_adj.select(["A", "B", "C"]).to_numpy()

    # NumPy implementation
    cor_numpy = _ewm_corr_numpy(_x, com=_com)

    # Pandas implementation (original approach)
    _ret_pd = pd.DataFrame(_x, columns=["A", "B", "C"])
    _ewm_corr = _ret_pd.ewm(com=_com, min_periods=_com).corr()
    _cor_long = _ewm_corr.reset_index(names=["t", "asset"])
    _cor_pandas_dict = {t: df_t.drop(columns=["t", "asset"]).to_numpy() for t, df_t in _cor_long.groupby("t")}

    # Convert pandas dict to (T, N, N) array aligned with the numpy output
    _n_time = len(_x)
    _n_assets = 3
    cor_pandas = np.full((_n_time, _n_assets, _n_assets), np.nan)
    for _t_key, _mat in _cor_pandas_dict.items():
        cor_pandas[_t_key] = _mat

    return cor_numpy, cor_pandas


@app.cell
def cell_10(cor_numpy, cor_pandas):
    """Compute numerical differences between the two implementations."""
    # Only compare timesteps where both are finite
    _valid = np.isfinite(cor_numpy).all(axis=(1, 2)) & np.isfinite(cor_pandas).all(axis=(1, 2))
    _diff = np.abs(cor_numpy[_valid] - cor_pandas[_valid])

    _max_diff = float(_diff.max()) if _valid.any() else float("nan")
    _mean_diff = float(_diff.mean()) if _valid.any() else float("nan")
    _valid_count = int(_valid.sum())
    _total = len(cor_numpy)

    mo.callout(
        mo.md(
            f"""
            **Numerical Agreement** (finite timesteps only: {_valid_count}/{_total})

            | Metric | Value |
            |---|---|
            | Max absolute difference | `{_max_diff:.2e}` |
            | Mean absolute difference | `{_mean_diff:.2e}` |
            | Numerically equivalent? | {"✅ Yes (< 1e-10)" if _max_diff < 1e-10 else "⚠️ No"} |
            """
        ),
        kind="success" if _max_diff < 1e-10 else "warn",
    )
    return _max_diff, _mean_diff, _valid_count


@app.cell
def cell_11():
    """Render separator."""
    mo.md("---")


@app.cell
def cell_12():
    """Introduce the visual comparison section."""
    mo.md(
        r"""
        ## 📊 Visual Comparison

        The charts below plot the off-diagonal correlation entries over time for
        both implementations.  Overlapping lines confirm numerical agreement.
        """
    )


@app.cell
def cell_13(cor_numpy, cor_pandas, prices):
    """Plot correlation time-series for each asset pair."""
    _dates = prices["date"].to_list()
    _pairs = [("A", "B", 0, 1), ("A", "C", 0, 2), ("B", "C", 1, 2)]
    _colours_np = ["#2FA4A9", "#E06C4B", "#6A9FD8"]
    _colours_pd = ["#A8DFE0", "#F4B8A4", "#B8D4F0"]

    _fig = go.Figure()
    for _idx, (_lbl_a, _lbl_b, _i, _j) in enumerate(_pairs):
        _np_series = cor_numpy[:, _i, _j]
        _pd_series = cor_pandas[:, _i, _j]
        _pair_lbl = f"{_lbl_a}–{_lbl_b}"

        _fig.add_trace(
            go.Scatter(
                x=_dates,
                y=_np_series,
                name=f"NumPy {_pair_lbl}",
                line={"color": _colours_np[_idx], "width": 2},
                legendgroup=_pair_lbl,
            )
        )
        _fig.add_trace(
            go.Scatter(
                x=_dates,
                y=_pd_series,
                name=f"Pandas {_pair_lbl}",
                line={"color": _colours_pd[_idx], "width": 2, "dash": "dot"},
                legendgroup=_pair_lbl,
            )
        )

    _fig.update_layout(
        title="EWMA Correlation Time-Series: NumPy (solid) vs Pandas (dotted)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis={"range": [-1.1, 1.1]},
        template="plotly_white",
        height=450,
        legend={"orientation": "h", "y": -0.2},
    )

    mo.ui.plotly(_fig)


@app.cell
def cell_14(cor_numpy, cor_pandas, prices):
    """Plot per-timestep max absolute difference."""
    _dates = prices["date"].to_list()
    _valid = np.isfinite(cor_numpy).all(axis=(1, 2)) & np.isfinite(cor_pandas).all(axis=(1, 2))
    _max_per_t = np.where(_valid, np.nanmax(np.abs(cor_numpy - cor_pandas), axis=(1, 2)), np.nan)

    _fig2 = go.Figure(
        go.Scatter(
            x=_dates,
            y=_max_per_t,
            mode="lines",
            line={"color": "#E06C4B", "width": 1.5},
            name="Max |Δcorr|",
        )
    )
    _fig2.update_layout(
        title="Per-Timestep Maximum Absolute Difference (NumPy − Pandas)",
        xaxis_title="Date",
        yaxis_title="|Δ correlation|",
        yaxis={"exponentformat": "e"},
        template="plotly_white",
        height=350,
    )

    mo.ui.plotly(_fig2)


@app.cell
def cell_15():
    """Render separator."""
    mo.md("---")


@app.cell
def cell_bench_intro():
    """Introduce the benchmark section."""
    mo.md(
        r"""
        ## ⏱️ Benchmark

        Which implementation is faster, and by how much?

        The cells below sweep across portfolio sizes **N** (number of assets)
        and sequence lengths **T** (number of timesteps), running each
        implementation **20 times** and recording the **minimum** wall time —
        the minimum is robust to OS scheduling noise and gives the closest
        estimate of true compute cost.

        | Sweep | Fixed parameter |
        |---|---|
        | Vary N (3 → 50), fixed T = 500 | shows how both scale with portfolio size |
        | Vary T (100 → 5 000), fixed N = 10 | shows linear scaling with history length |
        """
    )


@app.cell
def cell_bench_run():
    """Run benchmark sweep: vary N with fixed T, and vary T with fixed N."""
    import timeit

    _repeats = 20
    _com_bench = 20

    _ns = [3, 5, 10, 20, 30, 50]
    _t_fixed = 500
    _np_by_n: list[float] = []
    _pd_by_n: list[float] = []

    for _n in _ns:
        # Fixed seed ensures both implementations always receive the same data,
        # making the comparison fair regardless of loop order.
        _rng_b = np.random.default_rng(0)
        _x_b = _rng_b.normal(0, 1, (_t_fixed, _n))
        _cols_b = [f"X{i}" for i in range(_n)]
        _df_b = pd.DataFrame(_x_b, columns=_cols_b)

        _np_by_n.append(
            min(
                timeit.repeat(
                    lambda x=_x_b, c=_com_bench: _ewm_corr_numpy(x, com=c),
                    number=1,
                    repeat=_repeats,
                )
            )
            * 1000
        )

        def _run_pd_n(df=_df_b, c=_com_bench):
            _e = df.ewm(com=c, min_periods=c).corr()
            _r = _e.reset_index(names=["t", "asset"])
            return {t: g.drop(columns=["t", "asset"]).to_numpy() for t, g in _r.groupby("t")}

        _pd_by_n.append(min(timeit.repeat(_run_pd_n, number=1, repeat=_repeats)) * 1000)

    _ts = [100, 500, 1000, 2000, 5000]
    _n_fixed = 10
    _np_by_t: list[float] = []
    _pd_by_t: list[float] = []

    for _t in _ts:
        # Fixed seed so both implementations see the same data at each T value.
        _rng_b = np.random.default_rng(0)
        _x_b = _rng_b.normal(0, 1, (_t, _n_fixed))
        _cols_b = [f"X{i}" for i in range(_n_fixed)]
        _df_b = pd.DataFrame(_x_b, columns=_cols_b)

        _np_by_t.append(
            min(
                timeit.repeat(
                    lambda x=_x_b, c=_com_bench: _ewm_corr_numpy(x, com=c),
                    number=1,
                    repeat=_repeats,
                )
            )
            * 1000
        )

        def _run_pd_t(df=_df_b, c=_com_bench):
            _e = df.ewm(com=c, min_periods=c).corr()
            _r = _e.reset_index(names=["t", "asset"])
            return {t: g.drop(columns=["t", "asset"]).to_numpy() for t, g in _r.groupby("t")}

        _pd_by_t.append(min(timeit.repeat(_run_pd_t, number=1, repeat=_repeats)) * 1000)

    bench = {
        "ns": _ns,
        "np_by_n": _np_by_n,
        "pd_by_n": _pd_by_n,
        "ts": _ts,
        "np_by_t": _np_by_t,
        "pd_by_t": _pd_by_t,
    }
    return (bench,)


@app.cell
def cell_bench_chart(bench):
    """Plot benchmark results: scaling with N and T, plus speedup ratios."""
    from plotly.subplots import make_subplots

    _ns = bench["ns"]
    _np_n = bench["np_by_n"]
    _pd_n = bench["pd_by_n"]
    _speedup_n = [p / n for p, n in zip(_pd_n, _np_n, strict=True)]

    _ts = bench["ts"]
    _np_t = bench["np_by_t"]
    _pd_t = bench["pd_by_t"]
    _speedup_t = [p / n for p, n in zip(_pd_t, _np_t, strict=True)]

    _fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Time vs N assets (T = 500)",
            "Speedup vs N assets",
            "Time vs T timesteps (N = 10)",
            "Speedup vs T timesteps",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    _nx = [str(n) for n in _ns]
    _tx = [str(t) for t in _ts]
    _np_col = "#2FA4A9"
    _pd_col = "#E06C4B"
    _su_col = "#6A9FD8"

    # Row 1: vary N
    _fig.add_trace(go.Bar(name="NumPy", x=_nx, y=_np_n, marker_color=_np_col, legendgroup="np"), row=1, col=1)
    _fig.add_trace(go.Bar(name="Pandas", x=_nx, y=_pd_n, marker_color=_pd_col, legendgroup="pd"), row=1, col=1)
    _fig.add_trace(
        go.Bar(name="Speedup ×", x=_nx, y=_speedup_n, marker_color=_su_col, showlegend=True, legendgroup="su"),
        row=1,
        col=2,
    )
    _fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=1, col=2)

    # Row 2: vary T
    _fig.add_trace(go.Bar(x=_tx, y=_np_t, marker_color=_np_col, showlegend=False, legendgroup="np"), row=2, col=1)
    _fig.add_trace(go.Bar(x=_tx, y=_pd_t, marker_color=_pd_col, showlegend=False, legendgroup="pd"), row=2, col=1)
    _fig.add_trace(
        go.Bar(x=_tx, y=_speedup_t, marker_color=_su_col, showlegend=False, legendgroup="su"),
        row=2,
        col=2,
    )
    _fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=2, col=2)

    _fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=600,
        legend={"orientation": "h", "y": -0.08},
    )
    _fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    _fig.update_yaxes(title_text="Pandas / NumPy ×", row=1, col=2)
    _fig.update_yaxes(title_text="Time (ms)", row=2, col=1)
    _fig.update_yaxes(title_text="Pandas / NumPy ×", row=2, col=2)
    _fig.update_xaxes(title_text="N (assets)", row=1, col=1)
    _fig.update_xaxes(title_text="N (assets)", row=1, col=2)
    _fig.update_xaxes(title_text="T (timesteps)", row=2, col=1)
    _fig.update_xaxes(title_text="T (timesteps)", row=2, col=2)

    mo.ui.plotly(_fig)


@app.cell
def cell_bench_table(bench):
    """Show benchmark results as tables."""
    _rows_n = [
        {"N (assets)": n, "NumPy (ms)": f"{np:.2f}", "Pandas (ms)": f"{pd:.2f}", "Speedup ×": f"{pd / np:.1f}x"}
        for n, np, pd in zip(bench["ns"], bench["np_by_n"], bench["pd_by_n"], strict=True)
    ]
    _rows_t = [
        {"T (timesteps)": t, "NumPy (ms)": f"{np:.2f}", "Pandas (ms)": f"{pd:.2f}", "Speedup ×": f"{pd / np:.1f}x"}
        for t, np, pd in zip(bench["ts"], bench["np_by_t"], bench["pd_by_t"], strict=True)
    ]
    mo.vstack(
        [
            mo.md("### Scaling with N (T = 500, 20 repeats, min time)"),
            mo.ui.table(_rows_n),
            mo.md("### Scaling with T (N = 10, 20 repeats, min time)"),
            mo.ui.table(_rows_t),
        ]
    )


@app.cell
def cell_15b():
    """Render separator before conclusion."""
    mo.md("---")


@app.cell
def cell_16():
    """Render the conclusion."""
    mo.md(
        r"""
        ## 🎉 Conclusion

        The NumPy implementation reproduces the Pandas EWMA correlation output
        to **machine-precision accuracy** (max Δ < 10⁻¹⁰), and is consistently
        **faster** — especially at larger portfolio sizes — because it avoids
        the overhead of intermediate DataFrame and MultiIndex construction.

        Key properties of the NumPy implementation:

        - **Single-pass accumulator** — O(T · N²) time, no intermediate DataFrame allocation
        - **NaN handling** — rows with NaN are skipped (no update, no decay), matching pandas' pairwise behaviour
        - **Identical decay** — `com`-parameterised, `adjust=True`-equivalent weighting
        - **No external dependencies** — pure NumPy, already a core runtime dependency
        """
    )


if __name__ == "__main__":
    app.run()
