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
    """Run both implementations and measure timing."""
    import time

    _com = com_slider.value
    _x = ret_adj.select(["A", "B", "C"]).to_numpy()

    # NumPy implementation
    _t0 = time.perf_counter()
    cor_numpy = _ewm_corr_numpy(_x, com=_com)
    elapsed_numpy = time.perf_counter() - _t0

    # Pandas implementation (original approach)
    _t0 = time.perf_counter()
    _ret_pd = pd.DataFrame(_x, columns=["A", "B", "C"])
    _ewm_corr = _ret_pd.ewm(com=_com, min_periods=_com).corr()
    _cor_long = _ewm_corr.reset_index(names=["t", "asset"])
    cor_pandas_dict = {t: df_t.drop(columns=["t", "asset"]).to_numpy() for t, df_t in _cor_long.groupby("t")}
    elapsed_pandas = time.perf_counter() - _t0

    # Convert pandas dict to (T, N, N) array aligned with the numpy output
    _n_time = len(_x)
    _n_assets = 3
    cor_pandas = np.full((_n_time, _n_assets, _n_assets), np.nan)
    for _t_key, _mat in cor_pandas_dict.items():
        cor_pandas[_t_key] = _mat

    mo.callout(
        mo.md(
            f"""
            **Timings** (com={_com}, T=200, N=3)

            | Implementation | Wall time |
            |---|---|
            | NumPy | `{elapsed_numpy * 1000:.2f} ms` |
            | Pandas | `{elapsed_pandas * 1000:.2f} ms` |
            """
        ),
        kind="info",
    )
    return cor_numpy, cor_pandas, elapsed_numpy, elapsed_pandas


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
def cell_16():
    """Render the conclusion."""
    mo.md(
        r"""
        ## 🎉 Conclusion

        The NumPy implementation reproduces the Pandas EWMA correlation output
        to **machine-precision accuracy** (max Δ < 10⁻¹⁰), confirming that the
        removal of `pandas` and `pyarrow` as runtime dependencies is safe.

        Key properties of the NumPy implementation:

        - **Single-pass accumulator** — O(T · N²) time, no intermediate DataFrame allocation
        - **NaN handling** — rows with NaN are skipped (no update, no decay), matching pandas' pairwise behaviour
        - **Identical decay** — `com`-parameterised, `adjust=True`-equivalent weighting
        - **No external dependencies** — pure NumPy, already a core runtime dependency
        """
    )


if __name__ == "__main__":
    app.run()
