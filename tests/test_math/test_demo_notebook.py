"""CI execution gate for the demo notebook.

This module mirrors the data setup and cell logic from
``book/marimo/notebooks/demo.py`` so that any drift in the engine and
portfolio analytics API is caught by ``make test`` before it can silently
corrupt the notebook.

Covered API surface:

- ``BasanosEngine`` — construction, ``cash_position``, ``assets``
- ``Portfolio`` — ``stats.summary()``, ``turnover_summary()``,
  ``tilt_timing_decomp``, ``monthly``
- ``Portfolio.plots`` — ``snapshot()``, ``lead_lag_ir_plot()``,
  ``correlation_heatmap()``
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from basanos.math import BasanosConfig, BasanosEngine

# ─── Constants (mirror cell_04 of the notebook) ───────────────────────────────

_SEED = 42
_ASSETS = ["AAPL", "GOOGL", "MSFT", "AMZN"]
_N = 750
_DRIFT = [0.0003, 0.0002, 0.0004, 0.0001]
_SIGMA = [0.018, 0.022, 0.020, 0.025]
_S0 = [150.0, 2800.0, 300.0, 3300.0]


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def demo_prices() -> pl.DataFrame:
    """Synthetic price series matching demo notebook cell_04."""
    rng = np.random.default_rng(_SEED)
    start = pl.date(2021, 1, 1)
    dates = pl.date_range(start, start + pl.duration(days=_N - 1), interval="1d", eager=True)
    cols: dict[str, object] = {"date": dates}
    for i, asset in enumerate(_ASSETS):
        log_ret = rng.normal(_DRIFT[i], _SIGMA[i], _N)
        cols[asset] = _S0[i] * np.exp(np.cumsum(log_ret))
    return pl.DataFrame(cols)


@pytest.fixture(scope="module")
def demo_mu(demo_prices: pl.DataFrame) -> pl.DataFrame:
    """Momentum signal matching demo notebook cell_04."""
    dates = demo_prices["date"]
    cols: dict[str, object] = {"date": dates}
    for asset in _ASSETS:
        p = demo_prices[asset].to_numpy()
        ma5 = np.convolve(p, np.ones(5) / 5, mode="same") / p - 1.0
        ma20 = np.convolve(p, np.ones(20) / 20, mode="same") / p - 1.0
        cols[asset] = np.tanh(50.0 * (ma5 - ma20))
    return pl.DataFrame(cols)


@pytest.fixture(scope="module")
def demo_engine(demo_prices: pl.DataFrame, demo_mu: pl.DataFrame) -> BasanosEngine:
    """BasanosEngine with the notebook's default configuration (cell_09)."""
    cfg = BasanosConfig(vola=16, corr=32, clip=3.5, shrink=0.5, aum=1_000_000)
    return BasanosEngine(prices=demo_prices, mu=demo_mu, cfg=cfg)


# ─── Engine API ───────────────────────────────────────────────────────────────


class TestDemoEngineApi:
    """BasanosEngine properties used in the demo notebook (cell_12 / cell_13)."""

    def test_assets_list(self, demo_engine: BasanosEngine) -> None:
        assert demo_engine.assets == _ASSETS

    def test_cash_position_columns(self, demo_engine: BasanosEngine) -> None:
        cols = demo_engine.cash_position.columns
        assert "date" in cols
        assert all(a in cols for a in _ASSETS)

    def test_cash_position_row_count(self, demo_engine: BasanosEngine) -> None:
        assert demo_engine.cash_position.height == _N

    def test_cash_position_values_are_finite_or_nan(self, demo_engine: BasanosEngine) -> None:
        for asset in _ASSETS:
            vals = demo_engine.cash_position[asset].to_numpy()
            assert np.all(np.isfinite(vals) | np.isnan(vals))


# ─── Portfolio stats ──────────────────────────────────────────────────────────


class TestDemoPortfolioStats:
    """Portfolio analytics properties used in the demo notebook (cell_16 / cell_17)."""

    def test_stats_summary_is_dataframe(self, demo_engine: BasanosEngine) -> None:
        result = demo_engine.portfolio.stats.summary()
        assert isinstance(result, pl.DataFrame)

    def test_stats_summary_has_rows(self, demo_engine: BasanosEngine) -> None:
        assert demo_engine.portfolio.stats.summary().height > 0

    def test_sharpe_returns_dict(self, demo_engine: BasanosEngine) -> None:
        sharpe = demo_engine.portfolio.stats.sharpe(periods=252)
        assert isinstance(sharpe, dict)
        assert "returns" in sharpe

    def test_sharpe_is_finite(self, demo_engine: BasanosEngine) -> None:
        sharpe = demo_engine.portfolio.stats.sharpe(periods=252)
        assert np.isfinite(sharpe["returns"])

    def test_volatility_returns_dict(self, demo_engine: BasanosEngine) -> None:
        vol = demo_engine.portfolio.stats.volatility(periods=252)
        assert isinstance(vol, dict)
        assert "returns" in vol

    def test_volatility_is_positive(self, demo_engine: BasanosEngine) -> None:
        vol = demo_engine.portfolio.stats.volatility(periods=252)
        assert vol["returns"] > 0


# ─── Portfolio turnover ───────────────────────────────────────────────────────


class TestDemoPortfolioTurnover:
    """Turnover summary used in demo notebook cell_16b."""

    def test_turnover_summary_is_dataframe(self, demo_engine: BasanosEngine) -> None:
        result = demo_engine.portfolio.turnover_summary()
        assert isinstance(result, pl.DataFrame)

    def test_turnover_summary_has_metric_and_value_columns(self, demo_engine: BasanosEngine) -> None:
        cols = demo_engine.portfolio.turnover_summary().columns
        assert "metric" in cols
        assert "value" in cols

    def test_turnover_summary_contains_expected_metrics(self, demo_engine: BasanosEngine) -> None:
        metrics = set(demo_engine.portfolio.turnover_summary()["metric"].to_list())
        assert "mean_daily_turnover" in metrics
        assert "mean_weekly_turnover" in metrics
        assert "turnover_std" in metrics


# ─── Portfolio decomposition and monthly ─────────────────────────────────────


class TestDemoPortfolioDecomposition:
    """Tilt/timing decomposition and monthly returns (cell_17 / cell_18)."""

    def test_tilt_timing_decomp_is_dataframe(self, demo_engine: BasanosEngine) -> None:
        assert isinstance(demo_engine.portfolio.tilt_timing_decomp, pl.DataFrame)

    def test_tilt_timing_decomp_has_rows(self, demo_engine: BasanosEngine) -> None:
        assert demo_engine.portfolio.tilt_timing_decomp.height > 0

    def test_monthly_has_expected_columns(self, demo_engine: BasanosEngine) -> None:
        required = {"year", "month_name", "returns", "profit", "NAV_accumulated"}
        assert required.issubset(set(demo_engine.portfolio.monthly.columns))

    def test_monthly_has_rows(self, demo_engine: BasanosEngine) -> None:
        assert demo_engine.portfolio.monthly.height > 0


# ─── Portfolio plots ──────────────────────────────────────────────────────────


class TestDemoPortfolioPlots:
    """Plot methods return Plotly figures (cell_21 / cell_22 / cell_23)."""

    def test_snapshot_returns_plotly_figure(self, demo_engine: BasanosEngine) -> None:
        fig = demo_engine.portfolio.plots.snapshot()
        assert isinstance(fig, go.Figure)

    def test_lead_lag_ir_plot_returns_plotly_figure(self, demo_engine: BasanosEngine) -> None:
        fig = demo_engine.portfolio.plots.lead_lag_ir_plot(start=-5, end=10)
        assert isinstance(fig, go.Figure)

    def test_correlation_heatmap_returns_plotly_figure(self, demo_engine: BasanosEngine) -> None:
        fig = demo_engine.portfolio.plots.correlation_heatmap()
        assert isinstance(fig, go.Figure)
