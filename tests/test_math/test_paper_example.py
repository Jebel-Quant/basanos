"""Smoke-test for the end-to-end code listing in paper/basanos.tex (§6).

Every API call that appears in the paper's code example is exercised here so
that drift between the paper and the library is caught by ``make test``.
The synthetic data and configuration are taken verbatim from the listing;
only the file-system side-effects (``portfolio.report.save``) are omitted.

Covered API surface
-------------------
Engine construction
    ``BasanosEngine(prices, mu, cfg)``
Position outputs
    ``cash_position``, ``position_status``, ``risk_position``
Optimizer diagnostics
    ``condition_number``, ``effective_rank``, ``solver_residual``,
    ``signal_utilisation``
Signal quality metrics
    ``ic``, ``rank_ic``, ``icir``, ``naive_sharpe``
Portfolio analytics
    ``portfolio.stats.sharpe()``, ``portfolio.stats.max_drawdown()``,
    ``portfolio.tilt_timing_decomp``
Visualisations
    ``portfolio.plots.snapshot()``,
    ``portfolio.plots.lead_lag_ir_plot()``
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from basanos.math import BasanosConfig, BasanosEngine

# ─── Fixtures (verbatim from the paper listing) ───────────────────────────────

_N_DAYS = 500
_ASSETS = ["AAPL", "GOOGL", "MSFT"]


@pytest.fixture(scope="module")
def paper_prices() -> pl.DataFrame:
    """Synthetic price series exactly as written in the paper."""
    rng = np.random.default_rng(42)
    dates = pl.date_range(
        pl.date(2022, 1, 1),
        pl.date(2022, 1, 1) + pl.duration(days=_N_DAYS - 1),
        eager=True,
    )
    return pl.DataFrame(
        {
            "date": dates,
            "AAPL": 100.0 + np.cumsum(rng.normal(0, 1.0, _N_DAYS)),
            "GOOGL": 150.0 + np.cumsum(rng.normal(0, 1.2, _N_DAYS)),
            "MSFT": 200.0 + np.cumsum(rng.normal(0, 0.9, _N_DAYS)),
        }
    )


@pytest.fixture(scope="module")
def paper_mu(paper_prices: pl.DataFrame) -> pl.DataFrame:
    """Signal series exactly as written in the paper."""
    rng = np.random.default_rng(42)
    # Advance the RNG past the price draws so mu draws match the listing.
    rng.normal(0, 1.0, _N_DAYS)
    rng.normal(0, 1.2, _N_DAYS)
    rng.normal(0, 0.9, _N_DAYS)
    dates = paper_prices["date"]
    return pl.DataFrame(
        {
            "date": dates,
            "AAPL": np.tanh(rng.normal(0, 0.5, _N_DAYS)),
            "GOOGL": np.tanh(rng.normal(0, 0.5, _N_DAYS)),
            "MSFT": np.tanh(rng.normal(0, 0.5, _N_DAYS)),
        }
    )


@pytest.fixture(scope="module")
def paper_engine(paper_prices: pl.DataFrame, paper_mu: pl.DataFrame) -> BasanosEngine:
    """Engine with the configuration from the paper listing."""
    cfg = BasanosConfig(
        vola=16,
        corr=64,
        clip=3.5,
        shrink=0.6,
        aum=1e6,
    )
    return BasanosEngine(prices=paper_prices, mu=paper_mu, cfg=cfg)


# ─── Position outputs ─────────────────────────────────────────────────────────


class TestPaperPositionOutputs:
    """Verify cash_position, position_status, and risk_position from the paper listing."""

    def test_cash_position_shape(self, paper_engine: BasanosEngine) -> None:
        cp = paper_engine.cash_position
        assert cp.height == _N_DAYS
        assert set(_ASSETS).issubset(cp.columns)

    def test_cash_position_values_finite_or_nan(self, paper_engine: BasanosEngine) -> None:
        for asset in _ASSETS:
            vals = paper_engine.cash_position[asset].to_numpy()
            assert np.all(np.isfinite(vals) | np.isnan(vals))

    def test_position_status_shape(self, paper_engine: BasanosEngine) -> None:
        ps = paper_engine.position_status
        assert ps.height == _N_DAYS
        assert "status" in ps.columns

    def test_position_status_valid_codes(self, paper_engine: BasanosEngine) -> None:
        allowed = {"warmup", "zero_signal", "degenerate", "valid"}
        actual = set(paper_engine.position_status["status"].to_list())
        assert actual.issubset(allowed)

    def test_position_status_has_valid_rows(self, paper_engine: BasanosEngine) -> None:
        statuses = paper_engine.position_status["status"].to_list()
        assert "valid" in statuses


# ─── Optimizer diagnostics ────────────────────────────────────────────────────


class TestPaperOptimizerDiagnostics:
    """Verify condition_number, effective_rank, solver_residual, signal_utilisation."""

    def test_condition_number_is_dataframe(self, paper_engine: BasanosEngine) -> None:
        assert isinstance(paper_engine.condition_number, pl.DataFrame)

    def test_condition_number_positive(self, paper_engine: BasanosEngine) -> None:
        kappa = paper_engine.condition_number
        vals = kappa.select(pl.exclude("date")).to_numpy().flatten()
        finite = vals[np.isfinite(vals)]
        assert np.all(finite > 0)

    def test_effective_rank_is_dataframe(self, paper_engine: BasanosEngine) -> None:
        assert isinstance(paper_engine.effective_rank, pl.DataFrame)

    def test_effective_rank_bounded(self, paper_engine: BasanosEngine) -> None:
        er = paper_engine.effective_rank
        vals = er.select(pl.exclude("date")).to_numpy().flatten()
        finite = vals[np.isfinite(vals)]
        assert np.all(finite >= 1.0)
        assert np.all(finite <= len(_ASSETS) + 1e-9)

    def test_solver_residual_is_dataframe(self, paper_engine: BasanosEngine) -> None:
        assert isinstance(paper_engine.solver_residual, pl.DataFrame)

    def test_solver_residual_small(self, paper_engine: BasanosEngine) -> None:
        res = paper_engine.solver_residual
        vals = res.select(pl.exclude("date")).to_numpy().flatten()
        finite = vals[np.isfinite(vals)]
        assert np.all(finite < 1e-6)

    def test_signal_utilisation_is_dataframe(self, paper_engine: BasanosEngine) -> None:
        assert isinstance(paper_engine.signal_utilisation, pl.DataFrame)

    def test_signal_utilisation_columns(self, paper_engine: BasanosEngine) -> None:
        assert set(_ASSETS).issubset(paper_engine.signal_utilisation.columns)


# ─── Signal quality metrics ───────────────────────────────────────────────────


class TestPaperSignalQuality:
    """Verify ic, rank_ic, icir, and naive_sharpe from the paper listing."""

    def test_ic_is_dataframe(self, paper_engine: BasanosEngine) -> None:
        assert isinstance(paper_engine.ic(), pl.DataFrame)

    def test_ic_height(self, paper_engine: BasanosEngine) -> None:
        assert paper_engine.prices.height == _N_DAYS
        assert paper_engine.mu.height == _N_DAYS
        assert paper_engine.ic().height == _N_DAYS - 1  # no data for the last day!

    def test_rank_ic_is_dataframe(self, paper_engine: BasanosEngine) -> None:
        assert isinstance(paper_engine.rank_ic(), pl.DataFrame)

    def test_rank_ic_height(self, paper_engine: BasanosEngine) -> None:
        assert paper_engine.rank_ic().height == _N_DAYS - 1

    def test_icir_is_finite_float(self, paper_engine: BasanosEngine) -> None:
        assert isinstance(paper_engine.icir(), float)
        assert np.isfinite(paper_engine.icir())

    def test_naive_sharpe_is_finite_float(self, paper_engine: BasanosEngine) -> None:
        assert isinstance(paper_engine.naive_sharpe, float)
        assert np.isfinite(paper_engine.naive_sharpe)


# ─── Portfolio analytics ──────────────────────────────────────────────────────


class TestPaperPortfolioAnalytics:
    """Verify portfolio.stats, tilt_timing_decomp from the paper listing."""

    def test_stats_sharpe_is_finite(self, paper_engine: BasanosEngine) -> None:
        sharpe = paper_engine.portfolio.stats.sharpe()
        assert np.isfinite(sharpe["returns"])

    def test_stats_max_drawdown_is_finite(self, paper_engine: BasanosEngine) -> None:
        mdd = paper_engine.portfolio.stats.max_drawdown()
        assert np.isfinite(mdd["returns"])

    def test_tilt_timing_decomp_is_dataframe(self, paper_engine: BasanosEngine) -> None:
        assert isinstance(paper_engine.portfolio.tilt_timing_decomp, pl.DataFrame)

    def test_tilt_timing_decomp_has_rows(self, paper_engine: BasanosEngine) -> None:
        assert paper_engine.portfolio.tilt_timing_decomp.height > 0


# ─── Visualisations ───────────────────────────────────────────────────────────


class TestPaperVisualisations:
    """Verify snapshot() and lead_lag_ir_plot() return Plotly figures."""

    def test_snapshot_returns_figure(self, paper_engine: BasanosEngine) -> None:
        fig = paper_engine.portfolio.plots.snapshot()
        assert isinstance(fig, go.Figure)

    def test_lead_lag_ir_plot_returns_figure(self, paper_engine: BasanosEngine) -> None:
        fig = paper_engine.portfolio.plots.lead_lag_ir_plot(-5, 15)
        assert isinstance(fig, go.Figure)
