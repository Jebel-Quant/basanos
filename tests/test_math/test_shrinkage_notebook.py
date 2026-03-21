"""CI execution gate for the shrinkage_guide notebook.

This module mirrors the data setup and cell logic from
``book/marimo/notebooks/shrinkage_guide.py`` so that any drift in the
shrinkage API is caught by ``make test`` before it can silently corrupt
the notebook.

Covered API surface:

- ``shrink2id(corr, lamb)`` — linear shrinkage toward identity
- ``BasanosConfig`` + ``BasanosEngine`` with varying ``shrink`` values
- ``portfolio.stats.sharpe()`` and ``portfolio.stats.volatility()``
- ``portfolio.turnover_summary()``
- ``portfolio.plots.snapshot()``
- Condition number computation via ``np.linalg.eigvalsh``
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from basanos.math import BasanosConfig, BasanosEngine
from basanos.math._signal import shrink2id

# ─── Constants (mirror cell_05 of the notebook) ───────────────────────────────

_SEED = 42
_ASSETS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"]
_N = 750
_DRIFT = [0.0003, 0.0002, 0.0004, 0.0001, 0.0005, 0.0002]
_SIGMA = [0.018, 0.022, 0.020, 0.025, 0.030, 0.024]
_S0 = [150.0, 2800.0, 300.0, 3300.0, 700.0, 200.0]


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def shrinkage_prices() -> pl.DataFrame:
    """Synthetic price series matching shrinkage_guide notebook cell_05."""
    rng = np.random.default_rng(_SEED)
    start = pl.date(2021, 1, 1)
    dates = pl.date_range(start, start + pl.duration(days=_N - 1), interval="1d", eager=True)
    cols: dict[str, object] = {"date": dates}
    for i, asset in enumerate(_ASSETS):
        log_ret = rng.normal(_DRIFT[i], _SIGMA[i], _N)
        cols[asset] = _S0[i] * np.exp(np.cumsum(log_ret))
    return pl.DataFrame(cols)


@pytest.fixture(scope="module")
def shrinkage_mu(shrinkage_prices: pl.DataFrame) -> pl.DataFrame:
    """Momentum signal matching shrinkage_guide notebook cell_05."""
    dates = shrinkage_prices["date"]
    cols: dict[str, object] = {"date": dates}
    for asset in _ASSETS:
        p = shrinkage_prices[asset].to_numpy()
        ma5 = np.convolve(p, np.ones(5) / 5, mode="same") / p - 1.0
        ma20 = np.convolve(p, np.ones(20) / 20, mode="same") / p - 1.0
        cols[asset] = np.tanh(50.0 * (ma5 - ma20))
    return pl.DataFrame(cols)


@pytest.fixture(scope="module")
def shrinkage_engine(shrinkage_prices: pl.DataFrame, shrinkage_mu: pl.DataFrame) -> BasanosEngine:
    """BasanosEngine at the notebook's mid-range shrinkage value."""
    cfg = BasanosConfig(vola=16, corr=32, clip=3.5, shrink=0.5, aum=1_000_000)
    return BasanosEngine(prices=shrinkage_prices, mu=shrinkage_mu, cfg=cfg)


# ─── shrink2id ────────────────────────────────────────────────────────────────


class TestShrink2Id:
    """``shrink2id`` function used in condition-number cell (cell_12)."""

    def test_full_shrinkage_returns_identity(self) -> None:
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        result = shrink2id(corr, lamb=0.0)
        np.testing.assert_allclose(result, np.eye(2), atol=1e-12)

    def test_no_shrinkage_returns_original(self) -> None:
        corr = np.array([[1.0, 0.6], [0.6, 1.0]])
        result = shrink2id(corr, lamb=1.0)
        np.testing.assert_allclose(result, corr, atol=1e-12)

    def test_mid_shrinkage_is_convex_combination(self) -> None:
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        lamb = 0.5
        result = shrink2id(corr, lamb=lamb)
        expected = lamb * corr + (1 - lamb) * np.eye(2)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_output_diagonal_is_one(self) -> None:
        corr = np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0.4], [0.3, 0.4, 1.0]])
        for lamb in [0.0, 0.3, 0.7, 1.0]:
            result = shrink2id(corr, lamb=lamb)
            np.testing.assert_allclose(np.diag(result), np.ones(3), atol=1e-12)

    def test_shrinkage_reduces_condition_number(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(size=(30, 6))
        corr = np.corrcoef(data.T)
        kappa_raw = np.linalg.cond(corr)
        shrunk = shrink2id(corr, lamb=0.3)
        kappa_shrunk = np.linalg.cond(shrunk)
        assert kappa_shrunk < kappa_raw


# ─── Engine with varying shrinkage ────────────────────────────────────────────


class TestShrinkageEngineApi:
    """Engine construction at various shrinkage levels (mirrors cell_08 sweep)."""

    @pytest.mark.parametrize("shrink", [0.0, 0.5, 1.0])
    def test_engine_constructs_at_all_shrinkage_levels(
        self, shrinkage_prices: pl.DataFrame, shrinkage_mu: pl.DataFrame, shrink: float
    ) -> None:
        cfg = BasanosConfig(vola=16, corr=32, clip=3.5, shrink=shrink, aum=1_000_000)
        engine = BasanosEngine(prices=shrinkage_prices, mu=shrinkage_mu, cfg=cfg)
        assert engine.cash_position.height == _N

    def test_sharpe_returns_finite_float(self, shrinkage_engine: BasanosEngine) -> None:
        sharpe = shrinkage_engine.portfolio.stats.sharpe(periods=252)
        assert isinstance(sharpe, dict)
        assert np.isfinite(sharpe["returns"])

    def test_volatility_returns_positive_float(self, shrinkage_engine: BasanosEngine) -> None:
        vol = shrinkage_engine.portfolio.stats.volatility(periods=252)
        assert vol["returns"] > 0

    def test_turnover_summary_has_mean_daily_turnover(self, shrinkage_engine: BasanosEngine) -> None:
        summary = shrinkage_engine.portfolio.turnover_summary()
        metrics = summary["metric"].to_list()
        assert "mean_daily_turnover" in metrics

    def test_turnover_summary_mean_daily_is_non_negative_or_nan(self, shrinkage_engine: BasanosEngine) -> None:
        summary = shrinkage_engine.portfolio.turnover_summary()
        mean_to = float(summary.filter(pl.col("metric") == "mean_daily_turnover")["value"][0])
        assert np.isnan(mean_to) or mean_to >= 0.0

    def test_snapshot_returns_plotly_figure(self, shrinkage_engine: BasanosEngine) -> None:
        fig = shrinkage_engine.portfolio.plots.snapshot()
        assert isinstance(fig, go.Figure)


# ─── Condition number cell (cell_12) ─────────────────────────────────────────


class TestConditionNumberVsShrinkage:
    """Condition number decreases monotonically as shrinkage increases (cell_12)."""

    def test_condition_number_decreases_with_more_shrinkage(self, shrinkage_prices: pl.DataFrame) -> None:
        subset = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        data = shrinkage_prices.select(subset).to_numpy().astype(float)
        log_ret = np.diff(np.log(data), axis=0)
        sample = log_ret[-40:]
        corr = np.corrcoef(sample.T)

        lambdas = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
        kappas = []
        for lamb in lambdas:
            shrunk = shrink2id(corr, lamb=float(lamb))
            eigvals = np.linalg.eigvalsh(shrunk)
            kappa = eigvals[-1] / max(eigvals[0], 1e-14)
            kappas.append(kappa)

        # More shrinkage (lower lambda) → lower condition number
        for i in range(len(kappas) - 1):
            assert kappas[i] >= kappas[i + 1]

    def test_full_shrinkage_gives_condition_number_one(self, shrinkage_prices: pl.DataFrame) -> None:
        subset = ["AAPL", "GOOGL"]
        data = shrinkage_prices.select(subset).to_numpy().astype(float)
        log_ret = np.diff(np.log(data), axis=0)
        corr = np.corrcoef(log_ret.T)
        shrunk = shrink2id(corr, lamb=0.0)
        eigvals = np.linalg.eigvalsh(shrunk)
        kappa = eigvals[-1] / max(eigvals[0], 1e-14)
        np.testing.assert_allclose(kappa, 1.0, atol=1e-10)
