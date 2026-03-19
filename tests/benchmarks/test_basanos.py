"""Benchmark tests for basanos core components.

Covers the hot paths in Portfolio, Stats, and BasanosEngine across realistic
dataset sizes (daily, 1-year and 5-year horizons, 5 and 20 assets).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.analytics import Portfolio
from basanos.analytics._stats import Stats
from basanos.math.optimizer import BasanosConfig, BasanosEngine

# ─── Data factories ──────────────────────────────────────────────────────────


def _make_portfolio(n: int, n_assets: int, seed: int = 0) -> Portfolio:
    """Return a Portfolio with ``n`` daily rows and ``n_assets`` assets."""
    rng = np.random.default_rng(seed)
    start = date(2015, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    assets = [f"A{i}" for i in range(n_assets)]

    # Geometric random-walk prices (strictly positive, non-monotonic)
    prices_data: dict[str, pl.Series] = {"date": dates}
    positions_data: dict[str, pl.Series] = {"date": dates}
    for asset in assets:
        log_ret = rng.normal(0.0, 0.01, size=n)
        price = 100.0 * np.exp(np.cumsum(log_ret))
        prices_data[asset] = pl.Series(price.tolist())
        position = 1e5 + rng.normal(0, 5e3, size=n)
        positions_data[asset] = pl.Series(position.tolist())

    prices = pl.DataFrame(prices_data)
    positions = pl.DataFrame(positions_data)
    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e8)


def _make_engine(n: int, n_assets: int, seed: int = 42) -> BasanosEngine:
    """Return a BasanosEngine with ``n`` daily rows and ``n_assets`` assets."""
    rng = np.random.default_rng(seed)
    start = date(2015, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    assets = [f"A{i}" for i in range(n_assets)]

    prices_data: dict[str, pl.Series] = {"date": dates}
    mu_data: dict[str, pl.Series] = {"date": dates}
    for asset in assets:
        log_ret = rng.normal(0.0, 0.01, size=n)
        price = 100.0 * np.exp(np.cumsum(log_ret))
        prices_data[asset] = pl.Series(price.tolist())
        signal = rng.normal(0, 1, size=n)
        mu_data[asset] = pl.Series(signal.tolist())

    prices = pl.DataFrame(prices_data)
    mu = pl.DataFrame(mu_data)
    cfg = BasanosConfig(vola=32, corr=64, clip=3.0, shrink=0.5, aum=1e8)
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg)


def _make_sw_engine(n: int, n_assets: int, window: int, n_factors: int, seed: int = 42) -> BasanosEngine:
    """Return a sliding-window BasanosEngine with ``n`` rows and ``n_assets`` assets.

    Args:
        n: Number of daily rows (timesteps).
        n_assets: Number of asset columns.
        window: Rolling look-back window ``W`` for the factor model.
        n_factors: Number of PCA factors ``k`` retained per window.
        seed: RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    start = date(2015, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    assets = [f"A{i}" for i in range(n_assets)]

    prices_data: dict[str, pl.Series] = {"date": dates}
    mu_data: dict[str, pl.Series] = {"date": dates}
    for asset in assets:
        log_ret = rng.normal(0.0, 0.01, size=n)
        price = 100.0 * np.exp(np.cumsum(log_ret))
        prices_data[asset] = pl.Series(price.tolist())
        signal = rng.normal(0, 1, size=n)
        mu_data[asset] = pl.Series(signal.tolist())

    prices = pl.DataFrame(prices_data)
    mu = pl.DataFrame(mu_data)
    cfg = BasanosConfig(
        vola=32,
        corr=64,
        clip=3.0,
        shrink=0.5,
        aum=1e8,
        covariance_mode="sliding_window",
        window=window,
        n_factors=n_factors,
    )
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg)


# ─── Pre-built fixtures (constructed once per session) ───────────────────────


@pytest.fixture(scope="session")
def pf_252_5() -> Portfolio:
    """Portfolio: 252 days, 5 assets."""
    return _make_portfolio(252, 5)


@pytest.fixture(scope="session")
def pf_1260_5() -> Portfolio:
    """Portfolio: 1260 days, 5 assets."""
    return _make_portfolio(1260, 5)


@pytest.fixture(scope="session")
def pf_252_20() -> Portfolio:
    """Portfolio: 252 days, 20 assets."""
    return _make_portfolio(252, 20)


@pytest.fixture(scope="session")
def engine_252_5() -> BasanosEngine:
    """BasanosEngine: 252 days, 5 assets."""
    return _make_engine(252, 5)


@pytest.fixture(scope="session")
def engine_1260_5() -> BasanosEngine:
    """BasanosEngine: 1260 days, 5 assets."""
    return _make_engine(1260, 5)


@pytest.fixture(scope="session")
def engine_252_20() -> BasanosEngine:
    """BasanosEngine: 252 days, 20 assets."""
    return _make_engine(252, 20)


@pytest.fixture(scope="session")
def sw_engine_252_5_60_3() -> BasanosEngine:
    """Sliding-window BasanosEngine: T=252, N=5, W=60, k=3."""
    return _make_sw_engine(252, 5, window=60, n_factors=3)


@pytest.fixture(scope="session")
def sw_engine_252_20_60_5() -> BasanosEngine:
    """Sliding-window BasanosEngine: T=252, N=20, W=60, k=5."""
    return _make_sw_engine(252, 20, window=60, n_factors=5)


@pytest.fixture(scope="session")
def sw_engine_1260_5_60_3() -> BasanosEngine:
    """Sliding-window BasanosEngine: T=1260, N=5, W=60, k=3."""
    return _make_sw_engine(1260, 5, window=60, n_factors=3)


# ─── Portfolio benchmarks ─────────────────────────────────────────────────────


class TestPortfolioBenchmarks:
    """Benchmark hot paths on Portfolio."""

    def test_profits_252_5(self, benchmark, pf_252_5):
        """Benchmark profits on 252-day, 5-asset portfolio."""
        result = benchmark(lambda: pf_252_5.profits)
        assert result.shape[0] == 252

    def test_profits_1260_5(self, benchmark, pf_1260_5):
        """Benchmark profits on 1260-day, 5-asset portfolio."""
        result = benchmark(lambda: pf_1260_5.profits)
        assert result.shape[0] == 1260

    def test_profits_252_20(self, benchmark, pf_252_20):
        """Benchmark profits on 252-day, 20-asset portfolio."""
        result = benchmark(lambda: pf_252_20.profits)
        assert result.shape[0] == 252

    def test_nav_accumulated_252_5(self, benchmark, pf_252_5):
        """Benchmark accumulated NAV on 252-day, 5-asset portfolio."""
        result = benchmark(lambda: pf_252_5.nav_accumulated)
        assert "NAV_accumulated" in result.columns

    def test_nav_compounded_252_5(self, benchmark, pf_252_5):
        """Benchmark compounded NAV on 252-day, 5-asset portfolio."""
        result = benchmark(lambda: pf_252_5.nav_compounded)
        assert "NAV_compounded" in result.columns

    def test_drawdown_252_5(self, benchmark, pf_252_5):
        """Benchmark drawdown on 252-day, 5-asset portfolio."""
        result = benchmark(lambda: pf_252_5.drawdown)
        assert "drawdown" in result.columns
        assert (result["drawdown"] >= 0).all()

    def test_drawdown_1260_5(self, benchmark, pf_1260_5):
        """Benchmark drawdown on 1260-day, 5-asset portfolio."""
        result = benchmark(lambda: pf_1260_5.drawdown)
        assert "drawdown" in result.columns

    def test_monthly_252_5(self, benchmark, pf_252_5):
        """Benchmark monthly compounded returns on 252-day, 5-asset portfolio."""
        result = benchmark(lambda: pf_252_5.monthly)
        assert "returns" in result.columns
        assert result.shape[0] > 0

    def test_tilt_timing_decomp_252_5(self, benchmark, pf_252_5):
        """Benchmark tilt/timing decomposition on 252-day, 5-asset portfolio."""
        result = benchmark(lambda: pf_252_5.tilt_timing_decomp)
        assert {"portfolio", "tilt", "timing"}.issubset(result.columns)

    def test_all_252_5(self, benchmark, pf_252_5):
        """Benchmark full merged view (drawdown + compounded NAV) on 252-day portfolio."""
        result = benchmark(lambda: pf_252_5.all)
        assert "drawdown" in result.columns
        assert "NAV_compounded" in result.columns


# ─── Stats benchmarks ─────────────────────────────────────────────────────────


class TestStatsBenchmarks:
    """Benchmark Stats computations on portfolio returns."""

    @pytest.fixture(scope="class")
    def stats_252_5(self, pf_252_5) -> Stats:
        """Stats built from the 252-day portfolio."""
        return pf_252_5.stats

    @pytest.fixture(scope="class")
    def stats_1260_5(self, pf_1260_5) -> Stats:
        """Stats built from the 1260-day portfolio."""
        return pf_1260_5.stats

    def test_volatility_252(self, benchmark, stats_252_5):
        """Benchmark annualised volatility on 252-day returns."""
        result = benchmark(stats_252_5.volatility)
        assert all(v >= 0 for v in result.values())

    def test_sharpe_252(self, benchmark, stats_252_5):
        """Benchmark Sharpe ratio on 252-day returns."""
        result = benchmark(stats_252_5.sharpe)
        assert isinstance(result, dict)

    def test_value_at_risk_252(self, benchmark, stats_252_5):
        """Benchmark VaR (alpha=5%) on 252-day returns."""
        result = benchmark(lambda: stats_252_5.value_at_risk(alpha=0.05))
        assert all(v < 0 for v in result.values())

    def test_summary_252(self, benchmark, stats_252_5):
        """Benchmark full stats summary on 252-day returns."""
        result = benchmark(stats_252_5.summary)
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 20

    def test_summary_1260(self, benchmark, stats_1260_5):
        """Benchmark full stats summary on 1260-day returns."""
        result = benchmark(stats_1260_5.summary)
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 20


# ─── BasanosEngine benchmarks ─────────────────────────────────────────────────


class TestBasanosEngineBenchmarks:
    """Benchmark the optimizer's EWM correlation and position computation."""

    def test_ret_adj_252_5(self, benchmark, engine_252_5):
        """Benchmark volatility-adjusted returns on 252-day, 5-asset engine."""
        result = benchmark(lambda: engine_252_5.ret_adj)
        assert result.shape == engine_252_5.prices.shape

    def test_vola_252_5(self, benchmark, engine_252_5):
        """Benchmark EWMA volatility on 252-day, 5-asset engine."""
        result = benchmark(lambda: engine_252_5.vola)
        assert result.shape == engine_252_5.prices.shape

    def test_cor_252_5(self, benchmark, engine_252_5):
        """Benchmark EWM correlation matrices on 252-day, 5-asset engine."""
        result = benchmark(lambda: engine_252_5.cor)
        assert len(result) == 252

    def test_cor_1260_5(self, benchmark, engine_1260_5):
        """Benchmark EWM correlation matrices on 1260-day, 5-asset engine."""
        result = benchmark(lambda: engine_1260_5.cor)
        assert len(result) == 1260

    def test_cor_252_20(self, benchmark, engine_252_20):
        """Benchmark EWM correlation matrices on 252-day, 20-asset engine."""
        result = benchmark(lambda: engine_252_20.cor)
        assert len(result) == 252

    def test_cash_position_252_5(self, benchmark, engine_252_5):
        """Benchmark full position optimization on 252-day, 5-asset engine."""
        result = benchmark(lambda: engine_252_5.cash_position)
        assert result.shape == engine_252_5.prices.shape

    def test_cash_position_1260_5(self, benchmark, engine_1260_5):
        """Benchmark full position optimization on 1260-day, 5-asset engine."""
        result = benchmark(lambda: engine_1260_5.cash_position)
        assert result.shape == engine_1260_5.prices.shape

    def test_portfolio_252_5(self, benchmark, engine_252_5):
        """Benchmark end-to-end Portfolio construction from the optimizer."""
        result = benchmark(lambda: engine_252_5.portfolio)
        assert isinstance(result, Portfolio)
        assert result.assets == engine_252_5.assets


# ─── Sliding-window BasanosEngine benchmarks ─────────────────────────────────


class TestSlidingWindowBenchmarks:
    """Benchmark ``cash_position`` in sliding-window covariance mode.

    Each fixture name encodes ``sw_<T>_<N>_<W>_<k>`` where:

    - ``T`` – number of daily rows (timesteps)
    - ``N`` – number of asset columns
    - ``W`` – rolling look-back window
    - ``k`` – number of PCA factors retained per window

    The sliding-window path has complexity O(T·W·N·k) for the rolling SVDs
    and O(T·(k³ + kN)) for the Woodbury solves, with memory O(W·N) per step
    (independent of T).  These cases exercise the representative corners of
    that curve so that CI regression detection covers the hot path.
    """

    def test_cash_position_sw_252_5_60_3(self, benchmark, sw_engine_252_5_60_3):
        """Benchmark sliding-window cash_position: T=252, N=5, W=60, k=3."""
        result = benchmark(lambda: sw_engine_252_5_60_3.cash_position)
        assert result.shape == sw_engine_252_5_60_3.prices.shape

    def test_cash_position_sw_252_20_60_5(self, benchmark, sw_engine_252_20_60_5):
        """Benchmark sliding-window cash_position: T=252, N=20, W=60, k=5."""
        result = benchmark(lambda: sw_engine_252_20_60_5.cash_position)
        assert result.shape == sw_engine_252_20_60_5.prices.shape

    def test_cash_position_sw_1260_5_60_3(self, benchmark, sw_engine_1260_5_60_3):
        """Benchmark sliding-window cash_position: T=1260, N=5, W=60, k=3."""
        result = benchmark(lambda: sw_engine_1260_5_60_3.cash_position)
        assert result.shape == sw_engine_1260_5_60_3.prices.shape
