"""Tests for basanos.analytics._async (AsyncPortfolio)."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import polars as pl

from basanos.analytics import AsyncPortfolio, Portfolio
from basanos.analytics._stats import Stats

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_portfolio(n: int = 60) -> Portfolio:
    """Build a simple Portfolio with a 'date' column for testing."""
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([100.0 + i * 0.5 + (i % 7) * 0.3 for i in range(n)], dtype=pl.Float64),
            "B": pl.Series([200.0 - i * 0.2 + (i % 5) * 0.4 for i in range(n)], dtype=pl.Float64),
        }
    )
    cashposition = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([1000.0] * n, dtype=pl.Float64),
            "B": pl.Series([500.0] * n, dtype=pl.Float64),
        }
    )
    return Portfolio(prices=prices, cashposition=cashposition)


def _make_async_portfolio(n: int = 60) -> AsyncPortfolio:
    """Build an AsyncPortfolio wrapping _make_portfolio()."""
    return AsyncPortfolio(_make_portfolio(n))


# ─── Construction ─────────────────────────────────────────────────────────────


def test_async_portfolio_construction():
    """AsyncPortfolio can be constructed from a Portfolio."""
    pf = _make_async_portfolio()
    assert isinstance(pf, AsyncPortfolio)


def test_async_portfolio_from_cash_position():
    """from_cash_position() classmethod returns an AsyncPortfolio."""
    base = _make_portfolio()
    pf = AsyncPortfolio.from_cash_position(prices=base.prices, cash_position=base.cashposition, aum=base.aum)
    assert isinstance(pf, AsyncPortfolio)
    assert pf.assets == base.assets


def test_async_portfolio_from_risk_position():
    """from_risk_position() classmethod returns an AsyncPortfolio."""
    import numpy as np

    n = 40
    start = date(2021, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    rng = np.random.default_rng(7)
    prices = pl.DataFrame(
        {
            "date": dates,
            "X": pl.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)), dtype=pl.Float64),
        }
    )
    risk_pos = pl.DataFrame(
        {
            "date": dates,
            "X": pl.Series([1.0] * n, dtype=pl.Float64),
        }
    )
    pf = AsyncPortfolio.from_risk_position(prices=prices, risk_position=risk_pos, vola=16)
    assert isinstance(pf, AsyncPortfolio)


# ─── Synchronous passthrough properties ───────────────────────────────────────


def test_async_portfolio_assets():
    """Assets property returns a list of numeric column names."""
    pf = _make_async_portfolio()
    assert pf.assets == ["A", "B"]


def test_async_portfolio_prices():
    """Prices property returns a Polars DataFrame."""
    pf = _make_async_portfolio()
    assert isinstance(pf.prices, pl.DataFrame)


def test_async_portfolio_cashposition():
    """Cashposition property returns a Polars DataFrame."""
    pf = _make_async_portfolio()
    assert isinstance(pf.cashposition, pl.DataFrame)


def test_async_portfolio_aum():
    """Aum property returns the configured float."""
    pf = _make_async_portfolio()
    assert pf.aum == 1e8


# ─── Async methods ─────────────────────────────────────────────────────────────


def test_async_portfolio_profits():
    """profits() returns a DataFrame with the same shape as prices."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.profits())
    assert isinstance(result, pl.DataFrame)
    assert result.shape == pf.prices.shape


def test_async_portfolio_profit():
    """profit() returns a DataFrame with a 'profit' column."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.profit())
    assert isinstance(result, pl.DataFrame)
    assert "profit" in result.columns


def test_async_portfolio_nav_accumulated():
    """nav_accumulated() returns a DataFrame with 'NAV_accumulated'."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.nav_accumulated())
    assert isinstance(result, pl.DataFrame)
    assert "NAV_accumulated" in result.columns


def test_async_portfolio_returns():
    """returns() returns a DataFrame with 'returns' column."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.returns())
    assert isinstance(result, pl.DataFrame)
    assert "returns" in result.columns


def test_async_portfolio_monthly():
    """monthly() returns a DataFrame with 'year', 'month', and 'month_name'."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.monthly())
    assert isinstance(result, pl.DataFrame)
    assert "year" in result.columns
    assert "month" in result.columns
    assert "month_name" in result.columns


def test_async_portfolio_nav_compounded():
    """nav_compounded() returns a DataFrame with 'NAV_compounded'."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.nav_compounded())
    assert isinstance(result, pl.DataFrame)
    assert "NAV_compounded" in result.columns


def test_async_portfolio_highwater():
    """highwater() returns a DataFrame with 'highwater'."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.highwater())
    assert isinstance(result, pl.DataFrame)
    assert "highwater" in result.columns


def test_async_portfolio_drawdown():
    """drawdown() returns a DataFrame with 'drawdown' and 'drawdown_pct'."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.drawdown())
    assert isinstance(result, pl.DataFrame)
    assert "drawdown" in result.columns
    assert "drawdown_pct" in result.columns


def test_async_portfolio_all():
    """all() returns a DataFrame containing 'NAV_compounded'."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.all())
    assert isinstance(result, pl.DataFrame)
    assert "NAV_compounded" in result.columns


def test_async_portfolio_stats():
    """stats() returns a Stats instance."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.stats())
    assert isinstance(result, Stats)


def test_async_portfolio_tilt():
    """tilt() returns an AsyncPortfolio with constant weights."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.tilt())
    assert isinstance(result, AsyncPortfolio)
    assert result.assets == pf.assets


def test_async_portfolio_timing():
    """timing() returns an AsyncPortfolio."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.timing())
    assert isinstance(result, AsyncPortfolio)
    assert result.assets == pf.assets


def test_async_portfolio_tilt_timing_decomp():
    """tilt_timing_decomp() returns a DataFrame with 'portfolio', 'tilt', 'timing'."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.tilt_timing_decomp())
    assert isinstance(result, pl.DataFrame)
    assert "portfolio" in result.columns
    assert "tilt" in result.columns
    assert "timing" in result.columns


def test_async_portfolio_correlation():
    """correlation() returns a square correlation DataFrame."""
    base = _make_portfolio()
    pf = AsyncPortfolio(base)
    result = asyncio.run(pf.correlation(base.prices))
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == result.shape[1]


def test_async_portfolio_truncate():
    """truncate() returns an AsyncPortfolio with fewer rows."""
    pf = _make_async_portfolio(n=60)
    start = date(2020, 1, 10)
    end = date(2020, 2, 10)
    result = asyncio.run(pf.truncate(start=start, end=end))
    assert isinstance(result, AsyncPortfolio)
    assert result.prices.height < pf.prices.height


def test_async_portfolio_lag():
    """lag() returns an AsyncPortfolio."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.lag(3))
    assert isinstance(result, AsyncPortfolio)
    assert result.prices.shape == pf.prices.shape


def test_async_portfolio_smoothed_holding():
    """smoothed_holding() returns an AsyncPortfolio."""
    pf = _make_async_portfolio()
    result = asyncio.run(pf.smoothed_holding(5))
    assert isinstance(result, AsyncPortfolio)
    assert result.prices.shape == pf.prices.shape


# ─── Correctness: async matches sync ──────────────────────────────────────────


def test_async_profit_equals_sync():
    """Async profit() result matches synchronous Portfolio.profit."""
    base = _make_portfolio()
    pf = AsyncPortfolio(base)
    async_result = asyncio.run(pf.profit())
    sync_result = base.profit
    assert async_result.equals(sync_result)


def test_async_nav_accumulated_equals_sync():
    """Async nav_accumulated() result matches synchronous Portfolio.nav_accumulated."""
    base = _make_portfolio()
    pf = AsyncPortfolio(base)
    async_result = asyncio.run(pf.nav_accumulated())
    sync_result = base.nav_accumulated
    assert async_result.equals(sync_result)


# ─── Concurrent execution ──────────────────────────────────────────────────────


def test_async_portfolio_concurrent_calls():
    """Multiple async calls can be gathered concurrently without error."""

    async def run_concurrent() -> tuple:
        pf = _make_async_portfolio()
        profit, nav = await asyncio.gather(pf.profit(), pf.nav_accumulated())
        return profit, nav

    profit, nav = asyncio.run(run_concurrent())
    assert isinstance(profit, pl.DataFrame)
    assert isinstance(nav, pl.DataFrame)
